import argparse
import asyncio
import datetime
import logging
import traceback
from enum import Enum

import pyaudio

from agent.chat import AnthropicChatAgent, ChatAgent, OllamaChatAgent, OpenAIChatAgent
from config import Config
from input.audio import RATE, AudioInput, Input
from input.text import TextInput
from input.transcribe import Transcriber
from logging_config import setup_logging
from speech.kokoro import KokoroVoice, Voice
from speech.text import TextVoice
from tasks import chat_worker, cleanup_tasks, speech_worker, transcribe_worker

logger = logging.getLogger(__name__)


class INPUT(Enum):
    VOICE = "voice"
    TEXT = "text"

    def __str__(self):
        return self.value


class OUTPUT(Enum):
    SPEECH = "speech"
    TEXT = "text"

    def __str__(self):
        return self.value


def generate_thread_id() -> str:
    return str(datetime.datetime.now().timestamp())


async def main():
    parser = argparse.ArgumentParser(
        description="Real-time speech transcription and chat."
    )
    parser.add_argument(
        "--lang", type=str, default="en", help="Language for transcription"
    )
    parser.add_argument(
        "--input", type=str, choices=INPUT, default=INPUT.VOICE.value, help="Input type"
    )
    parser.add_argument(
        "--output",
        choices=OUTPUT,
        type=str,
        default=OUTPUT.SPEECH.value,
        help="Output type",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode for logging"
    )
    args = parser.parse_args()

    # Set up logging with debug flag
    setup_logging(args.debug)

    # Set log level for this specific logger as well
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    config = Config.from_env().validate()

    pa = pyaudio.PyAudio()
    audio_lock = asyncio.Lock()
    audio_queue = asyncio.Queue(maxsize=1)
    chat_queue = asyncio.Queue(maxsize=1)
    speech_queue = asyncio.Queue(maxsize=1)

    chat_agent: ChatAgent = None
    if config.anthropic_api_key:
        chat_agent = AnthropicChatAgent(
            api_key=config.anthropic_api_key,
            model_name=config.anthropic_model,
            lang=args.lang,
            thread_id=generate_thread_id(),
        )
    elif config.openai_api_key:
        chat_agent = OpenAIChatAgent(
            api_key=config.openai_api_key,
            model=config.openai_model,
            lang=args.lang,
            thread_id=generate_thread_id(),
        )
    elif config.ollama_host:
        chat_agent = OllamaChatAgent(
            host=config.ollama_host,
            port=config.ollama_port,
            model=config.ollama_model,
            lang=args.lang,
            thread_id=generate_thread_id(),
        )
    else:
        raise ValueError("No API key set.")

    voice: Voice = None
    if args.output == OUTPUT.SPEECH.value:
        voice = KokoroVoice(pa, audio_lock, lang=args.lang, debug=args.debug)
    elif args.output == OUTPUT.TEXT.value:
        voice = TextVoice()

    input: Input = None
    tasks = []
    if args.input == INPUT.VOICE.value:
        input = AudioInput(pa, audio_queue, audio_lock, debug=args.debug)
        transcriber = Transcriber(
            model_name=config.whisper_model, force_language=args.lang
        )
        tasks.append(
            asyncio.create_task(
                transcribe_worker(transcriber, audio_queue, chat_queue, RATE)
            )
        )
    elif args.input == INPUT.TEXT.value:
        input = TextInput(chat_queue)

    tasks.append(asyncio.create_task(chat_worker(chat_agent, chat_queue, speech_queue)))
    tasks.append(asyncio.create_task(speech_worker(voice, speech_queue)))

    try:
        while True:
            if args.input == INPUT.VOICE.value:
                await input.receive(
                    silence_duration=config.silence_duration,
                    min_speech_duration=config.min_speech_duration,
                    silence_threshold=config.silence_threshold,
                )
            elif args.input == INPUT.TEXT.value:
                await input.receive()
    except asyncio.CancelledError:
        logger.info("Cancelled.")
    except EOFError:
        logging.info("* Stopping...")
    except KeyboardInterrupt:
        logging.info("* Stopping...")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        # Properly cleanup tasks and queues
        await cleanup_tasks(
            tasks,
            speech_queue,
            chat_queue,
            audio_queue if args.input == INPUT.VOICE.value else None,
        )

        logger.info("Stopped.")
        pa.terminate()


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Received exit, exiting")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        traceback.print_exc()
    finally:
        loop.close()
