import argparse
import asyncio
import datetime
import logging
import traceback
from enum import Enum
from typing import Optional

import pyaudio

from agent.chat import (
    AnthropicChatAgent,
    OllamaChatAgent,
    OpenAIChatAgent,
)
from config import Config
from input.audio import RATE, AudioInput
from input.text import TextInput
from input.transcribe import Transcriber, TranscriberLike
from logging_config import setup_logging
from model.event import EventData
from repository.chat import ChatAgentLike
from repository.input import InputLike
from speech.kokoro import KokoroVoice, Voice
from speech.text import TextVoice
from tasks import (
    chat_worker,
    cleanup_tasks,
    input_worker,
    speech_worker,
    transcribe_worker,
)

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


async def main() -> None:
    """Main function to run the application."""
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
    audio_data = EventData()
    chat_data = EventData()
    speech_data = EventData()

    chat_agent = make_chat_agent(
        config,
        args.lang,
        generate_thread_id(),
    )

    voice = make_voice(args.output, pa, audio_lock, args.lang, args.debug)

    try:
        async with asyncio.TaskGroup() as tg:
            input: Optional[InputLike] = None
            if args.input == INPUT.VOICE.value:
                input = AudioInput(pa, audio_data, audio_lock, debug=args.debug)
                transcriber: TranscriberLike = Transcriber(
                    model_name=config.whisper_model, force_language=args.lang
                )
                tg.create_task(
                    transcribe_worker(transcriber, audio_data, chat_data, RATE),
                    name="transcribe_worker",
                )
                # Create the input worker task for voice input
                tg.create_task(
                    input_worker(input, config, is_voice_input=True),
                    name="input_worker",
                )
            elif args.input == INPUT.TEXT.value:
                input = TextInput(chat_data)
                # Create the input worker task for text input
                tg.create_task(
                    input_worker(input, config, is_voice_input=False),
                    name="input_worker",
                )

            tg.create_task(
                chat_worker(
                    chat_agent,
                    chat_data,
                    speech_data,
                ),
                name="chat_worker",
            )
            tg.create_task(
                speech_worker(
                    voice,
                    speech_data,
                    input,
                ),
                name="speech_worker",
            )

            # No more input loop here - input worker task handles it
    except* (asyncio.CancelledError, EOFError, KeyboardInterrupt) as e:
        # Handle expected termination conditions
        for _e in e.exceptions:
            if isinstance(_e, KeyboardInterrupt):
                logger.info("* Stopping...")
            else:
                logger.info("Cancelled.")
    except* ValueError as e:
        # Handle value errors specifically
        for _e in e.exceptions:
            logger.error(f"ValueError: {_e}")
            traceback.print_exc()
    except* Exception as e:
        # Catch any other exceptions
        for _e in e.exceptions:
            logger.error(f"Unexpected error: {_e}")
            traceback.print_exc()
    finally:
        # Properly cleanup tasks and queues
        await cleanup_tasks(
            speech_data,
            chat_data,
            audio_data if args.input == INPUT.VOICE.value else None,
        )

        logger.info("Stopped.")
        pa.terminate()


def make_voice(output: OUTPUT, pa, audio_lock, lang: str, debug: bool) -> Voice:
    if output == OUTPUT.SPEECH.value:
        return KokoroVoice(pa, audio_lock, lang=lang, debug=debug)

    assert output == OUTPUT.TEXT.value
    return TextVoice()


def make_chat_agent(
    config: Config,
    lang: str,
    thread_id: str,
) -> ChatAgentLike:
    """Create a chat agent based on the configuration."""

    if config.anthropic_api_key:
        return AnthropicChatAgent(
            api_key=config.anthropic_api_key,
            model_name=config.anthropic_model,
            lang=lang,
            thread_id=thread_id,
        )
    elif config.openai_api_key:
        return OpenAIChatAgent(
            api_key=config.openai_api_key,
            model=config.openai_model,
            lang=lang,
            thread_id=thread_id,
        )
    elif config.ollama_host:
        return OllamaChatAgent(
            host=config.ollama_host,
            port=config.ollama_port,
            model=config.ollama_model,
            lang=lang,
            thread_id=thread_id,
        )
    else:
        raise ValueError("No API key set.")


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
