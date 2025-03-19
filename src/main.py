import argparse
import asyncio
import datetime
import logging
import traceback
from enum import Enum

import pyaudio

from agent.chat import AnthropicChatAgent, ChatAgent
from config import Config
from input.audio import RATE, AudioInput, Input
from input.text import TextInput
from input.transcribe import Transcriber
from logging_config import setup_logging
from speech.kokoro import KokoroVoice, Voice
from speech.text import TextVoice

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


async def transcribe_worker(
    transcriber: Transcriber, audio_queue, chat_queue, sample_rate
):
    try:
        while True:
            audio_array = await audio_queue.get()
            logger.info("* Transcribing...")

            result = transcriber.transcribe_buffer(audio_array, sample_rate)

            # Skip if result doesn't contain valid word
            if result is None or not any(char.isalnum() for char in result):
                logger.warning(f"{result} does not contain valid word.")
                audio_queue.task_done()
                continue

            # Print the recognized text
            logger.info(f"Transcript: {result}")
            if result:
                chat_queue.put_nowait(result)

            audio_queue.task_done()
    except Exception as e:
        logger.error(f"transcribe_worker Error: {e}")


async def chat_worker(chat_agent: ChatAgent, chat_queue, speech_queue):
    try:
        while True:
            message = await chat_queue.get()

            logger.info("* Chatting...")
            speech = chat_agent.chat(message)
            speech_queue.put_nowait(speech)

            chat_queue.task_done()
    except Exception as e:
        logger.error(f"chat_worker Error: {e}")


async def speech_worker(voice: Voice, speech_queue):
    try:
        while True:
            speech = await speech_queue.get()
            await voice.say(speech)
            speech_queue.task_done()
    except Exception as e:
        logger.error(f"speech_worker Error: {e}")


async def cleanup_tasks(tasks, speech_queue, chat_queue, audio_queue=None):
    """Cancel all tasks and wait for them to complete."""
    for task in tasks:
        task.cancel()

    # Give tasks time to respond to cancellation
    await asyncio.sleep(0.1)

    # Wait for all tasks to complete cancellation
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    # Clear out any remaining items in the queues
    while not speech_queue.empty():
        speech_queue.get_nowait()
        speech_queue.task_done()

    while not chat_queue.empty():
        chat_queue.get_nowait()
        chat_queue.task_done()

    if audio_queue is not None:
        while not audio_queue.empty():
            audio_queue.get_nowait()
            audio_queue.task_done()


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

    chat_agent: ChatAgent = AnthropicChatAgent(
        api_key=config.anthropic_api_key,
        model_name=config.anthropic_model,
        lang=args.lang,
        thread_id=generate_thread_id(),
    )
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
