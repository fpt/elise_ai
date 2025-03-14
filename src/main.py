import argparse
import asyncio
import logging
import os
import traceback

import pyaudio

from audio import RATE, AudioInput
from chat import ChatAgent, ChatAnthropicAgent
from config import Config
from kk_voice import KokoroVoice, Voice
from logging_config import setup_logging
from transcribe import Transcriber

setup_logging()

logger = logging.getLogger(__name__)


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
            speech = await chat_agent.chat(message)
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


async def main():
    parser = argparse.ArgumentParser(
        description="Real-time speech transcription and chat."
    )
    parser.add_argument(
        "--lang", type=str, default="en", help="Language for transcription"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode for logging"
    )
    args = parser.parse_args()

    # Set log level based on --debug option
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_api_key is None:
        raise ValueError("Please set the ANTHROPIC_API_KEY environment variable.")
    config = Config(anthropic_api_key)

    pa = pyaudio.PyAudio()
    audio_lock = asyncio.Lock()
    audio_queue = asyncio.Queue()
    chat_queue = asyncio.Queue()
    speech_queue = asyncio.Queue()

    input = AudioInput(pa, audio_queue, audio_lock, debug=args.debug)
    transcriber = Transcriber(model_name=config.whisper_model, force_language=args.lang)
    chat_agent = ChatAnthropicAgent(
        api_key=config.anthropic_api_key, model=config.anthropic_model, lang=args.lang
    )
    voice = KokoroVoice(pa, audio_lock, lang=args.lang, debug=args.debug)

    task = asyncio.create_task(
        transcribe_worker(transcriber, audio_queue, chat_queue, RATE)
    )
    task2 = asyncio.create_task(chat_worker(chat_agent, chat_queue, speech_queue))
    task3 = asyncio.create_task(speech_worker(voice, speech_queue))

    try:
        await input.process_audio(
            silence_duration=config.silence_duration,
            min_speech_duration=config.min_speech_duration,
            silence_threshold=config.silence_threshold,
        )
    except asyncio.CancelledError:
        logger.info("Cancelled.")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        logger.info("Done.")

    task3.cancel()
    await speech_queue.join()
    task2.cancel()
    await chat_queue.join()
    task.cancel()
    await audio_queue.join()
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
