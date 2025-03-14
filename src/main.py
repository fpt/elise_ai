import argparse
import asyncio
import os
import traceback

import pyaudio

from audio import RATE, AudioInput
from chat import ChatAgent, ChatAnthropicAgent
from config import Config
from kk_voice import KokoroVoice, Voice
from transcribe import Transcriber


async def transcribe_worker(
    transcriber: Transcriber, audio_queue, chat_queue, sample_rate
):
    try:
        while True:
            audio_array = await audio_queue.get()
            print("* Transcribing...")

            result = transcriber.transcribe_buffer(audio_array, sample_rate)

            # Skip if result doesn't contain valid word
            if result is None or not any(char.isalnum() for char in result):
                print(f"{result} does not contain valid word.")
                audio_queue.task_done()
                continue

            # Print the recognized text
            print(f"Transcript: {result}")
            if result:
                chat_queue.put_nowait(result)

            audio_queue.task_done()
    except Exception as e:
        print(f"transcribe_worker Error: {e}")


async def chat_worker(chat_agent: ChatAgent, chat_queue, speech_queue):
    try:
        while True:
            message = await chat_queue.get()

            print("* Chatting...")
            speech = await chat_agent.chat(message)
            speech_queue.put_nowait(speech)

            chat_queue.task_done()
    except Exception as e:
        print(f"chat_worker Error: {e}")


async def speech_worker(voice: Voice, speech_queue):
    try:
        while True:
            speech = await speech_queue.get()
            # print(f"* Saying... {speech}")
            await voice.say(speech)
            speech_queue.task_done()
    except Exception as e:
        print(f"speech_worker Error: {e}")


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
        print("Cancelled.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Done.")

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
        print("Received exit, exiting")
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
    finally:
        loop.close()
