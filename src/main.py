import argparse
import asyncio

import numpy as np
import pyaudio

from audio import is_silence
from chat import ChatAgent, ChatAnthropicAgent
from kk_voice import KokoroVoice, Voice
from transcribe import Transcriber

INT16_MAX = 32768
CHUNK = 1024  # Must be larger than processing time.
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
SILENCE_THRESHOLD = 32768 * 0.02  # Adjust this value based on your environment
SILENCE_DURATION = 1.5  # Seconds of silence to consider speech ended
MIN_SPEECH_DURATION = 1.5  # Minimum duration of speech to consider valid
WHISPER_MODEL = "turbo"


async def process_audio(pa, audio_queue, audio_lock, chat_queue):
    print("* Listening for speech...")

    try:
        while True:
            frames = []
            silent_chunks = 0
            is_speaking = False

            # Keep listening until we detect speech followed by silence
            stream = pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )
            while True:
                await audio_lock.acquire()
                try:
                    audio_data = stream.read(CHUNK, exception_on_overflow=False)
                finally:
                    audio_lock.release()

                # If we detect sound
                if not is_silence(audio_data, SILENCE_THRESHOLD):
                    is_speaking = True
                    silent_chunks = 0
                    frames.append(audio_data)
                # If we detect silence after speech
                elif is_speaking:
                    frames.append(audio_data)
                    silent_chunks += 1

                    # If silence duration exceeds our threshold, stop recording
                    if silent_chunks * CHUNK / RATE > SILENCE_DURATION:
                        if len(frames) * CHUNK / RATE < MIN_SPEECH_DURATION:
                            frames = []
                            is_speaking = False
                            continue
                        break
            stream.stop_stream()
            stream.close()

            # If we captured some speech
            if frames and is_speaking:
                print("* Queue speech...")

                # Convert frames to audio buffer
                audio_buffer = b"".join(frames)

                # Convert to numpy array for Whisper - use float32 to match Whisper's expected dtype
                audio_array = (
                    np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32)
                    / INT16_MAX
                )

                # Process with Whisper
                audio_queue.put_nowait(audio_array)
                await asyncio.sleep(0.1)

                print("* Listening for speech...")

    except KeyboardInterrupt:
        print("* Stopping...")

    print("* Stopped.")


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
    args = parser.parse_args()

    pa = pyaudio.PyAudio()
    audio_lock = asyncio.Lock()
    # voice = MacVoice(audio_lock, lang=args.lang)
    voice = KokoroVoice(pa, audio_lock, lang=args.lang)
    transcriber = Transcriber(model_name=WHISPER_MODEL, force_language=args.lang)
    chat_agent = ChatAnthropicAgent(lang=args.lang)
    audio_queue = asyncio.Queue()
    chat_queue = asyncio.Queue()
    speech_queue = asyncio.Queue()

    task = asyncio.create_task(
        transcribe_worker(transcriber, audio_queue, chat_queue, RATE)
    )
    task2 = asyncio.create_task(chat_worker(chat_agent, chat_queue, speech_queue))
    task3 = asyncio.create_task(speech_worker(voice, speech_queue))

    try:
        await process_audio(pa, audio_queue, audio_lock, chat_queue)
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
    asyncio.run(main())
