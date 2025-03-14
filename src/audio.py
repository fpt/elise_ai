import asyncio

import numpy as np
import pyaudio

INT16_MAX = 32768
CHUNK = 1024  # Must be larger than processing time.
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100


async def process_audio(
    pa,
    audio_queue,
    audio_lock,
    silence_duration=1.5,
    min_speech_duration=1.5,
    silence_threshold=0.02,
):
    try:
        while True:
            frames = []
            silent_chunks = 0
            is_speaking = False

            print("* Listening for speech...")

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
                if not is_silence(audio_data, silence_threshold):
                    is_speaking = True
                    silent_chunks = 0
                    frames.append(audio_data)
                # If we detect silence after speech
                elif is_speaking:
                    frames.append(audio_data)
                    silent_chunks += 1

                    # If silence duration exceeds our threshold, stop recording
                    if silent_chunks * CHUNK / RATE > silence_duration:
                        if len(frames) * CHUNK / RATE < min_speech_duration:
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

    except KeyboardInterrupt:
        print("* Stopping...")

    print("* Stopped.")


def is_silence(audio_data: bytes, threshold: float) -> bool:
    """
    Determine if the audio chunk is silence based on amplitude threshold.
    Args:
        audio_data (bytes): Raw audio data of int16 samples
        threshold (float): RMS amplitude threshold
    Returns:
        bool: True if silence, False otherwise
    """
    # Convert audio data to numpy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    # Check if array is empty or contains only zeros
    if len(audio_array) == 0 or np.all(audio_array == 0):
        return True

    # Calculate the RMS value (Use float32 to avoid overflow)
    mn = np.mean(np.square(audio_array.astype(np.float32)))
    rms = np.sqrt(mn)
    return rms < INT16_MAX * threshold
