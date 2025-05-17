import asyncio
import datetime
import logging
import os

import numpy as np
import pyaudio
import soundfile as sf
import webrtcvad

INT16_MAX = 32768
CHUNK = 960  # 1024  # Must be larger than processing time.
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 32000  # 44100

logger = logging.getLogger(__name__)


class AudioInput:
    def __init__(self, pa, audio_data, audio_lock, debug=False):
        self.debug = debug
        self.pa = pa
        self.audio_data = audio_data  # EventData instead of Queue
        self.audio_lock = audio_lock
        vad = webrtcvad.Vad()
        vad.set_mode(3)
        self.vad = vad

    async def receive(
        self,
        wait_event: asyncio.Event,
        silence_duration=1.5,
        min_speech_duration=1.5,
        silence_threshold=0.015,
    ):
        frames = []
        silent_chunks = 0
        is_speaking = False

        # Wait until we've received a response before showing the next prompt
        await wait_event.wait()
        logging.info("* Listening for speech...")
        wait_event.clear()

        # Keep listening until we detect speech followed by silence
        stream = self.pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        while True:
            await self.audio_lock.acquire()
            try:
                audio_data = stream.read(CHUNK, exception_on_overflow=False)
            except Exception as e:
                logger.error(f"Error reading audio data: {e}")
                break
            finally:
                self.audio_lock.release()

            # Wait for a bit to avoid blocking
            await asyncio.sleep(0.02)

            # If we detect sound
            # if not _is_silence(audio_data, silence_threshold):
            if self.is_speaking(audio_data, silence_threshold):
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
            # Convert frames to audio buffer
            audio_buffer = b"".join(frames)
            if self.debug:
                _dump_audio(audio_buffer)

            # Convert to numpy array for Whisper - use float32 to match Whisper's expected dtype
            audio_array = (
                np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32)
                / INT16_MAX
            )

            # Process with Whisper
            await self.audio_data.set(audio_array)

    def is_speaking(self, audio_data: bytes, silence_threshold: float) -> bool:
        is_speech: bool = False
        if _is_silence(audio_data, silence_threshold):
            return False
        try:
            is_speech = self.vad.is_speech(audio_data, RATE)
        except Exception as e:
            logger.error(f"Error in VAD: {e}")
            is_speech = False
        return is_speech


AUDIO_DUMP_DIR = "logs"
AUDIO_DUMP_FORMAT = "audio_{:%Y%m%d_%H%M%S}.wav"


def _dump_audio(audio_data: bytes):
    """
    Dump audio data to a file with timestamp.
    Args:
    audio_data (bytes): Raw audio data
    """
    os.makedirs(AUDIO_DUMP_DIR, exist_ok=True)
    timestamp = datetime.datetime.now()
    filename = os.path.join(AUDIO_DUMP_DIR, AUDIO_DUMP_FORMAT.format(timestamp))
    sf.write(filename, np.frombuffer(audio_data, dtype=np.int16), RATE)


def _is_silence(audio_data: bytes, threshold: float) -> bool:
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
    rms: float = np.sqrt(mn)
    return rms < INT16_MAX * threshold
