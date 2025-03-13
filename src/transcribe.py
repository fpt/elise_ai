from typing import Optional, Protocol

import numpy as np
import pyaudio
import whisper

CHUNK = 8196  # Must be larger than processing time.
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
SILENCE_THRESHOLD = 500  # Adjust this value based on your environment
SILENCE_DURATION = 1.5  # Seconds of silence to consider speech ended


class Transcriber(Protocol):
    model: whisper.Whisper
    force_language: str

    def __init__(self, model_name="turbo", force_language=None):
        self.model = whisper.load_model(model_name)
        self.force_language = force_language

    def transcribe_buffer(self, audio_array, sample_rate) -> Optional[str]:
        """
        Transcribe audio directly from buffer without saving to file.
        """
        # Resample if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            # Simple resampling - in production you might want to use a proper resampling library
            target_length = int(len(audio_array) * 16000 / sample_rate)
            audio_array = np.interp(
                np.linspace(0, len(audio_array), target_length),
                np.arange(0, len(audio_array)),
                audio_array,
            )

        # Pad or trim to fit Whisper's expected input
        audio_array = audio_array.astype(np.float32)
        # print(f"Audio shape: {audio_array.shape} type: {audio_array.dtype}")
        audio_array = whisper.pad_or_trim(audio_array)
        # print(f"Audio shape: {audio_array.shape} type: {audio_array.dtype}")

        # Make log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(
            audio_array, n_mels=self.model.dims.n_mels
        ).to(self.model.device)
        # print(f"Mel shape: {mel.shape} type: {mel.dtype}")

        # Detect language
        _, probs = self.model.detect_language(mel)
        detected_language = max(probs, key=probs.get)
        print(f"Detected language: {detected_language}")
        if detected_language not in ("en", "ja"):
            print(f"Unsupported language: {detected_language}")
            return None

        if self.force_language:
            detected_language = self.force_language

        # Decode the audio
        options = whisper.DecodingOptions(language=detected_language, fp16=False)
        result = whisper.decode(self.model, mel, options)

        return result.text
