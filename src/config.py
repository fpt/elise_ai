import os
from dataclasses import dataclass

SILENCE_THRESHOLD = 0.015  # Adjust this value based on your environment
SILENCE_DURATION = 1.5  # Seconds of silence to consider speech ended
MIN_SPEECH_DURATION = 1.0  # Minimum duration of speech to consider valid
WHISPER_MODEL = "turbo"
ANTHROPIC_MODEL_NAME = "claude-3-7-sonnet-latest"


@dataclass
class Config:
    anthropic_api_key: str
    anthropic_model: str
    whisper_model: str
    silence_threshold: float
    silence_duration: float
    min_speech_duration: float

    @classmethod
    def from_env(cls):
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        model = os.getenv("ANTHROPIC_MODEL", ANTHROPIC_MODEL_NAME)
        whisper = os.getenv("WHISPER_MODEL", WHISPER_MODEL)
        silence_thresh = float(os.getenv("SILENCE_THRESHOLD", SILENCE_THRESHOLD))
        silence_dur = float(os.getenv("SILENCE_DURATION", SILENCE_DURATION))
        min_speech = float(os.getenv("MIN_SPEECH_DURATION", MIN_SPEECH_DURATION))

        return cls(
            anthropic_api_key=api_key,
            anthropic_model=model,
            whisper_model=whisper,
            silence_threshold=silence_thresh,
            silence_duration=silence_dur,
            min_speech_duration=min_speech,
        )

    def validate(self):
        if not self.anthropic_api_key:
            raise ValueError("Please set the ANTHROPIC_API_KEY environment variable.")
        return self
