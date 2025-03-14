from dataclasses import dataclass

SILENCE_THRESHOLD = 0.02  # Adjust this value based on your environment
SILENCE_DURATION = 1.5  # Seconds of silence to consider speech ended
MIN_SPEECH_DURATION = 1.5  # Minimum duration of speech to consider valid
WHISPER_MODEL = "turbo"
ANTHROPIC_MODEL_NAME = "claude-3-7-sonnet-latest"


@dataclass
class Config:
    anthropic_api_key: str
    anthropic_model: str = ANTHROPIC_MODEL_NAME
    whisper_model: str = WHISPER_MODEL
    silence_threshold: float = SILENCE_THRESHOLD
    silence_duration: float = SILENCE_DURATION
    min_speech_duration: float = MIN_SPEECH_DURATION
