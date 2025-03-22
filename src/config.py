import os
from dataclasses import dataclass

from dotenv import load_dotenv

SILENCE_THRESHOLD = 0.015  # Adjust this value based on your environment
SILENCE_DURATION = 1.5  # Seconds of silence to consider speech ended
MIN_SPEECH_DURATION = 1.0  # Minimum duration of speech to consider valid
WHISPER_MODEL = "turbo"
ANTHROPIC_MODEL_NAME = "claude-3-7-sonnet-latest"
OPENAI_MODEL_NAME = "gpt-4o"
OLLAMA_MODEL_NAME = "llama3.2"

load_dotenv()


@dataclass
class Config:
    anthropic_api_key: str
    anthropic_model: str
    openai_api_key: str
    openai_model: str
    ollama_host: str
    ollama_port: str
    ollama_model: str
    whisper_model: str
    silence_threshold: float
    silence_duration: float
    min_speech_duration: float

    @classmethod
    def from_env(cls):
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        anthropic_model = os.getenv("ANTHROPIC_MODEL", ANTHROPIC_MODEL_NAME)
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        openai_model = os.getenv("OPENAI_MODEL", OPENAI_MODEL_NAME)
        ollama_host = os.getenv("OLLAMA_HOST", "")
        ollama_port = os.getenv("OLLAMA_PORT", "")
        ollama_model = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL_NAME)
        whisper = os.getenv("WHISPER_MODEL", WHISPER_MODEL)
        silence_thresh = float(os.getenv("SILENCE_THRESHOLD", SILENCE_THRESHOLD))
        silence_dur = float(os.getenv("SILENCE_DURATION", SILENCE_DURATION))
        min_speech = float(os.getenv("MIN_SPEECH_DURATION", MIN_SPEECH_DURATION))

        return cls(
            anthropic_api_key=anthropic_api_key,
            anthropic_model=anthropic_model,
            openai_api_key=openai_api_key,
            openai_model=openai_model,
            ollama_host=ollama_host,
            ollama_port=ollama_port,
            ollama_model=ollama_model,
            whisper_model=whisper,
            silence_threshold=silence_thresh,
            silence_duration=silence_dur,
            min_speech_duration=min_speech,
        )

    def validate(self):
        # One ov Anthropic, OpenAI, and Ollama must be set
        if not any([self.anthropic_api_key, self.openai_api_key, self.ollama_host]):
            raise ValueError(
                "Please set one of ANTHROPIC_API_KEY, OPENAI_API_KEY, or OLLAMA_HOST environment variables."
            )
        # If Anthropic is set, the model name must be set
        if self.anthropic_api_key and not self.anthropic_model:
            raise ValueError("Please set the ANTHROPIC_MODEL environment variable.")
        # If OpenAI is set, the model name must be set
        if self.openai_api_key and not self.openai_model:
            raise ValueError("Please set the OPENAI_MODEL environment variable.")
        # If Ollama is set, the model name must be set
        if self.ollama_host and not (self.ollama_port and self.ollama_model):
            raise ValueError(
                "Please set the OLLAMA_PORT and OLLAMA_MODEL environment variable."
            )

        return self
