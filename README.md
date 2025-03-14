# Elise AI Voice Chat

A real-time voice assistant that uses OpenAI's Whisper for speech recognition, LangChain with Claude 3.7 for conversation, and Kokoro text-to-speech for responses.

## Overview

This project creates a voice-based conversational assistant that:

1. Listens for speech using your microphone
2. Detects when you start and stop speaking
3. Transcribes your speech using OpenAI's Whisper model
4. Processes your request using LangChain with Claude 3.7
5. Speaks the response using Kokoro text-to-speech

The system is designed to minimize latency in conversational AI by using efficient audio processing, speech detection, and response generation.

```mermaid
flowchart LR
    Mic[Microphone Input] -- audio data --> VAD[Voice Activity Detection]
    VAD -- audio_queue --> Transcribe[Whisper Transcription]
    Transcribe -- chat_queue --> ChatAgent[Claude 3.7 LLM]
    ChatAgent -- speech_queue --> Voice[Text-to-Speech]
    Voice -- audio output --> Speaker[Speaker Output]
    
    subgraph Audio Processing
        Mic
        VAD
        Speaker
    end
    
    subgraph Language Processing
        Transcribe
        ChatAgent
        Voice
    end
```


## Components

### Speech Recognition (ASR)

- Uses OpenAI's Whisper Turbo model for fast and accurate speech recognition
- Supports multiple languages with automatic language detection
- Processes audio in real-time without saving to disk

### Voice Activity Detection (VAD)

- Implements RMS-based silence detection to determine when speech starts and ends
- Configurable silence threshold and duration parameters
- Efficiently captures complete utterances without unnecessary audio

### Language Model (LLM)

- Uses LangChain with Claude 3.7 Sonnet for natural conversation
- Maintains conversation context for coherent multi-turn interactions
- Configured for concise responses suitable for voice interaction

### Text-to-Speech (TTS)

- Uses [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) text-to-speech capabilities

## Requirements

- MacOS
- Anthropic API Key

## Installation

1. Install required components.

```bash
brew install uv portaudio
# optional but recommended.
brew install make direnv
```

2. Clone this repository
3. Install dependencies using uv:

```bash
uv sync
```

4. Make `.env` file

```
ANTHROPIC_API_KEY=<your_anthropic_api_key>
```

## Usage

Run the application using the provided Makefile:

For English conversation:

```bash
make run
```

For Japanese conversation:

```bash
make run-ja
```

To stop the application, press Ctrl+C.

## Development

The project includes several Makefile commands to help with development:

- `make run`: Run the application
- `make test`: Run unit tests
- `make fmt`: Format code using ruff
- `make lint` or `make check`: Run linting checks
- `make fix`: Automatically fix linting issues
- `make help`: Display available commands

## Project Structure

- `src/main.py`: Main application logic and audio processing
- `src/transcribe.py`: Whisper integration for speech recognition
- `src/chat.py`: LangChain and Claude integration for conversation
- `src/voice.py`: Text-to-speech functionality

## References

- [OpenAI Whisper](https://github.com/openai/whisper)
- [LangChain](https://python.langchain.com/docs/get_started/introduction)
- [Claude API](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
- [Optimizing Latency for Conversational AI](https://elevenlabs.io/blog/how-do-you-optimize-latency-for-conversational-ai)

## TODO

- Ctrl-C doesn't stop asycio tasks correctly.
    - https://stackoverflow.com/questions/48562893/how-to-gracefully-terminate-an-asyncio-script-with-ctrl-c
- Anthropic responds 529 if requests get longer
```
INFO: HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 529 "
INFO: Retrying request to /v1/messages in 0.493675 seconds
INFO: HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 529 "
INFO: Retrying request to /v1/messages in 0.793765 seconds
INFO: HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 529 "
ERROR: chat_worker Error: Error code: 529 - {'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'Overloaded'}}
```

## License

[Add license information here]
