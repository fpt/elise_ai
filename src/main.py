import argparse
import asyncio
import datetime
import logging
import traceback
from enum import Enum
from typing import Optional

import pyaudio

from agent.chat import (
    AnthropicChatAgent,
    OllamaChatAgent,
    OpenAIChatAgent,
)
from config import Config
from input.audio import RATE, AudioInput
from input.text import TextInput
from input.transcribe import Transcriber, TranscriberLike
from logging_config import setup_logging
from model.event import EventData
from model.pipeline import PipelineController
from protocol.chat import ChatAgentLike
from protocol.input import InputLike
from speech.kokoro import KokoroVoice, Voice
from speech.text import TextVoice
from tasks import (
    chat_worker,
    input_worker,
    speech_worker,
    transcribe_worker,
)

logger = logging.getLogger(__name__)


class INPUT(Enum):
    VOICE = "voice"
    TEXT = "text"

    def __str__(self):
        return self.value


class OUTPUT(Enum):
    SPEECH = "speech"
    TEXT = "text"

    def __str__(self):
        return self.value


def generate_thread_id() -> str:
    return str(datetime.datetime.now().timestamp())


async def run_pipeline_cycle(
    chat_agent: ChatAgentLike,
    voice: Voice,
    config: Config,
    pa: pyaudio.PyAudio,
    audio_lock: asyncio.Lock,
    input_type: str,
    lang: str,
    debug: bool,
) -> None:
    """
    Run a single pipeline cycle from input to output.

    Creates input handlers first, then creates a TaskGroup to manage tasks,
    and finally creates the PipelineController after input is received.
    Uses the context manager pattern to automatically handle cleanup.
    """

    # Create the appropriate input handler first
    input_handler: Optional[InputLike] = None
    transcriber: Optional[TranscriberLike] = None

    # Create events for the input handlers
    audio_event = EventData()
    input_event = EventData()

    if input_type == INPUT.VOICE.value:
        input_handler = AudioInput(pa, audio_event, audio_lock, debug=debug)
        transcriber = Transcriber(model_name=config.whisper_model, force_language=lang)
    elif input_type == INPUT.TEXT.value:
        input_handler = TextInput(input_event)
    else:
        raise ValueError(f"Unsupported input type: {input_type}")

    if input_handler is None:
        raise ValueError("Input handler is not initialized.")

    # First create the TaskGroup to manage tasks
    async with asyncio.TaskGroup() as tg:
        # Start the input worker to get input
        if input_type == INPUT.VOICE.value and transcriber is not None:
            # Start the input worker for voice input
            tg.create_task(
                input_worker(input_handler, config, is_voice_input=True),
                name="input_worker_initial",
            )

            # Wait for audio input to be received
            audio_data = await audio_event.get()

            # Now create the PipelineController with the audio data
            async with PipelineController(audio_data=audio_data) as ctlr:
                # Create the transcribe worker to process the audio
                tg.create_task(
                    transcribe_worker(ctlr, transcriber, RATE),
                    name="transcribe_worker",
                )

                # Create chat and speech worker tasks
                tg.create_task(
                    chat_worker(ctlr, chat_agent),
                    name="chat_worker",
                )
                tg.create_task(
                    speech_worker(ctlr, voice),
                    name="speech_worker",
                )

                # Wait for pipeline completion
                await ctlr.wait_for_completion()

        elif input_type == INPUT.TEXT.value:
            # Start the input worker for text input
            tg.create_task(
                input_worker(input_handler, config, is_voice_input=False),
                name="input_worker_initial",
            )

            # Wait for text input to be received
            text_input = await input_event.get()

            # Now create the PipelineController with the text input
            async with PipelineController(text_input=text_input) as ctlr:
                # Create chat and speech worker tasks
                tg.create_task(
                    chat_worker(ctlr, chat_agent),
                    name="chat_worker",
                )
                tg.create_task(
                    speech_worker(ctlr, voice),
                    name="speech_worker",
                )

                # Wait for pipeline completion
                await ctlr.wait_for_completion()


async def main() -> None:
    """Main function to run the application."""
    parser = argparse.ArgumentParser(
        description="Real-time speech transcription and chat."
    )
    parser.add_argument(
        "--lang", type=str, default="en", help="Language for transcription"
    )
    parser.add_argument(
        "--input", type=str, choices=INPUT, default=INPUT.VOICE.value, help="Input type"
    )
    parser.add_argument(
        "--output",
        choices=OUTPUT,
        type=str,
        default=OUTPUT.SPEECH.value,
        help="Output type",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode for logging"
    )
    args = parser.parse_args()

    # Set up logging with debug flag
    setup_logging(args.debug)

    # Set log level for this specific logger as well
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    config = Config.from_env().validate()
    pa = pyaudio.PyAudio()
    audio_lock = asyncio.Lock()

    # Create a new chat agent with a new thread ID for this conversation
    chat_agent = make_chat_agent(
        config,
        args.lang,
        generate_thread_id(),
    )

    # Create the voice handler
    voice = make_voice(args.output, pa, audio_lock, args.lang, args.debug)

    try:
        # Run pipeline cycles continuously
        while True:
            await run_pipeline_cycle(
                chat_agent,
                voice,
                config,
                pa,
                audio_lock,
                args.input,
                args.lang,
                args.debug,
            )
    except* (asyncio.CancelledError, EOFError, KeyboardInterrupt) as term_errors:
        # Handle expected termination conditions
        for _e in term_errors.exceptions:
            if isinstance(_e, KeyboardInterrupt):
                logger.info("* Stopping...")
            else:
                logger.info("Cancelled.")
    except* ValueError as e:
        # Handle value errors specifically
        logger.error(f"ValueError: {e}")
        traceback.print_exc()
    except* Exception as e:
        logger.error(f"Unexpected error: {e}")
        traceback.print_exc()
    finally:
        logger.info("Stopped.")
        pa.terminate()


def make_voice(output: OUTPUT, pa, audio_lock, lang: str, debug: bool) -> Voice:
    if output == OUTPUT.SPEECH.value:
        return KokoroVoice(pa, audio_lock, lang=lang, debug=debug)

    assert output == OUTPUT.TEXT.value
    return TextVoice()


def make_chat_agent(
    config: Config,
    lang: str,
    thread_id: str,
) -> ChatAgentLike:
    """Create a chat agent based on the configuration."""

    if config.anthropic_api_key:
        return AnthropicChatAgent(
            api_key=config.anthropic_api_key,
            model_name=config.anthropic_model,
            lang=lang,
            thread_id=thread_id,
        )
    elif config.openai_api_key:
        return OpenAIChatAgent(
            api_key=config.openai_api_key,
            model=config.openai_model,
            lang=lang,
            thread_id=thread_id,
        )
    elif config.ollama_host:
        return OllamaChatAgent(
            host=config.ollama_host,
            port=config.ollama_port,
            model=config.ollama_model,
            lang=lang,
            thread_id=thread_id,
        )
    else:
        raise ValueError("No API key set.")


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
