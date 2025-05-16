import asyncio
import logging
import traceback

from input.transcribe import TranscriberLike
from model.event import EventData
from repository.chat import ChatAgentLike
from repository.input import InputLike
from speech.kokoro import Voice

logger = logging.getLogger(__name__)


async def input_worker(input_handler, config=None, is_voice_input=True):
    """Worker task for continuously receiving input (either voice or text)."""
    try:
        while True:
            if is_voice_input and config:
                await input_handler.receive(
                    silence_duration=config.silence_duration,
                    min_speech_duration=config.min_speech_duration,
                    silence_threshold=config.silence_threshold,
                )
            else:
                await input_handler.receive()
    except Exception as e:
        logger.error(f"input_worker Error: {e}\n{traceback.format_exc()}")


async def transcribe_worker(
    transcriber: TranscriberLike,
    audio_data: EventData,
    chat_data: EventData,
    sample_rate: int,
    input: InputLike,
):
    try:
        while True:
            audio_array = await audio_data.get()
            logger.info("* Transcribing...")

            result = transcriber.transcribe_buffer(audio_array, sample_rate)

            # Skip if result doesn't contain valid word
            if result is None or not any(char.isalnum() for char in result):
                logger.warning(f"{result} does not contain valid word.")

                # Signal that the audio data has been processed
                input.notify_response_complete()

                audio_data.task_done()
                continue

            # Print the recognized text
            logger.info(f"Transcript: {result}")
            if result:
                await chat_data.set(result)

            audio_data.task_done()
    except Exception as e:
        logger.error(f"transcribe_worker Error: {e}")


async def chat_worker(
    chat_agent: ChatAgentLike, chat_data: EventData, speech_data: EventData
):
    try:
        while True:
            message = await chat_data.get()

            logger.info("* Chatting...")
            response_received = False
            try:
                async for response in chat_agent.chat(message):
                    await speech_data.set(response)
                    response_received = True
            except Exception as e:
                logger.error(f"Chat Error: {e}\n{traceback.format_exc()}")

            # Signal that one full response was received and processed
            # Add a special marker to indicate end of the full response
            if response_received:
                await speech_data.set("__RESPONSE_COMPLETE__")

            chat_data.task_done()
    except Exception as e:
        logger.error(f"chat_worker Error: {e}\n{traceback.format_exc()}")


async def speech_worker(voice: Voice, speech_data: EventData, input: InputLike):
    try:
        while True:
            speech = await speech_data.get()

            # Check if this is our special marker
            if speech == "__RESPONSE_COMPLETE__":
                # Signal completion to input to show the next prompt
                input.notify_response_complete()
            else:
                await voice.say(speech)

            speech_data.task_done()
    except Exception as e:
        logger.error(f"speech_worker Error: {e}\n{traceback.format_exc()}")


async def cleanup_tasks(
    speech_data: EventData, chat_data: EventData, audio_data: EventData = None
):
    """Cancel all tasks and wait for them to complete."""

    # Give tasks time to respond to cancellation
    await asyncio.sleep(0.1)

    # Nothing to clear for EventData objects, as they don't have queues
    # Just ensure all data is marked as processed
    speech_data.task_done()
    chat_data.task_done()

    if audio_data is not None:
        audio_data.task_done()
