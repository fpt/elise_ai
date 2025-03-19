import asyncio
import logging

from agent.chat import ChatAgent
from input.transcribe import Transcriber
from speech.kokoro import Voice

logger = logging.getLogger(__name__)


async def transcribe_worker(
    transcriber: Transcriber, audio_queue, chat_queue, sample_rate
):
    try:
        while True:
            audio_array = await audio_queue.get()
            logger.info("* Transcribing...")

            result = transcriber.transcribe_buffer(audio_array, sample_rate)

            # Skip if result doesn't contain valid word
            if result is None or not any(char.isalnum() for char in result):
                logger.warning(f"{result} does not contain valid word.")
                audio_queue.task_done()
                continue

            # Print the recognized text
            logger.info(f"Transcript: {result}")
            if result:
                chat_queue.put_nowait(result)

            audio_queue.task_done()
    except Exception as e:
        logger.error(f"transcribe_worker Error: {e}")


async def chat_worker(chat_agent: ChatAgent, chat_queue, speech_queue):
    try:
        while True:
            message = await chat_queue.get()

            logger.info("* Chatting...")
            speech = chat_agent.chat(message)
            speech_queue.put_nowait(speech)

            chat_queue.task_done()
    except Exception as e:
        logger.error(f"chat_worker Error: {e}")


async def speech_worker(voice: Voice, speech_queue):
    try:
        while True:
            speech = await speech_queue.get()
            await voice.say(speech)
            speech_queue.task_done()
    except Exception as e:
        logger.error(f"speech_worker Error: {e}")


async def cleanup_tasks(tasks, speech_queue, chat_queue, audio_queue=None):
    """Cancel all tasks and wait for them to complete."""
    for task in tasks:
        task.cancel()

    # Give tasks time to respond to cancellation
    await asyncio.sleep(0.1)

    # Wait for all tasks to complete cancellation
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    # Clear out any remaining items in the queues
    while not speech_queue.empty():
        speech_queue.get_nowait()
        speech_queue.task_done()

    while not chat_queue.empty():
        chat_queue.get_nowait()
        chat_queue.task_done()

    if audio_queue is not None:
        while not audio_queue.empty():
            audio_queue.get_nowait()
            audio_queue.task_done()
