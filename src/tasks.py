import asyncio
import logging
import traceback

from input.transcribe import TranscriberLike
from model.pipeline import PipelineController
from protocol.chat import ChatAgentLike
from speech.kokoro import Voice

logger = logging.getLogger(__name__)


async def input_worker(
    ctlr: PipelineController, input_handler, config=None, is_voice_input=True
):
    """Worker task for continuously receiving input (either voice or text)."""
    try:
        while True:
            # Attempt to receive input based on the configuration
            if is_voice_input and config:
                await input_handler.receive(
                    wait_event=ctlr.input_wait,
                    silence_duration=config.silence_duration,
                    min_speech_duration=config.min_speech_duration,
                    silence_threshold=config.silence_threshold,
                )
            else:
                await input_handler.receive(
                    wait_event=ctlr.input_wait,
                )
    except* asyncio.CancelledError as cancel_exc:
        # Handle cancellation specifically
        for exc in cancel_exc.exceptions:
            logger.info(f"Input worker cancelled: {exc}")
        raise  # Re-raise to propagate cancellation
    except* Exception as e:
        logger.error(f"Input worker error: {e}\n{traceback.format_exc()}")
        # Allow the worker to restart on next iteration
        await asyncio.sleep(0.5)  # Brief pause before retrying


async def transcribe_worker(
    ctlr: PipelineController,
    transcriber: TranscriberLike,
    sample_rate: int,
):
    """Worker task for transcribing audio input to text."""
    try:
        while True:
            audio_array = await ctlr.audio_event.get()
            logger.info("* Transcribing...")

            result = transcriber.transcribe_buffer(audio_array, sample_rate)

            # Skip if result doesn't contain valid word
            if result is None or not any(char.isalnum() for char in result):
                logger.warning(f"{result} does not contain valid word.")

                # If the result is empty, we can choose to restart the pipeline
                ctlr.start_over()
                continue

            # Print the recognized text
            logger.info(f"Transcript: {result}")
            if result:
                await ctlr.input_event.set(result)

    except* asyncio.CancelledError as cancel_exc:
        # Handle cancellation specifically
        for exc in cancel_exc.exceptions:
            logger.info(f"Transcribe worker cancelled: {exc}")
        raise  # Re-raise to propagate cancellation
    except* Exception as e:
        logger.error(f"Transcribe worker error: {e}\n{traceback.format_exc()}")
        # Allow for restart on next iteration


async def chat_worker(
    ctlr: PipelineController,
    chat_agent: ChatAgentLike,
):
    """Worker task for processing chat interactions."""
    try:
        while True:
            message = await ctlr.input_event.get()
            logger.info("* Chatting...")
            response_received = False

            try:
                # Process chat responses
                async for response in chat_agent.chat(message):
                    await ctlr.speech_queue.put(response)
                    response_received = True

                # Mark batch complete only if we got at least one response
                if response_received:
                    logger.info("Chat response complete, marking batch as complete")
                    await ctlr.speech_queue.mark_batch_complete()
            except* asyncio.CancelledError:
                logger.info("Chat response generation cancelled")
                raise  # Re-raise to allow proper task cancellation
            except* Exception as eg:
                for e in eg.exceptions:
                    logger.error(f"Chat Error: {e}\n{traceback.format_exc()}")
                # Even if we had an error, if we got any responses at all,
                # we should mark the batch as complete
                if response_received:
                    await ctlr.speech_queue.mark_batch_complete()
    except* asyncio.CancelledError as e:
        logger.info(f"Chat worker cancelled: {e}")
        raise  # Re-raise to propagate cancellation
    except* Exception as e:
        logger.error(f"Chat worker error: {e}\n{traceback.format_exc()}")
        # Allow for restart on next iteration


async def speech_worker(ctlr: PipelineController, voice: Voice):
    """Worker task for processing speech output from the chat responses."""
    try:
        while True:
            # Get the first speech item
            speech = await ctlr.speech_queue.get()
            ctlr.speech_queue.task_done()

            # Process the speech if it's not None
            if speech is not None:
                await voice.say(speech)

            # Create a task to wait for batch completion
            batch_completion_task = asyncio.create_task(
                ctlr.speech_queue.wait_for_batch_completion()
            )

            try:
                # Process all items until batch completion
                while not batch_completion_task.done():
                    try:
                        # Try to get an item with a short timeout
                        speech = await asyncio.wait_for(
                            ctlr.speech_queue.get(), timeout=0.1
                        )
                        ctlr.speech_queue.task_done()

                        if speech is not None:
                            await voice.say(speech)
                    except asyncio.TimeoutError:
                        # No items available, just continue checking batch status
                        await asyncio.sleep(0.01)

                # Process any remaining items in the queue after batch is complete
                while True:
                    try:
                        speech = await asyncio.wait_for(
                            ctlr.speech_queue.get(), timeout=0.1
                        )
                        ctlr.speech_queue.task_done()

                        if speech is not None:
                            await voice.say(speech)
                    except asyncio.TimeoutError:
                        break

                logger.info("Batch complete, all speech processed")
                ctlr.start_over()  # Reset the pipeline for next conversation
            finally:
                # Always ensure we clean up the task
                if not batch_completion_task.done():
                    batch_completion_task.cancel()

    except* asyncio.CancelledError as cancel_exc:
        # Handle worker cancellation
        for exc in cancel_exc.exceptions:
            logger.info(f"Speech worker cancelled: {exc}")
        raise  # Re-raise to propagate cancellation
    except* Exception as e:
        logger.error(f"Speech worker error: {e}\n{traceback.format_exc()}")
        # Allow for restart on next iteration
