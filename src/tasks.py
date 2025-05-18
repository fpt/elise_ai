import asyncio
import logging
import traceback

from input.transcribe import TranscriberLike
from model.pipeline import PipelineController
from protocol.chat import ChatAgentLike
from speech.kokoro import Voice

logger = logging.getLogger(__name__)


async def input_worker(
    input_handler, config=None, is_voice_input=True, wait_event=None
):
    """
    Worker task for receiving input (either voice or text).

    This function simply receives input once and returns.

    Args:
        input_handler: The input handler to use (AudioInput or TextInput)
        config: Configuration for voice input (optional)
        is_voice_input: Whether this is voice input (True) or text input (False)
        wait_event: Event to wait for before receiving input (optional)
    """
    try:
        # Create a wait event if none is provided
        if wait_event is None:
            wait_event = asyncio.Event()
            wait_event.set()  # Set the event to allow input immediately

        # Attempt to receive input based on the configuration
        if is_voice_input and config:
            await input_handler.receive(
                wait_event=wait_event,
                silence_duration=config.silence_duration,
                min_speech_duration=config.min_speech_duration,
                silence_threshold=config.silence_threshold,
            )
        else:
            await input_handler.receive(
                wait_event=wait_event,
            )

    except* asyncio.CancelledError as cancel_exc:
        # Handle cancellation specifically
        for exc in cancel_exc.exceptions:
            logger.info(f"Input worker cancelled: {exc}")
        raise  # Re-raise to propagate cancellation
    except* Exception as e:
        logger.error(f"Input worker error: {e}\n{traceback.format_exc()}")


async def transcribe_worker(
    ctlr: PipelineController,
    transcriber: TranscriberLike,
    sample_rate: int,
):
    """Worker task for transcribing audio input to text for a single pipeline run."""
    try:
        audio_array = await ctlr.get_audio_data()
        logger.info("* Transcribing...")

        result = transcriber.transcribe_buffer(audio_array, sample_rate)

        # Skip if result doesn't contain valid word
        if result is None or not any(char.isalnum() for char in result):
            logger.warning(f"{result} does not contain valid word.")
            # Signal completion since we're not continuing with this input
            ctlr.complete()
            return

        # Print the recognized text
        logger.info(f"Transcript: {result}")
        if result:
            await ctlr.set_input_text(result)

        # Wait for pipeline completion or cancellation
        while not ctlr.is_completed() and not ctlr.is_cancellation_requested():
            await asyncio.sleep(0.01)  # Reduced sleep time for faster response

    except* asyncio.CancelledError as cancel_exc:
        # Handle cancellation specifically
        for exc in cancel_exc.exceptions:
            logger.info(f"Transcribe worker cancelled: {exc}")
        raise  # Re-raise to propagate cancellation
    except* Exception as e:
        logger.error(f"Transcribe worker error: {e}\n{traceback.format_exc()}")


async def chat_worker(
    ctlr: PipelineController,
    chat_agent: ChatAgentLike,
):
    """Worker task for processing chat interactions for a single pipeline run."""
    try:
        message = await ctlr.get_input_text()
        logger.info("* Chatting...")
        response_received = False

        try:
            # Process chat responses
            async for response in chat_agent.chat(message):
                await ctlr.add_to_speech_queue(response)
                response_received = True

            # Mark batch complete only if we got at least one response
            if response_received:
                logger.info("Chat response complete, marking batch as complete")
                await ctlr.mark_speech_batch_complete()
        except* asyncio.CancelledError:
            logger.info("Chat response generation cancelled")
            raise  # Re-raise to allow proper task cancellation
        except* Exception as eg:
            for e in eg.exceptions:
                logger.error(f"Chat Error: {e}\n{traceback.format_exc()}")
            # Even if we had an error, if we got any responses at all,
            # we should mark the batch as complete
            if response_received:
                await ctlr.mark_speech_batch_complete()

        # Wait for pipeline completion or cancellation
        while not ctlr.is_completed() and not ctlr.is_cancellation_requested():
            await asyncio.sleep(0.01)  # Reduced sleep time for faster response
    except* asyncio.CancelledError as e:
        logger.info(f"Chat worker cancelled: {e}")
        raise  # Re-raise to propagate cancellation
    except* Exception as e:
        logger.error(f"Chat worker error: {e}\n{traceback.format_exc()}")


async def speech_worker(ctlr: PipelineController, voice: Voice):
    """Worker task for processing speech output from the chat responses for a single pipeline run."""
    try:
        # Get the first speech item
        speech = await ctlr.get_from_speech_queue()
        ctlr.speech_queue_task_done()

        # Process the speech if it's not None
        if speech is not None:
            await voice.say(speech)

        # Create a task to wait for batch completion
        batch_completion_task = asyncio.create_task(
            ctlr.wait_for_speech_batch_completion()
        )

        try:
            # Process all items until batch completion
            while not batch_completion_task.done():
                try:
                    # Try to get an item with a shorter timeout for faster response
                    speech = await asyncio.wait_for(
                        ctlr.get_from_speech_queue(), timeout=0.05
                    )
                    ctlr.speech_queue_task_done()

                    if speech is not None:
                        await voice.say(speech)
                except asyncio.TimeoutError:
                    # No items available, just continue checking batch status
                    await asyncio.sleep(0.01)

            # Process any remaining items in the queue after batch is complete
            while True:
                try:
                    speech = await asyncio.wait_for(
                        ctlr.get_from_speech_queue(), timeout=0.05
                    )
                    ctlr.speech_queue_task_done()

                    if speech is not None:
                        await voice.say(speech)
                except asyncio.TimeoutError:
                    break

            logger.info("Batch complete, all speech processed")
            # Signal that the pipeline has completed
            ctlr.complete()
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
