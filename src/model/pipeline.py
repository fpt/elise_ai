import asyncio
import logging
import traceback
from asyncio import Event

from model.event import EventData
from model.queue_event import QueuedEventData

logger = logging.getLogger(__name__)


class PipelineController:
    """
    Controls the flow of data through the audio processing and chat pipeline.

    This class manages several asyncio events to coordinate the flow of data
    between audio input, transcription, chat processing, and speech output.

    Attributes:
        cancel_requested (Event): Signal to request cancellation of pipeline processing.
        input_wait (Event): Controls when new input is allowed to be processed.
        audio_event (EventData): Carries audio data from input to transcription.
        input_event (EventData): Carries text data from transcription to chat.
        speech_event (QueuedEventData): Queues text responses from chat to speech output,
                                      allowing for multiple responses to be processed in sequence.
    """

    def __init__(self):
        self.cancel_requested = Event()
        self.input_wait = Event()
        self.audio_event = EventData()
        self.input_event = EventData()
        self.speech_queue = (
            QueuedEventData()
        )  # Using QueuedEventData for sequential processing

        self.input_wait.set()  # Initially set to allow input

    def start_over(self):
        """
        Reset the pipeline to its initial state to start a new conversation.

        This clears all events and resets the data in all EventData objects,
        allowing the pipeline to start from the beginning with new input.
        """
        logger.debug("Resetting pipeline for new conversation")

        # Clear the cancellation flag
        self.cancel_requested.clear()

        # Reset all event data
        self.audio_event.reset()
        self.input_event.reset()
        self.speech_queue.reset()  # For QueuedEventData, this clears the queue

        # Set the input wait event to allow for new input
        self.input_wait.set()

    def request_cancellation(self):
        """パイプラインのキャンセルを要求する"""
        self.cancel_requested.set()
        # 待機中のタスクを解放するためにイベントをセット
        # Set events to allow waiting tasks to exit
        asyncio.create_task(self.audio_event.set(None))
        asyncio.create_task(self.input_event.set(None))
        asyncio.create_task(
            self.speech_queue.put(None)
        )  # Using put for QueuedEventData
        self.input_wait.set()

    async def cleanup(self):
        """
        Cancel all tasks and wait for them to complete.

        This method should be called during shutdown to ensure
        proper cleanup of all resources and tasks.
        """
        logger.info("Cleaning up tasks...")

        # Give tasks time to respond to cancellation
        await asyncio.sleep(0.1)

        try:
            # Ensure all data is marked as processed
            # Reset events to prevent hanging
            self.speech_queue.reset()  # For QueuedEventData, this clears the queue
            self.input_event.reset()
            self.audio_event.reset()

            # Make sure input_wait is set to allow any blocked tasks to continue
            self.input_wait.set()

            logger.info("All tasks cleaned up successfully")
        except* Exception as e:
            # Handle all cleanup exceptions using except*
            for exc in e.exceptions:
                logger.error(f"Error during cleanup: {exc}\n{traceback.format_exc()}")
            # Continue with shutdown even if cleanup fails

    def is_cancellation_requested(self) -> bool:
        """
        Check if cancellation of the pipeline has been requested.

        Returns:
            bool: True if cancellation has been requested, False otherwise.
        """
        return self.cancel_requested.is_set()
