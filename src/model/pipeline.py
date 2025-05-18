import asyncio
import logging
import traceback
from asyncio import Event
from types import TracebackType
from typing import Any, Optional, Self, Type

from model.event import EventData
from model.queue_event import QueuedEventData

logger = logging.getLogger(__name__)


class PipelineController:
    """
    Controls the flow of data through the audio processing and chat pipeline.

    This class manages several asyncio events to coordinate the flow of data
    between audio input, transcription, chat processing, and speech output.

    The controller is designed to be used for a single pipeline run (from input to output)
    and should be recreated for each new conversation.

    This class can be used as a context manager:
    ```
    # Get input first (audio_data or text_input)

    # Then create controller with the input data
    async with PipelineController(audio_data=audio_data) as ctlr:  # For audio input
        # Use the controller
    # or
    async with PipelineController(text_input=text_input) as ctlr:  # For text input
        # Use the controller
    # Cleanup is automatically handled when exiting the context
    ```

    Attributes:
        cancel_requested (Event): Signal to request cancellation of pipeline processing.
        audio_event (EventData): Carries audio data from input to transcription.
        input_event (EventData): Carries text data from transcription to chat.
        speech_queue (QueuedEventData): Queues text responses from chat to speech output,
                                      allowing for multiple responses to be processed in sequence.
        completed (Event): Signal that the pipeline has completed processing.
    """

    def __init__(self, *, audio_data: Any = None, text_input: str | None = None):
        """
        Initialize a new PipelineController with the specified input data.

        Args:
            audio_data: Audio data for voice input (mutually exclusive with text_input)
            text_input: Text input for text input (mutually exclusive with audio_data)
        """
        if audio_data is None and text_input is None:
            raise ValueError("Either audio_data or text_input must be provided")
        if audio_data is not None and text_input is not None:
            raise ValueError("Only one of audio_data or text_input can be provided")

        self.cancel_requested = Event()
        self.audio_event = EventData()
        self.input_event = EventData()
        self.speech_queue = (
            QueuedEventData()
        )  # Using QueuedEventData for sequential processing
        self.completed = Event()  # New event to signal pipeline completion

        # Store the initial input data for later use
        self._initial_audio_data = audio_data
        self._initial_text_input = text_input

    async def __aenter__(self) -> Self:
        """Enter the async context manager."""
        # Set the initial input data now that we have an event loop
        if self._initial_audio_data is not None:
            await self.audio_event.set(self._initial_audio_data)
        elif self._initial_text_input is not None:
            await self.input_event.set(self._initial_text_input)
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit the async context manager and clean up resources."""
        await self.cleanup()

    def complete(self):
        """
        Signal that the pipeline has completed processing.

        This should be called when all processing for the current conversation
        is finished and the pipeline can be disposed of.
        """
        logger.debug("Pipeline completed")
        self.completed.set()

    def request_cancellation(self):
        """Request cancellation of the pipeline processing."""
        self.cancel_requested.set()
        # 待機中のタスクを解放するためにイベントをセット
        # Set events to allow waiting tasks to exit
        asyncio.create_task(self.audio_event.set(None))
        asyncio.create_task(self.input_event.set(None))
        asyncio.create_task(
            self.speech_queue.put(None)
        )  # Using put for QueuedEventData

    async def cleanup(self):
        """
        Cancel all tasks and wait for them to complete.

        This method should be called during shutdown to ensure
        proper cleanup of all resources and tasks.
        """

        try:
            # Reset events to prevent hanging
            self.speech_queue.reset()
            self.input_event.reset()
            self.audio_event.reset()

            # Signal completion if not already done
            if not self.completed.is_set():
                self.completed.set()
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

    def is_completed(self) -> bool:
        """
        Check if the pipeline has completed processing.

        Returns:
            bool: True if the pipeline has completed, False otherwise.
        """
        return self.completed.is_set()

    async def wait_for_completion(self):
        """
        Wait until the pipeline has completed processing.
        """
        await self.completed.wait()
