import asyncio
import logging
from unittest.mock import patch

import pytest

from model.pipeline import PipelineController

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def pipeline_controller_with_text():
    """Fixture to provide a fresh PipelineController instance with text input for each test."""
    return PipelineController(text_input="Test input")


@pytest.fixture
def pipeline_controller_with_audio():
    """Fixture to provide a fresh PipelineController instance with audio input for each test."""
    return PipelineController(audio_data=b"Test audio data")


class TestPipelineController:
    """Test suite for the PipelineController class."""

    def test_initialization_with_text(self, pipeline_controller_with_text):
        """Test that the PipelineController initializes correctly with text input."""
        pc = pipeline_controller_with_text

        # Check that all events are created
        assert isinstance(pc.cancel_requested, asyncio.Event)
        assert isinstance(pc.completed, asyncio.Event)

        # Check that EventData instances are created
        assert pc.audio_event is not None
        assert pc.input_event is not None
        assert pc.speech_queue is not None

        # Check initial state
        assert not pc.cancel_requested.is_set()  # Initially not set
        assert not pc.completed.is_set()  # Initially not set

    def test_initialization_with_audio(self, pipeline_controller_with_audio):
        """Test that the PipelineController initializes correctly with audio input."""
        pc = pipeline_controller_with_audio

        # Check that all events are created
        assert isinstance(pc.cancel_requested, asyncio.Event)
        assert isinstance(pc.completed, asyncio.Event)

        # Check that EventData instances are created
        assert pc.audio_event is not None
        assert pc.input_event is not None
        assert pc.speech_queue is not None

        # Check initial state
        assert not pc.cancel_requested.is_set()  # Initially not set
        assert not pc.completed.is_set()  # Initially not set

    def test_initialization_errors(self):
        """Test that the PipelineController raises appropriate errors for invalid initialization."""
        # Test that it raises an error when neither audio_data nor text_input is provided
        with pytest.raises(
            ValueError, match="Either audio_data or text_input must be provided"
        ):
            PipelineController()

        # Test that it raises an error when both audio_data and text_input are provided
        with pytest.raises(
            ValueError, match="Only one of audio_data or text_input can be provided"
        ):
            PipelineController(audio_data=b"Test audio", text_input="Test text")

    def test_complete(self, pipeline_controller_with_text):
        """Test that complete sets the completed event."""
        pc = pipeline_controller_with_text

        # Initially not set
        assert not pc.is_completed()

        # Call complete
        pc.complete()

        # Check that completed is set
        assert pc.is_completed()

    def test_is_cancellation_requested(self, pipeline_controller_with_text):
        """Test the is_cancellation_requested method."""
        pc = pipeline_controller_with_text

        # Initially not set
        assert not pc.is_cancellation_requested()

        # After setting
        pc.cancel_requested.set()
        assert pc.is_cancellation_requested()

        # After clearing
        pc.cancel_requested.clear()
        assert not pc.is_cancellation_requested()

    def test_is_completed(self, pipeline_controller_with_text):
        """Test the is_completed method."""
        pc = pipeline_controller_with_text

        # Initially not set
        assert not pc.is_completed()

        # After setting
        pc.completed.set()
        assert pc.is_completed()

        # After clearing
        pc.completed.clear()
        assert not pc.is_completed()

    @pytest.mark.asyncio
    async def test_request_cancellation(self, pipeline_controller_with_text):
        """Test that request_cancellation sets the proper events."""
        pc = pipeline_controller_with_text

        # Setup mocks to check that set/put is called
        with (
            patch.object(pc.audio_event, "set") as mock_audio_set,
            patch.object(pc.input_event, "set") as mock_input_set,
            patch.object(pc.speech_queue, "put") as mock_speech_put,
        ):
            pc.request_cancellation()

            # Check that cancel_requested is set
            assert pc.cancel_requested.is_set()

            # Allow tasks to complete
            await asyncio.sleep(0.1)

            # Verify that event methods were called
            mock_audio_set.assert_called_once_with(None)
            mock_input_set.assert_called_once_with(None)
            mock_speech_put.assert_called_once_with(None)

    @pytest.mark.asyncio
    async def test_cleanup(self, pipeline_controller_with_text):
        """Test the cleanup method."""
        pc = pipeline_controller_with_text

        # Set up mocks to verify resets
        with (
            patch.object(pc.speech_queue, "reset") as mock_speech_reset,
            patch.object(pc.input_event, "reset") as mock_input_reset,
            patch.object(pc.audio_event, "reset") as mock_audio_reset,
        ):
            # Execute cleanup
            await pc.cleanup()

            # Verify resets were called
            mock_speech_reset.assert_called_once()
            mock_input_reset.assert_called_once()
            mock_audio_reset.assert_called_once()

            # Check completed is set
            assert pc.completed.is_set()

    @pytest.mark.asyncio
    async def test_cleanup_with_exception(self, pipeline_controller_with_text):
        """Test that cleanup handles exceptions properly."""
        pc = pipeline_controller_with_text

        # Set up mocks to raise exceptions
        with (
            patch.object(
                pc.speech_queue, "reset", side_effect=Exception("Test exception")
            ),
            patch("model.pipeline.logger.error") as mock_logger,
        ):
            # Execute cleanup
            await pc.cleanup()

            # Verify logger was called to log the error
            assert mock_logger.called

    @pytest.mark.asyncio
    async def test_event_flow(self, pipeline_controller_with_text):
        """Test the flow of events through the pipeline."""
        pc = pipeline_controller_with_text

        # Test data
        test_audio_data = b"audio data"
        test_input_text = "Hello, world!"
        test_speech_text = "Response text"

        # Set data
        asyncio.create_task(pc.audio_event.set(test_audio_data))
        asyncio.create_task(pc.input_event.set(test_input_text))
        asyncio.create_task(pc.speech_queue.put(test_speech_text))

        # Get data
        audio_data = await pc.audio_event.get()
        input_text = await pc.input_event.get()
        speech_text = await pc.speech_queue.get()

        # Verify data
        assert audio_data == test_audio_data
        assert input_text == test_input_text
        assert speech_text == test_speech_text

    @pytest.mark.asyncio
    async def test_wait_for_completion(self, pipeline_controller_with_text):
        """Test the wait_for_completion method."""
        pc = pipeline_controller_with_text

        # Create a task to wait for completion
        wait_task = asyncio.create_task(pc.wait_for_completion())

        # Ensure the task is not done yet
        await asyncio.sleep(0.1)
        assert not wait_task.done()

        # Complete the pipeline
        pc.complete()

        # Allow the task to process
        await asyncio.sleep(0.1)

        # Verify the task is now done
        assert wait_task.done()

    @pytest.mark.asyncio
    async def test_context_manager_with_text(self):
        """Test that the PipelineController works as a context manager with text input."""
        # Setup mock for cleanup method
        with patch.object(PipelineController, "cleanup") as mock_cleanup:
            # Use the controller as a context manager
            async with PipelineController(text_input="Test input") as pc:
                # Check that the controller is initialized
                assert pc is not None
                assert isinstance(pc, PipelineController)

                # Check that cleanup hasn't been called yet
                mock_cleanup.assert_not_called()

            # After exiting the context, cleanup should be called
            mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_with_audio(self):
        """Test that the PipelineController works as a context manager with audio input."""
        # Setup mock for cleanup method
        with patch.object(PipelineController, "cleanup") as mock_cleanup:
            # Use the controller as a context manager
            async with PipelineController(audio_data=b"Test audio data") as pc:
                # Check that the controller is initialized
                assert pc is not None
                assert isinstance(pc, PipelineController)

                # Check that cleanup hasn't been called yet
                mock_cleanup.assert_not_called()

            # After exiting the context, cleanup should be called
            mock_cleanup.assert_called_once()
