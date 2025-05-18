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

        # Check initial state using accessor methods
        assert not pc.is_cancellation_requested()  # Initially not set
        assert not pc.is_completed()  # Initially not set

    def test_initialization_with_audio(self, pipeline_controller_with_audio):
        """Test that the PipelineController initializes correctly with audio input."""
        pc = pipeline_controller_with_audio

        # Check initial state using accessor methods
        assert not pc.is_cancellation_requested()  # Initially not set
        assert not pc.is_completed()  # Initially not set

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

    @pytest.mark.asyncio
    async def test_is_cancellation_requested(self, pipeline_controller_with_text):
        """Test the is_cancellation_requested method."""
        pc = pipeline_controller_with_text

        # Initially not set
        assert not pc.is_cancellation_requested()

        # Direct access to the internal attribute for test purposes
        pc._cancel_requested.set()
        assert pc.is_cancellation_requested()

    def test_is_completed(self, pipeline_controller_with_text):
        """Test the is_completed method."""
        pc = pipeline_controller_with_text

        # Initially not set
        assert not pc.is_completed()

        # After setting
        pc.complete()
        assert pc.is_completed()

    @pytest.mark.asyncio
    async def test_request_cancellation(self, pipeline_controller_with_text):
        """Test that request_cancellation sets the proper events."""
        pc = pipeline_controller_with_text

        # Use the mock library to patch the private methods
        with (
            patch("model.pipeline.asyncio.create_task") as mock_create_task,
        ):
            pc.request_cancellation()

            # Check that cancel_requested is set
            assert pc.is_cancellation_requested()

            # Verify that create_task was called 3 times (for each event)
            assert mock_create_task.call_count == 3

    @pytest.mark.asyncio
    async def test_cleanup(self, pipeline_controller_with_text):
        """Test the cleanup method."""
        pc = pipeline_controller_with_text

        # Mock the private attributes with patch to verify internal behavior
        with (
            patch.object(pc, "_speech_queue") as mock_speech_queue,
            patch.object(pc, "_input_event") as mock_input_event,
            patch.object(pc, "_audio_event") as mock_audio_event,
            patch.object(pc, "_completed") as mock_completed,
        ):
            mock_completed.is_set.return_value = False

            # Execute cleanup
            await pc.cleanup()

            # Verify resets were called
            mock_speech_queue.reset.assert_called_once()
            mock_input_event.reset.assert_called_once()
            mock_audio_event.reset.assert_called_once()

            # Check completed is set
            mock_completed.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_with_exception(self, pipeline_controller_with_text):
        """Test that cleanup handles exceptions properly."""
        pc = pipeline_controller_with_text

        # Set up mocks to raise exceptions
        with (
            patch.object(
                pc._speech_queue, "reset", side_effect=Exception("Test exception")
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

        # Set data using accessor methods
        asyncio.create_task(pc.set_audio_data(test_audio_data))
        asyncio.create_task(pc.set_input_text(test_input_text))
        asyncio.create_task(pc.add_to_speech_queue(test_speech_text))

        # Get data using accessor methods
        audio_data = await pc.get_audio_data()
        input_text = await pc.get_input_text()
        speech_text = await pc.get_from_speech_queue()

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
