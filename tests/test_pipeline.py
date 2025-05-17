import asyncio
import logging
from unittest.mock import patch

import pytest

from model.pipeline import PipelineController

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def pipeline_controller():
    """Fixture to provide a fresh PipelineController instance for each test."""
    return PipelineController()


class TestPipelineController:
    """Test suite for the PipelineController class."""

    def test_initialization(self, pipeline_controller):
        """Test that the PipelineController initializes with the correct state."""
        pc = pipeline_controller

        # Check that all events are created
        assert isinstance(pc.cancel_requested, asyncio.Event)
        assert isinstance(pc.input_wait, asyncio.Event)

        # Check that EventData instances are created
        assert pc.audio_event is not None
        assert pc.input_event is not None
        assert pc.speech_queue is not None

        # Check initial state
        assert pc.input_wait.is_set()  # Initially set to allow input
        assert not pc.cancel_requested.is_set()  # Initially not set

    def test_start_over(self, pipeline_controller):
        """Test that start_over resets the pipeline state."""
        pc = pipeline_controller

        # Set some events
        pc.cancel_requested.set()
        pc.input_wait.clear()

        # Reset
        pc.start_over()

        # Check reset state
        assert not pc.cancel_requested.is_set()
        assert pc.input_wait.is_set()

    def test_is_cancellation_requested(self, pipeline_controller):
        """Test the is_cancellation_requested method."""
        pc = pipeline_controller

        # Initially not set
        assert not pc.is_cancellation_requested()

        # After setting
        pc.cancel_requested.set()
        assert pc.is_cancellation_requested()

        # After clearing
        pc.cancel_requested.clear()
        assert not pc.is_cancellation_requested()

    @pytest.mark.asyncio
    async def test_request_cancellation(self, pipeline_controller):
        """Test that request_cancellation sets the proper events."""
        pc = pipeline_controller

        # Setup mocks to check that set/put is called
        with (
            patch.object(pc.audio_event, "set") as mock_audio_set,
            patch.object(pc.input_event, "set") as mock_input_set,
            patch.object(pc.speech_queue, "put") as mock_speech_put,
        ):
            pc.request_cancellation()

            # Check that cancel_requested is set
            assert pc.cancel_requested.is_set()

            # Check that input_wait is set
            assert pc.input_wait.is_set()

            # Allow tasks to complete
            await asyncio.sleep(0.1)

            # Verify that event methods were called
            mock_audio_set.assert_called_once_with(None)
            mock_input_set.assert_called_once_with(None)
            mock_speech_put.assert_called_once_with(None)

    @pytest.mark.asyncio
    async def test_cleanup(self, pipeline_controller):
        """Test the cleanup method."""
        pc = pipeline_controller

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

            # Check input_wait is set
            assert pc.input_wait.is_set()

    @pytest.mark.asyncio
    async def test_cleanup_with_exception(self, pipeline_controller):
        """Test that cleanup handles exceptions properly."""
        pc = pipeline_controller

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
    async def test_event_flow(self, pipeline_controller):
        """Test the flow of events through the pipeline."""
        pc = pipeline_controller

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
