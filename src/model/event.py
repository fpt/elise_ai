import asyncio
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EventData:
    """Class to hold data between coroutines with Event for signaling."""

    data: Any = None
    event_set: asyncio.Event = None  # type: ignore
    event_processed: asyncio.Event = None  # type: ignore

    def __post_init__(self):
        if self.event_set is None:
            self.event_set = asyncio.Event()
        if self.event_processed is None:
            self.event_processed = asyncio.Event()
            # Initially the data has been "processed" (there is no data)
            self.event_processed.set()

    async def set(self, data):
        """Set the data and signal that it's ready."""
        # Wait until previous data has been processed
        await self.event_processed.wait()
        # Set the new data
        self.data = data
        # Clear the processed event
        self.event_processed.clear()
        # Signal that data is ready
        self.event_set.set()

    async def get(self):
        """Wait for data to be ready and return it."""
        # Wait for the data to be set
        await self.event_set.wait()
        # Clear the set event
        self.event_set.clear()
        # Return the data
        return self.data

    def get_nowait(self):
        """Get the data without waiting. Returns None if no data is ready."""
        if not self.event_set.is_set():
            return None
        self.event_set.clear()
        return self.data

    def task_done(self):
        """Signal that the data has been processed."""
        self.event_processed.set()
