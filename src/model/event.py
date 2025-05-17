import asyncio
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EventData:
    """Class to hold data between coroutines with Event for signaling."""

    def __init__(self):
        self.data: Any = None
        self._event_set = asyncio.Event()

    async def set(self, data: Any):
        """Set the data and signal that it's ready."""
        # Set the new data
        self.data = data
        # Signal that data is ready
        assert self._event_set is not None
        self._event_set.set()

    async def get(self) -> Any:
        """Wait for data to be ready and return it."""
        # Wait for the data to be set
        assert self._event_set is not None
        await self._event_set.wait()
        # Clear the set event
        self._event_set.clear()
        # Return the data
        return self.data

    def reset(self):
        """Reset the event and data."""
        self.data = None
        assert self._event_set is not None
        self._event_set.clear()
