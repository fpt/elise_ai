import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class QueuedEventData:
    """
    Class to manage a queue of data between coroutines.

    This is useful when multiple values need to be sent in sequence
    from one task to another, such as streamed responses from an AI model.
    """

    def __init__(self, maxsize: int = 0):
        """
        Initialize with an optional maximum queue size.

        Args:
            maxsize: The maximum size of the queue. 0 means unlimited.
        """
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=maxsize)
        self._closed: bool = False
        self._batch_complete: asyncio.Event = asyncio.Event()

    async def put(self, data: Any):
        """
        Add data to the queue for processing.

        Args:
            data: The data to add to the queue.
        """
        if self._closed:
            logger.warning("Attempt to put data into a closed queue")
            return

        await self._queue.put(data)

    async def mark_batch_complete(self):
        """
        Signal that a batch of related items is complete.
        This allows consumers to know when a logical group of items is fully processed.
        """
        self._batch_complete.set()

    async def wait_for_batch_completion(self):
        """
        Wait until the current batch is marked as complete.
        """
        await self._batch_complete.wait()
        self._batch_complete.clear()  # Reset for the next batch

    async def get(self) -> Any:
        """
        Wait for and retrieve the next item from the queue.

        Returns:
            The next item from the queue.
        """
        if self._closed and self._queue.empty():
            logger.warning("Attempt to get data from an empty closed queue")
            return None

        return await self._queue.get()

    def get_nowait(self) -> Optional[Any]:
        """
        Get an item from the queue without waiting.

        Returns:
            The next item from the queue or None if queue is empty.
        """
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def task_done(self):
        """Mark a queue item as done, required for join()."""
        self._queue.task_done()

    def close(self):
        """
        Mark the queue as closed. No more items should be added.
        Existing items can still be retrieved.
        """
        self._closed = True

    def reset(self):
        """
        Reset the queue, removing all pending items.
        This reopens the queue for new items if it was closed.
        """
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break

        self._closed = False
        self._batch_complete.clear()  # Make sure the batch completion event is cleared

    def is_batch_complete(self) -> bool:
        """
        Check if the current batch has been marked as complete.

        Returns:
            True if the batch is complete, False otherwise.
        """
        return self._batch_complete.is_set()
