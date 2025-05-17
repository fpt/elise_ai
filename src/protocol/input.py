import asyncio
from typing import Protocol


class InputLike(Protocol):
    async def receive(self, input_wait: asyncio.Event): ...
