from typing import Protocol


class InputLike(Protocol):
    async def receive(self): ...
