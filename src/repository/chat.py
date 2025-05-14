from typing import AsyncGenerator, Protocol


class ChatAgentLike(Protocol):
    def chat(self, msg: str) -> AsyncGenerator[str, None]: ...
