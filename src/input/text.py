import asyncio
import logging

import aioconsole

logging.basicConfig(level=logging.INFO)


class TextInput:
    def __init__(self, chat_queue):
        self.chat_queue = chat_queue

    async def receive(self):
        text = await aioconsole.ainput("> ")
        if not text.strip():
            return
        await self.chat_queue.put(text)
        await asyncio.sleep(0.1)
