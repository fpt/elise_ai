import asyncio
import logging

import aioconsole

logging.basicConfig(level=logging.INFO)


class TextInput:
    def __init__(self, chat_data):
        self.chat_data = chat_data

    async def receive(
        self,
        wait_event: asyncio.Event,
    ):
        # Wait until we've received a response before showing the next prompt
        await wait_event.wait()
        text = await aioconsole.ainput("> ")
        if not text.strip():
            return

        # Clear the event when we send a message
        wait_event.clear()
        await self.chat_data.set(text)
        await asyncio.sleep(0.1)
