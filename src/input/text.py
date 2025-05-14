import asyncio
import logging

import aioconsole

logging.basicConfig(level=logging.INFO)


class TextInput:
    def __init__(self, chat_data):
        self.chat_data = chat_data
        self.response_received_event = asyncio.Event()
        self.response_received_event.set()  # Initially set so first prompt shows

    async def receive(self):
        # Wait until we've received a response before showing the next prompt
        await self.response_received_event.wait()
        text = await aioconsole.ainput("> ")
        if not text.strip():
            return

        # Clear the event when we send a message
        self.response_received_event.clear()
        await self.chat_data.set(text)
        await asyncio.sleep(0.1)

    def notify_response_complete(self):
        """Call this when a response is complete to allow showing the next prompt"""
        self.response_received_event.set()
