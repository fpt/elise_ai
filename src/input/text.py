import logging

import aioconsole

logging.basicConfig(level=logging.INFO)


class TextInput:
    def __init__(self, chat_data):
        self.chat_data = chat_data

    async def receive(
        self,
    ):
        text = await aioconsole.ainput("> ")
        if not text.strip():
            return

        await self.chat_data.set(text)
