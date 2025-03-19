import asyncio


class TextVoice:
    async def say(self, text):
        print(f"TextVoice: {text}")
        await asyncio.sleep(0.1)
