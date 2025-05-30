import asyncio
import logging

logger = logging.getLogger(__name__)


class TextVoice:
    async def say(self, text):
        pending_texts = [text]
        while pending_texts:
            text = pending_texts.pop(0)
            try:
                print(f"TextVoice: {text}", flush=True)
                await asyncio.sleep(0.1)
            except BlockingIOError:
                # If the output is blocked, we can just wait and retry
                pending_texts.append(text)
                logger.warning(
                    "TextVoice output is blocked, retrying after a short delay."
                )
            except Exception as e:
                print(f"Error in TextVoice: {e}", flush=True)
                break
