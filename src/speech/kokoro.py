import logging
from typing import Protocol

from kokoro import KPipeline

SPEECH_FORMAT_WIDTH = 4
SPEECH_CHANNELS = 1
SPEECH_RATE = 24000

# 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
# 🇯🇵 'j' => Japanese: pip install misaki[ja]
# 🇨🇳 'z' => Mandarin Chinese: pip install misaki[zh]
language_voice_map = {"en": "a", "ja": "j"}

voice_name_map = {
    "a": "af_heart",
    "j": "jf_alpha",
}

logger = logging.getLogger(__name__)


class Voice(Protocol):
    def say(self, text): ...


class KokoroVoice:
    def __init__(self, pa, audio_lock, lang="en", debug=False):
        self.debug = debug
        lang_code = language_voice_map[lang]
        self.pipeline = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M")
        self.voice = voice_name_map[lang_code]
        self.audio_lock = audio_lock
        self.pa = pa
        logger.info(f"LangCode: {lang_code}, Voice: {self.voice}")

    async def say(self, text):
        async with self.audio_lock:
            generator = self.pipeline(
                text,
                voice=self.voice,
                speed=1.1,
                split_pattern=r"\n+",
            )

            try:
                stream = self.pa.open(
                    format=self.pa.get_format_from_width(
                        SPEECH_FORMAT_WIDTH, unsigned=False
                    ),
                    channels=SPEECH_CHANNELS,
                    rate=SPEECH_RATE,
                    output=True,
                )

                for i, (gs, ps, audio) in enumerate(generator):
                    logger.info(f"{i}: {gs}")  # i => index, gs => graphemes/text
                    if self.debug:
                        logger.debug(ps)  # ps => phonemes
                    stream.write(audio.numpy().tobytes())
            except Exception as e:
                logger.error(f"Error while playing audio: {e}")
            finally:
                stream.stop_stream()
                stream.close()
