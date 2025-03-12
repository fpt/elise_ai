from kokoro import KPipeline
from typing import Protocol

CHUNK_SIZE = 2**10

# 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
# 🇯🇵 'j' => Japanese: pip install misaki[ja]
# 🇨🇳 'z' => Mandarin Chinese: pip install misaki[zh]
language_voice_map = {"en": "a", "ja": "j"}

voice_name_map = {
    "a": "af_heart",
    "j": "jf_alpha",
}


class KokoroVoice(Protocol):
    def __init__(self, pa, audio_lock, lang="en"):
        lang_code = language_voice_map[lang]
        self.pipeline = KPipeline(lang_code=lang_code, repo_id='hexgrad/Kokoro-82M')
        self.voice = voice_name_map[lang_code]
        self.audio_lock = audio_lock
        self.pa = pa
        print(f"LangCode: {lang_code}, Voice: {self.voice}")

    async def say(self, text):
        await self.audio_lock.acquire()

        generator = self.pipeline(
            text,
            voice=self.voice,  # <= change voice here
            speed=1.1,
            split_pattern=r"\n+",
        )

        try:
            stream = self.pa.open(
                format=self.pa.get_format_from_width(4, unsigned=False),
                channels=1,
                rate=24000,
                output=True,
            )

            for i, (gs, ps, audio) in enumerate(generator):
                print(i)  # i => index
                print(gs)  # gs => graphemes/text
                print(ps)  # ps => phonemes
                stream.write(audio.numpy().tobytes())
        except Exception as e:
            print(f"Error: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            self.audio_lock.release()
