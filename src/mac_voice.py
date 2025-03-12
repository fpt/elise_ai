from AppKit import AVSpeechSynthesizer, AVSpeechSynthesisVoice, AVSpeechUtterance
from typing import Protocol
import asyncio

language_voice_map = {"en": "en-US", "ja": "ja-JP"}

voice_name_map = {
    "en-US": ["Alex", "Ava", "Samantha", "Evan", "Joelle"],
    "ja-JP": ["Kyoko", "Hattori"],
}


class MacVoice(Protocol):
    def __init__(self, audio_lock, lang="en"):
        if lang == "ja":
            self.voice_lang = "ja-JP"
        elif lang == "en":
            self.voice_lang = "en-US"
        else:
            raise ValueError(f"Unsupported language: {lang}")
        self.voice = _choose_voice(self.voice_lang)
        self.audio_lock = audio_lock
        self.synth = AVSpeechSynthesizer.alloc().init()

    async def say(self, text):
        await self.audio_lock.acquire()

        utterance = AVSpeechUtterance.speechUtteranceWithString_(text)
        utterance.setRate_(0.50)
        utterance.setPitchMultiplier_(0.9)
        utterance.setVolume_(1.0)

        utterance.setVoice_(self.voice)

        self.synth = AVSpeechSynthesizer.alloc().init()

        try:
            self.synth.speakUtterance_(utterance)

            # Wait for the speech to finish
            while self.synth.isSpeaking():
                await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.audio_lock.release()


def _choose_voice(voice_lang):
    available_voices = filter(
        lambda x: x.language() == voice_lang, AVSpeechSynthesisVoice.speechVoices()
    )
    voice_names = voice_name_map[voice_lang]
    for voice in available_voices:
        if voice.name() in voice_names:
            return voice
    return None
