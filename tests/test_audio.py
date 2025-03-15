import numpy as np

from input.audio import _is_silence


class TestAudio:
    def test_empty_audio(self):
        """Test that empty audio is detected as silence."""
        empty_audio = np.array([], dtype=np.int16).tobytes()
        assert _is_silence(empty_audio, threshold=0.02) is True

    def test_zero_audio(self):
        """Test that audio with all zeros is detected as silence."""
        zero_audio = np.zeros(1000, dtype=np.int16).tobytes()
        assert _is_silence(zero_audio, threshold=0.02) is True
