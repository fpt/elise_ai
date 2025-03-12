import numpy as np
from audio import is_silence


class TestAudio:
    def test_empty_audio(self):
        """Test that empty audio is detected as silence."""
        empty_audio = np.array([], dtype=np.int16).tobytes()
        assert is_silence(empty_audio, threshold=100) is True

    def test_zero_audio(self):
        """Test that audio with all zeros is detected as silence."""
        zero_audio = np.zeros(1000, dtype=np.int16).tobytes()
        assert is_silence(zero_audio, threshold=100) is True
