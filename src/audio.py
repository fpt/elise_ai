import numpy as np


def is_silence(audio_data: bytes, threshold: float) -> bool:
    """
    Determine if the audio chunk is silence based on amplitude threshold.
    """
    # Convert audio data to numpy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    # Check if array is empty or contains only zeros
    if len(audio_array) == 0 or np.all(audio_array == 0):
        return True

    # Calculate the RMS value (Use float32 to avoid overflow)
    mn = np.mean(np.square(audio_array.astype(np.float32)))
    rms = np.sqrt(mn)
    return rms < threshold
