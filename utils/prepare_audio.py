import numpy as np

def prepare_audio(audio, target_len=8192):
    if len(audio) > target_len:
        return audio[:target_len]
    elif len(audio) < target_len:
        pad_len = target_len - len(audio)
        return np.pad(audio, (0, pad_len))
    return audio