import numpy as np
from utils.dll_loader import load_dll
import scipy.signal
import numpy as np
from scipy.io.wavfile import write
import sounddevice as sd
from scipy.signal import decimate
from scipy.io import wavfile
import matplotlib.pyplot as plt

# load dll file
load_dll("./rtl-sdr-64bit-20180506")
from rtlsdr import RtlSdr



def fm_demodulate(x):
    y = x[1:] * np.conj(x[:-1])
    return np.angle(y) * 1.0

def normalize_audio(audio):
    audio = audio - np.mean(audio)
    max_val = np.max(np.abs(audio))
    if max_val == 0:
        return audio
    return (audio / max_val) * 0.8

def bandpass_filter(samples, fs, cutoff=100e3):
    nyq = fs / 2
    b, a = scipy.signal.butter(5, cutoff / nyq, btype='low')
    return scipy.signal.lfilter(b, a, samples)


sdr = RtlSdr()
sdr.sample_rate = 2.4e6
#sdr.center_freq = 107.2e6
sdr.center_freq = 99.6e6

sdr.gain = 25

fs_signal = int(sdr.sample_rate)
fs_audio = 48000
decimation_factor = fs_signal // fs_audio

with sd.OutputStream(samplerate=fs_audio, channels=1) as stream:
    while True:
        samples = sdr.read_samples(256*1024)
        filtered = bandpass_filter(samples, fs_signal, 100e3)
        fm_demod = fm_demodulate(filtered)
        #fm_demod = fm_demodulate(samples)
        fm_audio = decimate(fm_demod, decimation_factor)
        fm_audio = normalize_audio(fm_audio).astype(np.float32)

        stream.write(fm_audio)