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

class Radio():
    def __init__(self):
        self.sdr = RtlSdr()
        self.sdr.sample_rate = 2.4e6
        # sdr.center_freq = 107.2e6
        # sdr.center_freq = 99.6e6
        self.sdr.center_freq = 87.5e6

        self.sdr.gain = 30

        self.fs_signal = int(self.sdr.sample_rate)
        self.fs_audio = 48000
        self.decimation_factor = self.fs_signal // self.fs_audio

    def fm_demodulate(self, x):
        y = x[1:] * np.conj(x[:-1])
        return np.angle(y) * 1.0

    def normalize_audio(self, audio):
        audio = audio - np.mean(audio)
        max_val = np.max(np.abs(audio))
        if max_val == 0:
            return audio
        return (audio / max_val) * 0.8

    def bandpass_filter(self, samples, fs, cutoff=100e3):
        nyq = fs / 2
        b, a = scipy.signal.butter(5, cutoff / nyq, btype='low')
        return scipy.signal.lfilter(b, a, samples)


    def play(self):
        with sd.OutputStream(samplerate=self.fs_audio, channels=1) as stream:
            while True:
                samples = self.sdr.read_samples(256*1024)
                filtered = self.bandpass_filter(samples, self.fs_signal, 100e3)
                fm_demod = self.fm_demodulate(filtered)
                #fm_demod = fm_demodulate(samples)
                fm_audio = decimate(fm_demod, self.decimation_factor)
                fm_audio = self.normalize_audio(fm_audio).astype(np.float32)

                stream.write(fm_audio)


radio = Radio()
radio.sdr.center_freq = 98.8e6
print(radio.sdr.center_freq)
radio.play()