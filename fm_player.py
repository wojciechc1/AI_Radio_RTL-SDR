import numpy as np
from utils.dll_loader import load_dll
from utils.prepare_audio import prepare_audio
import scipy.signal
import numpy as np
from scipy.io.wavfile import write
import sounddevice as sd
from scipy.signal import decimate
from scipy.io import wavfile
import matplotlib.pyplot as plt

import torch

# load dll file
load_dll("./rtl-sdr-64bit-20180506")
from rtlsdr import RtlSdr

from model.net import UNet1D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet1D()
model.load_state_dict(torch.load("./model/saved_models/trained_model_1.pth", map_location=device))
model.to(device)
model.eval()


class Radio():
    def __init__(self):
        self.sdr = RtlSdr()
        self.sdr.sample_rate = 2.4e6
        # sdr.center_freq = 107.2e6
        # sdr.center_freq = 99.6e6
        self.sdr.center_freq = 89.6e6

        self.sdr.gain = 20
        self.is_playing = True
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
        self.is_playing = True
        with sd.OutputStream(samplerate=self.fs_audio, channels=1) as stream:
            while self.is_playing:
                samples = self.sdr.read_samples(256*1024)
                filtered = self.bandpass_filter(samples, self.fs_signal, 75e3)
                fm_demod = self.fm_demodulate(filtered)
                #fm_demod = self.fm_demodulate(samples)
                fm_audio = decimate(fm_demod, self.decimation_factor)
                fm_audio = self.normalize_audio(fm_audio).astype(np.float32)
                fm_audio = prepare_audio(fm_audio, target_len=8192)

                input_tensor = torch.from_numpy(fm_audio).unsqueeze(0).unsqueeze(0).to(device)  # shape: [1,1,length]
                #print(input_tensor.shape)
                with torch.no_grad():
                    denoised_tensor = model(input_tensor)  # shape: [1,1,length]

                denoised = denoised_tensor.squeeze().cpu().numpy()  # shape: [length]
                denoised = denoised.astype(np.float32)

                stream.write(denoised)

import time
import threading


radio = Radio()
radio.sdr.center_freq = 98.8e6
radio.play()

"""
def run_radio():
    radio.play()

for i in range(200):
    print("---------------------------------------")
    print("start:", radio.sdr.center_freq / 1e6, "MHz")

    # uruchom radio w osobnym wątku
    thread = threading.Thread(target=run_radio)
    thread.start()

    time.sleep(10)  # gra przez 2 sekundy

    # zatrzymaj odtwarzanie
    radio.is_playing = False
    thread.join()  # poczekaj aż się zakończy

    print("stop:", radio.sdr.center_freq / 1e6, "MHz")
    radio.sdr.center_freq += 0.2e6"""