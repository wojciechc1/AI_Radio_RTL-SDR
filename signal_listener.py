import numpy as np
from utils.dll_loader import load_dll
import time
from collections import deque

load_dll("./rtl-sdr-64bit-20180506")
from rtlsdr import RtlSdr

sdr = RtlSdr()
sdr.sample_rate = 1.2e6  # szerokosc pasma
sdr.center_freq = 433.92e6  # 433.92 MHz - standardowy pilot do auta w EU
sdr.gain = 35



check_interval = 0.1

background = deque(maxlen=20)

print("Nasłuchiwanie sygnału. Częstotliwość: ", sdr.center_freq)

try:
    while True:
        samples = sdr.read_samples(64*1024)
        power = np.abs(samples)**2
        rms = np.sqrt(np.mean(power))

        background.append(rms)
        avg_rms = np.mean(background)

        if rms > avg_rms * 1.1:
            print(f"[{time.strftime('%H:%M:%S')}] WYKRYTO SYGNAŁ | RMS = {rms:.4f}")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Cisza / Szum | RMS = {rms:.4f}")

        time.sleep(check_interval)

except KeyboardInterrupt:
    print("Zatrzymano nasłuchiwanie.")
    sdr.close()
