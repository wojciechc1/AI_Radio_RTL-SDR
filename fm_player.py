import numpy as np
from utils.dll_loader import load_dll
import scipy.signal
import sounddevice as sd

# load dll file
load_dll("./rtl-sdr-64bit-20180506")


from rtlsdr import RtlSdr
def fm_demodulate(samples, sample_rate, cutoff=100e3, decimation=5):
    # Deemphasis + low-pass filtering
    x = np.angle(samples[1:] * np.conj(samples[:-1]))
    b, a = scipy.signal.butter(5, cutoff / (sample_rate / 2), btype='low')
    x = scipy.signal.lfilter(b, a, x)
    # Decymacja (redukcja próbkowania)
    x = scipy.signal.decimate(x, decimation)
    return x


# Ustawienia RTL-SDR
sdr = RtlSdr()
sdr.sample_rate = 2.4e6   # szerokość pasma
sdr.center_freq = 100.0e6 # np. RMF FM
sdr.gain = 40


print("Odbieram FM... Naciśnij Ctrl+C by zakończyć.")
try:
    while True:
        samples = sdr.read_samples(256*1024)
        audio = fm_demodulate(samples, sdr.sample_rate)
        audio /= np.max(np.abs(audio))  # normalizacja
        sd.play(audio, samplerate=48000, blocking=True)

        #sd.play(audio, samplerate=int(sdr.sample_rate / 5), blocking=True)

except KeyboardInterrupt:
    print("Zakończono.")
finally:
    sdr.close()
