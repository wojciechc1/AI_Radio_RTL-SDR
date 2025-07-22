from utils.dll_loader import load_dll
from utils.plot import plt
import numpy as np

# load dll file
load_dll("./rtl-sdr-64bit-20180506")

from rtlsdr import RtlSdr



import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr

import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr


def scan_frequencies(start_freq=88e6, stop_freq=108e6, step=100e3, sample_rate=2.4e6, n_samples=2048, gain=40):
    sdr = RtlSdr()
    sdr.sample_rate = sample_rate
    sdr.gain = 'auto'

    freqs = np.arange(start_freq, stop_freq, step)
    powers = []

    print("Rozpoczynam skanowanie...")

    for f in freqs:
        sdr.center_freq = f
        samples = sdr.read_samples(n_samples)
        power = 20 * np.log10(np.mean(np.abs(samples)**2))
        powers.append(power)
        print(f"Freq: {f/1e6:.3f} MHz, Power: {power:.2f} dB")

    sdr.close()

    powers = np.array(powers)

    # Wypisz 5 najsilniejszych częstotliwości
    idx_top = np.argsort(powers)[-5:][::-1]
    print("\nTop 5 sygnałów:")
    for i in idx_top:
        print(f"{freqs[i]/1e6:.3f} MHz : {powers[i]:.2f} dB")

    # Wykres
    plt.figure(figsize=(12,6))
    plt.plot(freqs / 1e6, powers, marker='o')
    plt.xlabel("Częstotliwość [MHz]")
    plt.ylabel("Moc [dB]")
    plt.title("Skanowanie pasma FM")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    scan_frequencies()