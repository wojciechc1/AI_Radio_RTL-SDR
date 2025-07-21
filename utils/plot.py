import matplotlib.pyplot as plt
import numpy as np

def plot(samples, center_freq):
    sample_rate = 2.4e6
    fft_vals = np.fft.fftshift(np.fft.fft(samples))
    power = np.abs(fft_vals) / len(samples)
    power_db = 20 * np.log10(power + 1e-12)

    freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), 1 / sample_rate))
    freqs_real = freqs / 1e6 + center_freq / 1e6

    plt.plot(freqs_real, power_db)
    plt.xlabel("Częstotliwość (MHz)")
    plt.ylabel("Moc (dB)")
    plt.title("Widmo sygnału RTL-SDR")
    plt.grid()
    plt.ylim(power_db.min() - 10, power_db.max() + 5)
    plt.tight_layout()
    plt.show()
