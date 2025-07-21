from utils.dll_loader import load_dll
from utils.plot import plot

# load dll file
load_dll("./rtl-sdr-64bit-20180506")

from rtlsdr import RtlSdr

def main():
    sdr = RtlSdr()
    print("RTL-SDR działa")

    # Ustawienia
    sdr.sample_rate = 2.4e6    # 2.4 MHz
    sdr.center_freq = 100e6    # 100 MHz
    sdr.gain = 'auto'          # automatyczne wzmocnienie

    print("Pobieram próbki...")
    samples = sdr.read_samples(256*1024)  # 256k próbek

    print(f"Pobrano {len(samples)} próbek")
    print("Pierwsze 10 próbek (complex64):")
    for i in range(10):
        print(samples[i])

    plot(samples, sdr.center_freq)

    sdr.close()

if __name__ == "__main__":
    main()
