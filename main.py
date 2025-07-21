from dll_loader import load_dll

load_dll()

from rtlsdr import RtlSdr

def main():
    sdr = RtlSdr()
    print("RTL-SDR działa")

    # Ustawienia (przykładowe)
    sdr.sample_rate = 2.4e6    # 2.4 MHz
    sdr.center_freq = 100e6    # 100 MHz (FM radio)
    sdr.gain = 'auto'          # automatyczne wzmocnienie

    print("Pobieram próbki...")
    samples = sdr.read_samples(256*1024)  # 256k próbek

    print(f"Pobrano {len(samples)} próbek")
    print("Pierwsze 10 próbek (complex64):")
    for i in range(10):
        print(samples[i])

    sdr.close()

if __name__ == "__main__":
    main()
