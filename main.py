import ctypes
import os

dll_folder = os.path.abspath("rtl-sdr-64bit-20180506t")
dll_path = os.path.join(dll_folder, "librtlsdr.dll")

os.environ['PATH'] = dll_folder + ';' + os.environ['PATH']

try:
    ctypes.cdll.LoadLibrary(dll_path)
    print("librtlsdr.dll załadowany poprawnie!")
except Exception as e:
    print("Błąd ładowania librtlsdr.dll:", e)

from rtlsdr import RtlSdr

def main():
    sdr = RtlSdr()
    print("RTL-SDR działa!")

    # Ustawienia (przykładowe)
    sdr.sample_rate = 2.4e6    # 2.4 MHz
    sdr.center_freq = 100e6    # 100 MHz (FM radio)
    sdr.gain = 'auto'          # automatyczne wzmocnienie

    print("Pobieram próbki...")
    samples = sdr.read_samples(256*1024)  # 256k próbek

    print(f"Pobrano {len(samples)} próbek.")
    print("Pierwsze 10 próbek (complex64):")
    for i in range(10):
        print(samples[i])

    sdr.close()

if __name__ == "__main__":
    main()
