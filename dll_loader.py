import ctypes
import os

def load_dll():
    dll_folder = os.path.abspath("rtl-sdr-64bit-20180506")
    dll_path = os.path.join(dll_folder, "librtlsdr.dll")

    os.environ['PATH'] = dll_folder + ';' + os.environ['PATH']

    try:
        ctypes.cdll.LoadLibrary(dll_path)
        print("librtlsdr.dll załadowany poprawnie")
    except Exception as e:
        print("Błąd ładowania librtlsdr.dll:", e)