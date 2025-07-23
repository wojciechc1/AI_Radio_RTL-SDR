from pydub import AudioSegment


mp3_dir = './data/6e.mp3' # sciezka istnejacego pliku mp3
wav_dir = "./data/eval/1.wav" # docelowa sciezka utworzonego wav

def convert_mp3_to_wav(mp3_path):
    sound = AudioSegment.from_mp3(mp3_path)
    sound.export(wav_dir, format="wav")


convert_mp3_to_wav(mp3_dir)