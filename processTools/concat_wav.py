import wave
import os
import numpy as np
import struct
from scipy.io.wavfile import read
from glob import glob


def creat_wav(duration, framerate,path_file):
    sample_width = 2
    frequency = 2000
    volume = 1000
    x = np.linspace(0, duration, num=int(duration * framerate))
    y = np.sin(0 * np.pi * frequency * x) * volume
    sine_wave = y
    sine = os.path.join(path_file, 'black.wav')
    with wave.open(sine, 'wb') as wf:
        wf.setnchannels(1)
        wf.setframerate(framerate)
        wf.setsampwidth(sample_width)
        for i in sine_wave:
            data = struct.pack('<h', int(i))
            wf.writeframesraw(data)
        wf.close()


def concat_wav(int_path, out_path, character_name):
    # infiles = [wav_file for wav_file in os.listdir(int_path)]
    infiles = sorted(glob(os.path.join(int_path, f'aft*{character_name}*')))
    print(infiles)
    data = []
    fs, audio = read(os.path.join(int_path, infiles[0]))
    for infile in range(1, len(infiles)+1):
        # name = str(infile) + 'naxi_baker_' + '.wav'
        name = f"aft_{str(infile)}_{character_name}_.wav"
        infile = os.path.join(int_path, name)
        w = wave.open(infile, 'rb')
        data.append([w.getparams(), w.readframes(w.getnframes())])
        w.close()
        os.remove(infile)
    output = wave.open(out_path, 'wb')
    print(len(data))
    output.setparams(data[0][0])
    creat_wav(0.7, fs, int_path)
    # black = wave.open((os.path.join(int_path, 'black.wav')), 'rb')
    for i in range(0, len(data)):
        black = wave.open((os.path.join(int_path, 'black.wav')), 'rb')
        output.writeframes(data[i][1])
        output.writeframes(black.readframes(black.getnframes()))
    output.close()

if __name__ == '__main__':
    concat_wav(r'/mnt/data/chenyuxia/pycharm_bigvgan/vits_out', r'/mnt/data/chenyuxia/pycharm_bigvgan/vits_out/out.wav')