import os
import librosa
import pyloudnorm as pyln
import numpy as np
import soundfile as sf
import glob
from tqdm import tqdm

# 忽略警告
import warnings

warnings.filterwarnings('ignore')

sample_rate = 24000
trim_db = 30
wav_db = -20.0
st_save_len = 150  # ms
end_save_len = 200  # ms

meter = pyln.Meter(sample_rate)


def process_wav(filename, file_o):
    wav, sr = librosa.load(filename, sr=sample_rate)
    loudness = meter.integrated_loudness(wav)
    wav = pyln.normalize.loudness(wav, loudness, wav_db)
    if np.abs(wav).max() > 1.0:
        wav = wav / np.abs(wav).max()

    wav = wav.astype(np.float32)
    cut_wav, index = librosa.effects.trim(
        wav,
        top_db=trim_db,
        frame_length=512,
        hop_length=128,
    )

    # index
    st_start, st_end = index[0], index[1]  # 这个可以用来进行手动调整静音长度,上面 cut_wav = wav[st_start, st_end]

    # cut_wav = wav[st_start, st_end]
    # new_s = max(st_start - int(st_save_len * sample_rate / 1000), 0)
    # new_d = min(st_end + int(end_save_len * sample_rate / 1000), len(wav))

    sf.write(file_o, cut_wav, sample_rate)


if __name__ == '__main__':
    wav_dir = "/mnt/data/fangpengcheng/data/ziwei_record/audio_ziwei"  # 语音文件夹
    out_put_path = '/mnt/data/fangpengcheng/data/ziwei_record/audio_no_silence'
    # 对每一个文件进行操作
    for filename in tqdm(os.listdir(wav_dir)):
        save_path = os.path.join(out_put_path, filename.split('/')[-1])
        input_path = os.path.join(wav_dir, filename)
        # 去除静音段
        process_wav(input_path, save_path)
