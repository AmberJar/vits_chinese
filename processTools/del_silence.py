import sys

import librosa
import soundfile as sf
import numpy as np
from scipy.signal import medfilt
import os
from tqdm import tqdm
import pyloudnorm as pyln

# 忽略警告
import warnings

warnings.filterwarnings('ignore')

# 参数
sample_rate = 24000
wav_db = -20.0

meter = pyln.Meter(sample_rate)

'''
将帧转换为时间刻度
'''


def frame2Time(frameNum, framelen, inc, fs):
    frames = np.array(range(0, frameNum, 1))
    frames = frames * inc + framelen / 2
    frameTime = frames / fs
    return frameTime


'''
去除静音的函数
'''


def slience(filename, sr):
    frame_threshold = 10  # 该参数决定去掉连续多少帧的静音段，比如某段语音检测到有12帧的静音帧，则去掉这一段的语音，而如果检测到只有8帧，那么不操作

    # 求取MFCCs参数

    y, sr = librosa.load(filename, sr=sr)
    loudness = meter.integrated_loudness(y)
    y = pyln.normalize.loudness(y, loudness, wav_db)
    if np.abs(y).max() > 1.0:
        y = y / np.abs(y).max()
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24, win_length=1024, hop_length=512, n_fft=1024)

    # # 对mfcc进行中值滤波
    Mfcc1 = medfilt(mfccs[0, :], 9)
    pic = Mfcc1
    start = 0
    end = 0
    points = []
    min_data = min(pic) * 0.9

    for i in range((pic.shape[0])):
        if (pic[i] < min_data and start == 0):
            start = i
        if (pic[i] < min_data and start != 0):
            end = i

        elif (pic[i] > min_data and start != 0):
            hh = [start, end]
            points.append(hh)
            start = 0

    # 解决 文件的最后为静音
    if (pic[-1] < min_data and start != 0):
        hh = [start, end]
        points.append(hh)
        start = 0
    distances = []
    for i in range(len(points)):

        two_ends = points[i]
        distance = two_ends[1] - two_ends[0]
        if (distance > frame_threshold):
            distances.append(points[i])

    # 保存到本地文件夹
    name = filename.split('\\')[-1]

    # 取出来端点，按照端点，进行切割,分情况讨论：1.如果未检测到静音段 2.检测到静音段

    if (len(distances) == 0):
        # print('检测到的静音段的个数为： %s 未对文件进行处理：' % len(distances))
        return y
        # sf.write(slience_clean, clean_data, 16000)

    else:
        slience_data = []
        for i in range(len(distances)):
            if (i == 0):
                start, end = distances[i]
                # 将左右端点转换到 采样点

                if (start == 1):
                    internal_clean = y[0:0]
                else:
                    # 求取开始帧的开头
                    start = (start - 1) * 512
                    # 求取结束帧的结尾
                    end = (end - 1) * 512 + 1024
                    internal_clean = y[0:start - 1]

            else:
                _, end = distances[i - 1]
                start, _ = distances[i]
                start = (start - 1) * 512
                end = (end - 1) * 512 + 1024
                internal_clean = y[end + 1:start]

            hhh = np.array(internal_clean)
            # 开始拼接
            slience_data.extend(internal_clean)

        # 开始 添加 最后一部分,需要分情况讨论，1. 文件末尾本来就是静音的  2.文件末尾不是静音的
        ll = len(distances)
        _, end = distances[ll - 1]
        end = (end - 1) * 512 + 1024
        end_part_clean = y[end:len(y)]
        slience_data.extend(end_part_clean)
        # 写到本地
        # sf.write("./data/{}.wav".format(name), slience_data, 16000)

        return slience_data


'''
获取输入文件夹内的所有wav文件，并返回文件名全称列表
'''


def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.wav':
                filename = os.path.join(root, file)
                # print(get_label(filename))
                L.append(filename)
        return L


'''
储存wav
'''


def save_to_wav(data, path, sr):
    sf.write(path, data, sr, 'PCM_16')


if __name__ == '__main__':
    wav_dir = "/mnt/data/fangpengcheng/data/ziwei_record/audio_ziwei"  # 语音文件夹
    out_put_path = '/mnt/data/fangpengcheng/data/ziwei_record/audio_no_silence'
    sampling_rate = 24000
    wav_files = file_name(wav_dir)  # 获取文件夹内的所有语音文件
    # 对每一个文件进行操作
    for filename in tqdm(wav_files):
        # 去除静音段
        silence_data = slience(filename, sampling_rate)
        save_path = os.path.join(out_put_path, filename.split('/')[-1])
        save_to_wav(silence_data, save_path, sampling_rate)

