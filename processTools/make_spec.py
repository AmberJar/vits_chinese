import librosa
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.io.wavfile import read
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(dpi=600) # 将显示的所有图分辨率调高
matplotlib.rc("font", family='simhei') # 显示中文
matplotlib.rcParams['axes.unicode_minus']=False # 显示符号


def displayWaveform(wav_path,rate): # 显示语音时域波形
    """
    display waveform of a given speech sample
    :param sample_name: speech sample name
    :param fs: sample frequency
    :return:
    """
    sr, samples = read(wav_path, rate)
    # samples = samples[6000:16000]

    print(len(samples), sr)
    time = np.arange(0, len(samples)) * (1.0 / sr)

    plt.plot(time, samples)
    plt.title("语音信号时域波形")
    plt.xlabel("时长（秒）")
    plt.ylabel("振幅")
    plt.savefig(r"/mnt/data/fangpengcheng/projects/vitsBigGanSpanPSP/vits_out/语音信号时域波形图", dpi=600)
    plt.show()

def displaySpectrum(wav_path, rate): # 显示语音频域谱线
    sr, x = read(wav_path, rate)
    print(len(x))
    # ft = librosa.stft(x)
    # magnitude = np.abs(ft)  # 对fft的结果直接取模（取绝对值），得到幅度magnitude
    # frequency = np.angle(ft)  # (0, 16000, 121632)

    ft = fft(x)
    print(len(ft), type(ft), np.max(ft), np.min(ft))
    magnitude = np.absolute(ft)  # 对fft的结果直接取模（取绝对值），得到幅度magnitude
    frequency = np.linspace(0, sr, len(magnitude))  # (0, 16000, 121632)

    print(len(magnitude), type(magnitude), np.max(magnitude), np.min(magnitude))
    print(len(frequency), type(frequency), np.max(frequency), np.min(frequency))

    # plot spectrum，限定[:40000]
    # plt.figure(figsize=(18, 8))
    plt.plot(frequency[:40000], magnitude[:40000])  # magnitude spectrum
    plt.title("语音信号频域谱线")
    plt.xlabel("频率（赫兹）")
    plt.ylabel("幅度")
    plt.savefig(r"/mnt/data/fangpengcheng/projects/vitsBigGanSpanPSP/vits_out/语音信号频谱图", dpi=600)
    plt.show()

    # # plot spectrum，不限定 [对称]
    # plt.figure(figsize=(18, 8))
    # plt.plot(frequency, magnitude)  # magnitude spectrum
    # plt.title("语音信号频域谱线")
    # plt.xlabel("频率（赫兹）")
    # plt.ylabel("幅度")
    # plt.show()


def displaySpectrogram(wav_path, rate):
    sr, x = read(wav_path, rate)

    # compute power spectrogram with stft(short-time fourier transform):
    # 基于stft，计算power spectrogram
    spectrogram = librosa.amplitude_to_db(librosa.stft(x))

    # show
    librosa.display.specshow(spectrogram, y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('语音信号对数谱图')
    plt.xlabel('时长（秒）')
    plt.ylabel('频率（赫兹）')
    plt.show()

def main(wav_path, sample_rate):
    displayWaveform(wav_path, sample_rate)
    displaySpectrum(wav_path, sample_rate)
    # displaySpectrogram(wav_path, sample_rate)