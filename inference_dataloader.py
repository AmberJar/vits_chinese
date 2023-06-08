import time
import os
import random
import numpy as np
import torch
import torch.utils.data
from glob import glob
import commons
from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text
# from text import text_to_sequence, cleaned_text_to_sequence
import torchaudio


class TextAudioLoader(torch.utils.data.Dataset):
    """
        1) loads audio, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_and_text, hparams, is_train=False):

        self.spk_path = glob(os.path.join(audiopaths_and_text,'*'))

        print("Speaker num", len(self.spk_path))
        self.is_train = is_train
        self.npzs, self.spk_label = self.get_npz_path(self.spk_path)

        print("Total data len: ", len(self.npzs))
        # self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 1000)

        c = list(zip(self.npzs, self.spk_label))
        random.seed(1234)
        random.shuffle(c)
        self.npzs, self.spk_label = zip(*c)

        self._filter()
        print("filtered data len: ", len(self.npzs))

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length
        npz_new = []
        lengths = []
        spk_new = []
        for npz, spk in zip(self.npzs, self.spk_label):
            temp = np.load(npz)
            # if len(temp['audio'])//256 < 400:
            npz_new.append(npz)
            lengths.append(len(temp['audio']) // (self.hop_length))
            spk_new.append(spk)

        self.lengths = lengths
        self.npzs = npz_new
        self.spk_label = spk_new

    def get_npz_path(self, spk_path):
        npz_path = []
        speaker_label = []
        i = 0
        for spk in spk_path:
            if self.is_train:
                temp_path = glob(os.path.join(spk, os.path.join("train", "*.npz")))
                npz_path += temp_path
                speaker_label += [i]*len(temp_path)

            else:
                temp_path = glob(os.path.join(spk, os.path.join("test", "*.npz")))
                npz_path += temp_path
                speaker_label += [i]*len(temp_path)
            i +=1
        return npz_path, speaker_label

    def get_audio_text_pair(self, audiopath_and_text, spk_label):

        files = np.load(audiopath_and_text)
        text = self.add_blank_token(files['token'])
        spec, wav = self.get_audio(files['audio'], audiopath_and_text, language=0)
        spk_id = spk_label

        return (text, spec, wav, spk_id)

    def get_audio(self, audio, filename, language=0):

        audio = torch.FloatTensor(audio.astype(np.float32))
        audio_norm = audio / self.max_wav_value * 0.95
        audio_norm = audio_norm.unsqueeze(0)

        spec_filename = filename.replace(".npz", "." + str(language) + "_spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename, map_location='cpu')
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                                     self.sampling_rate, self.hop_length, self.win_length,
                                     center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)

        return spec, audio_norm

    def add_blank_token(self, text):
        if self.add_blank:
            text = commons.intersperse(text, 0)
        text_norm = torch.LongTensor(text)
        return text_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.npzs[index], self.spk_label[index])

    def __len__(self):
        return len(self.npzs)