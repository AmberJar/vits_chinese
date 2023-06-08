#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
import time
import torch
import numpy as np
from pypinyin import Style
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin, load_phrases_dict
from scipy.io import wavfile
import commons
import utils
from model_vits_with_bigvgan import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence
import prosody_txt
from pinyin_dict import pinyin_dict
import soundfile as sf
from pydub import AudioSegment
import random
import datetime
import re
from chinese2number import chinese_to_number
from chinese2number import number_to_chinese
from processTools.remove_noise import remove_noise
from xpinyin import Pinyin as py
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")

ctn = chinese_to_number()
ntc = number_to_chinese()

ROOT = os.path.dirname(os.path.abspath(__file__))


class MyConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass


# define Global Variable
cut_off_punctuation = "，。！？"
CUT_OFF = 200

del_punctuation = "[＊＋／＜＝＞＠＾＿｀｛｜｝～｟｠､〃｢｣『』【】〔〕〖〗〘〙〚〛〜〝〞＂＃＄％＆＇.!+-=——,$%^~@#￥%……&*<>♪「」{}/\\\[\]'\"（）《》”“]"


def E_trans_to_C(string):
    E_pun = u',.!?[]()<>"\':'
    C_pun = u'，。！？【】（）《》“‘：'
    table = {ord(f): ord(t) for f, t in zip(E_pun, C_pun)}
    return string.translate(table)


def load_pinyin_dict():
    my_dict = {}
    with open(ROOT + "/misc/pypinyin-local.dict", "r", encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            cuts = line.strip().split()
            hanzi = cuts[0]
            pinyin = cuts[1:]
            tmp = []
            for one in pinyin:
                onelist = [one]
                tmp.append(onelist)
            my_dict[hanzi] = tmp
    load_phrases_dict(my_dict)


def get_phoneme4pinyin(pinyins):
    result = []
    for pinyin in pinyins:
        if pinyin[:-1] in pinyin_dict:
            tone = pinyin[-1]
            a = pinyin[:-1]
            a1, a2 = pinyin_dict[a]
            result += [a1, a2 + tone, "#0"]
    result.append("sil")
    return result


def chinese_to_phonemes(pinyin_parser, text, single_zw):
    all = 'sil'
    zw_index = 0
    py_list_all = pinyin_parser.pinyin(text, style=Style.TONE3, errors="ignore")
    py_list = [single[0] for single in py_list_all]
    print(py_list)
    print(single_zw)
    for single in single_zw:
        if single == '#':
            all = all[:-2]
            all += single
        elif single.isdigit():
            all += single
        else:
            pyname = pinyin_dict.get(py_list[zw_index][:-1])
            all += ' ' + pyname[0] + ' ' + pyname[1] + py_list[zw_index][-1] + ' ' + '#0'
            zw_index += 1
    all = all + ' ' + 'sil' + ' ' + 'eos'
    return all


def save_wav(wav, path, rate):
    wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
    wavfile.write(path, rate, wav.astype(np.int16))


def save_to_wav1(data, sampling_rate, path):
    # 将音频数据转换为 pydub 音频对象
    audio = AudioSegment(
        data.tobytes(),
        frame_rate=sampling_rate,
        sample_width=data.dtype.itemsize,
        channels=1
    )

    # 将音频对象导出为 AMR 文件
    audio.export(path, format='amr', parameters=['-ar', '8000'])

    # 用 ffmpeg 转换 AMR 文件头
    # os.system('ffmpeg -i output_file.amr -ar 8000 -ac 1 -ab 12.2k -acodec libopencore_amrnb '+ path)


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def get_text(phones, hps):
    text_norm = cleaned_text_to_sequence(phones)
    if hps.data.add_blank:
        text_norm = intersperse(text_norm, 0)

    text_norm = torch.LongTensor(text_norm)
    return text_norm


# def inference(messages, net_g, yl_model):
def inference(messages):
    inference_dataset = torch.utils.data.TensorDataset(messages)
    sys.exit()
    load_pinyin_dict()
    pinyin_parser = Pinyin(MyConverter())
    # define model and load checkpoint
    hps = utils.get_hparams_from_file(ROOT + "/configs/baker_bigvgan_vits.json")
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cpu()
    _ = net_g.eval()

    _ = utils.load_checkpoint(ROOT + "/logs/ziwei/G_71000.pth", net_g, None)

    # check directory existence
    if not os.path.exists(ROOT + "/vits_out"):
        os.makedirs(ROOT + "/vits_out")

    # load rythm model
    yl_model = prosody_txt.init_model()

    # process text
    messages = E_trans_to_C(messages)

    # process text
    new_messages = ''
    num = ''
    for item in messages:
        if bool(re.search('[a-zA-Z]', item)):
            continue
        if not item.isdigit():
            res = ntc.decimal_chinese(num)
            new_messages += res
            num = ''
        else:
            num += item
            continue
        new_messages += item
    if num != '':
        new_messages += ntc.decimal_chinese(num)
    message = re.sub(del_punctuation, "，", new_messages)
    print('message ---------------', message)
    single_zw = ''
    prosody_txt.run_auto_labels(yl_model, message)  # 输入中文
    with open(ROOT + '/temp.txt', 'r') as r:
        for line in r.readlines():
            line = line.strip()
            single_zw += line + '#3'
    single_zw = single_zw[:-1] + '4'  # 输出中文+#

    new_single_zw = ''
    for index, item in enumerate(single_zw):
        if item.isdigit() and single_zw[index - 1] != '#':
            if single_zw[index - 1].isdigit():
                print(single_zw)
            else:
                print(single_zw)
                new_single_zw += '#'
                new_single_zw += item
        else:
            new_single_zw += item

    phonemes = chinese_to_phonemes(pinyin_parser, message, single_zw)
    input_ids = get_text(phonemes, hps)
    with torch.no_grad():
        x_tst = input_ids.unsqueeze(0).cuda()
        x_tst_lengths = torch.LongTensor([input_ids.size(0)]).cuda()
        noise_scale = random.uniform(0, 1)
        noise_scale_w = random.uniform(0, 1)
        audio = net_g.cuda().infer(x_tst, x_tst_lengths, noise_scale, noise_scale_w, length_scale=1)[0][
            0, 0].data.cpu().float().numpy()

    timestamp = int(time.time())
    tmp_audio_path_1 = ROOT + f"/vits_out/{timestamp}tmp_audio_path_1.wav"
    tmp_audio_path_2 = ROOT + f"/vits_out/{timestamp}tmp_audio_path_2.wav"

    sample_rate = 16000
    save_wav(audio, tmp_audio_path_1, sample_rate)

    audio = remove_noise(tmp_audio_path_1, tmp_audio_path_2, sample_rate)
    print(message)
    print(phonemes)
    print(input_ids)

    return audio


def split_message(message, mid):
    msg_list = []
    while len(message) > CUT_OFF:
        num = split_text(message, mid)
        msg_list.append(message[0:num])
        message = message[num:]

    msg_list.append(message)

    return msg_list


def split_text(text, mid):
    if mid - 1 == 0 or mid + 1 == len(text):
        return -1
    if text[mid] in cut_off_punctuation:
        return mid + 1
    else:
        if mid - 1 > 0:
            return split_text(text, mid - 1)
        if mid + 1 <= len(text):
            return split_text(text, mid + 1)


if __name__ == "__main__":
    text = '惊蛰一过，春寒加剧。先是料料峭峭，继而雨季开始，时而淋淋漓漓，时而淅淅沥沥，天潮潮地湿湿，即连在梦里，也似乎有把伞撑着。而就凭一把伞，躲过一阵潇潇的冷雨，也躲不过整个雨季。连思想也都是潮润润的。每天回家，曲折穿过金门街到厦门街迷宫式的长巷短巷，雨里风里，走入霏霏令人更想入非非。想这样子的台北凄凄切切完全是黑白片的味道，想整个中国整部中国的历史无非是一张黑白片子，片头到片尾，一直是这样下着雨的。这种感觉，不知道是不是从安东尼奥尼那里来的。不过那—块土地是久违了，二十五年，四分之一的世纪，即使有雨，也隔着千山万山，千伞万伞。 '
    text1 = '这种感觉，不知道是不是从安东尼奥尼那里来的。不过那—块土地是久违了，二十五年，四分之一的世纪，即使有雨，也隔着千山万山，千伞万伞。'
    text2 = '你好！有什么我可以帮助你的吗？'

    input = [text2] * 20
    input = np.asarray(input)
    print(input.shape)
    inference(input)
    # res = split_message(text, CUT_OFF)
    # print(res)
    # print(len(res[0]), len(res[1]))
