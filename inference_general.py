import os
import sys

sys.path.append('./vits')
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录的绝对路径
print("[current_dir]", current_dir)
sys.path.append(current_dir)  # 将上上级目录的路径添加到 sys.path 中
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import numpy as np
from pypinyin import Style
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin, load_phrases_dict
from scipy.io import wavfile

import utils
import prosody_txt
import commons
from model_vits_with_bigvgan import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence
from pinyin_dict import pinyin_dict

from processTools.remove_noise import remove_noise
from processTools.concat_wav import concat_wav

import random
import re
from chinese2number import chinese_to_number
from chinese2number import number_to_chinese

ctn = chinese_to_number()
ntc = number_to_chinese()


ROOT = os.path.dirname(os.path.abspath(__file__))


class MyConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass


del_punctuation = "[＊＋／＜＝＞＠＾＿｀｛｜｝～｟｠､〃｢｣『』【】〔〕〖〗〘〙〚〛〜〝〞＂＃＄％＆＇.!+-=——,$%^~@#￥%……&*<>♪「」{}/\\\[\]'\"（）《》]"

# define Global Variable
cut_off_punctuation = "，。！？"
CUT_OFF = 200


def E_trans_to_C(string):
    E_pun = u',.!?[]()<>"\':'
    C_pun = u'，。！？【】（）《》“‘：'
    table = {ord(f): ord(t) for f, t in zip(E_pun, C_pun)}
    return string.translate(table)


def load_pinyin_dict():
    my_dict = {}
    with open(ROOT + "./misc/pypinyin-local.dict", "r", encoding='utf-8') as f:
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


def get_text(phones, hps):
    text_norm = cleaned_text_to_sequence(phones)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def inference(sendfrom, userid, messages):
    tmp_path = r'./vits_out/tmp.wav'
    out_put_path = r'./vits_out/without_noise.wav'

    load_pinyin_dict()
    pinyin_parser = Pinyin(MyConverter())
    # define model and load checkpoint

    hps = utils.get_hparams_from_file("./configs/baker_bigvgan_vits.json")
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cpu()
    _ = net_g.eval()

    # define the dir name
    character_name = 'ziwei'
    _ = utils.load_checkpoint("./logs/{}/G_69000.pth".format(character_name), net_g, None)

    # load rythm model
    yl_model = prosody_txt.init_model()

    fo = open("vits_strings.txt", "r+")
    n = 0

    while (True):
        try:
            message = fo.readline().strip()
            txt = E_trans_to_C(message)
            new_messages = ''
            num = ''
            for item in txt:
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
            new_messages = re.sub(del_punctuation, "，", new_messages)
        except Exception as e:
            print('nothing of except:', e)
            break
        if (message == None):
            break
        if (message == ""):
            break
        n = n + 1
        single_zw = ''

        prosody_txt.run_auto_labels(yl_model, new_messages)  # 输入中文
        with open('temp.txt', 'r') as r:
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

        phonemes = chinese_to_phonemes(pinyin_parser, new_messages, new_single_zw)
        input_ids = get_text(phonemes, hps)
        with torch.no_grad():
            x_tst = input_ids.unsqueeze(0).cpu()
            x_tst_lengths = torch.LongTensor([input_ids.size(0)]).cpu()
            audio = net_g.cpu().infer(x_tst, x_tst_lengths, length_scale=1)[0][
                0, 0].data.cpu().float().numpy()

        print(audio.shape)

        pre_save_path = f"./vits_out/pre_{n}_{character_name}_.wav"
        aft_save_path = f"./vits_out/aft_{n}_{character_name}_.wav"

        sample_rate = 16000
        save_wav(audio, pre_save_path, sample_rate)
        remove_noise(pre_save_path, tmp_path, aft_save_path, sample_rate)

        # main(f"./vits_out/{n}_baker_.wav", 16000)
        print(message)
        print(phonemes)
        print(input_ids)
    fo.close()

    if n > 1:
        # 29
        # output_path = fr'/mnt/data/fangpengcheng/projects/vitsBigGanSpanPSP/vits_out/concat_out.wav'
        output_path = fr'/data/fpc/projects/vitsBigGan/vits_out/1_{character_name}_concat_out.wav'
        if os.path.exists(output_path):
            os.remove(output_path)
        # 29
        # concat_wav(r'/mnt/data/fangpengcheng/projects/vitsBigGanSpanPSP/vits_out/', output_path, character_name)
        concat_wav(r'/data/fpc/projects/vitsBigGan/vits_out/', output_path, character_name)
    # print(time.time() - start)