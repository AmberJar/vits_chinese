import warnings
warnings.warn('ignore')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import sys

import numpy as np
from pypinyin import Style
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin, load_phrases_dict
from scipy.io import wavfile
import torch
import commons
import utils
from model_vits_with_bigvgan import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence
import prosody_txt
from pinyin_dict import pinyin_dict
from tqdm import tqdm
from chinese2number import number_to_chinese

class MyConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass


def load_pinyin_dict():
    my_dict = {}
    with open("misc/pypinyin-local.dict", "r", encoding='utf-8') as f:
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


load_pinyin_dict()
pinyin_parser = Pinyin(MyConverter())

# define model and load checkpoint
hps = utils.get_hparams_from_file("configs/baker_bigvgan_vits.json")
# net_g = SynthesizerTrn(
#     len(symbols),
#     hps.data.filter_length // 2 + 1,
#     hps.train.segment_size // hps.data.hop_length,
#     **hps.model).cuda()
# _ = net_g.eval()
#
# _ = utils.load_checkpoint("/mnt/data/fangpengcheng/projects/vitsBigGanSpanPSP/log/baker/G_745000.pth", net_g, None)


if __name__ == "__main__":
    yl_model = prosody_txt.init_model()

    txt_path = '/data/fangpengcheng/data/Audios/senior_man/壮年男性.txt'
    # check directory existence
    output_dir = '/data/fangpengcheng/data/Audios/senior_man/output_file'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ntc = number_to_chinese()

    fo = open(txt_path, "r+")
    for line in tqdm(fo.readlines()):
        save_name = line.split('|')[0].split('/')[-1].split('.')[0] + '.txt'
        message = line.split('|')[1]

        tmp_message = ''
        number_save = ''
        for index, item in enumerate(message):
            if item.isalpha():
                continue
            elif item.isdigit():
                number_save += item
            elif number_save:
                res = ntc.decimal_chinese(number_save)
                if 9 < int(number_save) < 20:
                    res = res[1:]
                tmp_message += res
                tmp_message += item
                number_save = ''
            else:
                tmp_message += item

            if index == len(message)-1:
                res = ntc.decimal_chinese(number_save)
                if 9 < int(number_save) < 20:
                    res = res[1:]
                tmp_message += res

        message = tmp_message
        print(message)

        single_zw = ''
        prosody_txt.run_auto_labels(yl_model, message)
        with open('temp.txt', 'r') as r:
            for line in r.readlines():
                line = line.strip()
                single_zw += line + '#3'
        single_zw = single_zw[:-1] + '4'

        write_trigger = False
        print_or_not = False
        new_single_zw = ''
        for index, item in enumerate(single_zw):
            if item.isdigit() and single_zw[index-1] != '#':
                if single_zw[index - 1].isdigit():
                    print_or_not = True
                else:
                    print_or_not = True
                    new_single_zw += '#'
                    new_single_zw += item
            else:
                new_single_zw += item

        if print_or_not:
            # pass
            print(new_single_zw)
        phonemes = chinese_to_phonemes(pinyin_parser, message, new_single_zw)

        txt_save_path = os.path.join(output_dir, save_name)
        with open(txt_save_path, 'w') as r:
            r.write(phonemes)
        # except Exception as e:
        #     print(e)
        #     print('失败了！！！', line)
    fo.close()
