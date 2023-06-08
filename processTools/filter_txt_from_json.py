import json
import numpy as np
import cv2
import re
import os
from tqdm import tqdm
import shutil
word = '为将军大人实现｢愿望]'
del_punctuation = "[＊＋／＜＝＞＠＾＿｀｛｜｝～｟｠､〃｢｣『』【】〔〕〖〗〘〙〚〛〜〝〞＂＃＄％＆＇.!+-=——,$%^~@#￥%……&*<>「」{}/\\\[\]'\"（）《》]"

# del_punctuation = "[＂＃＄％＆＇＊＋－／＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､〃「」[]『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛„‟…‧﹏]"
punctuation_list = ['，', '。', '、', '；', '：', '？', '！', '“', '”', '‘', '’', '—', '…', '（', '）', '《', '》']
line = re.sub(del_punctuation, "", word)

names = {}
wav = {}

for name in os.listdir(r'/mnt/data/chenyuxia/yuanshen/yuanshen_labels'):
    mk_name = name.split('.')[0]
    names[mk_name] = []
    wav[mk_name] = []
print(names)

def E_trans_to_C(string):
    E_pun = u',.!?[]()<>"\':'
    C_pun = u'，。！？【】（）《》“‘：'
    table = {ord(f): ord(t) for f, t in zip(E_pun, C_pun)}
    return string.translate(table)


def load_json_file(json_path):
    _category_dict = ['fileName', 'language', 'npcName', 'text']
    with open(json_path, 'r', encoding='utf8') as fp:
        category_dict = json.load(fp)
        # print(category_dict)
        for nums in category_dict.values():
            if ('fileName' not in nums.keys()) or ('language' not in nums.keys()) or ('npcName' not in nums.keys()) or ('text' not in nums.keys()):
                continue
            if nums['npcName'] not in names:
                continue
            if nums['language'] == 'CHS':
                wav_name = nums['fileName'].split('\\')[-1]
                new_name = '{}.wav'.format(wav_name.split('.')[0])
                get_txt = nums['text']
                if bool(re.search('[a-zA-Z]', get_txt)):
                    print(get_txt)
                if 'UNRELEASED' in get_txt:
                    continue
                txt = E_trans_to_C(get_txt)
                chs_txt = re.sub(del_punctuation, "", txt)
                list = (names[nums['npcName']])
                list.append([new_name, chs_txt])
    return names


def load_chinese(json_path, wav_path, txt_path):
    names = load_json_file(json_path)
    for i in os.listdir(wav_path):
        role_path = os.path.join(wav_path, i)
        # for wav_dir in os.listdir(role_path):
        #     wavs = wav_dir.split('.')[0] + '.wav'
        wav['纳西妲'].append(i)
    # wav = [wav_dir.split('.')[0] + '.wav' for wav_dir in os.listdir(wav_path)]
    for name, massges in names.items():
        if wav[name] == None:
            continue
        if name == '纳西妲':
            role_txt_dir = os.path.join(txt_path, name + '.txt')
            with open(role_txt_dir, "w", encoding="utf-8") as f:
                f.writelines(["|".join(item) + "\n" for item in massges if item[0] in wav[name]])



def move(txt_dir,wav_dir,out_put_dir):
    if not os.path.exists(out_put_dir):
        os.mkdir(out_put_dir)
    for name in tqdm(os.listdir(txt_dir)):
        wav_name = name.split('.')[0] + '.wav'
        wav_file = os.path.join(wav_dir, wav_name)
        txt_file = os.path.join(txt_dir, name)

        shutil.copy(wav_file, os.path.join(out_put_dir, wav_name))
        shutil.copy(txt_file, os.path.join(out_put_dir, name))


if __name__ == '__main__':
    # load_chinese(r'/mnt/data/fangpengcheng/data/yuanshen/result.json', r'/mnt/data/fangpengcheng/data/yuanshen/nahida/wav', r'/mnt/data/fangpengcheng/data/yuanshen/nahida')
    # txt_dir = '/mnt/data/chenyuxia/pycharm_bigvgan/dataset/shenzi'
    # wav_dir = '/mnt/data/chenyuxia/yuanshen/yuanshen_big/八重神子'
    # out_put_dir = '/mnt/data/chenyuxia/pycharm_bigvgan/dataset/shenzi'
    move('/mnt/data/fangpengcheng/data/ziwei_record/output_file', '/mnt/data/fangpengcheng/data/yuanshen/paimon/wav', '/mnt/data/fangpengcheng/data/yuanshen/paimon/train_file')







