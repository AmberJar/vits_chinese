import os
import sys
from chinese2number import chinese_to_number
from chinese2number import number_to_chinese
from xpinyin import Pinyin

txt_path = '/mnt/data/fangpengcheng/projects/vitsBigGanSpanPSP/test/txt_file/paimon.txt'
save_path = '/mnt/data/fangpengcheng/projects/vitsBigGanSpanPSP/test/txt_file/paimon.txt'

txt_list = []
ctn = chinese_to_number()
ntc = number_to_chinese()
p = Pinyin()

fixed_text = []
with open(txt_path, 'r', encoding='utf-8') as f:
    num_word = [line.strip().split('|') for line in f]

    for line in num_word:
        new_line = []
        left = line[0]
        print_or_not = False
        for item in line[1].split(' '):
            if item.isdigit():
                print_or_not = True
                res = ntc.decimal_chinese(item)
                # ret1 = p.get_pinyin(res, tone_marks='numbers')
                # new_pinyin = ret1.split('-')
                # new_line += new_pinyin
                new_line += res
            else:
                new_line.append(item)

        if print_or_not:
            print(new_line)
        new_line_str = ' '.join(new_line)
        new_line_str = line[0] + '|' + new_line_str
        fixed_text.append(new_line_str)

with open(save_path, "w", encoding="utf-8") as f:
    f.writelines([item + "\n" for item in fixed_text])