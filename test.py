import random
import re

del_punctuation = "[＊＋／＜＝＞＠＾＿｀｛｜｝～｟｠､〃｢｣『』【】〔〕〖〗〘〙〚〛〜〝〞＂＃＄％＆＇.!+-=——,$%^~@#￥%……&*<>♪「」{}/\\\[\]'\"（）《》”“]"

txt_path = '/data/fangpengcheng/projects/vits/biaozhun_res.txt'
string_count = ''
with open(txt_path, 'r') as f:
    f1 = f.readlines()

process_list = []
for item in f1:
    item = item.strip()
    process_list.append(item)
    # try:
    #     item = item.split('|')[1]
    # except:
    #     continue

# print(process_list)
# process_list = sorted(process_list, key=lambda s: len(s), reverse=True)
random.shuffle(process_list)
print(process_list)
# with open('/data/fangpengcheng/projects/vits/biaozhun_final.txt', 'w') as f:
#     for index, item in enumerate(process_list):
#         f.writelines(item + '\n')
with open('/data/fangpengcheng/projects/vits/biaozhun_final.txt', 'w') as f:
    for index, item in enumerate(process_list):
        k = str(index + 1)
        other_url = k.zfill(6)
        try:
            item = item.split(' ')[1]
        except:
            continue
        f.writelines(other_url + ' ' + item + '\n')
