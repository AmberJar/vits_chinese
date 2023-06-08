import os.path

from pydub import AudioSegment
from pydub.silence import split_on_silence

path = '/data/fangpengcheng/data/loud_save/audios/sample.wav'
sound = AudioSegment.from_mp3(path)
loudness = sound.dBFS
# print(loudness)

chunks = split_on_silence(sound,
                          # must be silent for at least half a second,沉默半秒
                          min_silence_len=670,

                          # consider it silent if quieter than -16 dBFS
                          silence_thresh=-45,
                          keep_silence=400

                          )
print('总分段：', len(chunks))

# 放弃长度小于2秒的录音片段
# for i in list(range(len(chunks)))[::-1]:
#     if len(chunks[i]) <= 2000 or len(chunks[i]) >= 10000:
#         chunks.pop(i)
# print('取有效分段(大于2s小于10s)：', len(chunks))

'''
for x in range(0,int(len(sound)/1000)):
    print(x,sound[x*1000:(x+1)*1000].max_dBFS)
'''

save_dir = '/data/fangpengcheng/data/loud_save/senior_man'
for i, chunk in enumerate(chunks):
    number = str(i).zfill(5)
    chunk.export(f"{save_dir}/chunk_{number}.wav", format="wav")
    print(i)