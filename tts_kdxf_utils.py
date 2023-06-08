import struct

import websocket
import datetime
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import time
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import _thread as thread
import os
from pydub import AudioSegment
from swagger_server.dev_utils.tts_path_utils import VOICE_PATH

STATUS_FIRST_FRAME = 0  # 第一帧的标识
STATUS_CONTINUE_FRAME = 1  # 中间帧标识
STATUS_LAST_FRAME = 2  # 最后一帧的标识


def pcm_to_amr(pcm_file, amr_file):
    # 读取PCM文件并转换为AudioSegment对象
    sound = AudioSegment.from_file(pcm_file, format='s16le', channels=1, frame_rate=16000, sample_width=2)
    # 导出AudioSegment对象为AMR格式
    sound.export(amr_file, format='amr', parameters=['-ar', '8000'])
    return len(sound)


class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, Text):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.Text = Text

        # 公共参数(common)
        self.CommonArgs = {"app_id": self.APPID}
        # 业务参数(business)，更多个性化参数可在官网查看
        self.BusinessArgs = {"aue": "raw", "auf": "audio/L16;rate=16000", "vcn": "xiaoyan", "tte": "utf8"}
        self.Data = {"status": 2, "text": str(base64.b64encode(self.Text.encode("utf-8")), "UTF8")}
        #使用小语种须使用以下方式，此处的unicode指的是 utf16小端的编码方式，即"UTF-16LE"”
        #self.Data = {"status": 2, "text": str(base64.b64encode(self.Text.encode("utf-16")), "UTF8")}

    # 生成url
    def create_url(self):
        url = "wss://tts-api.xfyun.cn/v2/tts"
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/tts " + "HTTP/1.1"
        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode("utf-8"), signature_origin.encode("utf-8"),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding="utf-8")

        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode(encoding="utf-8")
        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }
        # 拼接鉴权参数，生成url
        url = url + "?" + urlencode(v)
        return url


class KDXF_TTS():
    def __init__(self, APPID, APIKey, APISecret, sendfrom="wechat", userid="xxxx",
                 Text="大家好我是国星宇航算法能力中心的小方和小陈欢迎您加入国星宇航大家庭。Perfect work."):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.Text = Text
        self.wsParam = Ws_Param(APPID="e9b00fb8", APISecret="YTI2MTY3Yzc3NGFmMzBjZmU2NDYwMzEz",
                           APIKey="9d4f57b935c5a99253d99cf55eec699d",
                           Text=Text)
        self.path_pcm = os.path.join(VOICE_PATH, sendfrom + "_" + userid + "_" +str(int(time.time() * 10000)) + ".pcm")

    def on_message(self, ws, message):
        try:
            message =json.loads(message)
            code = message["code"]
            sid = message["sid"]
            audio = message["data"]["audio"]
            audio = base64.b64decode(audio)
            status = message["data"]["status"]

            if status == 2:
                print("ws is closed")
                ws.close()
            if code != 0:
                errMsg = message["message"]
                print("sid:%s call error:%s code is:%s" % (sid, errMsg, code))
            else:
                with open(self.path_pcm, "ab") as f:
                    f.write(audio)
        except Exception as e:
            print("receive msg,but parse exception:", e)

    # 收到websocket错误的处理
    def on_error(self, ws, error):
        print("### error:", error)

    # 收到websocket关闭的处理
    def on_close(self, ws):
        print("### closed ###")


    # 收到websocket连接建立的处理
    def on_open(self,ws):
        def run(*args):
            d = {"common": self.wsParam.CommonArgs,
                 "business": self.wsParam.BusinessArgs,
                 "data": self.wsParam.Data,
                 }
            d = json.dumps(d)
            ws.send(d)

        thread.start_new_thread(run, ())

    def run_tts(self):
        websocket.enableTrace(False)
        wsUrl = self.wsParam.create_url()
        ws = websocket.WebSocketApp(wsUrl, on_message=self.on_message, on_error=self.on_error, on_close=self.on_close)
        ws.on_open = self.on_open
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        return self.path_pcm


def kdxf_voice(text="大家好我是国星宇航算法能力中心的小方和小陈欢迎您加入国星宇航大家庭。Perfect work.",
               sendfrom="wechat", userid="xxxx",):
    wsParam = KDXF_TTS(
        APPID="e9b00fb8",
        APISecret="YTI2MTY3Yzc3NGFmMzBjZmU2NDYwMzEz",
        APIKey="9d4f57b935c5a99253d99cf55eec699d",
        Text=text, sendfrom=sendfrom, userid=userid)
    path_pcm = wsParam.run_tts()
    path_amr = path_pcm[:-4] + ".amr"
    len_sound = pcm_to_amr(path_pcm, path_amr)
    return path_amr, len_sound


if __name__ == "__main__":
    kdxf_voice()