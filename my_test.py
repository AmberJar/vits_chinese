import torch.nn as nn
import torch
from torch.nn import functional as F
import math
from attentions_test import Encoder
import random
import commons
import attentions_test
import utils
from text.symbols import symbols


class TextEncoder(nn.Module):
    def __init__(self,
                 n_vocab,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        # self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):

        # x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        # x = torch.transpose(x, 1, -1)  # [b, h, t]
        # x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        #
        # x = self.encoder(x * x_mask, x_mask)
        # stats = self.proj(x) * x_mask
        #
        # m, logs = torch.split(stats, self.out_channels, dim=1)
        # return x, m, logs, x_mask
        return x, x_lengths

class MyModel(nn.Module):
    def __init__(self,
                 n_vocab,
                 spec_channels,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 n_speakers=0,
                 gin_channels=0,
                 rank=0,
                 use_sdp=True,
                 **kwargs):

        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels

        self.use_sdp = use_sdp

        self.enc_p = TextEncoder(n_vocab,
                                 inter_channels,
                                 hidden_channels,
                                 filter_channels,
                                 n_heads,
                                 n_layers,
                                 kernel_size,
                                 p_dropout)
        # self.dec = Generator(inter_channels, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
        #                      upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels, rank=rank)

        # self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16,
        #                               gin_channels=gin_channels)
        # self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, n_flows=4,
        #                                   gin_channels=gin_channels)
        #
        # self.dp = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels)
        #
        #
        # if n_speakers > 1:
        #     self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def forward(self, x, x_lst):
        # x, m_p, logs_p, x_mask = self.enc_p(x, x_lst)
        x, m_p = self.enc_p(x, x_lst)

        return x
        # if self.n_speakers > 0:
        #     g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        # else:
        #     g = None
        #
        # if self.use_sdp:
        #     logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        # else:
        #     logw = self.dp(x, x_mask, g=g)
        # w = torch.exp(logw) * x_mask * length_scale
        # w_ceil = torch.ceil(w)
        # y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        # y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        # attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        # attn = commons.generate_path(w_ceil, attn_mask)
        #
        # m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
        # logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1,
        #                                                                          2)  # [b, t', t], [b, t, d] -> [b, d, t']
        #
        # z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        # z = self.flow(z_p, y_mask, g=g, reverse=True)
        # o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        # return o, attn, y_mask, (z, z_p, m_p, logs_p)


hps = utils.get_hparams_from_file("./configs/ziwei.json")
net_g = MyModel(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cpu()
_ = net_g.eval()


with torch.no_grad():
    ######
    print("Load model complete.>>>")
    input_ids = torch.rand(1, 256).cuda()  # dummy data
    x_tst = input_ids.unsqueeze(0).cuda()
    x_tst_lengths = torch.LongTensor([input_ids.size(0)]).cuda()

    traced_script_module = torch.jit.script(net_g, (x_tst, x_tst_lengths))
    # traced_script_module = torch.jit.script(net_g, (x_tst, x_tst_lengths))
    traced_script_module.save(r"/data/fpc/projects/vitsBigGan/pt_file/test_model.pt")
    print("torch.jit save model complete.>>>")

    ######

