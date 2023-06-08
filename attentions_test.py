import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
from modules import LayerNorm


class Encoder(nn.Module):
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0., window_size=4,
                 **kwargs):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for i in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout,
                                                       window_size=window_size))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout))
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x, x_mask


class Decoder(nn.Module):
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0.,
                 proximal_bias=False, proximal_init=True, **kwargs):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init

        self.drop = nn.Dropout(p_dropout)
        self.self_attn_layers = nn.ModuleList()
        self.norm_layers_0 = nn.ModuleList()
        self.encdec_attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.self_attn_layers.append(
                MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout,
                                   proximal_bias=proximal_bias, proximal_init=proximal_init))
            self.norm_layers_0.append(LayerNorm(hidden_channels))
            self.encdec_attn_layers.append(
                MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout, causal=True))
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask, h, h_mask):
        """
        x: decoder input
        h: encoder output
        """
        self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(device=x.device, dtype=x.dtype)
        encdec_attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.self_attn_layers[i](x, x, self_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_0[i](x + y)

            y = self.encdec_attn_layers[i](x, h, encdec_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


# @torch.jit.script_method
class MultiHeadAttention(nn.Module):
    def __init__(self, channels, out_channels, n_heads, p_dropout=0., window_size=None, heads_share=True,
                 block_length=None, proximal_bias=False, proximal_init=False):
        super().__init__()
        # assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels ** -0.5
            self.emb_rel_k = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
            self.emb_rel_v = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x, c, attn_mask):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k, v, self.n_heads, self.k_channels, self.window_size,
                                      self.emb_rel_k, self.emb_rel_v, self.proximal_bias,
                                      self.p_dropout, attn_mask, 0)

        x = self.conv_o(x)
        return x

    # @torch.jit.script
    def attention(self, query, key, value, n_heads: int, k_channels: int, window_size: int,
                  emb_rel_k, emb_rel_v, proximal_bias: bool,
                  p_dropout: float, mask: torch.Tensor=None, zsbd:int=0):

        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s = key.size()
        t_t = query.size(2)
        query = query.view(b, n_heads, k_channels, t_t).transpose(2, 3)
        key = key.view(b, n_heads, k_channels, t_s).transpose(2, 3)
        value = value.view(b, n_heads, k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(k_channels), key.transpose(-2, -1))
        if window_size is not None:
            # assert t_s == t_t, "Relative attention is only available for self-attention."
            relative_embeddings, length, window_size = emb_rel_k, t_s, window_size

            "key_relative_embeddings = self._get_relative_embeddings(emb_rel_k, t_s, window_size)"
            max_relative_position = 2 * window_size + 1
            # Pad first before slice to avoid using cond ops.
            pad_length = max(length - (window_size + 1), 0)
            slice_start_position = max((window_size + 1) - length, 0)
            slice_end_position = slice_start_position + 2 * length - 1
            if pad_length > 0:
                # pad_list = commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]])
                # pad_list = [[0, 0], [pad_length, pad_length], [0, 0]]
                res_list = [0, 0, pad_length, pad_length, 0, 0]
                padded_relative_embeddings = F.pad(
                    relative_embeddings,
                    res_list)
            else:
                padded_relative_embeddings = relative_embeddings
            used_relative_embeddings = padded_relative_embeddings[:, slice_start_position:slice_end_position]

            key_relative_embeddings = used_relative_embeddings
            ###
            "self._matmul_with_relative_keys(query / math.sqrt(k_channels), key_relative_embeddings)"
            rel_logits = torch.matmul(query / math.sqrt(k_channels),
                                      key_relative_embeddings.unsqueeze(0).transpose(-2, -1))

            "scores_local = self._relative_position_to_absolute_position(rel_logits)"
            x = rel_logits
            batch, heads, length, _ = x.size()
            # Concat columns of pad to shift from relative to absolute indexing.
            # pad_list = [[0, 0], [0, 0], [0, 0], [0, 1]]
            pad_list = [0, 1, 0, 0, 0, 0, 0, 0]
            # x = F.pad(x, commons.convert_pad_shape(pad_list))
            x = F.pad(x, pad_list)

            # Concat extra elements so to add up to shape (len+1, 2*len-1).
            x_flat = x.view([batch, heads, length * 2 * length])
            pad_list = [0, length - 1, 0, 0, 0, 0]
            # x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))
            x_flat = F.pad(x_flat, pad_list)

            # Reshape and slice out the padded elements.
            x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1:]
            scores_local = x_final

            scores = scores + scores_local
        if proximal_bias:
            # assert t_s == t_t, "Proximal bias is only available for self-attention."
            "scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)"

            r = torch.arange(t_s, dtype=torch.float32)
            diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
            res = torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)
            res = res.to(device=scores.device, dtype=scores.dtype)

            scores = scores + res
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            # if block_length is not None:
            #     # assert t_s == t_t, "Local attention is only available for self-attention."
            #     block_mask = torch.ones_like(scores).triu(-block_length).tril(block_length)
            #     scores = scores.masked_fill(block_mask == 0, -1e4)
        p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        p_attn = F.dropout(p_attn, p_dropout, False, True)
        output = torch.matmul(p_attn, value)
        if window_size is not None:
            "relative_weights = self._absolute_position_to_relative_position(p_attn)"
            x = p_attn

            batch, heads, length, _ = x.size()
            batch, heads, length = int(batch), int(heads), int(length)
            input_list = [batch, heads, int(length ** 2 + length * (length - 1))]
            # padd along column
            pad_list = [0, length - 1, 0, 0, 0, 0, 0, 0]
            x = F.pad(x, pad_list)
            x_flat = x.view(input_list)
            # add 0's in the beginning that will skew the elements after reshape
            pad_list = [length, 0, 0, 0, 0, 0]
            x_flat = F.pad(x_flat, pad_list)
            x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
            relative_weights = x_final

            "value_relative_embeddings = self._get_relative_embeddings(emb_rel_v, t_s, window_size)"
            relative_embeddings, length, window_size = emb_rel_v, t_s, window_size
            max_relative_position = 2 * window_size + 1
            # Pad first before slice to avoid using cond ops.
            pad_length = max(length - (window_size + 1), 0)
            slice_start_position = max((window_size + 1) - length, 0)
            slice_end_position = slice_start_position + 2 * length - 1
            if pad_length > 0:
                pad_list = [0, 0, pad_length, pad_length, 0, 0]
                # pad_list = commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]])
                padded_relative_embeddings = F.pad(relative_embeddings, pad_list)
            else:
                padded_relative_embeddings = relative_embeddings
            used_relative_embeddings = padded_relative_embeddings[:, slice_start_position:slice_end_position]
            value_relative_embeddings = used_relative_embeddings

            "output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)"
            x, y = relative_weights, value_relative_embeddings
            ret = torch.matmul(x, y.unsqueeze(0))
            output = output + ret

        output = output.transpose(2, 3).contiguous().view(b, d, t_t)  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        """
        x: [b, h, l, d]
        y: [h or 1, m, d]
        ret: [b, h, l, m]
        """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length, window_size):
        max_relative_position = 2 * window_size + 1
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (window_size + 1), 0)
        slice_start_position = max((window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            pad_list = commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]])
            res_list = []
            for item in pad_list:
                item = item.item()
                res_list.append(item)
            padded_relative_embeddings = F.pad(
                relative_embeddings,
                res_list)
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[:, slice_start_position:slice_end_position]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        batch, heads, length, _ = x.size()
        # Concat columns of pad to shift from relative to absolute indexing.
        x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))

        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1:]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.size()
        # padd along column
        x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = x.view([batch, heads, length ** 2 + length * (length - 1)])
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length):
        """Bias for self-attention to encourage attention to close positions.
        Args:
          length: an integer scalar.
        Returns:
          a Tensor with shape [1, 1, length, length]
        """
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0., activation=None,
                 causal=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal

        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding(x * x_mask))
        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x

    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x
