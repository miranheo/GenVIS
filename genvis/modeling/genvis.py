from math import ceil

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from vita.modeling.transformer_decoder.vita import VITA

class GenVIS(VITA):
    def __init__(self, cfg):
        super().__init__(
            cfg=cfg,
            in_channels=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            aux_loss=cfg.MODEL.VITA.DEEP_SUPERVISION,
        )

        self.num_frames = cfg.MODEL.GENVIS.LEN_CLIP_WINDOW
        hidden_dim = cfg.MODEL.VITA.HIDDEN_DIM

        self.pre_memory_embed_k = nn.Linear(hidden_dim, hidden_dim)
        self.pre_memory_embed_v = nn.Linear(hidden_dim, hidden_dim)

        self.pre_query_embed_k = nn.Linear(hidden_dim, hidden_dim)
        self.pre_query_embed_v = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, frame_query, pre_memory, output):
        """
        L: Number of Layers.
        B: Batch size.
        T: Temporal window size. Number of frames per video.
        C: Channel size.
        fQ: Number of frame-wise queries from Mask2Former.
        cQ: Number of clip-wise queries to decode Q.
        """
        if not self.training:
            frame_query = frame_query[[-1]]

        pre_memory_k  = pre_memory["k"]
        pre_memory_v  = pre_memory["v"]

        L, BT, fQ, C = frame_query.shape
        B = BT // self.num_frames if self.training else 1
        T = self.num_frames if self.training else BT // B

        frame_query = frame_query.reshape(L*B, T, fQ, C)
        frame_query = frame_query.permute(1, 2, 0, 3).contiguous()
        frame_query = self.input_proj_dec(frame_query) # T, fQ, LB, C

        if self.window_size > 0:
            pad = int(ceil(T / self.window_size)) * self.window_size - T
            _T = pad + T
            frame_query = F.pad(frame_query, (0,0,0,0,0,0,0,pad))   # _T, fQ, LB, C
            enc_mask = frame_query.new_ones(L*B, _T).bool()         # LB, _T
            enc_mask[:, :T] = False
        else:
            enc_mask = None

        frame_query = self.encode_frame_query(frame_query, enc_mask)
        frame_query = frame_query[:T].flatten(0,1)              # TfQ, LB, C

        if self.use_sim:
            pred_fq_embed = self.sim_embed_frame(frame_query)   # TfQ, LB, C
            pred_fq_embed = pred_fq_embed.transpose(0, 1).reshape(L, B, T, fQ, C)
        else:
            pred_fq_embed = None

        src = self.src_embed(frame_query)   # TfQ, LB, C
        dec_pos = self.fq_pos.weight[None, :, None, :].repeat(T, 1, L*B, 1).flatten(0, 1) # TfQ, LB, C

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, L*B, 1) # cQ, LB, C

        cQ, LB, C = output.shape

        # pre query embed
        pre_query_k = self.pre_query_embed_k(output) # cQ, LB, C
        pre_query_v = self.pre_query_embed_v(output) # cQ, LB, C

        # pre memory read
        if pre_memory_k and pre_memory_v:
            pre_memory_k = torch.cat(pre_memory_k).flatten(1,2) # M, LB, cQ, C
            pre_memory_v = torch.cat(pre_memory_v).flatten(1,2) # M, LB, cQ, C
        else:
            pre_memory_k = torch.empty((0, LB, cQ, C), device=output.device)
            pre_memory_v = torch.empty((0, LB, cQ, C), device=output.device)

        qk_mk = torch.einsum("qbc, mbpc -> bqmp", pre_query_k, pre_memory_k) # LB, cQ, M, cQ
        qk_mk = torch.einsum("bqmq -> bqm", qk_mk) # LB, cQ, M
        qk_mk = F.softmax(qk_mk, dim=2) 
        qk_mk_mv = torch.einsum("bqm, mbqc-> qbc", qk_mk, pre_memory_v) # cQ, B, C

        pre_query_v = pre_query_v + qk_mk_mv # cQ, LB, C
        output = output + pre_query_v        # cQ, LB, C

        decoder_outputs = []
        for i in range(self.num_layers):
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src,
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=dec_pos, query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            if (self.training and self.aux_loss) or (i == self.num_layers - 1):
                dec_out = self.decoder_norm(output) # cQ, LB, C
                dec_out = dec_out.transpose(0, 1)   # LB, cQ, C
                decoder_outputs.append(dec_out.view(L, B, self.num_queries, C))

        decoder_outputs = torch.stack(decoder_outputs, dim=0)   # D, L, B, cQ, C

        pred_cls = self.class_embed(decoder_outputs)
        pred_mask_embed = self.mask_embed(decoder_outputs)
        if self.use_sim and self.sim_use_clip:
            pred_cq_embed = self.sim_embed_clip(decoder_outputs)
        else:
            pred_cq_embed = [None] * self.num_layers

        memory_input = decoder_outputs[-1]

        pre_memory_k = self.pre_memory_embed_k(memory_input)[None] # 1, L, B, cQ, C
        pre_memory_v = self.pre_memory_embed_v(memory_input)[None] # 1, L, B, cQ, C

        out = {
            'pred_logits': pred_cls[-1],
            'pred_mask_embed': pred_mask_embed[-1],
            'pred_fq_embed': pred_fq_embed,
            'pred_cq_embed': pred_cq_embed[-1],
            'aux_outputs': self._set_aux_loss(
                pred_cls, pred_mask_embed, pred_cq_embed, pred_fq_embed
            ),
            'pre_memory': {"k": pre_memory_k, "v": pre_memory_v},
        }

        return out, output
