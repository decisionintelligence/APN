# spectron/model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import repeat, rearrange

from utils.globals import logger
from utils.ExpConfigs import ExpConfigs


# --- 辅助模块 (基本不变) ---
class ContinuousTimeEncode(nn.Module):
    """连续时间编码 (正弦位置编码的直接计算版本)"""

    def __init__(self, d_model, max_timescale=10000.0):
        super(ContinuousTimeEncode, self).__init__()
        self.d_model = d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(max_timescale) / d_model))
        self.register_buffer('div_term', div_term.view(1, 1, -1))

    def forward(self, t):
        """
        Args:
            t (Tensor): 形状为 (..., 1) 的时间戳张量, 值应在 [0, 1] 区间。
        """
        position = t * 5000.0
        pe_sin = torch.sin(position * self.div_term)
        pe_cos = torch.cos(position * self.div_term)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)
        if self.d_model % 2 != 0:
            pe = F.pad(pe, (0, 1))
        return pe


def _build_mlp(dims, activation=nn.ReLU):
    """构建一个MLP的辅助函数"""
    layers = []
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


# --- 新增核心组件: 对应方法论 3.1.1 ---
class Patching(nn.Module):
    """
    不规则时间序列分块模块
    - 沿时间维度 (L) 将序列切分为重叠的块。
    """

    def __init__(self, patch_len, stride):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, v, t, mask):
        """
        Args:
            v (Tensor): 值, shape (B_channel, L, 1)
            t (Tensor): 时间戳, shape (B_channel, L, 1)
            mask (Tensor): 掩码, shape (B_channel, L, 1)
        Returns:
            v_patch (Tensor): 分块后的值, shape (B_channel * num_patches, P, 1)
            t_patch (Tensor): 分块后的时间戳, shape (B_channel * num_patches, P, 1)
            mask_patch (Tensor): 分块后的掩码, shape (B_channel * num_patches, P, 1)
            num_patches (int): 每个序列生成的块数量
        """
        B_c, L, _ = v.shape

        # 使用 unfold 实现高效的滑动窗口分块
        # (B_c, 1, L) -> (B_c, 1, num_patches, patch_len)
        v_unfolded = v.permute(0, 2, 1).unfold(dimension=2, size=self.patch_len, step=self.stride)
        t_unfolded = t.permute(0, 2, 1).unfold(dimension=2, size=self.patch_len, step=self.stride)
        mask_unfolded = mask.permute(0, 2, 1).unfold(dimension=2, size=self.patch_len, step=self.stride)

        num_patches = v_unfolded.shape[2]

        v_patch = v_unfolded.permute(0, 2, 3, 1)
        t_patch = t_unfolded.permute(0, 2, 3, 1)
        mask_patch = mask_unfolded.permute(0, 2, 3, 1)

        v_patch = rearrange(v_patch, 'b m p d -> (b m) p d', m=num_patches)
        t_patch = rearrange(t_patch, 'b m p d -> (b m) p d', m=num_patches)
        mask_patch = rearrange(mask_patch, 'b m p d -> (b m) p d', m=num_patches)

        t_start = t_patch[:, :1, :]  # Shape: (B*C*M, 1, 1)
        t_patch_normalized = t_patch - t_start

        return v_patch, t_patch_normalized, mask_patch, num_patches  # 返回归一化后的时间


# --- 核心组件: 对应方法论 3.1 (Patched ANST) ---
class PatchedANST(nn.Module):
    """
    局部分块化的自适应神经谱变换 (Patched ANST) - 【速度与内存均衡最终版】
    - 通过分块处理核 (Kernel Chunking) 来平衡计算速度和内存占用。
    """

    def __init__(self, d_model, n_heads, num_kernels, d_max, patch_len, kernel_chunk_size=16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_kernels = num_kernels
        self.d_max = d_max
        self.patch_len = patch_len
        self.d_head = d_model // n_heads
        self.kernel_chunk_size = kernel_chunk_size

        self.time_encoder = ContinuousTimeEncode(d_model)
        self.mlp_enc = _build_mlp([1 + d_model, d_model, d_model])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.mlp_basis = _build_mlp([d_model, d_model * 2, num_kernels * 3])

        self.W_trap = nn.Linear(1, self.d_model, bias=False)  # 处理单个积分值
        self.mlp_val = _build_mlp([1 + d_model, d_model, d_model])
        self.mlp_key = _build_mlp([d_model, d_model, d_model])
        self.mlp_query = _build_mlp([1 + d_model, d_model, d_model])
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, v_patch, t_patch, mask_patch):
        B_patch, P, _ = v_patch.shape

        t_emb = self.time_encoder(t_patch)
        v_masked = v_patch * mask_patch
        t_emb_masked = t_emb * mask_patch
        e = self.mlp_enc(torch.cat([v_masked, t_emb_masked], dim=-1))
        cls_tokens = self.cls_token.expand(B_patch, -1, -1)
        e_in = torch.cat((cls_tokens, e), dim=1)
        src_mask = torch.cat([torch.ones(B_patch, 1, device=v_patch.device), mask_patch.squeeze(-1)], dim=1)
        src_mask_bool = (src_mask == 0)
        h_out = self.transformer_encoder(e_in, src_key_padding_mask=src_mask_bool)
        z = h_out[:, 0, :]
        raw_params = self.mlp_basis(z).view(B_patch, self.num_kernels, 3)
        omega, phi, d = raw_params.split(1, dim=-1)
        basis_params = {'omega': F.softplus(omega), 'phi': math.pi * torch.tanh(phi), 'd': self.d_max * torch.tanh(d)}

        V = self.mlp_val(torch.cat([v_masked, t_emb_masked], dim=-1))
        K_mat = self.mlp_key(e)
        V_multihead = rearrange(V, 'b p (h d) -> b h p d', h=self.n_heads)
        K_multihead = rearrange(K_mat, 'b p (h d) -> b h p d', h=self.n_heads)

        all_coeffs = []
        for k_start in range(0, self.num_kernels, self.kernel_chunk_size):
            k_end = min(k_start + self.kernel_chunk_size, self.num_kernels)
            chunk_size = k_end - k_start

            omega_chunk = basis_params['omega'][:, k_start:k_end, :]
            phi_chunk = basis_params['phi'][:, k_start:k_end, :]
            d_chunk = basis_params['d'][:, k_start:k_end, :]

            f_k_t_chunk = torch.exp(-d_chunk.transpose(1, 2) * t_patch) * \
                          torch.cos(2 * math.pi * omega_chunk.transpose(1, 2) * t_patch + phi_chunk.transpose(1, 2))

            integrand = v_masked * f_k_t_chunk
            integrand_sum = (integrand[:, :-1, :] + integrand[:, 1:, :]) / 2.0
            dt = (t_patch[:, 1:] - t_patch[:, :-1])
            trapz_integral = torch.sum(integrand_sum * dt * mask_patch[:, :-1], dim=1)
            c_trap_chunk = self.W_trap(trapz_integral.unsqueeze(-1)).view(B_patch, chunk_size, self.d_model)

            # t_emb_masked shape: (B_patch, P, d_model)
            # f_k_t_chunk shape: (B_patch, P, chunk_size)

            # Expand time embedding to match the kernel chunk dimension
            # (B_patch, P, d_model) -> (B_patch, P, 1, d_model) -> (B_patch, P, chunk_size, d_model)
            t_emb_expanded = t_emb_masked.unsqueeze(2).expand(-1, -1, chunk_size, -1)

            # Unsqueeze f_k_t_chunk for concatenation
            # (B_patch, P, chunk_size) -> (B_patch, P, chunk_size, 1)
            f_k_t_unsqueezed = f_k_t_chunk.unsqueeze(-1)

            # Now both tensors have shape (B_patch, P, chunk_size, ...), concatenation is correct
            q_input_raw = torch.cat([f_k_t_unsqueezed, t_emb_expanded], dim=-1)

            # Flatten for MLP, then reshape back. This is more robust.
            # (B_patch, P, chunk_size, 1 + d_model) -> (B_patch * P * chunk_size, 1 + d_model)
            Q_mat_flat = self.mlp_query(q_input_raw.flatten(0, 2))
            # (B_patch * P * chunk_size, d_model) -> (B_patch, P, chunk_size, d_model)
            Q_mat_chunk = Q_mat_flat.view(B_patch, P, chunk_size, self.d_model)
            Q_multihead_chunk = rearrange(Q_mat_chunk, 'b p k (h d) -> b h k p d', h=self.n_heads)

            pointwise_scores = torch.einsum('bhkpd,bhpd->bhkp', Q_multihead_chunk, K_multihead) / (self.d_head ** 0.5)
            attn_mask = mask_patch.squeeze(-1).view(B_patch, 1, 1, P)
            pointwise_scores.masked_fill_(attn_mask == 0, -1e9)
            attn_weights = F.softmax(pointwise_scores, dim=-1)
            attended_values = torch.einsum('bhkp,bhpd->bhkd', attn_weights, V_multihead)

            c_learned_concat = rearrange(attended_values, 'b h k d -> b k (h d)')
            c_learned_chunk = self.W_O(c_learned_concat)

            chunk_coeffs = c_trap_chunk + c_learned_chunk
            all_coeffs.append(chunk_coeffs)

        spectral_coeffs = torch.cat(all_coeffs, dim=1)
        return spectral_coeffs, basis_params


# --- 核心组件: 对应方法论 3.2 ---
class HarmonicAttentionLayer(nn.Module):
    """带相对频率编码的自注意力层 (无变化, 可复用)"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.mlp_rel = _build_mlp([2, 64, n_heads])

    def forward(self, x, freqs, mask=None):
        # x: (B, SeqLen, D_model), SeqLen 现在是 M*K
        # freqs: (B, SeqLen, 1)
        B, SeqLen, _ = x.shape
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        Q = rearrange(Q, 'b s (h d) -> b h s d', h=self.n_heads)
        K = rearrange(K, 'b s (h d) -> b h s d', h=self.n_heads)
        V = rearrange(V, 'b s (h d) -> b h s d', h=self.n_heads)
        scores = torch.einsum('bhid, bhjd -> bhij', Q, K) / (self.d_head ** 0.5)

        freq_diff = freqs.unsqueeze(2) - freqs.unsqueeze(1)
        freq_ratio = torch.log((freqs.unsqueeze(2) + 1e-6) / (freqs.unsqueeze(1) + 1e-6))
        rel_freq_feat = torch.cat([freq_diff, freq_ratio], dim=-1)

        b_jk = self.mlp_rel(rel_freq_feat)
        b_jk = b_jk.permute(0, 3, 1, 2)
        scores = scores + b_jk

        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.einsum('bhij, bhjd -> bhid', attn, V)
        context = rearrange(context, 'b h s d -> b s (h d)')
        return self.out(context)


class PatchSpectralTransformer(nn.Module):
    """
    深度谱交互Transformer (修改版)
    - 在 patch 和 kernel 展平后的序列上操作，学习时频谱关系。
    - 依然采用通道独立策略。
    """

    def __init__(self, d_model, n_heads, num_layers_intra, dropout=0.1):
        super().__init__()
        self.intra_channel_layers = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(d_model),
                'attn': HarmonicAttentionLayer(d_model, n_heads, dropout),
                'norm2': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4), nn.ReLU(),
                    nn.Dropout(dropout), nn.Linear(d_model * 4, d_model)
                ),
                'dropout': nn.Dropout(dropout)
            }) for _ in range(num_layers_intra)
        ])

    def forward(self, spectral_coeffs_seq, basis_params_seq):
        """
        Args:
            spectral_coeffs_seq (Tensor): (B_c, M, K, D)
            basis_params_seq (dict): omega (B_c, M, K, 1)
        Returns:
            H_seq (Tensor): (B_c, M, K, D)
        """
        # --- 3.2.1 块间谐波注意力 ---
        # 展平 patch 和 kernel 维度，形成一个长序列
        x = rearrange(spectral_coeffs_seq, 'b m k d -> b (m k) d')
        freqs = rearrange(basis_params_seq['omega'], 'b m k d -> b (m k) d')

        for layer in self.intra_channel_layers:
            x_norm = layer['norm1'](x)
            attn_out = layer['attn'](x_norm, freqs)
            x = x + layer['dropout'](attn_out)
            x_norm = layer['norm2'](x)
            ffn_out = layer['ffn'](x_norm)
            x = x + layer['dropout'](ffn_out)

        # 将处理后的序列重塑回结构化形状
        H_seq = rearrange(x, 'b (m k) d -> b m k d', k=spectral_coeffs_seq.shape[2])
        return H_seq


# --- 主模型: SPECTRON (Patched Version) ---
class Model(nn.Module):
    def __init__(self, configs: ExpConfigs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len_max_irr or configs.pred_len

        # --- SPECTRON Hyperparameters from configs ---
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.num_kernels = configs.spectron_num_kernels
        self.d_max = configs.spectron_d_max
        self.num_intra_layers = configs.spectron_num_intra_layers
        self.dropout = configs.dropout
        # **新增**：从配置中读取分块参数
        self.patch_len = configs.spectron_patch_len
        self.patch_stride = configs.spectron_patch_stride
        self.kernel_chunk_size = configs.spectron_kernel_chunk_size
        self.num_last_patches = configs.spectron_num_last_patches
        # --- 模块实例化 (采用新模块) ---
        # 3.1.1 局部分块
        self.patching = Patching(self.patch_len, self.patch_stride)

        # 3.1 局部分块化的自适应神经谱变换
        self.anst = PatchedANST(
            self.d_model, self.n_heads, self.num_kernels, self.d_max, self.patch_len,
            kernel_chunk_size=self.kernel_chunk_size
        )

        # 3.2 深度谱交互Transformer
        self.spectral_transformer = PatchSpectralTransformer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_layers_intra=self.num_intra_layers,
            dropout=self.dropout
        )

        # 3.3 谱综合与预测
        self.time_encoder_pred = ContinuousTimeEncode(self.d_model)
        self.synthesis_linear = nn.Linear(self.d_model, 1)
        self.mlp_out = _build_mlp(
            [1 + self.d_model + self.d_model, self.d_model, self.d_model, 1]
        )

    def forward(
            self,
            x: Tensor,
            x_mark: Tensor,
            x_mask: Tensor,
            y: Tensor = None,
            y_mark: Tensor = None,
            y_mask: Tensor = None,
            **kwargs
    ):
        if x_mark.shape[-1] != x.shape[-1]:
            x_mark = repeat(x_mark[:, :, 0], "b l -> b l c", c=x.shape[-1])
        B, L, C = x.shape

        # --- (B, L, C) -> (B*C, L, 1) ---
        v_in = rearrange(x, 'b l c -> (b c) l 1')
        t_in = rearrange(x_mark, 'b l c -> (b c) l 1')
        mask_in = rearrange(x_mask, 'b l c -> (b c) l 1')

        # --- 3.1.1 局部分块 (Patching) ---
        # (B*C, L, 1) -> (B*C*M, P, 1)
        v_patch, t_patch, mask_patch, num_patches = self.patching(v_in, t_in, mask_in)

        # --- 3.1.2 - 3.1.4 Patched ANST ---
        # (B*C*M, P, 1) -> (B*C*M, K, D)
        spectral_coeffs_patches, basis_params_patches = self.anst(v_patch, t_patch, mask_patch)

        # --- 恢复序列结构，为谱 Transformer 做准备 ---
        # (B*C*M, K, D) -> (B*C, M, K, D)
        spectral_coeffs_seq = rearrange(spectral_coeffs_patches, '(bc m) k d -> bc m k d', m=num_patches)
        basis_params_seq = {}
        for key in basis_params_patches:
            basis_params_seq[key] = rearrange(basis_params_patches[key], '(bc m) k d -> bc m k d', m=num_patches)

        # --- 3.2 深度谱交互 ---
        # (B*C, M, K, D) -> (B*C, M, K, D)
        final_coeffs_seq = self.spectral_transformer(spectral_coeffs_seq, basis_params_seq)

        # --- 3.3 谱综合与预测 ---
        if self.task_name in ["short_term_forecast", "long_term_forecast", "imputation"]:
            H = self.pred_len
            t_query = y_mark
            if t_query.shape[-1] != C:
                t_query = repeat(t_query[:, :, 0], "b l -> b l c", c=C)
            t_query_flat = rearrange(t_query, 'b h c -> (b c) h 1')

            # 1. 确定实际要使用的patch数量 (处理序列过短的情况)
            actual_num_last = min(self.num_last_patches, num_patches)

            # 2. 提取最后N个patch的谱系数和基函数参数
            # final_coeffs_seq shape: (B*C, M, K, D)
            last_n_coeffs = final_coeffs_seq[:, -actual_num_last:, :, :]  # (B*C, N, K, D)

            # 3. 创建权重：越新的patch权重越高 (线性衰减 + softmax)
            # 权重形状需要能广播: (1, N, 1, 1)
            weights = torch.linspace(0.1, 1.0, actual_num_last, device=x.device)
            weights = F.softmax(weights, dim=0).view(1, -actual_num_last, 1, 1)

            # 4. 对谱系数进行加权平均，得到一个鲁棒的“等效最新”谱表示
            # (B*C, N, K, D) * (1, N, 1, 1) -> sum over N -> (B*C, K, D)
            H_robust_patch = torch.sum(last_n_coeffs * weights, dim=1)

            # 5. 对基函数参数也进行加权平均
            basis_params_robust_patch = {}
            for key in basis_params_seq:
                # basis_params_seq[key] shape: (B*C, M, K, 1)
                last_n_params = basis_params_seq[key][:, -actual_num_last:, :, :]  # (B*C, N, K, 1)
                # (B*C, N, K, 1) * (1, N, 1, 1) -> sum over N -> (B*C, K, 1)
                robust_param = torch.sum(last_n_params * weights, dim=1)
                basis_params_robust_patch[key] = robust_param

            # 6. 定义参考时间点：最简单、最鲁棒的方式是直接取输入序列的最后一个有效时间点
            t_end_of_context = t_in[:, -1, :].unsqueeze(1)  # (B*C, 1, 1)

            # 7. 计算相对查询时间 (与之前逻辑相同)
            t_query_relative = t_query_flat - t_end_of_context

            # 8. 核心信号合成 (使用加权平均后的鲁棒参数)
            amplitudes = self.synthesis_linear(H_robust_patch)  # (B*C, K, 1)
            omega = basis_params_robust_patch['omega'].unsqueeze(2)  # (B*C, K, 1, 1)
            phi = basis_params_robust_patch['phi'].unsqueeze(2)
            d = basis_params_robust_patch['d'].unsqueeze(2)
            tq_rel = t_query_relative.unsqueeze(1)  # (B*C, 1, H, 1)

            f_k_q = torch.exp(-d * tq_rel) * torch.cos(2 * math.pi * omega * tq_rel + phi)
            s_j = torch.sum(amplitudes.unsqueeze(2) * f_k_q, dim=1)

            # 9. 最终预测 (与之前逻辑相同，但输入已更新)
            t_query_emb = self.time_encoder_pred(t_query_flat)
            h_bar_flat = rearrange(final_coeffs_seq, 'bc m k d -> bc (m k) d')
            h_bar = torch.mean(h_bar_flat, dim=1, keepdim=True).expand(-1, H, -1)

            mlp_input = torch.cat([s_j, t_query_emb, h_bar], dim=-1)
            v_hat = self.mlp_out(mlp_input)

            # --- 新逻辑结束 ---

            outputs = rearrange(v_hat, '(b c) h 1 -> b h c', b=B, c=C)

            f_dim = -1 if self.configs.features == 'MS' else 0
            return {
                "pred": outputs[:, -self.pred_len:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:]
            }
        else:
            raise NotImplementedError(f"Task name {self.task_name} not implemented for SPECTRON")