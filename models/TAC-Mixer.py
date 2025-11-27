import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat


from utils.globals import logger
from utils.ExpConfigs import ExpConfigs

class SinusoidalPositionalEncoding(nn.Module):
    """
    正弦位置编码模块，用于将连续的时间值编码为高维向量
    """

    def __init__(self, d_model: int, max_timescale: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        inv_freq = 1.0 / (max_timescale ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, time_stamps: torch.Tensor):
        """
        输入 time_stamps: 形状为 (..., 1) 的时间戳张量
        输出: 形状为 (..., d_model) 的位置编码
        """
        encoded = torch.einsum("... , d -> ...d", time_stamps.squeeze(-1), self.inv_freq)
        encoded = torch.cat((encoded.sin(), encoded.cos()), dim=-1)
        return encoded

class TACMixerLayer(nn.Module):
    """
    完全复现论文 3.3.2 和 3.3.3 节描述的 TAC-Mixer 层。
    """

    def __init__(self, patch_num, num_variables, d_model, d_ff, hidden_dim_p, hidden_dim_c, dropout=0.1):
        super().__init__()

        # --- 时间混合模块 ---
        self.norm_temporal_token = nn.LayerNorm(d_model)
        self.temporal_token_mixer = nn.Sequential(
            nn.Linear(patch_num, hidden_dim_p),
            nn.GELU(),
            nn.Linear(hidden_dim_p, patch_num)
        )
        self.norm_temporal_channel = nn.LayerNorm(d_model)
        self.temporal_channel_mixer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # --- 变量混合模块 ---
        self.norm_variable_token = nn.LayerNorm(d_model)
        self.variable_token_mixer = nn.Sequential(
            nn.Linear(num_variables, hidden_dim_c),
            nn.GELU(),
            nn.Linear(hidden_dim_c, num_variables)
        )
        self.norm_variable_channel = nn.LayerNorm(d_model)
        self.variable_channel_mixer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        # x: [B, C, P, D]

        # --- 1. 时间混合 (Temporal Mixing) ---
        # 1.1 词元混合 (Token-Mixing across patches)
        residual = x
        x_norm = self.norm_temporal_token(x)
        x_permuted = x_norm.permute(0, 1, 3, 2)  # -> [B, C, D, P]
        x_mixed = self.temporal_token_mixer(x_permuted)
        x_time_mixed = x_mixed.permute(0, 1, 3, 2)  # -> [B, C, P, D]
        x = x + x_time_mixed

        # 1.2 通道混合 (Channel-Mixing within each patch)
        residual = x
        x_norm = self.norm_temporal_channel(x)
        x_channel_mixed = self.temporal_channel_mixer(x_norm)
        x = x + x_channel_mixed

        # --- 2. 变量混合 (Variable Mixing) ---
        # 2.1 变量混合 (Variable-Mixing across channels/variables)
        residual = x
        x_norm = self.norm_variable_token(x)
        x_permuted = x_norm.permute(0, 2, 3, 1)  # -> [B, P, D, C]
        x_mixed = self.variable_token_mixer(x_permuted)
        x_var_mixed = x_mixed.permute(0, 3, 1, 2)  # -> [B, C, P, D]
        x = x + x_var_mixed

        # 2.2 通道混合 (Channel-Mixing within each patch again)
        residual = x
        x_norm = self.norm_variable_channel(x)
        x_channel_mixed = self.variable_channel_mixer(x_norm)
        x = x + x_channel_mixed

        return x

class Model(nn.Module):
    '''
    TAC-Mixer: 修正后的完美复现版本
    '''

    def __init__(self, configs):  # configs: ExpConfigs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.pred_len = getattr(configs, 'pred_len_max_irr', configs.pred_len)

        self.patch_num = configs.tac_patch_num
        self.context_k = configs.tac_decoder_context_k

        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.n_layers = configs.n_layers
        self.dropout = configs.dropout

        # --- 阶段一：自适应时间聚合模块 (与之前相同，逻辑正确) ---
        self.time_encoder = SinusoidalPositionalEncoding(self.d_model // 2)
        self.point_val_embedding = nn.Linear(1, self.d_model // 2)
        self.point_time_embedding_mlp = nn.Sequential(
            nn.Linear(self.d_model // 2, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, self.d_model // 2)
        )
        self.point_fusion_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model)
        )
        self.patch_queries = nn.Parameter(torch.randn(self.patch_num, self.d_model))
        self.attention_pool = nn.MultiheadAttention(self.d_model, num_heads=configs.n_heads, batch_first=True)
        self.empty_patch_embedding = nn.Parameter(torch.randn(self.d_model))

        # --- 阶段二：双视图时空混合器 (C-Mixer) ---
        # 使用修正后的 TACMixerLayer
        self.mixer_layers = nn.ModuleList([
            TACMixerLayer(
                patch_num=self.patch_num,
                num_variables=self.enc_in,
                d_model=self.d_model,
                d_ff=self.d_ff,
                hidden_dim_p=configs.tac_mixer_hidden_dim_p,
                hidden_dim_c=configs.tac_mixer_hidden_dim_c,
                dropout=configs.dropout
            ) for _ in range(self.n_layers)
        ])
        # 最后增加一个 LayerNorm, 这是 Transformer/Mixer 架构的标准做法
        self.final_norm = nn.LayerNorm(self.d_model)

        # --- 阶段三：查询感知的上下文解码 (逻辑正确，进行向量化优化) ---
        self.decoder_time_encoder = SinusoidalPositionalEncoding(self.d_model // 2)
        self.decoder_attn_mlp = nn.Sequential(
            nn.Linear(self.d_model // 2, self.d_model // 4),
            nn.GELU(),
            nn.Linear(self.d_model // 4, 1)
        )
        self.decoder_output_mlp = nn.Sequential(
            nn.Linear(self.d_model + self.d_model // 2, self.d_ff),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff, 1)
        )

    def forward(
            self,
            x: Tensor,
            x_mark: Tensor,
            x_mask: Tensor,
            y: Tensor,
            y_mark: Tensor,
            y_mask: Tensor,
            **kwargs
    ):
        B, L, C = x.shape
        device = x.device

        # --- 阶段一：自适应时间聚合 (完全向量化版本) ---

        # 1. 点嵌入 (与之前相同，本身就是向量化的)
        valid_points_mask = x_mask.bool()  # [B, L, C]
        point_vals_in = rearrange(x, 'b l c -> (b l c) 1')
        point_times_in = rearrange(repeat(x_mark, 'b l 1 -> b l c', c=C), 'b l c -> (b l c) 1')

        val_embeds = self.point_val_embedding(point_vals_in)
        time_embeds = self.time_encoder(point_times_in)
        time_embeds = self.point_time_embedding_mlp(time_embeds)

        point_embeds = self.point_fusion_layer(torch.cat([val_embeds, time_embeds], dim=-1))
        # -> [(B*L*C), D]

        # 2. 计算每个点所属的补丁索引
        min_time, max_time = x_mark.min(), y_mark.max()
        # 防止 max_time 和 min_time 相等导致除零错误
        if max_time == min_time:
            max_time = min_time + 1
        time_span = max_time - min_time
        patch_width = (time_span + 1e-8) / self.patch_num

        # point_patch_indices: [B, L], 每个点的时间戳对应的补丁 P 的索引
        point_patch_indices = torch.clamp(
            ((x_mark.squeeze(-1) - min_time) / patch_width).floor().long(),
            0, self.patch_num - 1
        )

        # 3. 为所有有效点创建唯一的组ID和在组内的索引
        # 目标: 将所有点按 (b, c, p) 分组，并进行填充(padding)以形成规整的张量

        # 获取所有有效点的嵌入和它们的坐标 (b, c, p)
        valid_mask_flat = valid_points_mask.flatten()  # -> [(B*L*C)]
        valid_point_embeds = point_embeds[valid_mask_flat]  # -> [NumValidPoints, D]

        # 创建 b, c, p 的索引网格
        b_indices_grid = torch.arange(B, device=device).view(B, 1, 1).expand(B, L, C)
        c_indices_grid = torch.arange(C, device=device).view(1, 1, C).expand(B, L, C)
        p_indices_grid = point_patch_indices.unsqueeze(-1).expand(B, L, C)

        # 筛选出有效点对应的坐标
        b_indices_valid = b_indices_grid.flatten()[valid_mask_flat]
        c_indices_valid = c_indices_grid.flatten()[valid_mask_flat]
        p_indices_valid = p_indices_grid.flatten()[valid_mask_flat]

        # 计算每个点的唯一组ID，用于后续的 bincount 统计
        # 每个 (b, c, p) 组合都是一个独立的组
        group_ids = b_indices_valid * (C * self.patch_num) + c_indices_valid * self.patch_num + p_indices_valid

        # 使用 bincount 高效统计每个组有多少个点
        # 总组数 = B * C * P
        group_counts = torch.bincount(group_ids, minlength=B * C * self.patch_num)
        max_points_in_patch = group_counts.max().item()

        if max_points_in_patch == 0:
            # 如果整个批次都没有任何有效的观测点（极端情况），则全部使用空补丁嵌入
            x_mixer_in = self.empty_patch_embedding.repeat(B, C, self.patch_num, 1)
        else:
            # 计算每个点在自己组内的行索引 (用于填充)
            # 这是一个高级技巧: 先排序，然后通过比较相邻元素是否相同来确定新组的开始，再用 cumsum 累加
            sorted_group_ids, sorted_indices = torch.sort(group_ids)
            # is_new_group_start: 标记每个组的第一个点
            is_new_group_start = torch.cat(
                [torch.tensor([True], device=device), sorted_group_ids[1:] != sorted_group_ids[:-1]])
            # group_cumsum: 在每个组内从0开始计数
            group_cumsum = torch.cumsum(is_new_group_start, dim=0) - 1
            # point_row_indices_sorted: 每个点应该被放置在 padded 张量的哪一行
            point_row_indices_sorted = torch.arange(len(sorted_group_ids), device=device) - torch.gather(
                torch.nonzero(is_new_group_start).squeeze(1), 0, group_cumsum)
            # 恢复原始顺序
            point_row_indices = torch.zeros_like(point_row_indices_sorted)
            point_row_indices[sorted_indices] = point_row_indices_sorted

            # 4. 创建容器张量并执行一次性填充 (Scatter 操作)
            padded_points = torch.zeros(B, C, self.patch_num, max_points_in_patch, self.d_model, device=device)
            key_padding_mask = torch.ones(B, C, self.patch_num, max_points_in_patch, device=device, dtype=torch.bool)

            # 使用高级索引，将所有有效点的嵌入一次性放入 padded_points 的正确位置
            padded_points[b_indices_valid, c_indices_valid, p_indices_valid, point_row_indices] = valid_point_embeds
            key_padding_mask[
                b_indices_valid, c_indices_valid, p_indices_valid, point_row_indices] = False  # False表示这个位置是真实数据

            padded_points_flat = rearrange(padded_points, 'b c p n d -> (b c p) n d')
            key_padding_mask_flat = rearrange(key_padding_mask, 'b c p n -> (b c p) n')

            # --- 核心修正开始 ---
            # 识别出哪些组是完全空的（即，其掩码行全部为True）
            is_empty_group = key_padding_mask_flat.all(dim=1)

            # 创建一个“安全”的掩码副本用于注意力计算
            safe_key_padding_mask = key_padding_mask_flat.clone()

            # 如果存在任何完全空的组
            if is_empty_group.any():
                # 对于这些空组，我们手动将其掩码的第一个位置设为 False。
                # 这能防止 softmax 接收到全为 -inf 的输入，从而避免 nan 的产生。
                # 注意力模块将会在一个全零的 padded_points 上计算，结果是一个稳定的零向量。
                safe_key_padding_mask[is_empty_group, 0] = False
            # --- 核心修正结束 ---

            # queries: [ (B*C*P), 1, D ]
            queries = self.patch_queries.unsqueeze(0).expand(B * C, -1, -1)  # -> [B*C, P, D]
            queries = rearrange(queries, 'bc p d -> (bc p) 1 d')

            # MHA 输入: query, key, value, key_padding_mask (使用修正后的安全mask)
            pooled_tokens, _ = self.attention_pool(queries, padded_points_flat, padded_points_flat,
                                                   key_padding_mask=safe_key_padding_mask)

            x_mixer_in = rearrange(pooled_tokens, '(b c p) 1 d -> b c p d', b=B, c=C)

            # 处理完全没有观测点的补丁，用可学习的 embedding 填充
            # 这里的 empty_mask 现在可以正确地将注意力计算产生的零向量替换掉
            empty_mask = (group_counts == 0).view(B, C, self.patch_num)
            x_mixer_in[empty_mask] = self.empty_patch_embedding

        # --- 阶段二：双视图时空混合器 (C-Mixer) ---
        # (此部分代码无需改动，本身就是高效的)
        x_mixer_out = x_mixer_in
        for layer in self.mixer_layers:
            x_mixer_out = layer(x_mixer_out)
        x_mixer_out = self.final_norm(x_mixer_out)

        # --- 阶段三：查询感知的上下文解码 ---
        # (此部分代码无需改动，本身就是高效的)
        query_times = y_mark
        query_patch_indices = torch.clamp(
            ((query_times.squeeze(-1) - min_time) / patch_width).floor().long(),
            0, self.patch_num - 1
        )

        k_offsets = torch.arange(-self.context_k, self.context_k + 1, device=device)
        context_indices = query_patch_indices.unsqueeze(-1) + k_offsets
        context_indices = torch.clamp(context_indices, 0, self.patch_num - 1)

        expanded_indices = rearrange(context_indices, 'b pred k -> b 1 pred k 1').expand(-1, C, -1, -1, self.d_model)
        source_patches = x_mixer_out.unsqueeze(2).expand(-1, -1, self.pred_len, -1, -1)
        context_patches = torch.gather(source_patches, 3, expanded_indices)

        patch_centers = min_time + patch_width * (torch.arange(self.patch_num, device=device) + 0.5)
        gathered_centers = patch_centers[context_indices]
        relative_times = query_times - gathered_centers

        relative_time_embeds = self.decoder_time_encoder(relative_times.unsqueeze(-1))
        attn_weights = self.decoder_attn_mlp(relative_time_embeds).softmax(dim=-2)
        attn_weights = attn_weights.unsqueeze(1)

        context_vector = (context_patches * attn_weights).sum(dim=3)

        query_time_embeds = self.time_encoder(query_times)
        query_time_embeds = self.point_time_embedding_mlp(query_time_embeds)
        query_time_embeds = query_time_embeds.unsqueeze(1).expand(-1, C, -1, -1)

        decoder_input = torch.cat([context_vector, query_time_embeds], dim=-1)
        decoder_input = rearrange(decoder_input, 'b c p d -> (b p c) d')
        output = self.decoder_output_mlp(decoder_input)
        output = rearrange(output, '(b p c) 1 -> b p c', b=B, p=self.pred_len)

        f_dim = -1 if self.configs.features == 'MS' else 0
        PRED_LEN = y.shape[1]
        return {
            "pred": output[:, -PRED_LEN:, f_dim:],
            "true": y[:, :, f_dim:],
            "mask": y_mask[:, :, f_dim:]
        }