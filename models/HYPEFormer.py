import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import *

# 假设这些工具函数与 HyperIMTS 在同一目录下
from utils.globals import logger
from utils.ExpConfigs import ExpConfigs
# 沿用 HyperIMTS 的 MultiHeadAttentionBlock，因为它是一个通用的构建块
from .HyperIMTS import MultiHeadAttentionBlock


class Model(nn.Module):
    """
    HYPE-Former 模型主类。
    负责编排整个流程：数据适配 -> 超图构建 -> 信息学习 -> 预测解码 -> 输出适配。
    """

    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.configs = configs
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.n_layers = configs.n_layers
        self.n_heads = configs.n_heads

        # HYPE-Former 的核心超参数：时间补丁的长度
        self.patch_len = configs.patch_len  # 需要在配置文件中添加此参数, e.g., 24

        seq_len = configs.seq_len_max_irr or configs.seq_len
        pred_len = configs.pred_len_max_irr or configs.pred_len
        total_len = seq_len + pred_len

        self.hypergraph_encoder = HYPEFormerEncoder(
            enc_in=self.enc_in,
            total_len=total_len,
            d_model=self.d_model,
            patch_len=self.patch_len
        )

        self.hypergraph_learner = HYPEFormerLearner(
            n_layers=self.n_layers,
            d_model=self.d_model,
            n_heads=self.n_heads
        )

        # 解码器现在接收来自节点、时间补丁超边和变量超边的信息
        self.decoder = nn.Linear(3 * self.d_model, 1)

    def forward(
            self,
            x: Tensor,
            x_mark: Tensor = None,
            x_mask: Tensor = None,
            y: Tensor = None,
            y_mark: Tensor = None,
            y_mask: Tensor = None,
            # 以下参数是为了兼容性和扩展性，在此模型中不直接使用，但会通过适配器生成
            x_L_flattened: Tensor = None,
            x_y_mask_flattened: Tensor = None,
            y_L_flattened: Tensor = None,
            y_mask_L_flattened: Tensor = None,
            exp_stage: str = "train",
            **kwargs
    ):
        # =====================================================================
        # 阶段 1: 输入适配器 (与 HyperIMTS 保持一致)
        # 将框架传入的填充后数据 (Padded) 转换为无填充的扁平化数据 (Flattened)
        # =====================================================================
        # BEGIN adaptor
        BATCH_SIZE, SEQ_LEN, ENC_IN = x.shape
        Y_LEN = self.configs.pred_len if self.configs.pred_len != 0 else SEQ_LEN

        # 处理可能的 None 输入
        if x_mark is None: x_mark = repeat(torch.arange(end=x.shape[1], dtype=x.dtype, device=x.device), "L -> B L 1",
                                           B=x.shape[0])
        if x_mask is None: x_mask = torch.ones_like(x, device=x.device, dtype=torch.bool)
        if y is None: y = torch.zeros((BATCH_SIZE, Y_LEN, ENC_IN), dtype=x.dtype, device=x.device)
        if y_mark is None: y_mark = repeat(
            torch.arange(start=SEQ_LEN, end=SEQ_LEN + Y_LEN, dtype=y.dtype, device=y.device), "L -> B L 1",
            B=y.shape[0])
        if y_mask is None: y_mask = torch.zeros_like(y, device=y.device, dtype=torch.bool)

        # 合并历史和未来序列
        x_zeros = torch.zeros_like(y, dtype=x.dtype, device=x.device)
        x_L = torch.cat([x, x_zeros], dim=1)
        y_L = torch.cat([torch.zeros_like(x), y], dim=1)
        x_y_mask = torch.cat([x_mask, y_mask], dim=1).bool()

        # 生成时间戳和变量索引
        time_indices = torch.arange(SEQ_LEN + Y_LEN, device=x.device).unsqueeze(0).unsqueeze(-1).repeat(BATCH_SIZE, 1,
                                                                                                        ENC_IN)
        variable_indices = torch.arange(ENC_IN, device=x.device).unsqueeze(0).unsqueeze(0).repeat(BATCH_SIZE,
                                                                                                  SEQ_LEN + Y_LEN, 1)

        # 扁平化操作
        N_OBSERVATIONS_MAX = torch.max(x_y_mask.sum((1, 2))).to(torch.int64)

        def pad(v):
            return F.pad(v, [0, N_OBSERVATIONS_MAX - len(v)], value=0)

        x_L_flattened = torch.stack([pad(r[m]) for r, m in zip(x_L, x_y_mask)]).contiguous()
        y_L_flattened = torch.stack([pad(r[m]) for r, m in zip(y_L, x_y_mask)]).contiguous()
        time_indices_flattened = torch.stack([pad(r[m]) for r, m in zip(time_indices, x_y_mask)]).contiguous()
        variable_indices_flattened = torch.stack([pad(r[m]) for r, m in zip(variable_indices, x_y_mask)]).contiguous()

        # 生成一个指示哪些是真实观测点的掩码
        observation_mask_flattened = torch.arange(N_OBSERVATIONS_MAX, device=x.device).unsqueeze(0) < x_y_mask.sum(
            (1, 2)).unsqueeze(1)
        # END adaptor

        # =====================================================================
        # 阶段 2: 动态补丁超图构建
        # =====================================================================
        (
            observation_nodes,
            patch_hyperedges,
            variable_hyperedges,
            patch_incidence_matrix,
            variable_incidence_matrix
        ) = self.hypergraph_encoder(
            x_flattened=x_L_flattened,
            time_indices_flattened=time_indices_flattened,
            variable_indices_flattened=variable_indices_flattened,
            observation_mask_flattened=observation_mask_flattened,
        )

        # =====================================================================
        # 阶段 3: 层级化信息学习
        # =====================================================================
        (
            final_observation_nodes,
            final_patch_hyperedges,
            final_variable_hyperedges
        ) = self.hypergraph_learner(
            observation_nodes=observation_nodes,
            patch_hyperedges=patch_hyperedges,
            variable_hyperedges=variable_hyperedges,
            patch_incidence_matrix=patch_incidence_matrix,
            variable_incidence_matrix=variable_incidence_matrix,
            observation_mask_flattened=observation_mask_flattened
        )

        # =====================================================================
        # 阶段 4: 预测解码与输出适配
        # =====================================================================
        # 为每个节点收集其对应的超边信息
        gathered_patch_hyperedges = final_patch_hyperedges.gather(
            dim=1,
            index=repeat(time_indices_flattened // self.patch_len, "B N -> B N D", D=self.d_model)
        )
        gathered_variable_hyperedges = final_variable_hyperedges.gather(
            dim=1,
            index=repeat(variable_indices_flattened, "B N -> B N D", D=self.d_model)
        )

        # 解码
        pred_flattened = self.decoder(
            torch.cat([
                final_observation_nodes,
                gathered_patch_hyperedges,
                gathered_variable_hyperedges
            ], dim=-1)
        ).squeeze(-1)
        y_mask_L = torch.cat([torch.zeros_like(x_mask), y_mask], dim=1)
        # 为损失函数准备掩码
        # 训练和验证时，我们只关心对未来（y_mask部分）的预测准确性
        y_mask_flattened = torch.stack([pad(r[m]) for r, m in zip(y_mask_L.float(), x_y_mask)]).contiguous()
        if exp_stage in ["train", "val"]:
            return {
                "pred": pred_flattened,
                "true": y_L_flattened,
                "mask": y_mask_flattened  # 只在 y 的位置计算损失
            }
        else:  # test stage
            # 将扁平化预测 unpad 并 reshape 回 (B, L, C) 的格式
            pred_padded = self.unpad_and_reshape(
                tensor_flattened=pred_flattened,
                original_mask=x_y_mask,
                original_shape=(BATCH_SIZE, SEQ_LEN + Y_LEN, ENC_IN)
            )
            f_dim = -1 if self.configs.features == 'MS' else 0
            return {
                "pred": pred_padded[:, -self.configs.pred_len:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:].bool()
            }

    def unpad_and_reshape(
            self,
            tensor_flattened: Tensor,
            original_mask: Tensor,
            original_shape: Tuple
    ):
        # (与 HyperIMTS 保持一致)
        batch_size, total_len, n_vars = original_shape
        result = torch.zeros(original_shape, dtype=tensor_flattened.dtype, device=tensor_flattened.device)

        for i in range(batch_size):
            mask_i = original_mask[i].bool()
            num_obs = mask_i.sum()
            result[i][mask_i] = tensor_flattened[i, :num_obs]

        return result


class HYPEFormerEncoder(nn.Module):
    """
    HYPE-Former 编码器：将扁平化的时序数据构建成动态补丁超图。
    """

    def __init__(self, enc_in, total_len, d_model, patch_len):
        super().__init__()
        self.enc_in = enc_in
        self.d_model = d_model
        self.patch_len = patch_len
        self.n_patches = math.ceil(total_len / patch_len)

        # 节点编码器：编码 (值, 时间)
        self.node_value_proj = nn.Linear(1, d_model // 2)
        self.node_time_proj = nn.Linear(1, d_model // 2)

        # 可学习的超边嵌入
        self.variable_hyperedge_embedding = nn.Embedding(enc_in, d_model)
        self.patch_hyperedge_embedding = nn.Embedding(self.n_patches, d_model)

    def forward(
            self,
            x_flattened: Tensor,
            time_indices_flattened: Tensor,
            variable_indices_flattened: Tensor,
            observation_mask_flattened: Tensor,
    ):
        BATCH_SIZE, N_OBS_MAX = x_flattened.shape

        # 1. 节点初始化
        # 使用正弦编码时间戳，增加时间信息的非线性
        time_feat = torch.sin(self.node_time_proj(time_indices_flattened.float().unsqueeze(-1)))
        value_feat = self.node_value_proj(x_flattened.unsqueeze(-1))
        observation_nodes = torch.cat([value_feat, time_feat], dim=-1)
        # 应用掩码，将无观测值位置的节点特征置零
        observation_nodes = observation_nodes * observation_mask_flattened.unsqueeze(-1)

        # 2. 超边初始化
        variable_hyperedges = self.variable_hyperedge_embedding(torch.arange(self.enc_in, device=x_flattened.device))
        variable_hyperedges = repeat(variable_hyperedges, "V D -> B V D", B=BATCH_SIZE)  # (B, C, D)

        patch_hyperedges = self.patch_hyperedge_embedding(torch.arange(self.n_patches, device=x_flattened.device))
        patch_hyperedges = repeat(patch_hyperedges, "P D -> B P D", B=BATCH_SIZE)  # (B, P, D)

        # 3. 构建关联矩阵 (Incidence Matrices)
        # 变量关联矩阵: (B, C, N_obs)
        variable_incidence_matrix = F.one_hot(variable_indices_flattened, num_classes=self.enc_in).transpose(1,
                                                                                                             2).float()
        variable_incidence_matrix = variable_incidence_matrix * observation_mask_flattened.unsqueeze(1)

        # 时间补丁关联矩阵: (B, P, N_obs)
        patch_indices = time_indices_flattened // self.patch_len
        patch_incidence_matrix = F.one_hot(patch_indices, num_classes=self.n_patches).transpose(1, 2).float()
        patch_incidence_matrix = patch_incidence_matrix * observation_mask_flattened.unsqueeze(1)

        return (
            observation_nodes,
            patch_hyperedges,
            variable_hyperedges,
            patch_incidence_matrix,
            variable_incidence_matrix
        )


class HYPEFormerLearner(nn.Module):
    """
    HYPE-Former 学习器：在超图上执行层级化信息传递。
    """

    def __init__(self, n_layers, d_model, n_heads):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model

        self.layers = nn.ModuleList([
            HYPEFormerLayer(d_model, n_heads) for _ in range(n_layers)
        ])

    def forward(
            self,
            observation_nodes: Tensor,
            patch_hyperedges: Tensor,
            variable_hyperedges: Tensor,
            patch_incidence_matrix: Tensor,
            variable_incidence_matrix: Tensor,
            observation_mask_flattened: Tensor
    ):
        for layer in self.layers:
            observation_nodes, patch_hyperedges, variable_hyperedges = layer(
                observation_nodes,
                patch_hyperedges,
                variable_hyperedges,
                patch_incidence_matrix,
                variable_incidence_matrix,
                observation_mask_flattened
            )
        return observation_nodes, patch_hyperedges, variable_hyperedges


class HYPEFormerLayer(nn.Module):
    """
    单个 HYPE-Former 层，包含两个阶段的信息传递。
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        # 阶段一：补丁内时空融合
        self.node_to_patch_attn = MultiHeadAttentionBlock(d_model, d_model, d_model, d_model, n_heads)
        self.patch_ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model))
        self.patch_to_node_proj = nn.Linear(d_model, d_model)
        self.norm1_node = nn.LayerNorm(d_model)
        self.norm1_patch = nn.LayerNorm(d_model)

        # 阶段二：变量内时序传播
        self.node_to_var_attn = MultiHeadAttentionBlock(d_model, d_model, d_model, d_model, n_heads)
        # 使用自注意力作为变量内的时序模型，因为它对变长序列友好
        self.var_temporal_model = MultiHeadAttentionBlock(d_model, d_model, d_model, d_model, n_heads)
        self.var_ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model))
        self.var_to_node_proj = nn.Linear(d_model, d_model)
        self.norm2_node = nn.LayerNorm(d_model)
        self.norm2_var = nn.LayerNorm(d_model)

    def forward(
            self,
            observation_nodes: Tensor,  # (B, N_obs, D)
            patch_hyperedges: Tensor,  # (B, P, D)
            variable_hyperedges: Tensor,  # (B, C, D)
            patch_incidence_matrix: Tensor,  # (B, P, N_obs)
            variable_incidence_matrix: Tensor,  # (B, C, N_obs)
            observation_mask_flattened: Tensor  # (B, N_obs)
    ):
        # --- 阶段一: 补丁内时空融合 (Node -> Patch -> Node) ---

        # 1. Node -> Patch: 节点向补丁超边发送消息并更新超边
        patch_hyperedges_res = patch_hyperedges
        updated_patch_hyperedges = self.node_to_patch_attn(
            Q=patch_hyperedges,
            K=observation_nodes,
            mask=patch_incidence_matrix
        )
        updated_patch_hyperedges = self.norm1_patch(patch_hyperedges_res + self.patch_ffn(updated_patch_hyperedges))

        # 2. Patch -> Node: 更新后的补丁超边信息广播回节点，得到中间节点表示
        # (B, P, D) x (B, N_obs, P) -> (B, N_obs, D)
        patch_info_for_nodes = torch.bmm(patch_incidence_matrix.transpose(1, 2), updated_patch_hyperedges)
        nodes_mid = self.norm1_node(observation_nodes + self.patch_to_node_proj(patch_info_for_nodes))
        nodes_mid = nodes_mid * observation_mask_flattened.unsqueeze(-1)

        # --- 阶段二: 变量内时序传播 (Node -> Var -> Node) ---

        # 1. Node -> Var: 节点的中间表示向变量超边发送消息
        var_hyperedges_res = variable_hyperedges
        # 此处使用 node_to_var_attn 聚合信息，可以看作是时序模型的第一步：特征提取
        aggregated_var_info = self.node_to_var_attn(
            Q=variable_hyperedges,
            K=nodes_mid,  # 使用中间节点表示
            mask=variable_incidence_matrix
        )
        # 在变量超边内部使用自注意力作为时序模型，捕捉其内部节点间的关系
        # 这是一个简化但有效的实现，避免了复杂的排序和循环
        var_temporal_info = self.var_temporal_model(aggregated_var_info, aggregated_var_info)
        updated_var_hyperedges = self.norm2_var(var_hyperedges_res + self.var_ffn(var_temporal_info))

        # 2. Var -> Node: 更新后的变量超边信息广播回节点，得到最终节点表示
        var_info_for_nodes = torch.bmm(variable_incidence_matrix.transpose(1, 2), updated_var_hyperedges)
        nodes_final = self.norm2_node(nodes_mid + self.var_to_node_proj(var_info_for_nodes))
        nodes_final = nodes_final * observation_mask_flattened.unsqueeze(-1)

        return nodes_final, updated_patch_hyperedges, updated_var_hyperedges
