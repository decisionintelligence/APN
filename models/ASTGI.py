import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import repeat
import numpy as np

from utils.globals import logger
from utils.ExpConfigs import ExpConfigs


def MLP(dims, dropout=0.1):
    layers = []
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class SpatioTemporalPropagationLayer(nn.Module):
    def __init__(self, d_model, d_coord, k, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_coord = d_coord
        self.k = k

        hidden_dim = int(d_model * mlp_ratio)

        self.relation_kernel = MLP([d_coord + 2 * d_model, hidden_dim, 1], dropout)
        self.message_mlp = MLP([d_model + d_coord, hidden_dim, d_model], dropout)
        self.update_mlp = MLP([d_model, hidden_dim, d_model], dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: Tensor, p: Tensor, t: Tensor, neighbor_indices: Tensor) -> Tensor:
        num_points = h.shape[0]
        h_i = h.unsqueeze(1).expand(-1, self.k, -1)
        p_i = p.unsqueeze(1).expand(-1, self.k, -1)
        t_i = t.unsqueeze(1).expand(-1, self.k, -1)
        h_j = h[neighbor_indices]
        p_j = p[neighbor_indices]
        t_j = t[neighbor_indices]
        p_rel = p_i - p_j
        relation_vec = torch.cat([p_rel, h_i, h_j], dim=-1)
        attn_scores = self.relation_kernel(relation_vec).squeeze(-1)
        causal_mask = (t_j > t_i).squeeze(-1)
        attn_scores.masked_fill_(causal_mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        msg_input = torch.cat([h_j, p_rel], dim=-1)
        messages = self.message_mlp(msg_input)
        aggregated_message = torch.sum(attn_weights.unsqueeze(-1) * messages, dim=1)
        h_new = self.norm(h + self.update_mlp(aggregated_message))
        return h_new


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.pred_len = configs.pred_len
        self.task_name = configs.task_name
        self.configs = configs

        self.n_vars = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout

        self.d_c = configs.astgi_channel_dim
        self.d_t = configs.astgi_time_dim
        self.n_layers = configs.astgi_prop_layers
        self.k = configs.astgi_k_neighbors
        self.mlp_ratio = configs.astgi_mlp_ratio
        self.w_c = configs.astgi_channel_dist_weight
        self.w_t = 1.0
        d_coord = self.d_c + self.d_t

        self.channel_embedding = nn.Embedding(self.n_vars, self.d_c)
        self.time_encoder = MLP([1, self.d_t * 2, self.d_t], self.dropout)
        self.value_encoder = MLP([1, self.d_model * 2, self.d_model], self.dropout)

        self.propagation_layers = nn.ModuleList([
            SpatioTemporalPropagationLayer(self.d_model, d_coord, self.k, self.mlp_ratio, self.dropout)
            for _ in range(self.n_layers)
        ])

        self.query_kernel = MLP([d_coord + self.d_model, int(self.d_model * self.mlp_ratio), 1], self.dropout)
        self.value_mlp = MLP([self.d_model, self.d_model, self.d_model], self.dropout)
        self.regression_head = MLP([self.d_model, int(self.d_model * self.mlp_ratio), 1], self.dropout)

    def _find_neighbors_pytorch(self, p_query: Tensor, p_hist: Tensor, k: int,
                                query_batch_idx: Tensor, hist_batch_idx: Tensor, B: int) -> Tensor:
        all_neighbor_indices = []
        point_global_indices = torch.arange(p_hist.shape[0], device=p_hist.device)

        for i in range(B):
            query_mask = (query_batch_idx == i)
            hist_mask = (hist_batch_idx == i)

            num_queries_in_batch = query_mask.sum()
            num_hist_in_batch = hist_mask.sum()

            if num_queries_in_batch == 0:
                continue

            if num_hist_in_batch == 0:
                placeholder = torch.zeros((num_queries_in_batch, k), dtype=torch.long, device=p_query.device)
                all_neighbor_indices.append(placeholder)
                continue

            p_q_batch = p_query[query_mask]
            p_hist_batch = p_hist[hist_mask]

            p_q_c, p_q_t = torch.split(p_q_batch, [self.d_c, self.d_t], dim=1)
            p_hist_c, p_hist_t = torch.split(p_hist_batch, [self.d_c, self.d_t], dim=1)

            dist_sq_c = torch.cdist(p_q_c, p_hist_c, p=2.0).pow(2)
            dist_sq_t = torch.cdist(p_q_t, p_hist_t, p=2.0).pow(2)

            dist_matrix_sq = self.w_c * dist_sq_c + self.w_t * dist_sq_t

            k_actual = min(k, num_hist_in_batch)

            _, indices_batch_local = torch.topk(dist_matrix_sq, k=k_actual, dim=1, largest=False)

            global_indices_map = point_global_indices[hist_mask]
            neighbor_indices_global = global_indices_map[indices_batch_local]

            if k_actual < k:
                padding_needed = k - k_actual
                last_neighbor = neighbor_indices_global[:, -1:]
                padding = last_neighbor.expand(-1, padding_needed)
                neighbor_indices_global = torch.cat([neighbor_indices_global, padding], dim=1)

            all_neighbor_indices.append(neighbor_indices_global)

        if not all_neighbor_indices:
            return torch.empty(0, k, dtype=torch.long, device=p_query.device)

        return torch.cat(all_neighbor_indices, dim=0)

    def forward(
            self,
            x: Tensor,
            x_mark: Tensor,
            x_mask: Tensor,
            **kwargs
    ) -> dict:
        B, L, N = x.shape

        if x_mask is None:
            x_mask = torch.ones_like(x, device=x.device, dtype=torch.bool)

        valid_indices = torch.nonzero(x_mask)
        batch_idx = valid_indices[:, 0]
        time_idx = valid_indices[:, 1]
        var_idx = valid_indices[:, 2]

        point_values = x[batch_idx, time_idx, var_idx].unsqueeze(-1)
        point_times = x_mark[batch_idx, time_idx, 0].unsqueeze(-1)

        e_c = self.channel_embedding(var_idx)
        e_t = self.time_encoder(point_times)
        p = torch.cat([e_c, e_t], dim=-1)
        h = self.value_encoder(point_values)

        with torch.no_grad():
            neighbor_indices = self._find_neighbors_pytorch(
                p_query=p, p_hist=p, k=self.k,
                query_batch_idx=batch_idx, hist_batch_idx=batch_idx, B=B
            )

        for layer in self.propagation_layers:
            h = layer(h, p, point_times, neighbor_indices)

        y_mark = kwargs['y_mark']
        pred_L = y_mark.shape[1]

        query_times_flat = y_mark[:, :, 0].reshape(-1, 1)
        query_vars_flat = torch.arange(N, device=x.device).repeat(B * pred_L)
        query_times_expanded = query_times_flat.repeat_interleave(N, dim=0)

        query_batch_idx = torch.arange(B, device=x.device).repeat_interleave(pred_L * N)

        q_e_c = self.channel_embedding(query_vars_flat)
        q_e_t = self.time_encoder(query_times_expanded)
        p_q = torch.cat([q_e_c, q_e_t], dim=-1)

        query_neighbor_indices = self._find_neighbors_pytorch(
            p_query=p_q, p_hist=p, k=self.k,
            query_batch_idx=query_batch_idx, hist_batch_idx=batch_idx, B=B
        )

        h_neighbor = h[query_neighbor_indices]
        p_neighbor = p[query_neighbor_indices]

        p_q_expanded = p_q.unsqueeze(1).expand(-1, self.k, -1)
        p_q_rel = p_q_expanded - p_neighbor

        query_relation_vec = torch.cat([p_q_rel, h_neighbor], dim=-1)
        query_attn_scores = self.query_kernel(query_relation_vec).squeeze(-1)
        query_attn_weights = torch.softmax(query_attn_scores, dim=-1)

        h_neighbor_transformed = self.value_mlp(h_neighbor)
        h_q = torch.sum(query_attn_weights.unsqueeze(-1) * h_neighbor_transformed, dim=1)
        predictions_flat = self.regression_head(h_q)

        outputs = predictions_flat.view(B, pred_L, N)

        y = kwargs['y']
        y_mask = kwargs['y_mask']
        f_dim = -1 if self.configs.features == 'MS' else 0

        return {
            "pred": outputs[:, :, f_dim:],
            "true": y[:, :, f_dim:],
            "mask": y_mask[:, :, f_dim:] if y_mask is not None else None
        }