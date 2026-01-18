# -*- coding: utf-8 -*-

from typing import List, Optional

import torch
import torch.nn.functional as F
from einops import rearrange


@torch.compile
def naive_chunk_simple_mhla_fixed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mixing_matrix: torch.Tensor,
    output_final_state: bool = False,
    chunk_size: int = 64,
    *args, **kwargs
):
    """

    Args:
        q (torch.Tensor): _description_
        k (torch.Tensor): _description_
        v (torch.Tensor): _description_
        mixing_matrix (torch.Tensor): Shape: (chunk_number, chunk_number, 1, 1, 1, 1)
        local_kv_summaries (Optional[torch.Tensor], optional): _description_. Defaults to None.
        output_final_state (bool, optional): _description_. Defaults to False.
        chunk_size (int, optional): _description_. Defaults to 64.

    Returns:
        _type_: _description_
    """
    # Enable TF32 for CUDA matmul and cuDNN for higher performance
    # if torch.cuda.is_available():
    #     torch.backends.cuda.matmul.allow_tf32 = True
    #     torch.backends.cudnn.allow_tf32 = True
    dtype = q.dtype
    q, k, v = map(lambda x: rearrange(x, 'b t h ... -> b h t ...').to(torch.float32), [q, k, v])
    # if scale is None:
    #     scale = 1.0 / q.shape[-1] ** 0.5
    scale = q.shape[-1] ** -0.5

    T = q.shape[-2]
    BT = chunk_size
    pad_len = (BT - (T % BT)) % BT
    if pad_len > 0:
        # Pad all tensors
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
    # decay = k.new_zeros(B, H, T1, 1)
    B, H, T1, K = q.shape
    chunk_number = T1 // chunk_size
    mixing_matrix = mixing_matrix[:chunk_number, :chunk_number, ...]
    # q = q * scale
    q, k, v = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), [q, k, v])
    q = q * scale
    S_all_list = []
    for i in range(chunk_number):
        k_i, v_i = k[:, :, i], v[:, :, i]  # [B,H,c,K], [B,H,c,V]
        S_i = k_i.transpose(-1, -2) @ v_i  # [B,H,K,V]
        S_all_list.append(S_i)
    S_all = torch.stack(S_all_list, dim=0)  # [n,B,H,K,V]

    o = torch.zeros_like(v)
    L_mask = torch.tril(torch.ones(chunk_size, chunk_size, dtype=v.dtype, device=q.device))
    for i in range(chunk_number):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]

        attn = (q_i @ k_i.transpose(-1, -2)) * L_mask  # [B,H,c,c]

        mm = mixing_matrix[i, :i, ...]

        prefix_S = (mm * S_all[:i]).sum(dim=0)             # [B,H,K,V]

        o_inter = q_i @ prefix_S                              # [B,H,c,V]
        o[:, :, i] = (o_inter + mixing_matrix[i, i] * (attn @ v_i))

    S = S_all if output_final_state else None
    # unpad
    o = rearrange(o, 'b h n c d -> b (n c) h d')[:, :T].to(dtype)
    return o




def naive_recurrent_mhla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mixing_matrix: torch.Tensor,
    chunk_size: int = 64,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True
):
    dtype = q.dtype
    q, k, v= map(lambda x: x.transpose(1, 2).float(), (q, k, v))
    BT = chunk_size
    T = q.shape[-2]
    scale = q.shape[-1] ** -0.5
    pad_len = (BT - (T % BT)) % BT
    B, H, T, K = q.shape
    V = v.shape[-1]
    o = v.new_zeros(B, H, T, V)
    
    chunk_number = T // chunk_size
    if pad_len > 0:
        chunk_number += 1
    mixing_matrix = mixing_matrix[:chunk_number, :chunk_number, ...]

    S = q.new_zeros(B, H, K, V)
    if initial_state is not None:
        S += initial_state
        
    S_all = []
    # Z_all = []
    current_s = q.new_zeros(B, H, K, V)
    # current_z = q.new_zeros(B, H, 1, K)

    for i in range(T):
        chunk_index = i // chunk_size
        if i % chunk_size == 0:
            S_all.append(current_s)
            # Z_all.append(current_z)
            current_s = q.new_zeros(B, H, K, V)
            # current_z = q.new_zeros(B, H, 1, K)
        key = k[:, :, i]
        value = v[:, :, i]
        kv = key.unsqueeze(-1) * value.unsqueeze(-2)
        current_s += kv
        q_i = q[:, :, i, :] * scale
        # S = S * gate.unsqueeze(-1).unsqueeze(-1) + kv
        mm = mixing_matrix[chunk_index, :chunk_index + 1, ...]
        S_all_t = (mm * torch.stack([*S_all[:chunk_index], current_s], dim=0)).sum(dim=0)
        
        o_i = (q_i.unsqueeze(-1) * S_all_t).sum(-2) 
        o[:, :, i] = o_i
    if not output_final_state:
        S = None
    return o.transpose(1, 2).to(dtype), S
    


