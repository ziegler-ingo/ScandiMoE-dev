# This code is based on the litgpt implementation:
# https://github.com/Lightning-AI/litgpt/blob/main/litgpt/model.py

import math
import json
import yaml
from pathlib import Path
from functools import partial
from dataclasses import dataclass
from typing import Any, Literal, Optional, Type, Union, Tuple
from typing_extensions import Self

import torch
import torch.nn as nn

from scandimoe.utils.common import find_multiple


@dataclass
class Config:
    # model architecture
    n_layer: int = 16
    n_head: int = 32
    n_embd: int = 4096
    intermediate_size: int = 8192

    # attention
    head_size: Optional[int] = None
    parallel_residual: bool = True
    n_query_groups: Optional[int] = None

    # normalization
    norm_class_name: Literal["LayerNorm", "RMSNorm"] = "LayerNorm"
    norm_eps: float = 1e-5

    # feed-forward
    mlp_class_name: Literal["LlamaMLP", "LlamaMoE"] = "LlamaMLP"

    # positional embeddings
    rotary_percentage: float = 0.25
    rope_condense_ratio: int = 1
    rope_base: int = 10000

    # tokenization and vocabulary
    vocab_size: int = 32000
    padded_vocab_size: Optional[int] = None
    padding_multiple: int = 512

    # MoE
    n_expert: int = 0
    n_expert_per_token: int = 0

    # other
    name: Optional[str] = None
    bias: bool = True
    lm_head_bias: bool = False
    block_size: int = 4096  # context length

    def __post_init__(self):
        if self.head_size is None:
            assert self.n_embd % self.n_head == 0
            self.head_size = self.n_embd // self.n_head

        # vocab size should be a power of 2 to be optimal on hardware
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(
                self.vocab_size, self.padding_multiple
            )
        else:
            # vocab size shouldn't be larger than padded vocab size
            self.vocab_size = min(self.vocab_size, self.padded_vocab_size)

        # compute the number of query groups
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            # equals standard MHA
            self.n_query_groups = self.n_head

        self.rope_n_elem = int(self.rotary_percentage * self.head_size)

    @classmethod
    def from_file(cls, path: Union[str, Path], **kwargs: Any) -> Self:
        path = Path(path)
        if path.as_posix().endswith(".json"):
            with open(path, encoding="utf-8") as fp:
                file_kwargs = json.load(fp)
        elif path.as_posix().endswith(".yml"):
            with open(path, encoding="utf-8") as fp:
                file_kwargs = yaml.safe_load(fp)
        else:
            raise ValueError(f"{path} must be a `.json` or `.yml` file.")
        if file_kwargs is None:
            raise ValueError(f"{path} is empty which is likely unexpected.")
        file_kwargs.update(kwargs)
        return cls(**file_kwargs)

    @classmethod
    def from_checkpoint(cls, path: Union[str, Path], **kwargs: Any) -> Self:
        """Automatically load `model_config.yaml`."""
        if (config_path := path / "model_config.yaml").is_file():
            return cls.from_file(config_path, **kwargs)
        raise FileNotFoundError(f"For {str(path)!r} no 'model_config.yaml' exists.")

    @property
    def mlp_class(self) -> Type:
        # `self.mlp_class_name` cannot be the type to keep the config serializable
        if self.mlp_class_name.endswith("MoE"):
            return LlamaMoE
        else:
            return LlamaMLP

    @property
    def norm_class(self) -> Type:
        # `self.norm_class_name` cannot be the type to keep the config serializable
        if self.norm_class_name == "RMSNorm":
            return partial(RMSNorm, eps=self.norm_eps)
        return partial(nn.LayerNorm, eps=self.norm_eps)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
        # output projection
        # if `head_size` is explicitly specified in the config, `n_emd` might not be equal to `head_size * n_head`
        self.proj = nn.Linear(
            config.head_size * config.n_head, config.n_embd, bias=config.bias
        )
        # disabled by default
        self.kv_cache: Optional[KVCache] = None

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # batch size, sequence length, n_embd
        B, T, C = x.size()

        qkv = self.attn(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(
            B, T, self.config.n_query_groups, total_qkv, self.config.head_size
        )
        qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)

        # maybe repeat k and v if for the non multi-head attention cases
        # training: flash attention requires it
        # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
        if self.config.n_query_groups != self.config.n_head and (
            input_pos is None or self.config.n_query_groups != 1
        ):
            k = k.expand(
                B, self.config.n_query_groups, q_per_kv, T, self.config.head_size
            )
            v = v.expand(
                B, self.config.n_query_groups, q_per_kv, T, self.config.head_size
            )

        q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
        k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
        v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)

        q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
        k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)

        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)

        y = self.scaled_dot_product_attention(q, k, v, mask)

        # re-assemble all head outputs side by side
        y = y.reshape(B, T, self.config.head_size * self.config.n_head)

        return self.proj(y)

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.config.head_size)

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            scale=scale,
            is_causal=mask is None,
        )
        return y.transpose(1, 2)

    def build_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "KVCache":
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_head
        v_shape = (batch_size, heads, max_seq_length, self.config.head_size)
        if rope_cache_length is None:
            if self.config.rotary_percentage != 1.0:
                raise TypeError(
                    "Please pass the `rope_cache_length=gpt.cos.size(-1)` value"
                )
            k_shape = v_shape
        else:
            k_shape = (
                batch_size,
                heads,
                max_seq_length,
                rope_cache_length + self.config.head_size - self.config.rope_n_elem,
            )
        return KVCache(k_shape, v_shape, device=device, dtype=dtype)


class LlamaMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x)


class LlamaMoE(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.n_expert, bias=False)
        self.experts = nn.ModuleList(LlamaMLP(config) for _ in range(config.n_expert))

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Derived from: https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219
        See also figure 1 in https://arxiv.org/abs/2211.15841
        """
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        x = x.view(-1, C)  # (B*T, C)
        router = self.gate(x)  # (B*T, n_expert)
        probs, indices = torch.topk(
            router, self.config.n_expert_per_token
        )  # (B*T, n_expert_per_token)
        probs = probs.softmax(dim=1, dtype=torch.float).to(dtype=x.dtype)
        masks = indices.unsqueeze(-1) == torch.arange(
            self.config.n_expert, device=x.device
        )
        masks = masks.permute(2, 0, 1)  # (n_expert, B*T, n_expert_per_token)
        y = torch.zeros_like(x)  # (B*T, C)
        for mask, expert in zip(masks, self.experts):
            token_idx, expert_idx = torch.where(mask)
            y[token_idx] += probs[token_idx, expert_idx, None] * expert(x[token_idx])
        return y.view(B, T, C)


class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_normed = self.norm_1(x)
        attention_output = self.attn(x_normed, cos, sin, mask, input_pos)

        x = attention_output + x
        x = self.mlp(self.norm_2(x)) + x
        return x


class Mixtral(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.lm_head = nn.Linear(
            config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias
        )
        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[torch.Tensor] = None

    def forward(
        self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(
                f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}."
            )

        if input_pos is not None:  # use the kv cache
            cos = batched_index_select(self.cos, 0, input_pos)
            sin = batched_index_select(self.sin, 0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = batched_index_select(self.mask_cache, 2, input_pos)
            if mask.dim() > 4:
                # the mask cache has a batch dim of 1 in addition to the one
                # we get if input_pos has a batch dimension
                mask = mask.squeeze(1)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None

        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        for block in self.transformer.h:
            x = block(x, cos, sin, mask, input_pos)
        x = self.transformer.ln_f(x)
        x = self.lm_head(x)  # (b, t, vocab_size)
        return x

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        When doing inference, the sequences used might be shorter than the model's context length.
        This allows setting a smaller number to avoid allocating unused memory
        """
        if value > self.config.block_size:
            raise ValueError(
                f"Cannot attend to {value}, block size is only {self.config.block_size}."
                " This is likely because the input text exceeds the supported context length of this model."
            )
        self._max_seq_length = value
        if not hasattr(self, "cos"):
            # first call
            cos, sin = self.rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        # override
        elif value != self.cos.size(0):
            self.cos, self.sin = self.rope_cache(device=self.cos.device)
        # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
        # if the kv cache is expected

    def reset_parameters(self) -> None:
        # Trigger resetting the rope-cache
        self.cos, self.sin = self.rope_cache(device=self.cos.device)

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @classmethod
    def from_file(cls, path: Union[str, Path], **kwargs: Any) -> Self:
        return cls(Config.from_file(path, **kwargs))

    def rope_cache(
        self, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.config.rope_n_elem,
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
        )

    def set_kv_cache(
        self,
        batch_size: int,
        max_seq_length: Optional[int] = None,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if rope_cache_length is None:
            rope_cache_length = self.cos.size(-1)

        if max_seq_length is None:
            max_seq_length = self.max_seq_length

        # initialize the kv cache for all blocks
        for block in self.transformer.h:
            block.attn.kv_cache = block.attn.build_kv_cache(
                batch_size,
                max_seq_length,
                rope_cache_length,
                device,
                dtype,
            )

        if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
            # passing `attn_mask` to SDPA disables the flash implementation. since we only need the mask
            # for the kv-cache support (only during inference), we only create it in that situation
            self.mask_cache = build_mask_cache(max_seq_length, device)

    def clear_kv_cache(self) -> None:
        self.mask_cache = None
        for block in self.transformer.h:
            block.attn.kv_cache = None


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.size = size
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        # NOTE: the original RMSNorm paper implementation is not equivalent
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return (x_normed * self.weight.float()).to(dtype=dtype)

    def __repr__(self):
        return f"RMSNorm(size={self.size}, eps={self.eps})"

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)


class KVCache(nn.Module):
    def __init__(
        self,
        k_shape: Tuple[int, int, int, int],
        v_shape: Tuple[int, int, int, int],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "k", torch.zeros(k_shape, device=device, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "v", torch.zeros(v_shape, device=device, dtype=dtype), persistent=False
        )

    def forward(
        self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # move the buffer to the activation dtype for when AMP is used
        self.k = self.k.to(k.dtype)
        self.v = self.v.to(v.dtype)
        # update the cache
        n = k.size(0)
        k = batched_index_copy_(self.k[:n, ...], -2, input_pos, k)
        v = batched_index_copy_(self.v[:n, ...], -2, input_pos, v)
        return k, v

    def reset_parameters(self) -> None:
        torch.nn.init.zeros_(self.k)
        torch.nn.init.zeros_(self.v)


def build_mask_cache(
    max_seq_length: int, device: Optional[torch.device] = None
) -> torch.Tensor:
    ones = torch.ones((max_seq_length, max_seq_length), device=device, dtype=torch.bool)
    return torch.tril(ones).unsqueeze(0).unsqueeze(0)


def batched_index_select(t, dim, idx):
    """index_select for batched index and unbatched t"""
    if idx.dim() == 1:
        return torch.index_select(t, dim, idx)

    *batch_shape, idx_size = idx.shape
    res = torch.index_select(t, dim, idx.reshape(-1))  # flat index
    # split out single batch idx
    res = res.view(*t.shape[:dim], -1, idx_size, *t.shape[dim + 1 :])
    # move batch dim to front, this is np.rollaxis(res, dim, 0) for tensors
    dims = [dim] + list(range(res.dim()))
    del dims[dim + 1]
    res = res.permute(dims)
    # unflatten batch dims
    res = res.view(*batch_shape, *res.shape[1:])
    return res


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    device: Optional[torch.device] = None,
    base: int = 10000,
    condense_ratio: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

    return torch.cos(idx_theta), torch.sin(idx_theta)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    if cos.dim() > 1:
        # batch dimensions must align
        # sin/cos are (B, T, hs) so we unsqeeze -3 for nh
        # we count from back because all of apply_rope does
        cos = cos.unsqueeze(-3)
        sin = sin.unsqueeze(-3)

    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


def batched_index_copy_(t, dim, idx, val):
    """Index copy for batched t, idx, val"""

    if t.device.type == "mps":
        # Normalize negative dimensions
        if dim < 0:
            dim = t.dim() + dim
        if idx.dim() == 1:
            idx_shape = [1] * val.dim()
            idx_shape[dim] = -1
            idx_expanded = idx.view(*idx_shape)
            idx_expanded = idx_expanded.expand_as(val)
            t.scatter_(dim, idx_expanded, val)
            return t

        elif idx.dim() == 2:
            assert dim != 0, "Cannot index the batch dimension"
            batch_size = idx.size(0)
            idx_size = idx.size(1)
            assert batch_size == t.size(0) == val.size(0)

            idx_shape = [batch_size] + [1] * (val.dim() - 1)
            idx_shape[dim] = idx_size
            idx_expanded = idx.view(*idx_shape)
            idx_expanded = idx_expanded.expand_as(val)

            t.scatter_(dim, idx_expanded, val)
            return t
        else:
            raise NotImplementedError(f"idx.dim() == {idx.dim()} not supported")

    else:
        if idx.dim() == 1:
            return t.index_copy_(dim, idx, val)

        assert idx.dim() == 2, f"multiple batch dims not yet {idx.shape=}"
        assert dim != 0, f"cannot index batch dim"
        batch_size, idx_size = idx.shape
        assert batch_size == t.size(0)
        assert batch_size == val.size(0)
        t_indexed_dim = t.size(dim)

        # if we can view the batch and indexed dimensions together, we could
        # do index trickery. This is, sadly, not the case for kvcache so we
        # fall back to for loop
        for i in range(batch_size):
            unbatched_dim = dim if dim < 0 else dim - 1
            t[i].index_copy_(unbatched_dim, idx[i], val[i])
        return t
