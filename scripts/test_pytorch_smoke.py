#!/usr/bin/env python3
"""
Phase 10 PyTorch (ROCm) end-to-end anti-regression smoke test suite for gcn-sim.
Run with:
  LD_PRELOAD=build/libamdhip64.so .venv-rocm/bin/python scripts/test_pytorch_smoke.py
"""
import copy
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


def check_close(name: str, gpu_out: torch.Tensor, cpu_ref: torch.Tensor, tol: float):
    diff = (gpu_out.cpu().float() - cpu_ref.float()).abs().max().item()
    print(f"[PyTorch Smoke] {name:<28s} max_diff={diff:.3e} (tol={tol:.1e})")
    assert diff <= tol, f"{name} failed: max_diff={diff} > {tol}"


def main():
    # Stage 2: Device discovery & H2D/D2H
    assert torch.cuda.is_available(), "torch.cuda.is_available() returned False"
    dev_name = torch.cuda.get_device_name(0)
    print(f"[PyTorch Smoke] device={dev_name}")
    ones = torch.ones(16, device="cuda")
    check_close("stage2_ones_h2d_d2h", ones, torch.ones(16), 0.0)

    # Stage 3A: Elementwise ops
    torch.manual_seed(0)
    a = torch.randn(1024, dtype=torch.float32)
    b = torch.randn(1024, dtype=torch.float32)
    ag, bg = a.cuda(), b.cuda()
    check_close("stage3a_add", ag + bg, a + b, 1e-6)
    check_close("stage3a_mul", ag * bg, a * b, 1e-6)
    check_close("stage3a_relu", F.relu(ag), F.relu(a), 1e-6)
    check_close("stage3a_exp", torch.exp(ag), torch.exp(a), 1e-5)
    check_close("stage3a_tanh", torch.tanh(ag), torch.tanh(a), 1e-6)
    check_close("stage3a_gelu", F.gelu(ag), F.gelu(a), 1e-5)

    # Stage 3B: GEMM (FP32, FP16, BF16) & nn.Linear
    m1 = torch.randn(32, 64, dtype=torch.float32)
    m2 = torch.randn(64, 48, dtype=torch.float32)
    check_close("stage3b_matmul_fp32", torch.matmul(m1.cuda(), m2.cuda()), torch.matmul(m1, m2), 1e-4)

    m1_f16, m2_f16 = m1.half(), m2.half()
    check_close("stage3b_matmul_fp16", torch.matmul(m1_f16.cuda(), m2_f16.cuda()), torch.matmul(m1_f16, m2_f16), 5e-2)

    m1_bf16, m2_bf16 = m1.bfloat16(), m2.bfloat16()
    check_close("stage3b_matmul_bf16", torch.matmul(m1_bf16.cuda(), m2_bf16.cuda()), torch.matmul(m1_bf16, m2_bf16), 2.5e-1)

    linear = nn.Linear(64, 32)
    x_lin = torch.randn(8, 64)
    with torch.no_grad():
        ref_lin = linear(x_lin)
        gpu_lin = linear.cuda()(x_lin.cuda())
    check_close("stage3b_nn_linear_fp32", gpu_lin, ref_lin, 1e-4)

    # Stage 3C: Reductions, Softmax, LayerNorm
    r = torch.randn(4, 64, dtype=torch.float32)
    rg = r.cuda()
    check_close("stage3c_sum_all", rg.sum(), r.sum(), 1e-4)
    check_close("stage3c_sum_dim1", rg.sum(dim=1), r.sum(dim=1), 1e-4)
    check_close("stage3c_max_all", rg.max(), r.max(), 1e-6)
    check_close("stage3c_max_dim1", rg.max(dim=1).values, r.max(dim=1).values, 1e-6)
    check_close("stage3c_softmax", F.softmax(rg, dim=-1), F.softmax(r, dim=-1), 1e-5)
    check_close("stage3c_layer_norm", F.layer_norm(rg, (64,)), F.layer_norm(r, (64,)), 1e-5)

    # Stage 4: Mini-Model
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(64, 64),
        nn.LayerNorm(64),
        nn.GELU(),
        nn.Linear(64, 16),
    )
    x_mod = torch.randn(4, 64)
    with torch.no_grad():
        ref_mod = F.softmax(model(x_mod), dim=-1)
        gpu_mod = F.softmax(model.cuda()(x_mod.cuda()), dim=-1)
    check_close("stage4_mini_model", gpu_mod, ref_mod, 1e-4)

    # Stage 4B: 2-layer MiniGPT (Decoder-only Transformer)
    class CausalSelfAttention(nn.Module):
        def __init__(self, d_model, n_heads):
            super().__init__()
            self.n_heads = n_heads
            self.d_head = d_model // n_heads
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.proj = nn.Linear(d_model, d_model)

        def forward(self, x):
            B, T, C = x.shape
            qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
            return self.proj(y)

    class Block(nn.Module):
        def __init__(self, d_model, n_heads):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = CausalSelfAttention(d_model, n_heads)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x

    class MiniGPT(nn.Module):
        def __init__(self, vocab_size=64, block_size=16, d_model=64, n_heads=4, n_layers=2):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(block_size, d_model)
            self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx):
            B, T = idx.shape
            pos = torch.arange(0, T, device=idx.device)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            for blk in self.blocks:
                x = blk(x)
            return self.head(self.ln_f(x))

    torch.manual_seed(1234)
    gpt_cpu = MiniGPT().eval()
    idx_cpu = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        logits_cpu = gpt_cpu(idx_cpu)
        logits_gpu = copy.deepcopy(gpt_cpu).cuda().eval()(idx_cpu.cuda()).cpu()
    check_close("stage4_minigpt_logits", logits_gpu, logits_cpu, 1e-4)
    assert torch.equal(logits_gpu[:, -1, :].argmax(dim=-1), logits_cpu[:, -1, :].argmax(dim=-1)), \
        "MiniGPT next-token mismatch"

    print("[PyTorch Smoke] ALL PYTORCH SMOKE CHECKS PASSED!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
