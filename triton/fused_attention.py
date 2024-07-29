import triton
import triton.language as tl
import torch
import maia_athena
import torch_maia
from tqdm import tqdm
from time import time

torch_maia.set_device(0)
maia_athena.load_nepalrt_executive_firmware(0)
maia_athena.get_nepal_device(0).set_global_hbm_limit(int(1e9))


@triton.jit
def fused_attention_fwd_kernel(
    Q,
    K,
    V,
    M,
    Out,
    sm_scale,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    off_q = (
        off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    )
    off_k = (
        off_hz * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    )
    off_v = (
        off_hz * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    )
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)

    # loop over k, v and update accumulator
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(k_ptrs + start_n * stride_kn)
        qk = tl.dot(q, tl.trans(k))
        # -- causal masking
        # these 4 lines come from the tutorial kernel's STAGE 2 if body which computes
        # which includes causal masking before performing softmax
        mask = offs_m[:, None] >= (start_n + offs_n[None, :])
        qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(v_ptrs + start_n * stride_vn)
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        # update
        m_i = m_ij

    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    m_ptrs = M + off_hz * N_CTX + offs_m
    m_i += tl.math.log2(l_i)
    tl.store(m_ptrs, m_i)
    # initialize pointers to output
    offs_n = tl.arange(0, HEAD_DIM)
    off_o = (
        off_hz * stride_oh + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    )
    out_ptrs = Out + off_o
    acc = acc / l_i[:, None]
    tl.store(out_ptrs, acc)

def bench_flash_attention():
    BATCH = 1
    H = 32
    N_CTX = 4096
    BLOCK_M = 128
    D_HEAD = 128
    BLOCK_N = 128
    dtype = torch.bfloat16
    device = "maia"
    q = torch.randn(
        (BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=False
    )
    k = torch.randn(
        (BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=False
    )
    v = torch.randn(
        (BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=False
    )
    l = torch.zeros((BATCH, H, N_CTX), dtype=torch.float32, device=device)
    m = torch.zeros((BATCH, H, N_CTX), dtype=torch.float32, device=device)
    o = torch.zeros_like(q, device=device)
    sm_scale = 0.3
    grid = (triton.cdiv(N_CTX, BLOCK_M), BATCH * H, 1)

    fn = lambda: fused_attention_fwd_kernel[grid](
            q,
            k,
            v,
            # l,
            m,
            o,
            sm_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            BATCH,
            H,
            N_CTX,
            BLOCK_M,
            D_HEAD,
            BLOCK_N,
        )

    rounds = 100

    with torch.no_grad():
        # warm up
        for _ in tqdm(range(rounds)):
            fn()
        torch_maia.streams.current_stream().synchronize()
        # running
        tot = 0
        start = time()
        for _ in tqdm(range(rounds)):
            fn()
        torch_maia.streams.current_stream().synchronize()
        tot = time() - start
        print(tot / rounds * 1000, "ms")


bench_flash_attention()