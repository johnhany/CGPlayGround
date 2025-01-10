import time

import numpy as np
import taichi as ti
import torch
import torch.nn as nn

from torch.nn import functional as F

torch.backends.cudnn.benchmark = True


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base


def RUN_FORMULA_VERY_SLOW(w, k, B, C, T, eps):
    # this is the formula (very slow)
    out = torch.empty((B, C, T), device='cuda')
    for b in range(B):
        for c in range(C):
            for t in range(T):
                s = eps
                for u in range(0, t+1):
                    s += w[c][(T-1)-(t-u)] * k[b][c][u]
                out[b][c][t] = s
    return out


def RUN_PYTORCH(w, k, B, C, T, eps):
    # this shall equal the formula
    return F.conv1d(nn.ZeroPad2d((T-1, 0, 0, 0))(k), w.unsqueeze(1), groups=C) + eps


fwd_t_group = 6
fwd_b_group = 8
Tmax = 768
fwd_block_dim=128
ti_matf = ti.types.matrix(fwd_b_group, fwd_t_group, dtype=float)

@ti.kernel
def timex_taichi_forward(
        out: ti.types.ndarray(ndim=3), # type: ignore
        w: ti.types.ndarray(ndim=2), # type: ignore
        k: ti.types.ndarray(ndim=3), # type: ignore
        B: ti.i32, C: ti.i32, T: ti.i32, eps: ti.f32): # type: ignore
    ti.loop_config(block_dim=fwd_block_dim)
    for b_block, c, t_block in ti.ndrange(B // fwd_b_group, C, T // fwd_t_group):
        # Group both b and t with factor 4
        t = t_block * fwd_t_group
        b = b_block * fwd_b_group
        s_mat = ti_matf(((eps,) * fwd_t_group,) * fwd_b_group)
        k_pad = ti.simt.block.SharedArray((fwd_b_group, Tmax), ti.f32)
        w_pad = ti.simt.block.SharedArray((Tmax,), ti.f32)
        for bi in ti.static(range(fwd_b_group)):
            for i in ti.static(range(fwd_t_group)):
                k_pad[bi, t + i] = k[b + bi, c, t + i]
                if bi == 0:
                    w_pad[t + i] = w[c, t + i]
        ti.simt.block.sync()
        for u in range(0, t+1):
            for bi in ti.static(range(fwd_b_group)):
                k_val = k_pad[bi, u]
                for i in ti.static(range(fwd_t_group)):
                    s_mat[bi, i] += w_pad[(T-1)-(t-(u-i))] * k_val
        ti.simt.block.sync()
        # Compute the remaining triangle in the thread group.
        for bi in ti.static(range(fwd_b_group)):
            for i in ti.static(range(1, fwd_t_group)):
                for j in ti.static(range(i)):
                    s_mat[bi, i] += w_pad[T - i + j] * k_pad[bi, t+1+j]
            for i in ti.static(range(fwd_t_group)):
                out[b + bi, c, t+ i] = s_mat[bi, i]


bwd_t_group = 6
bwd_b_group = 4
bwd_block_dim = 128
ti_back_matf = ti.types.matrix(bwd_b_group, bwd_t_group, dtype=float)

@ti.kernel
def timex_taichi_backward(
        w: ti.types.ndarray(ndim=2), # type: ignore
        k: ti.types.ndarray(ndim=3), # type: ignore
        gwk: ti.types.ndarray(ndim=3), # type: ignore
        gw: ti.types.ndarray(ndim=3), # type: ignore
        gk: ti.types.ndarray(ndim=3), # type: ignore
        B: ti.i32, C: ti.i32, T: ti.i32): # type: ignore
    ti.loop_config(block_dim=bwd_block_dim)
    for b_block, c, t_block in ti.ndrange(B // bwd_b_group, C, T // bwd_t_group):
        t = t_block * bwd_t_group
        b = b_block * bwd_b_group
        s = ti_back_matf(((0.0,) * bwd_t_group,)*bwd_b_group)
        gwk_pad = ti.simt.block.SharedArray((fwd_b_group, Tmax), ti.f32)
        w_pad = ti.simt.block.SharedArray((Tmax,), ti.f32)
        for bi in ti.static(range(bwd_b_group)):
            for i in ti.static(range(bwd_t_group)):
                    gwk_pad[bi, t + i] = gwk[b + bi, c, t + i]
                    if bi == 0:
                        w_pad[t + i] = w[c, t + i]
        ti.simt.block.sync()
        for bi in ti.static(range(0, bwd_b_group)):
            for u in range(0, t+1):
                for i in ti.static(range(0, bwd_t_group)):
                    s[bi, i] += gwk_pad[bi, (T-1)-(t+i-u)] * k[b + bi, c, u]
        ti.simt.block.sync()
        # The last triangle is specialized
        # u is replaced with t+1+j
        for bi in ti.static(range(0, bwd_b_group)):
            for i in ti.static(range(1, bwd_t_group)):
                for j in ti.static(range(i)):
                    s[bi, i] += gwk_pad[bi, T-i+j] * k[b + bi, c, t+1+j]
        # write out
        for bi in ti.static(range(0, bwd_b_group)):
            for i in ti.static(range(bwd_t_group)):
                gw[b + bi, c, t+i] = s[bi, i]

        s = ti_back_matf(((0.0,) * bwd_t_group,)*bwd_b_group)
        # The first triangle is specialized
        # t' = t + i
        # u' = t' + j
        for bi in ti.static(range(0, bwd_b_group)):
            for i in ti.static(range(0, bwd_t_group-1)):
                for j in ti.static(range(i, bwd_t_group-1)):
                    s[bi, i] += gwk_pad[bi, T+i-j-1] * w_pad[t+j]

        for bi in ti.static(range(0, bwd_b_group)):
            for u in range(t+bwd_t_group-1, T):
                for i in ti.static(range(0, bwd_t_group)):
                    s[bi, i] += gwk_pad[bi, (T-1)+(t+i-u)] * w_pad[u]
        ti.simt.block.sync()
        # write out
        for bi in ti.static(range(0, bwd_b_group)):
            for i in ti.static(range(bwd_t_group)):
                gk[b+bi, c, t+i] = s[bi, i]


class TimeX_Taichi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, k, B, C, T, eps):
        ctx.B = B
        ctx.C = C
        ctx.T = T
        #assert ctx.T % 4 == 0 and ctx.T <= T_MAX and ctx.B % B_GROUP_FORWARD == 0 and ctx.B % B_GROUP_BACKWARD == 0, "require T % 4 == 0 and T <= T_MAX and B % B_GROUP_* == 0"
        w = w.contiguous()
        k = k.contiguous()
        ctx.save_for_backward(w, k)
        wk = torch.empty((B, C, T), device='cuda',
                         memory_format=torch.contiguous_format)
        timex_taichi_forward(wk, w, k, B, C, T, eps)
        ti.sync()
        return wk

    @staticmethod
    def backward(ctx, gwk):
        #assert ctx.T % 4 == 0 and ctx.T <= T_MAX and ctx.B % B_GROUP_FORWARD == 0 and ctx.B % B_GROUP_BACKWARD == 0, "require T % 4 == 0 and T <= T_MAX and B % B_GROUP_* == 0"
        w, k = ctx.saved_tensors
        gw = torch.empty((ctx.B, ctx.C, ctx.T), device='cuda',
                         memory_format=torch.contiguous_format)
        gk = torch.empty((ctx.B, ctx.C, ctx.T), device='cuda',
                         memory_format=torch.contiguous_format)
        timex_taichi_backward(w, k, gwk.contiguous(), gw,
                            gk, ctx.B, ctx.C, ctx.T)
        ti.sync()
        # actually pytorch will do gw.sum(dim=0) but we will do it anyway just to be safe
        return (gw.sum(dim=0), gk, None, None, None, None)


def RUN_TAICHI(w, k, B, C, T, eps):
    return TimeX_Taichi.apply(w.cuda(), k.cuda(), B, C, T, eps)


def CHECK_PYTORCH():
    B = 3
    C = 5
    T = 11
    eps = 0.1

    set_seed(42)
    w = torch.rand(C, T, requires_grad=True, device='cuda')
    k = torch.rand(B, C, T, requires_grad=True, device='cuda')

    r0 = RUN_FORMULA_VERY_SLOW(w, k, B, C, T, eps)
    r1 = RUN_PYTORCH(w, k, B, C, T, eps)

    print('--> pytorch correct =', torch.allclose(r0, r1),
          ', err ratio =', get_err_ratio(r0, r1))


def CHECK_TAICHI(silent=False):
    B = 32
    C = 768
    T = 768
    eps = 0.1

    set_seed(42)
    w = torch.rand(C, T, requires_grad=True, device='cuda')
    k = torch.rand(B, C, T, requires_grad=True, device='cuda')

    # check forward

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        r1 = RUN_PYTORCH(w, k, B, C, T, eps)
    if not silent:
        print('pytorch forward\n', prof.key_averages(group_by_stack_n=5).table(
            sort_by='self_cuda_time_total', row_limit=5))

    # check backward
    # a strange loss for better verification
    loss1 = ((r1 * r1) - torch.tanh(r1)).sum()
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        loss1.backward()
    if not silent:
        print('pytorch backward\n', prof.key_averages(group_by_stack_n=5).table(
            sort_by='self_cuda_time_total', row_limit=5))
    gw1 = w.grad.data.clone()
    gk1 = k.grad.data.clone()

    w.grad.data.zero_()
    k.grad.data.zero_()

    w.grad.data.zero_()
    k.grad.data.zero_()

    # Check Taichi
    ti.init(arch=ti.cuda, kernel_profiler=True)
    # Taichi
    r3 = RUN_TAICHI(w, k, B, C, T, eps)
    loss3 = ((r3 * r3) - torch.tanh(r3)).sum()
    loss3.backward()
    w.grad.data.zero_()
    k.grad.data.zero_()
    ti.profiler.clear_kernel_profiler_info()
    r3 = RUN_TAICHI(w, k, B, C, T, eps)
    ti.sync()

    print('--> Taichi fwd correct =', torch.allclose(r1, r3),
         ', err ratio =', get_err_ratio(r1, r3))
    loss3 = ((r3 * r3) - torch.tanh(r3)).sum()
    loss3.backward()
    if not silent:
        ti.profiler.print_kernel_profiler_info('trace')
    gw3 = w.grad.data.clone()
    gk3 = k.grad.data.clone()

    print('--> bwd gradW correct =', torch.allclose(gw1, gw3),
         ', err ratio =', get_err_ratio(gw1, gw3))
    print('--> bwd gradK correct =', torch.allclose(gk1, gk3),
         ', err ratio =', get_err_ratio(gk1, gk3))


if __name__ == "__main__":
    print('PyTorch benchmark...')
    CHECK_PYTORCH()

    print('Taichi benchmark...')
    CHECK_TAICHI(silent=False)  # benchmark
