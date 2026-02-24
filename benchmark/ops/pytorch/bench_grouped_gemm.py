###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import torch
import torch.utils.benchmark as benchmark

from primus_turbo.pytorch.ops import grouped_gemm
from tests.pytorch.ref.gemm_ref import (
    generate_grouped_gemm_group_lens,
)
from tests.pytorch.test_utils import compute_snr, get_tolerances

M_SIZE_LIST = [512, 1024, 2048, 4096]#, 8192, 16384]
EP_SIZE_LIST = [32, 16, 8]


def _generate_moe_test_cases(
    name_prefix: str,
    n_routed_experts: int,
    moe_intermediate_size: int,
    hidden_size: int,
):
    test_cases = []
    shapes_dict = {
        f"{name_prefix}-GateUP": (2 * moe_intermediate_size, hidden_size),
        f"{name_prefix}-Down": (hidden_size, moe_intermediate_size),
    }

    for ep in EP_SIZE_LIST:
        if n_routed_experts % ep != 0:
            continue
        B = n_routed_experts // ep
        if B < 1:
            continue
        for M in M_SIZE_LIST:
            for name, (N, K) in shapes_dict.items():
                for dtype in [torch.bfloat16]:
                    test_cases.append(
                        {
                            "Case": name,
                            "B": B,
                            "M": M,
                            "N": N,
                            "K": K,
                            "dtype": dtype,
                        }
                    )
    return test_cases


def generate_deepseekv3_test_cases():
    return _generate_moe_test_cases(
        "DSV3", n_routed_experts=256, moe_intermediate_size=2048, hidden_size=7168
    )


def generate_deepseekv2_test_cases():
    return _generate_moe_test_cases(
        "DSV2", n_routed_experts=160, moe_intermediate_size=1536, hidden_size=5120
    )


def generate_deepseekv2_lite_test_cases():
    return _generate_moe_test_cases(
        "DSV2-Lite", n_routed_experts=64, moe_intermediate_size=1408, hidden_size=2048
    )


def generate_grok_v2_test_cases():
    # https://huggingface.co/xai-org/grok-2/blob/main/config.json
    return _generate_moe_test_cases(
        "Grok-V2", n_routed_experts=8, moe_intermediate_size=16384, hidden_size=8192
    )


def make_fwd_bwd_funcs_te(x, w, group_lens, activation_dtype, return_dw_stacked=True):
    from transformer_engine.pytorch.module.base import get_multi_stream_cublas_workspace
    from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm

    B = int(group_lens.numel())
    N = int(w.shape[1])
    K = int(w.shape[2])

    m_splits = [int(v) for v in group_lens.tolist()]
    assert len(m_splits) == B
    sum_M = sum(m_splits)
    assert x.numel() > 0 and x.shape[0] == sum_M

    x_view = x.reshape(-1, x.shape[-1])
    xs = list(torch.split(x_view, m_splits))
    weights = [w[i] for i in range(B)]

    workspaces = get_multi_stream_cublas_workspace()

    # Forward output buffer
    out = torch.empty((sum_M, N), device=x.device, dtype=activation_dtype)

    def fwd_func_te():
        general_grouped_gemm(
            A=weights,
            B=xs,
            out=[out],
            out_dtype=activation_dtype,
            workspaces=workspaces,
            single_output=True,
            m_splits=m_splits,
            use_bias=False,
            bias=None,
            layout="TN",
        )
        return out

    # dx buffers
    dx = torch.empty((sum_M, K), device=x.device, dtype=activation_dtype)
    dxs = list(torch.split(dx, m_splits))

    # dw buffers
    dw_stacked = torch.empty((B, N, K), device=x.device, dtype=activation_dtype)
    dws = [dw_stacked[i] for i in range(B)]

    def bwd_func_te(grad_out):
        go = grad_out.view(-1, grad_out.shape[-1])

        splits = torch.split(go, m_splits)

        general_grouped_gemm(
            A=weights,
            B=splits,
            out=dxs,
            out_dtype=activation_dtype,
            workspaces=workspaces,
            single_output=False,
            layout="NN",
            m_splits=m_splits,
            grad=False,
            use_bias=False,
            bias=None,
        )

        general_grouped_gemm(
            A=xs,
            B=splits,
            out=dws,
            out_dtype=activation_dtype,
            workspaces=workspaces,
            single_output=False,
            layout="NT",
            m_splits=m_splits,
            grad=False,
            use_bias=False,
            bias=None,
            accumulate=False,
        )

        return dx, dw_stacked if return_dw_stacked else dws

    return fwd_func_te, bwd_func_te


from aiter.ops.triton.gmm import ptgmm
from aiter.ops.triton._triton_kernels.gmm import gmm_kernel, get_config


def aiter_gmm_forward(x: torch.Tensor, w: torch.Tensor, group_lens: torch.Tensor) -> torch.Tensor:
    B, N, K = w.shape
    sumL, N2 = x.shape

    group_sizes = group_lens.to(torch.int32)

    out = torch.empty((sumL, N), device=x.device, dtype=x.dtype)

    cfg = get_config("gmm", M=sumL, K=K, N=N, G=B, accumulate=False)

    BS_M = min(int(cfg.get("BLOCK_SIZE_M", 128)), 128)
    BS_N = min(int(cfg.get("BLOCK_SIZE_N", 128)), 128)
    BS_K = min(int(cfg.get("BLOCK_SIZE_K", 32)), 32)
    GROUP_SIZE = int(cfg.get("GROUP_SIZE", 1))
    GRID_DIM   = int(cfg.get("GRID_DIM", 240))

    num_warps  = int(cfg.get("NUM_WARPS", 8))
    num_stages = min(int(cfg.get("NUM_STAGES", 3)), 3)

    grid = (GRID_DIM,)

    gmm_kernel[grid](
        x, w, group_sizes, out,
        None,                 # bias_ptr
        sumL, K, N, B,
        TRANS_RHS=True,
        BLOCK_SIZE_M=BS_M,
        BLOCK_SIZE_K=BS_K,
        BLOCK_SIZE_N=BS_N,
        GROUP_SIZE=GROUP_SIZE,
        GRID_DIM=GRID_DIM,
        USE_BIAS=False,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def aiter_gmm_dx(grad_y: torch.Tensor, w: torch.Tensor, group_lens: torch.Tensor) -> torch.Tensor:
    B, N, K = w.shape
    sumL, N2 = grad_y.shape

    group_sizes = group_lens.to(torch.int32)

    dx = torch.empty((sumL, K), device=grad_y.device, dtype=grad_y.dtype)

    cfg = get_config("gmm", M=sumL, K=N, N=K, G=B, accumulate=False) or {}

    BS_M = int(cfg.get("BLOCK_SIZE_M", 128))
    BS_K = int(cfg.get("BLOCK_SIZE_K", 64))
    BS_N = int(cfg.get("BLOCK_SIZE_N", 128))
    GROUP_SIZE = int(cfg.get("GROUP_SIZE", 1))
    GRID_DIM = int(cfg.get("GRID_DIM", 240))

    grid = (GRID_DIM,)

    gmm_kernel[grid](
        grad_y, w, group_sizes, dx,
        None,
        sumL, N, K, B,
        TRANS_RHS=False,
        BLOCK_SIZE_M=BS_M,
        BLOCK_SIZE_K=BS_K,
        BLOCK_SIZE_N=BS_N,
        GROUP_SIZE=GROUP_SIZE,
        GRID_DIM=GRID_DIM,
        USE_BIAS=False,
        num_warps=int(cfg.get("NUM_WARPS", 8)),
        num_stages=min(int(cfg.get("NUM_STAGES", 3)), 3),
    )
    return dx


def aiter_ptgmm_dw(x: torch.Tensor, grad_y: torch.Tensor, group_lens: torch.Tensor) -> torch.Tensor:
    sumL, K = x.shape
    sumL2, N = grad_y.shape

    group_sizes = group_lens.to(torch.int32)

    dw = torch.empty((group_sizes.numel(), N, K), device=x.device, dtype=x.dtype)

    ptgmm(
        lhs=grad_y.t(),
        rhs=x,
        group_sizes=group_sizes,
        preferred_element_type=x.dtype,
        existing_out=dw,
        config=None,
        bias_grad=None,
        accumulate=False,
    )
    return dw


class GroupedGemmAiter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, group_lens):
        ctx.save_for_backward(x, w, group_lens)
        return aiter_gmm_forward(x, w, group_lens)

    @staticmethod
    def backward(ctx, grad_y):
        x, w, group_lens = ctx.saved_tensors
        dx = aiter_gmm_dx(grad_y, w, group_lens)
        dw = aiter_ptgmm_dw(x, grad_y, group_lens)
        return dx, dw, None


def bench_grouped_gemm(B, M, N, K, dtype):
    device = "cuda"
    # Prepare inputs
    x = torch.randn((B * M, K), dtype=dtype, device=device, requires_grad=True)
    w = torch.randn((B, N, K), dtype=dtype, device=device, requires_grad=True)
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=True).to(device)  # int64
    print("group_lens: ", group_lens)
    x_ref = x.clone().detach().requires_grad_()
    w_ref = w.clone().detach().requires_grad_()

    # Reference forward pass
    offs = torch.cumsum(group_lens, dim=0, dtype=torch.int32)
    ww = w_ref.transpose(-1, -2)
    fwd_func_ref = lambda: torch._grouped_mm(x_ref, ww, offs=offs, out_dtype=dtype)
    out_ref = fwd_func_ref()
    grad_out = torch.randn_like(out_ref)
    bwd_func_ref = lambda: out_ref.backward(grad_out, retain_graph=True)
    bwd_func_ref()

    # Forward pass for implementation
    fwd_func = lambda: grouped_gemm(x, w, group_lens, trans_b=True)
    bwd_func = lambda: out.backward(grad_out, retain_graph=True)
    out = fwd_func()
    bwd_func()

    # Check
    torch.testing.assert_close(out_ref, out, **get_tolerances(dtype))
    torch.testing.assert_close(x_ref.grad, x.grad, **get_tolerances(dtype))
    torch.testing.assert_close(w_ref.grad, w.grad, **get_tolerances(dtype))

    # TE grouped
    x_te = x.clone().detach()
    w_te = w.clone().detach()
    fwd_func_te, bwd_func_te_n = make_fwd_bwd_funcs_te(
        x_te, w_te, group_lens, activation_dtype=dtype, return_dw_stacked=True
    )

    out_te = fwd_func_te()

    bwd_func_te = lambda: bwd_func_te_n(grad_out)
    dx_te, dw_te = bwd_func_te()

    # Check TE
    torch.testing.assert_close(out_te, out_ref, **get_tolerances(dtype))
    torch.testing.assert_close(dx_te, x_ref.grad, **get_tolerances(dtype))
    torch.testing.assert_close(dw_te, w_ref.grad, **get_tolerances(dtype))

    # Aiter Grouped GEMM
    x_ai = x.clone().detach().requires_grad_()
    w_ai = w.clone().detach().requires_grad_()

    fwd_func_aiter = lambda: GroupedGemmAiter.apply(x_ai, w_ai, group_lens)
    out_aiter = fwd_func_aiter()

    bwd_func_aiter = lambda: out_aiter.backward(grad_out, retain_graph=True)
    bwd_func_aiter()

    # Check Aiter
    torch.testing.assert_close(out_aiter, out_ref, **get_tolerances(dtype))
    torch.testing.assert_close(x_ai.grad, x_ref.grad, **get_tolerances(dtype))
    torch.testing.assert_close(w_ai.grad, w_ref.grad, **get_tolerances(dtype))

    # Compute SNRs
    out_snr = compute_snr(out_ref, out)
    if out_snr <= 20:
        print(f"out_snr too low: {out_snr}")

    a_grad_snr = compute_snr(x_ref.grad, x.grad)
    b_grad_snr = compute_snr(w_ref.grad, w.grad)
    if a_grad_snr <= 20:
        print(f"x_grad_snr too low: {a_grad_snr}")
    if b_grad_snr <= 20:
        print(f"w_grad_snr too low: {b_grad_snr}")
    assert out_snr > 20, "out_snr too low"
    assert a_grad_snr > 20, "x_grad_snr too low"
    assert b_grad_snr > 20, "w_grad_snr too low"

    # Calculate FLOPs
    fwd_total_flops = 2 * B * M * N * K
    bwd_total_flops = 2 * fwd_total_flops

    # Warmup
    warmup = 20
    for _ in range(warmup):
        fwd_func()
        bwd_func()

        os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM"] = "1"
        fwd_func_te()
        bwd_func_te()
        os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM"] = "0"
        fwd_func_te()
        bwd_func_te()

        fwd_func_ref()
        bwd_func_ref()

        fwd_func_aiter()
        bwd_func_aiter()

    torch.cuda.synchronize()

    # Benchmark
    fwd_timer = benchmark.Timer(
        stmt="fn()",
        globals={"fn": fwd_func},
    )
    bwd_timer = benchmark.Timer(
        stmt="fn()",
        globals={"fn": bwd_func},
    )
    fwd_ref_timer = benchmark.Timer(
        stmt="fn()",
        globals={"fn": fwd_func_ref},
    )
    bwd_ref_timer = benchmark.Timer(
        stmt="fn()",
        globals={"fn": bwd_func_ref},
    )

    os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM"] = "1"
    fwd_te_timer = benchmark.Timer(stmt="fn()", globals={"fn": fwd_func_te})
    bwd_te_timer = benchmark.Timer(stmt="fn()", globals={"fn": bwd_func_te})

    fwd_te_measurement = fwd_te_timer.timeit(100)
    bwd_te_measurement = bwd_te_timer.timeit(100)

    fwd_te_time_ms = fwd_te_measurement.mean * 1e3
    bwd_te_time_ms = bwd_te_measurement.mean * 1e3

    fwd_te_tflops = fwd_total_flops / (fwd_te_time_ms * 1e-3) / 1e12
    bwd_te_tflops = bwd_total_flops / (bwd_te_time_ms * 1e-3) / 1e12

    os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM"] = "0"
    fwd_te_timer2 = benchmark.Timer(stmt="fn()", globals={"fn": fwd_func_te})
    bwd_te_timer2 = benchmark.Timer(stmt="fn()", globals={"fn": bwd_func_te})

    fwd_te_measurement2 = fwd_te_timer2.timeit(100)
    bwd_te_measurement2 = bwd_te_timer2.timeit(100)

    fwd_te_time_ms2 = fwd_te_measurement2.mean * 1e3
    bwd_te_time_ms2 = bwd_te_measurement2.mean * 1e3

    fwd_te_tflops2 = fwd_total_flops / (fwd_te_time_ms2 * 1e-3) / 1e12
    bwd_te_tflops2 = bwd_total_flops / (bwd_te_time_ms2 * 1e-3) / 1e12

    aiter_fwd_timer = benchmark.Timer(stmt="fn()", globals={"fn": fwd_func_aiter})
    aiter_bwd_timer = benchmark.Timer(stmt="fn()", globals={"fn": bwd_func_aiter})

    aiter_fwd_ms = aiter_fwd_timer.timeit(100)
    aiter_bwd_ms = aiter_bwd_timer.timeit(100)

    aiter_fwd_ms = aiter_fwd_ms.mean * 1e3
    aiter_bwd_ms = aiter_bwd_ms.mean * 1e3

    aiter_fwd_tflops = fwd_total_flops / (aiter_fwd_ms * 1e-3) / 1e12
    aiter_bwd_tflops = bwd_total_flops / (aiter_bwd_ms * 1e-3) / 1e12

    fwd_measurement = fwd_timer.timeit(100)
    bwd_measurement = bwd_timer.timeit(100)
    fwd_ref_measurement = fwd_ref_timer.timeit(100)
    bwd_ref_measurement = bwd_ref_timer.timeit(100)

    fwd_time_ms = fwd_measurement.mean * 1e3
    bwd_time_ms = bwd_measurement.mean * 1e3
    fwd_tflops = fwd_total_flops / (fwd_time_ms * 1e-3) / 1e12
    bwd_tflops = bwd_total_flops / (bwd_time_ms * 1e-3) / 1e12

    fwd_ref_time_ms = fwd_ref_measurement.mean * 1e3
    bwd_ref_time_ms = bwd_ref_measurement.mean * 1e3
    fwd_ref_tflops = fwd_total_flops / (fwd_ref_time_ms * 1e-3) / 1e12
    bwd_ref_tflops = bwd_total_flops / (bwd_ref_time_ms * 1e-3) / 1e12

    print(f"Primus-Turbo Forward  Mean time: {fwd_time_ms:.3f} ms | TFLOPS: {fwd_tflops:.2f}")
    print(f"Primus-Turbo Backward Mean time: {bwd_time_ms:.3f} ms | TFLOPS: {bwd_tflops:.2f}")
    print(f"Pytorch grouped Forward  Mean time: {fwd_ref_time_ms:.3f} ms | TFLOPS: {fwd_ref_tflops:.2f}")
    print(f"Pytorch grouped Backward Mean time: {bwd_ref_time_ms:.3f} ms | TFLOPS: {bwd_ref_tflops:.2f}")

    print(f"TE (CK_Tile) Forward  Mean time: {fwd_te_time_ms:.3f} ms | TFLOPS: {fwd_te_tflops:.2f}")
    print(f"TE (CK_Tile) Backward Mean time: {bwd_te_time_ms:.3f} ms | TFLOPS: {bwd_te_tflops:.2f}")

    print(f"TE (non-grouped) Forward  Mean time: {fwd_te_time_ms2:.3f} ms | TFLOPS: {fwd_te_tflops2:.2f}")
    print(f"TE (non-grouped) Backward Mean time: {bwd_te_time_ms2:.3f} ms | TFLOPS: {bwd_te_tflops2:.2f}")

    print(f"Aiter-Triton Forward  Mean time: {aiter_fwd_ms:.3f} ms | TFLOPS: {aiter_fwd_tflops:.2f}")
    print(f"Aiter-Triton Backward Mean time: {aiter_bwd_ms:.3f} ms | TFLOPS: {aiter_bwd_tflops:.2f}")

    return fwd_time_ms, fwd_tflops, bwd_time_ms, bwd_tflops, fwd_ref_time_ms, fwd_ref_tflops, bwd_ref_time_ms, bwd_ref_tflops, fwd_te_time_ms, fwd_te_tflops, bwd_te_time_ms, bwd_te_tflops, fwd_te_time_ms2, fwd_te_tflops2, bwd_te_time_ms2, bwd_te_tflops2, aiter_fwd_ms, aiter_fwd_tflops, aiter_bwd_ms, aiter_bwd_tflops


if __name__ == "__main__":
    dsv2_lite_test_cases = generate_deepseekv2_lite_test_cases()
    dsv2_test_cases = generate_deepseekv2_test_cases()
    dsv3_test_cases = generate_deepseekv3_test_cases()
    grok_v2_test_cases = generate_grok_v2_test_cases()
    test_cases = dsv2_lite_test_cases + dsv2_test_cases + dsv3_test_cases + grok_v2_test_cases

    import pandas as pd
    from tabulate import tabulate

    # DataFrame to store results
    results = pd.DataFrame(
        columns=[
            "TestID",
            "Case",
            "B",
            "M",
            "N",
            "K",
            "dtype",
            "Primus-Turbo Forward Time (ms)",
            "Primus-Turbo Forward TFLOPS",
            "Primus-Turbo Backward Time (ms)",
            "Primus-Turbo Backward TFLOPS",
            "Pytorch grouped Forward Time (ms)",
            "Pytorch grouped Forward TFLOPS",
            "Pytorch grouped Backward Time (ms)",
            "Pytorch grouped Backward TFLOPS",
            "TE (CK_Tile) Forward Time (ms)",
            "TE (CK_Tile) Forward TFLOPS",
            "TE (CK_Tile) Backward Time (ms)",
            "TE (CK_Tile) Backward TFLOPS",
            "TE (non-grouped) Forward Time (ms)",
            "TE (non-grouped) Forward TFLOPS",
            "TE (non-grouped) Backward Time (ms)",
            "TE (non-grouped) Backward TFLOPS",
            "Aiter-Triton Forward Time (ms)",
            "Aiter-Triton Forward TFLOPS",
            "Aiter-Triton Backward Time (ms)",
            "Aiter-Triton Backward TFLOPS",
        ]
    )
    test_id = 0

    # Run bench_grouped_gemm once to warmup
    B = test_cases[0]["B"]
    M = test_cases[0]["M"]
    N = test_cases[0]["N"]
    K = test_cases[0]["K"]
    dtype = test_cases[0]["dtype"]
    print(f"\n{'='*50}")
    print(f"WARMUP Case: {test_cases[0]}")
    print(f"{'='*50}")
    bench_grouped_gemm(
                B=B,
                M=M,
                N=N,
                K=K,
                dtype=dtype,
            )

    for case in test_cases:
        B = case["B"]
        M = case["M"]
        N = case["N"]
        K = case["K"]
        dtype = case["dtype"]
        print(f"\n{'='*50}")
        print(f"Testing Case: {case}")
        print(f"{'='*50}")
        test_id += 1
        try:
            # Run benchmark
            (
                fwd_time_ms,
                fwd_tflops,
                bwd_time_ms,
                bwd_tflops,
                fwd_ref_time_ms,
                fwd_ref_tflops,
                bwd_ref_time_ms,
                bwd_ref_tflops,
                fwd_te_time_ms, fwd_te_tflops, bwd_te_time_ms, bwd_te_tflops,
                fwd_te_time_ms2, fwd_te_tflops2, bwd_te_time_ms2, bwd_te_tflops2,
                aiter_fwd_ms, aiter_fwd_tflops, aiter_bwd_ms, aiter_bwd_tflops,
            ) = bench_grouped_gemm(
                B=B,
                M=M,
                N=N,
                K=K,
                dtype=dtype,
            )

            # Add to results table
            new_row = {
                "TestID": test_id,
                "Case": case["Case"],
                "B": B,
                "M": M,
                "N": N,
                "K": K,
                "dtype": dtype,
                "Primus-Turbo Forward Time (ms)": f"{fwd_time_ms:.2f}",
                "Primus-Turbo Forward TFLOPS": f"{fwd_tflops:.2f}",
                "Primus-Turbo Backward Time (ms)": f"{bwd_time_ms:.2f}",
                "Primus-Turbo Backward TFLOPS": f"{bwd_tflops:.2f}",
                "Pytorch grouped Forward Time (ms)": f"{fwd_ref_time_ms:.2f}",
                "Pytorch grouped Forward TFLOPS": f"{fwd_ref_tflops:.2f}",
                "Pytorch grouped Backward Time (ms)": f"{bwd_ref_time_ms:.2f}",
                "Pytorch grouped Backward TFLOPS": f"{bwd_ref_tflops:.2f}",
                "TE (CK_Tile) Forward Time (ms)": f"{fwd_te_time_ms:.2f}",
                "TE (CK_Tile) Forward TFLOPS": f"{fwd_te_tflops:.2f}",
                "TE (CK_Tile) Backward Time (ms)": f"{bwd_te_time_ms:.2f}",
                "TE (CK_Tile) Backward TFLOPS": f"{bwd_te_tflops:.2f}",
                "TE (non-grouped) Forward Time (ms)": f"{fwd_te_time_ms2:.2f}",
                "TE (non-grouped) Forward TFLOPS": f"{fwd_te_tflops2:.2f}",
                "TE (non-grouped) Backward Time (ms)": f"{bwd_te_time_ms2:.2f}",
                "TE (non-grouped) Backward TFLOPS": f"{bwd_te_tflops2:.2f}",
                "Aiter-Triton Forward Time (ms)": f"{aiter_fwd_ms:.2f}",
                "Aiter-Triton Forward TFLOPS": f"{aiter_fwd_tflops:.2f}",
                "Aiter-Triton Backward Time (ms)": f"{aiter_bwd_ms:.2f}",
                "Aiter-Triton Backward TFLOPS": f"{aiter_bwd_tflops:.2f}",
            }
            results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

        except Exception as e:
            raise
            print(f"Failed to run {case}: {str(e)}")
            new_row = {
                "TestID": test_id,
                "Case": case["Case"],
                "B": B,
                "M": M,
                "N": N,
                "K": K,
                "dtype": dtype,
                "Forward Time (ms)": "Failed",
                "Forward TFLOPS": "N/A",
                "Backward Time (ms)": "Failed",
                "Backward TFLOPS": "N/A",
                "Ref Forward Time (ms)": "Failed",
                "Ref Forward TFLOPS": "N/A",
                "Ref Backward Time (ms)": "Failed",
                "Ref Backward TFLOPS": "N/A",
            }
            results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

    # Print results
    print("\nFinal Results:")
    print(tabulate(results, headers="keys", tablefmt="grid", showindex=False))

    # Save to CSV
    results.to_csv("grouped_gemm_benchmark_results.csv", index=False)
    print("Results saved to grouped_gemm_benchmark_results.csv")
