import torch
import triton
import triton.language as tl
from torch.autograd import Function


# --------------------------------------------------------------------------
#                    1. 高性能 Triton 内核 (优化版)
# --------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_W': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_W': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_W': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE_W': 1024}, num_warps=8),
    ],
    key=['W'],
)
@triton.jit
def q_shift_fwd_kernel_optimized(
    output_ptr, input_ptr,
    C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    C1: tl.constexpr, C2: tl.constexpr, C3: tl.constexpr, C4: tl.constexpr,
    SHIFT_PIXEL: tl.constexpr,
    stride_b, stride_c, stride_h,
    BLOCK_SIZE_W: tl.constexpr
):

    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)

    row_base_ptr = input_ptr + pid_b * stride_b + pid_c * stride_c + pid_h * stride_h
    output_row_ptr = output_ptr + pid_b * stride_b + pid_c * stride_c + pid_h * stride_h

    for w_start in range(0, W, BLOCK_SIZE_W):
        w_offsets = w_start + tl.arange(0, BLOCK_SIZE_W)
        
        src_h_offsets = pid_h
        src_w_offsets = w_offsets
        
        if pid_c < C1:      # Group 1: Shift Left
            src_w_offsets = w_offsets - SHIFT_PIXEL
        elif pid_c < C2:    # Group 2: Shift Right
            src_w_offsets = w_offsets + SHIFT_PIXEL
        elif pid_c < C3:    # Group 3: Shift Up
            src_h_offsets = pid_h - SHIFT_PIXEL
        elif pid_c < C4:    # Group 4: Shift Down
            src_h_offsets = pid_h + SHIFT_PIXEL
        
        load_mask = (w_offsets < W) & (src_h_offsets >= 0) & (src_h_offsets < H) & (src_w_offsets >= 0) & (src_w_offsets < W)
        store_mask = w_offsets < W
        
        src_ptr = input_ptr + pid_b * stride_b + pid_c * stride_c + src_h_offsets * stride_h + src_w_offsets
        
        block = tl.load(src_ptr, mask=load_mask, other=0.0)
        tl.store(output_row_ptr + w_offsets, block, mask=store_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_W': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_W': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_W': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE_W': 1024}, num_warps=8),
    ],
    key=['W'],
)
@triton.jit
def q_shift_bwd_kernel_optimized(
    dx_ptr, dy_ptr,
    C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    C1: tl.constexpr, C2: tl.constexpr, C3: tl.constexpr, C4: tl.constexpr,
    SHIFT_PIXEL: tl.constexpr,
    stride_b, stride_c, stride_h,
    BLOCK_SIZE_W: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    dy_row_ptr = dy_ptr + pid_b * stride_b + pid_c * stride_c + pid_h * stride_h
    
    for w_start in range(0, W, BLOCK_SIZE_W):
        w_offsets = w_start + tl.arange(0, BLOCK_SIZE_W)
        store_mask = w_offsets < W
        
        grad_block = tl.load(dy_row_ptr + w_offsets, mask=store_mask, other=0.0)
        
        dst_h_offsets = pid_h
        dst_w_offsets = w_offsets
        if pid_c < C1:      # Fwd: Left, Bwd: Left
            dst_w_offsets = w_offsets - SHIFT_PIXEL
        elif pid_c < C2:    # Fwd: Right, Bwd: Right
            dst_w_offsets = w_offsets + SHIFT_PIXEL
        elif pid_c < C3:    # Fwd: Up, Bwd: Up
            dst_h_offsets = pid_h - SHIFT_PIXEL
        elif pid_c < C4:    # Fwd: Down, Bwd: Down
            dst_h_offsets = pid_h + SHIFT_PIXEL

        write_mask = (store_mask) & (dst_h_offsets >= 0) & (dst_h_offsets < H) & (dst_w_offsets >= 0) & (dst_w_offsets < W)
        
        dx_ptr_row = dx_ptr + pid_b * stride_b + pid_c * stride_c + dst_h_offsets * stride_h + dst_w_offsets
        tl.store(dx_ptr_row, grad_block, mask=write_mask)


class QShiftFunctionOptimized(Function):
    @staticmethod
    def forward(ctx, input_tensor, shift_pixel, gamma):
        input_tensor = input_tensor.contiguous()
        B, C, H, W = input_tensor.shape
        C1, C2, C3, C4 = (int(C * gamma * i) for i in range(1, 5))
        output_tensor = torch.empty_like(input_tensor)
        
        grid = (B, C, H)
        
        stride_b, stride_c, stride_h, _ = input_tensor.stride()

        q_shift_fwd_kernel_optimized[grid](
            output_tensor, input_tensor,
            C, H, W,
            C1, C2, C3, C4,
            shift_pixel,
            stride_b, stride_c, stride_h
        )
        
        ctx.params = (C, H, W, C1, C2, C3, C4, shift_pixel, (stride_b, stride_c, stride_h))
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        C, H, W, C1, C2, C3, C4, shift_pixel, strides = ctx.params
        stride_b, stride_c, stride_h = strides
        B = grad_output.shape[0]

        grad_input = torch.zeros_like(grad_output)
        grid = (B, C, H)

        q_shift_bwd_kernel_optimized[grid](
            grad_input, grad_output,
            C, H, W,
            C1, C2, C3, C4,
            shift_pixel,
            stride_b, stride_c, stride_h
        )
        return grad_input, None, None

def q_shift_triton_optimized(input_tensor: torch.Tensor, shift_pixel: int = 1, gamma: float = 1/4):
    return QShiftFunctionOptimized.apply(input_tensor, shift_pixel, gamma)

# --------------------------------------------------------------------------
#                      2. 最终优化的 Triton 内核
# --------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # 遍历块大小、线程束数量和流水线阶段的组合
        triton.Config({'BLOCK_SIZE_W': bs}, num_warps=nw, num_stages=ns)
        for bs in [128, 256, 512, 1024]
        for nw in [2, 4, 8]
        for ns in [1, 2, 4, 5] # 增加流水线深度选项
    ],
    key=['W'], # 仍然以宽度W作为调优的关键参数
)
@triton.jit
def q_shift_fwd_kernel_final(
    output_ptr, input_ptr,
    C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    C1: tl.constexpr, C2: tl.constexpr, C3: tl.constexpr, C4: tl.constexpr,
    SHIFT_PIXEL: tl.constexpr,
    stride_b, stride_c, stride_h,
    BLOCK_SIZE_W: tl.constexpr
):
    pid_b, pid_c, pid_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    row_base_ptr = input_ptr + pid_b * stride_b + pid_c * stride_c + pid_h * stride_h
    output_row_ptr = output_ptr + pid_b * stride_b + pid_c * stride_c + pid_h * stride_h
    for w_start in range(0, W, BLOCK_SIZE_W):
        w_offsets = w_start + tl.arange(0, BLOCK_SIZE_W)
        src_h_offsets, src_w_offsets = pid_h, w_offsets
        if pid_c < C1: src_w_offsets = w_offsets - SHIFT_PIXEL
        elif pid_c < C2: src_w_offsets = w_offsets + SHIFT_PIXEL
        elif pid_c < C3: src_h_offsets = pid_h - SHIFT_PIXEL
        elif pid_c < C4: src_h_offsets = pid_h + SHIFT_PIXEL
        load_mask = (w_offsets < W) & (src_h_offsets >= 0) & (src_h_offsets < H) & (src_w_offsets >= 0) & (src_w_offsets < W)
        store_mask = w_offsets < W
        src_ptr = input_ptr + pid_b * stride_b + pid_c * stride_c + src_h_offsets * stride_h + src_w_offsets
        block = tl.load(src_ptr, mask=load_mask, other=0.0)
        tl.store(output_row_ptr + w_offsets, block, mask=store_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_W': bs}, num_warps=nw, num_stages=ns)
        for bs in [128, 256, 512, 1024]
        for nw in [2, 4, 8]
        for ns in [1, 2, 4, 5]
    ],
    key=['W'],
)
@triton.jit
def q_shift_bwd_kernel_final(
    dx_ptr, dy_ptr,
    C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    C1: tl.constexpr, C2: tl.constexpr, C3: tl.constexpr, C4: tl.constexpr,
    SHIFT_PIXEL: tl.constexpr,
    stride_b, stride_c, stride_h,
    BLOCK_SIZE_W: tl.constexpr
):
    pid_b, pid_c, pid_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    dy_row_ptr = dy_ptr + pid_b * stride_b + pid_c * stride_c + pid_h * stride_h
    for w_start in range(0, W, BLOCK_SIZE_W):
        w_offsets = w_start + tl.arange(0, BLOCK_SIZE_W)
        store_mask = w_offsets < W
        grad_block = tl.load(dy_row_ptr + w_offsets, mask=store_mask, other=0.0)
        dst_h_offsets, dst_w_offsets = pid_h, w_offsets
        if pid_c < C1: dst_w_offsets = w_offsets - SHIFT_PIXEL
        elif pid_c < C2: dst_w_offsets = w_offsets + SHIFT_PIXEL
        elif pid_c < C3: dst_h_offsets = pid_h - SHIFT_PIXEL
        elif pid_c < C4: dst_h_offsets = pid_h + SHIFT_PIXEL
        write_mask = (store_mask) & (dst_h_offsets >= 0) & (dst_h_offsets < H) & (dst_w_offsets >= 0) & (dst_w_offsets < W)
        dx_ptr_row = dx_ptr + pid_b * stride_b + pid_c * stride_c + dst_h_offsets * stride_h + dst_w_offsets
        tl.store(dx_ptr_row, grad_block, mask=write_mask)


class QShiftFunctionFinal(Function):
    @staticmethod
    def forward(ctx, input_tensor, shift_pixel, gamma):
        input_tensor = input_tensor.contiguous()
        B, C, H, W = input_tensor.shape
        C1, C2, C3, C4 = (int(C * gamma * i) for i in range(1, 5))
        output_tensor = torch.empty_like(input_tensor)
        grid = (B, C, H)
        stride_b, stride_c, stride_h, _ = input_tensor.stride()

        q_shift_fwd_kernel_final[grid](
            output_tensor, input_tensor,
            C, H, W, C1, C2, C3, C4, shift_pixel,
            stride_b, stride_c, stride_h
        )
        ctx.params = (C, H, W, C1, C2, C3, C4, shift_pixel, (stride_b, stride_c, stride_h))
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        C, H, W, C1, C2, C3, C4, shift_pixel, strides = ctx.params
        stride_b, stride_c, stride_h = strides
        B = grad_output.shape[0]
        grad_input = torch.zeros_like(grad_output)
        grid = (B, C, H)

        q_shift_bwd_kernel_final[grid](
            grad_input, grad_output,
            C, H, W, C1, C2, C3, C4, shift_pixel,
            stride_b, stride_c, stride_h
        )
        return grad_input, None, None

def q_shift_triton_final(input_tensor: torch.Tensor, shift_pixel: int = 1, gamma: float = 1/4):
    return QShiftFunctionFinal.apply(input_tensor, shift_pixel, gamma)