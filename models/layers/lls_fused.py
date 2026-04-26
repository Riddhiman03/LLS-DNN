import torch
import triton
import triton.language as tl


@triton.jit
def fused_lls_forward_kernel(
    # Inputs
    latents_ptr,        # [B, H] — pre-pooled, flattened latents
    basis_ptr,          # [N_CLASSES, H] — your sin basis (cached)
    labels_ptr,         # [B] — int labels
    # Outputs
    loss_ptr,           # [B] — per-sample loss (we'll average outside)
    grad_latents_ptr,   # [B, H] — gradient w.r.t. latents (for backward)
    # Sizes
    B, H, N_CLASSES,
    temperature,
    # Block sizes (compile-time constants)
    BLOCK_H: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Fuses: F.normalize(latents) -> matmul(basis.T) -> CE loss -> backward to latents.

    One program (block) per sample in the batch.
    """
    pid = tl.program_id(0)
    if pid >= B:
        return

    # ─── 1. Load this sample's latent vector ───
    h_offsets = tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    latent = tl.load(latents_ptr + pid * H + h_offsets, mask=h_mask, other=0.0)

    # ─── 2. L2 normalize: latent = latent / ||latent|| ───
    norm_sq = tl.sum(latent * latent, axis=0)
    inv_norm = 1.0 / tl.sqrt(norm_sq + 1e-12)
    normed_latent = latent * inv_norm

    # ─── 3. Compute logits = normed_latent @ basis.T  (size N_CLASSES) ───
    # We compute logits in chunks of BLOCK_C
    # This produces logits[c] for all c in [0, N_CLASSES)
    # First pass: find max for numerical stability
    max_logit = -float('inf')
    for c_start in range(0, N_CLASSES, BLOCK_C):
        c_offsets = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offsets < N_CLASSES

        # basis is [N_CLASSES, H]; load BLOCK_C rows of size H
        # logits_chunk[c] = sum_h(normed_latent[h] * basis[c, h])
        # Loaded as a 2D tile [BLOCK_C, BLOCK_H]
        basis_tile = tl.load(
            basis_ptr + c_offsets[:, None] * H + h_offsets[None, :],
            mask=c_mask[:, None] & h_mask[None, :],
            other=0.0
        )
        logits_chunk = tl.sum(basis_tile * normed_latent[None, :], axis=1) / temperature
        max_logit = tl.maximum(max_logit, tl.max(tl.where(c_mask, logits_chunk, -float('inf'))))

    # Second pass: compute softmax denominator
    sum_exp = 0.0
    for c_start in range(0, N_CLASSES, BLOCK_C):
        c_offsets = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offsets < N_CLASSES
        basis_tile = tl.load(
            basis_ptr + c_offsets[:, None] * H + h_offsets[None, :],
            mask=c_mask[:, None] & h_mask[None, :],
            other=0.0
        )
        logits_chunk = tl.sum(basis_tile * normed_latent[None, :], axis=1) / temperature
        sum_exp += tl.sum(tl.where(c_mask, tl.exp(logits_chunk - max_logit), 0.0))

    log_z = max_logit + tl.log(sum_exp)

    # ─── 4. Cross-entropy loss ───
    label = tl.load(labels_ptr + pid)
    # We need logit at index `label`. Load it.
    label_basis = tl.load(basis_ptr + label * H + h_offsets, mask=h_mask, other=0.0)
    label_logit = tl.sum(label_basis * normed_latent, axis=0) / temperature
    loss = log_z - label_logit
    tl.store(loss_ptr + pid, loss)

    # ─── 5. Backward to normed_latent ───
    # d_loss/d_logits[c] = (softmax[c] - one_hot[c]) / temperature
    # d_loss/d_normed_latent = sum_c d_loss/d_logits[c] * basis[c, :]
    # Computed in chunks
    grad_normed = tl.zeros([BLOCK_H], dtype=tl.float32)
    for c_start in range(0, N_CLASSES, BLOCK_C):
        c_offsets = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offsets < N_CLASSES
        basis_tile = tl.load(
            basis_ptr + c_offsets[:, None] * H + h_offsets[None, :],
            mask=c_mask[:, None] & h_mask[None, :],
            other=0.0
        )
        logits_chunk = tl.sum(basis_tile * normed_latent[None, :], axis=1) / temperature
        softmax_chunk = tl.exp(logits_chunk - log_z)

        # subtract one-hot
        one_hot = tl.where(c_offsets == label, 1.0, 0.0)
        d_logits = (softmax_chunk - one_hot) / temperature
        d_logits = tl.where(c_mask, d_logits, 0.0)

        grad_normed += tl.sum(basis_tile * d_logits[:, None], axis=0)

    # ─── 6. Backward through normalize ───
    # d_normed_latent / d_latent = (I - normed_latent ⊗ normed_latent) * inv_norm
    # grad_latent = inv_norm * (grad_normed - normed_latent * (normed_latent · grad_normed))
    dot = tl.sum(normed_latent * grad_normed, axis=0)
    grad_latent = inv_norm * (grad_normed - normed_latent * dot)

    tl.store(grad_latents_ptr + pid * H + h_offsets, grad_latent, mask=h_mask)


# ─── Python-side autograd wrapper ────────────────────────────────────────
class FusedLLSLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, latents, basis, labels, temperature):
        """
        latents: [B, H] (already pooled, contiguous)
        basis:   [N_CLASSES, H] (cached, contiguous)
        labels:  [B] (long)
        """
        assert latents.is_cuda and basis.is_cuda and labels.is_cuda
        assert latents.is_contiguous() and basis.is_contiguous()

        B, H = latents.shape
        N_CLASSES = basis.shape[0]

        loss = torch.empty(B, device=latents.device, dtype=torch.float32)
        grad_latents = torch.empty_like(latents, dtype=torch.float32)

        # Pick block sizes (must be powers of 2)
        BLOCK_H = triton.next_power_of_2(H)
        BLOCK_C = min(triton.next_power_of_2(N_CLASSES), 32)

        fused_lls_forward_kernel[(B,)](
            latents, basis, labels,
            loss, grad_latents,
            B, H, N_CLASSES, temperature,
            BLOCK_H=BLOCK_H, BLOCK_C=BLOCK_C,
        )

        ctx.save_for_backward(grad_latents)
        return loss.mean()

    @staticmethod
    def backward(ctx, grad_output):
        (grad_latents,) = ctx.saved_tensors
        # grad_output is scalar (mean), so divide by batch and scale
        B = grad_latents.shape[0]
        return grad_latents * (grad_output / B), None, None, None


def fused_lls_loss(latents, basis, labels, temperature=1.0):
    """Drop-in replacement for normalize + matmul + cross_entropy."""
    return FusedLLSLoss.apply(latents, basis, labels, temperature)