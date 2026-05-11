import logging

import torch

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger("flag_gems." + __name__)


def _grouped_mm(
    self,
    mat2,
    offs=None,
    bias=None,
    out_dtype=None,
):
    """
    Perform grouped matrix multiplication.

    For 3D inputs (batch, M, K) x (batch, K, N), this delegates to bmm.
    For 2D inputs without offsets, this delegates to mm/addmm.
    For 2D inputs with offsets, this uses a loop over bmm operations.

    Args:
        self: First tensor of shape (total_M, K) or (batch, M, K)
        mat2: Second tensor of shape (K, total_N) or (batch, K, N)
        offs: Optional offsets tensor specifying group boundaries
        bias: Optional bias to add
        out_dtype: Optional output dtype

    Returns:
        Result tensor
    """
    logger.debug("METAX GEMS GROUPED_MM")

    # Handle the simple case: treat as bmm when no offsets provided
    # This covers the case where inputs are 3D (batch, M, K) and (batch, K, N)
    if self.dim() == 3 and mat2.dim() == 3:
        batch, M, K = self.shape
        _, _, N = mat2.shape

        logger.debug(
            "METAX GEMS GROUPED_MM (3D case): [shape info]: [%s, %s, %s, %s](batch, M, N, K)",
            batch,
            M,
            N,
            K,
        )

        self = self.contiguous()
        mat2 = mat2.contiguous()

        if bias is not None:
            # Broadcast bias for each batch element
            if bias.dim() == 1:
                # bias shape: (N,) -> expand to (batch, N)
                bias = bias.unsqueeze(0).expand(batch, -1)
            elif bias.dim() == 2:
                # bias shape: (batch, N)
                pass
            bias = bias.contiguous()

        if out_dtype is not None:
            out = torch.empty((batch, M, N), dtype=out_dtype, device=self.device)
        else:
            out = torch.empty((batch, M, N), dtype=self.dtype, device=self.device)

        # Use bmm for each group
        with torch_device_fn.device(self.device):
            from flag_gems.ops.bmm import bmm as gems_bmm

            out = gems_bmm(self, mat2)

            if bias is not None:
                out = out + bias

        return out

    # Handle the 2D case with offsets (grouped mm with explicit groups)
    # This is the case where self is (total_M, K), mat2 is (K, total_N)
    # and offs specifies where each group starts
    if self.dim() == 2 and mat2.dim() == 2:
        if offs is None:
            # No offsets provided, treat as single group (regular mm)
            logger.debug("METAX GEMS GROUPED_MM (2D no offs): using mm")
            # Use torch.mm directly for reference comparison
            # Note: In production, this could delegate to gems_mm if available
            self = self.contiguous()
            mat2 = mat2.contiguous()
            if bias is not None:
                if out_dtype is not None:
                    out = torch.full((self.shape[0], mat2.shape[1]), 0, dtype=out_dtype, device=self.device)
                else:
                    out = torch.zeros((self.shape[0], mat2.shape[1]), dtype=self.dtype, device=self.device)
                out = torch.addmm(bias, self, mat2, out=out)
            else:
                if out_dtype is not None:
                    out = torch.empty((self.shape[0], mat2.shape[1]), dtype=out_dtype, device=self.device)
                else:
                    out = torch.empty((self.shape[0], mat2.shape[1]), dtype=self.dtype, device=self.device)
                out = torch.mm(self, mat2, out=out)
            return out

        # With offsets - this is the true grouped_mm case
        # Use a loop over bmm operations for each group
        logger.debug("METAX GEMS GROUPED_MM (2D with offs): using loop over bmm")

        num_groups = offs.numel() + 1
        K = self.shape[1]

        # Calculate M and N for each group
        M_sizes = []
        N_sizes = []
        prev_off = 0
        for off in offs.tolist():
            M_sizes.append(off - prev_off)
            prev_off = off
        M_sizes.append(self.shape[0] - prev_off)

        prev_off = 0
        for off in offs.tolist():
            N_sizes.append(off - prev_off)
            prev_off = off
        N_sizes.append(mat2.shape[1] - prev_off)

        # Allocate output
        total_M = self.shape[0]
        total_N = mat2.shape[1]

        if out_dtype is not None:
            out = torch.empty((total_M, total_N), dtype=out_dtype, device=self.device)
        else:
            out = torch.empty((total_M, total_N), dtype=self.dtype, device=self.device)

        # Make inputs contiguous
        self = self.contiguous()
        mat2 = mat2.contiguous()

        # Compute offsets for each group
        a_offsets = [0]
        b_offsets = [0]
        o_offsets = [0]
        for i in range(1, num_groups):
            a_offsets.append(a_offsets[-1] + M_sizes[i - 1] * K)
            b_offsets.append(b_offsets[-1] + N_sizes[i - 1] * K)
            o_offsets.append(o_offsets[-1] + M_sizes[i - 1] * N_sizes[i - 1])

        # Use bmm for each group
        from flag_gems.ops.bmm import bmm as gems_bmm

        for i in range(num_groups):
            M_i = M_sizes[i]
            N_i = N_sizes[i]

            # Extract slices for this group
            a_i = self[a_offsets[i] : a_offsets[i] + M_i * K].view(M_i, K)
            b_i = mat2[b_offsets[i] : b_offsets[i] + K * N_i].view(K, N_i)

            # Compute bmm for this group
            out_i = gems_bmm(a_i.unsqueeze(0), b_i.unsqueeze(0)).squeeze(0)

            # Copy to output
            out[o_offsets[i] : o_offsets[i] + M_i * N_i] = out_i.view(M_i, N_i).T.ravel()

        if bias is not None:
            out = out + bias

        return out

    # Fallback: use bmm
    logger.debug("METAX GEMS GROUPED_MM fallback to bmm")
    if self.dim() == 2:
        # Reshape to (1, M, K)
        self = self.unsqueeze(0)
    if mat2.dim() == 2:
        mat2 = mat2.unsqueeze(0)

    batch = self.shape[0]
    if bias is not None:
        if bias.dim() == 1:
            bias = bias.unsqueeze(0).expand(batch, -1)
        elif bias.dim() == 2:
            pass

    from flag_gems.ops.bmm import bmm as gems_bmm

    with torch_device_fn.device(self.device):
        out = gems_bmm(self, mat2)

        if bias is not None:
            out = out + bias

    return out.squeeze(0) if out.shape[0] == 1 else out