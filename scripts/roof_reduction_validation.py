"""Numerical policy for explicitly opted-in reassociation experiments only.

No production default or old candidate bitwise gate is relaxed. References
use identical packed integers and *rounded FP32* G128 scale products. FP64
group dots are exact in our supported integer range; the FP64 final sum is a
high precision reference, not a claim of exact real arithmetic.
"""
import math


def mse_regression_ok(actual, baseline, rtol=1e-5, atol=1e-12):
    return math.isfinite(actual) and abs(actual-baseline) <= atol+rtol*abs(baseline)


def reference_fp64(variant, values):
    import torch
    packed_a, asc, packed_w, wsc = values

    def unpack(t, signed):
        q=torch.stack((t&15,t>>4),dim=-1).flatten(-2).to(torch.int16)
        return torch.where(q>=8,q-16,q) if signed else q

    m=packed_a.shape[0]//2
    a=unpack(packed_a[:m],False)+16*unpack(packed_a[m:],True)
    w=unpack(packed_w,True)
    if variant=='o3':
        from adangel.quantization.mxfp4 import decode_ue8m0_tensor
        wsc=decode_ue8m0_tensor(wsc)
        asc=asc[:,None].expand(-1,a.shape[1]//128)
    y=torch.zeros((m,w.shape[0]),device=a.device,dtype=torch.float64)
    for g in range(a.shape[1]//128):
        sl=slice(128*g,128*(g+1))
        partial=a[:,sl].double()@w[:,sl].double().T
        scale=(asc[:,g,None]*wsc[None,:,g]).float()
        y.add_(partial*scale.double())
    return y


def compare_output(y, baseline, reference, tune, kernel):
    import torch
    if y.dtype!=torch.float32 or not torch.isfinite(y).all():
        raise AssertionError('output must be finite FP32')
    changed=not torch.equal(y.view(torch.int32),baseline.view(torch.int32))
    reassociated=tune in (24,25,26,27,34,35,36)
    if not reassociated and changed:
        raise AssertionError('old candidate differs bitwise from production')
    if tune in (24,25,26,27):
        if (kernel.get('fp32_accumulation_chains')!=(2 if tune in (24,26) else 4)
                or not kernel.get('fp32_reassociated')):
            raise AssertionError('missing explicit reassociation metadata')
    if tune in (34,35,36):
        if (not kernel.get('fp32_reassociated') or not kernel.get('separate_rounded_products')
                or kernel.get('product_window_groups')!=(32 if tune==35 else 4)
                or kernel.get('eager_product_reduction')!=(tune==36)
                or kernel.get('group_accumulation')!='rounded_products_then_window_balanced_tree'):
            raise AssertionError('missing explicit product-tree metadata')
    torch.testing.assert_close(y.double(),reference,rtol=1e-3,atol=1e-3)
    delta=y.double()-baseline.double()
    error=y.double()-reference
    base_error=baseline.double()-reference
    return dict(bitwise_equal_production=not changed,
        mse_vs_production=delta.square().mean().item(),
        max_abs_vs_production=delta.abs().max().item(),
        changed_elements=int((delta!=0).sum().item()),
        mse_vs_semantic_fp64=error.square().mean().item(),
        baseline_mse_vs_semantic_fp64=base_error.square().mean().item(),
        max_abs_vs_semantic_fp64=error.abs().max().item(),
        semantic_tolerance_passed=True,fp32_reassociated=reassociated)
