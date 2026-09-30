"""Bitwise layout checks, always outside CUDA Event measurement intervals."""


def payload_reorder_bytes_for_stage(meta, mode, stage):
    weight=stage=='weight_conversion' or (mode=='conversion_only' and stage=='total')
    activation=stage=='activation_conversion' or (mode=='conversion_only' and stage=='total')
    return (int(meta.get('weight_payload_reorder_traffic_bytes',0)) if weight else 0)+(int(meta.get('activation_payload_reorder_traffic_bytes',0)) if activation else 0)


def verify_grouped_payload(result, tune, natural_a, natural_w):
    if tune not in (41,42,43,44,45,46,47,48):
        return {}
    import torch
    m=natural_a.shape[0]//2
    n,kbytes=natural_w.shape
    groups=kbytes//64
    expected_a=natural_a.reshape(2,m,groups,64).permute(0,2,1,3).contiguous()
    expected_w=natural_w.reshape(n,groups,64).permute(1,0,2).contiguous()
    actual_a=result['packed_activation_g128_major']
    actual_w=result['packed_weight_g128_major']
    assert actual_a.is_contiguous() and actual_w.is_contiguous()
    assert tuple(actual_a.shape)==(2,groups,m,64) and tuple(actual_w.shape)==(groups,n,64)
    assert torch.equal(actual_a,expected_a) and torch.equal(actual_w,expected_w)
    meta=result['kernel']
    assert meta['payload_layout']=='plane_group_row_k64_bytes'
    fused=tune in (43,44)
    assert meta['payload_reorder_in_conversion'] and meta['payload_reorder_fused']==fused
    assert meta['activation_payload_reorder_traffic_bytes']==(0 if fused else 4*m*kbytes)
    assert meta['weight_payload_reorder_traffic_bytes']==(0 if fused else 2*n*kbytes)
    assert meta['pipeline_stages']==(2 if tune in (41,43,45,47) else 3)
    if fused:
        assert meta['gemm_tune']==tune-2 and meta['conversion_kernels_per_operand']==1
        assert meta['natural_payload_export']=='diagnostic_inverse_layout_after_timing'
    assert meta['cta_tile']==[64,64 if tune in (45,46) else 128,128] and meta['threads']==128
    if tune in (45,46):
        assert meta['accumulators_per_thread']==32 and meta['warp_layout']==[2,2]
        assert meta['launch_bounds_min_blocks']==4
    if tune in (47,48):
        assert meta['compile_time_ring_slots'] and meta['accumulators_per_thread']==64
        assert meta['launch_bounds_min_blocks']==3 and meta['warp_layout']==[2,2]
    assert not meta['fp32_reassociated'] and meta['product_window_groups']==0
    return {'payload_layout_bitwise_verified':True}
