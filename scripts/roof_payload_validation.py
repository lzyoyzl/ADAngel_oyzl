"""Bitwise layout checks, always outside CUDA Event measurement intervals."""


def payload_reorder_bytes_for_stage(meta, mode, stage):
    weight=stage=='weight_conversion' or (mode=='conversion_only' and stage=='total')
    activation=stage=='activation_conversion' or (mode=='conversion_only' and stage=='total')
    return (int(meta.get('weight_payload_reorder_traffic_bytes',0)) if weight else 0)+(int(meta.get('activation_payload_reorder_traffic_bytes',0)) if activation else 0)


def verify_grouped_payload(result, tune, natural_a, natural_w, natural_ws=None):
    if tune not in (41,42,43,44,45,46,47,48,49,50)+(51,52)+(53,54)+(55,56)+(57,58)+(59,60)+(61,62):
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
    assert meta['pipeline_stages']==(2 if tune in (41,43,45,47,49,51,53,55,57,59,62) else 3)
    if fused:
        assert meta['gemm_tune']==tune-2 and meta['conversion_kernels_per_operand']==1
        assert meta['natural_payload_export']=='diagnostic_inverse_layout_after_timing'
    assert meta['cta_tile']==[32 if tune in (57,58) else 64,64 if tune in (45,46) else 128,128] and meta['threads']==128
    if tune in (57,58):
        assert meta['accumulators_per_thread']==32 and meta['warp_layout']==[1,4]
        assert meta['launch_bounds_min_blocks']==4 and meta['stream_n_slice']==128
    if tune in (45,46):
        assert meta['accumulators_per_thread']==32 and meta['warp_layout']==[2,2]
        assert meta['launch_bounds_min_blocks']==4
    if tune in (47,48):
        assert meta['compile_time_ring_slots'] and meta['accumulators_per_thread']==64
        assert meta['launch_bounds_min_blocks']==3 and meta['warp_layout']==[2,2]
    if tune in (49,50):
        assert meta['phased_fragment_finish'] and meta['finish_batch_values_per_thread']==16
        assert not meta['compile_time_ring_slots'] and meta['accumulators_per_thread']==64
        assert meta['launch_bounds_min_blocks']==3 and meta['warp_layout']==[2,2]
    assert meta['fp32_reassociated']==(tune in (51,52,53,54,61)) and meta['product_window_groups']==0
    if tune in (51,52,53,54,61):
        assert meta['row_scale_in_epilogue'] and meta['cross_group_accumulator_dtype']=='fp32'
        assert meta['unscaled_fp32_bound_checked']
    if tune in (53,54,61):
        assert natural_ws is not None and natural_ws.dtype==torch.uint8
        actual_ws=result['converted_weight_scale']
        assert actual_ws.is_contiguous() and tuple(actual_ws.shape)==(groups,n)
        assert torch.equal(actual_ws,natural_ws.T.contiguous())
        assert meta['weight_scale_layout']=='group_major' and meta['weight_scale_reorder_bytes']==2*n*groups
        assert meta['conversion_kernels_per_operand'] is None
        assert meta['activation_conversion_kernels']==2 and meta['weight_conversion_kernels']==3
    if tune in (55,56,57,58,59,60,62):
        assert meta['scale_copy_async'] and meta['scale_copy_transaction_bytes']==16
        assert meta['scale_copy_alignment_bytes']==16
        assert meta['scale_copy_buffering']=='same_ring_slot_as_g128_payload'
        assert meta['async_scale_reference_tune']=={55:41,56:42,57:55,58:56,59:55,60:56,62:55}[tune]
        assert not meta['scale_copy_combined_panels'] and not meta['fp32_reassociated']
        assert meta['conversion_kernels_per_operand']==2
        assert meta['cta_tile']==([32,128,128] if tune in (57,58) else [64,128,128])
        assert meta['warp_layout']==([1,4] if tune in (57,58) else [2,2])
    if tune in (59,60,62):
        assert meta['dimension_addressing']=='host_bounded_uint32'
        assert meta['address_products_max']==2147483647
        assert meta['accumulators_per_thread']==64 and meta['launch_bounds_min_blocks']==3
    if tune in (61,62):
        assert meta['copy_cache_policy']=='ca' and meta['cache_policy_only_candidate']
        assert not meta['gemm_math_changed_vs_reference']
        assert meta['cache_policy_reference_tune']=={61:54,62:59}[tune]
    return {'payload_layout_bitwise_verified':True,
            **({'async_scale_metadata_verified':True} if tune in (55,56,57,58,59,60,62) else {}),
            **({'weight_scale_layout_bitwise_verified':True} if tune in (53,54,61) else {})}
