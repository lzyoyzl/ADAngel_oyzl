from pathlib import Path
import runpy
import struct

ROOT = Path(__file__).resolve().parents[2]


def test_magic_bias_entire_o1_o3_bound_is_bitwise_exact():
    # O1's [-49152,49152] is a subset of this O3 conservative G128 bound.
    for partial in range(-131072, 131073):
        biased = struct.unpack('<f', struct.pack('<I', 0x4b400000+partial))[0]
        result = struct.pack('<f', biased-12582912.0)
        assert result == struct.pack('<f', float(partial))


def test_magic_candidates_do_not_change_production_or_scale_math():
    o1 = (ROOT/'csrc/sm120/o1_gemm.cu').read_text()
    o3 = (ROOT/'csrc/sm120/o3_gemm.cu').read_text()
    h = (ROOT/'include/adangel/exact_partial_cast.cuh').read_text()
    assert 'kProductionO1Implementation[] = "register_128x64_k64_scale_shared_row_dedup"' in o1
    assert 'kProductionO3Implementation = "m64_n32_k128_aligned_factor_16w"' in o3
    assert 'adangel_exact_partial_cast<kMagicCast, 32 * 128 * 12>' in o1
    assert 'adangel_exact_partial_cast<Config::kMagicCast, 128 * 128 * 8>' in o3
    assert '__fadd_rn(__int_as_float(0x4b400000 + partial), -12582912.0f)' in h
    assert 'AbsoluteBound < (1 << 22)' in h
    assert 'const float scale = __fmul_rn(row_scale, column_scale);' in o1
    assert 'Config::kFactorRowScaleAfterK' in o3
    # This legacy family does not derive from O3Config.
    assert 'struct O3SwizzledConfig {\n  static constexpr bool kMagicCast = false;' in o3


def test_magic_audit_classifies_only_exact_candidate_families():
    classify = runpy.run_path(str(ROOT/'scripts/audit_sm120_magic.py'))['classify']
    stem = 'adangel_o1_register_partial_128x64_k64_scale_shared_row_dedup'
    assert classify('_Z'+stem+'_magicILb1EE') == 'o1_magic'
    assert classify('_Z'+stem+'ILb1EE') == 'o1'
    assert classify('_Z'+stem+'_sparse_scaleILb1EE') is None
    assert classify('adangel_o1_probe') is None
    assert classify('adangel_o3_split_tma_ws_O3M64N32K128AlignedFactor16WMagicConfig') == 'o3_magic'
    assert classify('adangel_o3_split_tma_ws_O3M64N32K128AlignedFactor16WConfig') == 'o3'
    assert classify('adangel_o3_split_tma_ws_O3N16K128Config') is None


def test_magic_audit_preserves_producer_conversion_counts():
    counts = runpy.run_path(str(ROOT/'scripts/audit_sm120_magic.py'))['instruction_counts']
    old = counts('I2F.S16 R1; I2FP.F32.S32 R2; UTMALDG.2D R3; IMMA.16816.S8.S8 R4; I2FP.F32.S32 R5;')
    new = counts('I2F.S16 R1; I2FP.F32.S32 R2; UTMALDG.2D R3; IMMA.16816.S8.S8 R4; IADD3 R5; FADD R6;')
    assert old['I2FP'] == 2 and new['I2FP'] == 1 and new['I2F'] == 1
    assert old['UTMALDG'] == new['UTMALDG'] == 1
    # Instruction placement cannot identify a loop's execution order.
    reordered = counts('I2FP.F32.S32 R5; I2F.S16 R1; I2FP.F32.S32 R2; UTMALDG.2D R3; IMMA.16816.S8.S8 R4;')
    assert reordered == old


def test_shared_scale_has_post_store_release_and_matching_arrival_count():
    o1 = (ROOT/'csrc/sm120/o1_gemm.cu').read_text()
    o3 = (ROOT/'csrc/sm120/o3_gemm.cu').read_text()
    assert 'pipeline_params.num_producers = kShareColumnScale ? 33 : 1;' in o1
    assert 'pipeline_params.num_producers = 33;' in o1
    assert 'params.num_producers = 33;' in o3
    assert o1.count('cutlass::arch::ClusterBarrier::arrive(') == 2
    assert o3.count('cutlass::arch::ClusterBarrier::arrive(') == 1
    # Check all three producer-shared-scale paths. The empty-stage acquire
    # protects reuse; joined stores must precede the extra release and TMA.
    for source, storage in [
        (o1, 'shared_storage.column_scale_factor['),
        (o3, 'storage.column_scale['),
    ]:
        release = 'cutlass::arch::ClusterBarrier::arrive('
        start = 0
        while (pos := source.find(release, start)) >= 0:
            acquire = source.rfind('pipeline.producer_acquire(write_state)', 0, pos)
            assert 'if (lane == 0)' not in source[source.rfind('\n', 0, acquire):acquire]
            store = source.find(storage, acquire, pos)
            sync = source.rfind('__syncwarp();', acquire, pos)
            assert 0 <= acquire < store < sync < pos
            # Publication is before the lane0-only TMA branch, not inside it.
            assert source.find('if (lane == 0)', sync) > pos
            assert source.find('cute::copy(', pos) > pos
            start = pos + len(release)
    for source in [o1, o3]:
        assert 'per_writer_acquire_release_v3' in source
