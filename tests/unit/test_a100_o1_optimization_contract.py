from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_sm80_candidates_preserve_group_scale_and_register_partial():
    source = (ROOT / 'csrc/sm80/o1_optimized.cuh').read_text()
    assert 'SM80_16x8x32_S32S8S8S32_TN' in source
    assert 'Groups = K / 32' in source
    assert 'cute::clear(partial)' in source
    assert '__fmaf_rn(float(partial(i)),scale,acc(i))' in source
    assert 'float scales[Stages][Groups*N]' in source
    assert 'Swizzle<' in source
    assert 'cp.async.wait_group' in source
    assert 'partial[' not in source  # no shared/global partial array
    assert '__fmul_rn(rows(i),column)' in source
    assert '__float_as_uint(rows(i))+__float_as_uint(column)' in source
    host=(ROOT / 'csrc/sm80/o1_o3.cu').read_text()
    assert 'emin+wmin-128>=1' in host and 'emax+wmax-128<=254' in host


def test_sm80_candidates_do_not_replace_default_sm120_target():
    setup = (ROOT / 'setup.py').read_text()
    assert 'os.environ.get("ADANGEL_CUDA_TARGET", "sm120")' in setup
    assert '"csrc/sm80/o1_o3.cu"' in setup
    assert '"csrc/sm120/o1_gemm.cu"' in setup


def test_candidate_validation_is_bitwise_and_unfiltered():
    script = (ROOT / 'scripts/benchmark_a100_o1.py').read_text()
    assert 'torch.testing.assert_close(y,base,rtol=0,atol=0)' in script
    assert 'binary_sha256' in script
    assert "'timings_ms'" in script
    assert 'baseline_mse_vs_o0' in script
