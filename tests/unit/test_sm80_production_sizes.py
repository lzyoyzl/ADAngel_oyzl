"""Size specializations must retain the accepted algorithm and actual K."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
DIR = ROOT / 'csrc/sm80/production_sized_generated'


def test_group_count_is_the_only_gemm_body_change():
    for name, old_ns, new_ns in (
        ('o3.cuh', 'o3_grouped_cta_experiment', 'o3_sized_grouped_cta'),
        ('o78.cuh', 'o78_output_streaming_experiment', 'o78_sized_output_streaming'),
    ):
        original = (ROOT / 'csrc/sm80/production_generated' / name).read_text()
        sized = (DIR / name).read_text()
        # Compare executable tokens, ignoring comments/whitespace.
        import re
        clean = lambda s: re.sub(r'\s+', '', re.sub(r'//[^\n]*', '', s))
        restored = sized.replace(new_ns, old_ns).replace('template<int Groups>\n', '')
        if name == 'o3.cuh':
            restored = restored.replace('const int groups=Groups;', 'const int groups=32;')
            restored = restored.replace('[Groups*n+', '[32*n+')
        else:
            restored = restored.replace('prefetch(s,0,0,a,w,af,wf,m,n,k);',
                'constexpr int Groups=32;prefetch(s,0,0,a,w,af,wf,m,n,k);')
        assert clean(restored) == clean(original)


def test_conversion_masks_and_neutral_metadata():
    for name in ('nv4.cuh', 'mx8.cuh', 'hif4.cuh', 'nv6.cuh'):
        body = (DIR / name).read_text()
        assert 'if(g<groups)' in body
        assert body.index('if(g<groups)') < body.index('src_group=row*groups+g')
        assert 'square_by_group[threadIdx.x]=0;code_by_group[threadIdx.x]=0' in body
        assert 'if(unsigned(t)<groups) decode<' in body
        assert 'if(unsigned(t)<groups) factors[t*rows+row]=' in body
        assert body.count('__syncthreads()') == 2


def test_size_dispatch_metadata_and_profile_filters():
    from benchmark_a100_mixed import profile_spec
    host = (ROOT / 'csrc/sm80/production_sized_benchmark.cuh').read_text()
    assert '(k==512 || k==1024)' in host and 'using ::production::measure' in host
    assert 'physical_k' in host and 'legacy_shape_fallback' in host
    assert 'groups=k/128' in host and 'groups+1,n' in host
    assert 'api::o3_gemm' in host and 'api::mixed_gemm' in host
    gemm = (ROOT / 'csrc/sm80/production_sized_gemm.cu').read_text()
    assert 'O3_ENTRY(512,4)' in gemm and 'MIXED_ENTRY(1024,8)' in gemm
    for k in (512, 1024):
        for case, symbol in [('o3', 'adangel_sm80_o3_fullk_grouped'),
                             ('o7/64x128x256', 'adangel_sm80_o78_fullk_streaming'),
                             ('o8/64x128x256', 'adangel_sm80_o78_fullk_streaming')]:
            assert profile_spec(case, 50, k)['kernel_filter'] == f'regex:{symbol}_k{k}'
    source = (ROOT / 'csrc/sm80/o1_o3.cu').read_text()
    assert source.index('#include "production_sized_api.h"') < source.index('namespace {')
    assert 'return production::o3(' in source and 'return production_sized::o3(' in source
    assert 'no_empty_stage_events_compute_total_equals_gemm' in source
    assert 'const bool size_timing=!split' in source and '(k==512 || k==1024)' in source
