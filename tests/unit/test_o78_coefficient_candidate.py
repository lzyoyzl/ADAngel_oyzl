"""Exact v72 rewrite and isolation checks, not GPU or performance acceptance."""
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def test_guarded_coefficient_first_integer_rewrite():
    rng=np.random.default_rng(20261003)
    partial=rng.integers(-131072,131073,200000,dtype=np.int64)
    af=rng.integers(0,8192,200000,dtype=np.int64)
    wf=rng.integers(0,8192,200000,dtype=np.int64)
    coefficient=af*wf
    good=(coefficient<=2**31-1)&(np.abs(partial*coefficient)<=2**31-1)
    p,a,w=partial[good],af[good],wf[good]
    assert len(p)>1000
    assert np.array_equal((p*a)*w,p*(a*w))
    # Explicit zero/negative/endpoints, including a zero weight with huge A.
    for p,a,w in ((-1,2**31-1,1),(1,1,2**31-1),(0,2**31-1,1),
                  (-131072,2**31-1,0),(-131072,1,16383)):
        assert abs(p*(a*w))<=2**31-1
        assert p*(a*w)==(p*a)*w


def test_entire_body_unchanged_except_opaque_coefficient():
    old=(ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    new=(ROOT/'csrc/sm80/o78_coefficient_first_probe.cuh').read_text()
    left=new.index('            const int row_factor=')
    right=new.index('            // Host proves coefficient',left)
    old_left=old.index('            const int coefficient=')
    old_right=old.index('            // Host proves coefficient',old_left)
    restored=new[:left]+old[old_left:old_right]+new[right:]
    restored=restored.replace('o78_coefficient_first_experiment','o78_fullk_integer_experiment')
    assert restored[restored.index('namespace '):]==old[old.index('namespace '):]
    block=new[left:right]
    assert 'asm("mul.lo.s32 %0, %1, %2;"' in block
    assert '"r"(row_factor), "r"(column_factor)' in block
    assert 'volatile(' not in block and '"memory"' not in block


def test_compile_inputs_exist_and_old_default_untouched():
    sys.path.insert(0,str(ROOT/'scripts'))
    from probe_o78_coefficient_codegen import SOURCE_NAMES
    assert all((ROOT/name).is_file() for name in SOURCE_NAMES)
    source=(ROOT/'csrc/sm80/roof_o78_coefficient_probe.cu').read_text()
    assert source.count('if(flag>1u) return;')==2
    assert source.count('if(flag==1u)')==2
    assert source.count('O78::o3_body<64,128,128')==2
    assert 'o78_fullk_integer_experiment::body' in source
    assert 'o78_coefficient_first_experiment::body' in source
    for path in (ROOT/'csrc').rglob('*'):
        if path.is_file() and path.suffix in ('.cpp','.cu','.cuh','.h') and path.name!='roof_o78_coefficient_probe.cu':
            assert 'adangel_roof_o78_coefficient_' not in path.read_text(errors='replace'),path


def test_dependency_analysis_tracks_register_versions():
    from analyze_o78_coefficient_codegen import dependencies
    text = [
        'LDS R16, [R1]', 'LDS.64 R28, [R2]',
        'IMMA.16864.U4.S4 R56, R68.ROW, R100.COL, RZ',
        'IMMA.16864.S4.S4 R84, R80.ROW, R100.COL, RZ',
        'IMAD R30, R16, R28, RZ',
        'LEA R56, R84, R56, 0x4',
        'IMAD R11, R56, R30, R11',
        'IMAD R30, R57, R16, RZ',
        'IMAD R12, R30, R29, R12',
    ]
    result = dependencies([(i * 16, t.split()[0], t) for i, t in enumerate(text)])
    assert len(result['independent_coefficient_multiplies']) == 1
    assert len(result['partial_first_multiplies']) == 1
    assert len(result['coefficient_based_accumulator_updates']) == 1


def test_both_runtime_policies_keep_candidate_gpu_preparation(monkeypatch):
    from types import SimpleNamespace
    import benchmark_o78_coefficient_probe as probe
    calls = []
    def call(*args):
        calls.append(args)
        for i in range(len(args[-1])):
            args[-1][i] = 1.0
        return 0
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda: SimpleNamespace(cuda_stream=7)))
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    driver = object.__new__(probe.Driver)
    driver.handles = {0: 11, 1: 12, 2: 13}
    driver.lib = SimpleNamespace(roof_o78_gpu_benchmark=call)
    case = SimpleNamespace(variant='o7', oracle={'status_flat': np.array([0])}, a_source=1,
        w_source=2, state_pointers=3, m=64, n=128, a_multiplier=4.0, w_multiplier=1.0, state={'y': 'output'})
    for policy in (0, 1, 2):
        _, timings = driver.run(case, policy, 'compute_only', 5, 2, 100)
        assert timings == {'gemm': [1.0, 1.0], 'total': [1.0, 1.0]}
    assert [c[0] for c in calls] == [11, 12, 13]
    assert [c[2] for c in calls] == [1, 1, 1]
