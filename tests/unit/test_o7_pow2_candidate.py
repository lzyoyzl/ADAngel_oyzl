"""v71 exact-integer arithmetic and isolation contracts; no GPU claims."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def test_unsigned_shift_equals_guarded_integer_product():
    rng = np.random.default_rng(20261003)
    count = 0
    for d in range(31):
        af = 1 << d
        for wf in {0,1,3,7,15,255,32767,(2**31-1)//af}:
            if af*wf > 2**31-1:
                continue  # This is exactly the existing coefficient guard.
            bound = min(131072,(2**31-1)//(af*wf)) if wf else 131072
            p = np.r_[np.array([-bound,-1,0,1,bound]),rng.integers(-bound,bound+1,1024)]
            p = p[np.abs(p)<=bound].astype(np.int64)
            weighted = p*wf
            assert np.all(np.abs(weighted)<=2**31-1)
            bits = weighted.astype(np.int32).view(np.uint32).astype(np.uint64)
            shifted = ((bits << d) & (2**32-1)).astype(np.uint32).view(np.int32)
            assert np.array_equal(shifted.astype(np.int64),p*wf*af)
            count += len(p)
    assert count > 100000


def test_guard_fallback_and_source_isolation():
    source = (ROOT/'csrc/sm80/roof_o7_pow2_probe.cu').read_text()
    assert '#include "o78_fullk_integer_probe.cuh"' in source
    assert '#include "o7_fullk_pow2_probe.cuh"' in source
    assert source.count('if(flag>1u) return;') == 2
    assert source.count('if(flag==1u)') == 2
    assert source.count('O78::o3_body<64,128,128') == 2
    assert 'o78_fullk_integer_experiment::body' in source
    assert 'o7_fullk_pow2_experiment::body' in source
    candidate = (ROOT/'csrc/sm80/o7_fullk_pow2_probe.cuh').read_text()
    assert 'row_shifts(ri,mi)=31-__clz(factor)' in candidate
    assert 'static_cast<unsigned>(weighted)' in candidate
    assert '__float_as_int(__uint_as_float(shifted))' in candidate
    assert 'const int coefficient=' not in candidate
    # Same single final write structure and original 2-stage pipeline.
    assert candidate.count('cp.async.wait_group 0') == 1
    assert candidate.count('prefetch(s,1-slot') == 1
    assert candidate.count('make_float2(') == 1


def test_no_formal_binding_or_default_edit():
    for path in (ROOT/'csrc').rglob('*'):
        if path.is_file() and path.suffix in ('.cu','.cuh','.cpp','.h') and path.name not in {
            'roof_o7_pow2_probe.cu','o7_fullk_pow2_probe.cuh'}:
            assert 'adangel_roof_o7_pow2_' not in path.read_text(errors='replace'),path


def test_archived_codegen_and_loop_analysis_reproduce():
    sys.path.insert(0,str(ROOT/'scripts'))
    from analyze_o7_pow2_codegen import fullk_loops
    evidence = ROOT/'docs/evidence/a100_o378_roof_v71/reports/o378_roof_v71_codegen'
    saved = json.loads((evidence/'codegen.json').read_text())
    assert saved['coordinate_mapping_passed']
    assert saved['control_opcode_counts_match_v67'] and saved['control_instruction_count_match_v67']
    assert not saved['production_default_changed']
    for name,sha in saved['sources'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    assert hashlib.sha256((evidence/'o7_pow2.cubin').read_bytes()).hexdigest()==saved['cubin_sha256']
    rows = fullk_loops((evidence/'o7_pow2.sass').read_text())
    assert rows == json.loads((evidence/'loop_analysis.json').read_text())['rows']
    by = {r['symbol'].rsplit('_',1)[1]:r for r in rows}
    assert (by['control']['instructions'],by['candidate']['instructions'])==(378,453)
    assert not by['control']['mainloop_local_loads']
    assert len(by['candidate']['mainloop_local_loads'])==4
    for r in rows:
        ops=r['opcode_counts']
        assert ops['IMMA.16864.U4.S4']==ops['IMMA.16864.S4.S4']==32
        assert ops['LDSM.16.M88.4']==16
