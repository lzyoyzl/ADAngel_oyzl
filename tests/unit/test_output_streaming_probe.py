"""v99 is a store-policy experiment, not another MMA/payload-layout change."""
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from probe_output_streaming_codegen import generated_header, prior_memory_evidence, worth_runtime
from probe_grouped_cta_codegen import generated_headers
from probe_o78_eight_chain_codegen import generated_header as eight_header


def test_only_integer_epilogue_changes():
    root = Path(__file__).resolve().parents[2]
    for kind in ('o3', 'o78'):
        old = generated_headers('o3')[0] if kind == 'o3' else eight_header(
            (root/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
        new = generated_header(kind)
        boundary = '  auto final_value=' if kind == 'o3' else '  // Reuse the 64 INT32'
        old_namespace = 'o3_grouped_cta_experiment' if kind == 'o3' else 'o78_eight_chain_experiment'
        assert old.split(boundary)[0].replace(old_namespace, kind+'_output_streaming_experiment') == new.split(boundary)[0]
        assert new.count('__stcs(') == 3
        assert '__fmul_rn' in new and 'float(acc(i))' in new
        assert 'st.global' not in new  # use the documented compiler intrinsic


def test_authoritative_ldsm_already_conflict_free():
    for kind in ('o3', 'o78'):
        evidence = prior_memory_evidence(kind)
        assert evidence['executed_ldsm'] == 4194304
        assert evidence['ldsm_excess_wavefronts'] == 0
        assert evidence['executed_stg'] > 0
        assert not evidence['output_streaming_already_present']


def test_fixed_compile_budget_rejects_extra_work():
    loop = dict(kind='integer', max_live_gpr=166, static_instructions=383,
                opcode_counts={'IMMA.16864.S4.S4':32, 'IMMA.16864.U4.S4':32, 'LDSM.16.M88.4':16})
    control = dict(allocated_gpr=168, loops=[loop])
    stores = dict(control=dict(total=96, streaming=0), candidate=dict(total=96, streaming=32))
    assert worth_runtime(control, deepcopy(control), stores)
    for field, value in (('allocated_gpr', 169),):
        candidate = deepcopy(control)
        candidate[field] = value
        assert not worth_runtime(control, candidate, stores)
    for field in ('max_live_gpr', 'static_instructions'):
        candidate = deepcopy(control)
        candidate['loops'][0][field] += 1
        assert not worth_runtime(control, candidate, stores)
    candidate = deepcopy(control)
    candidate['loops'][0]['opcode_counts']['LDL.64'] = 1
    assert not worth_runtime(control, candidate, stores)
    candidate = deepcopy(control)
    candidate['loops'][0]['opcode_counts']['LDSM.16.M88.4'] = 17
    assert not worth_runtime(control, candidate, stores)
    extra = deepcopy(stores)
    extra['candidate']['total'] += 1
    assert not worth_runtime(control, control, extra)
