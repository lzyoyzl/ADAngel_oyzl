"""Read-only replay: temporary partial lifetime is longer than MMA lifetime."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from inspect_scaled_partial_overlap import inspect, reduction_updates


def one_chain():
    return [dict(id=0, end_pc='0x0', mma=[dict(destination=4)])]


def scaled_ops():
    ops = [(0, 'IMMA.16864.U4.S4 R4, R20.ROW, R24.COL, R4')]
    # Each lane is row-scaled, then accumulated with a column factor.
    for vi in range(4):
        ops.extend([(16*(2*vi+1), f'IMAD R{8+vi}, R{4+vi}, R20, RZ'),
                    (16*(2*vi+2), f'IMAD R{12+vi}, R{8+vi}, R21.reuse, R{12+vi}')])
    return ops


def test_row_multiplication_is_not_final_accumulation():
    updates = reduction_updates(scaled_ops(), one_chain())
    assert [(u['chain'],u['value']) for u in updates] == [(0,i) for i in range(4)]
    assert [u['instruction_index'] for u in updates] == [2,4,6,8]


def test_single_factor_and_register_rename():
    ops = [(0,'IMMA.16864.U4.S4 R4, R20.ROW, R24.COL, R4'),
           (16,'MOV R8, R4.reuse'), (32,'IMAD R12, R8, R20, R12')]
    ops += [(48+16*i,f'IMAD R{13+i}, R{5+i}, R20, R{13+i}') for i in range(3)]
    assert len(reduction_updates(ops, one_chain())) == 4


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'overwrite', 'multiply_partials'])
def test_incomplete_or_unsupported_dataflow_rejected(mutation):
    ops = scaled_ops()
    if mutation == 'missing':
        ops.pop()
    elif mutation == 'duplicate':
        ops.append((160,'IMAD R16, R8, R21, R16'))
    elif mutation == 'overwrite':
        ops.insert(1,(8,'LDS.128 R4, [R30]'))
    else:
        ops[1] = (16,'IMAD R8, R4, R5, RZ')
    with pytest.raises(ValueError):
        reduction_updates(ops, one_chain())


@pytest.mark.parametrize('version,kind,stem,old_after,new_after,new_before', [
    (96,'o3','interleaved_tail',12,12,9),
    (96,'o78','interleaved_tail',16,10,2),
    (97,'o3','tail_lookahead',12,12,9),
    (97,'o78','tail_lookahead',16,0,0),
])
def test_archived_native_kernel_dataflow(version,kind,stem,old_after,new_after,new_before):
    directory = ROOT / f'docs/evidence/a100_o378_roof_v{version}/reports/o378_roof_v{version}_{kind}_codegen'
    receipt = json.loads((directory/'codegen.json').read_text())
    sass = (directory/f'{kind}_{stem}.sass').read_text()
    control = f'adangel_roof_{kind}_'+('grouped_cta_candidate' if kind=='o3' else 'eight_chain_candidate')
    candidate = f'adangel_roof_{kind}_{stem}_candidate'
    old = inspect(sass,control,receipt['liveness'][control])
    new = inspect(sass,candidate,receipt['liveness'][candidate])
    assert old['scaled_accumulator_updates'] == new['scaled_accumulator_updates'] == 64
    assert old['old_slice_updates_after_next_mma'] == old_after
    assert new['old_slice_updates_after_next_mma'] == new_after
    assert new['next_slice_mma_before_old_scale_drains'] == new_before
    assert old['scope'].startswith('static interleaving only')


@pytest.mark.parametrize('version,kind', [(96,'o3'),(96,'o78'),(97,'o3'),(97,'o78')])
def test_frozen_reaudit_exact_replay_and_input_hashes(version,kind):
    path = ROOT/f'docs/evidence/o378_scale_overlap_reaudit_20261008/v{version}_{kind}.json'
    receipt = json.loads(path.read_text())
    assert receipt['new_performance_result'] is False
    assert receipt['production_default_changed'] is False
    for name, digest in receipt['input_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    sources = list(receipt['input_sha256'])
    codegen = json.loads((ROOT/next(x for x in sources if x.endswith('codegen.json'))).read_text())
    sass = (ROOT/next(x for x in sources if x.endswith('.sass'))).read_text()
    for row in receipt['analyses']:
        assert inspect(sass,row['symbol'],codegen['liveness'][row['symbol']]) == row
