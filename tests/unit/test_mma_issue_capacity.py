"""Capacity probes cannot be reported as a trace GEMM or an MSE improvement."""
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_a100_mma_issue_capacity import audit, summarize, SYMBOLS


def fake_sass():
    blocks=[]
    for mode,symbol in enumerate(SYMBOLS):
        body=[f'Function : {symbol}']
        for i in range(64):
            kind='U4' if mode==1 or (mode==2 and i>=32) else 'S4'
            body.append(f'/*{16*i:04x}*/ IMMA.16864.{kind}.S4 R32, R0.ROW, R8.COL, R32 ;')
        body.append('/*0400*/ @P0 BRA 0x0 ;')
        blocks.append('\n'.join(body))
    return '\n'.join(blocks)


def test_exact_native_shape_types_and_single_loop():
    result=audit(fake_sass())
    assert result[SYMBOLS[2]]['native_signed']==result[SYMBOLS[2]]['native_unsigned']==32
    assert result[SYMBOLS[0]]['loop']['instructions']==65
    for text in (fake_sass().replace('S4.S4','S8.S8',1),fake_sass().replace('BRA 0x0','BRA 0x1000',1)):
        with pytest.raises(ValueError):audit(text)


def test_operation_count_is_physical_not_effective_gemm_tops():
    rows=[dict(mode=i,symbol=s,checksum_passed=True,validation_checks=24,blocks=2048,threads=128,
               groups=256,mma_per_warp_group=64,active_ctas_per_sm=3,raw_ms=[2.]*200)
          for i,s in enumerate(SYMBOLS)]
    for r in summarize(rows):
        assert r['dynamic_mma_work']==134217728
        assert r['physical_int4_ops']==2199023255552
        assert r['normalized_32_group_ms']==.25
        assert r['measured_physical_tops']==pytest.approx(1099.511627776)
    rows[0]['active_ctas_per_sm']=4
    with pytest.raises(ValueError):summarize(rows)


def test_no_binding_default_or_trace_claim():
    text=(ROOT/'csrc/sm80/mma_issue_capacity_probe.cu').read_text()
    assert 'resident!=3 || attr.localSizeBytes!=0' in text
    assert 'if(prop.major!=8 || prop.minor!=0 || prop.multiProcessorCount!=108)' in text
    assert 'for(int groups:{1,32})' in text
    assert 'for(int slice=0;slice<2;++slice)' in text
    assert 'p[i][j]*=16u' in text  # Defined unsigned bit arithmetic for negative partials.
    assert 'if(got!=static_cast<uint32_t>(16*groups*value))' in text
    assert 'for(int i=0;i<8;++i) checksum+=p[i][0]' in text
