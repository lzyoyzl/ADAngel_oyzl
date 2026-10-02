from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_o78_warp_metadata_codegen import OLD, NEW, generated_header, eight_header


def test_only_copy_partition_changes_not_math_pipeline_or_epilogue():
    source=(ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    old=eight_header(source)
    new=generated_header(source).replace('o78_warp_metadata_experiment','o78_eight_chain_experiment')
    assert new.replace(NEW,OLD)==old
    with pytest.raises(ValueError):generated_header(source.replace(OLD,'changed'))


def test_every_factor_byte_written_once_by_a_complete_warp():
    a=[byte for lane in range(32) for byte in range(lane*8,lane*8+8)]
    w=[byte for tid in range(32,64) for byte in range((tid-32)*16,(tid-32)*16+16)]
    assert a==list(range(256)) and w==list(range(512))
    for rows in (64,128,4096):
        for start in range(0,rows,64):
            assert start*4+max(a)<rows*4
    for cols in (128,256,4096):
        for start in range(0,cols,128):
            assert start*4+max(w)<cols*4


def test_eight_byte_copy_uses_supported_ca_and_memory_clobber():
    src=(ROOT/'csrc/sm80/roof_o78_warp_metadata_probe.cu').read_text()
    assert 'cp.async.ca.shared.global [%0], [%1], 8;' in src
    assert ':"memory"' in src
    assert 'if(flag==1u)' in src and 'O78::o3_body' in src
    assert '__launch_bounds__(128,3)' in src
