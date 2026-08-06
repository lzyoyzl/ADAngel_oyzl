import pytest

from adangel.ops.extension import require_sm120_extension


@pytest.mark.sm120
def test_sm120_microscale_layout():
    native = require_sm120_extension()
    result = dict(native.verify_layout())
    assert result["passed"], result
    assert result["max_abs_error"] <= 1.0e-3
