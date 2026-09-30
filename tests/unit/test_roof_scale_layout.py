"""Check benchmark allocation contract without a PyTorch installation.

GPU numerical and stride validation is covered by the synthetic campaign.
"""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from benchmark_a100_roof_candidates import group_major_scales


class RoofScaleLayoutTest(unittest.TestCase):
    def test_single_and_multiple_groups_request_explicit_stride(self):
        for rows in (64, 128):
            for groups in (1, 3, 32):
                value = SimpleNamespace(ndim=2, shape=(rows, groups),
                                        device="cuda", dtype="float32")
                storage = Mock()
                storage.copy_.return_value = storage
                allocate = Mock(return_value=storage)
                with patch.dict(sys.modules, {"torch": SimpleNamespace(empty_strided=allocate)}):
                    result = group_major_scales(value)
                allocate.assert_called_once_with((rows, groups), (1, rows),
                                                 device="cuda", dtype="float32")
                storage.copy_.assert_called_once_with(value)
                self.assertIs(result, storage)

    def test_reject_non_matrix(self):
        with patch.dict(sys.modules, {"torch": SimpleNamespace()}):
            with self.assertRaises(ValueError):
                group_major_scales(SimpleNamespace(ndim=1))


if __name__ == "__main__":
    unittest.main()
