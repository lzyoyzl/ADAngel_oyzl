import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests/unit"))
from test_manifest_contract import extended_formal_manifest

spec = importlib.util.spec_from_file_location("mixed_trace_runner", ROOT / "scripts/benchmark_a100_mixed_trace.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MixedTraceRunnerTests(unittest.TestCase):
    def make_fixture(self, directory):
        manifest = extended_formal_manifest()
        for entry in manifest["samples"]:
            raw = entry["sample_id"].encode()  # Hash-only fixture, deliberately NOT torch tensors.
            (directory/entry["file"]).write_bytes(raw)
            entry["sha256"] = hashlib.sha256(raw).hexdigest()
        (directory/"manifest.json").write_text(json.dumps(manifest))
        return manifest

    def test_exact_files_and_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest = self.make_fixture(directory)
            checked, digest = module.inspect_inputs(directory)
            self.assertEqual(checked, manifest)
            self.assertEqual(len(digest), 64)
            extra = directory/"extra.pt"
            extra.write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "file set"):
                module.inspect_inputs(directory)
            extra.unlink()
            (directory/manifest["samples"][0]["file"]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                module.inspect_inputs(directory)

    def test_missing_file_and_escaping_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)/"input"
            directory.mkdir()
            manifest = self.make_fixture(directory)
            path = directory/manifest["samples"][0]["file"]
            content = path.read_bytes()
            path.unlink()
            with self.assertRaisesRegex(ValueError, "file set"):
                module.inspect_inputs(directory)
            target = Path(temp)/"outside.pt"
            target.write_bytes(content)
            path.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "path/hash mismatch"):
                module.inspect_inputs(directory)

    def test_execution_requires_explicit_data_policy(self):
        result = subprocess.run([sys.executable, str(ROOT/"scripts/benchmark_a100_mixed_trace.py"),
                                 "--data", "nonexistent_data_should_not_be_read"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires explicit --allow-secondary-quantization", result.stderr)

    def test_original_and_secondary_flags_are_exclusive(self):
        result = subprocess.run([sys.executable, str(ROOT/"scripts/benchmark_a100_mixed_trace.py"),
            "--data", "nonexistent", "--raw-data", "nonexistent",
            "--allow-secondary-quantization"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("not allowed with argument", result.stderr)

    def test_raw_manifest_link_and_deep_validation_boundary(self):
        # Isolate the hash/link/path boundary; this is not a fake tensor-deep test.
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest = {"samples": [{"sample_id": "s", "file": "s.pt"}]}
            path = directory/"trace_manifest.json"
            path.write_text(json.dumps(manifest))
            (directory/"s.pt").write_bytes(b"hash-only fixture")
            prepared = {"source_trace": {"manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
                        "samples": [{"sample_id": "s"}]}
            fake = ModuleType("adangel.trace.raw")
            fake.RAW_MANIFEST_NAME = "trace_manifest.json"
            fake.validate_raw_trace = mock.Mock(return_value=manifest)
            with mock.patch.dict(sys.modules, {"adangel.trace.raw": fake}):
                checked, digest = module.inspect_raw_inputs(directory, prepared, "config")
                self.assertEqual(checked, manifest)
                self.assertEqual(digest, prepared["source_trace"]["manifest_sha256"])
                fake.validate_raw_trace.assert_called_once_with(directory.resolve(), "config", deep=True)
                wrong = copy.deepcopy(prepared)
                wrong["source_trace"]["manifest_sha256"] = "0"*64
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    module.inspect_raw_inputs(directory, wrong, "config")
                wrong = copy.deepcopy(prepared)
                wrong["samples"][0]["sample_id"] = "another"
                with self.assertRaisesRegex(ValueError, "identities"):
                    module.inspect_raw_inputs(directory, wrong, "config")
                (directory/"s.pt").unlink()
                (directory/"s.pt").symlink_to(path.parent.parent)
                with self.assertRaisesRegex(ValueError, "escapes"):
                    module.inspect_raw_inputs(directory, prepared, "config")

    def records(self):
        result = []
        for sid, a, b, error in (("a", [10, 100], [5, 100], 7.), ("b", [20, 40], [10, 20], 9.)):
            for r in range(2):
                for case, latency, mse in (("o0", a[r], 0.), ("o5", a[r]*2, 0.), ("o7", b[r], error)):
                    result.append({"sample_id": sid, "round": r, "case": case, "mode": "compute_only",
                        "summary": {"gemm": {"median_ms": latency, "mean_ms": latency*1.1}},
                        "mse_vs_o0": mse, "timing_stable_cv3": True, "total_timing": "single_execution_cuda_event"})
        return result

    def test_rounds_are_collapsed_before_sample_summary(self):
        result = next(r for r in module.summarize_trace(self.records()) if r["case"] == "o7")
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["records"], 4)
        self.assertEqual(result["paired_speedup_vs_o0_median"], 1.75)
        self.assertEqual(result["paired_speedup_median_ci95"], [1.5, 2.])
        self.assertEqual(result["median_mse_vs_o0"], 8.)
        self.assertEqual(result["mean_mse_vs_o0"], 8.)

    def test_duplicate_mismatched_methods_and_changed_mse_fail(self):
        records = self.records()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            module.summarize_trace(records + [records[0]])
        mismatched = copy.deepcopy(records)
        mismatched[1]["total_timing"] = "different"
        with self.assertRaisesRegex(ValueError, "timing methods"):
            module.summarize_trace(mismatched)
        mismatched = copy.deepcopy(records)
        mismatched[1]["mse_vs_o0"] = 99
        with self.assertRaisesRegex(ValueError, "MSE changed"):
            module.summarize_trace(mismatched)

    def test_different_input_policies_are_not_aggregated(self):
        records = self.records()
        for record in records:
            record["input_policy"] = module.RAW_INPUT_POLICY
        records[1]["input_policy"] = module.INPUT_POLICY
        with self.assertRaisesRegex(ValueError, "input policies"):
            module.summarize_trace(records)


if __name__ == "__main__":
    unittest.main()
