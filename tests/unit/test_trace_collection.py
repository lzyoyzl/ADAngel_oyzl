import json
import tempfile
import unittest
from pathlib import Path

import yaml

from adangel.trace.collector import build_corpus, select_prefill_tokens
from adangel.trace.raw import (
    RAW_MANIFEST_NAME,
    RAW_TRACE_FORMAT,
    RAW_TRACE_VERSION,
    sha256_file,
    token_ids_sha256,
    validate_raw_trace,
    validate_trace_config,
)


ROOT = Path(__file__).resolve().parents[2]


class TestTraceCollection(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load(
            (ROOT / "configs/trace/llama2_7b_prefill.yaml").read_text(
                encoding="utf-8"
            )
        )

    def test_formal_trace_config(self):
        layers, projections, sequence_length, expected_shape = validate_trace_config(
            self.config
        )
        self.assertEqual(layers, [0, 6, 12, 18, 24, 31])
        self.assertEqual(projections, ["q_proj", "k_proj", "v_proj", "o_proj"])
        self.assertEqual(sequence_length, 4096)
        self.assertEqual(expected_shape, (4096, 4096))

    def test_seeded_token_window_is_deterministic(self):
        corpus = build_corpus(["alpha", "", "beta"], "\n\n")
        self.assertEqual(corpus, "alpha\n\n\n\nbeta")
        token_ids = list(range(2, 10002))
        first, first_start = select_prefill_tokens(
            token_ids,
            bos_token_id=1,
            seed=20250805,
            sampled_tokens=4095,
        )
        second, second_start = select_prefill_tokens(
            token_ids,
            bos_token_id=1,
            seed=20250805,
            sampled_tokens=4095,
        )
        self.assertEqual(first, second)
        self.assertEqual(first_start, second_start)
        self.assertEqual(len(first), 4096)
        self.assertEqual(first[0], 1)
        self.assertEqual(first.count(1), 1)

    def _manifest(self, directory: Path) -> dict:
        token_ids = [1] + [2 + (index % 1000) for index in range(4095)]
        samples = []
        for layer in self.config["layers"]:
            for projection in self.config["projections"]:
                sample_id = f"layer_{layer:02d}_{projection}"
                path = directory / f"{sample_id}.pt"
                path.write_bytes(sample_id.encode("utf-8"))
                samples.append(
                    {
                        "sample_id": sample_id,
                        "file": path.name,
                        "layer": layer,
                        "projection": projection,
                        "sha256": sha256_file(path),
                        "activation": {
                            "shape": [4096, 4096],
                            "dtype": "torch.float16",
                        },
                        "weight": {
                            "shape": [4096, 4096],
                            "dtype": "torch.float16",
                        },
                    }
                )
        runtime = {
            "python": "3.10.12",
            "platform": "Linux-test",
            "torch": "2.7.1+cu128",
            "torch_cuda": "12.8",
            "cuda_available": True,
            "gpu": "NVIDIA H100 80GB HBM3",
            "gpu_total_memory_bytes": 80 * 1024**3,
            "compute_capability": [9, 0],
            "driver": "570.00",
            "transformers": "4.49.0",
            "datasets": "3.3.2",
            "accelerate": "1.2.0",
            "pyyaml": "6.0.2",
            "safetensors": "0.5.3",
            "sentencepiece": "0.2.0",
            "protobuf": "5.29.3",
        }
        return {
            "version": RAW_TRACE_VERSION,
            "format": RAW_TRACE_FORMAT,
            "trace": {
                "batch_size": 1,
                "sequence_length": 4096,
                "valid_tokens": 4096,
                "dtype": "fp16",
                "layers": self.config["layers"],
                "projections": self.config["projections"],
                "expected_shape": [4096, 4096],
            },
            "dataset": {
                "dataset": "Salesforce/wikitext",
                "subset": "wikitext-2-raw-v1",
                "split": "train",
                "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
                "fingerprint": "test-fingerprint",
                "corpus_joiner": "\n\n",
                "corpus_sha256": "0" * 64,
                "row_count": 1,
            },
            "tokenization": {
                "seed": 20250805,
                "start_index": 5,
                "sampled_tokens": 4095,
                "prepend_bos": True,
                "bos_token_id": 1,
                "corpus_token_count": 5000,
                "input_ids": token_ids,
                "input_ids_sha256": token_ids_sha256(token_ids),
                "attention_mask_all_ones": True,
            },
            "model": {
                "model_type": "llama",
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 32,
                "max_position_embeddings": 4096,
                "artifact_sha256": {
                    "config.json": "1" * 64,
                    "tokenizer.model": "2" * 64,
                    "model-00001-of-00002.safetensors": "3" * 64,
                },
            },
            "runtime": runtime,
            "inference": {
                "dtype": "fp16",
                "attention_implementation": "sdpa",
                "use_cache": False,
                "local_files_only": True,
                "base_model_only": True,
            },
            "environment_source": self.config["environment_source"],
            "samples": samples,
        }

    def test_metadata_only_validation_detects_transfer_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = self._manifest(directory)
            (directory / RAW_MANIFEST_NAME).write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            observed = validate_raw_trace(directory, self.config, deep=False)
            self.assertEqual(len(observed["samples"]), 24)

            first = directory / manifest["samples"][0]["file"]
            first.write_bytes(first.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                validate_raw_trace(directory, self.config, deep=False)


if __name__ == "__main__":
    unittest.main()
