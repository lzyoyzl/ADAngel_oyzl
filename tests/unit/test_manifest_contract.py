import copy
import unittest

from adangel.trace.storage import validate_manifest


def formal_manifest():
    layers = [0, 6, 12, 18, 24, 31]
    projections = ["q_proj", "k_proj", "v_proj", "o_proj"]
    samples = []
    for layer in layers:
        for projection in projections:
            sample_id = f"layer_{layer:02d}_{projection}"
            samples.append(
                {
                    "sample_id": sample_id,
                    "file": f"{sample_id}.pt",
                    "sha256": "0" * 64,
                    "shape": [4096, 4096, 4096],
                    "layer": layer,
                    "projection": projection,
                }
            )
    return {
        "version": 2,
        "format": "adangel-prepared-mxfp4-k32",
        "matrix_shape": [4096, 4096, 4096],
        "dtypes": {
            "A_int8": "torch.int8",
            "A_scale": "torch.float32",
            "W_mxfp4": "torch.uint8",
            "W_scale": "torch.uint8",
        },
        "quantization": {
            "activation": "int8_symmetric_per_row",
            "weight": "mxfp4_e2m1_ue8m0_k32",
            "rounding": "round_ties_to_even",
        },
        "trace": {
            "source": "external_fp16_prefill_trace",
            "batch_size": 1,
            "valid_tokens": 4096,
            "layers": layers,
            "projections": projections,
        },
        "samples": samples,
    }


def extended_formal_manifest():
    manifest = copy.deepcopy(formal_manifest())
    manifest["version"] = 3
    manifest["format"] = "adangel-prepared-mxfp4-k32-g128-q4"
    manifest["dtypes"].update(
        {
            "W_mxfp4_g128": "torch.uint8",
            "W_scale_g128": "torch.uint8",
            "W_q4": "torch.uint8",
        }
    )
    manifest["quantization"]["arbitrary_bit_weight"] = (
        "mxfp4_e2m1_ue8m0_k128_to_q4_rne"
    )
    return manifest


class TestManifestContract(unittest.TestCase):
    def test_formal_manifest(self):
        validate_manifest(formal_manifest(), formal=True)
        validate_manifest(
            extended_formal_manifest(), formal=True, require_arbitrary_bits=True
        )

    def test_legacy_manifest_rejected_for_o3_o4(self):
        with self.assertRaisesRegex(ValueError, "extended"):
            validate_manifest(
                formal_manifest(), formal=True, require_arbitrary_bits=True
            )

    def test_missing_sample_is_rejected(self):
        manifest = formal_manifest()
        manifest["samples"].pop()
        with self.assertRaisesRegex(ValueError, "exactly the 24"):
            validate_manifest(manifest, formal=True)

    def test_unsafe_filename_is_rejected(self):
        manifest = copy.deepcopy(formal_manifest())
        manifest["samples"][0]["file"] = "../sample.pt"
        with self.assertRaisesRegex(ValueError, "filename"):
            validate_manifest(manifest, formal=True)


if __name__ == "__main__":
    unittest.main()
