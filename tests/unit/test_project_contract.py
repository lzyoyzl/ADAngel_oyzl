import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


class TestProjectContract(unittest.TestCase):
    def test_formal_config(self):
        config = yaml.safe_load((ROOT / "configs/experiment/o0_o1_o2_4096.yaml").read_text())
        self.assertEqual([config["matrix"][x] for x in ("m", "n", "k")], [4096, 4096, 4096])
        self.assertEqual(config["output"], {"accumulator": "fp32", "dtype": "fp32"})
        self.assertEqual(config["timing"]["warmup"], 50)
        self.assertEqual(config["timing"]["repeats"], 200)
        self.assertEqual(config["backend"]["cuda_arch"], "sm_120a")

    def test_target_o2_instruction_is_present(self):
        source = (ROOT / "csrc/sm120/o2_microkernel.cu").read_text()
        self.assertIn("kind::mxf4.block_scale.scale_vec::2X", source)
        self.assertIn("m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0", source)

    def test_native_build_embeds_sm120a_sass_and_ptx(self):
        setup = (ROOT / "setup.py").read_text()
        self.assertIn(
            "-gencode=arch=compute_120a,code=[sm_120a,compute_120a]", setup
        )
        self.assertIn("tools/util/include/cutlass/util/packed_stride.hpp", setup)
        self.assertIn('str(cutlass_root / "tools/util/include")', setup)

    def test_o0_production_backend_contract(self):
        source = (ROOT / "csrc/sm120/o0_gemm.cu").read_text()
        bindings = (ROOT / "csrc/bindings.cpp").read_text()
        self.assertIn("CUBLAS_COMPUTE_32F", source)
        self.assertIn("CUDA_R_16F", source)
        self.assertIn("CUDA_R_32F", source)
        self.assertIn("CUBLASLT_NUMERICAL_IMPL_FLAGS_HMMA", source)
        self.assertIn("CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F", source)
        self.assertIn("CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16F", source)
        self.assertIn("adangel_launch_mxfp4_to_fp16", source)
        self.assertIn("adangel_launch_int8_to_fp16", source)
        self.assertIn('module.def("run_o0", &adangel_run_o0)', bindings)
        self.assertNotIn('result["o0_fp16_tc"] = false', bindings)

    def test_o1_production_backend_contract(self):
        source = (ROOT / "csrc/sm120/o1_gemm.cu").read_text()
        conversion = (ROOT / "csrc/sm120/conversion.cu").read_text()
        bindings = (ROOT / "csrc/bindings.cpp").read_text()
        self.assertIn("#include <mma.h>", source)
        self.assertIn("wmma::fragment<wmma::matrix_a", source)
        self.assertIn("wmma::fragment<wmma::matrix_b", source)
        self.assertIn("wmma::fragment<wmma::accumulator", source)
        self.assertIn("wmma::mma_sync", source)
        self.assertIn("PipelineTmaAsync<kPipelineStages>", source)
        self.assertIn("CUTE_GRID_CONSTANT TmaA const tma_a", source)
        self.assertIn("CUTE_GRID_CONSTANT TmaB const tma_b", source)
        self.assertNotIn("CUTLASS_GRID_CONSTANT", source)
        self.assertIn("extern __shared__ __align__(128) uint8_t shared_bytes[]", source)
        self.assertIn("reinterpret_cast<O1SharedStorage*>(shared_bytes)", source)
        self.assertIn("sizeof(O1SharedStorage), stream", source)
        self.assertIn("make_tma_atom", source)
        self.assertIn("producer_acquire", source)
        self.assertIn("consumer_wait", source)
        self.assertIn("consumer_release", source)
        self.assertIn("NamedBarrier::sync", source)
        self.assertIn("shared_storage.partial", source)
        self.assertIn("accumulators[kOutputsPerThread]", source)
        self.assertIn("adangel_o1_tma_warp_specialized", source)
        self.assertIn('result["implementation"] = "tma_warp_specialized"', source)
        self.assertIn('result["data_movement"] = "TMA"', source)
        self.assertIn('result["kernel_schedule"] = "cooperative_warp_specialized"', source)
        self.assertIn('result["global_partial_buffer"] = false', source)
        self.assertNotIn("cublasLtMatmul(", source)
        self.assertIn("adangel_launch_mxfp4_to_int8", source)
        self.assertIn("e2m1_to_int8_base", conversion)
        self.assertIn('module.def("run_o1", &adangel_run_o1)', bindings)
        self.assertNotIn('result["o1_int8_tc"] = false', bindings)

    def test_o2_production_backend_contract(self):
        source = (ROOT / "csrc/sm120/o2_cutlass.cu").read_text()
        bindings = (ROOT / "csrc/bindings.cpp").read_text()
        audit = (ROOT / "scripts/audit_instructions.sh").read_text()
        self.assertIn("cutlass::arch::OpClassBlockScaledTensorOp", source)
        self.assertIn("cutlass::gemm::KernelTmaWarpSpecializedCooperative", source)
        self.assertIn("StageCountAutoCarveout", source)
        self.assertIn("GemmUniversalAdapter", source)
        self.assertIn("#include <cutlass/util/packed_stride.hpp>", source)
        self.assertIn("Sm1xxBlockScaledConfig<kGroupSize>", source)
        self.assertIn("tile_atom_to_shape_SFA", source)
        self.assertIn("tile_atom_to_shape_SFB", source)
        self.assertIn("adangel_o2_quantize_activation", source)
        self.assertIn("adangel_o2_repack_scale", source)
        self.assertIn("encode_e2m1_rne", source)
        shuffle = source.index("const int next_lane_code =")
        even_lane_pack = source.index("if ((lane & 1) == 0)", shuffle)
        self.assertLess(shuffle, even_lane_pack)
        self.assertIn(
            "__shfl_down_sync(0xffffffffu, static_cast<int>(code), 1)",
            source[shuffle:even_lane_pack],
        )
        self.assertIn("gemm_operator.can_implement", source)
        self.assertIn("gemm_operator.initialize", source)
        self.assertIn('result["data_movement"] = "TMA"', source)
        self.assertIn(
            'result["kernel_schedule"] = "cooperative_warp_specialized"', source
        )
        self.assertIn('result["mma_family"] = "MXFP4_BLOCK_SCALED"', source)
        self.assertIn('result["global_partial_buffer"] = false', source)
        self.assertIn("kWeightScaleRepackTimingInnerRepeats = 100", source)
        self.assertIn('result["weight_scale_repack_timing_method"]', source)
        self.assertIn('result["weight_scale_repack_timing_isolated"] = true', source)
        self.assertIn('result["total_timing_semantics"]', source)
        self.assertIn("std::vector<EventPair> weight_repack_events", source)
        self.assertEqual(source.count("convert_weight_timing_batch();"), 1)
        self.assertIn(
            "weight_batch / static_cast<float>(kWeightScaleRepackTimingInnerRepeats)",
            source,
        )
        self.assertIn('module.def("run_o2", &adangel_run_o2)', bindings)
        self.assertIn(
            'result["o2_mxf4_block_scale"] = adangel_o2_cutlass_is_implemented()',
            bindings,
        )
        self.assertIn(
            'result["o2_cutlass_tiled"] = adangel_o2_cutlass_is_implemented()',
            bindings,
        )
        self.assertIn("o2_mxf4_layout_probe", audit)
        self.assertIn("has_tma && has_mma", audit)
        self.assertIn(
            r"/^[[:space:]]*(\.visible[[:space:]]+)?\.entry[[:space:]]/",
            audit,
        )
        self.assertIn(
            r"/^[[:space:]]*Function[[:space:]]*:/",
            audit,
        )

    def test_o2_has_all_timing_modes(self):
        source = (ROOT / "csrc/sm120/o2_cutlass.cu").read_text()
        for mode in ("conversion_only", "compute_only", "cold", "steady_state"):
            self.assertIn(f'"{mode}"', source)

    def test_o1_tma_consumer_output_ownership(self):
        coordinates = [
            (thread // 32 + item * 8, thread % 32)
            for thread in range(256)
            for item in range(8)
        ]
        self.assertEqual(len(coordinates), 64 * 32)
        self.assertEqual(len(set(coordinates)), 64 * 32)
        self.assertEqual(
            set(coordinates), {(row, column) for row in range(64) for column in range(32)}
        )
        for row, column in coordinates:
            owner_warp = (row // 16) * 2 + column // 16
            owner_index = (row % 16) * 16 + column % 16
            self.assertGreaterEqual(owner_warp, 0)
            self.assertLess(owner_warp, 8)
            self.assertGreaterEqual(owner_index, 0)
            self.assertLess(owner_index, 256)

    def test_o1_has_all_timing_modes(self):
        source = (ROOT / "csrc/sm120/o1_gemm.cu").read_text()
        for mode in ("conversion_only", "compute_only", "cold", "steady_state"):
            self.assertIn(f'"{mode}"', source)

    def test_o0_has_all_timing_modes(self):
        source = (ROOT / "csrc/sm120/o0_gemm.cu").read_text()
        for mode in ("conversion_only", "compute_only", "cold", "steady_state"):
            self.assertIn(f'"{mode}"', source)


if __name__ == "__main__":
    unittest.main()
