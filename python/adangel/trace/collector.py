"""Collect selected Llama-2-7B projection inputs and weights from one FP16 prefill."""

from __future__ import annotations

from pathlib import Path


def collect_trace(model_name: str, text_path: str | Path, output_dir: str | Path, config: dict) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    layers = [int(x) for x in config["layers"]]
    projections = list(config["projections"])
    sequence_length = int(config["sequence_length"])
    expected = tuple(config["expected_shape"])
    if config["batch_size"] != 1 or sequence_length != 4096 or expected != (4096, 4096):
        raise ValueError("this experiment requires batch=1, sequence=4096 and 4096x4096 matrices")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text = Path(text_path).read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=sequence_length)
    if encoded["input_ids"].shape != (1, sequence_length):
        raise ValueError(f"prompt must tokenize to at least {sequence_length} valid tokens")
    if "attention_mask" in encoded and int(encoded["attention_mask"].sum()) != sequence_length:
        raise ValueError("prompt contains padding; exactly 4096 valid tokens are required")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).eval().cuda()
    handles = []

    def make_hook(layer_index: int, projection: str):
        def hook(module, args):
            activation = args[0].detach()
            if activation.ndim != 3 or activation.shape[:2] != (1, sequence_length):
                raise ValueError(f"unexpected activation shape {tuple(activation.shape)}")
            activation = activation[0].to(dtype=torch.float16, device="cpu").contiguous()
            weight = module.weight.detach().to(dtype=torch.float16, device="cpu").contiguous()
            if tuple(activation.shape) != expected or tuple(weight.shape) != expected:
                raise ValueError(
                    f"layer {layer_index} {projection}: expected {expected}, "
                    f"got A={tuple(activation.shape)}, W={tuple(weight.shape)}"
                )
            sample_id = f"layer_{layer_index:02d}_{projection}"
            torch.save(
                {
                    "sample_id": sample_id,
                    "layer": layer_index,
                    "projection": projection,
                    "activation_fp16": activation,
                    "weight_fp16": weight,
                },
                output_dir / f"{sample_id}.pt",
            )

        return hook

    for layer_index in layers:
        attention = model.model.layers[layer_index].self_attn
        for projection in projections:
            handles.append(getattr(attention, projection).register_forward_pre_hook(make_hook(layer_index, projection)))

    try:
        with torch.inference_mode():
            model(**{k: v.cuda() for k, v in encoded.items()}, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    files = list(output_dir.glob("*.pt"))
    expected_count = len(layers) * len(projections)
    if len(files) != expected_count:
        raise RuntimeError(f"expected {expected_count} samples, found {len(files)}")
