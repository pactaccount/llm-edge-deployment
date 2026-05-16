"""
inference.py — Optimized 4-bit Inference Pipeline
===================================================
Loads Mistral-7B-Instruct in 4-bit NF4 quantization via bitsandbytes.
Reduces VRAM footprint from ~14GB (fp16) → ~4.5GB while preserving
output quality. Benchmarks token throughput and latency on every call.
"""

import time
import torch
import psutil
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextStreamer,
)

# ── Model Configuration ──────────────────────────────────────────────────────

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# 4-bit NF4 quantization config — the core of hardware-aware compression
QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4 — optimal for normally-distributed weights
    bnb_4bit_use_double_quant=True,     # Double quantization saves an additional ~0.4 bits/param
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bf16 for numerical stability
)

# ── Hardware Profiling Utils ──────────────────────────────────────────────────

def get_vram_usage_gb() -> float:
    """Returns current GPU VRAM usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 3)
    return 0.0


def get_ram_usage_gb() -> float:
    """Returns current system RAM usage in GB."""
    return psutil.Process().memory_info().rss / (1024 ** 3)


def get_hardware_info() -> dict:
    """Returns a summary of the current hardware state."""
    info = {
        "device": "CPU",
        "vram_used_gb": 0.0,
        "vram_total_gb": 0.0,
        "ram_used_gb": get_ram_usage_gb(),
    }
    if torch.cuda.is_available():
        info["device"] = torch.cuda.get_device_name(0)
        info["vram_used_gb"] = round(get_vram_usage_gb(), 2)
        info["vram_total_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2
        )
    return info


# ── Model Loader ─────────────────────────────────────────────────────────────

def load_quantized_model(model_id: str = MODEL_ID):
    """
    Loads the model in 4-bit NF4 quantization.
    
    Key engineering trade-off: We sacrifice ~2-3% accuracy on standard benchmarks
    (MMLU, HellaSwag) in exchange for a 3× reduction in memory footprint,
    enabling deployment on hardware that cannot hold the fp16 model in VRAM.
    """
    print(f"\n[Loading] {model_id}")
    print(f"[Hardware] {get_hardware_info()}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=QUANTIZATION_CONFIG,
        device_map="auto",          # Automatically places layers across available GPU/CPU
        trust_remote_code=True,
    )
    model.eval()

    post_load_info = get_hardware_info()
    print(f"[Post-Load VRAM] {post_load_info['vram_used_gb']} GB / {post_load_info['vram_total_gb']} GB")
    return model, tokenizer


# ── Inference Pipeline ────────────────────────────────────────────────────────

def build_prompt(user_message: str, system_prompt: str = None) -> str:
    """
    Constructs a Mistral-formatted instruction prompt.
    Mistral uses [INST] tags — no system-level prompt in base spec.
    """
    if system_prompt:
        return f"<s>[INST] {system_prompt}\n\n{user_message} [/INST]"
    return f"<s>[INST] {user_message} [/INST]"


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stream: bool = False,
) -> dict:
    """
    Runs inference and returns the response with latency and throughput metrics.
    
    Returns:
        dict with keys: response, latency_ms, tokens_per_second, tokens_generated
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_token_count = inputs["input_ids"].shape[1]

    streamer = TextStreamer(tokenizer, skip_prompt=True) if stream else None

    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

    end_time = time.perf_counter()

    # Extract only the generated tokens (exclude the input prompt)
    generated_ids = outputs[0][input_token_count:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    latency_ms = (end_time - start_time) * 1000
    tokens_generated = len(generated_ids)
    tokens_per_second = tokens_generated / (end_time - start_time)

    return {
        "response": response_text.strip(),
        "latency_ms": round(latency_ms, 1),
        "tokens_generated": tokens_generated,
        "tokens_per_second": round(tokens_per_second, 1),
        "vram_used_gb": round(get_vram_usage_gb(), 2),
    }


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4-bit Quantized Mistral-7B Inference")
    parser.add_argument("--prompt", type=str, default="Explain the difference between supervised and unsupervised learning in 3 sentences.")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--stream", action="store_true", help="Stream output token by token")
    args = parser.parse_args()

    model, tokenizer = load_quantized_model()
    prompt = build_prompt(args.prompt)

    print(f"\n[Prompt] {args.prompt}")
    print("[Generating...]\n" + "-" * 60)

    result = generate_response(
        model, tokenizer, prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        stream=args.stream,
    )

    if not args.stream:
        print(result["response"])

    print("-" * 60)
    print(f"[Latency]          {result['latency_ms']} ms")
    print(f"[Throughput]       {result['tokens_per_second']} tokens/sec")
    print(f"[Tokens Generated] {result['tokens_generated']}")
    print(f"[VRAM Used]        {result['vram_used_gb']} GB")
