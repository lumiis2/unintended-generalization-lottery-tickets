from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def torch_dtype(name: str) -> torch.dtype:
    choices = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if name not in choices:
        raise ValueError(f"Unsupported dtype: {name}")
    return choices[name]


def load_tokenizer(config: dict[str, Any]):
    model_config = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"],
        revision=model_config.get("revision"),
        trust_remote_code=model_config.get("trust_remote_code", False),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(config: dict[str, Any], adapter: str | None = None, trainable: bool = False):
    model_config = config["model"]
    dtype = torch_dtype(model_config.get("dtype", "bfloat16"))
    kwargs: dict[str, Any] = {
        "revision": model_config.get("revision"),
        "trust_remote_code": model_config.get("trust_remote_code", False),
        "torch_dtype": dtype,
        "device_map": "auto",
    }
    if model_config.get("load_in_4bit", False):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_config["name"], **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    return model


def render_chat(tokenizer, user_prompt: str, system_prompt: str | None = None) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def batched_generate(model, tokenizer, prompts: list[str], generation: dict) -> list[str]:
    outputs: list[str] = []
    batch_size = generation["batch_size"]
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        generated = model.generate(
            **encoded,
            do_sample=True,
            temperature=generation["temperature"],
            top_p=generation["top_p"],
            max_new_tokens=generation["max_new_tokens"],
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return [output.strip() for output in outputs]

