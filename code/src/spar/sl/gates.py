from __future__ import annotations

import math


def meets_minimum(value: float, minimum: float, *, abs_tol: float = 1e-12) -> bool:
    """Return whether value meets a threshold, robust to floating-point roundoff."""
    return value > minimum or math.isclose(value, minimum, rel_tol=0.0, abs_tol=abs_tol)


def decoder_layer(model, hidden_state_index: int):
    """Return a Qwen decoder layer from a base or PEFT-wrapped model."""
    if hidden_state_index <= 0:
        raise ValueError("Layer 0 is the embedding state, not a decoder layer")
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model.layers[hidden_state_index - 1]


def component_module(model, layer: int, component: str):
    """Return one residual-producing transformer component."""
    if component not in {"attn", "mlp"}:
        raise ValueError(f"Unsupported component: {component}")
    name = "self_attn" if component == "attn" else "mlp"
    return getattr(decoder_layer(model, layer), name)


class FinalPromptChannelGate:
    """Temporarily zero selected component-output channels during prefill.

    Generation invokes the decoder once on the full prompt and then on one
    token at a time.  The gate deliberately acts only on that initial final
    prompt token, making it a causal activation intervention rather than a
    persistent edit to model weights or generated-token states.
    """

    def __init__(self, model, *, layer: int, component: str, indices: torch.Tensor | None):
        self.module = component_module(model, layer, component)
        self.indices = indices
        self.handle = None

    def __enter__(self):
        if self.indices is None:
            return self

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[1] <= 1:
                return output
            changed = hidden.clone()
            changed[:, -1, self.indices.to(hidden.device)] = 0
            return (changed, *output[1:]) if isinstance(output, tuple) else changed

        self.handle = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, *_exc):
        if self.handle is not None:
            self.handle.remove()
