"""Memory-bounded preparation of an already quantized, frozen base model."""


def prepare_frozen_kbit_model(model):
    """Freeze the base, retain large BF16 embeddings, upcast only small tensors.

    PEFT's general preparation upcasts all non-quantized tensors, including the
    billion-parameter embeddings and output head, creating a >13 GiB transient.
    This explicit variant preserves those frozen tensors in BF16, uses float32
    for small norms/biases and enables non-reentrant activation checkpointing.
    Adapter creation happens afterwards and makes only LoRA weights trainable.
    """
    import torch
    if not getattr(model, "is_loaded_in_4bit", False):
        raise ValueError("expected an already 4-bit-loaded base model")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        if (parameter.is_floating_point() and parameter.numel() < 1_000_000
                and parameter.dtype in (torch.float16, torch.bfloat16)):
            parameter.data = parameter.data.float()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model
