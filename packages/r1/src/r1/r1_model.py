import os
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def download_and_load_model(
    visible_gpus: int | list[int] = 0,
    device_index: int = 0,
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    cache_dir: str | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer, str]:
    """Load Model, Tokenizer, and Device for R1 Model."""
    # Constrain GPU Usage to Specific Device
    visible_gpus = (
        ",".join("x" for x in visible_gpus) if isinstance(visible_gpus, list) else str(visible_gpus)
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus
    if torch.cuda.is_available():
        device = f"cuda:{device_index}"
    else:
        warnings.warn("No GPU available. Running on CPU.")
        device = "cpu"

    # Load tokenizer and model
    # (Note: we directly load to GPU device 0 that is visible by pytorch,
    # and we load from /remote/shared/huggingface shared cache)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    cache_dir = cache_dir or os.environ["HF_HOME"]
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map={"": device_index}, cache_dir=cache_dir
    )
    return model, tokenizer, device
