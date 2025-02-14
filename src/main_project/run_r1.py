# %%
import os
from typing import Annotated

import torch
import typer
from r1.r1_model import download_and_load_model
from utils import get_function_status_string, load_environment, send_notification


def main(
    input_text: Annotated[str, typer.Option(help="Text input to LLM.")] = None,
    model_name: Annotated[str, typer.Option(help="Huggingface LLM Model Name.")] = None,
    visible_gpus: Annotated[
        list[int], typer.Option(help="GPU device indices that are visible to script.")
    ] = None,
    device_index: Annotated[
        int, typer.Option(help="Index of device to load GPU model onto.")
    ] = None,
) -> str:
    load_environment()

    # Default Values
    input_text = input_text or "How many 'r' letters are there in the word 'strawberry'?"
    model_name = model_name or "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    visible_gpus = visible_gpus or [0]
    device_index = device_index or 0

    # Load tokenizer and model
    model, tokenizer, device = download_and_load_model(
        model_name=model_name, visible_gpus=visible_gpus, device_index=device_index
    )

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Generate output
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id
        )
    # Decode output tokens to text
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    # Send Notification
    msg = get_function_status_string(filename=__file__)
    send_notification(title="Job Completed!", message=msg, url=os.environ["SLACK_WEBHOOK_URL"])

    return output_text


# %%
if __name__ == "__main__":
    typer.run(main)
