"""
CLI to run inference on a trained model
"""

from pathlib import Path

import fire
import transformers
from dotenv import load_dotenv
from typing import Optional, List

from axolotl.cli import (
    do_inference_cli,
    do_inference_gradio,
    load_cfg,
    print_axolotl_text_art,
    do_inference_api,
    api_create_model
)
from axolotl.common.cli import TrainerCliArgs
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class CompletionRequest(BaseModel):
    prompts: List[str]
    max_length: Optional[int] = None



def do_cli(
    config: Path = Path("examples/"), gradio: bool = False, chat: bool = False, api: bool = False, **kwargs
):
    # pylint: disable=duplicate-code
    print_axolotl_text_art()
    parsed_cfg = load_cfg(config, **kwargs)
    parsed_cfg.sample_packing = False
    parser = transformers.HfArgumentParser((TrainerCliArgs))
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    parsed_cli_args.inference = True

    if chat and not gradio:
        raise ValueError("Must use gradio for chat mode")

    app = FastAPI()

    model, tokenizer = None, None

    @app.post("/generate")
    async def generate_completion(request: CompletionRequest):
        try:
            completions = do_inference_api(request.prompts, request.max_length, tokenizer, model, cfg=parsed_cfg, cli_args=parsed_cli_args)
            return {"completions": completions}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    if gradio:
        do_inference_gradio(chat=chat, cfg=parsed_cfg, cli_args=parsed_cli_args)
    elif api:
        model, tokenizer = api_create_model(cfg=parsed_cfg, cli_args=parsed_cli_args)
        uvicorn.run(app, host="0.0.0.0", port=8080)
    else:
        do_inference_cli(cfg=parsed_cfg, cli_args=parsed_cli_args)


if __name__ == "__main__":
    load_dotenv()
    fire.Fire(do_cli)
