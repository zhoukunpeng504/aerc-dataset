# coding:utf-8
# write by zkp
import datetime
import time

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from typing import *
import sys
import threading
import os

app = FastAPI()


class Query(BaseModel):
    messages: List[Dict[str, str]]
    model: Optional[str] = None
    request_id: Optional[str] = None
    do_sample: Optional[bool] = None
    stream: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None

# nf4_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_compute_dtype=torch.float16,
# )

model_path = sys.argv[-1].strip() #'outputgct/Qwen2.5-0.5B-Instruct-train-528-r-16/v0-20250530-120257/checkpoint-2600-merged/'

# cache_size 修改
if 'gemma' in model_path.lower():
    torch._dynamo.config.cache_size_limit = 32768


model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True,
)
tok = AutoTokenizer.from_pretrained(model_path,
                                    trust_remote_code=True)


@app.post("/v1/chat/completions")
async def generate_response(query: Query):
    iids = tok.apply_chat_template(
        query.messages,
        add_generation_prompt=1,
    )
    gen_cfg = model.generation_config.to_dict()
    print(gen_cfg)
    gen_cfg['max_length'] = 4096
    gen_cfg['max_new_tokens'] = 4096
    gen_cfg['repetition_penalty'] = 1
    gen_cfg['do_sample'] = False
    gen_cfg["top_p"] = 1
    gen_cfg["temperature"] = 0

    #gen_cfg['do_sample'] = True
    #print(gen_cfg)
    torch.manual_seed(42)
    oids = model.generate(
        inputs=torch.tensor([iids]).to(model.device),
        # seed=42,
        **gen_cfg,
    )
    oids = oids[0][len(iids):-1].tolist()
    output = tok.decode(oids)
    return {
        "choices": [{
            'index': 0,
            'message': {'role': 'assistant', 'content': output}
        }]
    }

if __name__ == "__main__":
    eval_port_info = os.environ.get("eval_port",'').strip()
    if eval_port_info:
        eval_port = int(eval_port_info)
    else:
        eval_port = 8000

    print("start serve...", datetime.datetime.now())
    uvicorn.run(app, host="0.0.0.0", port=eval_port)


