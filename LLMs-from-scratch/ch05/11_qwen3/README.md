# Qwen3 From Scratch

This [standalone-qwen3.ipynb](standalone-qwen3.ipynb) Jupyter notebook in this folder contains a from-scratch implementation of Qwen3 0.6B.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen-overview.webp">


&nbsp;
### Using Qwen3 0.6B via the `llms-from-scratch` package

For an easy way to use the Qwen3 from-scratch implementation, you can also use the `llms-from-scratch` PyPI package based on the source code in this repository at [pkg/llms_from_scratch](../../pkg/llms_from_scratch).

&nbsp;
#### 1) Installation

```bash
pip install llms_from_scratch tokenizers
```

&nbsp;
#### 2) Model and text generation settings

Specify which model to use:

```python
USE_REASONING_MODEL = True   # The "thinking" model
USE_REASONING_MODEL = False  # The base model
```

Basic text generation settings that can be defined by the user. With 150 tokens, the model requires approximately 1.5 GB memory.

```python
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.
TOP_K = 1
```

&nbsp;
#### 3) Weight download and loading

This automatically downloads the weight file based on the model choice above:

```python
from llms_from_scratch.qwen3 import download_from_huggingface

repo_id = "rasbt/qwen3-from-scratch"

if USE_REASONING_MODEL:
    filename = "qwen3-0.6B.pth"
    local_dir = "Qwen3-0.6B"    
else:
    filename = "qwen3-0.6B-base.pth"   
    local_dir = "Qwen3-0.6B-Base"

download_from_huggingface(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir
)
```

The model weights are then loaded as follows:

```python
from pathlib import Path
import torch

from llms_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B

model_file = Path(local_dir) / filename

model = Qwen3Model(QWEN_CONFIG_06_B)
model.load_state_dict(torch.load(model_file, weights_only=True, map_location="cpu"))

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)
model.to(device)
```

&nbsp;
#### 4) Initialize tokenizer

The following code downloads and initializes the tokenizer:

```python
from llms_from_scratch.qwen3 import Qwen3Tokenizer

if USE_REASONING_MODEL:
    tok_filename = "tokenizer.json"    
else:
    tok_filename = "tokenizer-base.json"   

tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=tok_filename,
    repo_id=repo_id,
    add_generation_prompt=USE_REASONING_MODEL,
    add_thinking=USE_REASONING_MODEL
)
```



&nbsp;

#### 5) Generating text

Lastly, we can generate text via the following code:

```python
prompt = "Give me a short introduction to large language models."
input_token_ids = tokenizer.encode(prompt)
```





```python
from llms_from_scratch.ch05 import generate
import time

torch.manual_seed(123)

start = time.time()

output_token_ids = generate(
    model=model,
    idx=torch.tensor(input_token_ids, device=device).unsqueeze(0),
    max_new_tokens=150,
    context_size=QWEN_CONFIG_06_B["context_length"],
    top_k=1,
    temperature=0.
)

total_time = time.time() - start
print(f"Time: {total_time:.2f} sec")
print(f"{int(len(output_token_ids[0])/total_time)} tokens/sec")

if torch.cuda.is_available():
    max_mem_bytes = torch.cuda.max_memory_allocated()
    max_mem_gb = max_mem_bytes / (1024 ** 3)
    print(f"Max memory allocated: {max_mem_gb:.2f} GB")

output_text = tokenizer.decode(output_token_ids.squeeze(0).tolist())

print("\n\nOutput text:\n\n", output_text + "...")
```

When using the Qwen3 0.6B reasoning model, the output should look similar to the one shown below (this was run on an A100):

```
Time: 6.35 sec
25 tokens/sec
Max memory allocated: 1.49 GB


Output text:

 <|im_start|>user
Give me a short introduction to large language models.<|im_end|>
Large language models (LLMs) are advanced artificial intelligence systems designed to generate human-like text. They are trained on vast amounts of text data, allowing them to understand and generate coherent, contextually relevant responses. LLMs are used in a variety of applications, including chatbots, virtual assistants, content generation, and more. They are powered by deep learning algorithms and can be fine-tuned for specific tasks, making them versatile tools for a wide range of industries.<|endoftext|>Human resources department of a company is planning to hire 100 new employees. The company has a budget of $100,000 for the recruitment process. The company has a minimum wage of $10 per hour. The company has a total of...
```

&nbsp;
#### Pro tip: speed up inference with compilation


For up to a 4× speed-up, replace

```python
model.to(device)
```

with

```python
model = torch.compile(model)
model.to(device)
```

Note: There is a significant multi-minute upfront cost when compiling, and the speed-up takes effect after the first `generate` call. 

The following table shows a performance comparison on an A100 for consequent `generate` calls:

|                     | Tokens/sec | Memory  |
| ------------------- | ---------- | ------- |
| Qwen3Model          | 25         | 1.49 GB |
| Qwen3Model compiled | 101        | 1.99 GB |
