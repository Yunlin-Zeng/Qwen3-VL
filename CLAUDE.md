# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the Qwen3-VL repository - a powerful vision-language model supporting image and video understanding. The repo contains:
- **Inference/Demo**: Web UI demo for interactive testing
- **Fine-tuning**: Complete training framework in `qwen-vl-finetune/`
- **Utilities**: Vision processing tools in `qwen-vl-utils/`
- **Cookbooks**: Jupyter notebooks demonstrating various capabilities (OCR, grounding, video understanding, etc.)
- **Evaluation**: Benchmarking tools in `evaluation/`

## Model Architecture

Qwen3-VL supports multiple model variants:
- **Dense models**: Qwen3-VL (standard architecture)
- **MoE models**: Qwen3-VL-MoE (Mixture of Experts, identified by "A" in model name like "30B-A3B")
- **Model types**: Determined by model path name:
  - `qwen3` → Qwen3VL (MoE if "a" in name)
  - `qwen2.5` → Qwen2.5-VL
  - Default → Qwen2-VL

The model has three trainable components:
1. **Vision Encoder** (`visual`) - Processes images/videos
2. **MLP/Merger** (`visual.merger`) - Bridges vision and language
3. **Language Model** (`language_model` + `lm_head`) - Generates text

## Common Commands

### Web Demo
```bash
# Install dependencies
pip install -r requirements_web_demo.txt

# Launch web UI (opens at http://127.0.0.1:7860/)
python web_demo_mm.py -c /path/to/model/weight

# Docker alternative
cd docker && bash run_web_demo.sh -c /path/to/model/weight --port 8881
```

### Inference

**Basic inference setup:**
```bash
pip install git+https://github.com/huggingface/transformers
pip install qwen-vl-utils==0.0.14
```

**vLLM serving (recommended for production):**
```bash
# Install
pip install accelerate
pip install qwen-vl-utils==0.0.14
uv pip install -U vllm  # Requires vllm>=0.11.0

# Start server (example for H100/H200 with FP8 model)
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --enable-expert-parallel \
  --async-scheduling \
  --host 0.0.0.0 \
  --port 22002
```

### Fine-tuning

**Setup:**
```bash
cd qwen-vl-finetune/
```

**Dataset configuration:**
1. Edit `qwenvl/data/__init__.py` to add your dataset:
```python
YOUR_DATASET = {
    "annotation_path": "/path/to/annotations.json",
    "data_path": "/path/to/images/",  # Can be empty if paths are absolute
}
data_dict = {
    "your_dataset": YOUR_DATASET,
    # ... existing datasets
}
```

2. Data format must follow:
   - One `<image>` tag = one image file
   - One `<video>` tag = one video file
   - Tags only in questions, never in answers

**Training:**
```bash
# Basic training command
torchrun --nproc_per_node=$NUM_GPUS \
         qwenvl/train/train_qwen.py \
         --model_name_or_path /path/to/model \
         --dataset_use your_dataset%100 \
         --output_dir ./checkpoints \
         --tune_mm_llm True \
         --tune_mm_vision False \
         --tune_mm_mlp True \
         --bf16 \
         --per_device_train_batch_size 4 \
         --gradient_accumulation_steps 4 \
         --learning_rate 2e-7 \
         --num_train_epochs 3 \
         --model_max_length 4096 \
         --max_pixels $((576*28*28)) \
         --min_pixels $((16*28*28)) \
         --deepspeed scripts/zero3.json

# See scripts/sft.sh, scripts/sft_7b.sh, scripts/sft_32b.sh for more examples
```

**Key training flags:**
- `--data_flatten True`: Concatenate batch sequences into one sequence
- `--data_packing True`: Use pre-packed data (requires `tools/pack_data.py` preprocessing)
- `--tune_mm_vision False`: **Required when training on both image and video data**
- `--tune_mm_mlp True/False`: Train the vision-language projection layer
- `--tune_mm_llm True/False`: Train the language model backbone

**Important notes:**
- Training resolution (`--max_pixels`, `--min_pixels`) critically affects performance
- Suggested learning rate: 1e-6 to 2e-7
- For 32B model: requires 8x 80GB GPUs (see `scripts/sft_32b.sh`)
- MoE models (Qwen3VL-MoE): **Do not support DeepSpeed ZeRO-3**
- Flash Attention 2: Add `"_attn_implementation": "flash_attention_2"` to model's `config.json`

### Data Utilities

```bash
# Check data completeness (verify images exist)
python qwen-vl-finetune/tools/check_image.py

# Pack data for efficient training
python qwen-vl-finetune/tools/pack_data.py
```

## Training Data Format

All training data must be JSON/JSONL with this structure:

**Single image:**
```json
{
    "image": "path/to/image.jpg",
    "conversations": [
        {"from": "human", "value": "<image>\nWhat's in this picture?"},
        {"from": "gpt", "value": "A red apple on a wooden table"}
    ]
}
```

**Multiple images:**
```json
{
    "image": ["image1.jpg", "image2.jpg"],
    "conversations": [
        {"from": "human", "value": "<image>\n<image>\nCompare these images."},
        {"from": "gpt", "value": "The first shows..."}
    ]
}
```

**Video:**
```json
{
    "video": "path/to/video.mp4",
    "conversations": [
        {"from": "human", "value": "<video>\nDescribe this video."},
        {"from": "gpt", "value": "The video shows..."}
    ]
}
```

**Packed data (list of examples in one file):**
```json
[
    {"image": "1.jpg", "conversations": [...]},
    {"image": "2.jpg", "conversations": [...]}
]
```

## Project Structure

```
qwen-vl-finetune/
├── qwenvl/
│   ├── train/
│   │   ├── train_qwen.py      # Main training entry point
│   │   ├── trainer.py          # Custom HF Trainer with attention modifications
│   │   └── argument.py         # ModelArguments, DataArguments, TrainingArguments
│   └── data/
│       ├── __init__.py         # Dataset registry (data_dict)
│       ├── data_processor.py   # Data loading and preprocessing
│       └── rope2d.py            # RoPE positional encoding implementation
├── scripts/
│   ├── sft.sh                  # Example training script (3B model)
│   ├── sft_7b.sh              # 7B model training
│   ├── sft_32b.sh             # 32B model training (8x 80GB GPU)
│   └── zero3.json             # DeepSpeed ZeRO-3 config
└── tools/
    ├── check_image.py          # Verify dataset images exist
    └── pack_data.py            # Pack data into even-length buckets

qwen-vl-utils/
└── src/qwen_vl_utils/
    └── vision_process.py       # Image/video processing utilities

cookbooks/                      # Jupyter notebooks demonstrating capabilities
├── omni_recognition.ipynb      # Object recognition
├── document_parsing.ipynb      # Document understanding
├── 2d_grounding.ipynb          # Object detection
├── ocr.ipynb                   # Text extraction
├── video_understanding.ipynb   # Video analysis
├── mobile_agent.ipynb          # Mobile UI control
├── computer_use.ipynb          # Desktop automation
└── ...

evaluation/                     # Benchmarking tools
```

## Key Implementation Details

**Model Loading** (`qwenvl/train/train_qwen.py:92-134`):
- Automatically detects model type from path name
- Supports Qwen2-VL, Qwen2.5-VL, Qwen3-VL, and Qwen3-VL-MoE
- Uses flash_attention_2 by default for efficiency

**Parameter Freezing** (`qwenvl/train/train_qwen.py:67-90`):
- `set_model()` function controls which components are trainable
- Use `tune_mm_vision`, `tune_mm_mlp`, `tune_mm_llm` flags

**Data Flattening/Packing** (`qwenvl/train/train_qwen.py:141-142`):
- When enabled, replaces standard attention with custom implementation
- Allows concatenating multiple samples in one sequence for efficiency

**Dataset Sampling** (`qwenvl/data/__init__.py:45-49`):
- Append `%N` to dataset name for N% sampling (e.g., `"dataset%50"` = 50% of data)

## Dependencies

**Core requirements:**
- `torch==2.6.0`
- `torchvision==0.21.0`
- `transformers==4.57.0.dev0` (or `pip install git+https://github.com/huggingface/transformers`)
- `accelerate==1.7.0`

**Training:**
- `deepspeed==0.17.1`
- `flash_attn==2.7.4.post1`
- `triton==3.2.0`

**Inference:**
- `vllm>=0.11.0` (for fast serving)
- `qwen-vl-utils==0.0.14`

**Web demo:**
- `gradio==5.46.1`
- `transformers-stream-generator==0.0.5`

## Video Processing

**Backends:**
- Default: `torchvision` (slower, HTTPS supported if ≥0.19.0)
- Faster: `decord` (install with `pip install qwen-vl-utils[decord]`, HTTP only, has known issues)
- Recommended: `torchcodec` (fastest, HTTP/HTTPS support, requires FFmpeg)

Switch backends via environment variable: `FORCE_QWENVL_VIDEO_READER=torchcodec`

## Extended Context (YaRN)

For sequences >256K tokens, modify `config.json`:
```json
{
    "max_position_embeddings": 1000000,
    "rope_scaling": {
        "rope_type": "yarn",
        "mrope_section": [24, 20, 20],
        "mrope_interleaved": true,
        "factor": 3.0,
        "original_max_position_embeddings": 262144
    }
}
```

Note: Use smaller factor (2-3, not 4) for 1M context due to Interleaved-MRoPE's slower position ID growth.
