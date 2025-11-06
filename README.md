# Tiny Example of RL PPO 

## Install 

```bash
pip install "transformers>=4.44" "trl>=0.9.6" datasets accelerate bitsandbytes peft einops
```

## Run

### Titan X (12 GB)

```bash
python ppo_math_toy.py \
  --model_name distilgpt2 \
  --use_lora 0 \
  --batch_size 64 \
  --micro_bsz 4 \
  --ppo_epochs 2 \
  --total_ppo_steps 200 \
  --digits 2 \
  --max_new_tokens 32
```

### A100 — better capability (QLoRA on TinyLlama 1.1B)

```bash
python ppo_math_toy.py \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --use_lora 1 \
  --batch_size 128 \
  --micro_bsz 8 \
  --ppo_epochs 2 \
  --total_ppo_steps 300 \
  --digits 2 \
  --max_new_tokens 32
```

> Tip: Increase --digits to 3 for harder tasks (you’ll see reward/accuracy move more slowly). 
> If you see KL blow up or collapse, tweak --target_kl (e.g., 0.05–0.3) or lower --learning_rate.

## Showcase
