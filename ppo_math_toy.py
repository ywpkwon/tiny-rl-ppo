# ppo_math_toy.py
# Minimal PPO RL training for LLMs on a tiny arithmetic dataset.
# - default: distilgpt2 (fits 12GB Titan X)
# - optional: TinyLlama 1.1B with QLoRA (fits A100 nicely)

import os, math, random, argparse, json
from dataclasses import dataclass
from typing import List, Dict

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# --------------------------
# dataset: tiny arithmetic
# --------------------------
OPS = ["+", "-", "*"]

def gen_item(n_digits=2):
    op = random.choice(OPS)
    if op == "+":
        a = random.randint(0, 10**n_digits - 1)
        b = random.randint(0, 10**n_digits - 1)
        ans = a + b
    elif op == "-":
        a = random.randint(0, 10**n_digits - 1)
        b = random.randint(0, 10**n_digits - 1)
        ans = a - b
    else:
        a = random.randint(0, 10**n_digits - 1)
        b = random.randint(0, 10**(n_digits-1)) if n_digits>1 else random.randint(0,9)
        ans = a * b
    prompt = f"Question: {a} {op} {b}\nAnswer:"
    return {"prompt": prompt, "answer": str(ans)}

def build_dataset(n_train=2000, n_val=200, n_digits=2, seed=42):
    random.seed(seed)
    train = [gen_item(n_digits) for _ in range(n_train)]
    val   = [gen_item(n_digits) for _ in range(n_val)]
    return DatasetDict({
        "train": Dataset.from_list(train),
        "validation": Dataset.from_list(val)
    })

# --------------------------
# reward function
# --------------------------
def extract_last_int(s: str):
    # pull last integer in a response (simple but effective for toy math)
    tokens = []
    num = ""
    for ch in s:
        if ch.isdigit() or (ch == "-" and not num):
            num += ch
        else:
            if num:
                tokens.append(num)
                num = ""
    if num:
        tokens.append(num)
    # take last integer-like token
    if tokens:
        try:
            return int(tokens[-1])
        except:
            return None
    return None

def compute_reward(prompts: List[str], responses: List[str], answers: List[str], len_penalty=0.0):
    # +1 for exact-match numeric answer; small length penalty (per 20 tokens)
    rewards = []
    for p, r, gold in zip(prompts, responses, answers):
        guess = extract_last_int(r)
        correct = (guess is not None) and (str(guess) == gold)
        base = 1.0 if correct else 0.0
        lp = 0.0
        if len_penalty > 0.0:
            # crude length proxy: characters/80 ~ tokens/??? (ok for toy)
            lp = len_penalty * (len(r) / 80.0)
        rewards.append(base - lp)
    return rewards

# --------------------------
# evaluation
# --------------------------
@torch.no_grad()
def evaluate_accuracy(model, tokenizer, ds, gen_cfg, max_eval=200, device="cuda"):
    model.eval()
    n = min(max_eval, len(ds))
    correct = 0
    for i in range(n):
        prompt = ds[i]["prompt"]
        answer = ds[i]["answer"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs, do_sample=False, max_new_tokens=gen_cfg["max_new_tokens"]
        )
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        guess = extract_last_int(resp)
        if guess is not None and str(guess) == answer:
            correct += 1
    return correct / n

# --------------------------
# main
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                        help="Try TinyLlama/TinyLlama-1.1B-Chat-v1.0 on A100 with --use_lora 1")
    parser.add_argument("--use_lora", type=int, default=0, help="Enable QLoRA (bitsandbytes + PEFT)")
    parser.add_argument("--micro_bsz", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64, help="Number of prompts sampled per PPO step")
    parser.add_argument("--ppo_epochs", type=int, default=2)
    parser.add_argument("--total_ppo_steps", type=int, default=300)
    parser.add_argument("--target_kl", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1.0e-5)
    parser.add_argument("--len_penalty", type=float, default=0.0)
    parser.add_argument("--digits", type=int, default=2, help="difficulty: 1=easy, 2=default, 3=harder")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--save_dir", type=str, default="./ppo_math_ckpt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # dataset
    dsdict = build_dataset(n_train=3000, n_val=400, n_digits=args.digits, seed=args.seed)
    train_ds = dsdict["train"]
    val_ds = dsdict["validation"]

    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # model (policy + value head) and optional QLoRA
    quant_cfg = None
    peft_cfg = None
    if args.use_lora:
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        policy_base = AutoModelForCausalLM.from_pretrained(
            args.model_name, quantization_config=quant_cfg, device_map="auto"
        )
        policy = AutoModelForCausalLMWithValueHead.from_pretrained(policy_base)  # TRL wraps value head
        # TRL will attach a small value head; for LoRA adapters, use PEFT via PPOConfig.peft_config
        from peft import LoraConfig
        peft_cfg = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["c_attn","q_proj","v_proj","k_proj","o_proj"]  # adjust per arch
        )
    else:
        policy = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_name).to(device)

    # reference model for KL
    ref_model = None  # PPOTrainer will create its own ref by default if None

    # PPO config
    ppo_config = PPOConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,             # prompts per PPO step (rollout batch)
        mini_batch_size=args.micro_bsz,         # per optimizer minibatch
        ppo_epochs=args.ppo_epochs,
        target_kl=args.target_kl,
        remove_unused_columns=False,
        device=device,
        seed=args.seed,
        optimize_cuda_cache=True,
        accelerate_kwargs={"mixed_precision": "bf16" if torch.cuda.is_available() else "no"},
        peft_config=peft_cfg
    )

    trainer = PPOTrainer(
        config=ppo_config,
        model=policy,
        ref_model=ref_model,
        tokenizer=tok,
        dataset=train_ds
    )

    # Baseline eval (greedy)
    base_acc = evaluate_accuracy(trainer.model, tok, val_ds,
                                 {"max_new_tokens": args.max_new_tokens}, device=device)
    print(f"[baseline greedy accuracy] {base_acc:.3f}")

    # PPO training loop
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True, temperature=0.7, top_p=0.95, pad_token_id=tok.pad_token_id
    )

    for step_idx in range(args.total_ppo_steps):
        # 1) Sample a batch of prompts
        idxs = random.sample(range(len(train_ds)), k=ppo_config.batch_size)
        prompts = [train_ds[i]["prompt"] for i in idxs]
        golds   = [train_ds[i]["answer"] for i in idxs]

        # 2) Rollout: generate responses from current policy
        batch_inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        response_ids = trainer.generate(batch_inputs["input_ids"], attention_mask=batch_inputs["attention_mask"], **gen_kwargs)
        responses = tok.batch_decode(response_ids, skip_special_tokens=True)

        # 3) Compute scalar rewards
        rewards = compute_reward(prompts, responses, golds, len_penalty=args.len_penalty)

        # 4) PPO step: updates policy & value head
        trainer.step(prompts, responses, rewards)

        if (step_idx + 1) % 20 == 0:
            acc = evaluate_accuracy(trainer.model, tok, val_ds,
                                    {"max_new_tokens": args.max_new_tokens}, device=device)
            avg_r = sum(rewards)/len(rewards)
            print(f"[step {step_idx+1}] val_acc={acc:.3f} avg_reward={avg_r:.3f} kl={trainer.current_kl:.3f}")

    # Final eval + save
    final_acc = evaluate_accuracy(trainer.model, tok, val_ds,
                                  {"max_new_tokens": args.max_new_tokens}, device=device)
    print(f"[final greedy accuracy] {final_acc:.3f}")
    trainer.save_pretrained(args.save_dir)
    tok.save_pretrained(args.save_dir)
    print(f"Saved to {args.save_dir}")
    # Optionally dump a tiny eval sample
    sample = []
    for i in range(10):
        prompt = val_ds[i]["prompt"]
        inputs = tok(prompt, return_tensors="pt").to(device)
        out = trainer.model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
        resp = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        sample.append({"prompt": prompt, "response": resp, "gold": val_ds[i]["answer"]})
    with open(os.path.join(args.save_dir, "eval_samples.jsonl"), "w") as f:
        for s in sample:
            f.write(json.dumps(s) + "\n")

if __name__ == "__main__":
    main()
