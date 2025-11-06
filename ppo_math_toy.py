import os, random, argparse, json
from dataclasses import dataclass
from typing import List

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed, AutoModelForSequenceClassification,
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model

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
        b = random.randint(0, 10**(n_digits-1)) if n_digits > 1 else random.randint(0, 9)
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
    import re
    m = re.findall(r"-?\d+", s)
    return int(m[-1]) if m else None

def compute_reward(responses: List[str], answers: List[str], len_penalty: float = 0.0):
    rewards = []
    for y, gold in zip(responses, answers):
        g = extract_last_int(y)
        ok = (g is not None) and (str(g) == gold)
        base = 1.0 if ok else 0.0
        lp = len_penalty * (len(y) / 80.0) if len_penalty > 0 else 0.0
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
        out = model.generate(**inputs, do_sample=False, max_new_tokens=gen_cfg["max_new_tokens"])
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        guess = extract_last_int(resp)
        if guess is not None and str(guess) == answer:
            correct += 1
    return correct / n


# -----------------------------
# PPOTrainer subclass: rule-based rewards
# -----------------------------
class RuleBasedPPOTrainer(PPOTrainer):
    def __init__(self, *args, len_penalty: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._len_penalty = len_penalty

    def get_reward(self, samples, **kwargs):
        __import__('pdb').set_trace()

    def get_rewards(self, samples, **kwargs):
        """
        TRL 0.25 calls this after rollouts.
        `samples` contains:
          - "responses": list[str]  (decoded model outputs)
          - "answers":   list[str]  (from our dataset; we must keep this column)
        Return: torch.Tensor [batch]
        """
        __import__('pdb').set_trace()
        responses = samples["responses"]
        answers   = samples.get("answers", None)
        assert answers is not None, "Dataset must include 'answers' column for rule-based reward."
        rews = compute_reward(responses, answers, len_penalty=self._len_penalty)
        print(rews)
        return torch.tensor(rews, device=self.accelerator.device, dtype=torch.float32)

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
    # each row: {"prompt": "...", "answer": "42"}
    dsdict = build_dataset(n_train=3000, n_val=400, n_digits=args.digits, seed=args.seed)
    train_ds = dsdict["train"]
    # train_ds = train_ds.rename_columns({"prompt": "prompts", "answer": "answers"})
    val_ds = dsdict["validation"]

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # ========= pre-tokenize train dataset (no text_column needed) =========
    # Keep "answers" for reward; produce input_ids/attention_mask for prompts.
    def tok_batch(batch):
        enc = tokenizer(
            batch["prompt"],
            padding=True,
            truncation=True,
            return_attention_mask=True,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "answers": batch["answer"],
        }
    train_ds = train_ds.map(tok_batch, batched=True, remove_columns=["prompt", "answer"])
    train_ds = train_ds.with_format(type="torch", columns=["input_ids", "attention_mask"])
    # ======================================================================

    if not args.use_lora:
        # -------------------------------------------------
        # (A) Baseline full-precision policy/value/reward
        # -------------------------------------------------
        policy_model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype="auto").to(device)

        # add a small value head on the same backbone
        value_model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1).to(device)
        # value_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_name, dtype="auto").to(device)

        # ----- PATCH: expose base_model_prefix and backbone on wrapper -----
        # inner = getattr(value_model, "pretrained_model", None)
        # assert inner is not None, "Value wrapper missing .pretrained_model"
        # # discover the inner prefix (GPT-2/DistilGPT-2 => 'transformer'; LLaMA => 'model')
        # prefix = getattr(inner, "base_model_prefix", None)
        # if prefix is None:
        #     # simple heuristic fallback for GPT2-family
        #     prefix = "transformer" if "gpt" in args.model_name.lower() else "model"
        # value_model.base_model_prefix = prefix
        # setattr(value_model, prefix, getattr(inner, prefix))
        # <---- PATCH for TRL 0.25.0 + distilgpt2

        # reward model (sequence classifier; we compute rule-based rewards anyway)
        reward_model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1).to(device)
        reward_model.eval().requires_grad_(False)

        # explicit frozen reference
        ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype="auto").to(device)
        ref_model.eval().requires_grad_(False)
    else:
        # -------------------------------------------------
        # (B) LoRA / QLoRA mode
        # -------------------------------------------------
        from peft import PeftModel

        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        # Load quantized backbone
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quant_cfg,
            device_map="auto",
        )

        # Attach LoRA adapters to train only low-rank weights
        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        policy_model = get_peft_model(base_model, lora_cfg)

        # Value head can sit on a cloned backbone (sharing weights) with LoRA frozen
        value_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.model_name,
            quantization_config=quant_cfg,
            device_map="auto",
        )

        # reward_model = policy_model  # we compute rewards externally
        reward_model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1).to(device)
        reward_model.eval().requires_grad_(False)

        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quant_cfg,
            device_map="auto",
        )
        ref_model.eval().requires_grad_(False)

    # --- check memory footprint ---
    print(f"Policy dtype: {next(policy_model.parameters()).dtype}, LoRA={args.use_lora}")


    print("policy:", type(policy_model))
    print("value :", type(value_model))
    print("reward:", type(reward_model))
    print("ref   :", type(ref_model))

    # PPO config
    ppo_config = PPOConfig(
        seed=args.seed,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,     # total rollout batch
        gradient_accumulation_steps=1,                   # microbatching handled manually
        num_ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.micro_bsz,                  # optional alias if supported
        kl_coef=args.target_kl,                          # initial KL weight
        cliprange=0.2,
        vf_coef=0.5,
        response_length=args.max_new_tokens,
        temperature=0.7,
        stop_token_id=tokenizer.eos_token_id,
        remove_unused_columns=False,
        logging_steps=10,
        total_episodes=args.total_ppo_steps,
        output_dir=args.save_dir,
    )

    ppo_trainer = RuleBasedPPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=policy_model,
        value_model=value_model,
        reward_model=reward_model,   # unused in our subclass
        # reward_model=None,   # unused in our subclass
        ref_model=ref_model,
        train_dataset=train_ds,
        len_penalty=args.len_penalty,
    )

    # This now performs: rollout -> get_rewards(...) -> PPO update, repeatedly
    ppo_trainer.train()

    # trainer = PPOTrainer(
    #     args=ppo_config,
    #     processing_class=tokenizer,       # tokenizer or processor
    #     model=policy_model,               # policy network
    #     value_model=value_model,          # value head
    #     reward_model=reward_model,        # can be dummy; external rewards computed in Python
    #     ref_model=ref_model,
    #     train_dataset=train_ds,
    # )
    # # baseline eval
    # base_acc = evaluate_accuracy(trainer.model.policy, tokenizer, val_ds,
    #                              {"max_new_tokens": args.max_new_tokens}, device=device)
    # print(f"[baseline greedy accuracy] {base_acc:.3f}")
    #
    # # PPO training loop
    # gen_kwargs = dict(
    #     max_new_tokens=args.max_new_tokens,
    #     do_sample=True, temperature=0.7, top_p=0.95, pad_token_id=tokenizer.pad_token_id
    # )
    #
    # # PPO loop
    # for step_idx in range(args.total_ppo_steps):
    #     # 1) Sample a batch of prompts
    #     idxs = random.sample(range(len(train_ds)), k=ppo_config.per_device_train_batch_size)
    #     prompts = [train_ds[i]["prompt"] for i in idxs]
    #     golds   = [train_ds[i]["answer"] for i in idxs]
    #
    #     # 2) Rollout: generate responses from current policy
    #     batch_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    #
    #     policy = trainer.model.policy          # unwrap the HF CausalLM
    #     policy.eval()                          # (optional) no dropout during rollout
    #     with torch.no_grad():
    #         response_ids = policy.generate(batch_inputs["input_ids"], attention_mask=batch_inputs["attention_mask"], **gen_kwargs)
    #
    #     # 3) Compute scalar rewards
    #     responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    #     rewards = compute_reward(prompts, responses, golds, len_penalty=args.len_penalty)
    #
    #     # 4) PPO step: updates policy & value head
    #     policy.train()
    #     trainer.step(prompts, responses, rewards)
    #
    #     if (step_idx + 1) % 20 == 0:
    #         acc = evaluate_accuracy(trainer.model.policy, tokenizer, val_ds,
    #                                 {"max_new_tokens": args.max_new_tokens}, device=device)
    #         avg_r = sum(rewards)/len(rewards)
    #         print(f"[step {step_idx+1}] val_acc={acc:.3f} avg_reward={avg_r:.3f} kl={trainer.current_kl:.3f}")
    #
    # # Final eval + save
    # final_acc = evaluate_accuracy(trainer.model.policy, tokenizer, val_ds,
    #                               {"max_new_tokens": args.max_new_tokens}, device=device)
    # print(f"[final greedy accuracy] {final_acc:.3f}")
    # trainer.model.save_pretrained(args.save_dir)
    # tokenizer.save_pretrained(args.save_dir)
    # print(f"Saved to {args.save_dir}")
    # # Optionally dump a tiny eval sample
    # sample = []
    # for i in range(10):
    #     prompt = val_ds[i]["prompt"]
    #     inputs = tokenizer(prompt, return_tensors="pt").to(device)
    #     out = trainer.model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
    #     resp = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    #     sample.append({"prompt": prompt, "response": resp, "gold": val_ds[i]["answer"]})
    # with open(os.path.join(args.save_dir, "eval_samples.jsonl"), "w") as f:
    #     for s in sample:
    #         f.write(json.dumps(s) + "\n")

if __name__ == "__main__":
    main()
