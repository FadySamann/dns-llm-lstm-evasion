# Cell 1: Environment setup (Kaggle sessions are ephemeral -- reinstall pinned
# versions every session; Kaggle's base image ships different versions of
# several of these and --force-reinstall is what keeps runs reproducible)
!pip install --upgrade --force-reinstall \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    Pillow==10.4.0 \
    numpy==1.26.4 \
    pandas==2.2.3 \
    transformers==4.47.1 \
    trl==0.12.1 \
    peft==0.14.0 \
    accelerate==1.2.1 \
    bitsandbytes==0.45.0 \
    huggingface-hub==0.27.0 \
    datasets \
    tldextract
print("cell 1 done")
########################
# Cell 2: Imports + HF auth (back to kaggle_secrets -- safer than a pasted
# token in a notebook that might end up public, and it's the Kaggle-native way)
import os
# MUST be set before torch touches CUDA at all. device_map={"": 0} in Cell 4
# only controls where the model's weights live -- Trainer separately checks
# torch.cuda.device_count() and, seeing 2 GPUs visible, auto-wraps the model
# in nn.DataParallel regardless. DataParallel is incompatible with
# bitsandbytes 4-bit quantized params (their quant_state/absmax bookkeeping
# tensors don't replicate correctly across devices -- that's the
# "Input tensors need to be on the same GPU" error). Hiding GPU 1 entirely
# is what actually prevents the DataParallel wrap.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
os.environ["HF_TOKEN"] = hf_token
login(token=hf_token)

# Kaggle can have a separately-attached HF token via Notebook > Add-ons > Hugging
# Face, distinct from this Kaggle Secret. If they diverge, some code path relying
# on env-var auto-detection later could silently authenticate as the wrong
# account -- and Meta-Llama-3.1-8B-Instruct is gated, so that shows up as a
# 401/403 at Cell 4's from_pretrained(), not here.
assert os.environ["HF_TOKEN"] == hf_token, (
    "Env var HF_TOKEN diverges from the Kaggle Secret's token -- "
    "check Notebook > Add-ons > Hugging Face for a separately-attached token."
)
print("cell 2 done")
#################################
# Cell 3: Config
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# corrected filename: prepare_data_LLM_v5.py's output (no <|eot_id|> in
# "output" -- formatting_prompts_func in Cell 5 handles that defensively)
dataset_path = "/kaggle/input/fqdn-dataset/train_strict_v5.jsonl"

CHECKPOINT_DIR = "./dns_tunnel_checkpoints_v10_attnOnly_alpha25"
ADAPTER_OUT_DIR = "dns_tunnel_v10_adapter_attnOnly_alpha25_20k"
CSV_LOG_PATH = "training_loss_log_v10_attnOnly_20k.csv"

# LoRA configuration -- ABLATION: attention-only target_modules (matches
# v9's LoRA shape), everything else identical to v10 (single-GPU config,
# batch size 8, float16 compute dtype, alpha=25 EOT weighting, 20k step
# ceiling). Isolates target_modules as the only changed variable relative
# to the MLP-expanded run that collapsed at 20k steps.
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],      # attention
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 4-bit Quantization Config. NOTE: compute_dtype set to float16, not bfloat16.
# Kaggle's T4 GPUs are Turing (sm_75) and don't have native bf16 tensor-core
# support -- bf16 either silently falls back to slow emulation or misbehaves.
# This must match the fp16=True mixed-precision setting in Cell 6's
# TrainingArguments (the original Kaggle script had these mismatched:
# bnb_4bit_compute_dtype=bfloat16 with fp16=True -- fixed here).
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
print("cell 3 done")
##################################
# Cell 4: Load model + tokenizer
# device_map="auto" -- with CUDA_VISIBLE_DEVICES="0" set above, only one GPU
# is visible to this process, so "auto" has nothing to split across and
# resolves to the same placement as {"": 0} did. If CUDA_VISIBLE_DEVICES is
# ever removed/changed, "auto" would go back to splitting across both T4s
# and reintroduce the DataParallel/bitsandbytes crash fixed earlier -- don't
# remove that env var without re-testing.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    token=hf_token
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    token=hf_token
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("cell 4 done")
#################################
# Cell 5: Data loading + formatting (unchanged logic)
dataset = load_dataset("json", data_files=dataset_path, split="train")

def formatting_prompts_func(sample):
    outputs = []
    eot_token = "<|eot_id|>"

    for i in range(len(sample['instruction'])):
        raw_output = sample['output'][i].strip()
        if raw_output.endswith(eot_token):
            raw_output = raw_output[:-len(eot_token)].strip()

        text = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{sample['instruction'][i]}{eot_token}"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{sample['input'][i]}{eot_token}"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{raw_output}{eot_token}"
        )
        outputs.append(text)
    return outputs

test_output = formatting_prompts_func(dataset[:1])
print("--- VERIFICATION OF DYNAMIC FORMATTING ---")
print(test_output[0])
print("--- END VERIFICATION ---")
############################################
# Cell 6: Training arguments
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=15,
    load_best_model_at_end=False,
    num_train_epochs=3,               # outer cap; max_steps below is what actually
                                       # governs stopping
    max_steps=20000,                  # matches draft 11's calibration value.
                                       # NOTE: draft 11 reached 10,000 steps via
                                       # real DDP across 2 T4s (16 examples/step,
                                       # 160,000 examples total). This run is
                                       # single-GPU (8 examples/step, 80,000
                                       # examples total) -- same step count,
                                       # about half the actual data exposure.
                                       # At the measured 0.17 it/s: ~16.3h.
    per_device_train_batch_size=2,    # reverted to T4-16GB-safe values
    gradient_accumulation_steps=4,    # effective batch size = 8
    optim="paged_adamw_32bit",        # genuinely earns its keep on 16GB T4s
    logging_steps=10,
    learning_rate=5e-5,
    fp16=True,                        # T4 = Turing, no native bf16
    bf16=False,
    max_grad_norm=0.3,
    warmup_steps=500,
    lr_scheduler_type="constant",
    report_to="none",
    neftune_noise_alpha=5,
    dataloader_num_workers=2,         # Kaggle CPU allocation is modest; don't oversubscribe
)
print("cell 6 done")
####################################
# Cell 7: Trainer, collator, resume logic, train, save
import torch.nn as nn
from transformers import TrainerCallback
import warnings
import logging
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("trl").setLevel(logging.ERROR)

class LossLoggingCallback(TrainerCallback):
    def __init__(self, csv_file):
        self.csv_file = csv_file
        if not os.path.exists(self.csv_file):
            pd.DataFrame(columns=["step", "loss"]).to_csv(self.csv_file, index=False)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            step = state.global_step
            loss = logs["loss"]
            new_log = pd.DataFrame({"step": [step], "loss": [loss]})
            new_log.to_csv(self.csv_file, mode='a', header=False, index=False)

class DynamicWeightedSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        vocab_size = self.model.config.vocab_size
        stop_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        raw_loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        raw_loss = raw_loss.view(shift_labels.shape)

        alpha = 25.0
        positions = torch.arange(shift_labels.shape[1], device=self.model.device).repeat(shift_labels.shape[0], 1)
        dynamic_penalty = 1.0 + (positions.float() / alpha)

        weights = torch.where(
            shift_labels == stop_token_id,
            dynamic_penalty,
            torch.tensor(1.0, device=self.model.device, dtype=logits.dtype)
        )

        weights = torch.where(shift_labels == -100, torch.tensor(0.0, device=self.model.device), weights)
        weighted_loss = (raw_loss * weights).sum() / (weights > 0).sum()

        return (weighted_loss, outputs) if return_outputs else weighted_loss

response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

loss_callback = LossLoggingCallback(CSV_LOG_PATH)

trainer = DynamicWeightedSFTTrainer(
    model=model,
    train_dataset=dataset,
    data_collator=collator,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    args=training_args,
    max_seq_length=512,
    tokenizer=tokenizer,
    callbacks=[loss_callback]
)

# --- RESUME LOGIC ---
# IMPORTANT (Kaggle-specific): this only finds a checkpoint if CHECKPOINT_DIR
# already has one *in this session's /kaggle/working*. Kaggle wipes
# /kaggle/working between separate Edit sessions -- it does NOT persist
# automatically like a local disk. To actually resume across sessions:
#   1. At the end of a session, commit the notebook (Save Version) so its
#      /kaggle/working output becomes a versioned Kaggle Dataset/output.
#   2. In the next session, "Add Data" -> attach that previous output as an
#      input, then copy/symlink its checkpoint folder into CHECKPOINT_DIR
#      (a read-only /kaggle/input path can't be trained into directly)
#      BEFORE running this cell, e.g.:
#        !cp -r /kaggle/input/<your-committed-output>/dns_tunnel_checkpoints_v10_alpha25 .
# Skipping this means every new session starts from scratch, which is easy
# to miss until you notice the loss curve restarting.
resume_from_checkpoint = None

if os.path.exists(CHECKPOINT_DIR) and os.listdir(CHECKPOINT_DIR):
    checkpoints = [os.path.join(CHECKPOINT_DIR, d) for d in os.listdir(CHECKPOINT_DIR) if "checkpoint" in d]
    if checkpoints:
        resume_from_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")

        if os.path.exists(CSV_LOG_PATH):
            resume_step = int(resume_from_checkpoint.split('-')[-1])
            df_temp = pd.read_csv(CSV_LOG_PATH)
            df_temp = df_temp[df_temp['step'] < resume_step]
            df_temp.to_csv(CSV_LOG_PATH, index=False)
            print(f"CSV adjusted to step {resume_step} for clean resumption.")

# --- TRAIN ---
trainer.train(resume_from_checkpoint=resume_from_checkpoint)

# --- SAVE FINAL ADAPTER ---
trainer.model.save_pretrained(ADAPTER_OUT_DIR)
tokenizer.save_pretrained(ADAPTER_OUT_DIR)

print(f"Final model saved to '{ADAPTER_OUT_DIR}'. Loss logs saved to: {CSV_LOG_PATH}")
