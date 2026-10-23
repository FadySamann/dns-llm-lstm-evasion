# Cell 1: Environment Setup
#!rm -rf /usr/local/lib/python3.11/dist-packages/~*
!pip install -q --upgrade --force-reinstall \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    Pillow==10.4.0 \
    numpy==1.26.4 \
    pandas==2.2.3 \
    transformers==4.47.1 \
    peft==0.14.0 \
    accelerate==1.2.1 \
    bitsandbytes==0.45.0 \
    huggingface-hub==0.27.0

print("Cell 1 done")
####################################
#Cell 2
"""
subdomain_llm_modeb.py  --  Mode B Subdomain-LLM Generation (Kaggle GPU)
========================================================================
Adapts the current Mode A validation script (fine-tuning-llm.ipynb)
for Mode B payload-chunk generation.

The generation pipeline is kept IDENTICAL to Mode A to ensure results
are directly comparable. 
"""

import os
# Force single-GPU execution before torch initializes
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import pandas as pd
import math
import json
import re
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

# =============================================================================
# CONFIGURATION
# =============================================================================
HF_TOKEN      = UserSecretsClient().get_secret("HF_TOKEN")
BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ADAPTER_PATH  = "/kaggle/working/dns_tunnel_v10_adapter_attnOnly_alpha25_20k"

# JSONL produced by modeb_prepare_jsonl.py -- adjust dataset name
JSONL_PATH  = "/kaggle/input/fqdn-dataset/modeb_inference.jsonl"
OUTPUT_CSV  = "/kaggle/working/modeb_llm_results.csv"

BATCH_SIZE     = 64     
SAVE_EVERY     = 1000    
MAX_NEW_TOKENS = 128    

DO_SAMPLE    = True
TEMPERATURE  = 0.8
MAX_ATTEMPTS = 5    

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def calculate_entropy(text: str) -> float:
    """Shannon entropy -- same definition as the LSTM eval."""
    if not text or not isinstance(text, str):
        return 0.0
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in Counter(text).values())

def build_prompt(instruction: str, user_input: str) -> str:
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{instruction}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_input}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

_TOKEN_LEAK  = re.compile(
    r'(?:<\|start_header_id\|>|<\|end_header_id\|>|'
    r'<\|assistant\|>|<\|system\|>|<\|user\|>|'
    r'header_id|eot_id|<\|eot_id\|>)',
    re.IGNORECASE
)
_RFC_ILLEGAL = re.compile(r'[^a-z0-9\-\._]')  

def is_valid_fqdn(fqdn: str) -> tuple[bool, str]:
    if '\n' in fqdn or '\r' in fqdn:
        return False, "newline_in_fqdn"
    if _TOKEN_LEAK.search(fqdn):
        return False, "token_leak"
    if _RFC_ILLEGAL.search(fqdn.lower()):
        return False, "illegal_chars"
    if not fqdn or fqdn.startswith('.') or fqdn.endswith('.') or '..' in fqdn or '.' not in fqdn:
        return False, "boundary_error"
    if len(fqdn) > 253:
        return False, "fqdn_too_long"
    for label in fqdn.split('.'):
        if len(label) == 0:
            return False, "empty_label"
        if len(label) > 63:
            return False, "label_too_long"
    return True, ""

# =============================================================================
# RESUME LOGIC
# =============================================================================
processed_inputs: set[str] = set()
csv_exists = os.path.exists(OUTPUT_CSV)

if csv_exists:
    try:
        existing_df     = pd.read_csv(OUTPUT_CSV)
        processed_inputs = set(existing_df["input_field"].astype(str).tolist())
        print(f"Resuming: {len(processed_inputs):,} samples already completed.")
    except Exception as e:
        print(f"Could not read existing CSV, starting fresh. Reason: {e}")
        csv_exists = False

# =============================================================================
# DATA LOADING
# =============================================================================
print(f"Loading Mode B JSONL: {JSONL_PATH} ...")
validation_pool = []   

with open(JSONL_PATH, 'r', encoding='utf-8') as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        if entry.get("input", "") in processed_inputs:
            continue    
        validation_pool.append(entry)

print(f"Records to process: {len(validation_pool):,}")

if not validation_pool:
    print("Nothing to do -- all records already processed.")
else:
    # =========================================================================
    # MODEL & TOKENIZER LOADING
    # =========================================================================
    print("\nLoading tokeniser and model ...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"    

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    print("Model loaded successfully.")

    stop_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    # =========================================================================
    # BATCHED INFERENCE LOOP
    # =========================================================================
    all_results = []
    total_done  = len(processed_inputs)

    for batch_start in tqdm(
            range(0, len(validation_pool), BATCH_SIZE),
            desc="Mode B batches"):

        batch = validation_pool[batch_start : batch_start + BATCH_SIZE]

        pending        = list(range(len(batch)))   
        sample_results = [None] * len(batch)       
        attempt_counts = [0]   * len(batch)

        for attempt in range(1, MAX_ATTEMPTS + 1):
            if not pending:
                break   

            pending_entries = [batch[i] for i in pending]
            prompts = [
                build_prompt(e["instruction"], e["input"])
                for e in pending_entries
            ]

            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,   
            ).to("cuda")

            with torch.inference_mode():
                output_ids = model.generate(
                    **encoded,
                    max_new_tokens = MAX_NEW_TOKENS,
                    do_sample      = DO_SAMPLE,
                    temperature    = TEMPERATURE,
                    eos_token_id   = stop_token_id,
                    pad_token_id   = stop_token_id,
                )

            still_pending = []
            prompt_len    = encoded["input_ids"].shape[1]

            for k, batch_idx in enumerate(pending):
                attempt_counts[batch_idx] = attempt
                
                # Sliced isolation for generated tokens only
                gen_ids = output_ids[k][prompt_len:]
                
                # Detect termination mathematically via token ID
                clean_stop  = stop_token_id in gen_ids.tolist()
                termination = "clean_stop" if clean_stop else "max_tokens"
                
                decoded_new  = tokenizer.decode(gen_ids, skip_special_tokens=False)
                generated_fqdn = decoded_new.split("<|eot_id|>")[0].strip()
                
                valid, reason  = is_valid_fqdn(generated_fqdn)

                sample_results[batch_idx] = {
                    "generated_raw":  decoded_new,
                    "generated_fqdn": generated_fqdn,
                    "termination":    termination,
                    "valid":          valid,
                    "discard_reason": reason,
                }

                if not valid:
                    still_pending.append(batch_idx)

            pending = still_pending  

        for i, entry in enumerate(batch):
            res = sample_results[i]
            
            # Map max_tokens failures that ran out of retries
            discard_reason = res["discard_reason"]
            if not res["valid"] and res["termination"] == "max_tokens":
                discard_reason = "max_tokens_exhausted"
                
            all_results.append({
                "payload_id":           entry.get("_payload_id",          ""),
                "chunk_size_bytes":     entry.get("_chunk_size_bytes",    ""),
                "chunk_base32":         entry.get("_chunk_base32",        ""),
                "chunk_base32_len":     entry.get("_chunk_base32_len",    ""),
                "approx_seed_ratio":    entry.get("_approx_seed_ratio",   ""),
                "model_freedom_chars":  entry.get("_model_freedom_chars", ""),
                "assigned_root_domain": entry.get("_assigned_root_domain",""),
                "input_field":          entry["input"],
                "generated_raw":        res["generated_raw"],
                "model_output":         res["generated_fqdn"],  
                "termination":          res["termination"],
                "gen_entropy":          round(calculate_entropy(res["generated_fqdn"]), 3),
                "valid":                int(res["valid"]),
                "discard_reason":       discard_reason,
                "attempts":             attempt_counts[i],
            })

        total_done += len(batch)

        if len(all_results) >= SAVE_EVERY:
            flush_df = pd.DataFrame(all_results)
            flush_df.to_csv(
                OUTPUT_CSV, mode="a", index=False,
                header=not os.path.exists(OUTPUT_CSV),
            )
            all_results = []
            torch.cuda.empty_cache()
            print(f"  -> Flushed. Total done: {total_done:,} / "
                  f"{len(validation_pool) + len(processed_inputs):,}")

    if all_results:
        flush_df = pd.DataFrame(all_results)
        flush_df.to_csv(
            OUTPUT_CSV, mode="a", index=False,
            header=not os.path.exists(OUTPUT_CSV),
        )

# =============================================================================
# SUMMARY REPORT
# =============================================================================
if os.path.exists(OUTPUT_CSV):
    df = pd.read_csv(OUTPUT_CSV)
    df['chunk_size_bytes'] = pd.to_numeric(
        df['chunk_size_bytes'], errors='coerce')

    total   = len(df)
    valid   = df['valid'].sum()
    invalid = total - valid

    print("\n" + "=" * 55)
    print("       MODE B LLM -- GENERATION SUMMARY")
    print("=" * 55)
    print(f"  Total instances   : {total:,}")
    print(f"  Valid FQDNs       : {valid:,}  ({valid/total*100:.2f}%)")
    print(f"  Discarded         : {invalid:,}  ({invalid/total*100:.2f}%)")
    print(f"  Avg attempts      : {df['attempts'].mean():.2f}")
    
    clean_pct = (df['termination'] == 'clean_stop').mean() * 100
    print(f"  Clean-stop rate   : {clean_pct:.2f}%")

    if invalid > 0:
        print(f"\n  Discard reasons:")
        for reason, count in (df[df['valid'] == 0]
                               ['discard_reason'].value_counts().items()):
            print(f"    {reason:<25}: {count:,}")

    print(f"\n  Per chunk-size class:")
    print(f"  {'Bytes':>6}  {'Root domain':<20}  "
          f"{'Valid':>7}  {'Invalid':>8}  {'Valid%':>7}  {'Avg att':>8}")
    print(f"  {'-'*60}")
    for byte_len in sorted(df['chunk_size_bytes'].dropna().unique()):
        sub  = df[df['chunk_size_bytes'] == byte_len]
        v    = sub['valid'].sum()
        iv   = len(sub) - v
        root = sub['assigned_root_domain'].iloc[0]
        avg_att = sub['attempts'].mean()
        print(f"  {int(byte_len):>6}  {root:<20}  "
              f"{v:>7,}  {iv:>8,}  "
              f"{v/len(sub)*100:>6.1f}%  "
              f"{avg_att:>8.2f}")

    print(f"\n  Sample outputs (first 3 valid per class):")
    for byte_len in sorted(df['chunk_size_bytes'].dropna().unique()):
        sub = df[
            (df['chunk_size_bytes'] == byte_len) &
            (df['valid'] == 1)
        ].head(3)
        if len(sub) == 0:
            continue
        root   = sub.iloc[0]['assigned_root_domain']
        b32len = sub.iloc[0]['chunk_base32_len']
        print(f"\n  {int(byte_len)}-byte / {b32len}-char chunk ({root}):")
        for _, r in sub.iterrows():
            print(f"    seed : {r['chunk_base32']}")
            print(f"    fqdn : {r['model_output']}")

    print(f"\n  Results saved -> {OUTPUT_CSV}")
