#Mode A -- Test-set evaluation for the v10 QLoRA adapter
#Cell 1
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
import torch
import os
import pandas as pd
import math
import json
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
print("Cell 1 done")
#####################################
#Cell 2
# =============================================================================
# CONFIGURATION
# =============================================================================
HF_TOKEN      = UserSecretsClient().get_secret("HF_TOKEN")
BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# The v10 adapter produced by LLM_train_v10.py minus MLP layes
ADAPTER_PATH  = "/kaggle/working/dns_tunnel_v10_adapter_attnOnly_alpha25_20k"

# Both files produced by the same prepare_data_LLM_v5.py run.
# Row i in the CSV and line i in the JSONL are the same FQDN, same seed,
# same ratio -- guaranteed by construction (both appended in the same loop
# iteration in build_test_seed_data(), no intervening shuffle). No runtime
# alignment check is needed.
TEST_JSONL_PATH   = "/kaggle/input/datasets/fqdn-dataset/test_strict_v5.jsonl"
SEED_MATCHING_CSV = "/kaggle/input/datasets/fqdn-dataset/llm_test_seed_matching_v5.csv"

# Running per-sample results (crash-safe, appended in order).
# Resume uses len(RESULTS_CSV) as the row index to continue from.
RESULTS_CSV   = "/kaggle/working/llm_modeA_v10_results_attnOnly.csv" #"llm_modeA_v10_results_attnOnly.csv"

# Final merged output: original seed-matching columns +
# model_output, termination, gen_entropy, match, len_diff.
# Original SEED_MATCHING_CSV is left untouched.
MERGED_CSV    = "llm_test_seed_matching_v5_scored_attnOnly.csv"

BATCH_SIZE     = 64 #32
SAVE_EVERY     = 2000 #500
MAX_NEW_TOKENS = 128

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
    """Reconstruct the exact chat-template prompt used during fine-tuning."""
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{instruction}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_input}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

# =============================================================================
# INDEX-BASED RESUME
# The number of rows already written to RESULTS_CSV is the exact index into
# TEST_JSONL_PATH to continue from -- no set lookups, no reordering.
# =============================================================================
already_done = 0
if os.path.exists(RESULTS_CSV):
    try:
        already_done = len(pd.read_csv(RESULTS_CSV))
        print(f"Resuming from index {already_done:,} ({already_done:,} rows already scored).")
    except Exception as e:
        print(f"Could not read {RESULTS_CSV}, starting fresh. Reason: {e}")
        already_done = 0

# =============================================================================
# DATA LOADING -- skip the first `already_done` lines using the file index
# =============================================================================
test_pool = []
print(f"Loading test set from {TEST_JSONL_PATH} (skipping first {already_done:,} lines)...")
with open(TEST_JSONL_PATH, "r") as fh:
    for idx, line in enumerate(fh):
        if idx < already_done:
            continue
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        test_pool.append({
            "instruction": entry["instruction"],
            "input":       entry["input"],
            "output":      entry["output"],
        })

total_in_file = already_done + len(test_pool)
print(f"Total test-set size: {total_in_file:,} | Remaining to score: {len(test_pool):,}")

# =============================================================================
# MODEL & TOKENIZER LOADING
# =============================================================================
if test_pool:
    print("Loading tokeniser and model ...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, # matches T4 training config
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"   # CRITICAL for left-padded batched generation

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

    for batch_start in tqdm(range(0, len(test_pool), BATCH_SIZE),
                            desc="Scoring test batches"):

        batch   = test_pool[batch_start : batch_start + BATCH_SIZE]
        prompts = [build_prompt(e["instruction"], e["input"]) for e in batch]

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to("cuda")

        prompt_len = encoded["input_ids"].shape[1]

        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=stop_token_id,
                pad_token_id=stop_token_id,
            )

        for j, ids in enumerate(output_ids):
            # Isolate only the newly generated token IDs (past the prompt)
            gen_ids = ids[prompt_len:]

            # termination: detected at token-ID level on gen_ids only.
            # String-based detection is unreliable here because
            # pad_token_id == stop_token_id, so trailing padding also
            # decodes as "<|eot_id|>" -- a token-ID check is exact.
            clean_stop  = stop_token_id in gen_ids.tolist()
            termination = "clean_stop" if clean_stop else "max_tokens"

            decoded_new    = tokenizer.decode(gen_ids, skip_special_tokens=False)
            generated_fqdn = decoded_new.split("<|eot_id|>")[0].strip()

            clean_gt = batch[j]["output"].replace("<|eot_id|>", "").strip()

            all_results.append({
                "model_output": generated_fqdn,
                "termination":  termination,
                "gen_entropy":  round(calculate_entropy(generated_fqdn), 3),
                "match":        1 if generated_fqdn == clean_gt else 0,
                "len_diff":     len(generated_fqdn) - len(clean_gt),
            })

        # Periodic flush -- append only, preserving order
        if len(all_results) >= SAVE_EVERY:
            pd.DataFrame(all_results).to_csv(
                RESULTS_CSV, mode="a", index=False,
                header=not os.path.exists(RESULTS_CSV),
            )
            all_results = []
            torch.cuda.empty_cache()

    # Final flush
    if all_results:
        pd.DataFrame(all_results).to_csv(
            RESULTS_CSV, mode="a", index=False,
            header=not os.path.exists(RESULTS_CSV),
        )

# =============================================================================
# MERGE onto the seed-matching CSV (positional, guaranteed by
# prepare_data_LLM_v5.py's build_test_seed_data() construction).
# Writes MERGED_CSV; SEED_MATCHING_CSV is left untouched.
# =============================================================================
if os.path.exists(RESULTS_CSV) and os.path.exists(SEED_MATCHING_CSV):
    seed_df    = pd.read_csv(SEED_MATCHING_CSV)
    results_df = pd.read_csv(RESULTS_CSV)

    n_scored = len(results_df)
    if n_scored < len(seed_df):
        print(f"NOTE: {n_scored:,} / {len(seed_df):,} rows scored so far. "
              f"Merged file will contain only scored rows -- re-run to complete.")

    n = min(len(seed_df), n_scored)
    merged = seed_df.iloc[:n].reset_index(drop=True).copy()
    for col in ["model_output", "termination", "gen_entropy", "match", "len_diff"]:
        merged[col] = results_df[col].iloc[:n].values

    merged.to_csv(MERGED_CSV, index=False)
    print(f"Merged file written to: {MERGED_CSV}  ({n:,} rows)")
    print(f"Columns: {list(merged.columns)}")

# =============================================================================
# SUMMARY REPORT
# =============================================================================
if os.path.exists(RESULTS_CSV):
    df = pd.read_csv(RESULTS_CSV)
    print("\n" + "="*55)
    print("   TEST SET SUMMARY (Mode A, v10 adapter)")
    print("="*55)
    print(f"  Total samples scored    : {len(df):,}")
    print(f"  Exact-match accuracy    : {df['match'].mean() * 100:.2f}%")
    clean_pct = (df['termination'] == 'clean_stop').mean() * 100
    print(f"  Clean-stop rate         : {clean_pct:.2f}%")
    print(f"  max_tokens rate         : {100 - clean_pct:.2f}%")
    print(f"  Avg generated entropy   : {df['gen_entropy'].mean():.3f}")
    print(f"  Avg length difference   : {df['len_diff'].mean():.2f} chars")
    print(f"\nResults  : {RESULTS_CSV}")
    print(f"Merged   : {MERGED_CSV}")
