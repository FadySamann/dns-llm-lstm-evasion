import json
import math
import os
import random
from collections import Counter

import pandas as pd
import tldextract

# --- Configuration ---
# Inputs are the shared splits produced by combine_and_shuffle_v4.py.
# Per the reduced-scope redesign: the LLM trains on train_set ONLY
# (val_set is reserved for the LSTM's train_set + val_set combination),
# and test_set is used to build the shared seed-matching CSV for BOTH models.
TRAIN_INPUT_FILE = "train_domains.txt"
TEST_INPUT_FILE = "test_domains.txt"

# Binary QRNG file used for reproducible shuffling (same source used
# throughout the pipeline, e.g. combine_and_shuffle_v4.py).
BINARY_SEED_FILE = "ANU_13Oct2017_100MB"

# Outputs
TRAIN_OUTPUT_JSONL = "train_strict_v5.jsonl"
TEST_OUTPUT_JSONL = "test_strict_v5.jsonl"
TEST_OUTPUT_CSV = "llm_test_seed_matching_v5.csv"
SUMMARY_TXT_FILENAME = "prepare_data_LLM_v5_summary.txt"

# Multi-Tiered Seed Strategy ratios — must match the ratios used by the
# LSTM evaluation script (subdomain_100kgenerator_lstm_model_v10.py) so
# both models are tested against the exact same seed lengths.
TEST_RATIO_CHOICES = [0.25, 0.5, 1.0]

# STRICT TASK PROMPT: Teaches the task AND the formatting discipline
STRICT_TASK_PROMPT = (
    "You are a DNS traffic generator. Your task is to predict and complete the full FQDN "
    "using the provided domain and partial seed. Output ONLY the FQDN. "
    "Do not explain, do not use parentheses, and stop immediately after the FQDN."
)


def calculate_shannon_entropy(text: str) -> float:
    """Same entropy definition used in subdomain_100kgenerator_lstm_model_v10.py,
    kept identical so gt_entropy is directly comparable across models."""
    if not text or not isinstance(text, str):
        return 0.0
    char_counts = Counter(text)
    text_length = len(text)
    entropy = 0.0
    for count in char_counts.values():
        probability = count / text_length
        entropy -= probability * math.log2(probability)
    return entropy


def load_qrng_seed_bytes(binary_seed_file: str) -> bytes:
    """Reads the binary QRNG file to use as a reproducible random seed
    (same approach as shuffle_data_LLM.py)."""
    if not os.path.exists(binary_seed_file):
        raise FileNotFoundError(f"Binary seed file '{binary_seed_file}' not found.")
    print(f"Reading binary seed from '{binary_seed_file}'...")
    with open(binary_seed_file, 'rb') as f:
        return f.read()


def shuffle_lines_with_binary_seed(lines: list, rng: random.Random) -> list:
    """Shuffles a list of lines in place using a Random instance seeded
    from the binary QRNG file (logic ported from shuffle_data_LLM.py)."""
    print(f"Shuffling {len(lines)} training records...")
    rng.shuffle(lines)
    return lines


def build_training_records(input_file: str) -> tuple:
    """
    Reads FQDNs from the train_set file and expands each into the
    Multi-Tiered Seed Strategy training records (full / half / quarter
    subdomain seeds), WITHOUT appending <|eot_id|> to the output — Cell 5
    of LLM_traning.py strips/re-adds it cleanly when building the Llama-3.1
    prompt block, so appending it here would be redundant.
    """
    records = []
    total_domains = 0
    skipped_no_subdomain = 0

    print(f"Reading train_set domains from '{input_file}'...")
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            fqdn = line.strip().lower()
            if not fqdn:
                continue
            total_domains += 1

            ext = tldextract.extract(fqdn)
            root_domain = ext.top_domain_under_public_suffix
            subdomain = ext.subdomain

            if not subdomain:
                skipped_no_subdomain += 1
                continue

            sub_len = len(subdomain)
            seeds = {subdomain}  # Use a set to avoid duplicate seeds

            if sub_len > 3:
                seeds.add(subdomain[:sub_len // 2])
                seeds.add(subdomain[:sub_len // 4])

            for seed in seeds:
                data_record = {
                    "instruction": STRICT_TASK_PROMPT,
                    "input": f"domain: {root_domain}, seed: {seed}",
                    "output": fqdn,  # NOTE: no <|eot_id|> appended (see docstring)
                }
                records.append(json.dumps(data_record))

    return records, total_domains, skipped_no_subdomain


def build_test_seed_data(input_file: str, rng: random.Random) -> tuple:
    """
    Builds ONE seed-matching row per test_set FQDN, mirroring the seed
    selection logic in subdomain_100kgenerator_lstm_model_v10.py (a single
    randomly chosen ratio from TEST_RATIO_CHOICES per domain).

    Returns two index-aligned outputs built from the exact same seed/ratio
    draw for each domain, so there is no risk of the CSV and JSONL diverging:
      - rows: dicts for the seed-matching CSV (ratio_used, seed_input,
        root_domain, ground_truth, gt_entropy) — consumed by the LSTM script
        and later used to compare BOTH models' outputs against the same seed.
      - records: JSON-line prompt records (instruction/input/output, no
        <|eot_id|>) in the same format as the training JSONL — this is what
        gets fed to the LLM at inference time.
    """
    rows = []
    records = []
    total_domains = 0
    skipped_no_subdomain = 0

    print(f"Reading test_set domains from '{input_file}'...")
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            fqdn = line.strip().lower()
            if not fqdn:
                continue
            total_domains += 1

            ext = tldextract.extract(fqdn)
            root_domain = ext.top_domain_under_public_suffix
            subdomain = ext.subdomain

            if not subdomain:
                skipped_no_subdomain += 1
                continue

            ratio = rng.choice(TEST_RATIO_CHOICES)
            sub_len = len(subdomain)
            seed_len = max(1, int(sub_len * ratio))
            seed = subdomain[:seed_len]

            rows.append({
                "ratio_used": ratio,
                "seed_input": seed,
                "root_domain": root_domain,
                "ground_truth": fqdn,
                "gt_entropy": calculate_shannon_entropy(fqdn),
            })

            records.append(json.dumps({
                "instruction": STRICT_TASK_PROMPT,
                "input": f"domain: {root_domain}, seed: {seed}",
                "output": fqdn,  # ground truth kept for reference/scoring, no <|eot_id|>
            }))

    return rows, records, total_domains, skipped_no_subdomain


# --- Main Execution Logic ---
if __name__ == "__main__":
    try:
        for required_file in (TRAIN_INPUT_FILE, TEST_INPUT_FILE, BINARY_SEED_FILE):
            if not os.path.exists(required_file):
                print(f"Error: required file '{required_file}' was not found.")
                raise SystemExit(1)

        # Single RNG instance seeded from the QRNG binary file, reused for
        # both the training-set shuffle and the test-set ratio selection so
        # the whole script is deterministic/reproducible end to end.
        seed_bytes = load_qrng_seed_bytes(BINARY_SEED_FILE)
        rng = random.Random(seed_bytes)

        # 1. Build training records from train_set (LLM: train_set only)
        train_records, train_domains_total, train_skipped = build_training_records(TRAIN_INPUT_FILE)
        if not train_records:
            print("Error: No training records were generated from the train_set.")
            raise SystemExit(1)

        # 2. Shuffle the training records
        train_records = shuffle_lines_with_binary_seed(train_records, rng)

        # 3. Write shuffled training JSONL
        print(f"Writing {len(train_records)} training records to '{TRAIN_OUTPUT_JSONL}'...")
        with open(TRAIN_OUTPUT_JSONL, 'w', encoding='utf-8') as f:
            for record in train_records:
                f.write(record + '\n')

        # 4. Build the test-set seed data (test_set only): CSV rows for the
        #    LSTM + cross-model comparison, and JSONL prompts for the LLM —
        #    both built from the identical seed/ratio draw per domain.
        test_rows, test_records, test_domains_total, test_skipped = build_test_seed_data(TEST_INPUT_FILE, rng)
        if not test_rows:
            print("Error: No test-set seed rows were generated.")
            raise SystemExit(1)

        test_df = pd.DataFrame(test_rows, columns=[
            "ratio_used", "seed_input", "root_domain", "ground_truth", "gt_entropy"
        ])
        print(f"Writing {len(test_df)} test-set seed rows to '{TEST_OUTPUT_CSV}'...")
        test_df.to_csv(TEST_OUTPUT_CSV, index=False)

        print(f"Writing {len(test_records)} test-set prompt records to '{TEST_OUTPUT_JSONL}'...")
        with open(TEST_OUTPUT_JSONL, 'w', encoding='utf-8') as f:
            for record in test_records:
                f.write(record + '\n')

        # 5. Write the summary file
        print(f"Writing summary to '{SUMMARY_TXT_FILENAME}'...")
        with open(SUMMARY_TXT_FILENAME, 'w', encoding='utf-8') as f:
            f.write("--- LLM Data Preparation Summary (v5) ---\n\n")
            f.write("[Train set -> SFT JSONL]\n")
            f.write(f"Source file: '{TRAIN_INPUT_FILE}'\n")
            f.write(f"Domains read: {train_domains_total}\n")
            f.write(f"Domains skipped (no subdomain): {train_skipped}\n")
            f.write(f"Training records generated (Multi-Tiered Seed Strategy): {len(train_records)}\n")
            f.write(f"Output file (shuffled): '{TRAIN_OUTPUT_JSONL}'\n")
            f.write("Note: outputs do NOT include <|eot_id|>; it is added during "
                     "prompt construction in LLM_traning.py (Cell 5).\n\n")

            f.write("[Test set -> shared LSTM/LLM evaluation]\n")
            f.write(f"Source file: '{TEST_INPUT_FILE}'\n")
            f.write(f"Domains read: {test_domains_total}\n")
            f.write(f"Domains skipped (no subdomain): {test_skipped}\n")
            f.write(f"Seed rows/records generated: {len(test_df)}\n")
            f.write(f"Ratio choices used: {TEST_RATIO_CHOICES}\n")
            f.write(f"CSV output (LSTM input, cross-model comparison): '{TEST_OUTPUT_CSV}'\n")
            f.write("  Columns: ratio_used, seed_input, root_domain, ground_truth, gt_entropy\n")
            f.write(f"JSONL output (LLM inference prompts): '{TEST_OUTPUT_JSONL}'\n")
            f.write("  Fields: instruction, input (domain + seed), output (ground truth, no <|eot_id|>)\n")
            f.write("Row/record order in the CSV and JSONL is index-aligned: row i in the CSV and\n")
            f.write("line i in the JSONL were built from the same seed/ratio draw for the same FQDN,\n")
            f.write("so both models can be scored against the exact same seeds.\n\n")

            f.write("[Reproducibility]\n")
            f.write(f"Binary QRNG seed file: '{BINARY_SEED_FILE}'\n")
            f.write("A single random.Random instance, seeded from the QRNG file bytes, was\n")
            f.write("used sequentially for (1) shuffling the training JSONL and (2) selecting\n")
            f.write("test-set seed ratios, in that order.\n")

        print("\nSuccess! All files processed.")
        print(f"  -> Training JSONL saved to: '{TRAIN_OUTPUT_JSONL}'")
        print(f"  -> Test seed-matching CSV saved to: '{TEST_OUTPUT_CSV}'")
        print(f"  -> Test prompt JSONL saved to: '{TEST_OUTPUT_JSONL}'")
        print(f"  -> Summary saved to: '{SUMMARY_TXT_FILENAME}'")

    except SystemExit:
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise SystemExit(1)
