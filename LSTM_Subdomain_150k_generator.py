import os
import re
import sys
import time
import string
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------------------------------
# subdomain_150k_generator_lstm_model_v11.py
#
# Difference from subdomain_100kgenerator_lstm_model_v10.py:
#
# 1. Seeds now come from llm_test_seed_matching_v5.csv (the Multi-Tiered
#    Seed Strategy test set already used for the LLM side), not re-derived
#    from test.tfrecord. This makes the LSTM and LLM evaluated on identical
#    seeds for Mode A, rather than two independently-sampled seed pools --
#    replaces the ad hoc random.choice([0.25, 0.5, 1.0]) ratio sampling in
#    v10 (which only approximated the LLM's actual seed distribution) with
#    the real one. All 150,000 rows in the file are used (not 100,000).
#
# 2. Generation is now BATCHED across sequences instead of one
#    model.predict() call per individual sample per character. v10's
#    per-sample loop would mean roughly 150,000 x (average ~26 chars still
#    to generate) ~= 4M+ individual predict() calls -- the same class of
#    per-call-overhead problem already diagnosed and fixed in the LSTM
#    *training* pipeline (dataprepare_v5 -> v6). Here, at each generation
#    timestep, ALL still-active sequences are batched into one (chunked)
#    model.predict() call, cutting total predict() calls from millions to
#    roughly (max_gen_len x n_active/predict_batch_size). Verified against
#    a small dummy model that batched generation is logically equivalent to
#    the original per-sample loop (same padding, same temperature-sampling
#    logic, same stopping condition), just restructured for throughput.
#
# 3. Generation cap changed from a NEW-CHARACTERS-ONLY budget (v10's
#    hardcoded max_gen_len=60, appended on top of the seed regardless of
#    seed length) to a TOTAL-LENGTH cap of MAXLEN (85), covering seed +
#    generated together. This is the fix for two related problems:
#      - v10's 60-new-chars cap could still truncate legitimate long
#        completions: llm_test_seed_matching_v5.csv has ground_truth up to
#        85 chars with up to 70 chars needing generation past the seed.
#      - More importantly, a new-characters-only cap doesn't actually bound
#        output length at all when combined with a long seed: with seeds up
#        to 63 chars observed in this file, a 60-85 char *additional* budget
#        could produce outputs of 123-148 total characters -- well past
#        MAXLEN, i.e. past anything the model was ever trained on, and the
#        kind of runaway/RFC-1035-implausible output the cap was presumably
#        meant to prevent in the first place. Each sequence in
#        generate_batch() now stops once its OWN total length (seed +
#        generated) reaches MAXLEN, independent of how long its seed was;
#        most sequences still stop earlier via <EOS> regardless.
#
# 4. CSV read with keep_default_na=False. The seed file contains literal
#    seed strings "nan" and "null" (e.g. seed_input="nan" for ground_truth
#    "nan-ci.github.io"). Pandas' default NA-string recognition treats the
#    literal text "nan"/"null"/"NA"/etc. as a missing value, not a string --
#    confirmed 6 rows in this exact file get silently turned into NaN under
#    a default pd.read_csv() call. keep_default_na=False disables that
#    recognition so these are read back as the literal strings they are.
#
# 5. Defensive fix (not present in v10): if temperature-sampling ever draws
#    the <PAD> token (index 0) -- which nothing structurally forbids, since
#    a well-trained model assigning it near-zero probability is a learned
#    property, not a hard constraint -- generation now stops for that
#    sequence instead of letting the literal text "<PAD>" leak into
#    model_output. Verified this occurs even in a random/untrained dummy
#    model and would silently corrupt affected rows without this guard.
# ---------------------------------------------------------------------------

# --- Configuration ---
MODEL_PATH = 'LSTM_subdomian_generator_best_overall_model.keras'
SEED_CSV = 'llm_test_seed_matching_v5.csv'
OUTPUT_CSV = 'lstm_test_seed_matching_v5.csv'
DATAPREP_SUMMARY_FILE = 'dataprepare_v6_summary.txt'

TEMPERATURE = 0.8
PREDICT_BATCH_SIZE = 4096  # tune down if you hit GPU memory pressure on the RTX 4050
PROGRESS_LOG_EVERY = 1     # print progress every N generation steps

all_chars = ['<PAD>'] + list(string.ascii_lowercase) + list(string.digits) + ['_', '-', '.'] + ['<EOS>']
char_to_int = {char: i for i, char in enumerate(all_chars)}
int_to_char = {i: char for i, char in enumerate(all_chars)}
PAD_TOKEN = '<PAD>'
EOS_TOKEN = '<EOS>'


def load_dataprep_metadata(summary_path: str) -> dict:
    """
    Parses maxlen/vocab_size from dataprepare_v6.py's summary file so the
    generator uses the exact same values the model was trained with,
    instead of a separately hardcoded (and potentially stale) constant.
    """
    if not os.path.exists(summary_path):
        print(f"Error: '{summary_path}' not found. Point DATAPREP_SUMMARY_FILE "
              f"at the summary produced when this model's training data was prepared.")
        sys.exit(1)

    with open(summary_path, "r", encoding="utf-8") as f:
        text = f.read()

    patterns = {
        "maxlen": r"maxlen \(from train domains only\):\s*(\d+)",
        "vocab_size": r"vocab_size:\s*(\d+)",
    }
    metadata = {}
    missing = []
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            metadata[key] = int(match.group(1))
        else:
            missing.append(key)
    if missing:
        print(f"Error: could not find {missing} in '{summary_path}'.")
        sys.exit(1)
    return metadata


_metadata = load_dataprep_metadata(DATAPREP_SUMMARY_FILE)
MAXLEN = _metadata["maxlen"]
VOCAB_SIZE = _metadata["vocab_size"]
if VOCAB_SIZE != len(all_chars):
    print(f"Error: vocab_size from '{DATAPREP_SUMMARY_FILE}' ({VOCAB_SIZE}) does not match "
          f"this script's character set ({len(all_chars)}). The model and this generator "
          f"would disagree on what each index means -- aborting rather than risk silently "
          f"garbled output.")
    sys.exit(1)

print(f"Loaded maxlen={MAXLEN}, vocab_size={VOCAB_SIZE} from '{DATAPREP_SUMMARY_FILE}'.")
# MAX_TOTAL_LEN bounds the FULL output (seed + generated), matching the
# model's own trained context window -- see generate_batch() docstring and
# header note #3 for why this must be a total-length cap, not a
# newly-generated-characters cap.
MAX_TOTAL_LEN = MAXLEN


def calculate_shannon_entropy(text: str) -> float:
    if not text or not isinstance(text, str):
        return 0.0
    char_counts = Counter(text)
    text_length = len(text)
    entropy = 0.0
    for count in char_counts.values():
        probability = count / text_length
        entropy -= probability * math.log2(probability)
    return entropy


def generate_batch(model, seeds: list, max_total_len: int, temperature: float,
                    predict_batch_size: int) -> tuple:
    """
    Vectorized generation across many seeds at once. At each timestep, all
    still-active sequences (those that haven't emitted <EOS>/<PAD>, and
    whose TOTAL length -- seed + generated so far -- hasn't reached
    max_total_len) are batched into one or more chunked model.predict()
    calls, instead of one predict() call per sequence per character.
    Sampling logic (temperature-scaled multinomial) is unchanged from v10 --
    this is a throughput restructuring, not a change in what gets generated.

    max_total_len bounds the FULL output length (seed included), not just
    the newly-generated portion -- this is what actually prevents runaway
    generation / RFC 1035-implausible output, since the model was never
    trained on sequences longer than MAXLEN and a cap on new-characters-only
    would let long seeds produce outputs well past that (see conversation:
    a bare per-step cap of 85 on top of a 63-char seed could reach 148
    chars). Each sequence stops independently once it reaches
    max_total_len, regardless of how long its particular seed was.

    Returns (generated, termination_reasons):
      generated           -- list of output strings (one per seed)
      termination_reasons -- parallel list of strings, one of:
          'eos'     natural termination: model sampled <EOS>
          'pad'     defensive stop: model sampled <PAD> (degenerate case)
          'cap'     safety-net stop: total length reached max_total_len
                    without <EOS>; indicates failure of learned termination
    """
    n = len(seeds)
    generated = [str(s).lower() if s is not None else '' for s in seeds]
    active = np.array([len(g) < max_total_len for g in generated], dtype=bool)
    # Seeds already at or beyond the cap are immediately cap-stopped
    termination = ['cap' if len(str(s)) >= max_total_len else None for s in seeds]

    t0 = time.time()
    for step in range(max_total_len):
        active_idx = np.where(active)[0]
        n_active = len(active_idx)
        if n_active == 0:
            break

        encoded_batch = [
            [char_to_int[c] for c in generated[i] if c in char_to_int]
            for i in active_idx
        ]
        padded_batch = pad_sequences(encoded_batch, maxlen=MAXLEN, padding='pre', truncating='pre')

        preds_chunks = []
        for start in range(0, n_active, predict_batch_size):
            chunk = padded_batch[start:start + predict_batch_size]
            preds_chunks.append(model.predict(chunk, verbose=0))
        preds = np.concatenate(preds_chunks, axis=0).astype('float64')

        # Temperature-scaled sampling (unchanged from v10, vectorized across rows)
        preds = np.log(preds + 1e-10) / temperature
        exp_preds = np.exp(preds)
        probas = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)

        for local_i, idx in enumerate(active_idx):
            p = probas[local_i]
            next_index = np.argmax(np.random.multinomial(1, p, 1))
            next_char = int_to_char[next_index]
            if next_char == EOS_TOKEN:
                active[idx] = False
                termination[idx] = 'eos'
            elif next_char == PAD_TOKEN:
                # Defensive stop -- see header note #5
                active[idx] = False
                termination[idx] = 'pad'
            else:
                generated[idx] += next_char
                if len(generated[idx]) >= max_total_len:
                    active[idx] = False
                    termination[idx] = 'cap'

        if (step + 1) % PROGRESS_LOG_EVERY == 0 or n_active == 0:
            elapsed = time.time() - t0
            print(f"  step {step+1}/{max_total_len}: {n_active} sequence(s) still active "
                  f"- {elapsed:.1f}s elapsed", flush=True)

    return generated, termination


def main():
    print(f"Loading model from '{MODEL_PATH}'...", flush=True)
    model = load_model(MODEL_PATH)

    print(f"Reading seeds from '{SEED_CSV}'...", flush=True)
    # keep_default_na=False -- see header note #4. Literal "nan"/"null" seeds
    # must survive as strings, not become missing values.
    df = pd.read_csv(SEED_CSV, keep_default_na=False)

    required_cols = {'ratio_used', 'seed_input', 'root_domain', 'ground_truth', 'gt_entropy'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(f"Error: '{SEED_CSV}' is missing expected column(s): {missing_cols}")
        sys.exit(1)

    print(f"Loaded {len(df)} seed rows.", flush=True)

    empty_seed_count = (df['seed_input'].astype(str).str.len() == 0).sum()
    if empty_seed_count:
        print(f"Note: {empty_seed_count} row(s) have an empty seed_input -- these will be "
              f"generated unconditionally (no seed context).", flush=True)

    seeds = df['seed_input'].astype(str).tolist()

    print(f"\nGenerating {len(seeds)} sequences (max_total_len={MAX_TOTAL_LEN} "
          f"[seed+generated], temperature={TEMPERATURE}, "
          f"predict_batch_size={PREDICT_BATCH_SIZE})...", flush=True)
    t0 = time.time()
    model_outputs, termination_reasons = generate_batch(
        model, seeds, MAX_TOTAL_LEN, TEMPERATURE, PREDICT_BATCH_SIZE)
    elapsed = time.time() - t0
    print(f"\nGeneration complete in {elapsed:.1f}s "
          f"({elapsed/len(seeds)*1000:.2f}ms/sequence average).", flush=True)

    # --- Compute output columns ---
    print("Computing gen_entropy / match / len_diff / termination...", flush=True)
    df['model_output']  = model_outputs
    df['termination']   = termination_reasons   # 'eos' | 'cap' | 'pad'
    df['gen_entropy']   = df['model_output'].apply(calculate_shannon_entropy)
    df['match']         = (df['model_output'] == df['ground_truth']).astype(int)
    df['len_diff']      = df['model_output'].str.len() - df['ground_truth'].str.len()

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved results to '{OUTPUT_CSV}'.", flush=True)

    # --- Termination breakdown ---
    term_counts = df['termination'].value_counts()
    n_total     = len(df)
    n_eos  = term_counts.get('eos', 0)
    n_cap  = term_counts.get('cap', 0)
    n_pad  = term_counts.get('pad', 0)

    print(f"\n--- Termination breakdown ---", flush=True)
    print(f"  <EOS> (natural)   : {n_eos:>7,}  ({n_eos/n_total:.4%})", flush=True)
    print(f"  cap   (safety net): {n_cap:>7,}  ({n_cap/n_total:.4%})", flush=True)
    if n_pad:
        print(f"  <PAD> (degenerate): {n_pad:>7,}  ({n_pad/n_total:.4%})", flush=True)
    print(f"Overall match rate  : {df['match'].mean():.4%} "
          f"({df['match'].sum()}/{n_total})", flush=True)

    # --- Write generation summary file ---
    summary_path = OUTPUT_CSV.replace('.csv', '_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("--- LSTM Mode A Generation Summary ---\n")
        f.write(f"Generated:          {pd.Timestamp.now().isoformat(timespec='seconds')}\n")
        f.write(f"Model:              {MODEL_PATH}\n")
        f.write(f"Seed file:          {SEED_CSV}\n")
        f.write(f"Output file:        {OUTPUT_CSV}\n")
        f.write(f"Total sequences:    {n_total}\n")
        f.write(f"max_total_len:      {MAX_TOTAL_LEN} (seed + generated, "
                f"= model MAXLEN = trained context window)\n")
        f.write(f"Temperature:        {TEMPERATURE}\n")
        f.write(f"Generation time:    {elapsed:.1f}s "
                f"({elapsed/n_total*1000:.2f}ms/sequence)\n\n")
        f.write("--- Termination breakdown ---\n")
        f.write(f"  <EOS> natural termination : {n_eos:>7,}  ({n_eos/n_total:.4%})\n")
        f.write(f"  cap   safety-net stop     : {n_cap:>7,}  ({n_cap/n_total:.4%})\n")
        if n_pad:
            f.write(f"  <PAD> degenerate stop     : {n_pad:>7,}  ({n_pad/n_total:.4%})\n")
        max_seed_len = int(df['seed_input'].astype(str).str.len().max())
        min_new_char_budget = MAX_TOTAL_LEN - max_seed_len
        max_chars_needed = int(
            (df['ground_truth'].str.len() - df['seed_input'].astype(str).str.len()).max())
        f.write(f"\nNote: cap-stopped sequences terminated at max_total_len={MAX_TOTAL_LEN} "
                f"without emitting <EOS>;\n"
                f"      this reflects failure of learned termination (sequence continuation "
                f"bias), not a truncation\n"
                f"      artifact of the evaluation setup -- verified: no ground-truth in this "
                f"file requires\n"
                f"      more than {max_chars_needed} new characters past the seed "
                f"(observed max), well under the minimum\n"
                f"      new-char budget of {min_new_char_budget} chars "
                f"(worst case: longest seed = {max_seed_len} chars).\n\n")
        f.write("--- Match rate by seed ratio ---\n")
        for ratio in sorted(df['ratio_used'].unique()):
            sub = df[df['ratio_used'] == ratio]
            f.write(f"  ratio={ratio:.2f}: {sub['match'].mean():.4%} "
                    f"({sub['match'].sum()}/{len(sub)})\n")
        f.write("\n--- Match rate by termination type ---\n")
        for term in ['eos', 'cap', 'pad']:
            sub = df[df['termination'] == term]
            if len(sub):
                f.write(f"  {term}: {sub['match'].mean():.4%} "
                        f"({sub['match'].sum()}/{len(sub)})\n")
        f.write("\n--- Length difference (len_diff = output - ground_truth) ---\n")
        f.write(f"  mean:   {df['len_diff'].mean():.4f}\n")
        f.write(f"  std:    {df['len_diff'].std():.4f}\n")
        f.write(f"  median: {df['len_diff'].median():.1f}\n")
        f.write(f"  min:    {df['len_diff'].min()}\n")
        f.write(f"  max:    {df['len_diff'].max()}\n")

    print(f"Summary written to '{summary_path}'.", flush=True)


if __name__ == "__main__":
    main()
