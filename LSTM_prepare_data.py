import sys
import os
import string
import argparse
import datetime
import time
import tensorflow as tf

# ---------------------------------------------------------------------------
# dataprepare_v6.py
#
# Difference from v5: identical logic and output (same OOV hard-check, same
# domain-level pre-split consumption, same TFRecord schema) but the inner
# per-domain encoding loop no longer calls Keras's pad_sequences()/
# to_categorical(). Those are batch-oriented utilities with real per-call
# overhead (type coercion, array allocation, assertions) that becomes
# significant when called once per domain across hundreds of thousands of
# domains rather than once per large batch. v6 does the padding and
# one-hot encoding with plain Python/list ops instead, which is much
# cheaper at this call volume.
#
# v6 also adds periodic, FLUSHED progress logging (domain count, example
# count, elapsed time, ETA) inside process_and_write_tfrecords(), since v5
# printed nothing between "Writing N domains..." and completion -- on a
# slow run that looks identical to a hang from the outside.
# ---------------------------------------------------------------------------

VOCAB_CHARS = ['<PAD>'] + list(string.ascii_lowercase) + list(string.digits) + ['_', '-', '.'] + ['<EOS>']
CHAR_TO_INT = {char: i for i, char in enumerate(VOCAB_CHARS)}
VOCAB_SIZE = len(VOCAB_CHARS)
VALID_CHARS = set(string.ascii_lowercase) | set(string.digits) | {'_', '-', '.'}


def read_domains(filename: str) -> list:
    """Reads a pre-split domain file (one FQDN per line) and lowercases each line."""
    print(f"Reading domains from '{filename}'...", flush=True)
    if not os.path.exists(filename):
        print(f"Error: The file '{filename}' was not found.", flush=True)
        sys.exit(1)

    domains = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            domain = line.strip().lower()
            if domain:
                domains.append(domain)
    return domains


def find_oov_domains(domains: list, split_name: str) -> list:
    """
    Returns a list of (domain, offending_chars) for every domain that
    contains at least one character outside the LSTM vocabulary. Does not
    modify or drop anything -- flagging only.
    """
    oov = []
    for domain in domains:
        bad_chars = sorted(set(c for c in domain if c not in VALID_CHARS))
        if bad_chars:
            oov.append((domain, bad_chars))
    if oov:
        print(f"  -> [{split_name}] Found {len(oov)} domain(s) with out-of-vocabulary characters.", flush=True)
    return oov


def report_and_abort_on_oov(oov_by_split: dict, report_path: str):
    """
    Writes every flagged domain (across all splits) to a text report and
    aborts the script. Called only if at least one OOV domain was found.
    """
    total = sum(len(v) for v in oov_by_split.values())
    print(f"\nERROR: {total} domain(s) contain characters outside the LSTM vocabulary.", flush=True)
    print(f"Full list written to '{report_path}'. Fix or filter the source data and rerun.", flush=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("--- Out-of-Vocabulary Domain Report (dataprepare_v6.py) ---\n")
        f.write(f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Vocabulary allows: a-z, 0-9, '_', '-', '.'\n\n")
        for split_name, oov in oov_by_split.items():
            if not oov:
                continue
            f.write(f"--- {split_name} ({len(oov)} flagged) ---\n")
            for domain, bad_chars in oov:
                f.write(f"{domain}\t offending_chars={bad_chars}\n")
            f.write("\n")

    sys.exit(1)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def process_and_write_tfrecords(domains: list, filename: str, maxlen: int,
                                 char_to_int: dict, vocab_size: int,
                                 log_every: int = 25000) -> int:
    """
    Streams each domain's (prefix -> next-char) training pairs straight to a
    TFRecord file, one domain at a time, so the full expanded dataset is
    never held in memory. Padding and one-hot encoding are done manually
    (no pad_sequences/to_categorical calls) to avoid per-call Keras overhead
    at this volume. Prints flushed progress every `log_every` domains.
    Returns the total number of examples written.
    """
    total_domains = len(domains)
    print(f"Writing {total_domains} domains to '{filename}'...", flush=True)
    count = 0
    start_time = time.time()

    with tf.io.TFRecordWriter(filename) as writer:
        for domain_idx, sub in enumerate(domains, 1):
            encoded_sub_part = [char_to_int[c] for c in sub]
            encoded_sub = encoded_sub_part + [char_to_int['<EOS>']]

            for i in range(1, len(encoded_sub)):
                seq = encoded_sub[:i]

                # Manual pre-padding / pre-truncating to maxlen (equivalent
                # to pad_sequences(..., padding='pre', truncating='pre')).
                seq_len = len(seq)
                if seq_len < maxlen:
                    x_arr = [0] * (maxlen - seq_len) + seq
                elif seq_len > maxlen:
                    x_arr = seq[-maxlen:]
                else:
                    x_arr = seq

                # Manual one-hot encoding (equivalent to to_categorical).
                y_idx = encoded_sub[i]
                y_arr = [0.0] * vocab_size
                y_arr[y_idx] = 1.0

                feature = {
                    'x': _int64_feature(x_arr),
                    'y': _float_feature(y_arr),
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
                count += 1

            if domain_idx % log_every == 0 or domain_idx == total_domains:
                elapsed = time.time() - start_time
                rate = domain_idx / elapsed if elapsed > 0 else 0.0
                remaining = total_domains - domain_idx
                eta_s = remaining / rate if rate > 0 else float('nan')
                print(f"  ... {domain_idx}/{total_domains} domains "
                      f"({count} examples written) - {elapsed:.1f}s elapsed, "
                      f"~{eta_s:.1f}s remaining", flush=True)

    print(f"Successfully wrote {count} total examples to '{filename}'.", flush=True)
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Convert the pre-split train/val/test domain files from "
                    "combine_and_shuffle_v4.py/v5.py into TFRecord files for LSTM training."
    )
    parser.add_argument("--train", default="train_domains.txt", help="Path to train split (default: train_domains.txt)")
    parser.add_argument("--val", default="val_domains.txt", help="Path to val split (default: val_domains.txt)")
    parser.add_argument("--test", default="test_domains.txt", help="Path to test split (default: test_domains.txt)")
    parser.add_argument("--outdir", default=".", help="Directory to write TFRecord + summary files (default: current dir)")
    parser.add_argument("--summary", default="dataprepare_v6_summary.txt", help="Summary filename (default: dataprepare_v6_summary.txt)")
    parser.add_argument("--oov-report", default="oov_domains_report.txt", help="Filename for the OOV report if any flagged domains are found (default: oov_domains_report.txt)")
    parser.add_argument("--log-every", type=int, default=25000, help="Print progress every N domains per split (default: 25000)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- Step 1: Read the three pre-split files (no re-splitting) ---
    train_domains = read_domains(args.train)
    val_domains = read_domains(args.val)
    test_domains = read_domains(args.test)

    for name, domains in (("train", train_domains), ("val", val_domains), ("test", test_domains)):
        if not domains:
            print(f"Error: '{name}' split is empty.", flush=True)
            sys.exit(1)

    # --- Step 1b: Flag (don't silently filter) any out-of-vocabulary domains ---
    print("\nChecking for out-of-vocabulary characters...", flush=True)
    oov_by_split = {
        "train": find_oov_domains(train_domains, "train"),
        "val": find_oov_domains(val_domains, "val"),
        "test": find_oov_domains(test_domains, "test"),
    }
    if any(oov_by_split.values()):
        oov_report_path = os.path.join(args.outdir, args.oov_report)
        report_and_abort_on_oov(oov_by_split, oov_report_path)
    print("No out-of-vocabulary characters found. Proceeding.", flush=True)

    # --- Step 2: maxlen from TRAINING domains only (unchanged from v4/v5) ---
    maxlen = max(len(sub) for sub in train_domains)
    print(f"Max sequence length (maxlen) will be: {maxlen}", flush=True)

    # --- Step 3: Stream each split straight to its own TFRecord file ---
    print("\n--- Saving data to TFRecord format ---", flush=True)
    train_tfrecord = os.path.join(args.outdir, "train.tfrecord")
    val_tfrecord = os.path.join(args.outdir, "val.tfrecord")
    test_tfrecord = os.path.join(args.outdir, "test.tfrecord")

    t0 = time.time()
    train_count = process_and_write_tfrecords(train_domains, train_tfrecord, maxlen, CHAR_TO_INT, VOCAB_SIZE, args.log_every)
    val_count = process_and_write_tfrecords(val_domains, val_tfrecord, maxlen, CHAR_TO_INT, VOCAB_SIZE, args.log_every)
    test_count = process_and_write_tfrecords(test_domains, test_tfrecord, maxlen, CHAR_TO_INT, VOCAB_SIZE, args.log_every)
    total_elapsed = time.time() - t0

    # --- Step 4: Write summary file ---
    summary_path = os.path.join(args.outdir, args.summary)
    print(f"\nWriting summary to '{summary_path}'...", flush=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("--- LSTM Data Preparation Summary (dataprepare_v6.py) ---\n")
        f.write(f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Total TFRecord writing time: {total_elapsed:.1f}s\n\n")

        f.write("Input splits (pre-split upstream, shared with LLM pipeline):\n")
        f.write(f"  Train file: '{args.train}'\n")
        f.write(f"  Val file:   '{args.val}'\n")
        f.write(f"  Test file:  '{args.test}'\n\n")

        f.write("Domain counts (no domains dropped -- OOV check passed with 0 flagged):\n")
        f.write(f"  Train: {len(train_domains)} domains\n")
        f.write(f"  Val:   {len(val_domains)} domains\n")
        f.write(f"  Test:  {len(test_domains)} domains\n\n")

        f.write("Vocabulary:\n")
        f.write(f"  vocab_size: {VOCAB_SIZE}\n")
        f.write(f"  maxlen (from train domains only): {maxlen}\n\n")

        f.write("TFRecord output:\n")
        f.write(f"  '{train_tfrecord}': {train_count} examples\n")
        f.write(f"  '{val_tfrecord}': {val_count} examples\n")
        f.write(f"  '{test_tfrecord}': {test_count} examples\n")

    print("\n" + "=" * 50, flush=True)
    print("Data preparation and TFRecord conversion complete!", flush=True)
    print(f"Total writing time: {total_elapsed:.1f}s", flush=True)
    print("\nIMPORTANT: Use these values in your training script:", flush=True)
    print(f"  maxlen: {maxlen}", flush=True)
    print(f"  vocab_size: {VOCAB_SIZE}", flush=True)
    print(f"  num_train_samples: {train_count}", flush=True)
    print(f"  num_val_samples: {val_count}", flush=True)
    print(f"  num_test_samples: {test_count}", flush=True)
    print("=" * 50, flush=True)


if __name__ == "__main__":
    main()
