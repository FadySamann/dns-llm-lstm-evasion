import re
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
import gc
import math
import os
import sys

# ---------------------------------------------------------------------------
# lstm_model_v14.py
#
# Difference from v13:
#
# Parsing moved from per-example (tf.io.parse_single_example, called before
# .batch()) to batched (tf.io.parse_example, called after .batch()).
#
# Evidence this was the real bottleneck, not the shuffle buffer: dropping
# SHUFFLE_BUFFER_SIZE 20x (2,000,000 -> 100,000, v12 -> v13) produced NO
# measurable change --  GPU-Util held at 62% in both live nvidia-smi
# snapshots, and the training process stayed at ~190-200% CPU (~2 cores) in
# both `top` snapshots, taken under otherwise-matching conditions (no swap,
# no I/O wait, same machine). If the shuffle buffer refill were the limiter,
# a 20x smaller buffer should have shown a clear speed change. It didn't --
# which points at fixed per-call overhead in the parsing step itself as the
# actual constraint, since v11-v13 all called tf.io.parse_single_example()
# once per individual record (~29.4M calls/epoch) before batching, rather
# than once per batch (~230K calls/epoch). Each parse_single_example() call
# carries graph-dispatch and tensor-allocation overhead that is amortized
# far better when parsing a whole batch (128 records) in one call via
# tf.io.parse_example(), instead of paying that overhead 128x more often.
#
# Verified equivalence: tf.io.parse_example() on a batch of serialized
# examples was checked against tf.io.parse_single_example() + .batch() on
# the same synthetic records and produces byte-identical x/y tensors --
# this is a throughput change only, not a change to what data reaches the
# model. SHUFFLE_BUFFER_SIZE is left at v13's 100,000 since that change was
# harmless even though it wasn't the fix.
#
# If GPU-Util rises meaningfully above 62% and epoch time drops in this
# version, that confirms parsing was the bottleneck. If not, the next place
# to look is TFRecordDataset's own read buffer_size (currently left at
# TensorFlow's default, ~256KiB) or num_parallel_calls tuning.
# ---------------------------------------------------------------------------

# --- Configuration for Multiple Runs ---
NUM_RUNS = 10
BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 100_000  # unchanged from v13 -- ruled out as the bottleneck, but harmless to keep

# --- File Paths for Local Execution ---
TRAIN_TFRECORD = 'train.tfrecord'
VAL_TFRECORD = 'val.tfrecord'
DATAPREP_SUMMARY_FILE = 'dataprepare_v6_summary.txt'
BEST_MODEL_FILE = 'LSTM_subdomian_generator_best_overall_model.keras'
RESULTS_CSV_FILE = 'LSTM_subdomain_training_run_results.csv'
SUMMARY_FILE = 'LSTM_subdomain_training_summary.txt'


def load_dataprep_metadata(summary_path: str) -> dict:
    """
    Parses maxlen, vocab_size, and per-split example counts directly out of
    dataprepare_v6.py's summary file. Aborts with a clear error rather than
    silently falling back to a guessed/stale value if the file is missing or
    a field can't be found -- this metadata directly controls
    steps_per_epoch/validation_steps and the TFRecord parsing schema, so a
    wrong value fails silently (wrong epoch coverage) rather than loudly.
    """
    if not os.path.exists(summary_path):
        print(f"Error: '{summary_path}' not found. Run dataprepare_v6.py first, "
              f"or point DATAPREP_SUMMARY_FILE at the correct summary.")
        sys.exit(1)

    with open(summary_path, "r", encoding="utf-8") as f:
        text = f.read()

    patterns = {
        "maxlen": r"maxlen \(from train domains only\):\s*(\d+)",
        "vocab_size": r"vocab_size:\s*(\d+)",
        "num_train_samples": r"train\.tfrecord'?:\s*(\d+)\s*examples",
        "num_val_samples": r"val\.tfrecord'?:\s*(\d+)\s*examples",
        "num_test_samples": r"test\.tfrecord'?:\s*(\d+)\s*examples",
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
        print(f"Error: could not find the following field(s) in '{summary_path}': {missing}")
        print("The summary format may have changed -- check it against dataprepare_v6.py's output.")
        sys.exit(1)

    return metadata


def load_completed_runs(results_csv: str) -> tuple:
    """
    Reads RESULTS_CSV_FILE (if it exists) and returns:
      - a set of run numbers already completed (to skip on resume)
      - the best val_accuracy already on record (0.0 if none), so a fresh
        process invocation doesn't reset "best" tracking and risk
        overwriting BEST_MODEL_FILE with a worse model after a restart.
    """
    if not os.path.exists(results_csv):
        return set(), 0.0

    try:
        existing_df = pd.read_csv(results_csv)
    except pd.errors.EmptyDataError:
        return set(), 0.0

    if existing_df.empty or 'run' not in existing_df.columns:
        return set(), 0.0

    completed = set(existing_df['run'].astype(int).unique())
    best_val_acc = float(existing_df['val_accuracy'].max()) if 'val_accuracy' in existing_df.columns else 0.0
    return completed, best_val_acc


# --- Load Metadata (parsed, not hardcoded) ---
_metadata = load_dataprep_metadata(DATAPREP_SUMMARY_FILE)
maxlen = _metadata["maxlen"]
vocab_size = _metadata["vocab_size"]
num_train_samples = _metadata["num_train_samples"]
num_val_samples = _metadata["num_val_samples"]
num_test_samples = _metadata["num_test_samples"]

print(f"Loaded data-prep metadata from '{DATAPREP_SUMMARY_FILE}':")
print(f"  maxlen={maxlen}, vocab_size={vocab_size}")
print(f"  num_train_samples={num_train_samples}, num_val_samples={num_val_samples}, "
      f"num_test_samples={num_test_samples}")

steps_per_epoch = math.ceil(num_train_samples / BATCH_SIZE)
validation_steps = math.ceil(num_val_samples / BATCH_SIZE)

# --- Resume check ---
completed_runs, best_overall_val_accuracy = load_completed_runs(RESULTS_CSV_FILE)
if completed_runs:
    print(f"\nFound {len(completed_runs)} already-completed run(s) in '{RESULTS_CSV_FILE}': "
          f"{sorted(completed_runs)}")
    print(f"Resuming -- these will be skipped. Best val_accuracy on record so far: "
          f"{best_overall_val_accuracy:.4f}")
else:
    print("\nNo prior results found -- starting fresh.")

# --- TFRecord Parsing Function (BATCHED -- see header note) ---
_feature_description = {
    'x': tf.io.FixedLenFeature([maxlen], tf.int64),
    'y': tf.io.FixedLenFeature([vocab_size], tf.float32),
}

def parse_tfrecord_batch_fn(examples):
    """Parses a BATCH of serialized examples in one call via
    tf.io.parse_example, instead of one call per individual record. Must be
    applied AFTER .batch() in the pipeline, not before."""
    parsed = tf.io.parse_example(examples, _feature_description)
    x = tf.cast(parsed['x'], tf.int32)
    return x, parsed['y']

# --- Main Experiment Loop ---
all_run_results = []

for i in range(NUM_RUNS):
    run_num = i + 1

    if run_num in completed_runs:
        print(f"\nRun {run_num}/{NUM_RUNS} already completed -- skipping.")
        continue

    print("\n" + "="*50)
    print(f"STARTING RUN {run_num}/{NUM_RUNS}")
    print("="*50)
    
    # 1. Build and compile a fresh model
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=100),
        LSTM(units=256, return_sequences=True),
        Dropout(0.2),
        LSTM(units=256), #return_sequences=True),
        Dropout(0.2),
        Dense(units=vocab_size, activation='softmax')
    ])
    optimizer = Adam(learning_rate=0.0005) #Use 0.0005 instead of the default 0.001
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # 2. Set up datasets using TFRecord
    # MODIFIED (v14): .batch() now happens BEFORE .map() so parsing is done
    # once per batch (tf.io.parse_example) instead of once per record
    # (tf.io.parse_single_example) -- see header note for the evidence.
    train_dataset = tf.data.TFRecordDataset(TRAIN_TFRECORD)
    train_dataset = train_dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.map(parse_tfrecord_batch_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.repeat().prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.TFRecordDataset(VAL_TFRECORD)
    # Validation order doesn't affect metrics, so no shuffle needed here.
    val_dataset = val_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.map(parse_tfrecord_batch_fn, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.repeat().prefetch(tf.data.AUTOTUNE)
    
    # 3. Define callbacks for this run
    temp_model_file = f'subdomain_temp_best_model_run_{run_num}.keras'
    checkpoint = ModelCheckpoint(filepath=temp_model_file, save_best_only=True, monitor='val_accuracy', mode='max', verbose=0)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # 4. Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[checkpoint, early_stopping],
        verbose=1 
    )

    # 5. Record the results
    best_epoch_index = np.argmax(history.history['val_accuracy'])
    run_best_val_acc = history.history['val_accuracy'][best_epoch_index]
    run_train_acc = history.history['accuracy'][best_epoch_index]
    
    current_run_data = {
        'run': run_num,
        'train_accuracy': run_train_acc,
        'val_accuracy': run_best_val_acc
    }
    all_run_results.append(current_run_data)
    
    # --- Incremental Saving ---
    # Append the result of this run to the CSV immediately. Safe from
    # duplication on resume because completed runs are skipped above before
    # this point is ever reached for them.
    run_df = pd.DataFrame([current_run_data])
    header_needed = not os.path.exists(RESULTS_CSV_FILE)
    run_df.to_csv(RESULTS_CSV_FILE, mode='a', index=False, header=header_needed)
    print(f"Metrics for Run {run_num} saved to '{RESULTS_CSV_FILE}'.")
    
    # 6. Check if this is the best model overall (best_overall_val_accuracy
    # was seeded from prior completed runs on resume, not reset to 0.0)
    if run_best_val_acc > best_overall_val_accuracy:
        best_overall_val_accuracy = run_best_val_acc
        print(f"New best model found with val_accuracy: {run_best_val_acc:.4f}. Saving...")
        if os.path.exists(temp_model_file):
            os.rename(temp_model_file, BEST_MODEL_FILE)

    if os.path.exists(temp_model_file):
        os.remove(temp_model_file)
    
    # --- Release Resources ---
    print(f"Run {run_num} complete. Releasing resources...")
    K.clear_session()
    gc.collect()

# --- Final Step: Summarize and Save All Results ---
print("\n" + "="*50)
print("All training runs complete.")
print("="*50)

# Read the full results from disk rather than only in-memory all_run_results,
# so the summary reflects every run ever completed (including runs finished
# in a prior process invocation before a crash/restart), not just this run
# of the script.
if os.path.exists(RESULTS_CSV_FILE):
    results_df = pd.read_csv(RESULTS_CSV_FILE)
elif all_run_results:
    results_df = pd.DataFrame(all_run_results)
else:
    results_df = pd.DataFrame(columns=['run', 'train_accuracy', 'val_accuracy'])

if not results_df.empty:
    summary_stats = {
        'Metric': ['Training Accuracy', 'Validation Accuracy'],
        'Average': [results_df['train_accuracy'].mean(), results_df['val_accuracy'].mean()],
        'Std Deviation': [results_df['train_accuracy'].std(), results_df['val_accuracy'].std()]
    }
    summary_df = pd.DataFrame(summary_stats)
    
    print("\n--- Overall Performance Summary ---")
    print(f"(Based on {len(results_df)} completed run(s) across all process invocations)")
    print(summary_df)
    
    with open(SUMMARY_FILE, 'w') as f:
        f.write("Overall Performance Summary\n" + "="*30 + "\n")
        f.write(f"Total Runs Completed: {len(results_df)}\n\n")
        f.write(summary_df.to_string(index=False))
    
    print(f"\nSummary saved to '{SUMMARY_FILE}'")
    print(f"Detailed results saved to '{RESULTS_CSV_FILE}'")
    print(f"The single best model was saved to '{BEST_MODEL_FILE}'")
else:
    print("No training runs were completed.")
