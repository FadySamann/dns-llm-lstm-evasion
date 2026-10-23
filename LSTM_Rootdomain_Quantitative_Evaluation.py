import numpy as np
import tensorflow as tf
import string
import math
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Configuration ---
MODEL_PATH = 'rootdomain_generator_best_overall_model.keras'
TEST_TFRECORD = 'rootdomain_test.tfrecord'
OUTPUT_FILE = 'rootdomain_test_evaluation_results_withNgrams.txt'

BATCH_SIZE = 128
MAXLEN = 75
VOCAB_SIZE = 41

# --- Recreate Character Mapping ---
all_chars = ['<PAD>'] + list(string.ascii_lowercase) + list(string.digits) + ['_', '-', '.'] + ['<EOS>']
char_to_int = {char: i for i, char in enumerate(all_chars)}
int_to_char = {i: char for i, char in enumerate(all_chars)}

# --- TFRecord Parsing ---
def parse_tfrecord_fn(example):
    feature_description = {
        'x': tf.io.FixedLenFeature([MAXLEN], tf.int64),
        'y': tf.io.FixedLenFeature([VOCAB_SIZE], tf.float32),
    }
    example = tf.io.parse_single_example(example, feature_description)
    return tf.cast(example['x'], tf.int32), example['y']

# --- Sampling & Generation ---
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-10) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_fqdn(model, seed_text, max_gen_len=60, temperature=1.0):
    generated = seed_text.lower()
    for _ in range(max_gen_len):
        encoded_seq = [char_to_int[c] for c in generated if c in char_to_int]
        padded_seq = pad_sequences([encoded_seq], maxlen=MAXLEN, padding='pre')
        preds = model.predict(padded_seq, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = int_to_char[next_index]
        if next_char == '<EOS>':
            break
        generated += next_char
    return generated

# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
    else:
        # Open file for writing results
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("=== FQDN GENERATOR TEST REPORT ===\n\n")
            
            # 1. Load Model
            print(f"Loading model: {MODEL_PATH}...")
            model = load_model(MODEL_PATH)
            f.write(f"Model File: {MODEL_PATH}\n")

            # 2. Quantitative Evaluation
            """if os.path.exists(TEST_TFRECORD):
                print(f"Evaluating {TEST_TFRECORD}...")
                test_dataset = tf.data.TFRecordDataset(TEST_TFRECORD)
                test_dataset = test_dataset.map(parse_tfrecord_fn).batch(BATCH_SIZE)
                
                loss, accuracy = model.evaluate(test_dataset, verbose=1)
                
                eval_str = f"Test Loss: {loss:.4f}\nTest Accuracy: {accuracy:.4f}\n"
                print("\n" + eval_str)
                f.write("\n--- Quantitative Metrics ---\n")
                f.write(eval_str)
            else:
                f.write("\nWarning: test.tfrecord not found. Skipping metrics.\n")"""

            # 3. Qualitative Generation
            print("Generating inference samples...")
            f.write("\n--- Inference Samples (Qualitative Test) ---\n")
            f.write("Generated using different temperatures (higher = more creative)\n")
            
            ngrams_seeds = ["co", "in", "de", "re", "ne","er", "et", "al", "es", "en","or", "an", "on", "ar", "te", "st", "com", "net", "org", "ing", "lin"]
            test_seeds = ["cloud-", "api-", "server.", "dev-", "uk-", "internal-",
                        # Numeric / Cluster Seeds
                        "123-",      # Common for numbered server clusters
                        "001-",      # Zero-padded machine identifiers
                        "2025-",     # Year-based project or log prefixes
                        "10.",       # Internal IP-style or versioning starts
                        "99",        # High-range numeric identifiers
                        "v1-",        # Versioned API service style
                        "srv1-",      # Standard server numbering
                        "us-west-2",  # AWS/Cloud region style
                        "a1-",         # Short cluster prefix
                        "v2.api.",    # Versioned API subdomains
                        "node-00",    # Machine node identifiers
                        "x7",         # Start of a potential hex-code/hash
                        "prod-v3-"    # Environment + versioning
                        ]
            temperatures = [0.2, 0.5, 0.8, 1.0, 1.2]

            for temp in temperatures:
                header = f"\nTemperature: {temp}\n" + "-"*30 + "\n"
                print(header.strip())
                f.write(header)
                
                for seed in ngrams_seeds:
                    prediction = generate_fqdn(model, seed, temperature=temp)
                    line = f"Seed: '{seed:<10}' -> Predicted: {prediction}\n"
                    print(line.strip())
                    f.write(line)

        print(f"\nSuccess! Results have been saved to '{OUTPUT_FILE}'")
