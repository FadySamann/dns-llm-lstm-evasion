import numpy as np
import tensorflow as tf
import string
import math
import pandas as pd
import os
from tensorflow.keras.models import load_model

# --- Configuration ---
MODEL_PATH = 'LSTM_subdomian_generator_best_overall_model.keras'
TEST_TFRECORD = 'test.tfrecord'
METRICS_OUTPUT = 'LSTM_subdomain_test_quantitative_resultsV2.csv'

MAXLEN = 85
VOCAB_SIZE = 41
BATCH_SIZE = 128

# --- TFRecord Parsing Function ---
def parse_tfrecord_fn(example):
    feature_description = {
        'x': tf.io.FixedLenFeature([MAXLEN], tf.int64),
        'y': tf.io.FixedLenFeature([VOCAB_SIZE], tf.float32),
    }
    example = tf.io.parse_single_example(example, feature_description)
    return tf.cast(example['x'], tf.int32), example['y']

# --- Quantitative Evaluation ---
def run_quantitative_evaluation():
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    
    print(f"Loading test dataset from {TEST_TFRECORD}...")
    test_dataset = tf.data.TFRecordDataset(TEST_TFRECORD)
    test_dataset = test_dataset.map(parse_tfrecord_fn).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print("Executing model evaluation on test set...")
    results = model.evaluate(test_dataset, verbose=1)
    
    # results is a list: [loss, accuracy]
    test_loss = results[0]
    test_accuracy = results[1]
    
    print(f"\nTest Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_accuracy:.4f}")

    # Save results for paper documentation
    metrics_df = pd.DataFrame({
        'Metric': ['Test Loss', 'Test Accuracy'],
        'Value': [test_loss, test_accuracy]
    })
    metrics_df.to_csv(METRICS_OUTPUT, index=False)
    print(f"Metrics saved to {METRICS_OUTPUT}")

if __name__ == "__main__":
    run_quantitative_evaluation()
