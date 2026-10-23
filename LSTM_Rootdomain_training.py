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

# --- Configuration for Multiple Runs ---
NUM_RUNS = 3
BATCH_SIZE = 128

# --- File Paths for Local Execution ---
TRAIN_TFRECORD = 'rootdomain_train.tfrecord'
VAL_TFRECORD = 'rootdomain_val.tfrecord'
BEST_MODEL_FILE = 'rootdomain_generator_best_overall_model.keras'
RESULTS_CSV_FILE = 'LSTM_rootdomain_training_run_results.csv'
SUMMARY_FILE = 'LSTM_rootdomain_training_summary.txt'

# --- Load Metadata ---
# These values are hardcoded from your data preparation script's output
maxlen = 75
vocab_size = 41
num_train_samples = 9543381
num_val_samples = 2045010

steps_per_epoch = math.ceil(num_train_samples / BATCH_SIZE)
validation_steps = math.ceil(num_val_samples / BATCH_SIZE)

# --- TFRecord Parsing Function ---
def parse_tfrecord_fn(example):
    feature_description = {
        'x': tf.io.FixedLenFeature([maxlen], tf.int64),
        'y': tf.io.FixedLenFeature([vocab_size], tf.float32),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example['x'] = tf.cast(example['x'], tf.int32)
    return example['x'], example['y']

# --- Main Experiment Loop ---
all_run_results = []
best_overall_val_accuracy = 0.0

for i in range(NUM_RUNS):
    run_num = i + 1
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
    train_dataset = tf.data.TFRecordDataset(TRAIN_TFRECORD)
    train_dataset = train_dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    # MODIFIED: Added .repeat() to make the dataset loop
    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.TFRecordDataset(VAL_TFRECORD)
    val_dataset = val_dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    # MODIFIED: Added .repeat() here as well
    val_dataset = val_dataset.batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
    
    # 3. Define callbacks for this run
    temp_model_file = f'rootdomain_temp_best_model_run_{run_num}.keras'
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
    
    # --- NEW: Incremental Saving ---
    # Append the result of this run to the CSV immediately
    run_df = pd.DataFrame([current_run_data])
    # Write header only for the first run
    header_needed = not os.path.exists(RESULTS_CSV_FILE)
    run_df.to_csv(RESULTS_CSV_FILE, mode='a', index=False, header=header_needed)
    print(f"Metrics for Run {run_num} saved to '{RESULTS_CSV_FILE}'.")
    
    # 6. Check if this is the best model overall
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

if all_run_results:
    results_df = pd.DataFrame(all_run_results)
    summary_stats = {
        'Metric': ['Training Accuracy', 'Validation Accuracy'],
        'Average': [results_df['train_accuracy'].mean(), results_df['val_accuracy'].mean()],
        'Std Deviation': [results_df['train_accuracy'].std(), results_df['val_accuracy'].std()]
    }
    summary_df = pd.DataFrame(summary_stats)
    
    print("\n--- Overall Performance Summary ---")
    print(summary_df)
    
    results_df.to_csv(RESULTS_CSV_FILE, index=False)
    with open(SUMMARY_FILE, 'w') as f:
        f.write("Overall Performance Summary\n" + "="*30 + "\n")
        f.write(f"Total Runs Completed: {len(all_run_results)}\n\n")
        f.write(summary_df.to_string(index=False))
    
    print(f"\nSummary saved to '{SUMMARY_FILE}'")
    print(f"Detailed results saved to '{RESULTS_CSV_FILE}'")
    print(f"The single best model was saved to '{BEST_MODEL_FILE}'")
else:
    print("No training runs were completed.")
