import argparse
import math
import os
import sys
from collections import Counter
import pandas as pd

# --- Constants for Output Files ---
# OUT_PREFIX is set from --prefix in main() so LLM and LSTM runs do not overwrite each other.
OUT_PREFIX = ''
FEATURES_CSV = 'FQDN_feature_analysis.csv'
COMMON_NGRAMS_CSV = 'FQDN_common_2gram_ratios.csv'


# --- Feature Calculation Functions ---

def calculate_shannon_entropy(text: str) -> float:
    """Calculates the Shannon Entropy of a given string."""
    if not text:
        return 0.0
    char_counts = Counter(text)
    text_length = len(text)
    entropy = 0.0
    for count in char_counts.values():
        probability = count / text_length
        entropy -= probability * math.log2(probability)
    return entropy

def calculate_character_ratios(text: str) -> dict:
    """Calculates character-type ratios for a string."""
    text_length = len(text)
    if text_length == 0:
        return {
            'alpha_ratio': 0.0, 'num_ratio': 0.0, 'hyphen_ratio': 0.0,
            'underscore_ratio': 0.0, 'hex_ratio': 0.0, 'vowels_ratio': 0.0
        }

    hex_chars = "0123456789abcdef"
    vowels = "aeiou"

    alpha_count = sum(c.isalpha() for c in text)
    num_count = sum(c.isdigit() for c in text)
    hex_count = sum(c.lower() in hex_chars for c in text)

    # Calculate vowel ratio based on alphabetical characters only
    alpha_chars_lower = [char for char in text.lower() if char.isalpha()]
    if not alpha_chars_lower:
        vowel_ratio = 0.0
    else:
        vowel_count = sum(1 for char in alpha_chars_lower if char in vowels)
        vowel_ratio = vowel_count / len(alpha_chars_lower)

    return {
        'alpha_ratio': alpha_count / text_length,
        'num_ratio': num_count / text_length,
        'hyphen_ratio': text.count('-') / text_length,
        'underscore_ratio': text.count('_') / text_length,
        'hex_ratio': hex_count / text_length,
        'vowels_ratio': vowel_ratio
    }

def generate_ngrams(text: str, n: int) -> list:
    """Generates a list of n-grams from a given string."""
    return [text[i:i+n] for i in range(len(text) - n + 1)]

def calculate_domain_depth(domain: str) -> int:
    """Calculates the depth of a domain by counting its parts."""
    return len(domain.split('.')) if domain else 0

def calculate_label_lengths(domain: str) -> dict:
    """Calculates the length of each domain label from right to left."""
    if not domain:
        return {}
    labels = domain.split('.')[::-1]  # Split and reverse
    return {f'len_label_{i+1}': len(label) for i, label in enumerate(labels)}

def calculate_subdomain_length(domain: str) -> int:
    """
    Full subdomain length: every label to the LEFT of the registered domain
    (all labels except the rightmost two, SLD + TLD), joined with the dots
    between them.

    This is DEPTH-AGNOSTIC and is the correction for the depth-3 assumption
    that previously affected the generated 150K sets:
        depth-3  "srflow.github.io"            -> "srflow"            len = 6
        depth-4  "a.b.github.io"               -> "a.b"              len = 3
        depth-5  "x.a.b.github.io"             -> "x.a.b"            len = 5
    The earlier Table 8 pipeline used len_label_3 (the third label from the
    right), which equals the true subdomain ONLY at depth 3 (the 1M baseline
    and the LSTM, both exactly three labels). On the deeper degenerate LLM
    outputs it reported just one inner label, a systematic undercount. This
    definition matches extract_jawad_features_live() in
    jawad_isolation_forest_v7_modeb_with_holdout.py, so Table 8 and Table 11
    now measure the subdomain the same way.
    """
    if not domain:
        return 0
    labels = domain.split('.')
    if len(labels) <= 2:
        return 0
    return len('.'.join(labels[:-2]))

def calculate_consecutive_chars(text: str) -> dict:
    """
    Calculates the max consecutive letters, vowels, and consonants in a string.
    """
    vowels = "aeiou"
    max_letters, current_letters = 0, 0
    max_vowels, current_vowels = 0, 0
    max_consonants, current_consonants = 0, 0

    for char in text.lower():
        if char.isalpha():
            current_letters += 1
            if char in vowels:
                current_vowels += 1
                current_consonants = 0  # Reset consonant streak
            else:
                current_consonants += 1
                current_vowels = 0  # Reset vowel streak
        else:
            # Character is not a letter, so reset all streaks
            current_letters = 0
            current_vowels = 0
            current_consonants = 0

        # Update the max values found so far
        max_letters = max(max_letters, current_letters)
        max_vowels = max(max_vowels, current_vowels)
        max_consonants = max(max_consonants, current_consonants)

    return {
        'max_consecutive_letters': max_letters,
        'max_consecutive_vowels': max_vowels,
        'max_consecutive_consonants': max_consonants
    }

def analyze_subdomain_labels(domain_string: str) -> dict:
    """
    Analyzes each subdomain label for digit patterns, returning both
    per-label booleans and aggregated counts in a single pass.
    """
    if not domain_string:
        return {}

    labels = domain_string.split('.')
    subdomain_labels = labels[:-2]  # Isolate labels to the left of the SLD

    if not subdomain_labels:
        return {}

    results = {}
    total_starting_digit = 0
    total_ending_digit = 0

    # Single loop to perform all calculations
    for i, label in enumerate(subdomain_labels, 1):
        if not label:  # Skip empty labels
            continue

        # Calculate booleans for the current label
        starts_with_digit = label[0].isdigit()
        ends_with_digit = label[-1].isdigit()

        # Update aggregate counters
        if starts_with_digit:
            total_starting_digit += 1
        if ends_with_digit:
            total_ending_digit += 1

        # Add per-label results to the dictionary
        results[f'sub{i}_starts_with_digit'] = starts_with_digit
        results[f'sub{i}_ends_with_digit'] = ends_with_digit

    # Add final aggregate counts after the loop
    results['total_labels_starting_in_digit'] = total_starting_digit
    results['total_labels_ending_in_digit'] = total_ending_digit

    return results

# --- Main Analysis Functions ---

def generate_feature_reports(subdomains: list, n: int):
    """Generates the main feature report and the common N-gram report."""
    print("Generating core feature reports...")
    all_features = []

    # Generates N-grams based on the provided 'n'
    all_ngrams = [generate_ngrams(s, n) for s in subdomains]

    for i, s in enumerate(subdomains):
        features = {
            'subdomain': s,
            'total_length': len(s),
            'entropy': calculate_shannon_entropy(s),
            'domain_depth': calculate_domain_depth(s),
            'subdomain_length': calculate_subdomain_length(s),
            **calculate_character_ratios(s),
            **calculate_label_lengths(s),
            **calculate_consecutive_chars(s),
            **analyze_subdomain_labels(s)
        }
        all_features.append(features)

    # --- Create and save the main features DataFrame ---
    df_features = pd.DataFrame(all_features)
    features_csv = OUT_PREFIX + FEATURES_CSV
    df_features.to_csv(features_csv, index=False, float_format='%.4f')
    print(f"\nSuccessfully saved full feature analysis to '{features_csv}'")
    print("\n--- Summary Statistics (Features) ---")
    print(df_features.describe())
    # index=True keeps the statistic names (count, mean, std, ...) as row labels
    df_features.describe().to_csv(OUT_PREFIX + "Summary_Stat_FEATURES.csv", index=True, float_format='%.4f')
    # --- Analyze common N-grams using the pre-generated list ---
    list_of_ngram_sets = [set(ng) for ng in all_ngrams]
    if not list_of_ngram_sets:
        return

    common_ngrams = sorted(list(set.intersection(*list_of_ngram_sets)))
    # CHANGED: Updated print statement to be dynamic
    print(f"\n--- Common {n}-Grams Found in All Subdomains ---")
    if common_ngrams:
        print(common_ngrams)
        common_ngram_ratios = []
        for i, s in enumerate(subdomains):
            ratio_data = {'subdomain': s}
            total_ngrams_in_s = len(all_ngrams[i])
            if total_ngrams_in_s > 0:
                for ngram in common_ngrams:
                    count = all_ngrams[i].count(ngram)
                    ratio_data[f'ratio_{ngram}'] = count / total_ngrams_in_s
            common_ngram_ratios.append(ratio_data)

        df_ratios = pd.DataFrame(common_ngram_ratios)
        # CHANGED: Filename is now generated dynamically based on 'n'
        output_ratios_csv = f'{OUT_PREFIX}common_{n}gram_ratios.csv'
        df_ratios.to_csv(output_ratios_csv, index=False, float_format='%.4f')
        print(f"Successfully saved common n-gram ratios to '{output_ratios_csv}'")
        print(f"\n--- Summary Statistics (Common {n}-grams) ---")
        print(df_ratios.describe())
    else:
        print(f"No {n}-grams were found to be common across all subdomains.")


def analyze_most_common_ngrams(subdomains: list, n: int, x: int):
    """Finds and analyzes the X most common N-grams across all subdomains."""
    print(f"\n--- Analyzing The {x} Most Common {n}-grams ---")
    all_ngrams_flat = [ngram for s in subdomains for ngram in generate_ngrams(s, n)]

    if not all_ngrams_flat:
        print(f"Could not generate any {n}-grams. Skipping.")
        return

    ngram_counts = Counter(all_ngrams_flat)
    most_common_tuples = ngram_counts.most_common(x)

    if not most_common_tuples:
        print("No common n-grams found. Skipping.")
        return

    top_x_ngrams = [item[0] for item in most_common_tuples]
    print(f"The {len(top_x_ngrams)} most common {n}-grams are: {top_x_ngrams}")

    ratio_results = []
    for s in subdomains:
        ratio_data = {'subdomain': s}
        subdomain_ngrams = generate_ngrams(s, n)
        total_ngrams_in_s = len(subdomain_ngrams)
        if total_ngrams_in_s > 0:
            for ngram in top_x_ngrams:
                count = subdomain_ngrams.count(ngram)
                ratio_data[f'ratio_{ngram}'] = count / total_ngrams_in_s
        ratio_results.append(ratio_data)

    df_ratios = pd.DataFrame(ratio_results)
    output_filename = f'{OUT_PREFIX}top_{x}_most_common_{n}-grams_ratios.csv'
    df_ratios.to_csv(output_filename, index=False, float_format='%.4f')
    print(f"Successfully saved most common n-gram ratios to '{output_filename}'")


def clean_generated_output(value: str, mode: str):
    """
    Collapses one generated output to a single FQDN sample.
    Returns (cleaned_value, had_newline). cleaned_value is None when mode='drop'
    and the value contained a newline artifact.
      first: keep the first non-empty line (the FQDN emitted before the model looped)
      join:  delete the newline characters and keep the rest as one string
      drop:  exclude the sample

    NOTE: the default is now 'join'. The detector pipeline
    (jawad_isolation_forest_v7_modeb_with_holdout.py) measures the full
    generated string, so to keep Table 8 consistent with Table 11 the
    morphological pipeline must also measure the whole output rather than
    truncating a newline-looped output to its first line ('first'). 'first'
    and 'drop' are retained for comparison/audit runs.
    """
    had_newline = ('\n' in value) or ('\r' in value)
    if not had_newline:
        return value.strip(), False
    if mode == 'drop':
        return None, True
    if mode == 'join':
        return value.replace('\r', '').replace('\n', '').strip(), True
    # mode == 'first'
    for line in value.splitlines():
        if line.strip():
            return line.strip(), True
    return '', True


def read_subdomains_from_file(filename: str, column: str, newline_mode: str) -> list:
    """
    Reads the generated samples so that ONE CSV ROW = ONE SAMPLE.
    Newlines embedded inside a quoted CSV field are handled by
    clean_generated_output() instead of being split into extra samples.
    """
    print(f"Reading subdomains from '{filename}'...")
    if not os.path.isfile(filename):
        print(f"Error: The file '{filename}' was not found.")
        sys.exit(1)

    if not filename.lower().endswith('.csv'):
        # Plain-text fallback (one FQDN per line). Newline artifacts cannot be
        # recovered here because the row boundaries are already lost.
        print("WARNING: plain-text input. Multi-line model outputs cannot be told apart "
              "from separate samples. Pass the results CSV instead for exact counts.")
        with open(filename, "r") as f:
            return [line.strip() for line in f if line.strip()]

    # keep_default_na=False: outputs such as 'null' or 'nan' stay strings
    df = pd.read_csv(filename, dtype=str, keep_default_na=False)
    if column not in df.columns:
        print(f"Error: column '{column}' not in CSV. Available: {list(df.columns)}")
        sys.exit(1)

    subdomains, flagged = [], []
    for idx, raw in df[column].items():
        cleaned, had_nl = clean_generated_output(raw, newline_mode)
        if had_nl:
            row = {'row_index': idx, 'cleaned': cleaned, 'raw_output': raw,
                   'n_lines': len([l for l in raw.splitlines() if l.strip()])}
            if 'termination' in df.columns:
                row['termination'] = df.at[idx, 'termination']
            flagged.append(row)
        if cleaned is not None:
            subdomains.append(cleaned)

    # --- Report the artifact loudly instead of fixing it silently ---
    n_rows = len(df)
    print(f"\n--- Newline Artifact Report ({column}) ---")
    print(f"CSV rows (samples):              {n_rows}")
    print(f"Rows containing newline(s):      {len(flagged)} ({100 * len(flagged) / n_rows:.2f}%)")
    print(f"Handling mode:                   {newline_mode}")
    print(f"Samples passed to analysis:      {len(subdomains)}")
    if flagged:
        df_flag = pd.DataFrame(flagged)
        extra = int(df_flag['n_lines'].sum() - len(df_flag))
        print(f"Extra samples a line-split would have created: {extra}")
        if 'termination' in df_flag.columns:
            print(f"Termination of affected rows:    {df_flag['termination'].value_counts().to_dict()}")
        empties = int((df_flag['cleaned'] == '').sum())
        if empties:
            print(f"WARNING: {empties} affected rows were empty after cleaning (kept as empty samples).")
        flag_csv = OUT_PREFIX + 'newline_artifacts.csv'
        df_flag.to_csv(flag_csv, index=False)
        print(f"Affected rows saved to '{flag_csv}' for audit.")

    expected = n_rows if newline_mode != 'drop' else n_rows - len(flagged)
    assert len(subdomains) == expected, "Sample count does not match CSV row count."
    return subdomains


def main():
    """Main execution function to run the analysis."""
    global OUT_PREFIX
    parser = argparse.ArgumentParser(
        description="Feature and n-gram statistics for generated FQDNs (one CSV row = one sample).")
    parser.add_argument('filename', help="Results CSV (e.g. LLM_150K_generated_FQDN_results.csv) or a .txt with one FQDN per line")
    parser.add_argument('--column', default='model_output', help="CSV column holding the generated FQDN (default: model_output)")
    parser.add_argument('--newline', choices=['first', 'join', 'drop'], default='join',
                        help="How to handle outputs containing newlines (default: join -- "
                             "keeps the whole output, matching the detector pipeline; "
                             "'first' keeps only the first line, 'drop' excludes the sample)")
    parser.add_argument('--prefix', default='', help="Prefix for all output files, e.g. LLM_ or LSTM_")
    args = parser.parse_args()
    OUT_PREFIX = args.prefix

    input_filename = args.filename
    subdomains = read_subdomains_from_file(input_filename, args.column, args.newline)

    if not subdomains:
        print("The input file is empty.")
        return

    # --- CHANGED: Get user input for N and X at the beginning ---
    try:
        print("\n--- Analysis Configuration ---")
        n_value = int(input("Enter the size of the N-grams to analyze (e.g., 2, 3): "))
        x_value = int(input(f"Enter the number of 'most common' {n_value}-grams to find (X): "))
        if n_value <= 0 or x_value <= 0:
            raise ValueError("N and X must be positive integers.")
    except ValueError as e:
        print(f"Invalid input: {e}. Exiting.")
        sys.exit(1)

    # --- Perform the core feature and common N-gram analysis ---
    generate_feature_reports(subdomains, n_value) # CHANGED: Pass n_value here

    # --- Perform the "most common n-grams" analysis ---
    analyze_most_common_ngrams(subdomains, n_value, x_value)

if __name__ == "__main__":
    main()
