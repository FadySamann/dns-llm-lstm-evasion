"""
assess_structural_anomalies_modeb.py
=====================================
Morphological audit of Mode B generation outputs.
Adapted from assess_structural_anomaliesv3.py for:

    modeb_llm_fqdns.csv
    modeb_lstm_fqdns.csv

Key schema differences from Mode A CSVs:
  - No ground_truth column. The GT sanity block is repurposed to audit
    'generated_raw' (pre-cleanup text), providing a raw-vs-clean
    comparison baseline.
      * LLM:  generated_raw contains <|eot_id|> cascade (expected,
              cosmetic -- model_output is clean). Reported as a
              known pipeline artifact, not a model failure.
      * LSTM: generated_raw contains the training-domain template
              (e.g. akamaihd.net) before root-domain substitution.
  - model_output for LSTM invalid rows is NaN (label_too_long at 32B
    caused generation to be discarded entirely). NaN is handled
    explicitly throughout.
  - Stratification uses chunk_size_bytes (primary) and
    approx_seed_ratio (secondary) instead of seed_ratio.
  - discard_reason column records generation-time RFC failures.
    A reconciliation block cross-checks these against the independent
    morphological audit to verify pipeline consistency.
"""

import pandas as pd
import re
import sys

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------
LLM_CSV  = "modeb_llm_fqdns_v2.csv"
LSTM_CSV = "modeb_lstm_fqdns_v2.csv"

TARGET_COL      = "model_output"
RAW_COL         = "generated_raw"
STRAT_PRIMARY   = "chunk_size_bytes"
STRAT_SECONDARY = "approx_seed_ratio"
DISCARD_COL     = "discard_reason"
VALID_COL       = "valid"

# ---------------------------------------------------------------------------
# Core artifact detector
# ---------------------------------------------------------------------------

def extract_artifacts(series: pd.Series) -> pd.DataFrame:
    """
    Applies the full suite of morphological artifact checks to a string Series.
    NaN values are treated as empty outputs and flagged accordingly.
    Returns a DataFrame of boolean masks, one column per artifact type.
    """
    # Coerce to str; NaN becomes the literal string 'nan' -- handle separately
    is_null = series.isna() | (series.astype(str).str.strip() == '') | (series.astype(str) == 'nan')
    s = series.fillna('').astype(str).str.lower()

    masks = pd.DataFrame(index=series.index)

    # 1. Internal newlines
    masks['has_inner_newline'] = s.str.contains(r'[\r\n]', regex=True)

    # 2. RFC 1035 violations (underscore allowed for DKIM/TXT records)
    masks['has_rfc_violation'] = s.str.contains(r'[^a-z0-9\-\.\_]', regex=True)

    # 3. Boundary errors: leading/trailing/double dots, or no dot at all
    masks['has_boundary_error'] = s.apply(
        lambda x: (
            x.startswith('.') or
            x.endswith('.')   or
            '..' in x         or
            ('.' not in x and x != '')   # bare label: no TLD
        )
    )

    # 4. Prompt / special-token leaks
    token_leak_pattern = (
        r'(?:<\|start_header_id\|>|<\|end_header_id\|>|'
        r'<\|assistant\|>|<\|system\|>|<\|user\|>|'
        r'header_id|eot_id|<\|eot_id\|>)'
    )
    masks['has_token_leak'] = s.str.contains(
        token_leak_pattern, flags=re.IGNORECASE, regex=True
    )

    # 5. Empty / null outputs (complete generation failure or discard)
    masks['is_empty_output'] = is_null

    # 6. RFC length violations
    masks['exceeds_fqdn_length'] = s.str.len() > 253
    masks['has_long_label'] = s.apply(
        lambda x: any(len(lbl) > 63 for lbl in x.split('.')) if x else False
    )

    # 7. Composite flag
    masks['any_artifact'] = masks.any(axis=1)

    return masks


# ---------------------------------------------------------------------------
# Raw-column baseline audit (replaces GT sanity check)
# ---------------------------------------------------------------------------

def audit_raw_column(df: pd.DataFrame, model_label: str) -> None:
    """
    Audits the 'generated_raw' column as a pre-cleanup baseline.

    For the LLM this is expected to surface heavy token-leak hits due to
    the <|eot_id|> cascade in generated_raw -- these are cosmetic pipeline
    artifacts, not semantic failures (model_output is clean).

    For the LSTM this surfaces the training-domain template strings
    (e.g. akamaihd.net) present before root-domain substitution.
    """
    if RAW_COL not in df.columns:
        print(f"  [!] '{RAW_COL}' column not present -- skipping raw baseline.\n")
        return

    total = len(df)
    raw_masks = extract_artifacts(df[RAW_COL])
    raw_any   = raw_masks['any_artifact'].sum()

    print(f"--- Raw-Column Baseline: '{RAW_COL}' (pre-cleanup artifacts) ---")
    print(f"  Rows with ANY artifact in raw output : {raw_any:,} "
          f"({raw_any / total * 100:.2f}%)")

    # Token-leak count is especially meaningful for the LLM
    token_hits = raw_masks['has_token_leak'].sum()
    print(f"  Token leaks in raw (expected for LLM): {token_hits:,} "
          f"({token_hits / total * 100:.2f}%)")

    if model_label == "LLM" and token_hits > 0:
        print("  [NOTE] LLM <|eot_id|> cascade in generated_raw is a known "
              "pipeline artifact.\n"
              "         model_output is confirmed clean (see Token Leaks row "
              "in Model Evaluation below).")
    if model_label == "LSTM" and token_hits == 0:
        print("  [OK] No token leaks in LSTM raw outputs (expected).")
    print()


# ---------------------------------------------------------------------------
# Discard-reason reconciliation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Seed integrity audit (payload preservation check)
# ---------------------------------------------------------------------------

def check_seed_integrity(df: pd.DataFrame, model_label: str) -> pd.DataFrame:
    """
    Audits payload preservation in Mode B outputs.
    
    For each row, checks:
      1. Does the output end with '.{root_domain}' (correct dot separator)?
      2. Does the subdomain (part before .root_domain) start with chunk_base32?
      3. Is the Base32 seed corrupted even if dot separator exists?
    
    Returns a DataFrame with columns:
      - has_dot_sep: output ends with '.{root_domain}'
      - seed_intact_prefix: subdomain starts with chunk_base32 verbatim
      - seed_integrity_category: one of 'good', 'merged_no_dot', 
                                 'root_hallucinated', 'seed_corrupted_with_dot'
      - seed_corruption_rate: 1 if corrupted, 0 if intact
    
    Schema assumptions:
      - LLM: root_domain in 'assigned_root_domain' column
      - LSTM: root_domain in 'mode_b_root_domain' column
      - Both: payload seed in 'chunk_base32' column
    """
    df_check = df.copy()
    
    # Determine root-domain column name by model
    if model_label == "LLM":
        root_col = 'assigned_root_domain'
    else:  # LSTM
        root_col = 'mode_b_root_domain'
    
    if root_col not in df_check.columns:
        print(f"  [!] '{root_col}' column not found (model label: {model_label}).")
        return pd.DataFrame(index=df.index)
    
    def categorize_row(row):
        """Classify a single row's seed-integrity status."""
        out = str(row['model_output']).lower() if pd.notna(row['model_output']) else ''
        root = str(row[root_col]).lower() if pd.notna(row[root_col]) else ''
        seed = str(row['chunk_base32']).lower() if pd.notna(row['chunk_base32']) else ''
        
        if not out or not root or not seed:
            return pd.Series({
                'has_dot_sep': False,
                'seed_intact_prefix': False,
                'seed_integrity_category': 'empty_or_missing_fields',
                'seed_corruption_rate': 1
            })
        
        suffix = "." + root
        has_dot_sep = out.endswith(suffix)
        
        if has_dot_sep:
            # Correct structure: extract subdomain
            subdomain = out[:-len(suffix)]
            seed_intact = subdomain.startswith(seed)
            if seed_intact:
                category = 'good'
                corruption = 0
            else:
                category = 'seed_corrupted_with_dot'
                corruption = 1
        elif out.endswith(root):
            # Merged: no dot separator between seed and root
            category = 'merged_no_dot'
            corruption = 1
        else:
            # Root hallucinated: output doesn't end with assigned root at all
            category = 'root_hallucinated'
            corruption = 1
        
        return pd.Series({
            'has_dot_sep': has_dot_sep,
            'seed_intact_prefix': has_dot_sep and out[:-len(suffix)].startswith(seed),
            'seed_integrity_category': category,
            'seed_corruption_rate': corruption
        })
    
    integrity = df_check.apply(categorize_row, axis=1)
    return integrity


def apply_seed_integrity_filter(df: pd.DataFrame, integrity_masks: pd.DataFrame,
                                model_label: str, output_csv: str) -> pd.DataFrame:
    """
    Updates the 'valid' column in the DataFrame based on seed integrity.
    
    - If seed_integrity_category == 'good': keep valid=1 (or set to 1)
    - Otherwise: set valid=0 (mark as invalid for downstream analysis)
    
    Then writes the filtered DataFrame to output_csv and returns it.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original Mode B CSV loaded as DataFrame
    integrity_masks : pd.DataFrame
        Output from check_seed_integrity() with seed_integrity_category column
    model_label : str
        'LLM' or 'LSTM' (for logging)
    output_csv : str
        Path to write filtered CSV
    
    Returns
    -------
    pd.DataFrame
        The filtered DataFrame (with updated valid column)
    """
    df_filtered = df.copy()
    
    # Ensure valid column exists and is numeric
    if VALID_COL not in df_filtered.columns:
        df_filtered[VALID_COL] = 1  # Default to valid if missing
    
    df_filtered[VALID_COL] = pd.to_numeric(df_filtered[VALID_COL], errors='coerce').fillna(1).astype(int)
    
    # Mark as invalid (0) any row that doesn't have 'good' seed integrity
    before_count = (df_filtered[VALID_COL] == 1).sum()
    
    df_filtered[VALID_COL] = (integrity_masks['seed_integrity_category'] == 'good').astype(int)
    
    after_count = (df_filtered[VALID_COL] == 1).sum()
    newly_invalid = before_count - after_count
    
    # Write filtered CSV
    df_filtered.to_csv(output_csv, index=False)
    
    print(f"\n--- Seed Integrity Filter Applied ---")
    print(f"  Valid rows before filtering : {before_count:,}")
    print(f"  Valid rows after filtering  : {after_count:,}")
    print(f"  Newly marked invalid        : {newly_invalid:,}")
    print(f"  Filtered CSV written to     : {output_csv}\n")
    
    return df_filtered


def update_valid_column(df: pd.DataFrame, integrity_masks: pd.DataFrame, 
                        model_label: str) -> pd.DataFrame:
    """
    Updates the 'valid' column based on seed integrity results.
    
    A row is marked valid=0 (invalid) if seed_integrity_category is NOT 'good'.
    This ensures downstream detector evaluation only runs on payload-intact FQDNs.
    
    Returns the updated dataframe.
    """
    df_updated = df.copy()
    df_updated['valid'] = pd.to_numeric(df_updated['valid'], errors='coerce').fillna(0).astype(int)
    
    # Mark as invalid if not 'good' integrity
    previously_valid = (df_updated['valid'] == 1).sum()
    integrity_issues = (integrity_masks['seed_integrity_category'] != 'good')
    newly_invalid_count = (integrity_issues & (df_updated['valid'] == 1)).sum()
    
    df_updated.loc[integrity_issues, 'valid'] = 0
    currently_valid = (df_updated['valid'] == 1).sum()
    
    print(f"  Valid column updated based on seed integrity:")
    print(f"    Previously valid (valid=1): {previously_valid:,}")
    print(f"    Newly marked invalid due to seed issues: {newly_invalid_count:,}")
    print(f"    Now valid (valid=1): {currently_valid:,}")
    print(f"    Total invalid (valid=0): {(df_updated['valid'] == 0).sum():,}\n")
    
    return df_updated


def report_seed_integrity(df: pd.DataFrame, integrity_masks: pd.DataFrame, 
                          model_label: str) -> None:
    """
    Reports seed-integrity statistics and stratifications.
    """
    print("--- Seed Integrity (Payload Preservation) ---")
    
    total = len(df)
    
    # Overall counts
    print(f"  Seed integrity categories (of {total:,} rows):")
    cat_counts = integrity_masks['seed_integrity_category'].value_counts(normalize=False)
    for cat, count in cat_counts.items():
        pct = count / total * 100
        print(f"    {cat:<30}: {count:>6,}  ({pct:>6.2f}%)")
    
    good_count = (integrity_masks['seed_integrity_category'] == 'good').sum()
    print("-" * 60)
    print(f"  {'Payload-intact (good)':<37}: {good_count:>6,}  "
          f"({good_count / total * 100:.2f}%)\n")
    
    # Stratified by chunk_size_bytes
    if 'chunk_size_bytes' in df.columns:
        print(f"=== Seed Integrity by chunk_size_bytes (%) ===")
        df_strat = pd.concat([df[['chunk_size_bytes']].copy(), integrity_masks], axis=1)
        df_strat['chunk_size_bytes'] = pd.to_numeric(df_strat['chunk_size_bytes'], errors='coerce')
        
        strat_table = (
            df_strat.groupby('chunk_size_bytes')['seed_integrity_category']
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
            .unstack(fill_value=0)
        )
        print(strat_table.to_string())
        print()
    
    # Stratified by approx_seed_ratio
    if 'approx_seed_ratio' in df.columns:
        print(f"=== Seed Integrity by approx_seed_ratio (%) ===")
        df_strat = pd.concat([df[['approx_seed_ratio']].copy(), integrity_masks], axis=1)
        df_strat['approx_seed_ratio'] = pd.to_numeric(df_strat['approx_seed_ratio'], errors='coerce')
        
        strat_table = (
            df_strat.groupby('approx_seed_ratio')['seed_integrity_category']
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
            .unstack(fill_value=0)
        )
        print(strat_table.to_string())
        print()
    
    # Specimen examples
    print("=== Seed Integrity Violation Specimens ===")
    df_specimens = pd.concat([df, integrity_masks], axis=1)
    
    for category in ['merged_no_dot', 'root_hallucinated', 'seed_corrupted_with_dot']:
        flagged = df_specimens[df_specimens['seed_integrity_category'] == category]
        print(f"\n[{category.upper()}]")
        if flagged.empty:
            print("  --> No specimens detected.")
            continue
        
        samples = flagged[['chunk_base32', 'model_output']].head(3)
        for idx, (_, row) in enumerate(samples.iterrows(), 1):
            seed = row['chunk_base32']
            out = str(row['model_output'])[:80] + "..." if len(str(row['model_output'])) > 80 else row['model_output']
            print(f"  Specimen {idx}: seed='{seed}' → output='{out}'")


def reconcile_discard_reasons(df: pd.DataFrame, model_masks: pd.DataFrame) -> None:
    """
    Cross-checks generation-time discard_reason flags against the
    independent morphological audit. Verifies pipeline consistency:
      - Every row flagged by the generation pipeline (valid==0) should
        map to at least one morphological artifact.
      - Any morphological artifact caught in model_output rows that the
        generation pipeline marked valid==1 represents a missed filter.
    """
    print("--- Discard-Reason Reconciliation ---")

    df_r = df.copy()
    df_r['valid_int'] = pd.to_numeric(df_r[VALID_COL], errors='coerce').fillna(0).astype(int)
    df_r = pd.concat([df_r, model_masks], axis=1)

    # Generation-pipeline invalid rows
    pipeline_invalid = df_r['valid_int'] == 0
    audit_caught     = df_r['any_artifact']

    both    = (pipeline_invalid & audit_caught).sum()
    pipe_only = (pipeline_invalid & ~audit_caught).sum()
    audit_only = (~pipeline_invalid & audit_caught).sum()

    total_invalid = pipeline_invalid.sum()
    total_valid   = (~pipeline_invalid).sum()

    print(f"  Generation-pipeline discards           : {total_invalid:,}")
    print(f"  ... also caught by morphological audit : {both:,}  "
          f"({'100.00' if total_invalid == 0 else f'{both/total_invalid*100:.2f}'}%)")
    print(f"  ... NOT caught by audit (gap)          : {pipe_only:,}")
    if pipe_only > 0:
        print("      [!] These discards are invisible to the morphological audit.")
        print("          Likely cause: NaN model_output on fully-discarded rows "
              "(LSTM 32B) -- the audit treats NaN as 'is_empty_output'.")

    print(f"\n  Pipeline-valid rows with audit artifact : {audit_only:,}  "
          f"({audit_only / total_valid * 100:.4f}% of valid outputs)")
    if audit_only > 0:
        print("  [!] These rows passed the generation filter but contain "
              "morphological anomalies.")
        print("      Investigate: may indicate RFC edge cases not covered by "
              "is_valid_fqdn() in the generation script.")
    else:
        print("  [OK] No morphological artifacts detected in pipeline-valid outputs.")
    print()


# ---------------------------------------------------------------------------
# Per-chunk-size stratified breakdown
# ---------------------------------------------------------------------------

def stratified_breakdown(df: pd.DataFrame, model_masks: pd.DataFrame) -> None:
    """
    Reports artifact rates stratified by chunk_size_bytes and
    approx_seed_ratio (the Mode B analogue of seed_ratio).
    """
    artifact_cols = [
        'has_inner_newline', 'has_rfc_violation', 'has_boundary_error',
        'has_token_leak', 'is_empty_output',
        'exceeds_fqdn_length', 'has_long_label', 'any_artifact'
    ]
    df_s = pd.concat([df[[STRAT_PRIMARY, STRAT_SECONDARY]].copy(), model_masks], axis=1)
    df_s[STRAT_PRIMARY]   = pd.to_numeric(df_s[STRAT_PRIMARY],   errors='coerce')
    df_s[STRAT_SECONDARY] = pd.to_numeric(df_s[STRAT_SECONDARY], errors='coerce')

    print(f"=== Artifact Rates by {STRAT_PRIMARY} (%) ===")
    summary = (
        df_s.groupby(STRAT_PRIMARY)[artifact_cols]
        .mean()
        .multiply(100)
        .round(4)
    )
    summary.columns = [c.replace('has_', '').replace('is_', '') for c in summary.columns]
    print(summary.to_string())
    print()

    print(f"=== Artifact Rates by {STRAT_SECONDARY} (approx seed ratio) (%) ===")
    summary2 = (
        df_s.groupby(STRAT_SECONDARY)[artifact_cols]
        .mean()
        .multiply(100)
        .round(4)
    )
    summary2.columns = [c.replace('has_', '').replace('is_', '') for c in summary2.columns]
    print(summary2.to_string())
    print()


# ---------------------------------------------------------------------------
# Qualitative specimen extraction
# ---------------------------------------------------------------------------

def extract_specimens(df: pd.DataFrame, model_masks: pd.DataFrame) -> None:
    """
    Prints up to 2 example model_output values per artifact category.
    Skips token-leak specimens for model_output (clean by design for LLM);
    instead samples from generated_raw to illustrate the cascade.
    """
    metrics = [
        ('Internal Newline Loops',         'has_inner_newline'),
        ('RFC Violations (Illegal Chars)',  'has_rfc_violation'),
        ('Boundary Errors (Dots/No TLD)',   'has_boundary_error'),
        ('Prompt Token Leaks',             'has_token_leak'),
        ('Empty Outputs',                  'is_empty_output'),
        ('Exceeds Max FQDN Length (>253)', 'exceeds_fqdn_length'),
        ('Exceeds Label Length (>63)',     'has_long_label'),
    ]

    df_e = pd.concat([df, model_masks], axis=1)
    print("=== Qualitative Specimens ===")

    for label, col in metrics:
        flagged = df_e[df_e[col]]
        print(f"\n[{label}]")
        if flagged.empty:
            print("  --> No specimens detected.")
            continue

        # For token leaks: model_output is clean for LLM; sample generated_raw instead
        if col == 'has_token_leak':
            if RAW_COL in df_e.columns:
                raw_flagged = df_e[
                    df_e[RAW_COL].str.contains('<|eot_id|>', regex=False, na=False)
                ][RAW_COL].head(2)
                for idx, val in enumerate(raw_flagged):
                    preview = val[:95] + "..." if len(str(val)) > 95 else val
                    print(f"  Specimen {idx+1} [from generated_raw]: {repr(preview)}")
                if raw_flagged.empty:
                    print("  --> No token leaks in generated_raw either.")
            else:
                print("  --> generated_raw column absent; cannot sample token-leak specimens.")
            continue

        samples = flagged[TARGET_COL].dropna().head(2).tolist()
        if not samples:
            # Fallback: show chunk size and discard_reason for NaN rows
            nan_rows = flagged[[STRAT_PRIMARY, DISCARD_COL]].head(2)
            print("  --> model_output is NaN (fully discarded). Metadata:")
            print(nan_rows.to_string(index=False))
            continue

        for idx, sample in enumerate(samples):
            preview = str(sample)[:95] + "..." if len(str(sample)) > 95 else str(sample)
            print(f"  Specimen {idx+1}: {repr(preview)}")


# ---------------------------------------------------------------------------
# Main audit runner
# ---------------------------------------------------------------------------

def run_modeb_audit(csv_path: str, model_label: str) -> None:
    """
    Full morphological audit pipeline for a single Mode B CSV.

    Parameters
    ----------
    csv_path    : path to modeb_llm_fqdns.csv or modeb_lstm_fqdns.csv
    model_label : 'LLM' or 'LSTM' -- controls interpretation of known artifacts
    """
    print(f"\n{'='*65}")
    print(f"  Mode B Structural Compliance Audit  |  Model: {model_label}")
    print(f"  File: {csv_path}")
    print(f"{'='*65}\n")

    try:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        # Restore genuine NaN for model_output so is_empty_output fires correctly
        df[TARGET_COL] = df[TARGET_COL].replace('', pd.NA).replace('nan', pd.NA)
    except FileNotFoundError:
        print(f"  Error: '{csv_path}' not found -- skipping {model_label}.")
        return

    total = len(df)
    print(f"  Loaded {total:,} records.\n")

    # 1. Raw-column baseline (replaces GT sanity check)
    audit_raw_column(df, model_label)

    # 2. Model-output morphological evaluation
    print(f"--- Model Evaluation: '{TARGET_COL}' ---")
    model_masks = extract_artifacts(df[TARGET_COL])

    metrics = [
        ('Internal Newline Loops',         'has_inner_newline'),
        ('RFC Violations (Illegal Chars)',  'has_rfc_violation'),
        ('Boundary Errors (Dots/No TLD)',   'has_boundary_error'),
        ('Prompt Token Leaks',             'has_token_leak'),
        ('Empty Outputs',                  'is_empty_output'),
        ('Exceeds Max FQDN Length (>253)', 'exceeds_fqdn_length'),
        ('Exceeds Label Length (>63)',      'has_long_label'),
    ]
    for label, col in metrics:
        count = model_masks[col].sum()
        print(f"  {label:<37}: {count:>6,}  ({count / total * 100:.4f}%)")

    any_count = model_masks['any_artifact'].sum()
    print("-" * 60)
    print(f"  {'Outputs with ANY artifact':<37}: {any_count:>6,}  "
          f"({any_count / total * 100:.4f}%)\n")

    # 3. Seed integrity audit (Mode B payload preservation)
    print("--- Seed Integrity Audit (Mode B Payload Preservation) ---")
    integrity_masks = check_seed_integrity(df, model_label)
    if not integrity_masks.empty:
        report_seed_integrity(df, integrity_masks, model_label)
        
        # 3b. Update valid column based on integrity check
        print("--- Updating 'valid' Column ---")
        df_updated = update_valid_column(df, integrity_masks, model_label)
        
        # Persist updated dataframe back to CSV
        print(f"--- Persisting Updated CSV ---")
        df_updated.to_csv(csv_path, index=False)
        print(f"  CSV updated and written to: {csv_path}\n")
        
        # Use updated dataframe for remaining audit steps
        df = df_updated
        model_masks = extract_artifacts(df[TARGET_COL])
    else:
        print("  [!] Seed integrity check could not be completed.\n")
    
    # 4. Discard-reason reconciliation
    reconcile_discard_reasons(df, model_masks)

    # 5. Stratified breakdown
    stratified_breakdown(df, model_masks)

    # 6. Qualitative specimens
    extract_specimens(df, model_masks)

    print(f"\n{'='*65}")
    print(f"  Audit complete: {model_label}")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_modeb_audit(LLM_CSV,  model_label="LLM")
    run_modeb_audit(LSTM_CSV, model_label="LSTM")
