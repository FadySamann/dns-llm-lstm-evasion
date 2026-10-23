"""
Multi-Detector DNS Exfiltration Evaluation
===========================================
Three unsupervised anomaly detectors, all trained on the SAME benign
pool and SAME eight Jawad et al. (2019) features, scored against the
SAME LLM/LSTM generated test sets:

  1. Isolation Forest  (Jawad et al., 2019 -- IFIP/IEEE IM)
  2. One-Class SVM     (RBF kernel, classic margin-based anomaly baseline)
  3. Autoencoder       (reconstruction-error based; captures anomalous
                         FEATURE COMBINATIONS even when each individual
                         feature falls within the benign marginal range)

Rationale for three detectors instead of one (see project discussion):
  If all three agree on a similar evasion rate, that is strong evidence
  evasion is a property of the FEATURE SPACE the generators learned to
  mimic, not an artifact of one algorithm's decision geometry. If they
  diverge, that is itself a finding -- e.g. the autoencoder catching
  cases the others miss would indicate the generators reproduce marginal
  feature distributions well (consistent with the Section 4.4 JSD/KS
  results) but not joint/correlational structure between features.

Faithful reproduction notes (Jawad et al., Section III.B, IFIP/IEEE IM 2019):
  - Training data: real observed FQDNs (not bare root domain names),
    filtered to root domains present in a top-popularity list to
    guarantee benignity. Jawad et al. used 4 days of live enterprise
    DNS traffic filtered by Majestic Million top 10,000. Here, the
    1,000,000 VirusTotal-collected FQDNs (Table 1) are used, all of
    which were collected from 28 root domains independently confirmed
    present in the Cloudflare Radar Top 1M (Table 2) -- the same
    filter-for-benignity logic, applied to an independently collected
    dataset.
  - Eight stateless per-FQDN features (Section III.B):
      1. FQDN length            -> total_length
      2. Subdomain length       -> len_label_3 (this dataset is
                                    uniformly domain_depth=3, so the
                                    single subdomain label IS the full
                                    subdomain)
      3. Query name entropy     -> entropy
      4. Numeric character count-> num_ratio * total_length_nodot
      5. Uppercase character ct.-> 0 (source data already lowercased)
      6. Number of DNS labels   -> domain_depth
      7. Maximum label length   -> max(len_label_1, len_label_2, len_label_3)
      8. Average label length   -> total_length_nodot / domain_depth
         (dot separators excluded here only, since dots are not part
          of any individual label's content)

NOTE ON ONE-CLASS SVM SCALING:
  RBF-kernel OneClassSVM has roughly quadratic time complexity in the
  number of training rows. Empirically (see project discussion), fitting
  on the full ~1,000,000-row benign pool would take many hours. We
  therefore train the SVM on a fixed random subsample of the benign pool
  (SVM_TRAIN_SAMPLE_SIZE rows, default 50,000) -- a standard and
  well-documented practice for kernel methods at this scale. The
  Isolation Forest and autoencoder are NOT subsampled; both scale
  near-linearly and are trained on the FULL benign pool, exactly as
  before. This asymmetry is intentional and is reported explicitly in
  the script output and the results CSV metadata.
"""

import os
import sys
import math
import json
import time
import warnings
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Pre-computed features for the 1M VirusTotal FQDNs (Table 1) -- this IS the
# benign training pool, since all 28 root domains are confirmed present in
# the Cloudflare Top 1M (Table 2).
BENIGN_FEATURES_CSV = "FQDN_feature_analysis.csv"

# C1 control: a benign slice disjoint from BENIGN_FEATURES_CSV, drawn from
# the same QRNG shuffle and never used to fit or calibrate any detector.
# Same pre-computed-feature schema as BENIGN_FEATURES_CSV (create_full_
# baseline_v3.py output), NOT a generated-FQDN test CSV -- scored via
# map_to_jawad_features(), not extract_jawad_features_live().
HOLDOUT_BENIGN_CSV = "holdout_feature_analysis.csv"

# =============================================================================
# MODE SELECTION
# =============================================================================
# Set to "A" for Mode A (seed-ratio stratification: 0.25 / 0.5 / 1.0)
# Set to "B" for Mode B (chunk-size stratification: 4 / 8 / 16 / 32 bytes)
# This flag controls which column is used for per-group breakdowns in
# run_detection(), run_significance_tests(), and run_model_comparison_test().
# All other pipeline steps (training, scoring, crosstab) are identical.

EVAL_MODE = "B"

# Stratification variable per mode
STRAT_COL_A   = "seed_ratio"
STRAT_VALS_A  = ["0.25", "0.5", "1.0"]
STRAT_COL_B   = "chunk_size_bytes"
STRAT_VALS_B  = [4, 8, 16, 32]
STRAT_LABEL_B = {4:  "4B /7ch  (~0.25 ratio)",
                 8:  "8B /13ch (~0.50 ratio)",
                 16: "16B/26ch (~0.93 ratio)",
                 32: "32B/52ch (neg. control)"}

# Active stratification (set automatically from EVAL_MODE)
STRAT_COL  = STRAT_COL_B  if EVAL_MODE == "B" else STRAT_COL_A
STRAT_VALS = STRAT_VALS_B if EVAL_MODE == "B" else STRAT_VALS_A

# Generated FQDNs to test -- LLM and LSTM validation CSVs
LLM_TEST_CSV  = "modeb_llm_fqdns_v2_filtered.csv"  if EVAL_MODE == "B"                 else "LLM_150K_generated_FQDN_results.csv"
LSTM_TEST_CSV = "modeb_lstm_fqdns_v2_filtered.csv"   if EVAL_MODE == "B"                 else "LSTM_150K_generated_FQDN_results.csv"

# Optional: Zeek dns.log to confirm capture (set to None to skip)
ZEEK_DNS_LOG = "dns.log"

OUTPUT_DIR = "./modeB_detect_result"

# Isolation Forest hyperparameters -- matched to Jawad et al.'s reported
# ~2% false alarm rate on benign data
IF_CONTAMINATION = 0.02
IF_N_ESTIMATORS   = 100
IF_RANDOM_STATE   = 42

# One-Class SVM hyperparameters
# nu = upper bound on the fraction of training errors / lower bound on the
# fraction of support vectors -- set equal to IF_CONTAMINATION so all three
# detectors target the SAME nominal benign false-alarm rate, making their
# evasion rates directly comparable.
SVM_NU                 = 0.02
SVM_KERNEL              = "rbf"
SVM_GAMMA               = "scale"
SVM_TRAIN_SAMPLE_SIZE   = 50_000   # see scaling note in module docstring
SVM_RANDOM_STATE        = 42

# Convergence check: trains throwaway SVMs at several subsample sizes
# BEFORE the main run, to empirically verify SVM_TRAIN_SAMPLE_SIZE is
# large enough that the evasion rate has stabilized -- i.e. that going
# all the way to the full ~1M-row benign pool (a multi-hour-to-multi-day
# fit on a single CPU, since libsvm's solver is single-threaded per fit
# regardless of core count) would not meaningfully change the result.
# Runs in well under a minute total; set to False to skip it entirely.
SVM_RUN_CONVERGENCE_CHECK = True
SVM_CONVERGENCE_SIZES     = [10_000, 20_000, 50_000, 80_000]
# A small held-out slice of generated FQDNs is needed to measure
# "evasion rate" at each subsample size before the real test sets are
# even loaded -- we draw this from the benign pool itself (held OUT of
# every convergence-check training run) purely as a cheap, immediately-
# available proxy signal for whether predictions are still shifting as
# n grows. This is NOT a substitute for the real LLM/LSTM evaluation;
# it only answers "has the SVM's decision boundary stopped moving?".
SVM_CONVERGENCE_PROXY_SIZE = 20_000

# Autoencoder hyperparameters (reconstruction-error detector)
# Small bottleneck architecture appropriate for an 8-dimensional input.
AE_HIDDEN_LAYERS    = (6, 3, 6)   # 8 -> 6 -> 3 (bottleneck) -> 6 -> 8
AE_MAX_ITER          = 300
AE_RANDOM_STATE       = 42
# Reconstruction-error threshold is calibrated post-hoc on the benign
# training pool to hit the SAME nominal false-alarm rate as the other
# two detectors (the percentile equal to 100 - contamination*100).
AE_CONTAMINATION    = 0.02

# =============================================================================
# FPR OPERATING-POINT SWEEP
# =============================================================================
# Instead of hard-committing to a single 2% benign false-alarm rate (a value
# borrowed only as a nominal target, NOT reproduced from Ahmed et al.'s trained
# model -- their achieved benign FPR was ~1.6% in-distribution and up to ~13.7%
# per-domain, and their decision threshold of 0.54 is specific to a non-public
# dataset), we characterise every result across three operating points.
#
# Each detector is FIT ONCE and its continuous anomaly score is thresholded at
# the (FPR*100)th percentile of the BENIGN pool's own score distribution, so
# each target FPR is achieved EXACTLY by construction for all three detectors.
# This is threshold-sweeping on a fixed fitted model -- the standard, ROC-style
# way to test operating-point robustness -- not refitting, which would conflate
# "operating point moved" with "model changed".
FPR_SWEEP = [0.01, 0.02, 0.05]

# =============================================================================
# BUBNOV POSITIVE-CONTROL DATASET
# =============================================================================
# Ahmed et al.'s DET/Iodine malicious corpus (their only public artifact) is no
# longer retrievable (nozzle-data.sdn.unsw.edu.au is dead). We substitute
# Bubnov's labelled DNS-tunnelling dataset (Mendeley mzn9hvdcxg), which we DO
# already cite [7]. Running the three benign-trained detectors, unchanged,
# against this independent labelled tunnelling corpus establishes that they
# actually detect real tunnelling traffic (sensitivity / positive control),
# making the generator-evasion result meaningful by contrast rather than a
# "maybe the detectors are just weak" artifact.
#
# Bubnov schema: two splits (train / validation); label 0 = regular domain,
# label 1 = domain with tunnel. Set the two CSV paths; column names are
# auto-detected if left as None.
RUN_BUBNOV_POSITIVE_CONTROL = False
BUBNOV_TRAIN_CSV  = "bubnov_training.csv"
BUBNOV_VAL_CSV    = "bubnov_validating.csv"
BUBNOV_DOMAIN_COL = None   # e.g. "domain" / "query" / "fqdn"; None = auto-detect
BUBNOV_LABEL_COL  = None   # e.g. "label" / "class"; None = auto-detect
BUBNOV_LABEL_TUNNEL_VALUE = 1   # value in the label column that denotes a tunnel

os.makedirs(OUTPUT_DIR, exist_ok=True)

# The eight Jawad et al. feature names, in a fixed order used throughout
JAWAD_FEATURES = [
    "fqdn_length", "subdomain_length", "entropy", "numeric_count",
    "uppercase_count", "num_labels", "max_label_length", "avg_label_length"
]

# =============================================================================
# STEP 1 -- LOAD PRE-COMPUTED BENIGN-POOL FEATURES AND MAP TO JAWAD COLUMNS
# =============================================================================

def map_to_jawad_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map create_full_baseline_v3.py's output columns onto the eight
    Jawad et al. (2019) feature names. Assumes len_label_1 (TLD),
    len_label_2 (SLD), and -- where present -- len_label_3 (subdomain
    label) were produced by create_full_baseline_v3.py.

    NOTE: calculate_label_lengths() in create_full_baseline_v3.py only
    emits as many len_label_i columns as a given FQDN has labels. Rows
    whose domain_depth == 2 (bare registered domain, no subdomain)
    therefore have an EMPTY len_label_3 cell, not a "0" -- pandas with
    keep_default_na=False preserves that as a literal empty string,
    which crashes a direct .astype(float) call. Such rows are dropped
    here before feature mapping, since the Jawad model assumes the
    presence of a subdomain (subdomain_length is one of the eight
    features by definition) and a depth-2 row contributes nothing
    meaningful to that feature in any case.
    """
    df = df.copy()

    domain_depth_raw = pd.to_numeric(df["domain_depth"], errors="coerce")
    valid_mask = (domain_depth_raw == 3)
    n_dropped  = (~valid_mask).sum()
    if n_dropped:
        print(f"  NOTE: dropping {n_dropped:,} rows with domain_depth != 3 "
              f"(no usable len_label_3 / subdomain) "
              f"-- {n_dropped/len(df)*100:.3f}% of input")
    df = df[valid_mask].reset_index(drop=True)

    out = pd.DataFrame()

    total_length = df["total_length"].astype(float)
    domain_depth = df["domain_depth"].astype(float)
    num_ratio    = df["num_ratio"].astype(float)

    len_label_1 = df["len_label_1"].astype(float)
    len_label_2 = df["len_label_2"].astype(float)
    len_label_3 = df["len_label_3"].astype(float)

    # Dots are separators, not label content -- removed ONLY for the
    # average-label-length feature, where the numerator must be pure
    # label-character content. Every other feature keeps dots as-is,
    # since dots are a genuine structural part of the FQDN.
    total_length_nodot = total_length - (domain_depth - 1)

    out["fqdn_length"]      = total_length                        # Feature 1
    out["subdomain_length"] = len_label_3                         # Feature 2
    out["entropy"]          = df["entropy"].astype(float)         # Feature 3
    out["numeric_count"]    = (num_ratio * total_length).round()  # Feature 4 num_ratio was digits / length-with-dots
    out["uppercase_count"]  = 0.0                                  # Feature 5
    out["num_labels"]       = domain_depth                         # Feature 6
    out["max_label_length"] = pd.concat(
        [len_label_1, len_label_2, len_label_3], axis=1
    ).max(axis=1)                                                  # Feature 7
    out["avg_label_length"] = total_length_nodot / domain_depth    # Feature 8

    return out


print("=" * 65)
print("  Jawad et al. (2019) Isolation Forest -- DNS Tunnel Detector")
print("  Trained on pre-computed features (create_full_baseline_v3.py)")
print("=" * 65)

print(f"\n[1/5] Loading benign-pool features: {BENIGN_FEATURES_CSV}")
if not os.path.exists(BENIGN_FEATURES_CSV):
    print(f"ERROR: {BENIGN_FEATURES_CSV} not found.")
    sys.exit(1)

df_benign_raw = pd.read_csv(BENIGN_FEATURES_CSV, dtype=str, keep_default_na=False)
print(f"  Rows loaded: {len(df_benign_raw):,}")

required_cols = ["total_length", "entropy", "domain_depth", "num_ratio",
                  "len_label_1", "len_label_2", "len_label_3"]
missing = [c for c in required_cols if c not in df_benign_raw.columns]
if missing:
    print(f"ERROR: missing required columns: {missing}")
    print(f"Found columns: {df_benign_raw.columns.tolist()}")
    sys.exit(1)

df_benign_jawad = map_to_jawad_features(df_benign_raw)

print(f"\n  Jawad feature means (benign training pool, n={len(df_benign_jawad):,}):")
for col in JAWAD_FEATURES:
    print(f"    {col:<20}: {df_benign_jawad[col].mean():.4f}")

# =============================================================================
# STEP 2 -- TRAIN ALL THREE DETECTORS (BENIGN-ONLY, ONE-CLASS)
# =============================================================================

X_benign = df_benign_jawad[JAWAD_FEATURES].values

scaler  = StandardScaler()
X_train_full = scaler.fit_transform(X_benign)

detectors = {}      # name -> fitted model (or dict of objects for autoencoder)
fa_rates  = {}       # name -> false alarm rate on benign training data

# ── Detector 1: Isolation Forest ─────────────────────────────────────────
print(f"\n[2a/5] Training Isolation Forest on full benign pool "
      f"(n={len(X_train_full):,}) ...")
print(f"  contamination = {IF_CONTAMINATION}")
print(f"  n_estimators  = {IF_N_ESTIMATORS}")

iforest = IsolationForest(
    n_estimators  = IF_N_ESTIMATORS,
    contamination = IF_CONTAMINATION,
    random_state  = IF_RANDOM_STATE,
    n_jobs        = -1
)
iforest.fit(X_train_full)
detectors["IsolationForest"] = iforest

preds = iforest.predict(X_train_full)
fa_rates["IsolationForest"] = (preds == -1).mean() * 100
print(f"  Benign training false alarm rate: {fa_rates['IsolationForest']:.2f}%  "
      f"(Jawad et al. report ~2%)")
# Save benign decision_function scores for later distributional comparison
# against the generated test sets (diagnostic for the anomalous IF result
# on Mode B LLM output — see project discussion).
benign_if_scores = iforest.decision_function(X_train_full)
pd.DataFrame({"anomaly_score_IsolationForest": benign_if_scores}).to_csv(
    os.path.join(OUTPUT_DIR, "benign_if_scores.csv"), index=False
)
print(f"  Benign IF scores saved -> {os.path.join(OUTPUT_DIR, 'benign_if_scores.csv')}")
# ── Detector 2: One-Class SVM ────────────────────────────────────────────
# RBF-kernel OneClassSVM scales roughly quadratically with training-set
# size -- empirically impractical (many hours) on the full ~1M-row pool.
# Trained instead on a fixed random subsample; see module docstring.

def run_svm_convergence_check(X_full: np.ndarray, sizes: list,
                               proxy_size: int, nu: float,
                               kernel: str, gamma: str,
                               base_seed: int) -> dict:
    """
    Trains a throwaway One-Class SVM at each subsample size in `sizes`
    and scores the SAME held-out proxy set each time, reporting how the
    detected/flagged RATE on that proxy set shifts as training-set size
    grows. If the rate has stabilized (changing by well under a
    percentage point) between the two largest sizes tested, that is
    direct empirical evidence that SVM_TRAIN_SAMPLE_SIZE is already
    large enough and a full ~1M-row fit would not change conclusions.

    The proxy set is a random slice of the BENIGN pool, held out of
    every training subsample in this check (drawn from the remaining
    rows after sampling the largest training size, to avoid the proxy
    overlapping with what the largest-n model was trained on). It is
    a quick stand-in signal only -- the real evaluation against the
    LLM/LSTM test sets happens later, against the one SVM actually
    selected for the main run.
    """
    print(f"\n  ── One-Class SVM convergence check "
          f"(sizes={sizes}, proxy_size={proxy_size:,}) ──")
    print(f"  Purpose: verify SVM_TRAIN_SAMPLE_SIZE={SVM_TRAIN_SAMPLE_SIZE:,} "
          f"is large enough that a full ~1M-row fit (multi-hour, single-")
    print(f"  threaded regardless of CPU core count) would not "
          f"meaningfully change results.\n")

    rng = np.random.RandomState(base_seed + 999)
    max_size  = max(sizes)
    n_total   = len(X_full)

    # Reserve a proxy set from rows NOT used by the largest training size,
    # so even the largest-n model in this check is evaluated out-of-sample.
    perm = rng.permutation(n_total)
    train_pool_idx = perm[:max_size]
    remaining_idx  = perm[max_size:]
    if len(remaining_idx) < proxy_size:
        # Fallback for small synthetic/test runs where the pool itself is
        # not much larger than max_size -- just reuse a held-out slice.
        proxy_idx = perm[-min(proxy_size, n_total // 5):]
    else:
        proxy_idx = remaining_idx[:proxy_size]
    X_proxy = X_full[proxy_idx]

    results = []
    prev_rate = None
    for n in sorted(sizes):
        sub_idx = rng.choice(train_pool_idx, size=min(n, len(train_pool_idx)),
                              replace=False)
        X_sub = X_full[sub_idx]

        t0 = time.time()
        svm_check = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        svm_check.fit(X_sub)
        fit_time = time.time() - t0

        preds = svm_check.predict(X_proxy)
        flagged_rate = (preds == -1).mean() * 100

        delta_str = ""
        if prev_rate is not None:
            delta = flagged_rate - prev_rate
            delta_str = f"  (Δ from previous size: {delta:+.3f} pp)"
        prev_rate = flagged_rate

        print(f"    n={n:>7,}  fit_time={fit_time:>6.2f}s  "
              f"flagged_rate_on_proxy={flagged_rate:.3f}%{delta_str}")

        results.append({"n": n, "fit_time_s": fit_time,
                         "flagged_rate": flagged_rate})

    # Stability verdict based on the two largest sizes tested
    if len(results) >= 2:
        last_delta = abs(results[-1]["flagged_rate"] - results[-2]["flagged_rate"])
        stable = last_delta < 1.0   # under 1 percentage point = stable
        verdict = "STABLE" if stable else "STILL SHIFTING"
        print(f"\n  Verdict: {verdict} -- change between largest two sizes "
              f"({results[-2]['n']:,} -> {results[-1]['n']:,}) "
              f"was {last_delta:.3f} percentage points.")
        if stable:
            print(f"  --> SVM_TRAIN_SAMPLE_SIZE={SVM_TRAIN_SAMPLE_SIZE:,} should be "
                  f"sufficient; a full ~1M-row fit is unlikely to change results.")
        else:
            print(f"  --> Consider increasing SVM_TRAIN_SAMPLE_SIZE, or treat the "
                  f"main SVM result as provisional pending a larger-n check.")

    return results


if SVM_RUN_CONVERGENCE_CHECK:
    convergence_results = run_svm_convergence_check(
        X_train_full,
        sizes=SVM_CONVERGENCE_SIZES,
        proxy_size=SVM_CONVERGENCE_PROXY_SIZE,
        nu=SVM_NU, kernel=SVM_KERNEL, gamma=SVM_GAMMA,
        base_seed=SVM_RANDOM_STATE,
    )
    pd.DataFrame(convergence_results).to_csv(
        os.path.join(OUTPUT_DIR, "svm_convergence_check.csv"), index=False)
    print(f"\n  Convergence check results saved -> "
          f"{os.path.join(OUTPUT_DIR, 'svm_convergence_check.csv')}")

rng = np.random.RandomState(SVM_RANDOM_STATE)
if len(X_train_full) > SVM_TRAIN_SAMPLE_SIZE:
    svm_idx = rng.choice(len(X_train_full), size=SVM_TRAIN_SAMPLE_SIZE,
                          replace=False)
    X_train_svm = X_train_full[svm_idx]
else:
    X_train_svm = X_train_full

print(f"\n[2b/5] Training One-Class SVM on benign SUBSAMPLE "
      f"(n={len(X_train_svm):,} of {len(X_train_full):,} -- "
      f"see scaling note in module docstring) ...")
print(f"  kernel = {SVM_KERNEL}, nu = {SVM_NU}, gamma = {SVM_GAMMA}")

ocsvm = OneClassSVM(kernel=SVM_KERNEL, nu=SVM_NU, gamma=SVM_GAMMA)
ocsvm.fit(X_train_svm)
detectors["OneClassSVM"] = ocsvm

# False-alarm rate is still measured against the FULL benign pool (not
# just the training subsample) for a fair, like-for-like comparison with
# the Isolation Forest and autoencoder false-alarm rates below.
preds = ocsvm.predict(X_train_full)
fa_rates["OneClassSVM"] = (preds == -1).mean() * 100
print(f"  Benign FULL-POOL false alarm rate: {fa_rates['OneClassSVM']:.2f}%  "
      f"(target ~{SVM_NU*100:.0f}%, evaluated on all {len(X_train_full):,} "
      f"benign rows, not just the training subsample)")

# ── Detector 3: Autoencoder (reconstruction error) ───────────────────────
print(f"\n[2c/5] Training Autoencoder on full benign pool "
      f"(n={len(X_train_full):,}) ...")
print(f"  architecture = 8 -> {' -> '.join(str(h) for h in AE_HIDDEN_LAYERS)} -> 8")

autoencoder = MLPRegressor(
    hidden_layer_sizes = AE_HIDDEN_LAYERS,
    activation          = "tanh",
    solver               = "adam",
    max_iter             = AE_MAX_ITER,
    random_state          = AE_RANDOM_STATE,
    early_stopping         = True,
    n_iter_no_change        = 15,
)
# Autoencoder target == input (reconstruction task)
autoencoder.fit(X_train_full, X_train_full)

def reconstruction_error(model, X):
    """Per-row mean squared reconstruction error."""
    X_hat = model.predict(X)
    return np.mean((X - X_hat) ** 2, axis=1)

train_errors = reconstruction_error(autoencoder, X_train_full)
# Calibrate the anomaly threshold post-hoc: the error value at the
# (100 - contamination*100)th percentile of BENIGN training errors,
# so the autoencoder targets the same nominal false-alarm rate as the
# other two detectors.
ae_threshold = np.percentile(train_errors, 100 * (1 - AE_CONTAMINATION))
detectors["Autoencoder"] = {"model": autoencoder, "threshold": ae_threshold}

fa_rates["Autoencoder"] = (train_errors > ae_threshold).mean() * 100
print(f"  Reconstruction-error threshold (calibrated to {AE_CONTAMINATION*100:.0f}%"
      f" percentile of benign training error): {ae_threshold:.6f}")
print(f"  Benign training false alarm rate: {fa_rates['Autoencoder']:.2f}%  "
      f"(calibrated to target {AE_CONTAMINATION*100:.0f}%, by construction)")

print(f"\n  ── False-alarm rate summary (all three detectors) ──────")
for name, rate in fa_rates.items():
    print(f"    {name:<18}: {rate:.2f}%")

# =============================================================================
# STEP 3 -- FEATURE EXTRACTION FOR GENERATED FQDNs (LIVE, NOT PRE-COMPUTED)
# The validation CSVs only contain the raw FQDN string in model_output, so
# features must be computed here -- reusing the exact same logic as
# create_full_baseline_v3.py for full consistency with the benign pool.
# =============================================================================

def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in Counter(text).values())

def extract_jawad_features_live(fqdn: str) -> dict:
    """
    Compute the eight Jawad et al. features directly from a raw FQDN
    string. Handles any domain depth dynamically -- depth-3 FQDNs
    (the entire benign training pool) and depth>3 FQDNs (which the
    LLM or LSTM may legitimately generate) are treated consistently.

    SUBDOMAIN LENGTH -- the critical dynamic computation:
      The subdomain is everything to the LEFT of the registered domain
      (rightmost two labels: SLD + TLD). Its length is the full string
      length of that leftmost portion, INCLUDING any separating dots
      between nested subdomain labels.
        depth-3: "srflow.github.io"         -> subdomain = "srflow"       len=6
        depth-4: "a.b.github.io"            -> subdomain = "a.b"          len=3
        depth-5: "x.a.b.github.io"          -> subdomain = "x.a.b"        len=5
      Previous implementation used rev[2] (third label from right),
      which equals the full subdomain ONLY at depth-3. At depth>3 it
      reports only the single label immediately left of the SLD, losing
      all further-left content -- a systematic undercount that would make
      adversarially deep structures appear shorter than they truly are,
      reducing their anomaly score and inflating the apparent evasion rate.

    MAX LABEL LENGTH -- also made dynamic:
      Previously compared only len_label_1, len_label_2, len_label_3.
      Now iterates across ALL labels so the true longest label is found
      regardless of depth, consistent with Jawad et al.'s definition
      ('maximum label length of the query').

    AVG LABEL LENGTH -- already correct (total_length_nodot / n_labels);
      no change needed since it never depended on depth assumptions.
    """
    fqdn = fqdn.replace("<|eot_id|>", "").strip().lower()
    labels   = fqdn.split(".")
    n_labels = len(labels)

    # ── Dynamic subdomain length ─────────────────────────────────────
    # Subdomain = all labels to the left of the rightmost two (SLD+TLD).
    # Joining with "." gives the full subdomain string whose len() matches
    # what Jawad et al. define as "subdomain length" for any depth.
    if n_labels <= 2:
        subdomain_str = ""
    else:
        subdomain_str = ".".join(labels[:-2])   # all labels except SLD+TLD
    subdomain_len = len(subdomain_str)

    # ── Dynamic max label length ─────────────────────────────────────
    label_lengths = [len(l) for l in labels if l]
    max_label_len = max(label_lengths) if label_lengths else 0

    # ── Other features ───────────────────────────────────────────────
    total_length       = len(fqdn)
    total_length_nodot = total_length - (n_labels - 1)  # strip dot separators
    num_count          = sum(c.isdigit() for c in fqdn)
    avg_label_len      = total_length_nodot / n_labels if n_labels > 0 else 0.0

    return {
        "fqdn_length":      float(total_length),
        "subdomain_length": float(subdomain_len),
        "entropy":          round(shannon_entropy(fqdn), 4),
        "numeric_count":    float(num_count),
        "uppercase_count":  0.0,
        "num_labels":       float(n_labels),
        "max_label_length": float(max_label_len),
        "avg_label_length": avg_label_len,
    }


def normalize_strat_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalises the stratification column for both Mode A and Mode B.

    Mode A (EVAL_MODE="A"):
      - LLM CSV has 'seed_ratio' with values '0.25','0.5','1.0'
      - LSTM CSV has 'ratio_used' with values '0.25','0.5','1' (note: '1' not '1.0')
      Fix: rename ratio_used->seed_ratio if needed, float round-trip
      so '1' and '1.0' collapse to '1.0'.

    Mode B (EVAL_MODE="B"):
      - Both CSVs have 'chunk_size_bytes' with integer values 4/8/16/32
      - Column may be stored as string due to dtype=str on read_csv
      Fix: cast to int so STRAT_VALS integer comparisons work correctly.
    """
    df = df.copy()
    if EVAL_MODE == "A":
        if "ratio_used" in df.columns and "seed_ratio" not in df.columns:
            df = df.rename(columns={"ratio_used": "seed_ratio"})
        if "seed_ratio" in df.columns:
            df["seed_ratio"] = df["seed_ratio"].astype(float).astype(str)
    else:  # Mode B
        if "chunk_size_bytes" in df.columns:
            df["chunk_size_bytes"] = pd.to_numeric(
                df["chunk_size_bytes"], errors="coerce").astype("Int64")
    return df


def score_all_detectors(X_test_sc: np.ndarray) -> dict:
    """
    Score a scaled feature matrix against all three fitted detectors.
    Returns dict: detector_name -> (binary detected array, continuous score array)
    For all three, 'detected=1' means flagged as anomalous, and the score
    is oriented so that LOWER = more anomalous (matches IsolationForest's
    decision_function convention; SVM and AE scores are sign-flipped /
    inverted accordingly for consistency in the output columns).
    """
    results = {}

    iforest_ = detectors["IsolationForest"]
    preds  = iforest_.predict(X_test_sc)
    scores = iforest_.decision_function(X_test_sc)
    results["IsolationForest"] = ((preds == -1).astype(int), scores)

    ocsvm_ = detectors["OneClassSVM"]
    preds  = ocsvm_.predict(X_test_sc)
    scores = ocsvm_.decision_function(X_test_sc)
    results["OneClassSVM"] = ((preds == -1).astype(int), scores)

    ae_      = detectors["Autoencoder"]["model"]
    ae_thr   = detectors["Autoencoder"]["threshold"]
    errors   = reconstruction_error(ae_, X_test_sc)
    detected = (errors > ae_thr).astype(int)
    # Flip sign so lower score = more anomalous, consistent with the other two
    results["Autoencoder"] = (detected, -errors)

    return results


def continuous_scores_all_detectors(X_test_sc: np.ndarray) -> dict:
    """
    Return only the continuous anomaly scores (oriented so LOWER = more
    anomalous, consistent with score_all_detectors) for every detector.
    Used for FPR-percentile thresholding and ROC-AUC, independent of any
    single fitted operating point.
    """
    return {name: score for name, (_, score) in
            score_all_detectors(X_test_sc).items()}


def derive_fpr_thresholds(benign_scores: dict, fpr_list: list) -> dict:
    """
    For each detector, derive the score threshold that yields each target
    benign FPR EXACTLY by construction. Scores are oriented lower = more
    anomalous, so "flagged" == score < threshold, and the threshold for a
    target FPR f is the (f*100)th percentile of the benign score
    distribution -- i.e. exactly f of benign rows fall below it.

    Returns: {detector_name: {fpr: threshold_value}}
    """
    thresholds = {}
    print("\n  ── FPR operating-point thresholds (percentile-calibrated on benign pool) ──")
    header = "    {:<18}".format("Detector") + "".join(
        f"  FPR={int(f*100)}% (thr / realized)".rjust(28) for f in fpr_list)
    print(header)
    for name, scores in benign_scores.items():
        thresholds[name] = {}
        row = f"    {name:<18}"
        for f in fpr_list:
            thr = float(np.percentile(scores, f * 100.0))
            realized = float((scores < thr).mean() * 100.0)
            thresholds[name][f] = thr
            row += f"  {thr:>10.5f} / {realized:>5.2f}%".rjust(28)
        print(row)
    return thresholds


def flagged_at_fpr(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Binary flag array: 1 = anomalous (score below the FPR threshold)."""
    return (scores < threshold).astype(int)


def run_fpr_sweep_report(scores_dict: dict, model_name: str,
                         fpr_thresholds: dict, n_rows: int,
                         out_tag: str = "") -> list:
    """
    Report per-detector and consensus (any / all) detection & evasion rates
    at every FPR operating point in FPR_SWEEP. `scores_dict` maps detector
    name -> continuous score array (lower = anomalous) for the test set.

    Returns a list of dict rows (one per detector x FPR, plus consensus)
    suitable for writing to a CSV.
    """
    det_names = list(scores_dict.keys())
    print(f"\n  ── FPR sweep: detection / evasion vs operating point ({model_name}) ──")
    print(f"  {'Detector':<18} " + "".join(
        f"{'FPR='+str(int(f*100))+'%':>16}" for f in FPR_SWEEP))
    print(f"  {'-'*(18 + 16*len(FPR_SWEEP))}")

    rows = []
    # Per-detector detection rates
    per_det_flags = {f: {} for f in FPR_SWEEP}
    for name in det_names:
        line = f"  {name:<18} "
        for f in FPR_SWEEP:
            thr = fpr_thresholds[name][f]
            flags = flagged_at_fpr(scores_dict[name], thr)
            per_det_flags[f][name] = flags
            det_rate = flags.mean() * 100
            line += f"{det_rate:>15.2f}%"
            rows.append({"model": model_name, "tag": out_tag,
                         "detector": name, "fpr_target": f,
                         "detected": int(flags.sum()), "n": n_rows,
                         "detection_rate_pct": det_rate,
                         "evasion_rate_pct": 100 - det_rate})
        print(line)

    # Consensus: flagged by ANY / ALL three, at each FPR
    line_any = f"  {'ANY detector':<18} "
    line_all = f"  {'ALL three':<18} "
    line_evade = f"  {'EVADED all three':<18} "
    for f in FPR_SWEEP:
        stack = np.vstack([per_det_flags[f][name] for name in det_names])
        any_flag = (stack.sum(axis=0) >= 1).astype(int)
        all_flag = (stack.sum(axis=0) == len(det_names)).astype(int)
        any_rate = any_flag.mean() * 100
        all_rate = all_flag.mean() * 100
        evade_rate = 100 - any_rate
        line_any   += f"{any_rate:>15.2f}%"
        line_all   += f"{all_rate:>15.2f}%"
        line_evade += f"{evade_rate:>15.2f}%"
        rows.append({"model": model_name, "tag": out_tag,
                     "detector": "ANY", "fpr_target": f,
                     "detected": int(any_flag.sum()), "n": n_rows,
                     "detection_rate_pct": any_rate,
                     "evasion_rate_pct": evade_rate})
        rows.append({"model": model_name, "tag": out_tag,
                     "detector": "ALL", "fpr_target": f,
                     "detected": int(all_flag.sum()), "n": n_rows,
                     "detection_rate_pct": all_rate,
                     "evasion_rate_pct": 100 - all_rate})
    print(f"  {'-'*(18 + 16*len(FPR_SWEEP))}")
    print(line_any)
    print(line_all)
    print(line_evade)
    return rows


def run_holdout_benign_check(holdout_csv: str, fpr_thresholds: dict):
    """
    C1 control (benign-vs-benign): score a held-out benign slice -- disjoint
    from BENIGN_FEATURES_CSV, drawn from the same QRNG shuffle, and never
    used to fit or calibrate any of the three detectors -- through the
    ALREADY-FITTED scaler/detectors and ALREADY-CALIBRATED FPR thresholds.
    No refitting, no recalibration on this data: the resulting flag rate is
    the true out-of-sample benign false-alarm floor. If LLM/LSTM flag rates
    (Section 3.5.1-3.5.2) are indistinguishable from this floor, that is the
    cleanest possible "indistinguishable from benign" result; any excess
    above it is the genuine synthetic-evasion signal.

    Uses the SAME pre-computed-feature pathway as the training pool
    (map_to_jawad_features), NOT extract_jawad_features_live -- the holdout
    file has the same create_full_baseline_v3.py schema as
    BENIGN_FEATURES_CSV (total_length, entropy, domain_depth, num_ratio,
    len_label_1/2/3), not raw FQDN strings in a model_output column.
    """
    print("\n" + "=" * 65)
    print("  C1 CONTROL: HELD-OUT BENIGN FLOOR")
    print("=" * 65)
    print(f"\n[H/5] Loading held-out benign features: {holdout_csv}")
    if not os.path.exists(holdout_csv):
        print(f"  WARNING: {holdout_csv} not found -- skipping holdout control")
        return None

    df_holdout_raw = pd.read_csv(holdout_csv, dtype=str, keep_default_na=False)
    print(f"  Rows loaded: {len(df_holdout_raw):,}")

    missing = [c for c in required_cols if c not in df_holdout_raw.columns]
    if missing:
        print(f"  ERROR: missing required columns: {missing}")
        print(f"  Found columns: {df_holdout_raw.columns.tolist()}")
        return None

    df_holdout_jawad = map_to_jawad_features(df_holdout_raw)
    print(f"\n  Jawad feature means (held-out benign, n={len(df_holdout_jawad):,}):")
    for col in JAWAD_FEATURES:
        print(f"    {col:<20}: {df_holdout_jawad[col].mean():.4f}")

    print(f"\n[H2] Scoring held-out benign against the three ALREADY-FITTED "
          f"detectors (no refit, no recalibration) ...")
    X_holdout    = df_holdout_jawad[JAWAD_FEATURES].values
    X_holdout_sc = scaler.transform(X_holdout)   # reuse the FITTED scaler

    holdout_scores = continuous_scores_all_detectors(X_holdout_sc)  # reuse FITTED detectors
    rows = run_fpr_sweep_report(
        holdout_scores, "HeldOutBenign", fpr_thresholds,
        n_rows=len(df_holdout_jawad), out_tag="C1_holdout"
    )
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "fpr_sweep_holdout_benign.csv"), index=False
    )
    print(f"\n  Saved -> {os.path.join(OUTPUT_DIR, 'fpr_sweep_holdout_benign.csv')}")
    return rows


def _autodetect_bubnov_columns(df: pd.DataFrame):
    """Best-effort detection of the domain-string column and the 0/1 label."""
    domain_col = BUBNOV_DOMAIN_COL
    label_col  = BUBNOV_LABEL_COL

    if label_col is None:
        for cand in ["label", "class", "is_tunnel", "tunnel", "target", "y"]:
            for c in df.columns:
                if c.lower() == cand:
                    label_col = c
                    break
            if label_col:
                break
        if label_col is None:
            # Fallback: any column whose unique values are a subset of {0,1}
            for c in df.columns:
                vals = pd.to_numeric(df[c], errors="coerce").dropna().unique()
                if len(vals) > 0 and set(np.unique(vals)).issubset({0, 1}):
                    label_col = c
                    break

    if domain_col is None:
        for cand in ["domain", "query", "fqdn", "name", "qname", "host", "url"]:
            for c in df.columns:
                if c.lower() == cand:
                    domain_col = c
                    break
            if domain_col:
                break
        if domain_col is None:
            # Fallback: the object column with the most dot-containing values
            best, best_score = None, -1
            for c in df.columns:
                if c == label_col:
                    continue
                s = df[c].astype(str)
                score = s.str.contains(r"\.").mean()
                if score > best_score:
                    best, best_score = c, score
            domain_col = best

    return domain_col, label_col


def run_bubnov_positive_control(fpr_thresholds: dict):
    """
    Positive control: score Bubnov's labelled tunnelling corpus (Mendeley
    mzn9hvdcxg) with the three benign-trained detectors, unchanged.

    Reports, per detector:
      - ROC-AUC (threshold-free) on Bubnov benign (0) vs tunnel (1)
      - True-positive (tunnel-detection) rate at each FPR operating point
      - External benign FPR on Bubnov's own regular domains at each point
    Plus consensus (ANY / ALL) tunnel-detection at each FPR.
    """
    print("\n" + "=" * 65)
    print("  POSITIVE CONTROL: Bubnov labelled DNS-tunnelling corpus [7]")
    print("  (are the benign-trained detectors actually sensitive to")
    print("   real tunnelling traffic?)")
    print("=" * 65)

    paths = [p for p in [BUBNOV_TRAIN_CSV, BUBNOV_VAL_CSV] if p and os.path.exists(p)]
    if not paths:
        print(f"  WARNING: no Bubnov CSVs found at {BUBNOV_TRAIN_CSV!r} / "
              f"{BUBNOV_VAL_CSV!r} -- skipping positive control.")
        return None

    frames = []
    for p in paths:
        d = pd.read_csv(p, dtype=str, keep_default_na=False,
                        engine="python", on_bad_lines="skip")
        d["_split"] = os.path.basename(p)
        frames.append(d)
        print(f"  Loaded {len(d):,} rows from {p}")
    df_b = pd.concat(frames, ignore_index=True)

    domain_col, label_col = _autodetect_bubnov_columns(df_b)
    if domain_col is None or label_col is None:
        print(f"  ERROR: could not identify domain/label columns "
              f"(domain={domain_col}, label={label_col}). "
              f"Set BUBNOV_DOMAIN_COL / BUBNOV_LABEL_COL explicitly.")
        return None
    print(f"  Using domain column = {domain_col!r}, label column = {label_col!r}")

    labels = pd.to_numeric(df_b[label_col], errors="coerce")
    is_tunnel = (labels == BUBNOV_LABEL_TUNNEL_VALUE).values
    is_benign = (labels == 0).values
    n_tunnel, n_benign = int(is_tunnel.sum()), int(is_benign.sum())
    print(f"  Tunnel rows (label={BUBNOV_LABEL_TUNNEL_VALUE}): {n_tunnel:,}   "
          f"Benign rows (label=0): {n_benign:,}")
    if n_tunnel == 0:
        print("  ERROR: no tunnel-labelled rows found -- check BUBNOV_LABEL_TUNNEL_VALUE.")
        return None

    # Feature extraction (identical logic to benign pool + generated FQDNs)
    print("  Extracting the eight features from Bubnov domains ...")
    feats = [extract_jawad_features_live(s) for s in df_b[domain_col].astype(str)]
    X_b = pd.DataFrame(feats)[JAWAD_FEATURES].values
    X_b_sc = scaler.transform(X_b)

    scores = continuous_scores_all_detectors(X_b_sc)  # lower = more anomalous
    det_names = list(scores.keys())

    # ── Threshold-free ROC-AUC (benign 0 vs tunnel 1) ────────────────────
    print("\n  ── Threshold-free ROC-AUC (Bubnov benign vs tunnel) ──")
    auc_rows = []
    if n_benign > 0:
        y_true = np.concatenate([np.ones(n_tunnel), np.zeros(n_benign)])
        for name in det_names:
            s_tunnel = -scores[name][is_tunnel]   # higher = more anomalous
            s_benign = -scores[name][is_benign]
            y_score = np.concatenate([s_tunnel, s_benign])
            auc = roc_auc_score(y_true, y_score)
            print(f"    {name:<18}: AUC = {auc:.4f}")
            auc_rows.append({"detector": name, "auc": auc,
                             "n_tunnel": n_tunnel, "n_benign": n_benign})
    else:
        print("    (no benign-labelled rows in Bubnov splits -- AUC skipped)")

    # ── Sensitivity at each FPR operating point ──────────────────────────
    print("\n  ── Tunnel-detection (TPR) and external benign-FPR at each operating point ──")
    print(f"  {'Detector':<18} " + "".join(
        f"{'FPR='+str(int(f*100))+'%':>22}" for f in FPR_SWEEP))
    print(f"  {'':<18} " + "".join(f"{'(TPR / benignFPR)':>22}" for _ in FPR_SWEEP))
    print(f"  {'-'*(18 + 22*len(FPR_SWEEP))}")

    control_rows = []
    per_det_tunnel_flags = {f: {} for f in FPR_SWEEP}
    for name in det_names:
        line = f"  {name:<18} "
        for f in FPR_SWEEP:
            thr = fpr_thresholds[name][f]
            flags = flagged_at_fpr(scores[name], thr)
            per_det_tunnel_flags[f][name] = flags[is_tunnel]
            tpr = flags[is_tunnel].mean() * 100 if n_tunnel else float("nan")
            ext_fpr = flags[is_benign].mean() * 100 if n_benign else float("nan")
            line += f"{tpr:>10.2f}% /{ext_fpr:>6.2f}%".rjust(22)
            control_rows.append({"detector": name, "fpr_target": f,
                                 "tunnel_tpr_pct": tpr,
                                 "bubnov_benign_fpr_pct": ext_fpr,
                                 "n_tunnel": n_tunnel, "n_benign": n_benign})
        print(line)

    # Consensus tunnel-detection at each FPR
    line_any = f"  {'ANY detector':<18} "
    line_all = f"  {'ALL three':<18} "
    for f in FPR_SWEEP:
        stack = np.vstack([per_det_tunnel_flags[f][name] for name in det_names])
        any_tpr = (stack.sum(axis=0) >= 1).mean() * 100
        all_tpr = (stack.sum(axis=0) == len(det_names)).mean() * 100
        line_any += f"{any_tpr:>13.2f}%".rjust(22)
        line_all += f"{all_tpr:>13.2f}%".rjust(22)
        control_rows.append({"detector": "ANY", "fpr_target": f,
                             "tunnel_tpr_pct": any_tpr,
                             "bubnov_benign_fpr_pct": np.nan,
                             "n_tunnel": n_tunnel, "n_benign": n_benign})
        control_rows.append({"detector": "ALL", "fpr_target": f,
                             "tunnel_tpr_pct": all_tpr,
                             "bubnov_benign_fpr_pct": np.nan,
                             "n_tunnel": n_tunnel, "n_benign": n_benign})
    print(f"  {'-'*(18 + 22*len(FPR_SWEEP))}")
    print(line_any)
    print(line_all)

    pd.DataFrame(auc_rows).to_csv(
        os.path.join(OUTPUT_DIR, "bubnov_auc.csv"), index=False)
    pd.DataFrame(control_rows).to_csv(
        os.path.join(OUTPUT_DIR, "bubnov_positive_control_fpr_sweep.csv"), index=False)
    print(f"\n  Saved -> {os.path.join(OUTPUT_DIR, 'bubnov_auc.csv')}")
    print(f"  Saved -> {os.path.join(OUTPUT_DIR, 'bubnov_positive_control_fpr_sweep.csv')}")
    return control_rows


def run_detection(test_csv: str, model_name: str):
    print(f"\n[3/5] Loading test data ({model_name}): {test_csv}")
    if not os.path.exists(test_csv):
        print(f"  WARNING: {test_csv} not found -- skipping {model_name}")
        return None

    df_test = pd.read_csv(
        test_csv, dtype=str, keep_default_na=False,
        engine="python", on_bad_lines="skip"
    )
    print(f"  Rows loaded: {len(df_test):,}")
    
    if EVAL_MODE == "B":
        # Restrict to pipeline-valid FQDNs only -- model_output for discarded rows
        # retains the last failed attempt (empty string for LSTM, newline-contaminated
        # text for LLM) and must not enter the detection feature space.
        n_before = len(df_test)
        df_test = df_test[df_test["valid"] == "1"].reset_index(drop=True)
        print(f"  Rows after valid==1 filter: {len(df_test):,} "
              f"(dropped {n_before - len(df_test):,} pipeline-discarded rows)")

    # Normalize seed_ratio / ratio_used column naming and string formatting
    # so the per-ratio breakdown below works identically for both the LLM
    # and LSTM CSVs, regardless of which native column name/format each uses.
    df_test = normalize_strat_column(df_test)

    print(f"  Extracting Jawad features from model_output ...")
    feats = [extract_jawad_features_live(f) for f in df_test["model_output"]]
    df_feat = pd.DataFrame(feats)

    print(f"  Jawad feature means ({model_name} generated FQDNs):")
    for col in JAWAD_FEATURES:
        print(f"    {col:<20}: {df_feat[col].mean():.4f}")

    print(f"\n[4/5] Classifying {model_name} FQDNs against all three detectors ...")
    X_test    = df_feat[JAWAD_FEATURES].values
    X_test_sc = scaler.transform(X_test)

    all_results = score_all_detectors(X_test_sc)

    for det_name, (detected, scores) in all_results.items():
        df_test[f"detected_{det_name}"]      = detected
        df_test[f"anomaly_score_{det_name}"] = scores.round(6)

    # "Consensus" columns: how many of the three detectors flagged each
    # FQDN, and whether ALL THREE / ANY ONE flagged it -- useful for
    # reporting agreement/disagreement between detectors directly.
    detected_cols = [f"detected_{n}" for n in all_results.keys()]
    df_test["detected_count"] = df_test[detected_cols].sum(axis=1)
    df_test["detected_any"]   = (df_test["detected_count"] >= 1).astype(int)
    df_test["detected_all"]   = (df_test["detected_count"] == len(all_results)).astype(int)

    df_test["model"] = model_name
    for col in JAWAD_FEATURES:
        df_test[f"feat_{col}"] = df_feat[col].values

    # Optional Zeek capture confirmation
    if ZEEK_DNS_LOG and os.path.exists(ZEEK_DNS_LOG):
        zeek_queries = set()
        with open(ZEEK_DNS_LOG, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        rec = json.loads(line)
                        q = rec.get("query", "")
                        if q:
                            zeek_queries.add(q.lower())
                    except json.JSONDecodeError:
                        pass
        df_test["zeek_captured"] = df_test["model_output"].str.lower().str.strip().apply(
            lambda x: 1 if x in zeek_queries else 0
        )
    else:
        df_test["zeek_captured"] = 0

    print(f"\n[5/5] Detection metrics -- {model_name}")
    total = len(df_test)

    print(f"\n  ── Per-detector results ─────────────────────────────")
    print(f"  {'Detector':<18} {'Detected':>10} {'Evaded':>10} {'Det. rate':>10}")
    print(f"  {'-'*52}")
    for det_name in all_results.keys():
        d = df_test[f"detected_{det_name}"].sum()
        e = total - d
        print(f"  {det_name:<18} {d:>10,} {e:>10,} {d/total*100:>9.2f}%")

    print(f"\n  ── Consensus across all three detectors ─────────────")
    any_d = df_test["detected_any"].sum()
    all_d = df_test["detected_all"].sum()
    print(f"  Flagged by ANY detector  : {any_d:,} ({any_d/total*100:.2f}%)  "
          f"-- evaded all three: {total-any_d:,} ({(total-any_d)/total*100:.2f}%)")
    print(f"  Flagged by ALL THREE     : {all_d:,} ({all_d/total*100:.2f}%)")

    # ── FPR operating-point sweep (robustness of the evasion result) ─────
    # Reuse the continuous scores already computed above (oriented lower =
    # anomalous) and re-threshold them at the 1% / 2% / 5% benign operating
    # points derived from the benign pool, so no refitting is needed.
    sweep_scores = {name: score for name, (_, score) in all_results.items()}
    sweep_rows = run_fpr_sweep_report(sweep_scores, model_name,
                                      fpr_thresholds, n_rows=total,
                                      out_tag=f"{EVAL_MODE}")
    pd.DataFrame(sweep_rows).to_csv(
        os.path.join(OUTPUT_DIR, f"fpr_sweep_{model_name.lower()}.csv"),
        index=False)
    # Also persist per-row flags at each FPR for downstream stratified analysis
    for f in FPR_SWEEP:
        pct = int(f * 100)
        stack = []
        for det_name in all_results.keys():
            flags = flagged_at_fpr(sweep_scores[det_name], fpr_thresholds[det_name][f])
            df_test[f"detected_{det_name}_fpr{pct}"] = flags
            stack.append(flags)
        stack = np.vstack(stack)
        df_test[f"detected_any_fpr{pct}"] = (stack.sum(axis=0) >= 1).astype(int)
        df_test[f"detected_all_fpr{pct}"] = (stack.sum(axis=0) == len(all_results)).astype(int)

    if STRAT_COL in df_test.columns:
        label = ("chunk-size class" if EVAL_MODE == "B" else "seed ratio")
        print(f"\n  ── By {label} (detected_any) ─────────────────────────")
        for val in STRAT_VALS:
            sub = df_test[df_test[STRAT_COL] == val]
            if len(sub) == 0:
                continue
            n, d = len(sub), sub["detected_any"].sum()
            tag = STRAT_LABEL_B.get(val, str(val)) if EVAL_MODE == "B" else str(val)
            print(f"    {tag:<28}: {d:>6,} / {n:<6,} detected "
                  f"({d/n*100:5.1f}%)  |  evaded {n-d:,} ({(n-d)/n*100:5.1f}%)")

    out_path = os.path.join(OUTPUT_DIR, f"multidetector_{model_name.lower()}.csv")
    df_test.to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")
    return df_test


def is_structural_collapse(generated_fqdn) -> bool:
    """
    Mirrors the authoritative structural-collapse definition used in
    assess_structural_anomaliesv3.py, plotLLMResults_Final_v2.py, and
    plotLSTMResults_Final_v2.py, so that the crosstab below compares the
    Jawad Isolation Forest's 'detected' flag against the SAME structural
    failure population already reported in Section 4.2.2 -- rather than
    an inconsistent or re-derived definition.
    """
    import re
    import tldextract

    if pd.isna(generated_fqdn) or not isinstance(generated_fqdn, str) \
            or not generated_fqdn.strip():
        return True
    clean = generated_fqdn.replace("<|eot_id|>", "").strip().lower()
    if len(clean) > 253:
        return True
    if any(len(label) > 63 for label in clean.split(".")):
        return True
    if re.search(r"[^a-z0-9\-\.\_]", clean):
        return True
    if clean.startswith(".") or clean.endswith(".") or ".." in clean \
            or "." not in clean:
        return True
    ext = tldextract.extract(clean)
    if not ext.suffix:
        return True
    return False


def crosstab_against_audit(df_test: pd.DataFrame, model_name: str,
                            detected_col: str = "detected_any"):
    """
    Follow-up 1: how much of a detector's flagged population overlaps
    with the structural-collapse population already identified by the
    morphological audit (Section 4.2.2)? If the overlap is large, the
    detector and the morphological audit are largely measuring the same
    underlying failure cases rather than two independent signals -- this
    needs to be stated explicitly either way.

    detected_col: which detection column to crosstab against. Defaults
    to 'detected_any' (flagged by at least one of the three detectors).
    Pass 'detected_IsolationForest', 'detected_OneClassSVM', or
    'detected_Autoencoder' to crosstab a single detector individually.
    """
    df_test = df_test.copy()
    df_test["struct_collapse"] = df_test["model_output"].apply(is_structural_collapse)

    detected      = df_test[detected_col] == 1
    struct        = df_test["struct_collapse"]

    both       = (detected &  struct).sum()
    det_only   = (detected & ~struct).sum()
    struct_only= (~detected &  struct).sum()
    neither    = (~detected & ~struct).sum()
    total      = len(df_test)

    print(f"\n  ── Crosstab: {detected_col} vs. structural collapse ({model_name}) ──")
    print(f"  {'':<28} {'struct_collapse':>16} {'NOT struct_collapse':>20}")
    print(f"  {'detected':<28} {both:>16,} {det_only:>20,}")
    print(f"  {'NOT detected':<28} {struct_only:>16,} {neither:>20,}")
    print(f"  Total: {total:,}")

    n_detected = detected.sum()
    n_struct   = struct.sum()
    overlap_pct_of_detected = both / n_detected * 100 if n_detected > 0 else 0.0
    overlap_pct_of_struct   = both / n_struct   * 100 if n_struct   > 0 else 0.0

    print(f"\n  Of {n_detected:,} detected FQDNs, {both:,} ({overlap_pct_of_detected:.1f}%) "
          f"are also structural-collapse cases.")
    print(f"  Of {n_struct:,} structural-collapse FQDNs, {both:,} ({overlap_pct_of_struct:.1f}%) "
          f"were caught by this detector.")
    print(f"  Detector found {det_only:,} additional anomalies NOT flagged "
          f"by the structural audit ({det_only/total*100:.2f}% of all samples) --")
    print(f"  these represent genuinely novel detections (e.g. statistically unusual "
          f"but RFC-compliant FQDNs).")

    return {
        "model": model_name, "detected_col": detected_col,
        "both": both, "det_only": det_only,
        "struct_only": struct_only, "neither": neither,
        "overlap_pct_of_detected": overlap_pct_of_detected,
        "overlap_pct_of_struct": overlap_pct_of_struct,
    }


def two_proportion_ztest(count1: int, n1: int, count2: int, n2: int):
    """
    Follow-up 2: two-proportion z-test. Used both for (a) comparing
    detection rate between seed-ratio buckets within one model, and
    (b) comparing aggregate detection rate between LLM and LSTM.
    Returns (z_statistic, two_tailed_p_value).
    No scipy dependency -- implemented directly since the z-test for
    two proportions has a closed-form normal-approximation formula.
    """
    p1 = count1 / n1
    p2 = count2 / n2
    p_pool = (count1 + count2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    # Two-tailed p-value from the standard normal CDF via erf
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return z, p_value


def run_significance_tests(df_test: pd.DataFrame, model_name: str,
                            detected_col: str = "detected_any"):
    """
    Tests whether the seed-ratio detection-rate differences within a
    single model are statistically significant, using a two-proportion
    z-test on each adjacent pair of seed ratios.
    """
    label = "chunk-size" if EVAL_MODE == "B" else "seed-ratio"
    print(f"\n  ── Significance tests: {label} detection rates "
          f"({model_name}, {detected_col}) ──")
    strat_stats = {}
    for val in STRAT_VALS:
        sub = df_test[df_test[STRAT_COL] == val]
        if len(sub) == 0:
            continue
        strat_stats[val] = (int(sub[detected_col].sum()), len(sub))

    # All adjacent pairs
    vals = list(strat_stats.keys())
    pairs = [(vals[i], vals[j]) for i in range(len(vals))
             for j in range(i+1, len(vals))]
    for v1, v2 in pairs:
        c1, n1 = strat_stats[v1]
        c2, n2 = strat_stats[v2]
        z, p = two_proportion_ztest(c1, n1, c2, n2)
        sig = "significant (p<0.05)" if p < 0.05 else "NOT significant (p>=0.05)"
        tag1 = STRAT_LABEL_B.get(v1, str(v1)) if EVAL_MODE == "B" else str(v1)
        tag2 = STRAT_LABEL_B.get(v2, str(v2)) if EVAL_MODE == "B" else str(v2)
        print(f"    {tag1} ({c1/n1*100:.2f}%) vs {tag2} ({c2/n2*100:.2f}%): "
              f"z={z:+.3f}, p={p:.4f} -- {sig}")

    return strat_stats


def run_model_comparison_test(df_llm: pd.DataFrame, df_lstm: pd.DataFrame,
                               detected_col: str = "detected_any"):
    """
    Tests whether the aggregate LLM vs LSTM detection-rate difference is
    statistically significant at n=100,000 each, and repeats the test
    within each seed-ratio bucket so a single aggregate p-value doesn't
    mask ratio-specific differences (or vice versa).
    """
    print(f"\n  ── Significance test: LLM vs LSTM aggregate detection rate "
          f"({detected_col}) ──")
    c_llm,  n_llm  = int(df_llm[detected_col].sum()),  len(df_llm)
    c_lstm, n_lstm = int(df_lstm[detected_col].sum()), len(df_lstm)
    z, p = two_proportion_ztest(c_llm, n_llm, c_lstm, n_lstm)
    sig = "significant (p<0.05)" if p < 0.05 else "NOT significant (p>=0.05)"
    print(f"    LLM ({c_llm/n_llm*100:.2f}%) vs LSTM ({c_lstm/n_lstm*100:.2f}%): "
          f"z={z:+.3f}, p={p:.4f} -- {sig}")

    label = "chunk-size" if EVAL_MODE == "B" else "seed ratio"
    print(f"\n  ── Same comparison, broken out by {label} ──")
    for val in STRAT_VALS:
        sub_llm  = df_llm[df_llm[STRAT_COL]  == val]
        sub_lstm = df_lstm[df_lstm[STRAT_COL] == val]
        if len(sub_llm) == 0 or len(sub_lstm) == 0:
            continue
        cl, nl = int(sub_llm[detected_col].sum()),  len(sub_llm)
        cs, ns = int(sub_lstm[detected_col].sum()), len(sub_lstm)
        z, p = two_proportion_ztest(cl, nl, cs, ns)
        sig = "significant" if p < 0.05 else "not significant"
        tag = STRAT_LABEL_B.get(val, str(val)) if EVAL_MODE == "B" else str(val)
        print(f"    {tag}: LLM {cl/nl*100:.2f}% vs LSTM {cs/ns*100:.2f}% "
              f"-- z={z:+.3f}, p={p:.4f} ({sig})")


def run_cross_detector_agreement(df_test: pd.DataFrame, model_name: str):
    """
    Follow-up 3 (new): how much do the three detectors agree with EACH
    OTHER, independent of the structural audit? High pairwise agreement
    means evasion is a property of the shared feature space, not an
    artifact of one algorithm's geometry. Low agreement -- especially
    cases the Autoencoder catches that the other two miss -- would
    indicate the generators reproduce per-feature marginal distributions
    well but not joint/correlational feature structure.
    """
    print(f"\n  ── Cross-detector agreement ({model_name}) ──")
    det_names = ["IsolationForest", "OneClassSVM", "Autoencoder"]
    total = len(df_test)

    print(f"  Pairwise overlap (Jaccard index = |A∩B| / |A∪B|):")
    for i in range(len(det_names)):
        for j in range(i + 1, len(det_names)):
            a = df_test[f"detected_{det_names[i]}"] == 1
            b = df_test[f"detected_{det_names[j]}"] == 1
            intersection = (a & b).sum()
            union        = (a | b).sum()
            jaccard      = intersection / union if union > 0 else 0.0
            print(f"    {det_names[i]:<16} ∩ {det_names[j]:<16}: "
                  f"{intersection:,} / {union:,} union  "
                  f"(Jaccard = {jaccard:.3f})")

    print(f"\n  Detection count distribution (how many of 3 detectors flagged each FQDN):")
    for k in [0, 1, 2, 3]:
        n = (df_test["detected_count"] == k).sum()
        print(f"    flagged by exactly {k}: {n:,} ({n/total*100:.2f}%)")

    # FQDNs the Autoencoder catches that NEITHER of the other two catch --
    # the specific population motivating the "joint feature structure"
    # hypothesis in the module docstring.
    ae_only = (
        (df_test["detected_Autoencoder"] == 1) &
        (df_test["detected_IsolationForest"] == 0) &
        (df_test["detected_OneClassSVM"] == 0)
    ).sum()
    print(f"\n  Autoencoder-EXCLUSIVE detections (missed by both IF and SVM): "
          f"{ae_only:,} ({ae_only/total*100:.2f}%)")
    print(f"  (candidates for qualitative specimen extraction -- these are cases")
    print(f"   where individual feature VALUES look normal but the feature")
    print(f"   COMBINATION is unusual)")



# =============================================================================
# DERIVE FPR OPERATING-POINT THRESHOLDS (percentile-calibrated on benign pool)
# =============================================================================
# Score the full benign pool once with all three detectors, then derive the
# score cut-points that yield exactly 1% / 2% / 5% benign FPR. These global
# thresholds are reused for the generated FQDNs and the Bubnov positive
# control, so every result is reported at the SAME set of operating points.
print("\n" + "=" * 65)
print("  FPR OPERATING-POINT CALIBRATION (on benign pool)")
print("=" * 65)
benign_scores = continuous_scores_all_detectors(X_train_full)
fpr_thresholds = derive_fpr_thresholds(benign_scores, FPR_SWEEP)

# =============================================================================
# C1 CONTROL -- held-out benign floor (unchanged scaler/detectors/thresholds)
# =============================================================================
holdout_rows = run_holdout_benign_check(HOLDOUT_BENIGN_CSV, fpr_thresholds)

# =============================================================================
# RUN FOR BOTH MODELS
# =============================================================================
df_llm  = run_detection(LLM_TEST_CSV,  "LLM")
df_lstm = run_detection(LSTM_TEST_CSV, "LSTM")

DETECTOR_COLS = ["detected_IsolationForest", "detected_OneClassSVM",
                  "detected_Autoencoder", "detected_any", "detected_all"]

# =============================================================================
# FOLLOW-UP 1 -- crosstab vs structural-collapse audit, per detector + consensus
# =============================================================================
print("\n" + "=" * 65)
print("  FOLLOW-UP 1: Overlap with morphological audit (Section 4.2.2)")
print("=" * 65)
for col in DETECTOR_COLS:
    if df_llm is not None:
        crosstab_against_audit(df_llm, "LLM", detected_col=col)
    if df_lstm is not None:
        crosstab_against_audit(df_lstm, "LSTM", detected_col=col)

# =============================================================================
# FOLLOW-UP 2 -- significance tests, per detector + consensus
# =============================================================================
print("\n" + "=" * 65)
print("  FOLLOW-UP 2: Statistical significance tests")
print("=" * 65)
for col in DETECTOR_COLS:
    if df_llm is not None:
        run_significance_tests(df_llm, "LLM", detected_col=col)
    if df_lstm is not None:
        run_significance_tests(df_lstm, "LSTM", detected_col=col)
    if df_llm is not None and df_lstm is not None:
        run_model_comparison_test(df_llm, df_lstm, detected_col=col)

# =============================================================================
# FOLLOW-UP 3 (NEW) -- cross-detector agreement
# =============================================================================
print("\n" + "=" * 65)
print("  FOLLOW-UP 3: Cross-detector agreement (do the three detectors")
print("  agree with EACH OTHER, independent of the structural audit?)")
print("=" * 65)
if df_llm is not None:
    run_cross_detector_agreement(df_llm, "LLM")
if df_lstm is not None:
    run_cross_detector_agreement(df_lstm, "LSTM")

# =============================================================================
# POSITIVE CONTROL -- Bubnov labelled tunnelling corpus at every FPR point
# =============================================================================
if RUN_BUBNOV_POSITIVE_CONTROL:
    run_bubnov_positive_control(fpr_thresholds)

print("\n" + "=" * 65)
print("  Done.")
print("  References:")
print("    Ahmed et al., IFIP/IEEE IM 2019, pp. 649-653 (Isolation Forest features)")
print("    IEEE TNSM 17(1):265-279")
print("    One-Class SVM and Autoencoder: standard unsupervised anomaly")
print("    detection baselines, trained on the identical 8-feature benign pool.")
print("=" * 65)
