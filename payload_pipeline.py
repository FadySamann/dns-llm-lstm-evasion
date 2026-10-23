"""
payload_pipeline.py  --  Mode B Payload Corpus Generator
=========================================================
Generates payload chunks for DNS tunnel simulation (Mode B).

Design rationale
----------------
Each payload CHUNK becomes the SEED (partial prefix) for the
Subdomain-LLM or Subdomain-LSTM model. The model then completes
the subdomain with additional CDN-mimicking characters before the
root domain is appended, producing a FQDN that embeds the payload
chunk inside a morphologically realistic subdomain structure:

    <model-generated-prefix><chunk><model-generated-suffix>.<root-domain>

This design is the critical difference from naive DNS tunneling:
instead of stuffing a full high-entropy encoded payload into the
subdomain label directly (which is trivially detected by entropy-
based detectors), the chunk occupies only a fraction of the label
and the model wraps it with low-entropy, CDN-like character
patterns that reduce the overall FQDN entropy toward the benign
training distribution.

The only variable that affects per-query detection by the three
stateless detectors (Isolation Forest, One-Class SVM, Autoencoder)
is CHUNK SIZE, because chunk size determines:
  - How much of the subdomain is high-entropy Base32 payload
  - How much room remains for the model to add cover characters
  - The resulting per-FQDN entropy, length, and character-
    composition features that the detectors score

Total payload size is irrelevant here because all three detectors
operate per-query with no session context -- a 256-byte payload
split into 32 × 8-byte chunks produces the same per-FQDN feature
vectors as 32 independently generated FQDNs with 8-byte seeds.

Four chunk sizes
----------------
  4 bytes  ->   7 Base32 chars  (seed ratio ~0.25, high model freedom)
  8 bytes  ->  13 Base32 chars  (seed ratio ~0.50, moderate freedom)
 16 bytes  ->  26 Base32 chars  (seed ratio ~0.90, low model freedom)
 32 bytes  ->  52 Base32 chars  (seed ratio >1.0,  negative control --
                                  chunk exceeds typical label length,
                                  model cannot meaningfully wrap it;
                                  expected high detection rate)

The 32-byte class is a deliberate negative control that demonstrates
the contrast between "chunk small enough to wrap" versus "chunk that
overwhelms the label budget" -- directly supporting the argument that
the chunked-seed design is necessary for realistic evasion.

Encoding
--------
Base32 (RFC 4648), padding stripped, lowercased:
  - Alphabet [a-z2-7] is entirely valid under RFC 1035 label rules
  - DNS-case-insensitive safe (unlike Base64 which uses +/= and
    upper/lower ambiguity lost in DNS case-folding)
  - Standard encoding used by real tunneling tools (iodine, dns2tcp)
    making Mode B specimens directly comparable to the threat model
    assumed by the detectors in the literature

Source: os.urandom(n) -- cryptographically random OS entropy pool.
BIP39 mnemonic phrases were considered and rejected because their
text byte-lengths vary with word character counts, introducing an
uncontrolled variable into a design where byte-length is the sole
independent variable between chunk-size classes.

Output
------
  payload_corpus.csv
    payload_id          : unique integer (0-indexed, global across classes)
    chunk_size_bytes    : raw byte length (4, 8, 16, or 32)
    chunk_base32        : lowercase Base32 string, padding stripped
    chunk_base32_len    : character length of chunk_base32 (constant
                          within each class: 7, 13, 26, 52 respectively)
    chunk_hex           : lowercase hex string (for debugging / audit)
    approx_seed_ratio   : approximate seed ratio relative to benign
                          training distribution mean subdomain length
                          (27.95 chars from Table 1 of the paper)
    model_freedom_chars : estimated chars the model can add after the
                          chunk to reach mean subdomain length
                          (= 27.95 - chunk_base32_len, floored at 0)

  payload_corpus_summary.csv
    one row per chunk-size class with summary statistics
"""

import os
import base64
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

N_PER_CLASS = 5_000          # instances per chunk-size class

# Four chunk sizes in bytes.
# Base32 output lengths (padding-stripped):
#   4 bytes  ->  7 chars
#   8 bytes  -> 13 chars
#  16 bytes  -> 26 chars
#  32 bytes  -> 52 chars
CHUNK_BYTE_SIZES = [4, 8, 16, 32]

# Mean subdomain length in the benign training pool (Table 1 of the paper,
# from FQDNs_feature_analysis.csv: len_label_3 mean = 27.95 chars).
# Used to compute approx_seed_ratio and model_freedom_chars.
BENIGN_MEAN_SUBDOMAIN_LEN = 27.95

OUTPUT_CSV         = "payload_corpus.csv"
OUTPUT_SUMMARY_CSV = "payload_corpus_summary.csv"

# =============================================================================
# ENCODING
# =============================================================================

def bytes_to_base32(raw: bytes) -> str:
    """
    Encode raw bytes as lowercase Base32 with padding stripped.

    Padding ('=') is stripped because '=' is not a valid DNS label
    character (RFC 1035). Decoders infer padding from string length,
    which is standard practice in DNS tunneling implementations.
    """
    return base64.b32encode(raw).decode("ascii").rstrip("=").lower()

# =============================================================================
# PRE-FLIGHT: verify exact Base32 output lengths for all chunk sizes
# =============================================================================

B32_LENGTHS = {}
for n in CHUNK_BYTE_SIZES:
    b32 = bytes_to_base32(os.urandom(n))
    B32_LENGTHS[n] = len(b32)

print("=" * 62)
print("  Mode B -- Payload Corpus Generator")
print("=" * 62)
print(f"\n  N per chunk-size class : {N_PER_CLASS:,}")
print(f"  Chunk sizes (bytes)    : {CHUNK_BYTE_SIZES}")
print(f"  Total payload rows     : {N_PER_CLASS * len(CHUNK_BYTE_SIZES):,}")
print(f"\n  Base32 output lengths (verified):")
for n, b32len in B32_LENGTHS.items():
    approx_ratio   = round(b32len / BENIGN_MEAN_SUBDOMAIN_LEN, 2)
    freedom        = max(0, round(BENIGN_MEAN_SUBDOMAIN_LEN - b32len, 1))
    label          = "negative control" if b32len > BENIGN_MEAN_SUBDOMAIN_LEN \
                     else f"~{approx_ratio} seed ratio"
    print(f"    {n:>3} bytes -> {b32len:>3} Base32 chars  "
          f"({label}, model freedom ~{freedom:.0f} chars)")

# =============================================================================
# GENERATE CORPUS
# =============================================================================

print(f"\n  Generating payload chunks ...")
rows    = []
summary = []

for class_idx, n_bytes in enumerate(CHUNK_BYTE_SIZES):
    b32_len        = B32_LENGTHS[n_bytes]
    approx_ratio   = round(b32_len / BENIGN_MEAN_SUBDOMAIN_LEN, 2)
    freedom        = max(0.0, round(BENIGN_MEAN_SUBDOMAIN_LEN - b32_len, 1))
    is_neg_control = b32_len > BENIGN_MEAN_SUBDOMAIN_LEN
    id_base        = class_idx * N_PER_CLASS

    for i in range(N_PER_CLASS):
        raw = os.urandom(n_bytes)
        b32 = bytes_to_base32(raw)

        # Defensive assertion: length must be constant within class
        assert len(b32) == b32_len, \
            f"Unexpected Base32 length {len(b32)} != {b32_len} at {n_bytes} bytes"

        rows.append({
            "payload_id":           id_base + i,
            "chunk_size_bytes":     n_bytes,
            "chunk_base32":         b32,
            "chunk_base32_len":     b32_len,
            "chunk_hex":            raw.hex(),
            "approx_seed_ratio":    approx_ratio,
            "model_freedom_chars":  freedom,
        })

    summary.append({
        "chunk_size_bytes":     n_bytes,
        "chunk_base32_len":     b32_len,
        "n_instances":          N_PER_CLASS,
        "approx_seed_ratio":    approx_ratio,
        "model_freedom_chars":  freedom,
        "role": "negative control (chunk > mean label len)"
                if is_neg_control else "experimental",
    })

    role_str = " [NEGATIVE CONTROL]" if is_neg_control else ""
    print(f"    {n_bytes:>3}-byte class: "
          f"{N_PER_CLASS:,} chunks × {b32_len} chars{role_str}")

# =============================================================================
# SAVE
# =============================================================================

df = pd.DataFrame(rows)
df_summary = pd.DataFrame(summary)

df.to_csv(OUTPUT_CSV, index=False)
df_summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)

# =============================================================================
# REPORT
# =============================================================================

print(f"\n{'='*62}")
print(f"  CORPUS SUMMARY")
print(f"{'='*62}")
print(f"\n  {'Bytes':>6}  {'B32 len':>8}  {'~Seed ratio':>12}  "
      f"{'Model freedom':>14}  {'Role':>20}")
print(f"  {'-'*66}")
for row in summary:
    print(f"  {row['chunk_size_bytes']:>6}  "
          f"{row['chunk_base32_len']:>8}  "
          f"{row['approx_seed_ratio']:>12.2f}  "
          f"{row['model_freedom_chars']:>13.0f}c  "
          f"{row['role']:>20}")

print(f"\n  Total rows  : {len(df):,}")
print(f"  Saved -> {OUTPUT_CSV}")
print(f"  Saved -> {OUTPUT_SUMMARY_CSV}")

# =============================================================================
# SAMPLE OUTPUT (first row per class, for visual inspection)
# =============================================================================

print(f"\n  Sample chunk per class (first instance):")
for n_bytes in CHUNK_BYTE_SIZES:
    sample = df[df["chunk_size_bytes"] == n_bytes]["chunk_base32"].iloc[0]
    print(f"    {n_bytes:>3} bytes -> '{sample}'  (len={len(sample)})")

print(f"""
  Next steps:
    1. Feed chunk_base32 column as LSTM/LLM seed in subdomain generation
    2. The model completes the subdomain around the chunk
    3. Append the LSTM-generated root domain to form the full FQDN
    4. Run completed FQDNs through jawad_isolation_forest.py detectors

  NOTE: the 32-byte class (52-char chunks) is a negative control.
  Chunks exceed the benign mean subdomain length ({BENIGN_MEAN_SUBDOMAIN_LEN} chars),
  leaving the model no room to add cover characters. Expect high
  detection rates for this class -- the contrast with the 4-byte
  and 8-byte classes quantifies the benefit of small-chunk design.
""")

# =============================================================================
# CHUNK_BASE32_PAYLOAD helper (imported by downstream generation scripts)
# =============================================================================

def chunk_base32_payload(b32: str, chunk_size: int = 50) -> list:
    """
    Splits a long Base32 string into DNS-label-safe chunks.
    Only needed if you later decide to send a full multi-chunk payload
    in a single DNS query using multiple subdomain labels. For the
    primary Mode B design (one chunk per query, chunk is the LSTM/LLM
    seed), this function is not used -- each chunk_base32 value from
    the corpus CSV is already the right size for one generation call.
    Retained here as a utility for completeness.
    """
    return [b32[i:i + chunk_size] for i in range(0, len(b32), chunk_size)]
