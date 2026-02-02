# Prosody-Based Rhyme Engine (Developer Documentation)

## Overview

This module implements a **ranked, pronunciation-based rhyme engine** for English words.
It finds and orders rhyming candidates using **phonetic prosody**, not orthography.

Key properties:

* Deterministic (non-ML)
* Pronunciation-first (stress, vowels, syllables)
* Multi-pronunciation aware
* Optional G2P fallback for OOV (out-of-vocabulary) words
* Designed for lyric / poetry workflows where *sound* > spelling

The engine is intentionally modular: pronunciation, prosody extraction, similarity scoring, and ranking are all separable concerns.

---

## External Dependencies

### Required

* `pronouncing` (CMU Pronouncing Dictionary interface)
* `nltk` (only for `g2p_en` internals; tagger auto-downloads)

### Optional

* `g2p_en`
  Enables grapheme-to-phoneme fallback for words not in CMU.

```bash
pip install pronouncing nltk
pip install g2p_en   # optional
```

---

## Data Model

### Prosody Representation

Internally, every pronunciation is reduced to:

```python
Prosody = {
    "stress":    List[int],   # e.g. [0, 1, 0, 0]
    "vowels":    List[str],   # e.g. ["EH", "IY", "AH"]
    "syllables": int
}
```

Notes:

* Stress values are normalized: secondary stress (`2`) → primary (`1`)
* Syllables are counted via vowel nuclei
* Consonants are not used directly (for now)

### ProsodyStore

Candidate pronunciations from the database can be either:

```python
ProsodyStore = Prosody | List[Prosody]
```

The engine treats both uniformly via `as_prosody_list`.

This allows backward compatibility with older single-pronunciation dictionaries.

---

## Pronunciation Resolution Pipeline

Pronunciations for the **query word** are resolved in this order:

1. **CMU dictionary** (`pronouncing`)
2. **Cached G2P results** (`g2p_cache.json`)
3. **G2P inference** (if `--use_g2p` and `g2p_en` available)

> ⚠️ Note
> Manual pronunciation overrides are not yet wired into this file.
> The current architecture anticipates them but they must be added explicitly (as discussed earlier).

Each step returns **zero or more pronunciations**.
If multiple pronunciations exist, *all* are retained.

---

## Core Algorithm (High-Level)

For a given query word:

1. Extract all possible **base prosodies**
2. Iterate over candidate words in `stress_dictionary.json`
3. For each candidate:

   * Compare *every base prosody* × *every candidate prosody*
   * Keep the **best-scoring pairing**
4. Apply a hard **core score threshold**
5. Rank remaining candidates by **final score**
6. Return top *N*

---

## Similarity Computation

### Tail-Aligned Comparison

All comparisons are **right-aligned** (from the end of the sequence).

This models how rhymes behave acoustically.

#### Utilities

```python
tail_match_stats(a, b) -> (matches, min_len, max_len)
```

Used by both stress and vowel similarity.

---

### Stress Similarity

```python
stress_similarity(a: List[int], b: List[int]) -> float
```

* Exact matches only
* Penalized for unequal sequence length
* Returns a small epsilon (`1e-7`) instead of zero to keep candidates rankable

---

### Vowel Similarity

```python
vowel_similarity(a: List[str], b: List[str]) -> float
```

* Exact vowel symbol matches
* Tail-aligned
* No vowel-distance modeling (by design, for now)

---

### Core Score

```python
core_score = 0.6 * stress_similarity + 0.4 * vowel_similarity
```

This is the **semantic gate**:

> If two words don’t rhyme *in principle*, they are rejected here.

The `threshold` CLI flag applies **only to this score**, not to ranking bonuses.

---

## Ranking Layer (Tie-Breakers)

Once a candidate passes the core threshold, the engine refines ordering using **small, bounded modifiers**.

These are deliberately weak compared to the core score.

### 1. Syllable Gating

```python
syllable_ok(base, cand, max_diff, max_syllables)
```

* Enforces rhythmic compatibility
* Used differently depending on mode:

  * `strict`: `max_diff = 0`
  * `end`: `max_diff = --max_syll_diff`

---

### 2. Syllable Closeness Bonus

```python
syllable_closeness_bonus(diff, max_diff)
```

* Smoothly favors equal syllable counts
* Multiplier ∈ `[1.0, 1.1]`

---

### 3. Rhyme-Tail Match Bonus

Adds a small bonus proportional to:

* number of matching tail vowels
* number of matching tail stresses

```python
tail_bonus = 0.03 * vowel_matches + 0.015 * stress_matches
```

---

### 4. Suffix Family Bonus

```python
suffix_bonus(base_word, cand_word)
```

Heuristic bias for morphological rhyme families:

Examples:

* `-ious` ↔ `-ious`
* `-arian` ↔ `-arian`
* `-ium` ↔ `-ium`

This is intentionally *non-linguistic* and *non-learned* — it encodes writer intuition.

---

### 5. Length Penalty

```python
length_penalty(base_word, cand_word)
```

Discourages extreme character-length mismatches.

---

### 6. Deterministic Jitter

```python
deterministic_jitter(token)
```

* Tiny (<1e-6)
* Stable across runs
* Prevents pathological score ties

---

## Final Score

```python
final_score = (
    core_score
  * syllable_closeness_bonus
  + tail_bonus
  + suffix_bonus
  - length_penalty
  + deterministic_jitter
)
```

Sorting is done on `final_score`.
Filtering is done on `core_score`.

This separation is **intentional**.

---

## Modes of Operation

### Strict Mode

```bash
--mode strict
```

* Requires exact syllable match
* Best for meter-sensitive writing (bars, hooks)

### End-Rhyme Mode

```bash
--mode end --max_syll_diff N
```

* Allows limited syllable variance
* Ranking still prefers closer matches
* Better for exploratory or free verse usage

---

## CLI Interface

### Minimal Example

```bash
python rhyme_engine.py --word precarious
```

### With G2P Enabled

```bash
python rhyme_engine.py --word mspacium --use_g2p
```

### End-Rhyme Exploration

```bash
python rhyme_engine.py --word nation --mode end --max_syll_diff 1
```

---

## Caching

### G2P Cache

* Stored in `g2p_cache.json`
* Keyed by lowercase word
* Written only when new entries are added
* Automatically reused across runs

This avoids repeated G2P inference and stabilizes results.

---

## Design Constraints (Intentional)

This engine **does not**:

* Learn from data
* Model semantics
* Use embeddings
* Generate words

It is a **phonetic comparator**, not a creative agent.

---

## Extension Points (For Future Devs)

Natural next steps:

1. **Manual pronunciation overrides**
   Insert before CMU resolution.

2. **Last-stressed-vowel anchoring**
   Compare rhyme tails starting from the final stressed vowel only.

3. **Vowel-distance metrics**
   Replace exact vowel match with phonetic proximity.

4. **Consonant coda modeling**
   Extend rhyme tail beyond vowels.

All of these can be added without changing the external API.

---

## Mental Model (TL;DR for Devs)

* CMU/G2P → phones
* phones → prosody
* prosody → tail-aligned similarity
* similarity → gated + ranked results

If you understand that pipeline, you understand the system.

