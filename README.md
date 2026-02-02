# Lyrical Lab – Prosody-Based Rhyme Engine

A **pronunciation-driven rhyme engine** designed for lyricists, poets, and music-oriented writing tools.
This engine finds and ranks rhymes based on **how words sound**, not how they’re spelled.

It is a core component of **Lyrical Lab**.

---

## Overview

Traditional rhyme tools rely on orthography (letters).
This engine relies on **prosody**:

* syllable count
* stress patterns
* vowel sequences
* rhyme-tail alignment

The result is a system that supports:

* multisyllabic rhymes
* slant rhymes
* rhyme ranking (not just yes/no matches)
* rhythm-aware constraints
* invented words and brand names

This is a **non-ML**, deterministic engine intended to model how writers *hear* rhyme.

---

## Key Features

* **Pronunciation-first design** (phonetics, not spelling)
* **Tail-aligned rhyme comparison**
* **Multiple pronunciations per word**
* **Strict vs exploratory rhyme modes**
* **Ranked results by rhyme quality**
* **G2P fallback for out-of-vocabulary words**
* **Caching for deterministic behavior**

---

## How It Works (High-Level)

1. Resolve pronunciations for the query word
2. Convert pronunciations into prosodic representations
3. Compare against a pronunciation dictionary
4. Filter candidates using a core prosody threshold
5. Rank valid rhymes using controlled tie-breakers
6. Return top-N results

The engine separates **rhyme detection** from **rhyme ranking** intentionally.

---

## Prosody Representation

Each pronunciation is reduced to:

```python
{
  "stress":    [0, 1, 0, 0],
  "vowels":    ["EH", "IY", "AH"],
  "syllables": 4
}
```

* Stress values are normalized (`2 → 1`)
* Vowels define rhyme resonance
* Consonants are not directly modeled (by design)

---

## Pronunciation Resolution Order

For each word:

1. **Manual overrides** (if present)
2. **CMU Pronouncing Dictionary** (`pronouncing`)
3. **Cached G2P result**
4. **G2P inference** (optional, via `g2p_en`)

Multiple pronunciations are preserved at every step.

---

## Rhyme Comparison

### Tail-Aligned Similarity

All comparisons are made **from the end of the word backward**, reflecting how rhyme perception works acoustically.

Similarity is computed separately for:

* stress patterns
* vowel sequences

These are combined into a **core prosody score**.

---

## Core Score vs Ranking

### Core Prosody Score (Hard Gate)

Determines whether two words rhyme *in principle*.

* Stress similarity (tail-aligned)
* Vowel similarity (tail-aligned)

If the score is below the threshold, the word is rejected.

---

### Ranking Layer (Soft Refinement)

Applied **only after** passing the core gate:

* syllable-count closeness bonus
* rhyme-tail depth bonus
* suffix-family affinity (`-ious`, `-arian`, `-ium`, etc.)
* word-length penalty
* deterministic tie-breaking jitter

These modifiers refine ordering but **cannot override** core rhyme validity.

---

## Modes

### Strict Mode

```bash
--mode strict
```

* Requires exact syllable match
* Best for rhythm-sensitive writing (bars, hooks)

---

### End-Rhyme Mode

```bash
--mode end --max_syll_diff N
```

* Allows limited syllable variation
* Still ranked by rhythmic closeness
* Useful for exploratory writing

---

## CLI Usage

Basic usage:

```bash
python rhyme_engine.py --word precarious
```

With G2P enabled:

```bash
python rhyme_engine.py --word mspacium --use_g2p
```

End-rhyme exploration:

```bash
python rhyme_engine.py --word nation --mode end --max_syll_diff 1
```


## What This Engine Does *Not* Do

* No lyric generation
* No semantic understanding
* No embeddings or ML models
* No spelling-based heuristics

This is a **phonetic comparison engine**, not a creative agent.

---

## Intended Use

This engine is designed to be embedded into:

* Lyrical Lab’s writing environment
* creative analysis tools
* educational software for rhyme and prosody
* advanced rhyme exploration workflows

It augments human judgment rather than replacing it.

---

## Development Roadmap

Planned improvements include:

* anchoring rhyme comparison at the **last stressed vowel**
* improved consonant-coda modeling
* UI-exposed pronunciation overrides
* deeper integration with Lyrical Lab’s editor

---

## Design Philosophy

> Rhymes are not binary.
> They have depth, strength, and rhythm.

This engine exists to make that structure explicit and inspectable.

