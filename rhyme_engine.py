
"""
rhyme_engine_g2p_ready.py

Ranked prosody rhyme engine + OPTIONAL G2P fallback (via g2p_en).

- Uses CMU (pronouncing) when available.
- If --use_g2p and g2p_en is installed, generates pronunciations for OOV words.
- Caches G2P prosodies to g2p_cache.json to avoid recomputation.
- Compatible with existing stress_dictionary.json formats:
    * word -> {stress,vowels,syllables}
    * word -> [ {..}, {..}, ... ]

Install optional dependency:
    pip install g2p_en

Examples:
    python rhyme_engine.py --word precarious --mode strict --top 20
    python rhyme_engine.py --word mspacium --mode strict --top 20 --use_g2p
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pronouncing
import logging
logging.basicConfig(level=logging.DEBUG)
import nltk
from pathlib import Path

Prosody = Dict[str, Any]
ProsodyStore = Union[Prosody, List[Prosody]]

def setup_nltk():

    base = Path(__file__).parent
    nltk_path = base / "nltk_data"

    nltk.data.path.append(str(nltk_path))

    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        raise RuntimeError(
            "NLTK tagger missing. Ensure nltk_data folder is included."
        )

# -----------------------------
# Optional G2P
# -----------------------------
_G2P_AVAILABLE = False
try:
    from g2p_en import G2p  # type: ignore
    _G2P_AVAILABLE = True
except Exception:
    _G2P_AVAILABLE = False


def _g2p_phones_for_word(word: str) -> List[str]:
    """Return a list of phones strings (CMU-ish) from g2p_en."""
    if not _G2P_AVAILABLE:
        return []
    g2p = G2p()
    toks = g2p(word) # example: ['M', 'AH0', 'S', 'P', 'IY1', 'Z', 'AH0', 'M']
    phones: List[str] = []
    for t in toks:
        if not t or t.isspace():
            continue
        if t in {"'", '"', ".", ",", "!", "?", ":", ";", "-", "—", "–", "(", ")", "[", "]", "{", "}"}:
            continue
        phones.append(t) 
    print(f"The phones: {phones}") # example: ['G', 'IY1', 'P', 'AH0', 'T', 'IY0']
    return [" ".join(phones)] if phones else []


# -----------------------------
# Prosody extraction
# -----------------------------
def _prosodies_from_phones(phones_list: List[str], normalize_secondary_stress: bool = True) -> List[Prosody]:
    prosodies: List[Prosody] = []
    for phones in phones_list:
        stresses = pronouncing.stresses(phones) #example:  0100
        logging.debug(f"The stresses: {stresses}")
        stress = [int(s) for s in stresses if s.isdigit()]
        if normalize_secondary_stress:
            stress = [1 if x == 2 else x for x in stress]

        parts = phones.split()
        vowels = [ph for ph in parts if any(ch.isdigit() for ch in ph)]
        vowel_seq = [v[:-1] for v in vowels]
        syllables = len(vowel_seq)
        if syllables == 0:
            continue
        prosodies.append({"stress": stress, "vowels": vowel_seq, "syllables": syllables})
        logging.debug(f"The prosody list: {prosodies}")

    # de-dup
    uniq: List[Prosody] = []
    seen = set()
    for p in prosodies:
        key = (tuple(p["stress"]), tuple(p["vowels"]), int(p["syllables"]))
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def get_prosodies_cmu(word: str) -> List[Prosody]:
    phones_list = pronouncing.phones_for_word(word)
    logging.debug(f"The phones_list of the word: {phones_list}") #['TH AH1 N D ER0 B AO2 L T']
    if not phones_list:
        return []
    return _prosodies_from_phones(phones_list, normalize_secondary_stress=True)


def as_prosody_list(value: Optional[ProsodyStore]) -> List[Prosody]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


# -----------------------------
# Similarity (tail-aligned)
# -----------------------------
def tail_match_stats(a: List[Any], b: List[Any]) -> Tuple[int, int, int]:
    min_len = min(len(a), len(b))
    max_len = max(len(a), len(b))
    if min_len == 0:
        return 0, 0, max_len
    matches = sum(1 for i in range(1, min_len + 1) if a[-i] == b[-i])
    return matches, min_len, max_len


def stress_similarity(a: List[int], b: List[int]) -> float:
    matches, min_len, max_len = tail_match_stats(a, b)
    if min_len == 0:
        return 0.0
    length_penalty = (min_len / max_len) if max_len else 1.0
    return ((matches / min_len) * length_penalty) if matches else 1e-7


def vowel_similarity(a: List[str], b: List[str]) -> float:
    matches, min_len, _ = tail_match_stats(a, b)
    if min_len == 0:
        return 0.0
    return (matches / min_len) if matches else 1e-7


# -----------------------------
# Ranking tie-breakers
# -----------------------------
def syllable_ok(base_s: int, cand_s: int, max_diff: int, max_syllables: Optional[int]) -> bool:
    if max_syllables is not None and cand_s > max_syllables:
        return False
    return abs(base_s - cand_s) <= max_diff


def syllable_closeness_bonus(diff: int, max_diff: int, weight: float = 0.10) -> float:
    if max_diff <= 0:
        return 1.0
    diff = max(0, min(diff, max_diff))
    closeness = (max_diff - diff) / max_diff
    return 1.0 + (weight * closeness)


_SUFFIX_GROUPS = [
    "ification", "ization", "ational",
    "entious", "arious", "erious", "orious",
    "acious", "icious",
    "eous", "ious",
    "arium", "arian",
    "ennial",
    "eria", "alia",
    "ium", "ial", "ia",
]


def suffix_bonus(base_word: str, cand_word: str) -> float:
    bw, cw = base_word.lower(), cand_word.lower()
    for suf in _SUFFIX_GROUPS:
        if bw.endswith(suf) and cw.endswith(suf):
            return min(0.06, 0.01 + 0.01 * (len(suf) / 4))
    if len(bw) >= 4 and len(cw) >= 4 and bw[-4:] == cw[-4:]:
        return 0.015
    return 0.0


def length_penalty(base_word: str, cand_word: str) -> float:
    a, b = len(base_word), len(cand_word)
    if not a or not b:
        return 0.0
    rel = abs(a - b) / max(a, b)
    return 0.02 * rel


def deterministic_jitter(token: str) -> float:
    h = 0
    for ch in token:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return (h % 1_000_000) / 1_000_000_000_000  # < 1e-6


# -----------------------------
# Scoring
# -----------------------------
def core_score(a: Prosody, b: Prosody, w_stress: float = 0.6, w_vowel: float = 0.4) -> float:
    return (w_stress * stress_similarity(a["stress"], b["stress"])) + (w_vowel * vowel_similarity(a["vowels"], b["vowels"]))


def best_score(
    base_word: str,
    cand_word: str,
    base_pros: List[Prosody],
    cand_pros: List[Prosody],
    *,
    max_syll_diff: int,
    max_syllables: Optional[int],
) -> Tuple[float, float]:
    best_final = 0.0
    best_core = 0.0

    suf_b = suffix_bonus(base_word, cand_word)
    len_pen = length_penalty(base_word, cand_word)
    jit = deterministic_jitter(cand_word)

    for bp in base_pros:
        bs = int(bp["syllables"])
        for cp in cand_pros:
            cs = int(cp["syllables"])
            if not syllable_ok(bs, cs, max_syll_diff, max_syllables):
                continue

            c = core_score(bp, cp)
            c *= syllable_closeness_bonus(abs(bs - cs), max_syll_diff, weight=0.10)

            v_matches, _, _ = tail_match_stats(bp["vowels"], cp["vowels"])
            s_matches, _, _ = tail_match_stats(bp["stress"], cp["stress"])
            tail_bonus = 0.03 * v_matches + 0.015 * s_matches

            final = c + tail_bonus + suf_b - len_pen + jit
            if final > best_final:
                best_final = final
                best_core = c
    return best_final, best_core


# -----------------------------
# G2P cache + fallback
# -----------------------------
def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_prosodies(word: str, *, use_g2p: bool, cache: Dict[str, Any], dirty: List[bool]) -> List[Prosody]:
    cmu = get_prosodies_cmu(word)
    if cmu:
        return cmu

    key = word.lower()
    if key in cache:
        return as_prosody_list(cache[key])

    if not use_g2p:
        return []

    phones_list = _g2p_phones_for_word(word)
    pros = _prosodies_from_phones(phones_list, normalize_secondary_stress=True)
    if pros:
        cache[key] = pros
        dirty[0] = True
    return pros


# -----------------------------
# Search
# -----------------------------
def find_rhymes(
    word: str,
    db: Dict[str, ProsodyStore],
    *,
    top_n: int,
    threshold: float,
    strict_length: bool,
    max_syll_diff_loose: int,
    max_syllables: Optional[int],
    use_g2p: bool,
    g2p_cache: Dict[str, Any],
    dirty: List[bool],
) -> List[Tuple[str, float]]:
    base_pros = get_prosodies(word, use_g2p=use_g2p, cache=g2p_cache, dirty=dirty)
    if not base_pros:
        return []

    max_diff = 0 if strict_length else max_syll_diff_loose

    out: List[Tuple[str, float]] = []
    for cand, stored in db.items():
        if cand.lower() == word.lower():
            continue
        cand_pros = as_prosody_list(stored)
        if not cand_pros:
            continue
        final, core = best_score(word, cand, base_pros, cand_pros, max_syll_diff=max_diff, max_syllables=max_syllables)
        if core >= threshold and final > 0:
            out.append((cand, final))

    out.sort(key=lambda x: x[1], reverse=True)
    return out[:top_n]



#---------------------
# For APIs and desktop
#---------------------
# def find_rhymes_api(
#     word: str,
#     *,
#     db_path: Optional[Path] = None,
#     g2p_cache_path: Optional[Path] = None,
#     top_n: int = 50,
#     threshold: float = 0.6,
#     strict_length: bool = False,
#     max_syll_diff_loose: int = 2,
#     max_syllables: Optional[int] = None,
#     use_g2p: bool = True
# ) -> List[Tuple[str, float]]:

#     """
#     Public function to find rhymes without using CLI.

#     Can be used in APIs, desktop apps, or other modules.
#     """
#     default_path_db = Path(__file__).parent / "stress_dictionary.json"
#     default_path_g2p = Path(__file__).parent / "g2p_cache.json"

#     setup_nltk()

#     # Load database
#     db: Dict[str, Any] = load_json(default_path_db)

#     # Load G2P cache
#     g2p_cache: Dict[str, Any] = load_json(default_path_g2p)

#     dirty = [False]

#     results = find_rhymes(
#         word,
#         db,
#         top_n=top_n,
#         threshold=threshold,
#         strict_length=strict_length,
#         max_syll_diff_loose=max_syll_diff_loose,
#         max_syllables=max_syllables,
#         use_g2p=use_g2p,
#         g2p_cache=g2p_cache,
#         dirty=dirty
#     )

#     # Save cache if updated
#     if dirty[0]:
#         save_json(g2p_cache_path, g2p_cache)

#     return results




import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from itertools import product


# -----------------------------
# Split phrase into words
# -----------------------------
def split_phrase(phrase: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", phrase.lower())


# -----------------------------
# Build phrasal rhymes
# -----------------------------
import random
from itertools import product


def build_phrasal_rhymes(
    phrase_results,
    top_phrases=30,
    min_phrase_score=0.7,
    randomize=True,
    diversity_strength=0.3
):

    word_lists = list(phrase_results.values())
    combinations = list(product(*word_lists))

    phrases = []

    for combo in combinations:
        words = [w[0] for w in combo]
        score = sum(w[1] for w in combo) / len(combo)

        if score >= min_phrase_score:
            phrases.append((" ".join(words), score))

    # sort by score
    phrases.sort(key=lambda x: x[1], reverse=True)

    if not randomize:
        return phrases[:top_phrases]

    # weighted randomness
    top_pool = phrases[:int(len(phrases) * diversity_strength) or 10]

    random.shuffle(top_pool)

    return top_pool[:top_phrases]

# -----------------------------
# Find rhymes for phrase
# -----------------------------
def find_rhymes_api(
    phrase: str,
    *,
    db_path: Optional[Path] = None,
    g2p_cache_path: Optional[Path] = None,
    top_n: int = 30,
    threshold: float = 0.8,
    strict_length: bool = False,
    max_syll_diff_loose: int = 2,
    max_syllables: Optional[int] = None,
    use_g2p: bool = True,
    top_phrases: int = 50,
    min_phrase_score: float = 0.8
) -> Dict[str, Any]:

    default_db = Path(__file__).parent / "stress_dictionary.json"
    default_g2p = Path(__file__).parent / "g2p_cache.json"

    db_path = db_path or default_db
    g2p_cache_path = g2p_cache_path or default_g2p

    setup_nltk()

    db: Dict[str, Any] = load_json(db_path)
    g2p_cache: Dict[str, Any] = load_json(g2p_cache_path)

    dirty = [False]

    words = split_phrase(phrase)

    print(f"words are: {words}")

    phrase_results: Dict[str, List[Tuple[str, float]]] = {}


    if len(words) == 1:
        results = find_rhymes(
            words[0],
            db,
            top_n=top_n,
            threshold=threshold,
            strict_length=strict_length,
            max_syll_diff_loose=max_syll_diff_loose,
            max_syllables=max_syllables,
            use_g2p=use_g2p,
            g2p_cache=g2p_cache,
            dirty=dirty
        )

        phrase_results[words[0]] = results
        phrasal_rhymes = {}
    else:
        # -----------------------------
        # Find rhymes per word
        # -----------------------------
        for word in words:
            results = find_rhymes(
                word,
                db,
                top_n=top_n,
                threshold=threshold,
                strict_length=strict_length,
                max_syll_diff_loose=max_syll_diff_loose,
                max_syllables=max_syllables,
                use_g2p=use_g2p,
                g2p_cache=g2p_cache,
                dirty=dirty
            )

            phrase_results[word] = results

        # -----------------------------
        # Build phrasal rhymes
        # -----------------------------
        phrasal_rhymes = build_phrasal_rhymes(
            phrase_results,
            top_phrases=top_phrases,
            min_phrase_score=min_phrase_score
        )

    # Save cache
    if dirty[0]:
        save_json(g2p_cache_path, g2p_cache)

    return {
        "input": phrase,
        "words": words,
        "word_rhymes": phrase_results,
        "phrasal_rhymes": phrasal_rhymes
    }




def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--word", required=True)
    p.add_argument("--db", default="stress_dictionary.json")
    p.add_argument("--mode", choices=["strict", "end"], default="strict")
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--threshold", type=float, default=0.65)
    p.add_argument("--max_syll_diff", type=int, default=1)
    p.add_argument("--max_syllables", type=int, default=None)
    p.add_argument("--use_g2p", action="store_true")
    p.add_argument("--g2p_cache", default="g2p_cache.json")
    args = p.parse_args()

    strict = args.mode == "strict"
    db_path = Path(args.db)
    with open(db_path, "r", encoding="utf-8") as f:
        db = json.load(f)

    cache_path = Path(args.g2p_cache)
    cache = load_json(cache_path)
    dirty = [False]

    if args.use_g2p and not _G2P_AVAILABLE:
        print(" --use_g2p set but g2p_en is not installed. Install with: pip install g2p_en")
        print("    Continuing without G2P fallback.\n")

    rhymes = find_rhymes(
        args.word, db,
        top_n=args.top,
        threshold=args.threshold,
        strict_length=strict,
        max_syll_diff_loose=args.max_syll_diff,
        max_syllables=args.max_syllables,
        use_g2p=args.use_g2p and _G2P_AVAILABLE,
        g2p_cache=cache,
        dirty=dirty,
    )

    if dirty[0]:
        save_json(cache_path, cache)

    print(f"\nWord: {args.word} | mode={args.mode} | threshold(core)={args.threshold} | top={args.top}")
    if args.max_syllables is not None:
        print(f"Max candidate syllables: {args.max_syllables}")
    print("-" * 60)
    for w, s in rhymes:
        print(f"{w:<20} {s:.4f}")


if __name__ == "__main__":
    import json
    # main()
    rhymes = find_rhymes_api("Time will tell")
    rhymes = json.dumps(rhymes)
    print(rhymes)
