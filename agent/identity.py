"""
agent/identity.py — Deterministic and Fuzzy Employee Identity Resolution.

Backed by the authoritative SQLite operators database.
Features:
- Candidate name extraction from speech transcript
- Normalization (whitespace, case, punctuation)
- Exact matching (full name, first name, employee code, operator ID)
- Fuzzy and phonetic matching with configurable confidence thresholds
- Strict ambiguity detection (refuses to guess between multiple close matches)
- Rejection of auto-inference from active shifts (spoken identity must be verified)
"""

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agent import db, shift_store
from agent.logging_setup import get_logger

log = get_logger("IDENTITY")

# Stopwords and common action words that must never be parsed as names
_NAME_STOPWORDS = {
    "i", "im", "i'm", "am", "main", "mera", "meri", "here", "speaking", "bol", "raha",
    "rahi", "hoon", "hu", "this", "is", "for", "by", "start", "starting", "started",
    "end", "ending", "ended", "shift", "now", "the", "a", "an", "and", "on", "in", "at", "to",
    "kelvin", "kelvin6k",
    "robot", "camera", "operator", "wall", "walls", "completed", "complete", "printing",
    "print", "line", "line_1", "line_2", "today", "yesterday", "please", "yes", "no",
    "okay", "ok", "done", "work", "working", "two", "three", "one", "four", "five",
    "facing", "having", "going", "ready", "clean", "cleaned", "clog", "issue", "problem",
    "actually", "record", "logged", "log", "mark", "marked",
    "when", "what", "who", "which", "how", "where", "why", "did", "does", "do",
    "was", "were", "have", "has", "had", "can", "could", "would", "should", "tell",
}

FUZZY_THRESHOLD = 0.85
AMBIGUITY_DELTA = 0.05


@dataclass
class IdentityResolution:
    status: str  # "MATCHED", "AMBIGUOUS", "UNKNOWN", "NONE"
    operator: Optional[Dict[str, Any]] = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    candidate_name: Optional[str] = None
    message: str = ""


def normalize_name(name: Any) -> str:
    """Strip punctuation, excessive whitespace, and lower-case."""
    if name is None:
        return ""
    cleaned = re.sub(r"[^\w\s]", " ", str(name))
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned



def extract_name_candidate(text: str) -> Optional[str]:
    """
    Extracts an employee name candidate from a spoken utterance.
    Supports English, Hindi, and Hinglish self-introductions and attributions.
    """
    if not text:
        return None

    clean = text.strip()

    # Pattern 1: "<Name> here", "<Name> speaking", "<Name> bol raha hoon/hu"
    m = re.search(r"\b([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(?:here|speaking|bol\s+rah[ia]\s+(?:hoon|hu))\b", clean, re.IGNORECASE)
    if m:
        cand = m.group(1).strip()
        first_word = cand.split()[0].lower()
        if first_word not in _NAME_STOPWORDS and len(cand) >= 3:
            return cand

    # Pattern 2: "completed by <Name>", "for <Name>", "recorded by <Name>", "start shift for <Name>"
    m = re.search(r"\b(?:completed|done|logged|printed|recorded|start(?:\s+(?:morning|day|evening|night|afternoon))?\s+shift)\s+(?:by|for)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b", clean, re.IGNORECASE)
    if m:
        cand = m.group(1).strip()
        first_word = cand.split()[0].lower()
        if first_word not in _NAME_STOPWORDS and len(cand) >= 3:
            return cand

    # Pattern 3: "No, it was <Name>", "It was <Name>", "Actually <Name>"
    m = re.search(r"\b(?:it\s+was|actually|no[, ]+it\s+was)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b", clean, re.IGNORECASE)
    if m:
        cand = m.group(1).strip()
        first_word = cand.split()[0].lower()
        if first_word not in _NAME_STOPWORDS and len(cand) >= 3:
            return cand

    # Pattern 4: "<Name> completed...", "<Name> printed...", "<Name> ne complete kiya"
    m = re.search(r"^([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(?:completed|printed|logged|ne\s+complete|did)\b", clean, re.IGNORECASE)
    if m:
        cand = m.group(1).strip()
        first_word = cand.split()[0].lower()
        if first_word not in _NAME_STOPWORDS and len(cand) >= 3:
            return cand

    # Pattern 5: "I am <Name>", "I'm <Name>", "Im <Name>", "This is <Name>", "Main <Name>"
    # Make sure we don't capture verbs ("I am starting work", "I'm done")
    m = re.search(r"\b(?:i\s*am|i['’]?m|this\s+is|main|my\s+name\s+is)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b", clean, re.IGNORECASE)
    if m:
        cand = m.group(1).strip()
        words = [w.lower() for w in cand.split()]
        if words and words[0] not in _NAME_STOPWORDS and len(cand) >= 3:
            # Check if second word is a stopword/verb
            if len(words) > 1 and words[1] in _NAME_STOPWORDS:
                cand = cand.split()[0]
            return cand

    return None


def resolve_employee(
    candidate_name: Optional[str],
    db_path: Optional[str] = None,
    active_only: bool = True,
) -> IdentityResolution:
    """
    Authoritative resolution of a candidate employee name against the database.
    Does NOT infer from active shifts.
    """
    if not candidate_name:
        log.info("[IDENTITY] No employee name candidate provided.")
        return IdentityResolution(status="NONE", message="No employee name candidate provided.")

    norm_cand = normalize_name(candidate_name)
    if not norm_cand or norm_cand in _NAME_STOPWORDS or len(norm_cand) < 3:
        log.info("[IDENTITY] Candidate %r normalized to empty, stopword, or too short.", candidate_name)
        return IdentityResolution(status="NONE", candidate_name=candidate_name, message="Name candidate not recognized.")

    log.info("[IDENTITY] Resolving candidate: %r (normalized: %r)", candidate_name, norm_cand)

    # 1. Fetch operators from DB
    try:
        with db.connect(db_path) as conn:
            sql = "SELECT * FROM operators"
            if active_only:
                sql += " WHERE (active = 1 OR status = 'Active')"
            rows = [dict(r) for r in conn.execute(sql)]
    except Exception as exc:
        log.error("[IDENTITY] Database error fetching operators: %s", exc)
        return IdentityResolution(status="UNKNOWN", candidate_name=candidate_name, message=f"Database error: {exc}")

    if not rows:
        log.warning("[IDENTITY] No operators registered in database.")
        return IdentityResolution(status="UNKNOWN", candidate_name=candidate_name, message="No operators found in database.")

    # 2. Exact and near-exact match check
    exact_matches = []
    near_matches = []
    for r in rows:
        r_name = normalize_name(r.get("name", ""))
        r_first = r_name.split()[0] if r_name else ""
        r_code = normalize_name(r.get("employee_code", "") or "")
        r_id = normalize_name(r.get("operator_id", "") or "")

        if norm_cand in (r_name, r_first, r_code, r_id):
            exact_matches.append(r)
        else:
            # Check for near-identical alternate transliterations (e.g. Hamdan vs Hamdaan)
            score = max(
                difflib.SequenceMatcher(None, norm_cand, r_name).ratio(),
                difflib.SequenceMatcher(None, norm_cand, r_first).ratio() if r_first else 0.0,
            )
            if score >= 0.88:
                near_matches.append((score, r))

    if len(exact_matches) == 1:
        # If another operator has an extremely close alternate spelling (e.g. both Hamdan and Hamdaan exist),
        # flag ambiguity to ensure clarification as required.
        if near_matches:
            all_ambiguous = [exact_matches[0]] + [nm[1] for nm in near_matches]
            log.warning("[IDENTITY] Exact match found (%s) but near-spelling collisions exist (%s)",
                        exact_matches[0]["name"], [o["name"] for o in all_ambiguous[1:]])
            return IdentityResolution(
                status="AMBIGUOUS",
                candidates=all_ambiguous,
                confidence=1.0,
                candidate_name=candidate_name,
                message=f"Multiple operators match '{candidate_name}' ({', '.join(o['name'] for o in all_ambiguous)}). Please clarify.",
            )

        matched = exact_matches[0]
        log.info("[IDENTITY] Exact match found: %s (%s)", matched["operator_id"], matched["name"])
        return IdentityResolution(
            status="MATCHED",
            operator=matched,
            candidates=exact_matches,
            confidence=1.0,
            candidate_name=candidate_name,
            message=f"Matched operator {matched['name']}.",
        )
    elif len(exact_matches) > 1:
        log.warning("[IDENTITY] Exact match ambiguous: %d candidates found for %r", len(exact_matches), candidate_name)
        return IdentityResolution(
            status="AMBIGUOUS",
            candidates=exact_matches,
            confidence=1.0,
            candidate_name=candidate_name,
            message=f"Multiple operators match '{candidate_name}'. Please clarify.",
        )

    # 3. Fuzzy matching
    scored_candidates: List[Tuple[float, Dict[str, Any]]] = []
    for r in rows:
        r_name = normalize_name(r.get("name", ""))
        r_first = r_name.split()[0] if r_name else ""

        # Candidates must share first character or be close phonetically (or length diff <= 2)
        cand_no_spaces = norm_cand.replace(" ", "")
        r_name_no_spaces = r_name.replace(" ", "")

        # Prevent short names matching substrings of completely different names (e.g. Aman inside Hamdan)
        # If first letter doesn't match and lengths differ by more than 1, skip
        if norm_cand and r_name:
            first_match = (norm_cand[0] == r_name[0]) or (norm_cand[0] in "ao" and r_name[0] in "ao")
            if not first_match and abs(len(cand_no_spaces) - len(r_name_no_spaces)) > 1:
                continue

        score_full = difflib.SequenceMatcher(None, norm_cand, r_name).ratio()
        score_first = difflib.SequenceMatcher(None, norm_cand, r_first).ratio() if r_first else 0.0
        score_no_spaces = difflib.SequenceMatcher(None, cand_no_spaces, r_name_no_spaces).ratio()

        best_score = max(score_full, score_first, score_no_spaces)
        if best_score >= FUZZY_THRESHOLD:
            scored_candidates.append((best_score, r))

    if not scored_candidates:
        log.info("[IDENTITY] No candidate exceeded threshold %.2f for %r", FUZZY_THRESHOLD, candidate_name)
        return IdentityResolution(
            status="UNKNOWN",
            candidate_name=candidate_name,
            confidence=0.0,
            message=f"Operator '{candidate_name}' not found in database.",
        )

    # Sort descending by score
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    top_score, top_op = scored_candidates[0]

    # Check for ambiguity: if second candidate has score close to top
    if len(scored_candidates) > 1:
        second_score, second_op = scored_candidates[1]
        if (top_score - second_score) < AMBIGUITY_DELTA:
            log.warning(
                "[IDENTITY] Ambiguous fuzzy match between %s (%.2f) and %s (%.2f)",
                top_op["name"], top_score, second_op["name"], second_score,
            )
            return IdentityResolution(
                status="AMBIGUOUS",
                candidates=[top_op, second_op],
                confidence=top_score,
                candidate_name=candidate_name,
                message=f"Ambiguous match between {top_op['name']} and {second_op['name']}. Please clarify.",
            )

    log.info("[IDENTITY] Fuzzy match found: %s (%s) with score %.2f", top_op["operator_id"], top_op["name"], top_score)
    return IdentityResolution(
        status="MATCHED",
        operator=top_op,
        candidates=[top_op],
        confidence=top_score,
        candidate_name=candidate_name,
        message=f"Matched operator {top_op['name']} (confidence {top_score:.2f}).",
    )
