"""extras.py: report drafting + patient visit history (SQLite, no extra setup).
Only ID, age, scores and the top finding are stored (never names or images)."""
import json
import re
import sqlite3
import time
from datetime import datetime

DB_PATH = "visits.db"


def _first_sentences(text, n=2):
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return " ".join(parts[:n]).strip()


GLOSS = {
    "Atelectasis": "partial collapse of lung tissue",
    "Cardiomegaly": "an enlarged heart",
    "Consolidation": "a solid-looking patch in the lung",
    "Edema": "fluid in the lungs",
    "Pleural Effusion": "fluid around the lung",
    "Pneumonia": "infection of the lung tissue",
    "Pneumothorax": "air around the lung",
}
NEXT_STEP = "Please share this report with your doctor, who can review the image directly and decide on any further tests."


def _phrase(p):
    """Wording that follows the score band, so a mid-range score isn't overstated."""
    return "consistent with" if p >= 0.5 else "suggestive of" if p >= 0.35 else "that may indicate"


def build_report(top, age, symptoms, ranked=None, close_to=None):
    """top: {"name", "probability", "description"} for the most likely finding, or None.
    ranked: the other classes as [{"name", "probability"}], best first."""
    ctx = "; ".join(c for c in [f"{age}-year-old patient" if age is not None else "",
                                f"reported symptoms: {symptoms}" if symptoms else ""] if c)
    if not top:
        return {"engine": "template", "context": ctx,
                "summary": "No finding stood out on this frontal chest X-ray.",
                "findings": "None of the seven conditions screened was singled out by the model on this image.",
                "impression": "No clear finding. This does not rule out disease.",
                "meaning": "", "next": NEXT_STEP, "note": ""}
    name, p = top["name"], top["probability"]
    low, ph = name.lower(), _phrase(p)
    gloss = GLOSS.get(name)
    summary = (f"The analysis of this frontal chest X-ray found features {ph} {low}"
               f"{f' ({gloss})' if gloss else ''}, at {p:.1%} model confidence. "
               "The area that most influenced the result is marked on the image.")
    notable = [c for c in (ranked or []) if c["name"] != "No Finding"][:3]
    findings = f"Strongest signal: {low} ({p:.1%})."
    if notable:
        findings += " Other notable scores: " + ", ".join(f"{c['name'].lower()} ({c['probability']:.1%})" for c in notable) + "."
    findings += " Scores are assessed independently and do not add up to 100%."
    impression = f"Features {ph} {low}."
    if close_to:
        impression += f" The score is close to that of {close_to.lower()}, so both deserve attention."
    return {"engine": "template", "context": ctx, "summary": summary, "findings": findings,
            "impression": impression, "meaning": _first_sentences(top.get("description"), 2),
            "next": NEXT_STEP, "note": ""}


def _connect():
    db = sqlite3.connect(DB_PATH)
    db.execute("CREATE TABLE IF NOT EXISTS visits (pid TEXT, ts REAL, age INTEGER, probs TEXT, flagged TEXT, top TEXT)")
    cols = [r[1] for r in db.execute("PRAGMA table_info(visits)")]
    if "top" not in cols:                       # database made by the earlier version
        db.execute("ALTER TABLE visits ADD COLUMN top TEXT")
    return db


def _fmt(ts):
    return datetime.fromtimestamp(ts).strftime("%d %b %Y, %H:%M")


def _row_top(probs_json, flagged_json, top):
    """Top finding of a stored visit. Older rows have no 'top', so use the highest flagged score."""
    if top:
        return top
    probs, flagged = json.loads(probs_json), json.loads(flagged_json)
    return max(flagged, key=lambda n: probs.get(n, 0)) if flagged else None


def save_and_compare(pid, age, classes, top_name):
    """Store this visit and compare its top finding with the previous visit for the same patient ID."""
    if not pid:
        return None
    now = {c["name"]: c["probability"] for c in classes}
    hot = sorted(c["name"] for c in classes if c["positive"] and c["name"] != "No Finding")
    db = _connect()
    try:
        rows = db.execute("SELECT ts, probs, flagged, top FROM visits WHERE pid=? ORDER BY ts DESC LIMIT 6", (pid,)).fetchall()
        db.execute("INSERT INTO visits VALUES (?,?,?,?,?,?)",
                   (pid, time.time(), age, json.dumps(now), json.dumps(hot), top_name))
        db.commit()
    finally:
        db.close()
    if not rows:
        return {"returning": False}
    ts, probs, flagged, top = rows[0]
    prev_top = _row_top(probs, flagged, top)
    return {
        "returning": True,
        "when": _fmt(ts),
        "prev_top": prev_top,
        "prev_score": json.loads(probs).get(prev_top) if prev_top else None,
        "now_top": top_name,
        "now_score": now.get(top_name) if top_name else None,
        "changed": prev_top != top_name,
        "earlier": [{"when": _fmt(r[0]), "top": _row_top(r[1], r[2], r[3])} for r in rows[1:5]],
    }


# ---- Hooks for the later BioGPT + LoRA and MongoDB work ----
# Flip these when the real implementations are connected; the UI already reads them.
CONFIG = {"report_engine": "template", "database": "sqlite", "biogpt": False, "mongodb": False}


def status():
    return dict(CONFIG)


def history(pid, limit=10):
    """Stored visits for a patient ID, newest first. Swap the body for a MongoDB query later."""
    if not pid:
        return []
    db = _connect()
    try:
        rows = db.execute("SELECT ts, age, probs, flagged, top FROM visits WHERE pid=? ORDER BY ts DESC LIMIT ?",
                          (pid, limit)).fetchall()
    finally:
        db.close()
    return [{"when": _fmt(t), "age": a, "flagged": json.loads(f), "top": _row_top(p, f, tp)} for t, a, p, f, tp in rows]


def generate_report(generator, findings, age, symptoms):
    """Plug BioGPT + LoRA in here. Return the same dict shape as build_report() with engine="biogpt"."""
    raise NotImplementedError("BioGPT + LoRA isn't connected yet.")