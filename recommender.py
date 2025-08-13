
import os, json, pickle, math, numpy as np

BASE = r"C:\Users\sagni\Downloads\SkillTracer Knowledge Tracing"
PP_PATH   = os.path.join(BASE, "preprocessor.pkl")
CATALOG   = os.path.join(BASE, "reco_catalog.pkl")
THRESHOLD = os.path.join(BASE, "threshold.json")

_pre = None
_cat = None
_thr = None

def _sigmoid(x):
    return 1.0/(1.0+math.exp(-x))

def _logit(p, eps=1e-6):
    p = min(max(p, eps), 1-eps)
    return math.log(p/(1-p))

def _load_pre():
    global _pre
    if _pre is None:
        with open(PP_PATH, "rb") as f:
            _pre = pickle.load(f)
    return _pre

def _load_cat():
    global _cat
    if _cat is None:
        with open(CATALOG, "rb") as f:
            _cat = pickle.load(f)
    return _cat

def _load_thr(default=0.5):
    global _thr
    if _thr is None:
        try:
            with open(THRESHOLD, "r", encoding="utf-8") as f:
                _thr = float(json.load(f).get("best_threshold", default))
        except Exception:
            _thr = default
    return _thr

def mastery_from_history(history, decay=0.3):
    """
    Exponential moving average per-skill:
    m_new = (1-decay)*m_prev + decay*correct
    Returns dict: skill -> mastery in [0,1], plus global p_student.
    """
    skill_m = {}
    skill_w = {}
    total_c = 0
    for e in history:
        if isinstance(e, dict):
            sk, cr = str(e.get("skill")), int(e.get("correct", 0))
        else:
            sk, cr = str(e[0]), int(e[1])
        if not sk:
            continue
        prev = skill_m.get(sk, 0.5)  # neutral start
        skill_m[sk] = (1.0 - decay)*prev + decay*cr
        skill_w[sk] = skill_w.get(sk, 0) + 1
        total_c += cr
    n = sum(skill_w.values())
    p_student = (total_c + 1) / (n + 2) if n > 0 else 0.5  # Laplace
    return skill_m, p_student

def recommend(history, top_k=5, target_low=0.60, target_high=0.75, min_item_count=30):
    """
    Returns top_k recommended problems (if available) or skills, aiming for predicted success in [target_low, target_high].
    Uses a simple 1PL-IRT estimate: for each skill, theta_skill=logit(p_student_skill) via EMA; for each item, P=σ(theta - b_item).
    Falls back to skill-level if no problem column in catalog.
    """
    pre = _load_pre()
    cat = _load_cat()
    skill_stats = cat["skill_stats"]
    item_stats  = cat["item_stats"]
    skill_to_items = cat["skill_to_items"]
    has_items = cat["meta"].get("problem_col_found", False)

    # per-skill mastery from history (EMA)
    skill_m, _ = mastery_from_history(history, decay=0.3)

    recs = []
    if has_items and len(item_stats) > 0:
        # item-level
        for sk, mastery in skill_m.items():
            base_p = skill_stats.get(sk, {}).get("p_correct", 0.5)
            theta = _logit( (0.9*mastery + 0.1*base_p) )
            for it in skill_to_items.get(sk, []):
                st = item_stats[it]
                if st["n"] < min_item_count:
                    continue
                p_hat = _sigmoid(theta - st["b"])
                if target_low <= p_hat <= target_high:
                    score = -abs((target_low+target_high)/2.0 - p_hat)
                    recs.append({"problem_id": it, "skill": sk, "pred_success": float(p_hat),
                                 "seen": int(st["n"]), "p_item": float(st["p_correct"]), "difficulty_b": float(st["b"]),
                                 "score": float(score)})
        # widen if not enough
        if len(recs) < top_k:
            extra = []
            band_low  = max(0.50, target_low - 0.10)
            band_high = min(0.85, target_high + 0.10)
            for sk, mastery in skill_m.items():
                base_p = skill_stats.get(sk, {}).get("p_correct", 0.5)
                theta = _logit( (0.9*mastery + 0.1*base_p) )
                for it in skill_to_items.get(sk, []):
                    st = item_stats[it]
                    if st["n"] < min_item_count:
                        continue
                    p_hat = _sigmoid(theta - st["b"])
                    if band_low <= p_hat <= band_high:
                        score = -abs((target_low+target_high)/2.0 - p_hat)
                        extra.append({"problem_id": it, "skill": sk, "pred_success": float(p_hat),
                                      "seen": int(st["n"]), "p_item": float(st["p_correct"]), "difficulty_b": float(st["b"]),
                                      "score": float(score)})
            recs = (recs + extra)
        recs.sort(key=lambda x: (x["score"], -x["seen"]), reverse=True)
        return recs[:top_k]
    else:
        # skill-level only
        for sk, mastery in skill_m.items():
            gap = 0.65 - mastery
            score = -abs(gap)
            recs.append({"skill": sk, "mastery": float(mastery), "score": float(score)})
        recs.sort(key=lambda x: x["score"], reverse=True)
        return recs[:top_k]
