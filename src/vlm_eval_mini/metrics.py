import re


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(ref) else 0.0


def token_f1(pred: str, ref: str) -> float:
    p = _normalize(pred).split()
    r = _normalize(ref).split()
    if not p or not r:
        return 0.0

    p_counts = {}
    r_counts = {}
    for token in p:
        p_counts[token] = p_counts.get(token, 0) + 1
    for token in r:
        r_counts[token] = r_counts.get(token, 0) + 1

    overlap = 0
    for token, c in p_counts.items():
        overlap += min(c, r_counts.get(token, 0))

    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def keyword_recall(pred: str, required_keywords: list[str]) -> float:
    if not required_keywords:
        return 1.0
    pred_n = _normalize(pred)
    hit = sum(1 for kw in required_keywords if _normalize(kw) in pred_n)
    return hit / len(required_keywords)


def modality_score(pred: str, required_modalities: list[str]) -> float:
    if not required_modalities:
        return 1.0
    pred_n = _normalize(pred)
    hits = 0
    for m in required_modalities:
        if m == "image" and any(x in pred_n for x in ["image", "chart", "figure", "diagram"]):
            hits += 1
        elif m == "text" and any(x in pred_n for x in ["text", "document", "caption", "note"]):
            hits += 1
        elif m == "video" and any(x in pred_n for x in ["video", "frame", "clip"]):
            hits += 1
    return hits / len(required_modalities)
