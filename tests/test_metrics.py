from vlm_eval_mini.metrics import exact_match, token_f1, keyword_recall, modality_score


def test_exact_match():
    assert exact_match("Hello", "hello") == 1.0


def test_token_f1_nonzero():
    assert token_f1("chart recall improved", "recall improved") > 0.0


def test_keyword_recall():
    assert keyword_recall("chart recall reranking", ["chart", "reranking"]) == 1.0


def test_modality_score():
    assert modality_score("Video frame evidence", ["video"]) == 1.0
