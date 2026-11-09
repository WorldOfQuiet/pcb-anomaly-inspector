from __future__ import annotations

import math


def ratio(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def normal_threshold(scores, alpha: float) -> dict:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1, exclusive")
    if not scores or not all(math.isfinite(score) for score in scores):
        raise ValueError("Calibration scores must be nonempty and finite")
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        raise ValueError(f"Insufficient calibration samples: n={n}, k={k}; increase n or alpha")
    return {"status": "calibrated", "threshold": sorted(scores)[k - 1], "n": n, "k": k,
            "alpha": alpha, "comparison": "greater_than"}


def latency(values) -> dict:
    import numpy as np
    return {"n": len(values), "unit": "seconds", "mean": float(np.mean(values)) if values else None,
            "p50": float(np.percentile(values, 50)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None,
            "max": max(values) if values else None}


def benchmark_metrics(rows, predictions) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    valid = [prediction for prediction in predictions if prediction["status"] == "ok"]
    errors = [prediction for prediction in predictions if prediction["status"] == "error"]
    if any(row.label is None for row in rows):
        raise ValueError("Benchmark requires labels for every selected input")
    labels = [prediction["label"] for prediction in valid]
    scores = [prediction["score"] for prediction in valid]
    tp = sum(p["label"] == 1 and p["decision"] == "suspect" for p in valid)
    fp = sum(p["label"] == 0 and p["decision"] == "suspect" for p in valid)
    tn = sum(p["label"] == 0 and p["decision"] == "normal" for p in valid)
    fn = sum(p["label"] == 1 and p["decision"] == "normal" for p in valid)
    decided = tp + fp + tn + fn
    both_classes = len(set(labels)) == 2
    return {"schema_version": 1, "status": "complete" if len(valid) == len(rows) and decided == len(rows) else "incomplete",
            "result_origin": "real_inference", "positive_class": "anomaly",
            "counts": {"inputs": len(rows), "attempted": len(predictions), "processed": len(valid), "failed": len(errors),
                       "unprocessed": len(rows) - len(predictions), "decided": decided, "undecided": len(valid) - decided,
                       "normal": sum(row.label == 0 for row in rows), "anomaly": sum(row.label == 1 for row in rows)},
            "accuracy": ratio(tp + tn, decided), "complete_case_accuracy": ratio(tp + tn, decided),
            "all_input_accuracy": ratio(tp + tn, len(rows)), "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn), "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "metric_denominators": {"accuracy": decided, "complete_case_accuracy": decided,
                                    "all_input_accuracy": len(rows), "precision": tp + fp,
                                    "recall": tp + fn, "f1": 2 * tp + fp + fn,
                                    "image_auroc": len(valid), "image_average_precision": len(valid)},
            "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
            "image_auroc": float(roc_auc_score(labels, scores)) if both_classes else None,
            "image_average_precision": float(average_precision_score(labels, scores)) if both_classes else None,
            "inference_latency": latency([p["inference_seconds"] for p in valid]),
            "total_latency": latency([p["total_seconds"] for p in valid]),
            "failures": [{"sample_id": p["sample_id"], "input_path": p["input_path"], "error": p["error"]} for p in errors],
            "scope": "Image ranking metrics use successful labeled inputs; accuracy is an alias of complete_case_accuracy on decided inputs. All_input_accuracy counts failed, undecided, and unprocessed inputs as incorrect. Every denominator and failure count is explicit.",
            "latency_protocol": "Batch size 1, no warmup; forward latency excludes decode/preprocessing/artifacts; total latency includes them and checksum verification. Model loading is excluded from per-image latency."}
