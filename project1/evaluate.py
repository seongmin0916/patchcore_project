import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import pandas as pd
import torch

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

from project1.imageloading import collect_image_list
from project1.modeling import (
    Autoencoder,
    create_model,
    epoch_loss,
    fit_memory,
    predict_autoencoder,
    predict_scores,
    set_seed,
    train_autoencoder,
)
from project1.preprocessing import make_loaders, split_records


EXPERIMENTS = [
    ("baseline", "patchcore", 224, 0.01),
    ("memory_10pct", "patchcore", 224, 0.10),
    ("size_384", "patchcore", 384, 0.01),
    ("autoencoder", "autoencoder", 224, None),
]

SEED = 42
BATCH_SIZE = 8
VALID_RATIO = 0.2
THRESHOLD_QUANTILE = 0.95
AE_TRAINING = dict(max_epochs=20, patience=3, learning_rate=0.001, min_delta=0.00001)


def calculate_metrics(scores, labels, threshold):
    predictions = (scores >= threshold).long()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels.numpy(), predictions.numpy(), average="binary", zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(labels.numpy(), predictions.numpy(), labels=[0, 1]).ravel()
    return {
        "auroc": roc_auc_score(labels.numpy(), scores.numpy()),
        "precision": precision, "recall": recall, "f1": f1,
        "false_positive_rate": fp / (tn + fp),
        "missed_defects": int(fn), "threshold": threshold,
    }


def evaluate_experiment(groups, model_type, size, ratio, device):
    set_seed(SEED)
    train_loader, valid_loader, test_loader = make_loaders(
        *groups, size=size, batch_size=BATCH_SIZE, seed=SEED
    )
    if model_type == "patchcore":
        model = create_model(pretrained=True, seed=SEED)
        predict = predict_scores
    elif model_type == "autoencoder":
        model = Autoencoder().to(device)
        predict = predict_autoencoder
    else:
        raise ValueError(f"지원하지 않는 모델: {model_type}")

    try:
        if model_type == "patchcore":
            model = fit_memory(model, train_loader, ratio=ratio, device=device)
        else:
            model = train_autoencoder(model, train_loader, valid_loader, device, **AE_TRAINING)
        valid_scores, _ = predict(model, valid_loader)
        threshold = torch.quantile(valid_scores, THRESHOLD_QUANTILE).item()
        test_scores, test_labels = predict(model, test_loader)
        return calculate_metrics(test_scores, test_labels, threshold)
    finally:
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()


def experiment_metadata(data_dir, grouped_data, device):
    dataset_entries = []
    for product, groups in grouped_data.items():
        for split, records in zip(("train", "valid", "test"), groups):
            for record in records:
                path = Path(record["image_path"])
                stat = path.stat()
                dataset_entries.append((product, split, str(path.relative_to(data_dir)),
                                        record["label"], stat.st_size, stat.st_mtime_ns))
    source_dir = Path(__file__).resolve().parent
    sources = {
        name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
        for name in ("evaluate.py", "modeling.py", "preprocessing.py", "imageloading.py")
    }
    packages = ("torch", "torchvision", "anomalib", "timm", "numpy", "pandas",
                "scikit-learn", "pillow", "lightning")
    return {
        "experiments": [list(experiment) for experiment in EXPERIMENTS],
        "seed": SEED, "batch_size": BATCH_SIZE, "valid_ratio": VALID_RATIO,
        "threshold_quantile": THRESHOLD_QUANTILE, "autoencoder_training": AE_TRAINING,
        "backbone": "wide_resnet50_2.racm_in1k", "device": device,
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "sources": sources,
        "dataset_signature": hashlib.sha256(json.dumps(dataset_entries).encode()).hexdigest(),
    }


def compare_models(data_dir, product=None, result_path=None, resume=False):
    data_dir = Path(data_dir).resolve()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    product_dirs = [data_dir / product] if product else sorted(p for p in data_dir.iterdir() if p.is_dir())
    grouped_data = {
        folder.name: split_records(collect_image_list(folder), valid_ratio=VALID_RATIO, seed=SEED)
        for folder in product_dirs
    }
    results = []
    if result_path is not None:
        result_path = Path(result_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        if result_path.exists() and not resume:
            raise FileExistsError("결과 파일이 존재합니다. --resume 또는 새 --result-path를 사용하세요.")
        if resume and result_path.exists():
            results = pd.read_json(result_path).to_dict("records")
    completed = {(row["product"], row["experiment"]) for row in results}
    expected = {(product, name) for product in grouped_data for name, *_ in EXPERIMENTS}
    if len(completed) != len(results) or not completed.issubset(expected):
        raise ValueError("저장된 결과에 중복 또는 이번 평가 범위에 없는 조건이 있습니다.")
    if result_path is not None and expected - completed:
        metadata = experiment_metadata(data_dir, grouped_data, device)
        metadata_path = result_path.with_suffix(".meta.json")
        if results:
            if not metadata_path.exists() or json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
                raise ValueError("실험 설정·코드·환경·데이터가 다르거나 실행 기록이 없습니다. 새 --result-path를 사용하세요.")
        temporary = metadata_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        temporary.replace(metadata_path)
    for product_name, groups in grouped_data.items():
        paths = [{r["image_path"] for r in group} for group in groups]
        if paths[0] & paths[1] or paths[0] & paths[2] or paths[1] & paths[2]:
            raise ValueError("학습·검증·테스트 경로가 겹칩니다.")
        for name, model_type, size, ratio in EXPERIMENTS:
            if (product_name, name) in completed:
                print(f"RESUME {product_name} / {name}", flush=True)
                continue
            print(f"\nEXPERIMENT {product_name} / {name}", flush=True)
            metrics = evaluate_experiment(groups, model_type, size, ratio, device)
            row = {"product": product_name, "experiment": name, **metrics}
            results.append(row)
            if result_path is not None:
                temporary = result_path.with_suffix(".tmp")
                pd.DataFrame(results).to_json(temporary, orient="records", indent=2, double_precision=12)
                temporary.replace(result_path)
            print(row, flush=True)
    if not results:
        raise ValueError("평가할 상품이 없습니다.")
    results = pd.DataFrame(results)
    baseline = results[results["experiment"] == "baseline"].set_index("product")
    for metric in ("auroc", "precision", "recall", "f1"):
        results[f"delta_{metric}"] = results[metric] - results["product"].map(baseline[metric])
    return results


def summarize_models(results):
    metrics = ["auroc", "precision", "recall", "f1", "false_positive_rate"]
    summary = results.groupby("experiment")[metrics].mean()
    summary["delta_recall"] = summary["recall"] - summary.loc["baseline", "recall"]
    return summary.sort_values("recall", ascending=False).round(4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--result-path", type=Path, default=None)
    args = parser.parse_args()
    torch.set_num_threads(4)
    data_dir = Path(__file__).resolve().parent.parent / "mvtec_anomaly_detection"
    output_dir = data_dir.parent / "outputs"
    suffix = args.product or "all"
    result_path = args.result_path or output_dir / f"model_comparison_{suffix}.json"
    if args.summary_only:
        results = pd.read_json(result_path)
    else:
        results = compare_models(data_dir, args.product, result_path, args.resume)
    columns = ["product", "experiment", "auroc", "precision", "recall", "f1", "missed_defects", "delta_recall"]
    final_results = results[columns + ["false_positive_rate", "threshold"]].round(4)
    summary = summarize_models(results)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        if not args.summary_only:
            print("\n상품별 최종 평가 DataFrame")
            print(final_results)
        print("\n모형별 상품 평균 지표 (각 상품에 동일 가중치)")
        print(summary)
    if not args.summary_only:
        results.to_json(result_path, orient="records", indent=2, double_precision=12)
    print(f"COMPLETE {len(results)} experiments", flush=True)
