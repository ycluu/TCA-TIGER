"""Summarize the five-token RQ-VAE/RQ-KMeans/RQ-OPQ comparison."""
import argparse
import json
from pathlib import Path


def metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        return json.load(file).get("metrics", {})


def number(values: dict, key: str) -> str:
    value = values.get(key)
    return "-" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="beauty")
    parser.add_argument("--out-root", default="out/tiger/amazon")
    args = parser.parse_args()
    root = Path(args.out_root) / args.split
    methods = [("RQ-VAE (5 RQ)", "rqvae_5token"),
               ("RQ-KMeans (5 RQ)", "rqkmeans_5token"),
               ("RQ-OPQ (3 RQ + 2 OPQ)", "rqopq")]
    print("| Method | SID length | Collision | Recon MSE | Recall@10 | NDCG@10 |")
    print("|---|---:|---:|---:|---:|---:|")
    for label, directory in methods:
        sid, tiger = metrics(root / directory / "results.json"), metrics(root / directory / "tiger" / "results.json")
        print("| {label} | 5 | {collision} | {mse} | {recall} | {ndcg} |".format(
            label=label, collision=number(sid, "collision_rate"),
            mse=number(sid, "reconstruction_mse"),
            recall=number(tiger, "best_test_Recall@10"),
            ndcg=number(tiger, "best_test_NDCG@10"),
        ))


if __name__ == "__main__":
    main()
