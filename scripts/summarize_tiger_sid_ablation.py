"""Print a compact Markdown table from the three SID ablation result files."""
import argparse
import json
from pathlib import Path


def read_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle).get("metrics", {})


def value(metrics: dict, key: str) -> str:
    number = metrics.get(key)
    return "-" if number is None else f"{number:.4f}" if isinstance(number, float) else str(number)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="beauty")
    parser.add_argument("--out-root", default="out/tiger/amazon")
    args = parser.parse_args()
    base = Path(args.out_root) / args.split
    methods = [("RQ-VAE", "rqvae"), ("RQ-KMeans", "rqkmeans"), ("RQ-OPQ (M=2)", "rqopq")]
    print("| Method | Collision | Recon MSE | Recall@10 | NDCG@10 |")
    print("|---|---:|---:|---:|---:|")
    for label, directory in methods:
        sid = read_metrics(base / directory / "results.json")
        tiger = read_metrics(base / directory / "tiger" / "results.json")
        print("| {label} | {collision} | {mse} | {recall} | {ndcg} |".format(
            label=label,
            collision=value(sid, "collision_rate"),
            mse=value(sid, "reconstruction_mse"),
            recall=value(tiger, "best_test_Recall@10"),
            ndcg=value(tiger, "best_test_NDCG@10"),
        ))


if __name__ == "__main__":
    main()
