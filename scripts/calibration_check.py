"""
Calibration check for the deployed mosquito classifier.

Runs the full test split through the live API, bins predictions by confidence,
and checks whether confidence actually tracks accuracy — e.g. do the model's
"90% confident" predictions get it right ~90% of the time?

This matters more than raw accuracy for a real field tool: if confidence is
well-calibrated, a low-confidence prediction is a meaningful signal that a
human should double-check the specimen. If confidence is poorly calibrated,
it can't be trusted for that purpose even if raw accuracy looks fine.

Usage:
    python calibration_check.py --root /path/to/mosquito_cdc --api-url https://mosquito-classifier-207189004007.us-central1.run.app
"""

import argparse
import os
import pandas as pd
import requests
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--api-url", required=True)
    args = parser.parse_args()

    labels_csv = os.path.join(args.root, "labels.csv")
    df = pd.read_csv(labels_csv)
    test_df = df[df["split"] == "test"]

    print(f"Running {len(test_df)} test images through {args.api_url} ...\n")

    records = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        filepath = os.path.join(args.root, row["filepath"])
        if not os.path.exists(filepath):
            continue

        with open(filepath, "rb") as f:
            resp = requests.post(f"{args.api_url}/predict", files={"file": f})

        if resp.status_code != 200:
            print(f"ERROR on {row['filename']}: {resp.status_code}")
            continue

        result = resp.json()
        records.append({
            "filename": row["filename"],
            "true_species": row["species"],
            "pred_species": result["species"],
            "confidence": result["species_confidence"],
            "correct": result["species"] == row["species"],
        })

        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(test_df)} done")

    results_df = pd.DataFrame(records)

    # ---------- Bin by confidence, compare to actual accuracy per bin ----------
    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    labels = ["<0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    results_df["conf_bin"] = pd.cut(results_df["confidence"], bins=bins, labels=labels, right=False)

    print("\n" + "=" * 60)
    print("CALIBRATION TABLE")
    print("=" * 60)
    print(f"{'Confidence bin':<15} {'N':<6} {'Actual accuracy':<18} {'Avg confidence'}")
    print("-" * 60)

    for b in labels:
        subset = results_df[results_df["conf_bin"] == b]
        if len(subset) == 0:
            continue
        actual_acc = subset["correct"].mean()
        avg_conf = subset["confidence"].mean()
        print(f"{b:<15} {len(subset):<6} {actual_acc:<18.3f} {avg_conf:.3f}")

    overall_acc = results_df["correct"].mean()
    print("-" * 60)
    print(f"Overall test accuracy: {overall_acc:.3f}")

    # ---------- Simple calibration verdict ----------
    print("\nHow to read this:")
    print("- If 'Actual accuracy' roughly matches 'Avg confidence' within each bin,")
    print("  the model is well-calibrated: its confidence can be trusted as a signal.")
    print("- If actual accuracy is much LOWER than avg confidence in a bin, the model")
    print("  is overconfident there — risky for a 'defer to human' threshold.")
    print("- If actual accuracy is much HIGHER than avg confidence, the model is")
    print("  underconfident — safe, but you're deferring to humans more than needed.")

    results_df.to_csv("calibration_results.csv", index=False)
    print("\nFull results saved to calibration_results.csv")


if __name__ == "__main__":
    main()
