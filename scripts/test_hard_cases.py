"""
Test the deployed mosquito classifier API against known-hard cases:
An. arabiensis <-> An. coluzzii, the confusion pair your offline
confusion matrix showed. Run this from Cloud Shell or any machine
with your labels.csv + test images available.

Usage:
    python test_hard_cases.py --root /path/to/mosquito_cdc --api-url https://mosquito-classifier-207189004007.us-central1.run.app
"""

import argparse
import os
import pandas as pd
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Path to dataset root (contains labels.csv)")
    parser.add_argument("--api-url", required=True, help="Base URL of deployed API, no trailing slash")
    parser.add_argument("--n-per-class", type=int, default=5, help="How many images per hard class to test")
    args = parser.parse_args()

    labels_csv = os.path.join(args.root, "labels.csv")
    df = pd.read_csv(labels_csv)

    # The known confusion pair from your confusion matrix
    hard_species = ["anopheles_arabiensis", "anopheles_coluzzi"]

    test_df = df[(df["split"] == "test") & (df["species"].isin(hard_species))]

    print(f"Testing {len(test_df)} hard-case images against {args.api_url}\n")
    print(f"{'filename':<45} {'true_species':<25} {'pred_species':<25} {'conf':<8} {'true_sex':<8} {'pred_sex':<8} {'match'}")
    print("-" * 130)

    correct = 0
    total = 0

    for species in hard_species:
        subset = test_df[test_df["species"] == species].head(args.n_per_class)
        for _, row in subset.iterrows():
            filepath = os.path.join(args.root, row["filepath"])
            if not os.path.exists(filepath):
                print(f"MISSING: {filepath}")
                continue

            with open(filepath, "rb") as f:
                resp = requests.post(f"{args.api_url}/predict", files={"file": f})

            if resp.status_code != 200:
                print(f"ERROR on {row['filename']}: {resp.status_code} {resp.text}")
                continue

            result = resp.json()
            pred_species = result["species"]
            pred_sex = result["sex"]
            conf = result["species_confidence"]

            species_match = (pred_species == row["species"])
            sex_match = (pred_sex == row["sex"])
            match_str = "OK" if species_match else "WRONG"

            print(f"{row['filename']:<45} {row['species']:<25} {pred_species:<25} "
                  f"{conf:<8.3f} {row['sex']:<8} {pred_sex:<8} {match_str}")

            total += 1
            if species_match:
                correct += 1

    print("-" * 130)
    print(f"\nHard-case species accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    print("Compare this to your overall test accuracy (89.4%) — a lower number here")
    print("confirms the confusion is concentrated in this species pair, as your")
    print("confusion matrix showed, rather than being spread evenly across classes.")


if __name__ == "__main__":
    main()
