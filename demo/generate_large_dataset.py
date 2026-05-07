#!/usr/bin/env python3
"""Generate a large-scale fraud detection dataset for AnamDB benchmarking.

Produces a CSV with configurable row counts (default 100K) that mirrors the
demo transactions.csv schema but with realistic distributions.

Usage:
    python3 demo/generate_large_dataset.py [num_rows] [output_path]
"""

import csv
import random
import sys
import os

REGIONS = ["US", "EU", "APAC", "LATAM", "MEA"]
MERCHANTS = ["grocery", "electronics", "luxury", "wire_transfer", "travel", "dining", "fuel"]

# Distribution: 80% legit (low fraud_prob), 15% suspicious, 5% high-risk
def generate_row(i: int) -> dict:
    roll = random.random()

    if roll < 0.80:
        # Legit transaction
        amount = round(random.uniform(5.0, 500.0), 2)
        fraud_prob = round(random.uniform(0.01, 0.15), 4)
        region = random.choice(["US", "US", "US", "EU", "EU"])
        merchant = random.choice(["grocery", "dining", "fuel", "grocery"])
        is_international = random.random() < 0.05
        hour = random.randint(8, 22)
        age = random.randint(22, 65)
    elif roll < 0.95:
        # Suspicious
        amount = round(random.uniform(1000.0, 15000.0), 2)
        fraud_prob = round(random.uniform(0.40, 0.85), 4)
        region = random.choice(["EU", "APAC", "LATAM", "MEA"])
        merchant = random.choice(["electronics", "luxury", "travel"])
        is_international = random.random() < 0.50
        hour = random.randint(0, 23)
        age = random.randint(25, 70)
    else:
        # High-risk
        amount = round(random.uniform(10000.0, 100000.0), 2)
        fraud_prob = round(random.uniform(0.88, 0.9999), 4)
        region = random.choice(["APAC", "LATAM", "MEA"])
        merchant = random.choice(["wire_transfer", "luxury"])
        is_international = True
        hour = random.randint(0, 5)
        age = random.randint(40, 80)

    return {
        "transaction_id": f"TXN-{i:07d}",
        "amount": amount,
        "fraud_prob": fraud_prob,
        "region": region,
        "merchant_type": merchant,
        "customer_age": age,
        "is_international": str(is_international).lower(),
        "hour_of_day": hour,
    }


def main():
    num_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    output_path = sys.argv[2] if len(sys.argv) > 2 else "demo/data/transactions_large.csv"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    random.seed(42)  # Deterministic for reproducibility

    fields = [
        "transaction_id", "amount", "fraud_prob", "region",
        "merchant_type", "customer_age", "is_international", "hour_of_day"
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(1, num_rows + 1):
            writer.writerow(generate_row(i))
            if i % 25000 == 0:
                print(f"  {i:>7,} / {num_rows:,} rows...", flush=True)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Generated {num_rows:,} rows → {output_path} ({size_mb:.1f} MB)")

    # Quick stats
    high_risk = sum(1 for _ in range(num_rows) if random.random() > 0.95)
    print(f"  ~{num_rows * 5 // 100:,} high-risk transactions (5%)")
    print(f"  ~{num_rows * 15 // 100:,} suspicious transactions (15%)")
    print(f"  ~{num_rows * 80 // 100:,} legitimate transactions (80%)")


if __name__ == "__main__":
    main()
