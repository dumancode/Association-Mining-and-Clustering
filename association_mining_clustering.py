from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN


RANDOM_SEED = 42
FIGURE_DIR = Path("figures")
OUTPUT_DIR = Path("outputs")


PAGE_AREAS = {
    "hero_product": (500, 250),
    "headline": (260, 170),
    "navigation": (620, 70),
    "cta_button": (370, 390),
    "price_info": (760, 390),
    "footer": (520, 690),
}


def generate_scanpath_transactions(n_participants: int = 120) -> list[list[str]]:
    """Generate reproducible gaze-area sequences resembling website scanpaths."""
    rng = np.random.default_rng(RANDOM_SEED)
    transactions = []
    area_names = list(PAGE_AREAS)

    for _ in range(n_participants):
        intent = rng.choice(["product_focused", "price_focused", "navigation_focused"], p=[0.48, 0.34, 0.18])

        if intent == "product_focused":
            sequence = ["hero_product", "headline"]
            if rng.random() < 0.78:
                sequence.append("cta_button")
            if rng.random() < 0.35:
                sequence.append("price_info")
            if rng.random() < 0.52:
                sequence.append("navigation")
        elif intent == "price_focused":
            sequence = ["price_info", "cta_button"]
            if rng.random() < 0.72:
                sequence.append("hero_product")
            if rng.random() < 0.42:
                sequence.append("headline")
            if rng.random() < 0.28:
                sequence.append("footer")
        else:
            sequence = ["navigation", "footer"]
            if rng.random() < 0.48:
                sequence.append("headline")
            if rng.random() < 0.32:
                sequence.append("hero_product")
            if rng.random() < 0.22:
                sequence.append("cta_button")

        if rng.random() < 0.25:
            sequence.append(rng.choice(area_names))
        transactions.append(list(dict.fromkeys(sequence)))

    return transactions


def generate_fixation_points(transactions: list[list[str]]) -> pd.DataFrame:
    """Create fixation coordinates around page areas for clustering diagnostics."""
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for participant_id, sequence in enumerate(transactions, start=1):
        timestamp = 0
        for area in sequence:
            center_x, center_y = PAGE_AREAS[area]
            n_fixations = rng.integers(3, 8)

            for fixation_index in range(n_fixations):
                rows.append(
                    {
                        "participant": f"P{participant_id:03d}",
                        "area": area,
                        "fixation_index": fixation_index,
                        "timestamp": timestamp,
                        "fixation_duration": rng.gamma(shape=4.0, scale=55.0),
                        "x": rng.normal(center_x, 45),
                        "y": rng.normal(center_y, 35),
                    }
                )
                timestamp += rng.integers(110, 320)

    data = pd.DataFrame(rows)
    data["x"] = data["x"].clip(0, 1024)
    data["y"] = data["y"].clip(0, 768)
    return data


def support(itemset: tuple[str, ...], transactions: list[list[str]]) -> float:
    itemset_values = set(itemset)
    matches = sum(itemset_values.issubset(transaction) for transaction in map(set, transactions))
    return matches / len(transactions)


def mine_frequent_itemsets(
    transactions: list[list[str]], min_support: float = 0.35, max_size: int = 3
) -> pd.DataFrame:
    unique_items = sorted({item for transaction in transactions for item in transaction})
    itemsets = []

    for size in range(1, max_size + 1):
        for candidate in combinations(unique_items, size):
            candidate_support = support(candidate, transactions)
            if candidate_support >= min_support:
                itemsets.append({"itemset": candidate, "support": candidate_support})

    return pd.DataFrame(itemsets).sort_values(["support", "itemset"], ascending=[False, True])


def build_association_rules(
    frequent_itemsets: pd.DataFrame,
    transactions: list[list[str]],
    min_confidence: float = 0.65,
) -> pd.DataFrame:
    rules = []

    for itemset in frequent_itemsets["itemset"]:
        if len(itemset) < 2:
            continue

        itemset_support = support(itemset, transactions)
        itemset_values = set(itemset)

        for antecedent_size in range(1, len(itemset)):
            for antecedent in combinations(itemset, antecedent_size):
                consequent = tuple(sorted(itemset_values - set(antecedent)))
                antecedent_support = support(antecedent, transactions)
                consequent_support = support(consequent, transactions)
                confidence = itemset_support / antecedent_support
                lift = confidence / consequent_support

                if confidence >= min_confidence:
                    rules.append(
                        {
                            "antecedent": " + ".join(antecedent),
                            "consequent": " + ".join(consequent),
                            "support": itemset_support,
                            "confidence": confidence,
                            "lift": lift,
                        }
                    )

    return pd.DataFrame(rules).sort_values(
        ["lift", "confidence", "support"], ascending=[False, False, False]
    )


def cluster_fixations(fixations: pd.DataFrame) -> pd.DataFrame:
    model = DBSCAN(eps=30, min_samples=10)
    clustered = fixations.copy()
    clustered["cluster"] = model.fit_predict(clustered[["x", "y"]])
    return clustered


def plot_area_frequency(transactions: list[list[str]]) -> None:
    counts = Counter(item for transaction in transactions for item in transaction)
    frequency = pd.DataFrame(
        {"area": list(counts.keys()), "count": list(counts.values())}
    ).sort_values("count", ascending=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=frequency, x="count", y="area", color="#89a7ff")
    plt.title("Most Frequent Areas in Scanpath Transactions")
    plt.xlabel("Transaction count")
    plt.ylabel("Page area")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "scanpath_area_frequency.png", dpi=180)
    plt.close()


def plot_top_rules(rules: pd.DataFrame) -> None:
    top_rules = rules.head(8).copy()
    top_rules["rule"] = top_rules["antecedent"] + " -> " + top_rules["consequent"]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=top_rules,
        x="support",
        y="confidence",
        size="lift",
        hue="lift",
        palette="viridis",
        sizes=(80, 420),
    )

    for _, row in top_rules.iterrows():
        plt.text(row["support"] + 0.004, row["confidence"], row["rule"], fontsize=8)

    plt.title("Top Association Rules by Support, Confidence, and Lift")
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "association_rules_bubble.png", dpi=180)
    plt.close()


def plot_clustered_fixations(clustered: pd.DataFrame) -> None:
    plt.figure(figsize=(11, 7))
    ax = plt.gca()
    ax.set_facecolor("#f4f6fb")

    for area, (center_x, center_y) in PAGE_AREAS.items():
        rectangle = plt.Rectangle(
            (center_x - 85, center_y - 45),
            170,
            90,
            fill=False,
            edgecolor="#222222",
            linewidth=1.1,
            alpha=0.45,
        )
        ax.add_patch(rectangle)
        ax.text(center_x - 70, center_y - 52, area, fontsize=8, alpha=0.75)

    clean = clustered[clustered["cluster"] != -1]
    noise = clustered[clustered["cluster"] == -1]

    sns.scatterplot(
        data=clean,
        x="x",
        y="y",
        hue="cluster",
        palette="tab10",
        s=22,
        alpha=0.68,
        legend=False,
    )
    plt.scatter(noise["x"], noise["y"], s=12, color="#777777", alpha=0.25, label="noise")
    plt.gca().invert_yaxis()
    plt.xlim(0, 1024)
    plt.ylim(768, 0)
    plt.title("DBSCAN Clusters over Synthetic Eye-Tracking Fixations")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "dbscan_fixation_clusters.png", dpi=180)
    plt.close()


def main() -> None:
    FIGURE_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    transactions = generate_scanpath_transactions()
    fixations = generate_fixation_points(transactions)

    frequent_itemsets = mine_frequent_itemsets(transactions)
    rules = build_association_rules(frequent_itemsets, transactions)
    clustered_fixations = cluster_fixations(fixations)

    frequent_itemsets.to_csv(OUTPUT_DIR / "frequent_itemsets.csv", index=False)
    rules.to_csv(OUTPUT_DIR / "association_rules.csv", index=False)
    clustered_fixations.to_csv(OUTPUT_DIR / "clustered_fixations.csv", index=False)

    plot_area_frequency(transactions)
    plot_top_rules(rules)
    plot_clustered_fixations(clustered_fixations)

    n_clusters = clustered_fixations.loc[clustered_fixations["cluster"] != -1, "cluster"].nunique()
    noise_ratio = (clustered_fixations["cluster"] == -1).mean()

    print("Frequent itemsets")
    print(frequent_itemsets.head(10).to_string(index=False))
    print("\nTop association rules")
    print(rules.head(10).round(3).to_string(index=False))
    print(f"\nDBSCAN clusters: {n_clusters}")
    print(f"Noise ratio: {noise_ratio:.2%}")
    print("Saved outputs under outputs/ and figures/.")


if __name__ == "__main__":
    main()
