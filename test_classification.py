
import pandas as pd
import re

from sklearn.model_selection import train_test_split

from src.ingestion import load_config, load_csv
from src.classification import classify_records


def test_company_a_classification():

    config = load_config("configs/company_a.yaml")

    taxonomy = load_csv(config["taxonomy_file"])
    labels = load_csv(config["labels_file"])

    source = config["sources"][0]
    df = load_csv(source["file"])

    print("\n" + "=" * 55)
    print("COMPANY A CLASSIFICATION EVALUATION")
    print("=" * 55)

    print("ERP records:", len(df))
    print("Taxonomy records:", len(taxonomy))
    print("Total labelled records:", len(labels))

    # --------------------------------------------------
    # 1. Split labels into training and testing sets
    # --------------------------------------------------

    train_labels, test_labels = train_test_split(
        labels,
        test_size=0.10,
        random_state=42
    )

    print("\nTraining labels:", len(train_labels))
    print("Testing labels:", len(test_labels))

    # --------------------------------------------------
    # 2. Classify using training labels only
    # --------------------------------------------------

    result = classify_records(
        df,
        source,
        taxonomy,
        labels=train_labels,
        confidence_threshold=0.15,
        batch_size=128
    )

    # --------------------------------------------------
    # 3. Prepare predictions for evaluation
    # --------------------------------------------------

    id_column = source["id_column"]
    prefix = source.get("id_prefix", "")

    predictions = result[
        [
            id_column,
            "predicted_l1",
            "predicted_l2",
            "predicted_l3",
            "confidence",
            "classification_method"
        ]
    ].copy()

    predictions["_match_id"] = (
        predictions[id_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    actual = test_labels[
        ["record_id", "category_l1", "category_l2", "category_l3"]
    ].copy()

    actual["_match_id"] = (
        actual["record_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if prefix:
        actual["_match_id"] = actual["_match_id"].str.replace(
            rf"^{re.escape(str(prefix))}:",
            "",
            regex=True,
            case=False
        )

    # Only compare predictions against hidden labels
    evaluation = predictions.merge(
        actual[
            [
                "_match_id",
                "category_l1",
                "category_l2",
                "category_l3"
            ]
        ],
        on="_match_id",
        how="inner"
    )

    # --------------------------------------------------
    # 4. Calculate classification metrics
    # --------------------------------------------------

    classified = (
        evaluation["predicted_l3"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    print("\n" + "=" * 55)
    print("EVALUATION RESULTS")
    print("=" * 55)

    print("Hidden test records:", len(evaluation))
    print("Successfully classified:", int(classified.sum()))
    print("Unclassified:", int((~classified).sum()))

    coverage = classified.mean() * 100

    print(f"Classification coverage: {coverage:.2f}%")

    for level in ["l1", "l2", "l3"]:

        predicted_col = f"predicted_{level}"
        actual_col = f"category_{level}"

        correct = (
            evaluation[predicted_col]
            .fillna("")
            .astype(str)
            .str.strip()
            ==
            evaluation[actual_col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        accuracy_classified = (
            correct[classified].mean() * 100
            if classified.any()
            else 0
        )

        accuracy_overall = correct.mean() * 100

        print(f"\nCategory {level.upper()} accuracy:")
        print(
            f"  Among classified records: "
            f"{accuracy_classified:.2f}%"
        )
        print(
            f"  Overall accuracy: "
            f"{accuracy_overall:.2f}%"
        )

    # --------------------------------------------------
    # 5. Classification method breakdown
    # --------------------------------------------------

    print("\nClassification methods on hidden records:")

    print(
        evaluation["classification_method"]
        .value_counts()
        .to_string()
    )

    print("\nAverage confidence on hidden records:")

    print(
        round(
            evaluation.loc[classified, "confidence"].mean(),
            4
        ) if classified.any() else 0
    )

    print("\n" + "=" * 55)
    print("EVALUATION COMPLETED")
    print("=" * 55)
    output_path = "outputs/company_a_classification_evaluation.csv"
    evaluation.to_csv(output_path, index=False)
    print(f"\nEvaluation results saved to: {output_path}")


if __name__ == "__main__":
    test_company_a_classification()
