
import pandas as pd
from sklearn.model_selection import train_test_split

from src.ingestion import load_config, load_csv
from src.classification import classify_records


def test_company_b_classification():

    config = load_config("configs/company_b.yaml")

    taxonomy = load_csv(config["taxonomy_file"])
    labels = load_csv(config["labels_file"])

    source = config["sources"][0]
    df = load_csv(source["file"])

    print("\n" + "=" * 55)
    print("COMPANY B CLASSIFICATION EVALUATION")
    print("=" * 55)

    print("Invoice records:", len(df))
    print("Taxonomy records:", len(taxonomy))
    print("Total labelled records:", len(labels))

    # Company B uses line_id instead of record_id.
    # Add record_id so the existing classifier can use it.
    labels["record_id"] = labels["line_id"].astype(str).str.strip()

    # Split labels into training and hidden test records.
        # Keep only valid procurement labels.
    labels = labels[
        labels["include"].astype(str).str.strip().eq("1")
    ].copy()

    print("\nValid procurement labels:", len(labels))

    # Split valid procurement labels into training and hidden test records.
    train_labels, test_labels = train_test_split(
        labels,
        test_size=0.10,
        random_state=42
    )

    print("\nTraining labels:", len(train_labels))
    print("Testing labels:", len(test_labels))
    # Classify invoice records using training labels.
    result = classify_records(
        df,
        source,
        taxonomy,
        labels=train_labels,
        confidence_threshold=0.15,
        batch_size=128
    )

    # Prepare predictions.
    id_column = source["id_column"]

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

    # Prepare actual hidden labels.
    actual = test_labels[
        ["line_id", "category_l1", "category_l2", "category_l3"]
    ].copy()

    actual["_match_id"] = (
        actual["line_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Compare predictions against hidden labels.
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
    print("Successfully classified:", classified.sum())
    print("Unclassified:", (~classified).sum())

    coverage = classified.mean() * 100 if len(evaluation) else 0
    print(f"Classification coverage: {coverage:.2f}%")

    for level in ["l1", "l2", "l3"]:

        predicted = evaluation[f"predicted_{level}"]
        expected = evaluation[f"category_{level}"]

        correct = predicted.fillna("").astype(str).str.strip().eq(
            expected.fillna("").astype(str).str.strip()
        )

        accuracy_classified = (
            correct[classified].mean() * 100
            if classified.sum() else 0
        )

        accuracy_overall = (
            correct.mean() * 100
            if len(evaluation) else 0
        )

        print(f"\nCategory {level.upper()} accuracy:")
        print(f"  Among classified records: {accuracy_classified:.2f}%")
        print(f"  Overall accuracy: {accuracy_overall:.2f}%")

    print("\nClassification methods on hidden records:")
    print(evaluation["classification_method"].value_counts())

    print("\nAverage confidence on hidden records:")
    print(round(evaluation["confidence"].mean(), 4))

    # Save evaluation results.
    output_path = "outputs/company_b_classification_evaluation.csv"
    evaluation.to_csv(output_path, index=False)

    print("\n" + "=" * 55)
    print("EVALUATION COMPLETED")
    print("=" * 55)
    print(f"Evaluation results saved to: {output_path}")


if __name__ == "__main__":
    test_company_b_classification()
