import pytest


from src.ingestion import load_config, load_csv
from src.cleaning import clean_dataset
from src.exclusions import apply_exclusion_rules


@pytest.mark.parametrize("config_path", ["configs/company_a.yaml", "configs/company_b.yaml"])
def test_company(config_path):
    config = load_config(config_path)

    print("\n" + "=" * 55)
    print("TESTING:", config["company"].upper())

    for source in config["sources"]:
        df = load_csv(source["file"])
        cleaned = clean_dataset(df, source)

        result = apply_exclusion_rules(cleaned, source, config)

        print("\nSource:", source["name"])
        print("Original rows:", len(df))
        print("Output rows:", len(result))
        print("Included:", result["include"].sum())
        print("Excluded:", (~result["include"]).sum())

        print("\nExclusion reasons:")
        print(
            result["exclusion_reason"]
            .replace("", "INCLUDED")
            .value_counts()
        )


def validate_company_b_exclusions(config_path):
    config = load_config(config_path)

    source = config["sources"][0]
    df = load_csv(source["file"])
    cleaned = clean_dataset(df, source)
    result = apply_exclusion_rules(cleaned, source, config)

    labels = load_csv(config["labels_file"])

    # Match predictions with manually labelled records.
    comparison = result[["line_id", "include"]].merge(
        labels[["line_id", "include"]],
        on="line_id",
        suffixes=("_predicted", "_actual"),
        how="inner"
    )

    predicted_excluded = ~comparison["include_predicted"].astype(bool)
    actual_excluded = ~comparison["include_actual"].astype(bool)

    true_positive = (predicted_excluded & actual_excluded).sum()
    false_positive = (predicted_excluded & ~actual_excluded).sum()
    false_negative = (~predicted_excluded & actual_excluded).sum()
    true_negative = (~predicted_excluded & ~actual_excluded).sum()

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive else 0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative else 0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    print("\n" + "=" * 55)
    print("COMPANY B EXCLUSION VALIDATION")
    print("=" * 55)
    print("Matched labelled records:", len(comparison))
    print("True positives:", true_positive)
    print("False positives:", false_positive)
    print("False negatives:", false_negative)
    print("True negatives:", true_negative)
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1 score:  {f1:.2%}")

    print("\nFalse positive examples (genuine spend excluded):")
    print(
        comparison.loc[
            predicted_excluded & ~actual_excluded,
            ["line_id", "include_predicted", "include_actual"]
        ].head(10).to_string(index=False)
    )

    print("\nFalse negative examples (non-procurement not excluded):")
    print(
        comparison.loc[
            ~predicted_excluded & actual_excluded,
            ["line_id", "include_predicted", "include_actual"]
        ].head(10).to_string(index=False)
    )

if __name__ == "__main__":
    test_company("configs/company_a.yaml")
    test_company("configs/company_b.yaml")
    validate_company_b_exclusions("configs/company_b.yaml")
