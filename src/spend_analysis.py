import pandas as pd
from pathlib import Path


def prepare_spend_data(df):
    """Prepare included procurement transactions for spend analysis."""

    data = df.copy()

    data = data[data["include"] == True].copy()

    data["spend_usd"] = pd.to_numeric(
        data["spend_usd"], errors="coerce"
    )

    data["parsed_date"] = pd.to_datetime(
        data["parsed_date"], errors="coerce"
    )

    # Keep all included records, including unclassified records.
    for column in ["predicted_l1", "predicted_l2", "predicted_l3"]:
        data[column] = data[column].fillna("UNCLASSIFIED")

    data["vendor_id"] = data["vendor_id"].fillna("UNKNOWN_VENDOR")

    data["month"] = data["parsed_date"].dt.to_period("M").astype(str)

    return data


def aggregate_spend(df, group_columns):
    """Aggregate converted spend by selected dimensions."""

    valid = df[df["spend_usd"].notna()].copy()

    result = (
        valid.groupby(group_columns, dropna=False)
        .agg(
            total_spend_usd=("spend_usd", "sum"),
            transaction_count=("line_id", "nunique"),
            positive_spend_usd=(
                "spend_usd",
                lambda values: values[values > 0].sum()
            ),
            negative_spend_usd=(
                "spend_usd",
                lambda values: values[values < 0].sum()
            )
        )
        .reset_index()
    )

    result["total_spend_usd"] = result["total_spend_usd"].round(2)
    result["positive_spend_usd"] = result["positive_spend_usd"].round(2)
    result["negative_spend_usd"] = result["negative_spend_usd"].round(2)

    return result.sort_values(
        "total_spend_usd",
        ascending=False
    )


def generate_company_b_reports(
    input_path="outputs/company_b_procurement_master.csv",
    output_dir="outputs/company_b_analytics"
):
    """Generate procurement spend reports for Company B."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, low_memory=False)

    data = prepare_spend_data(df)

    reports = {
        "supplier_spend.csv": ["vendor_id"],
        "category_l1_spend.csv": ["predicted_l1"],
        "category_l2_spend.csv": ["predicted_l1", "predicted_l2"],
        "category_l3_spend.csv": [
            "predicted_l1", "predicted_l2", "predicted_l3"
        ],
        "monthly_spend.csv": ["month"],
        "region_spend.csv": ["region"]
    }

    print("=" * 60)
    print("COMPANY B - PROCUREMENT SPEND ANALYTICS")
    print("=" * 60)

    for filename, dimensions in reports.items():

        report = aggregate_spend(data, dimensions)

        report.to_csv(output / filename, index=False)

        print("\nGenerated:", filename)
        print("Groups:", len(report))

    converted = data[data["spend_usd"].notna()]

    print("\n" + "=" * 60)
    print("SPEND ANALYTICS VALIDATION")
    print("=" * 60)

    print("Included procurement records:", len(data))
    print("Converted records:", len(converted))
    print("Missing spend records:", data["spend_usd"].isna().sum())

    print(
        "Total procurement spend (USD):",
        round(converted["spend_usd"].sum(), 2)
    )

    print(
        "Unclassified spend (USD):",
        round(
            converted.loc[
                converted["predicted_l3"] == "UNCLASSIFIED",
                "spend_usd"
            ].sum(),
            2
        )
    )

    print("\nReports saved in:", output)


from pathlib import Path


def generate_company_a_reports(
    input_path="outputs/company_a_procurement_master.csv",
    output_dir="outputs/company_a_analytics"
):
    """Generate procurement spend reports for Company A in ZAR."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, low_memory=False)

    # Keep included procurement records
    data = df[df["include"].astype(str).str.lower() == "true"].copy()

    # Standardize spend and date fields
    data["spend_zar"] = pd.to_numeric(
        data["spend_zar"], errors="coerce"
    )

    data["transaction_date"] = pd.to_datetime(
        data["transaction_date"], errors="coerce"
    )

    # Preserve unclassified records in spend reporting
    for column in ["category_l1", "category_l2", "category_l3"]:
        data[column] = data[column].fillna("").replace("", "UNCLASSIFIED")

    data["supplier_name"] = data["supplier_name"].fillna("UNKNOWN_SUPPLIER")
    data["site_code"] = data["site_code"].fillna("UNKNOWN_SITE")
    data["site_name"] = data["site_name"].fillna("UNKNOWN_SITE")

    data["month"] = data["transaction_date"].dt.to_period("M").astype(str)

    def aggregate_company_a(group_columns):
        valid = data[data["spend_zar"].notna()].copy()

        report = (
            valid.groupby(group_columns, dropna=False)
            .agg(
                total_spend_zar=("spend_zar", "sum"),
                transaction_count=("record_id", "nunique"),
                positive_spend_zar=(
                    "spend_zar",
                    lambda values: values[values > 0].sum()
                ),
                negative_spend_zar=(
                    "spend_zar",
                    lambda values: values[values < 0].sum()
                )
            )
            .reset_index()
        )

        # Preserve full precision for accurate reconciliation.
# Round only when displaying totals, not during aggregation.

        return report.sort_values(
            "total_spend_zar",
            ascending=False
        )

    reports = {
        "supplier_spend.csv": ["supplier_name"],
        "category_l1_spend.csv": ["category_l1"],
        "category_l2_spend.csv": ["category_l1", "category_l2"],
        "category_l3_spend.csv": [
            "category_l1", "category_l2", "category_l3"
        ],
        "monthly_spend.csv": ["month"],
        "region_spend.csv": ["site_code", "site_name"]
    }

    print("=" * 60)
    print("COMPANY A - PROCUREMENT SPEND ANALYTICS")
    print("=" * 60)

    for filename, dimensions in reports.items():
        report = aggregate_company_a(dimensions)
        report.to_csv(output / filename, index=False)

        print("\nGenerated:", filename)
        print("Groups:", len(report))

    valid = data[data["spend_zar"].notna()]

    print("\n" + "=" * 60)
    print("COMPANY A SPEND ANALYTICS VALIDATION")
    print("=" * 60)

    print("Included procurement records:", len(data))
    print("Records with spend:", len(valid))
    print("Missing spend records:", data["spend_zar"].isna().sum())

    print(
        "Total procurement spend (ZAR):",
        round(valid["spend_zar"].sum(), 2)
    )

    print(
        "Unclassified spend (ZAR):",
        round(
            valid.loc[
                valid["category_l3"] == "UNCLASSIFIED",
                "spend_zar"
            ].sum(),
            2
        )
    )

    print("\nReports saved in:", output)
