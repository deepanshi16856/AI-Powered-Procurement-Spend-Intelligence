
import pandas as pd


def parse_vat_rate(value, default_rate=0.15):
    """Convert VAT values such as 0.15, 15, and 15% into decimal rates."""

    if pd.isna(value) or str(value).strip() == "":
        return default_rate

    value = str(value).strip().replace("%", "")

    try:
        rate = float(value)
        if rate > 1:
            rate = rate / 100
        return rate
    except ValueError:
        return None


def clean_dataset(df, source_config):
    """
    Clean a dataset without deleting or overwriting original records.
    Adds standardized fields while preserving the source columns.
    """

    cleaned = df.copy()

    # Standardize text values without changing the original columns.
    for column in cleaned.select_dtypes(include=["object"]).columns:
        cleaned[f"{column}_clean"] = (
            cleaned[column]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )

    # Parse dates.
    date_column = source_config.get("date_column")

    if date_column and date_column in cleaned.columns:
        cleaned["parsed_date"] = pd.to_datetime(
            cleaned[date_column],
            errors="coerce",
            dayfirst=False
        )
        cleaned["date_parse_failed"] = (
            cleaned[date_column].notna()
            & cleaned["parsed_date"].isna()
        )
    else:
        cleaned["parsed_date"] = pd.NaT
        cleaned["date_parse_failed"] = False

    # Parse monetary amounts, including values with comma separators.
    amount_column = source_config.get("amount_column")

    if amount_column and amount_column in cleaned.columns:
        amount_values = (
            cleaned[amount_column]
            .astype("string")
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        cleaned["original_amount"] = pd.to_numeric(
            amount_values,
            errors="coerce"
        )

        cleaned["amount_parse_failed"] = (
            cleaned[amount_column].notna()
            & cleaned["original_amount"].isna()
        )
    else:
        cleaned["original_amount"] = pd.NA
        cleaned["amount_parse_failed"] = True

    # Convert gross invoice amounts into net amounts.
    amount_type = source_config.get("amount_type", "net")

    if amount_type == "gross":
        vat_column = source_config.get("vat_rate_column")
        default_vat = source_config.get("default_vat_rate", 0.15)

        if vat_column and vat_column in cleaned.columns:
            vat_rates = cleaned[vat_column].apply(
                lambda value: parse_vat_rate(value, default_vat)
            )
        else:
            vat_rates = pd.Series(
                default_vat,
                index=cleaned.index
            )

        cleaned["vat_rate_normalized"] = pd.to_numeric(
            vat_rates,
            errors="coerce"
        )

        cleaned["net_amount"] = (
            cleaned["original_amount"]
            / (1 + cleaned["vat_rate_normalized"])
        )

        cleaned["vat_parse_failed"] = (
            cleaned["vat_rate_normalized"].isna()
        )

    else:
        cleaned["net_amount"] = cleaned["original_amount"]
        cleaned["vat_rate_normalized"] = 0.0
        cleaned["vat_parse_failed"] = False

    # Preserve rows with missing descriptions; flag them for later processing.
    description_columns = [
        source_config.get("item_description_column"),
        source_config.get("secondary_description_column")
    ]

    available_descriptions = [
        column for column in description_columns
        if column and column in cleaned.columns
    ]

    if available_descriptions:
        cleaned["description_missing"] = (
            cleaned[available_descriptions]
            .isna()
            .all(axis=1)
        )
    else:
        cleaned["description_missing"] = True

    return cleaned
