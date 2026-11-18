
import pandas as pd


def convert_to_usd(df, fx_rates, source_config, config):
    """
    Convert cleaned transaction amounts into USD.

    FX convention:
        units_per_usd = local currency units per 1 USD

    Formula:
        spend_usd = net_amount / units_per_usd

    Preserves original records and existing columns.
    """

    converted = df.copy()
    rates = fx_rates.copy()

    # --------------------------------------------------
    # 1. Read configuration
    # --------------------------------------------------

    amount_column = "net_amount"
    original_amount_column = "original_amount"
    currency_column = source_config.get("currency_column")
    date_column = "parsed_date"

    fx_month_column = config.get("fx_month_column", "month")
    fx_rate_column = config.get("fx_rate_column", "units_per_usd")
    base_currency = config.get("base_currency", "USD").upper()

    # --------------------------------------------------
    # 2. Validate required columns
    # --------------------------------------------------

    required_invoice_columns = [
        amount_column,
        original_amount_column,
        currency_column,
        date_column
    ]

    for column in required_invoice_columns:
        if column not in converted.columns:
            raise ValueError(
                f"Required cleaned column missing: {column}. "
                "Run clean_dataset() first."
            )

    required_fx_columns = [
        "currency",
        fx_month_column,
        fx_rate_column
    ]

    for column in required_fx_columns:
        if column not in rates.columns:
            raise ValueError(
                f"Required FX column missing: {column}"
            )

    # --------------------------------------------------
    # 3. Normalize currency codes
    # --------------------------------------------------

    converted["currency_normalized"] = (
        converted[currency_column]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    rates["currency_normalized"] = (
        rates["currency"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------
    # 4. Prepare transaction months
    # --------------------------------------------------

    converted[date_column] = pd.to_datetime(
        converted[date_column],
        errors="coerce"
    )

    converted["fx_month"] = (
        converted[date_column].dt.to_period("M")
    )

    # --------------------------------------------------
    # 5. Prepare monthly FX rates
    # --------------------------------------------------

    rates["fx_month"] = (
        pd.to_datetime(
            rates[fx_month_column],
            errors="coerce"
        ).dt.to_period("M")
    )

    rates["units_per_usd"] = pd.to_numeric(
        rates[fx_rate_column],
        errors="coerce"
    )

    # Reject invalid FX data rather than silently dropping it.
    if rates["fx_month"].isna().any():
        raise ValueError("Invalid month found in FX dataset.")

    if rates["currency_normalized"].isna().any():
        raise ValueError("Missing currency found in FX dataset.")

    if rates["units_per_usd"].isna().any():
        raise ValueError("Invalid or missing FX rate found.")

    if (rates["units_per_usd"] <= 0).any():
        raise ValueError("FX rates must be greater than zero.")

    # --------------------------------------------------
    # 6. Check duplicate FX records
    # --------------------------------------------------

    duplicate_rates = rates.duplicated(
        subset=["currency_normalized", "fx_month"],
        keep=False
    )

    if duplicate_rates.any():

        duplicate_data = rates.loc[
            duplicate_rates,
            ["currency_normalized", "fx_month", "units_per_usd"]
        ]

        conflicts = duplicate_data.groupby(
            ["currency_normalized", "fx_month"]
        )["units_per_usd"].nunique()

        if (conflicts > 1).any():
            raise ValueError(
                "Conflicting FX rates found for a currency/month."
            )

        rates = rates.drop_duplicates(
            subset=["currency_normalized", "fx_month"]
        )

    # --------------------------------------------------
    # 7. Match invoices with monthly FX rates
    # --------------------------------------------------

    lookup = rates.set_index(
        ["currency_normalized", "fx_month"]
    )["units_per_usd"]

    keys = pd.MultiIndex.from_arrays([
        converted["currency_normalized"],
        converted["fx_month"]
    ])

    converted["units_per_usd"] = lookup.reindex(
        keys
    ).to_numpy()

    # Base currency always has an exchange rate of 1.
    is_base_currency = (
        converted["currency_normalized"] == base_currency
    )

    converted.loc[
        is_base_currency, "units_per_usd"
    ] = 1.0

    # --------------------------------------------------
    # 8. Calculate converted amounts
    # --------------------------------------------------

    converted["net_amount"] = pd.to_numeric(
        converted["net_amount"],
        errors="coerce"
    )

    converted["original_amount"] = pd.to_numeric(
        converted["original_amount"],
        errors="coerce"
    )

    converted["spend_usd"] = (
        converted["net_amount"] /
        converted["units_per_usd"]
    )

    # --------------------------------------------------
    # 9. Track conversion status
    # --------------------------------------------------

    converted["fx_status"] = "CONVERTED"

    converted.loc[
        converted["original_amount"].isna(),
        "fx_status"
    ] = "MISSING_AMOUNT"

    converted.loc[
        converted["currency_normalized"].isna(),
        "fx_status"
    ] = "MISSING_CURRENCY"

    converted.loc[
        converted[date_column].isna(),
        "fx_status"
    ] = "MISSING_POSTING_DATE"

    converted.loc[
        converted["units_per_usd"].isna()
        & converted["currency_normalized"].notna()
        & converted[date_column].notna(),
        "fx_status"
    ] = "MISSING_FX_RATE"

    converted["fx_rate_source"] = "MONTHLY_FX"

    converted.loc[
        is_base_currency,
        "fx_rate_source"
    ] = "BASE_CURRENCY_FIXED"

    converted["fx_missing"] = (
        converted["fx_status"] == "MISSING_FX_RATE"
    )

    # Invalid transactions must not produce converted amounts.
    invalid = converted["fx_status"] != "CONVERTED"

    converted.loc[invalid, "spend_usd"] = float("nan")

    converted["spend_usd"] = converted[
        "spend_usd"
    ].round(2)

    # --------------------------------------------------
    # 10. Return converted dataset
    # --------------------------------------------------

    return converted
