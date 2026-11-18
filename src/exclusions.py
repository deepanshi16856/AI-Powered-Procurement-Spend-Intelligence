
import pandas as pd


def apply_exclusion_rules(df, source_config, company_config=None):
    """
    Apply configurable exclusion rules without deleting records.
    """

    result = df.copy()

    result["include"] = True
    result["exclusion_reason"] = ""

    def exclude(mask, reason):
        mask = mask.fillna(False) & result["include"]
        result.loc[mask, "include"] = False
        result.loc[mask, "exclusion_reason"] = reason

    # 1. Invoice approval status.
    status_column = source_config.get("status_column")

    if status_column and status_column in result.columns:
        status = (
            result[status_column]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        exclude(status.eq("CANCELLED"), "CANCELLED")

        exclude(
            status.isin(["REJECTED", "ON HOLD", "ON_HOLD"]),
            "NOT_APPROVED"
        )

    # 2. Configurable non-procurement rules.
    rules = (company_config or {}).get("non_procurement_rules", {})

    description_column = rules.get("description_column")

    if description_column in result.columns:
        description = (
            result[description_column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        keywords = rules.get("description_keywords", [])

        if keywords:
            pattern = "|".join(
                pd.Series(keywords).str.lower().str.replace(
                    r"([.^$*+?{}\[\]\\|()])", r"\\\1", regex=True
                )
            )

            exclude(
                description.str.contains(pattern, na=False, regex=True),
                "NON_PROCUREMENT"
            )

    gl_description_column = rules.get("gl_description_column")

    if gl_description_column in result.columns:
        gl_description = (
            result[gl_description_column]
            .astype("string")
            .str.strip()
            .str.casefold()
        )

        exact_values = [
            value.casefold()
            for value in rules.get("gl_description_exact", [])
        ]

        exclude(
            gl_description.isin(exact_values),
            "NON_PROCUREMENT"
        )

    gl_account_column = rules.get("gl_account_column")

    if gl_account_column in result.columns:
        gl_account = (
            result[gl_account_column]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        account_values = [
            value.upper()
            for value in rules.get("gl_account_exact", [])
        ]

        exclude(
            gl_account.isin(account_values),
            "NON_PROCUREMENT"
        )

    # 3. Additional explicit non-procurement column, if configured.
    non_procurement_column = source_config.get(
        "non_procurement_column"
    )

    if non_procurement_column in result.columns:
        values = (
            result[non_procurement_column]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        exclude(
            values.isin(["TAX", "REVENUE", "INTERCOMPANY", "ROYALTY"]),
            "NON_PROCUREMENT"
        )

    return result
