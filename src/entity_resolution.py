
import pandas as pd
import re
import hashlib


def normalize_value(value):
    """Clean and standardize text values."""
    if pd.isna(value):
        return ""

    value = str(value).strip().upper()

    if value in ["", "NAN", "NONE", "NULL", "N/A", "MISSING"]:
        return ""

    value = re.sub(r"[^A-Z0-9]", "", value)

    return value


def generate_key(prefix, value):
    """
    Generate a stable key.
    Hashing keeps the key compact and consistent.
    """
    if not value:
        return ""

    hashed = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]

    return f"{prefix}_{hashed}"


def create_supplier_keys(df, source_config):
    """Create standardized supplier keys."""

    result = df.copy()

    id_column = source_config.get("supplier_id_column")
    name_column = source_config.get("supplier_name_column")

    supplier_keys = []

    for _, row in result.iterrows():

        supplier_id = (
            normalize_value(row[id_column])
            if id_column and id_column in result.columns
            else ""
        )

        supplier_name = (
            normalize_value(row[name_column])
            if name_column and name_column in result.columns
            else ""
        )

        if supplier_id:
            key_value = f"ID_{supplier_id}"
        elif supplier_name:
            key_value = f"NAME_{supplier_name}"
        else:
            key_value = ""

        supplier_keys.append(
            generate_key("SUP", key_value)
        )

    result["supplier_key"] = supplier_keys

    return result


def create_item_keys(df, source_config):
    """Create standardized item keys."""

    result = df.copy()

    id_column = source_config.get("item_id_column")
    description_column = source_config.get("item_description_column")
    secondary_column = source_config.get("secondary_description_column")

    item_keys = []

    for _, row in result.iterrows():

        item_id = (
            normalize_value(row[id_column])
            if id_column and id_column in result.columns
            else ""
        )

        description = (
            normalize_value(row[description_column])
            if description_column and description_column in result.columns
            else ""
        )

        secondary_description = (
            normalize_value(row[secondary_column])
            if secondary_column and secondary_column in result.columns
            else ""
        )

        if item_id:
            key_value = f"ID_{item_id}"
        elif description:
            key_value = f"DESC_{description}"
        elif secondary_description:
            key_value = f"DESC_{secondary_description}"
        else:
            key_value = ""

        item_keys.append(
            generate_key("ITEM", key_value)
        )

    result["item_key"] = item_keys

    return result
