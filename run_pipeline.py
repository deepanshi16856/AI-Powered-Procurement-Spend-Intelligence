from pathlib import Path
import hashlib
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submission"
OUT.mkdir(exist_ok=True)

COLUMNS = [
    "record_id", "include", "exclusion_reason", "supplier_key",
    "item_key", "spend", "l1", "l2", "l3", "l4", "confidence"
]

def clean_key(value):
    if pd.isna(value) or str(value).strip().lower() in {
        "", "nan", "none", "null", "n/a", "missing"
    }:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())

def make_key(value, prefix):
    value = clean_key(value)
    if not value:
        return ""
    return hashlib.sha256(f"{prefix}:{value}".encode()).hexdigest()[:16]

def save_submission(df, filename):
    result = pd.DataFrame({
        "record_id": df["record_id"],
        "include": df["include"].fillna(False).astype(bool).astype(int),
        "exclusion_reason": df.get("exclusion_reason", pd.Series("", index=df.index)).fillna(""),
        "supplier_key": df["supplier_key"],
        "item_key": df["item_key"],
        "spend": pd.to_numeric(df["spend"], errors="coerce"),
        "l1": df["l1"].fillna(""),
        "l2": df["l2"].fillna(""),
        "l3": df["l3"].fillna(""),
        "l4": "",
        "confidence": pd.to_numeric(df["confidence"], errors="coerce")
    })[COLUMNS]

    result.to_csv(OUT / filename, index=False)
    print(f"Saved {filename}: {len(result):,} rows")

def company_a():
    master = pd.read_csv(ROOT / "outputs/company_a_procurement_master.csv")
    erp = pd.read_csv(ROOT / "data/company_a/erp_goods_receipts.csv", low_memory=False)
    invoices = pd.read_csv(ROOT / "data/company_a/invoice_tool_export.csv", low_memory=False)

    item_lookup = {}

    for _, row in erp.iterrows():
        rid = "ERP:" + str(row.get("line_id", "")).strip()
        item_lookup[rid] = (
            row.get("material_number"),
            row.get("material_short_text") or row.get("service_short_text")
        )

    for _, row in invoices.iterrows():
        rid = "TOOL:" + str(row.get("tool_record_id", "")).strip()
        item_lookup[rid] = ("", row.get("line_text"))

    supplier_values = master.get("supplier_id", pd.Series("", index=master.index))
    supplier_names = master.get("supplier_name", pd.Series("", index=master.index))
    supplier_keys, item_keys = [], []

    for _, row in master.iterrows():
        supplier = row.get("supplier_id")
        if pd.isna(supplier) or not str(supplier).strip():
            supplier = row.get("supplier_name")
        supplier_keys.append(make_key(supplier, "supplier"))

        material, description = item_lookup.get(str(row["record_id"]), ("", ""))
        item_value = material if clean_key(material) else description
        item_keys.append(make_key(item_value, "item"))

    master["supplier_key"] = supplier_keys
    master["item_key"] = item_keys

    export = pd.DataFrame({
        "record_id": master["record_id"],
        "include": master["include"],
        "exclusion_reason": "",
        "supplier_key": master["supplier_key"],
        "item_key": master["item_key"],
        "spend": master["spend_zar"],
        "l1": master["category_l1"],
        "l2": master["category_l2"],
        "l3": master["category_l3"],
        "confidence": master["classification_confidence"]
    })
    save_submission(export, "Deepanshi__A2.csv")

def company_b():
    master = pd.read_csv(ROOT / "outputs/company_b_procurement_master.csv", low_memory=False)

    supplier_keys, item_keys = [], []
    for _, row in master.iterrows():
        supplier = row.get("vendor_id")
        if pd.isna(supplier) or not str(supplier).strip():
            supplier = row.get("supplier_name")
        supplier_keys.append(make_key(supplier, "supplier"))

        item = row.get("material_number")
        if pd.isna(item) or not str(item).strip():
            item = row.get("line_description")
        item_keys.append(make_key(item, "item"))

    export = pd.DataFrame({
        "record_id": master["line_id"],
        "include": master["include"],
        "exclusion_reason": master["exclusion_reason"],
        "supplier_key": supplier_keys,
        "item_key": item_keys,
        "spend": master["spend_usd"],
        "l1": master["predicted_l1"],
        "l2": master["predicted_l2"],
        "l3": master["predicted_l3"],
        "confidence": master["confidence"]
    })
    save_submission(export, "Deepanshi__B2.csv")

if __name__ == "__main__":
    company_a()
    company_b()
    print("A/B submission exports completed.")
