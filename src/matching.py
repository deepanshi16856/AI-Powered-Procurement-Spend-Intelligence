
import pandas as pd
import re


def normalize_text(value):
    """Normalize text for reliable matching."""
    if pd.isna(value):
        return ""

    value = str(value).strip().upper()
    value = re.sub(r"[^A-Z0-9]", "", value)

    return value


def mark_company_a_duplicates(
    erp_df,
    invoice_df,
    erp_config,
    invoice_config,
    amount_tolerance=1.0
):
    """
    Match invoice-tool records against ERP records.
    Each ERP record can match at most one invoice record.
    """

    erp = erp_df.copy()
    invoices = invoice_df.copy()

    # 1. Normalize ERP purchase order references.
    erp_po = erp_config.get("po_reference_column")

    if erp_po and erp_po in erp.columns:
        erp["_match_po"] = erp[erp_po].apply(normalize_text)
    elif "po_number" in erp.columns:
        erp["_match_po"] = erp["po_number"].apply(normalize_text)
    else:
        erp["_match_po"] = ""

    # Append PO item to ERP PO number.
    if "po_item" in erp.columns:
        erp["_match_po"] = (
            erp["_match_po"]
            + erp["po_item"].apply(normalize_text)
        )

    # 2. Normalize invoice-tool PO references.
    invoice_po = invoice_config.get("po_reference_column")

    if invoice_po and invoice_po in invoices.columns:
        invoices["_match_po"] = (
            invoices[invoice_po].apply(normalize_text)
        )
    else:
        invoices["_match_po"] = ""

    # 3. Normalize supplier names.
    erp_supplier = erp_config.get("supplier_name_column")
    invoice_supplier = invoice_config.get("supplier_name_column")

    if erp_supplier and erp_supplier in erp.columns:
        erp["_match_supplier"] = (
            erp[erp_supplier].apply(normalize_text)
        )
    else:
        erp["_match_supplier"] = ""

    if invoice_supplier and invoice_supplier in invoices.columns:
        invoices["_match_supplier"] = (
            invoices[invoice_supplier].apply(normalize_text)
        )
    else:
        invoices["_match_supplier"] = ""

    # 4. Read cleaned net amounts.
    erp_amount = pd.to_numeric(
        erp["net_amount"], errors="coerce"
    )

    invoice_amount = pd.to_numeric(
        invoices["net_amount"], errors="coerce"
    )

    # 5. Build ERP lookup.
    erp_lookup = {}

    for index, row in erp.iterrows():
        po = row["_match_po"]
        supplier = row["_match_supplier"]

        if not po or not supplier:
            continue

        key = (po, supplier)

        erp_lookup.setdefault(key, []).append(
            (index, erp_amount.loc[index])
        )

    # 6. Match invoice records against ERP records.
    invoices["is_duplicate"] = False
    invoices["matched_erp_id"] = ""

    matched_erp_indices = set()

    for index, row in invoices.iterrows():
        po = row["_match_po"]
        supplier = row["_match_supplier"]

        if not po or not supplier:
            continue

        key = (po, supplier)
        candidates = erp_lookup.get(key, [])

        for erp_index, amount in candidates:

            # Do not reuse an ERP record.
            if erp_index in matched_erp_indices:
                continue

            invoice_value = invoice_amount.loc[index]

            if pd.isna(amount) or pd.isna(invoice_value):
                continue

            if abs(amount - invoice_value) <= amount_tolerance:

                invoices.loc[index, "is_duplicate"] = True

                # Save matched ERP ID for audit.
                invoices.loc[index, "matched_erp_id"] = str(
                    erp.loc[erp_index, erp_config["id_column"]]
                )

                matched_erp_indices.add(erp_index)
                break

    # 7. Mark duplicates without overwriting existing exclusions.
    if "include" not in invoices.columns:
        invoices["include"] = True

    if "exclusion_reason" not in invoices.columns:
        invoices["exclusion_reason"] = ""

    duplicate_mask = (
        invoices["is_duplicate"] & invoices["include"]
    )

    invoices.loc[duplicate_mask, "include"] = False
    invoices.loc[duplicate_mask, "exclusion_reason"] = "DUPLICATE"

    return erp, invoices
