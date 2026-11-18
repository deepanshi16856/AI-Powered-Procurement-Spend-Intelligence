
from src.ingestion import load_config, load_csv
from src.cleaning import clean_dataset
from src.matching import mark_company_a_duplicates


def test_matching():
    config = load_config("configs/company_a.yaml")

    erp_config = config["sources"][0]
    invoice_config = config["sources"][1]

    # Load and clean ERP data.
    erp_df = load_csv(erp_config["file"])
    erp_cleaned = clean_dataset(erp_df, erp_config)

    # Load and clean invoice-tool data.
    invoice_df = load_csv(invoice_config["file"])
    invoice_cleaned = clean_dataset(invoice_df, invoice_config)

    # Match records.
    erp_result, invoice_result = mark_company_a_duplicates(
        erp_cleaned,
        invoice_cleaned,
        erp_config,
        invoice_config
    )

    duplicate_count = invoice_result["is_duplicate"].sum()

    print("\n" + "=" * 55)
    print("COMPANY A DUPLICATE MATCHING RESULTS")
    print("=" * 55)

    print("ERP records:", len(erp_result))
    print("Invoice records:", len(invoice_result))
    print("Matched duplicates:", duplicate_count)
    print(
        "Unmatched invoice records:",
        len(invoice_result) - duplicate_count
    )

    print("\nSample matched records:")

    print(
    invoice_result.loc[
        invoice_result["is_duplicate"],
        [
            "tool_record_id",
            "matched_erp_id",
            "po_reference",
            "supplier_name_as_captured",
            "net_amount"
        ]
    ].head(10).to_string(index=False)
)
 # Compare matched invoice records with their ERP records.
    erp_id_column = erp_config["id_column"]
    erp_supplier_column = erp_config["supplier_name_column"]

    matched_pairs = invoice_result[
        invoice_result["is_duplicate"]
    ].merge(
        erp_result[
            [
                erp_id_column,
                "po_number",
                "po_item",
                erp_supplier_column,
                "net_amount"
            ]
        ],
        left_on="matched_erp_id",
        right_on=erp_id_column,
        suffixes=("_invoice", "_erp")
    )

    print("\nMatched pair verification:")

    print(
        matched_pairs[
            [
                "tool_record_id",
                "matched_erp_id",
                "po_reference",
                "po_number",
                "po_item",
                "supplier_name_as_captured",
                erp_supplier_column,
                "net_amount_invoice",
                "net_amount_erp"
            ]
        ].head(10).to_string(index=False)
    )


    print("\nMATCH QUALITY CHECK")

    matched_pairs["amount_difference"] = (
        matched_pairs["net_amount_invoice"]
        - matched_pairs["net_amount_erp"]
    ).abs()

    print(
        "Matches with amount difference <= 1:",
        (matched_pairs["amount_difference"] <= 1.0).sum()
    )

    print(
        "Maximum amount difference:",
        matched_pairs["amount_difference"].max()
    )

    print(
        "Unique ERP records matched:",
        matched_pairs[erp_id_column].nunique()
    )

    print(
        "Unique invoice records matched:",
        matched_pairs["tool_record_id"].nunique()
    )

if __name__ == "__main__":
    test_matching()
