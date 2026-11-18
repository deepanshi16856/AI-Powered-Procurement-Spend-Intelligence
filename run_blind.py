import argparse
import pandas as pd

from src.ingestion import load_config, load_csv, PROJECT_ROOT
from src.cleaning import clean_dataset
from src.exclusions import apply_exclusion_rules
from src.entity_resolution import create_supplier_keys, create_item_keys
from src.classification import classify_records, classify_cross_source
from src.matching import mark_company_a_duplicates
from src.fx_converter import convert_to_usd


def prepare_source(source, config):
    df = load_csv(source["file"])
    df = clean_dataset(df, source)
    df = apply_exclusion_rules(df, source, config)
    df = create_supplier_keys(df, source)
    df = create_item_keys(df, source)
    return df


def export_source(df, source, spend_column, add_prefix=True):
    ids = df[source["id_column"]].fillna("").astype(str).str.strip()
    prefix = source.get("id_prefix", "")
    if prefix and add_prefix:
        ids = prefix + ":" + ids

    return pd.DataFrame({
        "record_id": ids,
        "include": df["include"].fillna(False).astype(int),
        "exclusion_reason": df["exclusion_reason"].fillna(""),
        "supplier_key": df["supplier_key"],
        "item_key": df["item_key"],
        "spend": pd.to_numeric(df[spend_column], errors="coerce"),
        "l1": df["predicted_l1"],
        "l2": df["predicted_l2"],
        "l3": df["predicted_l3"],
        "l4": "",
        "confidence": df["confidence"]
    })


def run_company_a(config, taxonomy, labels):
    sources = {s["name"]: s for s in config["sources"]}
    erp_config = sources["erp"]
    invoice_config = sources["invoice_tool"]

    print("\nPreparing ERP records...")
    erp = prepare_source(erp_config, config)

    print("\nPreparing invoice-tool records...")
    invoice = prepare_source(invoice_config, config)

    print("\nMatching duplicate invoices...")
    erp, invoice = mark_company_a_duplicates(
        erp, invoice, erp_config, invoice_config
    )

    print("\nClassifying ERP using labels...")
    erp = classify_records(erp, erp_config, taxonomy, labels)

    print("\nClassifying invoice-tool using ERP training examples...")
    invoice = classify_cross_source(
        erp, invoice, erp_config, invoice_config,
        labels, taxonomy
    )

    return [
        export_source(erp, erp_config, "net_amount"),
        export_source(invoice, invoice_config, "net_amount")
    ]


def run_generic(config, taxonomy, labels, fx_rates):
    outputs = []

    for source in config["sources"]:
        print(f"\nProcessing source: {source['name']}")
        df = prepare_source(source, config)

        if fx_rates is not None:
            df = convert_to_usd(df, fx_rates, source, config)
            spend_column = "spend_usd"
        else:
            spend_column = "net_amount"

        df = classify_records(df, source, taxonomy, labels)
        outputs.append(export_source(df, source, spend_column, add_prefix=False))

    return outputs


def run(config_path, output_path):
    config = load_config(config_path)
    taxonomy = load_csv(config["taxonomy_file"])
    labels = load_csv(config["labels_file"]) if config.get("labels_file") else None
    fx_rates = load_csv(config["fx_file"]) if config.get("fx_file") else None

    if config["company"].lower() == "company_a":
        outputs = run_company_a(config, taxonomy, labels)
    else:
        outputs = run_generic(config, taxonomy, labels, fx_rates)

    final = pd.concat(outputs, ignore_index=True)

    if final["record_id"].eq("").any() or final["record_id"].duplicated().any():
        raise ValueError("Missing or duplicate record IDs detected.")

    output = PROJECT_ROOT / output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output, index=False)

    classified = final["l1"].fillna("").astype(str).str.strip().ne("")

    print("\nPipeline completed.")
    print("Records:", len(final))
    print("Included:", int(final["include"].sum()))
    print("Excluded:", int((final["include"] == 0).sum()))
    print("Classified:", int(classified.sum()))
    print("Unclassified:", int((~classified).sum()))
    print("Saved:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, args.output)


