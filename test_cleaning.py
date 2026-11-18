import pytest


from src.ingestion import load_config, load_csv
from src.cleaning import clean_dataset


@pytest.mark.parametrize("company_config_path", ["configs/company_a.yaml", "configs/company_b.yaml"])
def test_company(company_config_path):
    config = load_config(company_config_path)

    print(f"\nTesting: {config['company'].upper()}")

    for source in config["sources"]:
        df = load_csv(source["file"])

        cleaned = clean_dataset(df, source)

        print(f"\nSource: {source['name']}")
        print("Original rows:", len(df))
        print("Cleaned rows:", len(cleaned))
        print("Failed amount parsing:", cleaned["amount_parse_failed"].sum())
        print("Failed date parsing:", cleaned["date_parse_failed"].sum())
        print("Missing descriptions:", cleaned["description_missing"].sum())

        if source.get("amount_type") == "gross":
            print("Sample gross amounts:")
            print(cleaned[["original_amount", "vat_rate_normalized", "net_amount"]].head())


if __name__ == "__main__":
    test_company("configs/company_a.yaml")
    test_company("configs/company_b.yaml")
