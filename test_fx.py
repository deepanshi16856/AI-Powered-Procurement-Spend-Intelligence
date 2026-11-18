
from src.ingestion import load_config, load_csv
from src.cleaning import clean_dataset
from src.fx_converter import convert_to_usd


if __name__ == "__main__":
    config = load_config("configs/company_b.yaml")

    source = config["sources"][0]

    invoices = load_csv(source["file"])
    fx_rates = load_csv(config["fx_file"])

    cleaned = clean_dataset(invoices, source)

    converted = convert_to_usd(
        cleaned,
        fx_rates,
        source,
        config
    )

    print("\nCOMPANY B CURRENCY CONVERSION")
    print("=" * 45)

    print("Original rows:", len(invoices))
    print("Converted rows:", len(converted))
    print("Missing FX rates:", converted["fx_missing"].sum())

    print("\nCurrency conversion samples:")

    print(
        converted[
            [
                "currency_normalized",
                "original_amount",
                "units_per_usd",
                "spend_usd"
            ]
        ].head(10).to_string(index=False)
    )
