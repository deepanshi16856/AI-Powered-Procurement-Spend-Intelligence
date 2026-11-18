
from src.ingestion import load_config, load_csv
from src.cleaning import clean_dataset
from src.fx_converter import convert_to_usd


def test_company_b_fx():

    print("\n" + "=" * 60)
    print("COMPANY B FX CONVERSION TEST")
    print("=" * 60)

    config = load_config("configs/company_b.yaml")

    source_config = config["sources"][0]

    invoices = load_csv(source_config["file"])
    fx_rates = load_csv(config["fx_file"])

    print("\nOriginal records:", len(invoices))

    # Clean invoice data
    cleaned = clean_dataset(invoices, source_config)

    # Convert amounts into USD
    converted = convert_to_usd(
        cleaned,
        fx_rates,
        source_config,
        config
    )

    print("\nConversion results:")
    print(converted["fx_status"].value_counts())

    print("\nFX rate sources:")
    print(converted["fx_rate_source"].value_counts())

    print("\nCurrency-wise results:")
    print(
        converted.groupby(
            ["currency_normalized", "fx_status"]
        ).size().to_string()
    )

    print("\nFirst 10 converted transactions:")

    print(
        converted[
            [
                "currency",
                "original_amount",
                "net_amount",
                "units_per_usd",
                "spend_usd",
                "fx_status"
            ]
        ].head(10).to_string(index=False)
    )

    # Validation
    assert len(converted) == len(invoices)

    assert converted["original_amount"].equals(
        cleaned["original_amount"]
    )

    usd_rows = converted["currency_normalized"] == "USD"

    assert (
        converted.loc[usd_rows, "units_per_usd"] == 1.0
    ).all()

    print("\nRecord count preserved:", len(converted))
    print("FX conversion test completed.")

    # Save output
    converted.to_csv(
        "outputs/company_b_fx_converted.csv",
        index=False
    )

    print("\nOutput saved successfully.")


if __name__ == "__main__":
    test_company_b_fx()
