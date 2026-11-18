
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str) -> dict:
    """Load a company's YAML configuration."""
    path = PROJECT_ROOT / config_path

    if not path.exists():
        raise FileNotFoundError(f"Configuration not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")

    return config


def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file safely."""
    path = PROJECT_ROOT / file_path

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path, low_memory=False)


def ingest_company(config_path: str) -> dict:
    """Load all configured datasets for one company."""
    config = load_config(config_path)

    company_name = config["company"]
    datasets = {}

    print(f"\n{'=' * 55}")
    print(f"INGESTING: {company_name.upper()}")
    print(f"{'=' * 55}")

    for source in config["sources"]:
        source_name = source["name"]
        file_path = source["file"]

        print(f"\nLoading: {source_name}")

        df = load_csv(file_path)

        datasets[source_name] = df

        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")
        print(f"Loaded successfully.")

    # Load taxonomy
    taxonomy = load_csv(config["taxonomy_file"])
    datasets["taxonomy"] = taxonomy

    print(f"\nTaxonomy categories: {len(taxonomy)}")

    # Load labels if configured
    if config.get("labels_file"):
        labels = load_csv(config["labels_file"])
        datasets["labels"] = labels
        print(f"Label records: {len(labels):,}")

    # Load FX rates for multi-currency companies
    if config.get("fx_file"):
        fx_rates = load_csv(config["fx_file"])
        datasets["fx_rates"] = fx_rates
        print(f"FX rate records: {len(fx_rates):,}")

    print(f"\n{company_name.upper()} INGESTION COMPLETED")

    return datasets
