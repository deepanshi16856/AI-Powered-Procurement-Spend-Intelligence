
from src.ingestion import ingest_company


if __name__ == "__main__":
    company_a = ingest_company("configs/company_a.yaml")
    company_b = ingest_company("configs/company_b.yaml")

    print("\nBOTH COMPANIES LOADED SUCCESSFULLY!")
