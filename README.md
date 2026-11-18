\# AI-Powered Procurement Spend Intelligence Engine



An end-to-end procurement analytics solution that ingests, cleans, classifies, reconciles, and analyses procurement transaction data from multiple enterprise sources.



\## Key Features



\- Multi-source procurement data ingestion

\- Data cleaning and standardisation

\- Duplicate detection and cross-source matching

\- Supplier entity resolution

\- Hierarchical procurement classification

\- TF-IDF and cosine similarity

\- Contextual classification using transaction metadata

\- Low-confidence classification review

\- Non-procurement transaction exclusion

\- Foreign exchange conversion

\- Procurement master dataset generation

\- Supplier, category, monthly, and regional spend analytics

\- Interactive Streamlit dashboard



\## Technology Stack



\- Python

\- Pandas and NumPy

\- Scikit-learn

\- PyYAML

\- Streamlit

\- Plotly

\- Pytest



\## Project Structure



```text

├── app.py

├── requirements.txt

├── README.md

├── configs/

├── data/

│   ├── company\_a/

│   └── company\_b/

├── src/

│   ├── ingestion.py

│   ├── cleaning.py

│   ├── matching.py

│   ├── entity\_resolution.py

│   ├── exclusions.py

│   ├── fx\_converter.py

│   ├── classification.py

│   └── spend\_analysis.py

├── outputs/

│   ├── company\_a\_analytics/

│   └── company\_b\_analytics/

└── test\_\*.py

```



\## Company A



\- Base currency: ZAR

\- Sources: ERP goods receipts and invoice-tool exports

\- VAT handling and duplicate matching

\- Hierarchical procurement classification



\## Company B



\- Base currency: USD

\- Source: Invoice-line transactions

\- Monthly foreign exchange conversion

\- Contextual classification and non-procurement exclusions



\## Dashboard



The Streamlit dashboard provides five sections:



1\. Executive Overview

2\. Category Analytics

3\. Supplier Analytics

4\. Trends \& Locations

5\. Data Quality



Users can filter data by company, transaction date, category, and top suppliers.



\## Installation



Clone the repository:



```bash

git clone YOUR\_GITHUB\_REPOSITORY\_URL

cd "AI-Powered Procurement Spend Intelligence Engine"

```



Create and activate a virtual environment:



```bash

python -m venv venv

```



Windows PowerShell:



```powershell

.\\venv\\Scripts\\Activate.ps1

```



Install dependencies:



```bash

python -m pip install -r requirements.txt

```



\## Run the Dashboard



```bash

python -m streamlit run app.py

```



Open http://localhost:8501 in your browser.



\## Run Tests



```bash

python -m pytest -v

```



Latest verified result: \*\*9 passed, 3 warnings\*\*.



\## Project Results



\### Company A



\- Procurement master records: 61,121

\- Classified records: 59,561

\- Unclassified records: 1,560

\- Classification coverage: 97.45%

\- Total spend: R4,227,798,606.01



\### Company B



\- Included procurement records: 38,577

\- Classified records: 37,479

\- Unclassified records: 1,098

\- Classification coverage: 97.15%

\- Total spend: $327,730,574.46



\## Testing and Validation



Automated tests cover classification, cleaning, entity resolution, exclusions, FX conversion, and duplicate matching.



Spend analytics reports have been reconciled against the procurement master datasets.



\## Future Improvements



\- Dashboard data export

\- Enhanced classification review

\- Additional supplier intelligence metrics

\- Cloud deployment

\- Expanded end-to-end testing



\## Disclaimer



Results are based on the datasets used in this project and should not be treated as audited financial statements.



