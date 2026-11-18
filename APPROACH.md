# Approach — AI-Powered Procurement Spend Intelligence Engine

## 1. Objective

The objective is to transform raw procurement and purchasing data into clean, standardized, classified spend data that can be used for procurement analytics and reporting.

The solution is designed to support different company data formats and category taxonomies while retaining records that cannot be confidently classified.

## 2. Processing Workflow

The solution follows these major stages:

1. **Data ingestion**
   - Load procurement records, category taxonomies, labelled examples and supporting foreign-exchange data.
   - Support company-specific input configurations.

2. **Data cleaning and normalization**
   - Standardize relevant text and field values.
   - Normalize supplier, item and transaction information.
   - Handle missing and inconsistent values.

3. **Spend eligibility and exclusions**
   - Identify records that should not be included in procurement spend.
   - Preserve exclusion decisions for traceability.

4. **Supplier and item matching**
   - Identify potential duplicate suppliers and items.
   - Use entity-resolution and matching logic to improve consistency.

5. **Currency conversion**
   - Use the supplied exchange-rate data where conversion is required.
   - Normalize spend values into the configured reporting currency.

6. **Category classification**
   - Use available labelled examples and matching logic to assign procurement records to the relevant category hierarchy.
   - Use similarity-based classification where direct matching is unavailable.
   - Retain uncertain or unmatched records as unclassified instead of forcing an unsupported category.

7. **Spend analytics**
   - Aggregate spend by supplier, category, month and region.
   - Generate classification and data-quality summaries.

8. **Reporting**
   - Produce structured CSV outputs.
   - Present results through the Streamlit dashboard.

## 3. Company Data Handling

Company-specific configurations are maintained separately from the processing modules.

Company A and Company B have different source formats, labels, taxonomies and supporting data. Their configurations and processing paths are kept separate where required.

The solution should use the category taxonomy supplied for each company rather than assume that category names or hierarchy levels are identical across companies.

## 4. Classification and Evaluation

Classification performance is evaluated using labelled records held out from the training examples.

The Company B evaluation includes:
- Level 1 classification accuracy
- Level 2 classification accuracy
- Level 3 classification accuracy

Unclassified records are retained for review. Classification coverage and accuracy are treated as separate measures.

## 5. Outputs and Dashboard

The project generates processed procurement master data and spend-analysis outputs for Company A and Company B.

The Streamlit dashboard contains:
- Executive Overview
- Category Analytics
- Supplier Analytics
- Trends & Locations
- Data Quality

## 6. Reproducibility and Confidentiality

- Company-specific settings are maintained in configuration files.
- Original challenge datasets are kept locally and excluded from Git using `.gitignore`.
- Original challenge data must not be committed to the public repository or included in Git history.
- Output files intended for submission are maintained separately from original input data.

## 7. AI Tool Disclosure

ChatGPT was used as a development assistant for problem-solving, code review, debugging and documentation support. The implementation details and runtime dependencies should be verified against the submitted source code and `requirements.txt`.

## 8. Execution

The project includes Python processing modules, company configurations, tests, generated outputs and a Streamlit dashboard.

The final end-to-end execution command and its behaviour must be verified before the build is frozen for the blind test.