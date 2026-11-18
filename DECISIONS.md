# Design Decisions — AI-Powered Procurement Spend Intelligence Engine

## 1. Company-Specific Configurations

**Decision:** Maintain separate configuration files for Company A and Company B.

**Reason:** The companies have different input structures, taxonomies, labelled examples and supporting data. Keeping configuration separate reduces assumptions about shared schemas.

## 2. Category Taxonomy

**Decision:** Use the taxonomy supplied for the relevant company.

**Reason:** Procurement categories are company-specific. A category from one company's taxonomy should not automatically be assigned to another company's taxonomy.

## 3. Classification Strategy

**Decision:** Use direct label matching where reliable labelled records are available, and similarity-based classification when direct matching is unavailable.

**Reason:** This combines reliable existing labels with a fallback for records that do not have an exact match.

## 4. Uncertain Classification

**Decision:** Preserve records that cannot be confidently classified as unclassified.

**Reason:** Assigning an unsupported category could distort procurement reporting. Unclassified records remain visible for further review.

## 5. Spend Eligibility

**Decision:** Apply exclusion rules before calculating eligible procurement spend.

**Reason:** Records outside the intended procurement-spend scope should not inflate spend totals.

## 6. Duplicate Supplier and Item Handling

**Decision:** Use entity-resolution and matching logic to identify potential duplicate suppliers and items.

**Reason:** Supplier and item variations can fragment spend across multiple names and reduce the accuracy of procurement analysis.

## 7. Currency Normalization

**Decision:** Convert currencies using the supplied foreign-exchange rates and the configured reporting currency.

**Reason:** Spend across different currencies must be normalized before meaningful aggregation and comparison.

## 8. Data Quality and Traceability

**Decision:** Retain classification and data-quality information in the generated results wherever available.

**Reason:** Users should be able to distinguish classified, unclassified and excluded records and understand the quality of the resulting analysis.

## 9. Evaluation

**Decision:** Evaluate classification using held-out labelled examples and report performance at each applicable taxonomy level.

**Reason:** Evaluation on unseen labelled examples provides a more useful indication of classification performance than measuring only training records.

## 10. Confidentiality

**Decision:** Keep original challenge datasets outside the public Git repository and exclude the `data/` directory through `.gitignore`.

**Reason:** The challenge instructions require the original task data to remain confidential.

## 11. Development and AI Assistance

**Decision:** ChatGPT was used as a development assistant for debugging, problem-solving and documentation support.

**Reason:** AI assistance was used during development. The submitted implementation and its actual runtime dependencies remain the source of truth.

## 12. Limitations and Assumptions

- Classification quality depends on the quality and coverage of available labels and taxonomy descriptions.
- Similarity-based matches may require manual review.
- Supplier and item matching can produce uncertain matches.
- Currency conversion depends on the accuracy and coverage of supplied exchange rates.
- Company-specific rules should not be assumed to generalize automatically to an unseen company.
- Unclassified records should not be treated as zero-spend records.