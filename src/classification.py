
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_text(value):
    """Normalize procurement descriptions."""

    if pd.isna(value):
        return ""

    value = str(value).upper()
    value = re.sub(r"[^A-Z0-9]+", " ", value)

    return " ".join(value.split())


def build_record_text(df, source_config):
    """Combine procurement descriptions and supporting context."""

    description_columns = [
        source_config.get("item_description_column"),
        source_config.get("secondary_description_column")
    ]

    context_columns = [
        "source_system",
        "document_type",
        "gl_account_description",
        "plant",
        "company_code",
        "cost_centre",
        "po_type"
    ]

    selected_columns = description_columns + context_columns

    available_columns = list(dict.fromkeys([
        col for col in selected_columns
        if col and col in df.columns
    ]))

    if not available_columns:
        return pd.Series("", index=df.index)

    text = df[available_columns].fillna("").astype(str)

    return text.agg(" ".join, axis=1).apply(normalize_text)


def classify_records(
    df,
    source_config,
    taxonomy,
    labels=None,
    confidence_threshold=0.15,
    batch_size=256
):
    """
    Classify procurement records using:
    1. Direct label matching by record ID.
    2. Batch-wise similarity against labelled examples.
    3. Taxonomy similarity as fallback.
    """

    result = df.copy()

    required = [
        "category_l1",
        "category_l2",
        "category_l3"
    ]

    # --------------------------------------------------
    # 1. Validate taxonomy
    # --------------------------------------------------

    for col in required:
        if col not in taxonomy.columns:
            raise ValueError(
                f"Missing taxonomy column: {col}"
            )

    taxonomy_clean = (
        taxonomy[required]
        .fillna("")
        .astype(str)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    record_texts = build_record_text(
        df,
        source_config
    )

    # Initialize prediction columns
    result["predicted_l1"] = ""
    result["predicted_l2"] = ""
    result["predicted_l3"] = ""
    result["confidence"] = 0.0
    result["classification_method"] = "UNCLASSIFIED"

    valid_records = record_texts.ne("")

    if not valid_records.any():
        return result

    # --------------------------------------------------
    # 2. Prepare labelled examples and match record IDs
    # --------------------------------------------------

    training_texts = []
    training_labels = []

    id_column = source_config.get("id_column")
    id_prefix = source_config.get("id_prefix", "")

    # Support both Company A and Company B label schemas
    label_id_column = None

    if labels is not None and not labels.empty:

        if "record_id" in labels.columns:
            label_id_column = "record_id"

        elif id_column and id_column in labels.columns:
            label_id_column = id_column

    if (
        labels is not None
        and not labels.empty
        and id_column
        and id_column in df.columns
        and label_id_column is not None
        and all(col in labels.columns for col in required)
    ):

        label_data = labels[
            [label_id_column] + required
        ].copy()

        # Standardize label ID column
        label_data = label_data.rename(
            columns={label_id_column: "record_id"}
        )

        # Normalize label IDs
        label_data["_match_id"] = (
            label_data["record_id"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        # Remove configured prefix, e.g. ERP:
        if id_prefix:
            label_data["_match_id"] = (
                label_data["_match_id"].str.replace(
                    rf"^{re.escape(str(id_prefix))}:",
                    "",
                    regex=True,
                    case=False
                )
            )

        label_data = label_data[
            label_data["_match_id"].ne("")
        ]

        label_data = label_data.drop_duplicates(
            subset="_match_id",
            keep="first"
        )

        label_data[required] = (
            label_data[required]
            .fillna("")
            .astype(str)
        )

        # Remove labels without a complete L3 category
        label_data = label_data[
            label_data["category_l3"].str.strip().ne("")
        ]

        label_lookup = label_data.set_index(
            "_match_id"
        )[required]

        # Normalize source IDs
        source_ids = (
            df[id_column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        # Directly assign existing labels
        matched_mask = source_ids.isin(
            label_lookup.index
        )

        print(
            "Matched labelled records:",
            int(matched_mask.sum())
        )

        for category, prediction in zip(
            required,
            [
                "predicted_l1",
                "predicted_l2",
                "predicted_l3"
            ]
        ):

            result.loc[matched_mask, prediction] = (
                source_ids[matched_mask]
                .map(label_lookup[category])
                .values
            )

        result.loc[
            matched_mask,
            "confidence"
        ] = 1.0

        result.loc[
            matched_mask,
            "classification_method"
        ] = "DIRECT_LABEL"

        # Prepare labelled examples for other records
        source_data = pd.DataFrame({
            "_match_id": source_ids.values,
            "text": record_texts.values
        })

        aligned = source_data.merge(
            label_data[
                ["_match_id"] + required
            ],
            on="_match_id",
            how="inner"
        )

        aligned = aligned[
            aligned["text"].ne("")
            & aligned["category_l3"].str.strip().ne("")
        ]

        training_texts = aligned["text"].tolist()

        training_labels = (
            aligned[required].values.tolist()
        )

    else:
        print(
            "Warning: Label ID matching unavailable. "
            "Using taxonomy fallback only."
        )

    # --------------------------------------------------
    # 3. Identify records still requiring classification
    # --------------------------------------------------

    remaining_mask = (
        valid_records
        & result["classification_method"].eq("UNCLASSIFIED")
    )

    remaining_indices = result.index[remaining_mask]

    if len(remaining_indices) == 0:
        print("All records already have matching labels.")
        return result

    remaining_texts = (
        record_texts.loc[remaining_indices].tolist()
    )

    print(
        "Records requiring similarity classification:",
        len(remaining_texts)
    )

    print(
        "Available labelled training examples:",
        len(training_texts)
    )

    # --------------------------------------------------
    # 4. Prepare taxonomy fallback examples
    # --------------------------------------------------

    taxonomy_texts = (
        taxonomy_clean["category_l1"] + " "
        + taxonomy_clean["category_l2"] + " "
        + taxonomy_clean["category_l3"]
    ).apply(normalize_text).tolist()

    valid_taxonomy = [
        i for i, text in enumerate(taxonomy_texts)
        if text.strip()
    ]

    taxonomy_texts = [
        taxonomy_texts[i]
        for i in valid_taxonomy
    ]

    taxonomy_reference = (
        taxonomy_clean.iloc[valid_taxonomy]
        .reset_index(drop=True)
    )

    reference_texts = (
        training_texts + taxonomy_texts
    )

    if not reference_texts:
        return result

    # --------------------------------------------------
    # 5. Fit TF-IDF vectorizer
    # --------------------------------------------------

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=50000,
        sublinear_tf=True
    )

    vectorizer.fit(
        reference_texts + remaining_texts
    )

    reference_matrix = vectorizer.transform(
        reference_texts
    )

    # --------------------------------------------------
    # 6. Classify in batches to control memory usage
    # --------------------------------------------------

    total_records = len(remaining_texts)

    for start in range(0, total_records, batch_size):

        end = min(
            start + batch_size,
            total_records
        )

        batch_texts = remaining_texts[start:end]

        batch_indices = remaining_indices[start:end]

        batch_matrix = vectorizer.transform(
            batch_texts
        )

        similarities = cosine_similarity(
            batch_matrix,
            reference_matrix
        )

        best_indices = similarities.argmax(axis=1)

        best_scores = similarities[
            range(len(batch_texts)),
            best_indices
        ]

        predictions = []
        scores = []
        methods = []

        for best_index, best_score in zip(
            best_indices,
            best_scores
        ):

            best_score = float(best_score)

            if best_score < confidence_threshold:

                predictions.append(["", "", ""])
                scores.append(0.0)
                methods.append("UNCLASSIFIED")
                continue

            if best_index < len(training_texts):

                prediction = training_labels[best_index]
                method = "LABEL_MATCH"

            else:

                taxonomy_index = (
                    best_index - len(training_texts)
                )

                prediction = taxonomy_reference.iloc[
                    taxonomy_index
                ][required].tolist()

                method = "TAXONOMY_MATCH"

            predictions.append(prediction)
            scores.append(round(best_score, 4))
            methods.append(method)

        # Assign batch predictions
        prediction_df = pd.DataFrame(
            predictions,
            columns=[
                "predicted_l1",
                "predicted_l2",
                "predicted_l3"
            ],
            index=batch_indices
        )

        prediction_df["confidence"] = scores

        prediction_df["classification_method"] = methods

        result.loc[
            batch_indices,
            [
                "predicted_l1",
                "predicted_l2",
                "predicted_l3"
            ]
        ] = prediction_df[
            [
                "predicted_l1",
                "predicted_l2",
                "predicted_l3"
            ]
        ].values

        result.loc[
            batch_indices,
            "confidence"
        ] = scores

        result.loc[
            batch_indices,
            "classification_method"
        ] = methods

        print(
            f"Processed {end:,}/{total_records:,} records"
        )

    return result


def classify_cross_source(
    training_df,
    target_df,
    training_config,
    target_config,
    labels,
    taxonomy,
    confidence_threshold=0.15,
    batch_size=256
):
    """
    Train on labelled ERP records and classify records
    from another source, such as Company A invoice-tool.
    """

    required = [
        "category_l1",
        "category_l2",
        "category_l3"
    ]

    for col in required:
        if col not in labels.columns:
            raise ValueError(f"Missing label column: {col}")

        if col not in taxonomy.columns:
            raise ValueError(f"Missing taxonomy column: {col}")

    # Build ERP and invoice-tool description text
    training_text = build_record_text(
        training_df,
        training_config
    )

    target_text = build_record_text(
        target_df,
        target_config
    )

    # Match ERP records to their labels using record IDs
    training_id = training_config["id_column"]

    label_data = labels.copy()

    label_data["_match_id"] = (
        label_data["record_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    prefix = training_config.get("id_prefix", "")

    if prefix:
        label_data["_match_id"] = (
            label_data["_match_id"].str.replace(
                rf"^{re.escape(str(prefix))}:",
                "",
                regex=True,
                case=False
            )
        )

    label_data = label_data.drop_duplicates(
        subset="_match_id",
        keep="first"
    )

    label_data[required] = (
        label_data[required]
        .fillna("")
        .astype(str)
    )

    label_data = label_data[
        label_data["category_l3"].str.strip().ne("")
    ]

    training_records = pd.DataFrame({
        "_match_id": (
            training_df[training_id]
            .fillna("")
            .astype(str)
            .str.strip()
            .values
        ),
        "text": training_text.values
    })

    training_records = training_records.merge(
        label_data[["_match_id"] + required],
        on="_match_id",
        how="inner"
    )

    training_records = training_records[
        training_records["text"].ne("")
    ]

    if training_records.empty:
        raise ValueError("No labelled ERP training examples found.")

    training_texts = training_records["text"].tolist()
    training_labels = training_records[required].values.tolist()

    print("ERP training examples:", len(training_texts))

    # Prepare taxonomy fallback
    taxonomy_clean = (
        taxonomy[required]
        .fillna("")
        .astype(str)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    taxonomy_texts = (
        taxonomy_clean["category_l1"] + " "
        + taxonomy_clean["category_l2"] + " "
        + taxonomy_clean["category_l3"]
    ).apply(normalize_text).tolist()

    valid_taxonomy = [
        i for i, value in enumerate(taxonomy_texts)
        if value.strip()
    ]

    taxonomy_reference = (
        taxonomy_clean.iloc[valid_taxonomy]
        .reset_index(drop=True)
    )

    taxonomy_texts = [
        taxonomy_texts[i] for i in valid_taxonomy
    ]

    reference_texts = training_texts + taxonomy_texts

    # Initialize target results
    result = target_df.copy()

    result["predicted_l1"] = ""
    result["predicted_l2"] = ""
    result["predicted_l3"] = ""
    result["confidence"] = 0.0
    result["classification_method"] = "UNCLASSIFIED"

    valid_mask = target_text.ne("")
    target_indices = target_df.index[valid_mask]
    target_texts = target_text.loc[target_indices].tolist()

    if not target_texts:
        print("No valid target descriptions.")
        return result

    # Fit vectorizer
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=50000,
        sublinear_tf=True
    )

    vectorizer.fit(reference_texts + target_texts)

    reference_matrix = vectorizer.transform(reference_texts)

    print("Invoice-tool records to classify:", len(target_texts))

    # Batch classification
    for start in range(0, len(target_texts), batch_size):

        end = min(start + batch_size, len(target_texts))

        batch_texts = target_texts[start:end]
        batch_indices = target_indices[start:end]

        batch_matrix = vectorizer.transform(batch_texts)

        similarities = cosine_similarity(
            batch_matrix,
            reference_matrix
        )

        best_indices = similarities.argmax(axis=1)

        best_scores = similarities[
            range(len(batch_texts)),
            best_indices
        ]

        predictions = []
        scores = []
        methods = []

        for best_index, best_score in zip(
            best_indices,
            best_scores
        ):

            best_score = float(best_score)

            if best_score < confidence_threshold:
                predictions.append(["", "", ""])
                scores.append(0.0)
                methods.append("UNCLASSIFIED")
                continue

            if best_index < len(training_texts):

                prediction = training_labels[best_index]
                method = "CROSS_SOURCE_LABEL_MATCH"

            else:

                taxonomy_index = best_index - len(training_texts)

                prediction = taxonomy_reference.iloc[
                    taxonomy_index
                ][required].tolist()

                method = "TAXONOMY_MATCH"

            predictions.append(prediction)
            scores.append(round(best_score, 4))
            methods.append(method)

        prediction_df = pd.DataFrame(
            predictions,
            columns=[
                "predicted_l1",
                "predicted_l2",
                "predicted_l3"
            ],
            index=batch_indices
        )

        result.loc[
            batch_indices,
            [
                "predicted_l1",
                "predicted_l2",
                "predicted_l3"
            ]
        ] = prediction_df.values

        result.loc[batch_indices, "confidence"] = scores

        result.loc[
            batch_indices,
            "classification_method"
        ] = methods

        print(f"Processed {end:,}/{len(target_texts):,} records")

    return result
