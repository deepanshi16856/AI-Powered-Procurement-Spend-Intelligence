
from src.ingestion import load_config, load_csv
from src.entity_resolution import (
    create_supplier_keys,
    create_item_keys
)


def test_entity_resolution():
    config = load_config("configs/company_a.yaml")

    labels = load_csv(config["labels_file"])

    print("\n" + "=" * 55)
    print("COMPANY A ENTITY RESOLUTION TEST")
    print("=" * 55)

    for source in config["sources"]:

        df = load_csv(source["file"])

        # Generate supplier and item keys.
        result = create_supplier_keys(df, source)
        result = create_item_keys(result, source)

        print("\nSource:", source["name"])
        print("Total records:", len(result))

        print("Unique supplier keys:", result["supplier_key"].nunique())
        print("Missing supplier keys:", result["supplier_key"].eq("").sum())

        print("Unique item keys:", result["item_key"].nunique())
        print("Missing item keys:", result["item_key"].eq("").sum())

        # Compare generated keys with reference labels.
        if source["name"] == "erp":

            result["record_id"] = (
                source["id_prefix"] + ":"
                + result[source["id_column"]].astype(str)
            )

            comparison = result.merge(
                labels[["record_id", "supplier_key", "item_key"]],
                on="record_id",
                how="inner",
                suffixes=("_generated", "_reference")
            )

            print("\nReference label comparison:")
            print("Matched labelled records:", len(comparison))

            for entity in ["supplier", "item"]:

                generated = comparison[f"{entity}_key_generated"]
                reference = comparison[f"{entity}_key_reference"]

                valid = (
                    generated.ne("")
                    & reference.notna()
                    & reference.astype(str).ne("")
                )

                mapping = comparison.loc[
                    valid,
                    [
                        f"{entity}_key_generated",
                        f"{entity}_key_reference"
                    ]
                ]

                # Check whether one generated key maps to
                # multiple reference keys.
                conflicts = (
                    mapping.groupby(
                        f"{entity}_key_generated"
                    )[f"{entity}_key_reference"]
                    .nunique()
                    .gt(1)
                    .sum()
                )

                print(
                    f"\n{entity.title()} keys with conflicting labels:",
                    conflicts
                )

                # Check whether one reference key maps to
                # multiple generated keys.
                reverse_conflicts = (
                    mapping.groupby(
                        f"{entity}_key_reference"
                    )[f"{entity}_key_generated"]
                    .nunique()
                    .gt(1)
                    .sum()
                )

                print(
                    "Reference keys mapped to multiple generated keys:",
                    reverse_conflicts
                )

                print(
                    "Unique generated-to-reference mappings:",
                    mapping.drop_duplicates().shape[0]
                )


if __name__ == "__main__":
    test_entity_resolution()
