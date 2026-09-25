import pandas as pd
import re
from collections import Counter

FILE = "dataset/sif_sanket_synthetic_safety_reports_v3.csv"

df = pd.read_csv(FILE)

print("\n========================================")
print("SIF-SANKET V3 DATASET AUDIT")
print("========================================")

# =========================================================
# 1. BASIC DATASET CHECK
# =========================================================

print("\n[1] BASIC CHECK")

print("Total rows:", len(df))
print("Total columns:", len(df.columns))

print("\nLabel distribution:")
print(df["sif_potential"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())


# =========================================================
# 2. EXACT DUPLICATE CHECK
# =========================================================

print("\n[2] EXACT DUPLICATE CHECK")

duplicate_texts = df["report_text"].duplicated().sum()
unique_texts = df["report_text"].nunique()

print("Duplicate report texts:", duplicate_texts)
print("Unique report texts:", unique_texts)


# =========================================================
# 3. REPORT LENGTH ANALYSIS
# =========================================================

print("\n[3] REPORT LENGTH")

df["word_count"] = df["report_text"].str.split().str.len()

print("Minimum words:", df["word_count"].min())
print("Maximum words:", df["word_count"].max())
print("Average words:", round(df["word_count"].mean(), 2))

print("\nAverage words by label:")
print(
    df.groupby("sif_potential")["word_count"]
    .mean()
    .round(2)
)


# =========================================================
# 4. LABEL LEAKAGE CHECK
# =========================================================

print("\n[4] LABEL LEAKAGE CHECK")

text_lower = df["report_text"].str.lower()

leakage_words = [
    "sif",
    "serious injury",
    "fatality",
    "fatal",
    "high-consequence",
    "life-threatening",
    "no credible precursor",
    "no sif potential",
    "no sif precursor",
    "serious harm"
]

for word in leakage_words:
    count = text_lower.str.contains(
        re.escape(word),
        regex=True
    ).sum()

    if count > 0:
        print(f"{word}: {count}")


# =========================================================
# 5. YES / NO WORD FREQUENCY
# =========================================================

print("\n[5] COMMON WORDS BY LABEL")

stop_words = {
    "the", "a", "an", "and", "or", "to", "of",
    "in", "at", "on", "for", "was", "were",
    "during", "with", "before", "after",
    "the", "team", "work", "area"
}


def get_words(series):

    words = []

    for text in series:

        tokens = re.findall(
            r"[a-zA-Z]+",
            text.lower()
        )

        for token in tokens:

            if (
                len(token) > 3
                and token not in stop_words
            ):
                words.append(token)

    return Counter(words)


yes_words = get_words(
    df[df["sif_potential"] == "YES"]["report_text"]
)

no_words = get_words(
    df[df["sif_potential"] == "NO"]["report_text"]
)

print("\nTop YES words:")
print(yes_words.most_common(20))

print("\nTop NO words:")
print(no_words.most_common(20))


# =========================================================
# 6. ACTIVITY DISTRIBUTION
# =========================================================

print("\n[6] ACTIVITY DISTRIBUTION")

activity_table = pd.crosstab(
    df["activity"],
    df["sif_potential"]
)

print(activity_table)


# =========================================================
# 7. SITE DISTRIBUTION
# =========================================================

print("\n[7] SITE DISTRIBUTION")

site_table = pd.crosstab(
    df["site"],
    df["sif_potential"]
)

print(site_table)


# =========================================================
# 8. REPORT TYPE DISTRIBUTION
# =========================================================

print("\n[8] REPORT TYPE DISTRIBUTION")

type_table = pd.crosstab(
    df["report_type"],
    df["sif_potential"]
)

print(type_table)


# =========================================================
# 9. PRECURSOR DISTRIBUTION
# =========================================================

print("\n[9] PRECURSOR DISTRIBUTION")

print(
    pd.crosstab(
        df["precursor"],
        df["sif_potential"]
    )
)


# =========================================================
# 10. LIFE SAVING RULE DISTRIBUTION
# =========================================================

print("\n[10] LIFE SAVING RULE DISTRIBUTION")

print(
    pd.crosstab(
        df["life_saving_rule"],
        df["sif_potential"]
    )
)


# =========================================================
# 11. IDENTICAL SENTENCE STRUCTURE CHECK
# =========================================================

print("\n[11] SENTENCE STRUCTURE CHECK")

def normalize_text(text):

    text = text.lower()

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove location/site/activity-specific words
    text = re.sub(
        r"\b(refinery|pipeline|compressor|tank|maintenance|"
        r"electrical|crane|lifting|welding|vehicle|"
        r"grinding|vessel|production|utility|drilling|loading)\w*\b",
        "",
        text
    )

    # Remove punctuation
    text = re.sub(r"[^a-z\s]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["normalized_text"] = df["report_text"].apply(
    normalize_text
)

normalized_duplicates = (
    df["normalized_text"].duplicated().sum()
)

print(
    "Duplicate normalized structures:",
    normalized_duplicates
)


# =========================================================
# 12. CROSS-LABEL STRUCTURE COLLISION
# =========================================================

print("\n[12] CROSS-LABEL STRUCTURE CHECK")

structure_groups = (
    df.groupby("normalized_text")["sif_potential"]
    .nunique()
)

cross_label_structures = (
    structure_groups[structure_groups > 1]
)

print(
    "Structures appearing in both YES and NO:",
    len(cross_label_structures)
)


# =========================================================
# 13. SAMPLE REPORTS
# =========================================================

print("\n========================================")
print("SAMPLE YES REPORTS")
print("========================================")

for text in df[
    df["sif_potential"] == "YES"
]["report_text"].head(5):

    print("\n-", text)


print("\n========================================")
print("SAMPLE NO REPORTS")
print("========================================")

for text in df[
    df["sif_potential"] == "NO"
]["report_text"].head(5):

    print("\n-", text)


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n========================================")
print("AUDIT COMPLETE")
print("========================================")

print("Rows:", len(df))
print("Unique texts:", df["report_text"].nunique())
print("Exact duplicates:", duplicate_texts)
print("Normalized duplicates:", normalized_duplicates)
print(
    "Cross-label structures:",
    len(cross_label_structures)
)

print("\nNext step:")
print("Review the audit results before model training.")
