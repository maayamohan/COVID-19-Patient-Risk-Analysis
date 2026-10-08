import pandas as pd

# Load dataset
df = pd.read_csv("dataset/Covid Data.csv")

# --------------------------------------------------
# FEATURES WE NEED TO INVESTIGATE
# --------------------------------------------------

features = [
    "USMER",
    "MEDICAL_UNIT",
    "SEX",
    "PATIENT_TYPE",
    "PNEUMONIA",
    "AGE",
    "PREGNANT",
    "DIABETES",
    "COPD",
    "ASTHMA",
    "INMSUPR",
    "HIPERTENSION",
    "OTHER_DISEASE",
    "CARDIOVASCULAR",
    "OBESITY",
    "RENAL_CHRONIC",
    "TOBACCO",
    "CLASIFFICATION_FINAL"
]

for column in features:
    print("\n" + "=" * 60)
    print(f"{column}")
    print("=" * 60)

    print(df[column].value_counts(dropna=False).sort_index())


# --------------------------------------------------
# UNKNOWN / SPECIAL VALUES
# --------------------------------------------------

print("\n" + "=" * 60)
print("SPECIAL VALUES")
print("=" * 60)

special_values = [97, 98, 99]

for column in features:
    counts = df[column].value_counts()

    special_counts = {
        value: counts.get(value, 0)
        for value in special_values
        if value in counts.index
    }

    if special_counts:
        print(f"\n{column}: {special_counts}")


# --------------------------------------------------
# AGE SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("AGE SUMMARY")
print("=" * 60)

print(df["AGE"].describe())


# --------------------------------------------------
# CLASSIFICATION FINAL × TARGETS
# --------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION_FINAL vs ICU")
print("=" * 60)

icu_known = df[df["ICU"].isin([1, 2])]

print(pd.crosstab(
    icu_known["CLASIFFICATION_FINAL"],
    icu_known["ICU"]
))


print("\n" + "=" * 60)
print("CLASSIFICATION_FINAL vs INTUBATION")
print("=" * 60)

mv_known = df[df["INTUBED"].isin([1, 2])]

print(pd.crosstab(
    mv_known["CLASIFFICATION_FINAL"],
    mv_known["INTUBED"]
))


# --------------------------------------------------
# AGE vs TARGETS
# --------------------------------------------------

print("\n" + "=" * 60)
print("AGE BY ICU")
print("=" * 60)

print(
    icu_known.groupby("ICU")["AGE"]
    .describe()
)


print("\n" + "=" * 60)
print("AGE BY INTUBATION")
print("=" * 60)

print(
    mv_known.groupby("INTUBED")["AGE"]
    .describe()
)


# --------------------------------------------------
# PNEUMONIA vs TARGETS
# --------------------------------------------------

print("\n" + "=" * 60)
print("PNEUMONIA vs ICU")
print("=" * 60)

print(pd.crosstab(
    icu_known["PNEUMONIA"],
    icu_known["ICU"]
))


print("\n" + "=" * 60)
print("PNEUMONIA vs INTUBATION")
print("=" * 60)

print(pd.crosstab(
    mv_known["PNEUMONIA"],
    mv_known["INTUBED"]
))