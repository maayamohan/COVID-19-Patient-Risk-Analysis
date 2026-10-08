import pandas as pd

# Load dataset
df = pd.read_csv("dataset/Covid Data.csv")

print("\n===== DATASET =====")
print(f"Rows: {len(df):,}")
print(f"Columns: {df.shape[1]}")

# --------------------------------------------------
# ICU
# --------------------------------------------------

print("\n===== ICU TARGET =====")
print(df["ICU"].value_counts().sort_index())

icu_known = df[df["ICU"].isin([1, 2])]

print("\nICU known outcomes:")
print(icu_known["ICU"].value_counts())

print("\nICU by PATIENT_TYPE:")
print(pd.crosstab(icu_known["PATIENT_TYPE"], icu_known["ICU"]))


# --------------------------------------------------
# INTUBATION / MECHANICAL VENTILATION
# --------------------------------------------------

print("\n===== INTUBATION TARGET =====")
print(df["INTUBED"].value_counts().sort_index())

mv_known = df[df["INTUBED"].isin([1, 2])]

print("\nIntubation known outcomes:")
print(mv_known["INTUBED"].value_counts())

print("\nIntubation by PATIENT_TYPE:")
print(pd.crosstab(mv_known["PATIENT_TYPE"], mv_known["INTUBED"]))


# --------------------------------------------------
# DEATH
# --------------------------------------------------

print("\n===== DEATH TARGET =====")

df["DEATH"] = (df["DATE_DIED"] != "9999-99-99").astype(int)

print(df["DEATH"].value_counts())

print("\nDeath by PATIENT_TYPE:")
print(pd.crosstab(df["PATIENT_TYPE"], df["DEATH"]))


# --------------------------------------------------
# RELATIONSHIPS BETWEEN OUTCOMES
# --------------------------------------------------

print("\n===== ICU vs INTUBATION =====")

known_both = df[
    df["ICU"].isin([1, 2]) &
    df["INTUBED"].isin([1, 2])
]

print(pd.crosstab(
    known_both["ICU"],
    known_both["INTUBED"],
    rownames=["ICU"],
    colnames=["INTUBED"]
))


print("\n===== ICU vs DEATH =====")

icu_death = df[df["ICU"].isin([1, 2])]

print(pd.crosstab(
    icu_death["ICU"],
    icu_death["DEATH"],
    rownames=["ICU"],
    colnames=["DEATH"]
))


print("\n===== INTUBATION vs DEATH =====")

mv_death = df[df["INTUBED"].isin([1, 2])]

print(pd.crosstab(
    mv_death["INTUBED"],
    mv_death["DEATH"],
    rownames=["INTUBED"],
    colnames=["DEATH"]
))