import csv
import random

icu_count_1 = icu_count_2 = icu_count_9x = 0
mv_count_1 = mv_count_2 = mv_count_9x = 0
death_count_1 = death_count_2 = 0

icu_count_1_1 = icu_count_2_1 = icu_count_9x_1 = 0
mv_count_1_1 = mv_count_2_1 = mv_count_9x_1 = 0
death_count_1_1 = death_count_2_1 = 0

icu_count_1_2 = icu_count_2_2 = icu_count_9x_2 = 0
mv_count_1_2 = mv_count_2_2 = mv_count_9x_2 = 0
death_count_1_2 = death_count_2_2 = 0

with open("dataset/Covid Data.csv", "r") as file:
    reader = csv.reader(file)
    header = next(reader)
    reader = list(reader)
    reader_filtered = []

    for row in reader:
        if (int(row[7]) >= 18) and not ((row[20] == "97" or row[20] == "99") and  (row[5] == "97" or row[5] == "99") and row[4] == "9999-99-99"):
            reader_filtered.append(row)

    cohort_1 = []
    cohort_2 = []

    go3 = reader_filtered[:106413]
    go4 = reader_filtered[106413:]

    sv = 0
    ev = 3
    for gn in range(35471):
        chunk = go3[sv:ev]
        cohort_2.append(chunk.pop(random.randrange(0, 3)))
        cohort_1.extend(chunk)
        sv += 3
        ev += 3
    sv = 0
    ev = 4
    for gn in range(21217):
            chunk = go4[sv:ev]
            cohort_2.append(chunk.pop(random.randrange(0, 4)))
            cohort_1.extend(chunk)
            sv += 4
            ev += 4
    
    for row in reader_filtered:
        if row[20] == "1":
            icu_count_1 += 1
        if row[20] == "2":
            icu_count_2 += 1
        if row[20] == "97" or row[20] == "99":
            icu_count_9x += 1
        if row[5] == "1":
            mv_count_1 += 1
        if row[5] == "2":
            mv_count_2 += 1
        if row[5] == "97" or row[5] == "99":
            mv_count_9x += 1
        if row[4] == "9999-99-99":
            death_count_2 += 1
        if row[4] != "9999-99-99":
            death_count_1 += 1
    
    for row in cohort_1:
        if row[20] == "1":
            icu_count_1_1 += 1
        if row[20] == "2":
            icu_count_2_1 += 1
        if row[20] == "97" or row[20] == "99":
            icu_count_9x_1 += 1
        if row[5] == "1":
            mv_count_1_1 += 1
        if row[5] == "2":
            mv_count_2_1 += 1
        if row[5] == "97" or row[5] == "99":
            mv_count_9x_1 += 1
        if row[4] == "9999-99-99":
            death_count_2_1 += 1
        if row[4] != "9999-99-99":
            death_count_1_1 += 1
    
    for row in cohort_2:
        if row[20] == "1":
            icu_count_1_2 += 1
        if row[20] == "2":
            icu_count_2_2 += 1
        if row[20] == "97" or row[20] == "99":
            icu_count_9x_2 += 1
        if row[5] == "1":
            mv_count_1_2 += 1
        if row[5] == "2":
            mv_count_2_2 += 1
        if row[5] == "97" or row[5] == "99":
            mv_count_9x_2 += 1
        if row[4] == "9999-99-99":
            death_count_2_2 += 1
        if row[4] != "9999-99-99":
            death_count_1_2 += 1

print("\nTotal")

print("\nICU")
print("Yes:", icu_count_1)
print("No:", icu_count_2)
print("Unknown:", icu_count_9x)
print("Total:", icu_count_9x + icu_count_2 + icu_count_1)

print("\nMV")
print("Yes:", mv_count_1)
print("No:", mv_count_2)
print("Unknown:", mv_count_9x)
print("Total:", mv_count_9x + mv_count_2 + mv_count_1)

print("\nDeath")
print("Yes:", death_count_1)
print("No:", death_count_2)
print("Total:", death_count_2 + death_count_1)

print("\nCohort 1")

print("\nICU")
print("Yes:", icu_count_1_1)
print("No:", icu_count_2_1)
print("Unknown:", icu_count_9x_1)
print("Total:", icu_count_9x_1 + icu_count_2_1 + icu_count_1_1)

print("\nMV")
print("Yes:", mv_count_1_1)
print("No:", mv_count_2_1)
print("Unknown:", mv_count_9x_1)
print("Total:", mv_count_9x_1 + mv_count_2_1 + mv_count_1_1)

print("\nDeath")
print("Yes:", death_count_1_1)
print("No:", death_count_2_1)
print("Total:", death_count_2_1 + death_count_1_1)

print("\nCohort 2")

print("\nICU")
print("Yes:", icu_count_1_2)
print("No:", icu_count_2_2)
print("Unknown:", icu_count_9x_2)
print("Total:", icu_count_9x_2 + icu_count_2_2 + icu_count_1_2)

print("\nMV")
print("Yes:", mv_count_1_2)
print("No:", mv_count_2_2)
print("Unknown:", mv_count_9x_2)
print("Total:", mv_count_9x_2 + mv_count_2_2 + mv_count_1_2)

print("\nDeath")
print("Yes:", death_count_1_2)
print("No:", death_count_2_2)
print("Total:", death_count_2_2 + death_count_1_2)
