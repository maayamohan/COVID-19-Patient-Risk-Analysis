import csv
import random

icu_count_1 = icu_count_2 = icu_count_9x = 0
mv_count_1 = mv_count_2 = mv_count_9x = 0
death_count_1 = death_count_2 = 0

with open("dataset/Covid Data.csv", "r") as file:
    reader = csv.reader(file)
    header = next(reader)
    reader = list(reader)

    for row in reader:
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
