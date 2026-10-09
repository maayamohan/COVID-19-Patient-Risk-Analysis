import csv
import random

COVID_DATA_PATH = "dataset/Covid Data.csv"
COHORT_1_PATH = "dataset/cohort_1.csv"
COHORT_2_PATH = "dataset/cohort_2.csv"

def outcome_summary(data, target):
    count_yes = count_no = count_unknown = 0
    targets = {
        "ICU": {
            "index": 20,
            "yes": lambda value: value == "1",
            "no": lambda value: value == "2",
            "unknown": lambda value: value in ["97", "98", "99"]
        },
        "MV": {
            "index": 5,
            "yes": lambda value: value == "1",
            "no": lambda value: value == "2",
            "unknown": lambda value: value in ["97", "98", "99"]
        },
        "Death": {
            "index": 4,
            "yes": lambda value: value != "9999-99-99",
            "no": lambda value: value == "9999-99-99",
            "unknown": lambda value: False
        }
    }

    config = targets[target]

    for row in data:
        value = row[config["index"]]

        if config["yes"](value):
            count_no += 1
        elif config["no"](value):
            count_yes += 1
        elif config["unknown"](value):
            count_unknown += 1

    print("\n", target, sep="")
    print("Yes:", count_yes)
    print("No:", count_no)
    print("Unknown:", count_unknown)

    return(count_yes, count_no, count_unknown)


with open(COVID_DATA_PATH, "r") as file:
    reader = csv.reader(file)
    header = next(reader)
    reader = list(reader)
    reader_filtered = []

    for row in reader:
        if (int(row[7]) >= 18) and not ((row[20] in ["97", "98", "99"]) and  (row[5] in ["97", "98", "99"]) and row[4] == "9999-99-99"):
            reader_filtered.append(row)

    cohort_1 = []
    cohort_2 = []

    go3 = reader_filtered[:106413]
    go4 = reader_filtered[106413:]

    sv = 0
    ev = 3
    for gn in range(35471):
        chunk = go3[sv:ev]
        cohort_2.append(chunk.pop(random.randrange(len(chunk))))
        cohort_1.extend(chunk)
        sv += len(chunk)
        ev += len(chunk)
    sv = 0
    ev = 4
    for gn in range(21217):
            chunk = go4[sv:ev]
            cohort_2.append(chunk.pop(random.randrange(len(chunk))))
            cohort_1.extend(chunk)
            sv += len(chunk)
            ev += len(chunk)

    print("\n\nTOTAL")
    outcome_summary(reader_filtered, "ICU")
    outcome_summary(reader_filtered, "MV")
    outcome_summary(reader_filtered, "Death")
    print("\n\nCOHORT 1")
    outcome_summary(cohort_1, "ICU")
    outcome_summary(cohort_1, "MV")
    outcome_summary(cohort_1, "Death")
    print("\n\nCOHORT 2\n")
    outcome_summary(cohort_2, "ICU")
    outcome_summary(cohort_2, "MV")
    outcome_summary(cohort_2, "Death")

with open(COHORT_1_PATH, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(cohort_1)

with open(COHORT_2_PATH, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(cohort_2)