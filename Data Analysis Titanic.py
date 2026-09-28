# ==========================================
# TITANIC DATA ANALYSIS
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# LOAD CSV FILE
# ==========================================

file_path = r"C:\Users\Kittu\OneDrive\Desktop\titanic.csv"

df = pd.read_csv(file_path)


# ==========================================
# CLEAN COLUMN NAMES
# ==========================================

# Convert column names to strings
df.columns = df.columns.astype(str)

# Remove spaces
df.columns = df.columns.str.strip()

# Convert to lowercase
df.columns = df.columns.str.lower()

# Replace spaces with underscore
df.columns = df.columns.str.replace(" ", "_")


print("\n==========================================")
print("COLUMNS PRESENT IN YOUR CSV")
print("==========================================")

print(df.columns.tolist())


# ==========================================
# CHECK REQUIRED COLUMNS
# ==========================================

print("\n==========================================")
print("CHECKING REQUIRED COLUMNS")
print("==========================================")

required_columns = ["age", "sex", "survived"]

for column in required_columns:

    if column in df.columns:
        print("FOUND:", column)

    else:
        print("NOT FOUND:", column)


# ==========================================
# 1. FIRST 10 RECORDS
# ==========================================

print("\n==========================================")
print("1. FIRST 10 RECORDS")
print("==========================================")

print(df.head(10))


# ==========================================
# 2. NUMBER OF ROWS AND COLUMNS
# ==========================================

print("\n==========================================")
print("2. NUMBER OF ROWS AND COLUMNS")
print("==========================================")

rows, columns = df.shape

print("Number of rows:", rows)
print("Number of columns:", columns)


# ==========================================
# 3. DATA TYPES
# ==========================================

print("\n==========================================")
print("3. DATA TYPES")
print("==========================================")

print(df.dtypes)


# ==========================================
# 4. MISSING VALUES
# ==========================================

print("\n==========================================")
print("4. MISSING VALUES")
print("==========================================")

print(df.isnull().sum())


# ==========================================
# 5. MEAN AGE
# ==========================================

print("\n==========================================")
print("5. MEAN AGE")
print("==========================================")

# Try to find age column
age_column = None

possible_age_columns = [
    "age",
    "passenger_age",
    "passengerage",
    "ages"
]

for column in possible_age_columns:

    if column in df.columns:
        age_column = column
        break


if age_column is not None:

    mean_age = df[age_column].mean()

    print("Age column found:", age_column)
    print("Mean age:", mean_age)

else:

    print("ERROR: No age column was found.")
    print("Available columns are:")
    print(df.columns.tolist())


# ==========================================
# 6. MALE AND FEMALE PASSENGERS
# ==========================================

print("\n==========================================")
print("6. MALE AND FEMALE PASSENGERS")
print("==========================================")


# Find gender column
sex_column = None

possible_sex_columns = [
    "sex",
    "gender"
]

for column in possible_sex_columns:

    if column in df.columns:
        sex_column = column
        break


if sex_column is not None:

    gender_count = df[sex_column].value_counts()

    print("Gender column found:", sex_column)
    print(gender_count)

else:

    print("ERROR: No gender/sex column was found.")


# ==========================================
# 7. SURVIVAL RATE
# ==========================================

print("\n==========================================")
print("7. SURVIVAL RATE")
print("==========================================")


# Find survival column
survival_column = None

possible_survival_columns = [
    "survived",
    "survival",
    "survival_status"
]

for column in possible_survival_columns:

    if column in df.columns:
        survival_column = column
        break


if survival_column is not None:

    survival_rate = df[survival_column].mean() * 100

    print("Survival column found:", survival_column)
    print("Survival Rate:", survival_rate, "%")

else:

    print("ERROR: No survival column was found.")


# ==========================================
# 8. BAR CHART - SURVIVAL BY GENDER
# ==========================================

print("\n==========================================")
print("8. SURVIVAL BY GENDER")
print("==========================================")


if sex_column is not None and survival_column is not None:

    survival_gender = pd.crosstab(
        df[sex_column],
        df[survival_column]
    )

    print(survival_gender)

    survival_gender.plot(
        kind="bar",
        figsize=(7, 5)
    )

    plt.title("Survival by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Number of Passengers")
    plt.xticks(rotation=0)

    plt.legend(
        ["Did Not Survive", "Survived"]
    )

    plt.tight_layout()

    plt.show()


# ==========================================
# 9. HISTOGRAM OF PASSENGER AGES
# ==========================================

print("\n==========================================")
print("9. HISTOGRAM OF PASSENGER AGES")
print("==========================================")


if age_column is not None:

    plt.figure(figsize=(7, 5))

    plt.hist(
        df[age_column].dropna(),
        bins=20
    )

    plt.title("Distribution of Passenger Ages")
    plt.xlabel("Age")
    plt.ylabel("Number of Passengers")

    plt.tight_layout()

    plt.show()

else:

    print("Cannot create histogram because age column is missing.")


# ==========================================
# 10. CORRELATION
# ==========================================

print("\n==========================================")
print("10. CORRELATION BETWEEN NUMERICAL VARIABLES")
print("==========================================")

numeric_data = df.select_dtypes(
    include=np.number
)

correlation = numeric_data.corr()

print(correlation)


# ==========================================
# PROGRAM COMPLETED
# ==========================================

print("\n==========================================")
print("ANALYSIS COMPLETED")
print("==========================================")

