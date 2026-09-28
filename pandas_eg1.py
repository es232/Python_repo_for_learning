"""
===========================================================
                    PANDAS - ESSENTIALS
===========================================================

WHAT?
Pandas is a Python library used to work with
structured/tabular data.

WHY?
It makes it easy to:
- Load data
- Clean data
- Filter data
- Transform data
- Analyze data

WHEN?
Use Pandas when working with CSV, Excel, SQL,
JSON or other structured datasets.

WHERE?
Commonly used in:
- Data Science
- Machine Learning
- Data Analysis
- EDA
- AI data preprocessing

INSTALL:
    python -m pip install pandas

RUN:
    python pandas_learning.py
===========================================================
"""

import pandas as pd
import numpy as np


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ===========================================================
# 1. BASIC CONCEPTS
# ===========================================================

section("1. BASIC CONCEPTS")

print("""
Pandas has two main objects:

Series
------
1D labelled data → like one column.

DataFrame
---------
2D table → rows + columns.

Example:

    Name     Age
    ----------
    Arun     22
    Priya    21

DataFrame = complete table
Series    = one column

Pandas vs NumPy:

NumPy  → numerical arrays
Pandas → tables + data analysis
""")


# ===========================================================
# 2. CREATE DATA
# ===========================================================

section("2. CREATE A DATAFRAME")

data = {
    "Name": ["Arun", "Priya", "Rahul", "Meena", "Kavin"],
    "City": ["Chennai", "Madurai", "Chennai", "Trichy", "Madurai"],
    "Marks": [85, 92, 78, np.nan, 88],
    "Age": [21, 22, 21, 23, 22]
}

df = pd.DataFrame(data)

print(df)


# ===========================================================
# 3. LOAD + INSPECT DATA
# ===========================================================

section("3. LOAD AND INSPECT DATA")

print("""
In real projects:

    df = pd.read_csv("data.csv")

Important inspection functions:
""")

print("head():")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nStatistics:")
print(df.describe())


# ===========================================================
# 4. SELECT DATA
# ===========================================================

section("4. SELECT DATA")

print("""
Select one column:
    df["Marks"]

Select multiple columns:
    df[["Name", "Marks"]]

iloc → select using position
loc  → select using labels/conditions
""")

print("Marks:")
print(df["Marks"])

print("\nFirst 2 rows:")
print(df.iloc[:2])

print("\nStudents from Chennai:")
print(df.loc[df["City"] == "Chennai"])


# ===========================================================
# 5. FILTER DATA
# ===========================================================

section("5. FILTER DATA")

print("""
Filtering means selecting rows based on conditions.

Example:
    df[df["Marks"] > 80]

Multiple conditions:
    & → AND
    | → OR
""")

print("Marks > 80:")
print(df[df["Marks"] > 80])

print("\nChennai AND Marks > 80:")
print(
    df[
        (df["City"] == "Chennai") &
        (df["Marks"] > 80)
    ]
)


# ===========================================================
# 6. CLEAN DATA
# ===========================================================

section("6. DATA CLEANING")

print("""
Real-world data often contains:

- Missing values
- Duplicate rows
- Incorrect data

Important functions:

isna()          → find missing values
fillna()        → replace missing values
dropna()        → remove missing rows
drop_duplicates() → remove duplicates
""")

print("Missing values:")
print(df.isna().sum())

# Fill missing marks with average
df["Marks"] = df["Marks"].fillna(df["Marks"].mean())

print("\nAfter filling missing Marks:")
print(df)


# ===========================================================
# 7. CREATE / TRANSFORM DATA
# ===========================================================

section("7. CREATE NEW COLUMNS")

print("""
You can create new columns from existing data.

Example:

    df["Result"] = ...
""")

df["Result"] = df["Marks"].apply(
    lambda x: "Pass" if x >= 40 else "Fail"
)

print(df)


# ===========================================================
# 8. SORTING
# ===========================================================

section("8. SORTING")

print("""
sort_values() → sort rows based on a column.
""")

print(
    df.sort_values(
        by="Marks",
        ascending=False
    )
)


# ===========================================================
# 9. GROUPBY
# ===========================================================

section("9. GROUPBY")

print("""
groupby() is very important for data analysis.

It is similar to SQL GROUP BY.

Example:
    Average marks by city
""")

print(
    df.groupby("City")["Marks"].mean()
)


# ===========================================================
# 10. VALUE COUNTS
# ===========================================================

section("10. VALUE COUNTS")

print("""
value_counts() counts how many times
each value appears.
""")

print(df["City"].value_counts())


# ===========================================================
# 11. REAL-WORLD MINI PROJECT
# ===========================================================

section("11. MINI PROJECT - STUDENT ANALYSIS")

print("""
Business question:

Analyze student performance and find:

1. Average marks
2. Highest marks
3. Top student
4. Students who scored > 80
5. Average marks by city
""")

print("Average marks:", round(df["Marks"].mean(), 2))

print("Highest marks:", df["Marks"].max())

top_student = df.loc[df["Marks"].idxmax(), "Name"]
print("Top student:", top_student)

print("\nStudents scoring > 80:")
print(df[df["Marks"] > 80][["Name", "Marks"]])

print("\nAverage marks by city:")
print(df.groupby("City")["Marks"].mean())


# ===========================================================
# 12. PANDAS IN MACHINE LEARNING
# ===========================================================

section("12. PANDAS + MACHINE LEARNING")

print("""
Typical ML workflow:

Raw Dataset
     ↓
Pandas
     ↓
Inspect
     ↓
Clean
     ↓
Transform
     ↓
Analyze
     ↓
NumPy / Scikit-learn
     ↓
ML Model

Pandas is mainly used for DATA PREPARATION,
not for building the ML algorithm itself.
""")


# ===========================================================
# 13. INTERVIEW ESSENTIALS
# ===========================================================

section("13. INTERVIEW QUESTIONS")

print("""
1. What is Pandas?
   → Python library for data manipulation and analysis.

2. Series vs DataFrame?
   → Series = 1D, DataFrame = 2D.

3. loc vs iloc?
   → loc = labels/conditions, iloc = positions.

4. How do you handle missing values?
   → isna(), fillna(), dropna().

5. What is groupby()?
   → Groups data for analysis, similar to SQL GROUP BY.

6. Pandas vs NumPy?
   → Pandas handles tabular data;
     NumPy handles numerical arrays.

7. How is Pandas used in ML?
   → Loading, cleaning and preparing datasets.
""")


# ===========================================================
# 14. PRACTICE
# ===========================================================

section("14. PRACTICE")

print("""
Try these yourself:

1. Find students with Marks > 90.

2. Find the youngest student.

3. Find the average age.

4. Count students in each city.

5. Sort students by Marks.

6. Find the student with the lowest marks.

7. Create a new column:
       Grade = A/B/C based on marks.

8. Save the DataFrame:
       df.to_csv("students.csv", index=False)
""")


# ===========================================================
# FINAL
# ===========================================================

section("PANDAS CHECKLIST")

print("""
Before moving to Matplotlib + Seaborn,
you should know:

[✓] What / Why / When / Where
[✓] Series
[✓] DataFrame
[✓] read_csv()
[✓] head(), shape, info(), describe()
[✓] Selecting data
[✓] loc / iloc
[✓] Filtering
[✓] Missing values
[✓] Creating columns
[✓] Sorting
[✓] groupby()
[✓] value_counts()
[✓] Basic data analysis
[✓] Pandas in ML

NEXT:
    Matplotlib + Seaborn
""")

