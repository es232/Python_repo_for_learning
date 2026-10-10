"""
======================================================================
       JUPYTER NOTEBOOK + DATA CLEANING + COMPLETE EDA
======================================================================

WHAT WILL YOU LEARN?
--------------------

PART 1  -> Jupyter Notebook
PART 2  -> Data Cleaning
PART 3  -> Complete EDA Workflow
PART 4  -> ML Data Preparation
PART 5  -> Interview Questions
PART 6  -> Common Mistakes
PART 7  -> Practice Exercises
PART 8  -> Mini Project

IMPORTANT WORKFLOW:

Raw Dataset
     ↓
Load Data
     ↓
Understand Data
     ↓
Clean Data
     ↓
Explore Data
     ↓
Visualize Data
     ↓
Find Patterns
     ↓
Feature Engineering
     ↓
Prepare for ML
     ↓
Machine Learning

INSTALL:
    python -m pip install jupyter pandas numpy matplotlib seaborn

START JUPYTER:
    jupyter notebook

======================================================================
"""


# ====================================================================
# IMPORTS
# ====================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ====================================================================
# 1. JUPYTER NOTEBOOK
# ====================================================================

print("\n" + "=" * 70)
print("1. JUPYTER NOTEBOOK")
print("=" * 70)

print("""
WHAT IS JUPYTER?
----------------

Jupyter Notebook is an interactive environment where you can
write and execute Python code in separate cells.

Instead of running an entire .py file at once, you can run
one block of code at a time.

WHY USE JUPYTER?
----------------

It is extremely useful for:

    - Data Science
    - Machine Learning
    - Data Analysis
    - EDA
    - Visualization
    - Experimentation
    - Learning Python/ML

WHY IS IT IMPORTANT FOR ML?
---------------------------

Machine Learning is highly experimental.

You often need to:

    1. Load data
    2. Inspect it
    3. Try cleaning
    4. Visualize
    5. Try transformations
    6. Train a model
    7. Change something
    8. Run again

Jupyter makes this interactive.


JUPYTER NOTEBOOK FILE
---------------------

Jupyter files usually have:

    .ipynb

Example:

    sales_analysis.ipynb


BASIC CELL TYPES
----------------

Code Cell:
    Used to execute Python.

Markdown Cell:
    Used for explanations, headings and notes.


IMPORTANT SHORTCUTS
-------------------

Shift + Enter
    Run current cell and move to next cell.

Ctrl + Enter
    Run current cell and stay there.

A
    Insert cell above.

B
    Insert cell below.

M
    Convert cell to Markdown.

Y
    Convert cell to Code.

DD
    Delete cell.


TYPICAL NOTEBOOK STRUCTURE
--------------------------

1. Project title
2. Imports
3. Load dataset
4. Understand dataset
5. Data cleaning
6. EDA
7. Visualization
8. Feature engineering
9. ML preparation
10. Conclusions
""")


# ====================================================================
# 2. DATASET
# ====================================================================

print("\n" + "=" * 70)
print("2. CREATE A REALISTIC DATASET")
print("=" * 70)

print("""
We will simulate an e-commerce customer dataset.

Real datasets often contain problems such as:

    - Missing values
    - Duplicate rows
    - Incorrect data types
    - Inconsistent text
    - Invalid values
    - Outliers

Our goal is to identify and fix these problems.
""")

data = {
    "Customer_ID": [
        101, 102, 103, 104, 105,
        106, 107, 108, 109, 110,
        110, 112, 113, 114, 115
    ],

    "Name": [
        "Arun", "PRIYA", "rahul", "Meena", "Kavin",
        "divya", "Vijay", "Anu", "Suresh", "Nila",
        "Nila", "Hari", "Keerthi", "Manoj", "Asha"
    ],

    "City": [
        "Chennai", "chennai", "Madurai", "Trichy",
        "Chennai", "MADURAI", "Chennai", "Trichy",
        "chennai", "Madurai", "Madurai", "Chennai",
        "Trichy", "Chennai", "madurai"
    ],

    "Age": [
        21, 22, np.nan, 24, 23,
        25, 22, 21, 150, 24,
        24, np.nan, 23, 26, 22
    ],

    "Product": [
        "Laptop", "Phone", "Laptop", "Tablet", "Phone",
        "Laptop", "Phone", "Tablet", "Laptop", "Phone",
        "Phone", "Laptop", "Tablet", "Laptop", "Phone"
    ],

    "Quantity": [
        1, 2, 1, 3, 2,
        1, 4, 2, 1, 2,
        2, 1, 3, 1, 2
    ],

    "Price": [
        75000, 30000, 75000, 25000, 32000,
        80000, 31000, 26000, 90000, 30000,
        30000, 78000, 27000, 82000, 31000
    ],

    "Rating": [
        4.5, 4.2, np.nan, 4.0, 4.3,
        4.7, 4.1, np.nan, 4.8, 4.2,
        4.2, 4.6, 4.0, 4.9, 4.3
    ]
}

df = pd.DataFrame(data)

print("\nRAW DATA:")
print(df)


# ====================================================================
# 3. UNDERSTANDING THE DATA
# ====================================================================

print("\n" + "=" * 70)
print("3. UNDERSTANDING THE DATA")
print("=" * 70)

print("""
Before cleaning anything, understand the dataset.

Important questions:

    - How many rows?
    - How many columns?
    - What are the column names?
    - What are the data types?
    - Are values missing?
    - Are there duplicates?
    - What are the numerical statistics?

This is the first step of EDA.
""")


print("\nFirst 5 rows:")
print(df.head())


print("\nLast 5 rows:")
print(df.tail())


print("\nDataset shape:")
print(df.shape)


print("\nColumn names:")
print(df.columns.tolist())


print("\nData types:")
print(df.dtypes)


print("\nBasic information:")
df.info()


print("\nStatistical summary:")
print(df.describe())


# ====================================================================
# 4. MISSING VALUES
# ====================================================================

print("\n" + "=" * 70)
print("4. MISSING VALUES")
print("=" * 70)

print("""
WHAT?
-----
A missing value means information is unavailable.

In Pandas, missing values are commonly represented by:

    NaN

WHY IS IT IMPORTANT?
--------------------

Many ML algorithms cannot directly handle missing values.

We need to decide what to do with them.

COMMON OPTIONS:

1. Remove rows
2. Remove columns
3. Fill with mean
4. Fill with median
5. Fill with mode
6. Use domain-specific values
7. Use advanced imputation


CHECK MISSING VALUES:
---------------------

    df.isna()

COUNT MISSING VALUES:

    df.isna().sum()
""")

print("\nMissing values:")
print(df.isna().sum())


# ====================================================================
# 5. HANDLING MISSING VALUES
# ====================================================================

print("\n" + "=" * 70)
print("5. HANDLING MISSING VALUES")
print("=" * 70)

print("""
METHOD 1 — DROP ROWS
--------------------

    df.dropna()

Use this when only a very small number of rows are missing
and removing them will not significantly affect the dataset.


METHOD 2 — DROP COLUMNS
-----------------------

    df.drop(columns=["column"])


METHOD 3 — FILL NUMERICAL VALUES
---------------------------------

Mean:
    df["Age"].fillna(df["Age"].mean())

Median:
    df["Age"].fillna(df["Age"].median())


WHEN TO USE MEAN?
-----------------

Mean works reasonably when the distribution is fairly balanced
and there are no major outliers.


WHEN TO USE MEDIAN?
-------------------

Median is often safer when the data contains outliers or
is skewed.


METHOD 4 — MODE
---------------

For categorical data:

    df["City"].fillna(df["City"].mode()[0])
""")


# Use median for Age because age contains an extreme value.
df["Age"] = df["Age"].fillna(df["Age"].median())

# Use median for Rating.
df["Rating"] = df["Rating"].fillna(df["Rating"].median())

print("\nAfter filling missing values:")
print(df.isna().sum())


# ====================================================================
# 6. DUPLICATES
# ====================================================================

print("\n" + "=" * 70)
print("6. DUPLICATE DATA")
print("=" * 70)

print("""
WHAT?
-----
A duplicate is a repeated record.

WHY?
----
Duplicates can cause:

    - Incorrect statistics
    - Double counting
    - Biased ML training
    - Incorrect business decisions


CHECK:

    df.duplicated()

COUNT:

    df.duplicated().sum()

REMOVE:

    df.drop_duplicates()
""")

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("\nDataset after removing exact duplicates:")
print(df)


# ====================================================================
# 7. DUPLICATES BASED ON A COLUMN
# ====================================================================

print("\n" + "=" * 70)
print("7. DUPLICATES USING A SPECIFIC COLUMN")
print("=" * 70)

print("""
Sometimes rows may not be completely identical.

Example:

Customer_ID = 110

appears more than once.

You can check:

    df.duplicated(subset=["Customer_ID"])

And remove using:

    df.drop_duplicates(
        subset=["Customer_ID"]
    )

IMPORTANT:
----------
Do not automatically remove duplicates based on one column.

First understand what the column represents.

A customer may legitimately have multiple orders.
""")

print(
    "\nDuplicate Customer IDs:"
)

print(
    df[df.duplicated(
        subset=["Customer_ID"],
        keep=False
    )]
)


# ====================================================================
# 8. DATA TYPES
# ====================================================================

print("\n" + "=" * 70)
print("8. DATA TYPES")
print("=" * 70)

print("""
Machine learning requires data to be in usable formats.

Common Pandas types:

int64
    Integer

float64
    Decimal number

object
    Usually text

bool
    True/False

datetime
    Date/time


CHECK:

    df.dtypes


CONVERT:

    df["Age"].astype(int)

For dates:

    pd.to_datetime(df["Date"])
""")


print(df.dtypes)


# ====================================================================
# 9. INVALID VALUES
# ====================================================================

print("\n" + "=" * 70)
print("9. INVALID VALUES")
print("=" * 70)

print("""
Not every problem is represented by NaN.

Example:

Age = 150

Technically this is a number.

But for an employee/customer dataset,
150 may be impossible or invalid.

This is called a DATA VALIDATION problem.

Always understand the domain before defining valid ranges.
""")


print("\nPossible invalid ages:")
print(df[df["Age"] > 100])


# Replace unrealistic age with median.
median_age = df.loc[df["Age"] <= 100, "Age"].median()

df.loc[df["Age"] > 100, "Age"] = median_age

print("\nAge after handling invalid value:")
print(df["Age"])


# ====================================================================
# 10. INCONSISTENT CATEGORICAL DATA
# ====================================================================

print("\n" + "=" * 70)
print("10. INCONSISTENT CATEGORICAL DATA")
print("=" * 70)

print("""
Problem:

    Chennai
    chennai
    CHENNAI

These may represent the same city.

ML treats different strings as different categories
until we normalize them.

COMMON SOLUTION:

    df["City"] = df["City"].str.strip().str.title()

strip()
    Removes unnecessary spaces.

lower()
    Converts to lowercase.

upper()
    Converts to uppercase.

title()
    Converts words to title case.
""")

df["City"] = (
    df["City"]
    .str.strip()
    .str.title()
)

print("\nCleaned cities:")
print(df["City"].unique())


# ====================================================================
# 11. CHECK UNIQUE VALUES
# ====================================================================

print("\n" + "=" * 70)
print("11. UNIQUE VALUES")
print("=" * 70)

print("""
Useful functions:

unique()
    Shows unique values.

nunique()
    Counts unique values.

value_counts()
    Counts occurrences of each value.
""")

print("\nUnique cities:")
print(df["City"].unique())

print("\nNumber of cities:")
print(df["City"].nunique())

print("\nCity counts:")
print(df["City"].value_counts())


# ====================================================================
# 12. OUTLIERS
# ====================================================================

print("\n" + "=" * 70)
print("12. OUTLIERS")
print("=" * 70)

print("""
WHAT?
-----
An outlier is an observation unusually far from the majority.

IMPORTANT:
----------
An outlier is not automatically an error.

Possible causes:

    - Data entry error
    - Measurement error
    - Genuine rare event
    - Special customer
    - Fraud
    - High-value transaction

COMMON METHODS:

1. Box plot
2. IQR method
3. Z-score
4. Domain rules


IQR METHOD:

    Q1 = 25th percentile
    Q3 = 75th percentile

    IQR = Q3 - Q1

    Lower = Q1 - 1.5 * IQR
    Upper = Q3 + 1.5 * IQR
""")

Q1 = df["Price"].quantile(0.25)
Q3 = df["Price"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower limit:", lower)
print("Upper limit:", upper)

outliers = df[
    (df["Price"] < lower) |
    (df["Price"] > upper)
]

print("\nPossible price outliers:")
print(outliers)


# ====================================================================
# 13. DATA TRANSFORMATION
# ====================================================================

print("\n" + "=" * 70)
print("13. DATA TRANSFORMATION")
print("=" * 70)

print("""
Data transformation means changing data into a more useful form.

Examples:

    - Convert text
    - Scale values
    - Create categories
    - Extract date components
    - Log-transform skewed data
    - Create ratios

In ML, transformation is often required before modeling.
""")


# ====================================================================
# 14. FEATURE ENGINEERING
# ====================================================================

print("\n" + "=" * 70)
print("14. FEATURE ENGINEERING")
print("=" * 70)

print("""
WHAT?
-----
Feature engineering means creating useful input variables
from existing data.

WHY?
----
Good features can help an ML model learn useful patterns.

Example:

    Quantity
    Price

can become:

    Total_Sales = Quantity × Price
""")

df["Total_Sales"] = (
    df["Quantity"] * df["Price"]
)

print("\nNew feature:")
print(
    df[
        [
            "Product",
            "Quantity",
            "Price",
            "Total_Sales"
        ]
    ]
)


# ====================================================================
# 15. BASIC EDA
# ====================================================================

print("\n" + "=" * 70)
print("15. COMPLETE EDA — STEP 1")
print("=" * 70)

print("""
EDA means Exploratory Data Analysis.

The purpose is to understand the dataset before modeling.

QUESTIONS:

    What does the dataset contain?
    How large is it?
    What are the distributions?
    Are values missing?
    Are there outliers?
    Which categories dominate?
    Which variables are related?
""")


print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nSummary:")
print(df.describe())


# ====================================================================
# 16. UNIVARIATE ANALYSIS
# ====================================================================

print("\n" + "=" * 70)
print("16. UNIVARIATE ANALYSIS")
print("=" * 70)

print("""
Univariate = analyzing ONE variable.

Examples:

    Age distribution
    Price distribution
    Rating distribution

Common tools:

    mean()
    median()
    min()
    max()
    std()
    histogram
    boxplot
""")

print("\nAge statistics:")
print(df["Age"].describe())

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Age",
    bins=6,
    kde=True
)

plt.title("Age Distribution")

plt.show()


# ====================================================================
# 17. CATEGORICAL ANALYSIS
# ====================================================================

print("\n" + "=" * 70)
print("17. CATEGORICAL ANALYSIS")
print("=" * 70)

print("""
For categorical variables, use:

    value_counts()
    countplot()
    barplot()

Example:
How many customers are from each city?
""")

print(df["City"].value_counts())

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="City"
)

plt.title("Customers by City")

plt.show()


# ====================================================================
# 18. BIVARIATE ANALYSIS
# ====================================================================

print("\n" + "=" * 70)
print("18. BIVARIATE ANALYSIS")
print("=" * 70)

print("""
Bivariate = analyzing TWO variables.

Examples:

    Age vs Total Sales
    Quantity vs Total Sales
    City vs Sales

For numerical + numerical:

    scatter plot

For categorical + numerical:

    barplot / boxplot
""")

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Age",
    y="Total_Sales"
)

plt.title("Age vs Total Sales")

plt.show()


# ====================================================================
# 19. MULTIVARIATE ANALYSIS
# ====================================================================

print("\n" + "=" * 70)
print("19. MULTIVARIATE ANALYSIS")
print("=" * 70)

print("""
Multivariate analysis means studying multiple variables together.

Example:

    Age
    Quantity
    Price
    Rating
    Total Sales

Heatmaps and pairplots are useful.
""")

numeric_columns = [
    "Age",
    "Quantity",
    "Price",
    "Rating",
    "Total_Sales"
]

correlation = df[numeric_columns].corr()

print("\nCorrelation matrix:")
print(correlation)

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()


# ====================================================================
# 20. GROUPBY ANALYSIS
# ====================================================================

print("\n" + "=" * 70)
print("20. GROUPBY ANALYSIS")
print("=" * 70)

print("""
groupby() is extremely important in EDA.

It is similar to SQL GROUP BY.

Example:

    Total sales by city
    Average rating by product
    Average price by category
""")

print("\nTotal sales by city:")

print(
    df.groupby("City")["Total_Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nAverage rating by product:")

print(
    df.groupby("Product")["Rating"]
    .mean()
    .sort_values(ascending=False)
)


# ====================================================================
# 21. VISUAL EDA
# ====================================================================

print("\n" + "=" * 70)
print("21. VISUAL EDA")
print("=" * 70)

print("""
A good EDA combines:

    Statistics
       +
    Visualization

IMPORTANT PLOTS:

1. Histogram
   -> distribution

2. Boxplot
   -> spread and outliers

3. Countplot
   -> category frequency

4. Barplot
   -> category comparison

5. Scatterplot
   -> relationship

6. Heatmap
   -> correlation
""")


# ====================================================================
# 22. DATA CLEANING CHECKLIST
# ====================================================================

print("\n" + "=" * 70)
print("22. DATA CLEANING CHECKLIST")
print("=" * 70)

print("""
Before ML, check:

[ ] Missing values
[ ] Duplicate rows
[ ] Duplicate IDs
[ ] Incorrect data types
[ ] Invalid values
[ ] Inconsistent text
[ ] Outliers
[ ] Incorrect units
[ ] Impossible values
[ ] Incorrect labels
[ ] Data leakage
[ ] Target column problems

IMPORTANT:

Cleaning is NOT simply:

    "Delete everything unusual."

Cleaning means understanding the data and making
justified decisions.
""")


# ====================================================================
# 23. DATA LEAKAGE
# ====================================================================

print("\n" + "=" * 70)
print("23. DATA LEAKAGE")
print("=" * 70)

print("""
WHAT?
-----
Data leakage happens when information that should not be available
during prediction accidentally enters model training.

Example:

You want to predict whether a customer will purchase.

But you create a feature using:

    customer's future purchase amount

That information would not exist at prediction time.

The model may appear extremely accurate,
but the evaluation is invalid.

IMPORTANT RULE:

Only use information that would genuinely be available
at prediction time.

Data leakage is one of the most important ML data-preparation
concepts.
""")


# ====================================================================
# 24. TRAIN / TEST SPLIT CONCEPT
# ====================================================================

print("\n" + "=" * 70)
print("24. TRAIN / TEST SPLIT")
print("=" * 70)

print("""
After EDA and preprocessing, ML data is usually divided.

TRAINING DATA
-------------
Used to learn patterns.

TEST DATA
---------
Used to evaluate performance on unseen data.

Typical idea:

    Dataset
       |
       +---- Training
       |
       +---- Testing


IMPORTANT:

Do not allow information from the test set to influence
training decisions.

This is why many preprocessing steps are fitted only
on training data.

Example:

    Scaling
    Imputation
    Feature selection
""")

print("""
Example using scikit-learn:

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
""")


# ====================================================================
# 25. ML DATA PIPELINE
# ====================================================================

print("\n" + "=" * 70)
print("25. COMPLETE DATA SCIENCE / ML PIPELINE")
print("=" * 70)

print("""
RAW DATA
   ↓
Load with Pandas
   ↓
Understand data
   ↓
Check missing values
   ↓
Check duplicates
   ↓
Check data types
   ↓
Check invalid values
   ↓
Handle outliers
   ↓
Normalize categories
   ↓
Feature engineering
   ↓
EDA
   ↓
Train/Test Split
   ↓
Preprocessing
   ↓
Machine Learning
   ↓
Evaluation
   ↓
Visualization
   ↓
Deployment


IMPORTANT:

EDA helps you understand the data.

Data cleaning makes the data usable.

Feature engineering creates useful representations.

ML uses the prepared data to learn patterns.
""")


# ====================================================================
# 26. COMMON MISTAKES
# ====================================================================

print("\n" + "=" * 70)
print("26. COMMON DATA CLEANING / EDA MISTAKES")
print("=" * 70)

print("""
1. Removing every missing row
--------------------------------
You may lose too much useful data.

2. Filling every missing value with zero
----------------------------------------
Zero may have a completely different meaning.

3. Using mean blindly
----------------------
Median may be more appropriate for skewed data.

4. Removing every outlier
-------------------------
Some outliers are genuine.

5. Ignoring duplicates
----------------------
Duplicates can distort statistics and ML training.

6. Ignoring inconsistent categories
------------------------------------
"Chennai", "chennai" and "CHENNAI" may become
different categories.

7. Looking only at averages
---------------------------
Distributions and outliers also matter.

8. Correlation = causation
--------------------------
It does not.

9. Data leakage
---------------
Never use future/unavailable information.

10. Cleaning test data using information from the test set
-----------------------------------------------------------
Preprocessing decisions should generally be learned from
training data and then applied to test data.

11. Building ML before understanding the data
---------------------------------------------
Always inspect the dataset first.

12. Making conclusions from one chart
--------------------------------------
Combine statistics and multiple relevant visualizations.
""")


# ====================================================================
# 27. INTERVIEW QUESTIONS
# ====================================================================

print("\n" + "=" * 70)
print("27. INTERVIEW QUESTIONS + ANSWERS")
print("=" * 70)

print("""
Q1. What is Jupyter Notebook?
-----------------------------
A:
Jupyter Notebook is an interactive environment where code,
output, explanations and visualizations can be organized
in cells.

---------------------------------------------------------------

Q2. Why is Jupyter popular in Data Science?
--------------------------------------------
A:
Because it supports interactive experimentation, visualization,
documentation and step-by-step analysis.

---------------------------------------------------------------

Q3. What is EDA?
----------------
A:
Exploratory Data Analysis is the process of understanding
data using statistics, summaries and visualizations.

---------------------------------------------------------------

Q4. What is data cleaning?
---------------------------
A:
Data cleaning is the process of identifying and correcting
missing, duplicate, inconsistent, invalid or incorrect data.

---------------------------------------------------------------

Q5. How do you find missing values in Pandas?
----------------------------------------------
A:
Use:

    df.isna().sum()

---------------------------------------------------------------

Q6. How can missing values be handled?
---------------------------------------
A:
They can be removed or imputed using methods such as mean,
median, mode or domain-specific values.

---------------------------------------------------------------

Q7. Mean vs Median for missing values?
---------------------------------------
A:
Mean can work well for relatively symmetric data.
Median is often more robust when data is skewed or contains
outliers.

---------------------------------------------------------------

Q8. How do you find duplicates?
--------------------------------
A:
Use:

    df.duplicated()

and count them with:

    df.duplicated().sum()

---------------------------------------------------------------

Q9. How do you remove duplicates?
----------------------------------
A:
Use:

    df.drop_duplicates()

---------------------------------------------------------------

Q10. What is an outlier?
-------------------------
A:
An observation that is unusually far from the majority
of observations.

---------------------------------------------------------------

Q11. Should outliers always be removed?
---------------------------------------
A:
No. They should first be investigated because they may be
valid observations.

---------------------------------------------------------------

Q12. What is IQR?
------------------
A:
IQR is the Interquartile Range:

    IQR = Q3 - Q1

It measures the spread of the middle 50% of the data.

---------------------------------------------------------------

Q13. What is feature engineering?
----------------------------------
A:
Creating useful features from existing data to improve
representation for analysis or machine learning.

---------------------------------------------------------------

Q14. What is data leakage?
---------------------------
A:
Data leakage occurs when information unavailable at prediction
time accidentally influences model training.

---------------------------------------------------------------

Q15. Why is data leakage dangerous?
------------------------------------
A:
It can make evaluation look unrealistically good while the
model performs poorly on genuinely unseen situations.

---------------------------------------------------------------

Q16. What is univariate analysis?
---------------------------------
A:
Analysis of one variable.

Example:
Studying salary distribution.

---------------------------------------------------------------

Q17. What is bivariate analysis?
--------------------------------
A:
Analysis involving two variables.

Example:
Experience vs Salary.

---------------------------------------------------------------

Q18. What is multivariate analysis?
-----------------------------------
A:
Analysis involving multiple variables simultaneously.

---------------------------------------------------------------

Q19. What is train-test split?
------------------------------
A:
It separates data into training data used to learn and test
data used to evaluate generalization on unseen data.

---------------------------------------------------------------

Q20. Why should preprocessing avoid test-set leakage?
------------------------------------------------------
A:
Because using information from the test set during training
or preprocessing decisions can make evaluation overly optimistic.

---------------------------------------------------------------

Q21. What is the purpose of EDA before ML?
-------------------------------------------
A:
To understand distributions, missing data, outliers,
relationships, class balance and possible data-quality problems.

---------------------------------------------------------------

Q22. What is the difference between cleaning and EDA?
------------------------------------------------------
A:
Cleaning focuses on making data reliable and usable.
EDA focuses on understanding patterns and characteristics
of the data. They often overlap and are iterative.

---------------------------------------------------------------

Q23. Why use groupby() during EDA?
----------------------------------
A:
It allows analysis of metrics across categories, such as
average sales by city or average salary by department.

---------------------------------------------------------------

Q24. Why normalize categorical values?
--------------------------------------
A:
Different spellings or capitalization can represent the same
category and otherwise create incorrect separate groups.

---------------------------------------------------------------

Q25. What is the difference between df.info() and df.describe()?
-----------------------------------------------------------------
A:
info() provides structure, data types and non-null counts.
describe() provides descriptive statistics for numerical data
by default.
""")


# ====================================================================
# 28. PRACTICE EXERCISES
# ====================================================================

print("\n" + "=" * 70)
print("28. PRACTICE EXERCISES")
print("=" * 70)

print("""
LEVEL 1 — PANDAS
-----------------

1. Find the number of rows and columns.

2. Find all missing values.

3. Find unique cities.

4. Count customers in each city.

5. Find average price.

6. Find median age.

7. Find the highest rating.

8. Find the cheapest product.


LEVEL 2 — DATA CLEANING
-----------------------

9. Find duplicate Customer_ID values.

10. Normalize all city names.

11. Find invalid ages.

12. Replace invalid ages with the median.

13. Create:

       Total_Sales = Quantity * Price

14. Check whether Total_Sales contains missing values.


LEVEL 3 — EDA
-------------

15. Plot age distribution.

16. Plot price distribution.

17. Create a boxplot for Price.

18. Create a countplot for City.

19. Create a barplot of average sales by city.

20. Create a scatter plot of Quantity vs Total_Sales.

21. Create a heatmap of numerical columns.

22. Write 5 observations from your EDA.


LEVEL 4 — ML PREPARATION
------------------------

23. Decide which column should be the target if the goal
    is to predict Total_Sales.

24. Identify numerical features.

25. Identify categorical features.

26. Identify columns that should NOT be used as features.

27. Explain possible sources of data leakage.

28. Perform a train/test split using scikit-learn.

29. Explain which preprocessing steps should be fitted
    only on training data.
""")


# ====================================================================
# 29. COMPLETE EDA PROJECT STRUCTURE
# ====================================================================

print("\n" + "=" * 70)
print("29. YOUR REAL JUPYTER PROJECT STRUCTURE")
print("=" * 70)

print("""
When you work with a real dataset, your Jupyter Notebook
can follow this structure:

==============================================================
1. PROJECT TITLE
==============================================================

# Customer Sales Analysis

Business question:
    What factors are associated with customer sales?

==============================================================
2. IMPORT LIBRARIES
==============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

==============================================================
3. LOAD DATA
==============================================================

df = pd.read_csv("sales.csv")

==============================================================
4. UNDERSTAND DATA
==============================================================

df.head()
df.shape
df.info()
df.describe()

==============================================================
5. DATA QUALITY
==============================================================

df.isna().sum()
df.duplicated().sum()
df.dtypes

==============================================================
6. CLEAN DATA
==============================================================

Handle:
    - missing values
    - duplicates
    - invalid values
    - inconsistent categories
    - incorrect types

==============================================================
7. FEATURE ENGINEERING
==============================================================

Create useful variables.

Example:

df["Total_Sales"] = (
    df["Quantity"] * df["Price"]
)

==============================================================
8. UNIVARIATE EDA
==============================================================

Histograms
Boxplots
Value counts

==============================================================
9. BIVARIATE EDA
==============================================================

Scatter plots
Bar plots
Box plots

==============================================================
10. MULTIVARIATE EDA
==============================================================

Correlation heatmap
Pairplot

==============================================================
11. FIND INSIGHTS
==============================================================

Write observations.

==============================================================
12. PREPARE FOR ML
==============================================================

Separate:

X = features
y = target

Then:

train_test_split()

==============================================================
13. SAVE CLEAN DATA
==============================================================

df.to_csv(
    "cleaned_data.csv",
    index=False
)

==============================================================
""")


# ====================================================================
# 30. FINAL MASTERY CHECKLIST
# ====================================================================

print("\n" + "=" * 70)
print("30. MASTERY CHECKLIST")
print("=" * 70)

"""
JUPYTER
-------
[ ] What is Jupyter?
[ ] Code cells
[ ] Markdown cells
[ ] Running cells
[ ] Notebook workflow


PANDAS INSPECTION
-----------------
[ ] head()
[ ] tail()
[ ] shape
[ ] columns
[ ] dtypes
[ ] info()
[ ] describe()


DATA CLEANING
-------------
[ ] Missing values
[ ] fillna()
[ ] dropna()
[ ] Duplicates
[ ] drop_duplicates()
[ ] Data types
[ ] Invalid values
[ ] String normalization
[ ] Outliers
[ ] IQR


EDA
---
[ ] Univariate analysis
[ ] Bivariate analysis
[ ] Multivariate analysis
[ ] groupby()
[ ] value_counts()
[ ] Distribution
[ ] Correlation
[ ] Outliers
[ ] Visualization


FEATURE ENGINEERING
-------------------
[ ] Create new features
[ ] Understand useful features
[ ] Avoid leakage


ML PREPARATION
--------------
[ ] Features X
[ ] Target y
[ ] Train/test split
[ ] Avoid test leakage
[ ] Numerical features
[ ] Categorical features


MOST IMPORTANT IDEA
-------------------

Do NOT think:

    "Data cleaning = deleting bad data."

Think:

    "Understand the data
     → identify problems
     → make justified decisions
     → verify the result."


COMPLETE FLOW:

Jupyter
   ↓
Pandas
   ↓
Load Dataset
   ↓
Inspect
   ↓
Clean
   ↓
EDA
   ↓
Visualize
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Preprocessing
   ↓
Machine Learning

======================================================================
END
======================================================================

NEXT MAJOR PHASE:

MATHEMATICS FOR MACHINE LEARNING

1. Linear Algebra
2. Statistics
3. Probability
4. Calculus Basics
5. Optimization

Then:

CLASSICAL MACHINE LEARNING
    ↓
Supervised Learning
Unsupervised Learning
Regression
Classification
Decision Trees
Random Forest
XGBoost
Clustering
Dimensionality Reduction
Evaluation
Feature Engineering
Pipelines
======================================================================
"""
