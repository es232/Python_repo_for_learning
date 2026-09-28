


"""
PANDAS - QUICK LEARNING MODULE
==============================

Real-world example:
E-Commerce Sales Data

How to run:
1. Install Pandas:
       python -m pip install pandas

2. Run:
       python pandas_learning.py

Windows alternative:
       py pandas_learning.py

This file covers the Pandas concepts commonly used in
Data Science, Machine Learning and data analysis.
"""

import pandas as pd
import numpy as np


# ============================================================
# HELPER
# ============================================================

def section(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


# ============================================================
# 1. WHAT IS PANDAS?
# ============================================================

def basics():
    section("1. WHAT IS PANDAS?")

    print("""
Pandas is a Python library used for:

- Data manipulation
- Data cleaning
- Data analysis
- Working with CSV/Excel/SQL data
- Preparing data for Machine Learning

Two important Pandas objects:

1. Series     → one-dimensional data
2. DataFrame  → two-dimensional table
""")

    # Series
    prices = pd.Series([100, 200, 300])

    print("Example Series:")
    print(prices)

    # DataFrame
    data = {
        "Product": ["Laptop", "Phone", "Mouse"],
        "Price": [60000, 30000, 1000]
    }

    df = pd.DataFrame(data)

    print("\nExample DataFrame:")
    print(df)


# ============================================================
# REAL-WORLD DATASET
# ============================================================

def create_dataset():
    """
    Creates a realistic e-commerce sales dataset.

    In real projects, this data would normally come from:
    CSV / Excel / Database / API.
    """

    data = {
        "Order_ID": [
            "ORD001", "ORD002", "ORD003", "ORD004", "ORD005",
            "ORD006", "ORD007", "ORD008", "ORD009", "ORD010",
            "ORD011", "ORD012"
        ],

        "Customer": [
            "Arun", "Priya", "Rahul", "Meena", "Kavin",
            "Divya", "Arun", "Priya", "Vijay", "Meena",
            "Kavin", "Divya"
        ],

        "Product": [
            "Laptop", "Phone", "Mouse", "Keyboard", "Laptop",
            "Phone", "Headphones", "Mouse", "Laptop", "Keyboard",
            "Phone", "Headphones"
        ],

        "Category": [
            "Electronics", "Electronics", "Accessories", "Accessories",
            "Electronics", "Electronics", "Accessories", "Accessories",
            "Electronics", "Accessories", "Electronics", "Accessories"
        ],

        "City": [
            "Chennai", "Madurai", "Chennai", "Coimbatore",
            "Chennai", "Madurai", "Trichy", "Chennai",
            "Coimbatore", "Madurai", "Chennai", "Trichy"
        ],

        "Quantity": [
            1, 2, 3, 2, 1,
            1, 2, 4, 1, 2,
            1, 3
        ],

        "Price": [
            60000, 30000, 1000, 2500, 60000,
            30000, 3000, 1000, 60000, 2500,
            30000, 3000
        ],

        "Rating": [
            4.5, 4.2, 4.0, 3.8, 4.7,
            4.1, np.nan, 4.3, 4.8, 3.9,
            4.4, np.nan
        ]
    }

    return pd.DataFrame(data)


# ============================================================
# 2. CREATING A DATAFRAME
# ============================================================

def create_dataframe():
    section("2. CREATING A DATAFRAME")

    df = create_dataset()

    print("E-Commerce Dataset:")
    print(df)

    print("""
A DataFrame is basically a table
with rows and columns.
""")

    return df


# ============================================================
# 3. LOADING DATA
# ============================================================

def loading_data():
    section("3. LOADING DATA")

    print("""
In real projects, data commonly comes from:

CSV:
    pd.read_csv("sales.csv")

Excel:
    pd.read_excel("sales.xlsx")

SQL:
    pd.read_sql(query, connection)

JSON:
    pd.read_json("data.json")

For this tutorial we create the data directly.
""")


# ============================================================
# 4. INSPECTING DATA
# ============================================================

def inspect_data(df):
    section("4. INSPECTING DATA")

    print("First 5 rows:")
    print(df.head())

    print("\nLast 5 rows:")
    print(df.tail())

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns)

    print("\nData types:")
    print(df.dtypes)

    print("\nBasic information:")
    df.info()

    print("\nStatistical summary:")
    print(df.describe())


# ============================================================
# 5. SELECTING COLUMNS
# ============================================================

def selecting_columns(df):
    section("5. SELECTING COLUMNS")

    print("Product column:")
    print(df["Product"])

    print("\nMultiple columns:")
    print(df[["Product", "Quantity", "Price"]])

    print("""
Remember:

df["Column"]          → one column
df[["A", "B"]]        → multiple columns
""")


# ============================================================
# 6. LOC AND ILOC
# ============================================================

def loc_iloc(df):
    section("6. LOC AND ILOC")

    print("First row using iloc:")
    print(df.iloc[0])

    print("\nFirst 3 rows:")
    print(df.iloc[0:3])

    print("\nSpecific rows and columns:")
    print(df.iloc[0:3, 0:4])

    print("""
iloc → position based

loc → label/condition based
""")

    print("Orders from Chennai using loc:")
    print(df.loc[df["City"] == "Chennai"])


# ============================================================
# 7. FILTERING
# ============================================================

def filtering(df):
    section("7. FILTERING DATA")

    print("Orders where price > 10000:")
    print(df[df["Price"] > 10000])

    print("\nOrders from Chennai:")
    print(df[df["City"] == "Chennai"])

    print("\nElectronics products:")
    print(df[df["Category"] == "Electronics"])

    print("\nPrice > 10000 AND quantity >= 1:")
    print(
        df[
            (df["Price"] > 10000) &
            (df["Quantity"] >= 1)
        ]
    )

    print("""
Important:

Use & for AND
Use | for OR

Each condition should be inside parentheses.
""")


# ============================================================
# 8. CREATING NEW COLUMNS
# ============================================================

def creating_columns(df):
    section("8. CREATING NEW COLUMNS")

    df = df.copy()

    # Total sales for each order
    df["Total_Sales"] = df["Quantity"] * df["Price"]

    print(df[[
        "Product",
        "Quantity",
        "Price",
        "Total_Sales"
    ]])

    print("""
This is one of the most common Pandas operations:

df["new_column"] = calculation
""")

    return df


# ============================================================
# 9. SORTING
# ============================================================

def sorting(df):
    section("9. SORTING")

    print("Highest sales first:")

    sorted_df = df.sort_values(
        by="Total_Sales",
        ascending=False
    )

    print(
        sorted_df[
            ["Product", "City", "Total_Sales"]
        ]
    )

    print("""
sort_values() is commonly used to
find highest/lowest records.
""")


# ============================================================
# 10. MISSING VALUES
# ============================================================

def missing_values(df):
    section("10. MISSING VALUES")

    print("Missing values:")
    print(df.isna().sum())

    print("""
Common functions:

isna()       → find missing values
notna()      → find non-missing values
fillna()     → replace missing values
dropna()     → remove rows containing missing values
""")

    # Fill missing ratings with average rating
    df = df.copy()

    average_rating = df["Rating"].mean()

    df["Rating"] = df["Rating"].fillna(average_rating)

    print("After filling missing ratings:")
    print(df[["Product", "Rating"]])

    return df


# ============================================================
# 11. DUPLICATES
# ============================================================

def duplicates(df):
    section("11. DUPLICATE DATA")

    print("Number of duplicate rows:")
    print(df.duplicated().sum())

    print("""
Common functions:

duplicated()       → find duplicates
drop_duplicates()  → remove duplicates
""")

    df = df.drop_duplicates()

    print("\nAfter removing duplicates:")
    print("Rows:", len(df))


# ============================================================
# 12. DATA TYPES
# ============================================================

def data_types(df):
    section("12. DATA TYPES")

    print(df.dtypes)

    print("""
Common data types:

int
float
string/object
bool
datetime

astype() can be used to change a data type.
""")

    df = df.copy()

    df["Quantity"] = df["Quantity"].astype(int)

    print("\nQuantity type after conversion:")
    print(df["Quantity"].dtype)


# ============================================================
# 13. GROUPBY
# ============================================================

def groupby_example(df):
    section("13. GROUPBY")

    print("Total sales by city:")

    city_sales = df.groupby("City")["Total_Sales"].sum()

    print(city_sales)

    print("\nAverage rating by category:")

    category_rating = df.groupby("Category")["Rating"].mean()

    print(category_rating)

    print("""
groupby() is one of the MOST IMPORTANT
Pandas functions for Data Analysis.

Think:

GROUP BY in SQL
        ↓
groupby() in Pandas
""")


# ============================================================
# 14. GROUPBY + AGG
# ============================================================

def aggregation(df):
    section("14. GROUPBY + AGG")

    result = df.groupby("Category").agg(
        Total_Sales=("Total_Sales", "sum"),
        Average_Price=("Price", "mean"),
        Total_Quantity=("Quantity", "sum"),
        Average_Rating=("Rating", "mean")
    )

    print(result)

    print("""
agg() allows us to perform multiple
calculations at the same time.
""")


# ============================================================
# 15. VALUE_COUNTS
# ============================================================

def value_counts(df):
    section("15. VALUE_COUNTS")

    print("Number of orders by city:")
    print(df["City"].value_counts())

    print("\nNumber of orders by category:")
    print(df["Category"].value_counts())

    print("""
value_counts() tells us how frequently
each unique value appears.
""")


# ============================================================
# 16. UNIQUE VALUES
# ============================================================

def unique_values(df):
    section("16. UNIQUE VALUES")

    print("Unique cities:")
    print(df["City"].unique())

    print("\nNumber of unique cities:")
    print(df["City"].nunique())

    print("""
unique()  → actual unique values
nunique() → number of unique values
""")


# ============================================================
# 17. APPLY
# ============================================================

def apply_example(df):
    section("17. APPLY")

    df = df.copy()

    def price_category(price):
        if price >= 30000:
            return "Expensive"
        elif price >= 5000:
            return "Medium"
        else:
            return "Affordable"

    df["Price_Category"] = df["Price"].apply(price_category)

    print(
        df[
            ["Product", "Price", "Price_Category"]
        ]
    )

    print("""
apply() allows us to apply a function
to values in a Series.

Use it when a simple vectorized operation
is not enough.
""")


# ============================================================
# 18. STRING OPERATIONS
# ============================================================

def string_operations(df):
    section("18. STRING OPERATIONS")

    print("Products in uppercase:")
    print(df["Product"].str.upper())

    print("\nCities in lowercase:")
    print(df["City"].str.lower())

    print("""
Common string operations:

.str.upper()
.str.lower()
.str.contains()
.str.replace()
.str.strip()
""")


# ============================================================
# 19. MERGE
# ============================================================

def merge_example():
    section("19. MERGING DATAFRAMES")

    customers = pd.DataFrame({
        "Customer": ["Arun", "Priya", "Rahul"],
        "Membership": ["Gold", "Silver", "Gold"]
    })

    orders = pd.DataFrame({
        "Customer": ["Arun", "Priya", "Rahul"],
        "Orders": [5, 3, 7]
    })

    result = pd.merge(
        orders,
        customers,
        on="Customer",
        how="inner"
    )

    print("Customer data:")
    print(customers)

    print("\nOrder data:")
    print(orders)

    print("\nMerged data:")
    print(result)

    print("""
merge() is similar to SQL JOIN.

Common types:

inner
left
right
outer
""")


# ============================================================
# 20. SAVE DATA
# ============================================================

def save_data(df):
    section("20. SAVING DATA")

    # Save processed data
    df.to_csv(
        "processed_sales.csv",
        index=False
    )

    print("""
Data saved as:

processed_sales.csv

Common:

df.to_csv()
df.to_excel()
""")


# ============================================================
# 21. REAL-WORLD MINI PROJECT
# ============================================================

def sales_analysis_project(df):
    section("21. REAL-WORLD PROJECT - SALES ANALYSIS")

    print("""
Business Question:

An e-commerce company wants to understand
its sales performance.

We need to find:

1. Total revenue
2. Best-selling product
3. Best city
4. Best category
5. Average order value
6. Number of orders
7. Top 3 orders
""")

    # 1. Total revenue
    total_revenue = df["Total_Sales"].sum()

    print("\n1. Total Revenue:")
    print(total_revenue)

    # 2. Best-selling product by revenue
    product_sales = (
        df.groupby("Product")["Total_Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\n2. Sales by Product:")
    print(product_sales)

    # 3. Best city
    city_sales = (
        df.groupby("City")["Total_Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\n3. Sales by City:")
    print(city_sales)

    # 4. Best category
    category_sales = (
        df.groupby("Category")["Total_Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\n4. Sales by Category:")
    print(category_sales)

    # 5. Average order value
    average_order = df["Total_Sales"].mean()

    print("\n5. Average Order Value:")
    print(round(average_order, 2))

    # 6. Number of orders
    number_of_orders = df["Order_ID"].nunique()

    print("\n6. Number of Orders:")
    print(number_of_orders)

    # 7. Top 3 orders
    top_orders = df.nlargest(
        3,
        "Total_Sales"
    )

    print("\n7. Top 3 Orders:")
    print(
        top_orders[
            [
                "Order_ID",
                "Product",
                "City",
                "Total_Sales"
            ]
        ]
    )


# ============================================================
# 22. IMPORTANT FUNCTIONS
# ============================================================

def important_functions():
    section("22. IMPORTANT PANDAS FUNCTIONS")

    print("""
DATA CREATION / LOADING
-----------------------
pd.DataFrame()
pd.Series()
pd.read_csv()
pd.read_excel()

INSPECTION
----------
df.head()
df.tail()
df.shape
df.columns
df.dtypes
df.info()
df.describe()

SELECTION
---------
df["column"]
df[["col1", "col2"]]
df.loc[]
df.iloc[]

FILTERING
---------
df[condition]

CLEANING
--------
df.isna()
df.fillna()
df.dropna()
df.duplicated()
df.drop_duplicates()
df.astype()

ANALYSIS
--------
df.sort_values()
df.groupby()
df.agg()
df.value_counts()
df.unique()
df.nunique()

TRANSFORMATION
--------------
df["new_column"] = ...
df.apply()

COMBINING DATA
--------------
pd.merge()

OUTPUT
------
df.to_csv()
df.to_excel()
""")


# ============================================================
# 23. INTERVIEW QUESTIONS
# ============================================================

def interview_questions():
    section("23. PANDAS INTERVIEW QUESTIONS")

    questions = [
        "1. What is Pandas?",
        "2. Difference between Series and DataFrame?",
        "3. How do you read a CSV file?",
        "4. How do you inspect a DataFrame?",
        "5. Difference between loc and iloc?",
        "6. How do you filter rows?",
        "7. How do you handle missing values?",
        "8. Difference between dropna() and fillna()?",
        "9. What is groupby()?",
        "10. What is agg()?",
        "11. How do you remove duplicates?",
        "12. How do you change a column's data type?",
        "13. What is merge()?",
        "14. Difference between merge() and concat()?",
        "15. How do you create a new column?",
        "16. When would you use apply()?",
        "17. How do you find unique values?",
        "18. How do you sort a DataFrame?"
    ]

    for question in questions:
        print(question)


# ============================================================
# 24. PRACTICE
# ============================================================

def practice():
    section("24. YOUR PRACTICE")

    print("""
Try these yourself using the sales DataFrame:

1. Find all orders from Chennai.

2. Find all products where Price > 10,000.

3. Find the total quantity sold for each product.

4. Find the average price for each category.

5. Find the city with the highest total sales.

6. Find the top 3 most expensive products.

7. Find all rows where Rating is missing.

8. Replace missing ratings with the average rating.

9. Create a new column:
       Total_Sales = Quantity × Price

10. Find how many unique customers exist.

11. Save the cleaned dataset as:
       cleaned_sales.csv
""")


# ============================================================
# MAIN
# ============================================================

def main():

    print("""
=================================================================
                    PANDAS QUICK COURSE
=================================================================

Real-world example:
E-Commerce Sales Data

Goal:
Learn the Pandas concepts commonly required for
Data Science and Machine Learning.

After this → Matplotlib + Seaborn
""")

    basics()

    df = create_dataframe()

    loading_data()

    inspect_data(df)

    selecting_columns(df)

    loc_iloc(df)

    filtering(df)

    df = creating_columns(df)

    sorting(df)

    df = missing_values(df)

    duplicates(df)

    data_types(df)

    groupby_example(df)

    aggregation(df)

    value_counts(df)

    unique_values(df)

    apply_example(df)

    string_operations(df)

    merge_example()

    save_data(df)

    sales_analysis_project(df)

    important_functions()

    interview_questions()

    practice()

    section("PANDAS COMPLETE")

    print("""
You should now understand:

✓ Series
✓ DataFrame
✓ Reading data
✓ head / tail / info / describe
✓ Selecting columns
✓ loc / iloc
✓ Filtering
✓ Creating columns
✓ Sorting
✓ Missing values
✓ Duplicates
✓ Data types
✓ groupby
✓ agg
✓ value_counts
✓ unique / nunique
✓ apply
✓ String operations
✓ merge
✓ Saving data
✓ Real-world sales analysis

Next:
                MATPLOTLIB + SEABORN
""")


if __name__ == "__main__":
    main()