"""
===========================================================
          MATPLOTLIB + SEABORN - ESSENTIALS
===========================================================

WHAT?
Matplotlib → Python library for creating visualizations.
Seaborn    → Higher-level visualization library built
              on top of Matplotlib.

WHY?
Visualization helps us:
- Understand data 
- Find patterns
- Find outliers
- Compare values
- Understand relationships
- Perform EDA (Exploratory Data Analysis)

WHEN?
Use them after loading and cleaning data with Pandas.

WHERE?
Commonly used in:
- Data Science
- Machine Learning
- EDA
- Data Analysis
- Reports and dashboards

INSTALL:
    python -m pip install matplotlib seaborn pandas

RUN:
    python matplotlib_seaborn_learning.py
===========================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ===========================================================
# 1. MATPLOTLIB VS SEABORN
# ===========================================================

print("=" * 60)
print("1. MATPLOTLIB VS SEABORN")
print("=" * 60)

print("""
Matplotlib
-----------
More control over plots.
Good for custom visualizations.

Seaborn
-------
Built on Matplotlib.
Makes statistical plots easier and usually looks cleaner.

Simple rule:

Matplotlib → control
Seaborn    → easy statistical visualization

Both can be used together.
""")


# ===========================================================
# 2. DATASET
# ===========================================================

print("=" * 60)
print("2. SAMPLE DATA")
print("=" * 60)

data = {
    "Product": ["Laptop", "Phone", "Tablet", "Laptop",
                "Phone", "Tablet", "Laptop", "Phone"],
    "Category": ["Electronics", "Electronics", "Electronics",
                 "Electronics", "Electronics", "Electronics",
                 "Electronics", "Electronics"],
    "City": ["Chennai", "Madurai", "Chennai", "Trichy",
             "Chennai", "Madurai", "Trichy", "Chennai"],
    "Sales": [75000, 30000, 25000, 68000,
              32000, 22000, 72000, 35000],
    "Quantity": [2, 3, 2, 1, 4, 2, 2, 5],
    "Rating": [4.5, 4.2, 4.0, 4.7, 4.3, 3.9, 4.8, 4.1]
}

df = pd.DataFrame(data)

print(df)


# ===========================================================
# 3. LINE PLOT
# ===========================================================

print("=" * 60)
print("3. LINE PLOT")
print("=" * 60)

print("""
Used for:
- Trends
- Changes over time
- Continuous data

Example:
Sales changing across orders.
""")

plt.plot(df.index, df["Sales"], marker="o")
plt.title("Sales Trend")
plt.xlabel("Order")
plt.ylabel("Sales")
plt.grid()
plt.show()


# ===========================================================
# 4. BAR PLOT
# ===========================================================

print("=" * 60)
print("4. BAR PLOT")
print("=" * 60)

print("""
Used to compare categories.

Example:
Sales by product.
""")

product_sales = df.groupby("Product")["Sales"].sum()

plt.bar(product_sales.index, product_sales.values)
plt.title("Sales by Product")
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.show()


# ===========================================================
# 5. SCATTER PLOT
# ===========================================================

print("=" * 60)
print("5. SCATTER PLOT")
print("=" * 60)

print("""
Used to understand the relationship between
two numerical variables.

Example:
Quantity vs Sales.
""")

plt.scatter(df["Quantity"], df["Sales"])
plt.title("Quantity vs Sales")
plt.xlabel("Quantity")
plt.ylabel("Sales")
plt.show()


# ===========================================================
# 6. HISTOGRAM
# ===========================================================

print("=" * 60)
print("6. HISTOGRAM")
print("=" * 60)

print("""
Used to understand the distribution of numerical data.

Example:
Distribution of sales.
""")

plt.hist(df["Sales"], bins=5)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()


# ===========================================================
# 7. BOX PLOT
# ===========================================================

print("=" * 60)
print("7. BOX PLOT")
print("=" * 60)

print("""
Used to understand:
- Median
- Spread
- Outliers

Very useful during data cleaning.
""")

plt.boxplot(df["Sales"])
plt.title("Sales Box Plot")
plt.ylabel("Sales")
plt.show()


# ===========================================================
# 8. SEABORN COUNT PLOT
# ===========================================================

print("=" * 60)
print("8. SEABORN COUNT PLOT")
print("=" * 60)

print("""
countplot() counts how many observations
belong to each category.
""")

sns.countplot(data=df, x="City")
plt.title("Orders by City")
plt.show()


# ===========================================================
# 9. SEABORN BARPLOT
# ===========================================================

print("=" * 60)
print("9. SEABORN BARPLOT")
print("=" * 60)

print("""
barplot() is useful for comparing
a numerical value across categories.

Example:
Average sales by city.
""")

sns.barplot(data=df, x="City", y="Sales")
plt.title("Average Sales by City")
plt.show()


# ===========================================================
# 10. SEABORN HEATMAP
# ===========================================================

print("=" * 60)
print("10. SEABORN HEATMAP")
print("=" * 60)

print("""
Heatmaps are commonly used to visualize
correlations between numerical features.

Correlation:
+1  → strong positive relationship
 0  → little/no linear relationship
-1  → strong negative relationship
""")

correlation = df[["Sales", "Quantity", "Rating"]].corr()

sns.heatmap(
    correlation,
    annot=True
)

plt.title("Correlation Heatmap")
plt.show()


# ===========================================================
# 11. IMPORTANT CUSTOMIZATION
# ===========================================================

print("=" * 60)
print("11. BASIC CUSTOMIZATION")
print("=" * 60)

print("""
Important functions:

plt.title()       → title
plt.xlabel()      → X-axis label
plt.ylabel()      → Y-axis label
plt.legend()      → legend
plt.grid()        → grid
plt.figure()      → figure size
plt.show()        → display
plt.savefig()     → save plot
""")

plt.figure(figsize=(8, 5))

plt.bar(product_sales.index, product_sales.values)

plt.title("Product Sales")
plt.xlabel("Product")
plt.ylabel("Sales")
plt.grid(axis="y")

plt.savefig("product_sales.png")
plt.show()


# ===========================================================
# 12. REAL-WORLD MINI EDA
# ===========================================================

print("=" * 60)
print("12. MINI PROJECT - SALES EDA")
print("=" * 60)

print("""
Business questions:

1. Which product sells the most?
2. Which city generates the most sales?
3. How are sales distributed?
4. Is quantity related to sales?
5. Are there unusual/outlier sales?
""")

print("\nTotal sales by product:")
print(
    df.groupby("Product")["Sales"]
      .sum()
      .sort_values(ascending=False)
)

print("\nTotal sales by city:")
print(
    df.groupby("City")["Sales"]
      .sum()
      .sort_values(ascending=False)
)

print("\nCorrelation:")
print(
    df[["Sales", "Quantity", "Rating"]].corr()
)

print("""
The plots help us visually understand these results.

This process is called:

EDA
↓
Exploratory Data Analysis
""")


# ===========================================================
# 13. MATPLOTLIB + SEABORN IN ML
# ===========================================================

print("=" * 60)
print("13. ROLE IN MACHINE LEARNING")
print("=" * 60)

print("""
Typical workflow:

Dataset
   ↓
Pandas
   ↓
Data Cleaning
   ↓
Matplotlib / Seaborn
   ↓
EDA
   ↓
Feature Engineering
   ↓
ML Model

Visualization helps us decide:
- Which features matter
- Whether data has outliers
- Whether variables are correlated
- How data is distributed
""")


# ===========================================================
# 14. INTERVIEW ESSENTIALS
# ===========================================================

print("=" * 60)
print("14. INTERVIEW QUESTIONS")
print("=" * 60)

print("""
1. What is Matplotlib?
   → Python library for data visualization.

2. What is Seaborn?
   → Statistical visualization library built on Matplotlib.

3. Matplotlib vs Seaborn?
   → Matplotlib gives more control;
     Seaborn makes statistical visualization easier.

4. When do you use a histogram?
   → To understand data distribution.

5. When do you use a scatter plot?
   → To study the relationship between two numerical variables.

6. When do you use a box plot?
   → To see spread, median and outliers.

7. What is a heatmap?
   → A color-based visualization of values,
     commonly used for correlation matrices.

8. What is EDA?
   → Exploratory Data Analysis:
     understanding data using statistics and visualizations.
""")


# ===========================================================
# 15. PRACTICE
# ===========================================================

print("=" * 60)
print("15. PRACTICE")
print("=" * 60)

print("""
Try these yourself:

1. Create a bar chart of quantity by product.

2. Create a histogram of ratings.

3. Create a scatter plot of Rating vs Sales.

4. Create a box plot of Quantity.

5. Create a countplot of products.

6. Create a heatmap using all numerical columns.

7. Create a chart showing total sales by city.

8. Save one of your plots as a PNG file.
""")


# ===========================================================
# FINAL CHECKLIST
# ===========================================================

print("=" * 60)
print("MATPLOTLIB + SEABORN CHECKLIST")
print("=" * 60)

print("""
[✓] What / Why / When / Where
[✓] Matplotlib vs Seaborn
[✓] Line plot
[✓] Bar plot
[✓] Scatter plot
[✓] Histogram
[✓] Box plot
[✓] Countplot
[✓] Barplot
[✓] Heatmap
[✓] Basic customization
[✓] Save plots
[✓] EDA basics
[✓] Interview concepts

NEXT:
    Jupyter + Data Cleaning + EDA
""")