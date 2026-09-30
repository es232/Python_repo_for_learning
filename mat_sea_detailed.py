"""
======================================================================
             MATPLOTLIB + SEABORN — DETAILED STUDY MODULE
======================================================================

WHAT?
-----
Matplotlib:
    A Python library used to create visualizations and charts.

Seaborn:
    A statistical visualization library built on top of Matplotlib.

WHY?
----
Visualization helps us:
    - Understand data
    - Find patterns
    - Find trends
    - Compare categories
    - Detect outliers
    - Understand distributions
    - Find relationships between variables
    - Perform Exploratory Data Analysis (EDA)
    - Understand ML datasets and model results

WHEN?
------
Usually after loading and cleaning data with Pandas.

Typical Data Science workflow:

    Raw Data
       ↓
    Pandas
       ↓
    Data Cleaning
       ↓
    EDA
       ↓
    Matplotlib / Seaborn
       ↓
    Feature Engineering
       ↓
    Machine Learning
       ↓
    Model Evaluation
       ↓
    Visualization

WHERE?
-------
Used in:
    - Data Science
    - Machine Learning
    - Data Analysis
    - EDA
    - Research
    - Business Analytics
    - Reports
    - Dashboards
    - ML model evaluation

INSTALL:
--------
python -m pip install matplotlib seaborn pandas numpy

RUN:
----
python matplotlib_seaborn_detailed.py

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
# 1. INTRODUCTION
# ====================================================================

print("\n" + "=" * 70)
print("1. WHAT ARE MATPLOTLIB AND SEABORN?")
print("=" * 70)

print("""
MATPLOTLIB
----------

Matplotlib is one of the most important Python visualization
libraries.

It allows us to create:

    - Line charts
    - Bar charts
    - Scatter plots
    - Histograms
    - Box plots
    - Pie charts
    - Subplots
    - Custom visualizations

Example:

    plt.plot(x, y)

Matplotlib gives us detailed control over almost every part
of a chart.


SEABORN
-------

Seaborn is a statistical visualization library built on top
of Matplotlib.

It makes many common statistical visualizations easier.

Example:

    sns.scatterplot(data=df, x="Age", y="Salary")

Seaborn works especially well with Pandas DataFrames.


SIMPLE DIFFERENCE
-----------------

Matplotlib:
    More control and customization.

Seaborn:
    Easier statistical visualization and EDA.

Seaborn uses Matplotlib internally.

So learning Matplotlib first makes Seaborn easier.
""")


# ====================================================================
# 2. MATPLOTLIB BASIC STRUCTURE
# ====================================================================

print("\n" + "=" * 70)
print("2. MATPLOTLIB BASIC CONCEPTS")
print("=" * 70)

print("""
Important Matplotlib concepts:

FIGURE
------
The complete canvas/window.

AXES
----
The actual area where the graph is drawn.

AXIS
----
The X-axis and Y-axis.

A Figure can contain multiple Axes.

Example:

    Figure
    └── Axes
        ├── X-axis
        └── Y-axis


Basic plotting flow:

    plt.figure()
    plt.plot(...)
    plt.title(...)
    plt.xlabel(...)
    plt.ylabel(...)
    plt.show()


Important functions:

    plt.figure()
    plt.subplots()
    plt.title()
    plt.xlabel()
    plt.ylabel()
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig()
    plt.tight_layout()
""")


# ====================================================================
# 3. CREATE A REAL-WORLD DATASET
# ====================================================================

print("\n" + "=" * 70)
print("3. REAL-WORLD DATASET")
print("=" * 70)

print("""
We will use an employee dataset.

Imagine this data came from a company's HR system.

Columns:

    Employee
    Department
    Experience
    Salary
    Performance
    Projects

We will use visualization to answer real business questions.
""")

data = {
    "Employee": [
        "Arun", "Priya", "Rahul", "Meena",
        "Kavin", "Divya", "Vijay", "Anu",
        "Suresh", "Nila", "Hari", "Keerthi"
    ],

    "Department": [
        "Engineering", "Engineering", "HR", "Sales",
        "Engineering", "Sales", "HR", "Engineering",
        "Sales", "Engineering", "HR", "Sales"
    ],

    "Experience": [
        1, 3, 2, 5, 4, 6,
        7, 2, 8, 5, 10, 3
    ],

    "Salary": [
        400000, 650000, 450000, 700000,
        800000, 850000, 900000, 500000,
        1000000, 780000, 1200000, 600000
    ],

    "Performance": [
        72, 85, 68, 80,
        91, 88, 90, 75,
        94, 86, 96, 79
    ],

    "Projects": [
        1, 3, 2, 4,
        5, 5, 6, 2,
        7, 4, 8, 3
    ]
}

df = pd.DataFrame(data)

print(df)


# ====================================================================
# 4. LINE PLOT
# ====================================================================

print("\n" + "=" * 70)
print("4. LINE PLOT")
print("=" * 70)

print("""
WHAT?
-----
A line plot connects data points using lines.

WHY?
----
Useful for understanding trends and changes.

WHEN?
------
Use it when data has an order.

Especially useful for:

    - Time series
    - Monthly sales
    - Temperature
    - Stock prices
    - Training loss
    - Website traffic

REAL-WORLD EXAMPLE:
-------------------
A company wants to know how sales changed every month.

IMPORTANT FUNCTION:
-------------------

    plt.plot(x, y)
""")

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
sales = [100, 120, 135, 128, 160, 180]

plt.figure(figsize=(8, 5))

plt.plot(
    months,
    sales,
    marker="o",
    label="Sales"
)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.grid()

plt.show()


# ====================================================================
# 5. BAR CHART
# ====================================================================

print("\n" + "=" * 70)
print("5. BAR CHART")
print("=" * 70)

print("""
WHAT?
-----
A bar chart compares values across categories.

WHY?
----
It makes category comparisons easy.

WHEN?
------
Use it when you want to compare separate categories.

Examples:

    Product vs Sales
    Department vs Employees
    City vs Revenue
    Model vs Accuracy

IMPORTANT FUNCTIONS:

    plt.bar()
    plt.barh()

bar()     -> vertical bars
barh()    -> horizontal bars
""")

department_salary = (
    df.groupby("Department")["Salary"]
    .mean()
)

plt.figure(figsize=(8, 5))

plt.bar(
    department_salary.index,
    department_salary.values
)

plt.title("Average Salary by Department")
plt.xlabel("Department")
plt.ylabel("Average Salary")

plt.xticks(rotation=15)

plt.show()


# ====================================================================
# 6. SCATTER PLOT
# ====================================================================

print("\n" + "=" * 70)
print("6. SCATTER PLOT")
print("=" * 70)

print("""
WHAT?
-----
A scatter plot displays individual observations as points.

WHY?
----
It helps us understand the relationship between TWO numerical
variables.

WHEN?
------
Use it when asking:

    "Are X and Y related?"

Examples:

    Experience vs Salary
    Hours studied vs Marks
    Advertising vs Sales
    Height vs Weight
    Age vs Income

INTERPRETATION:

If points generally move upward:
    Positive relationship

If points generally move downward:
    Negative relationship

If points look random:
    Weak/no obvious linear relationship

IMPORTANT:
----------
Correlation does NOT automatically mean causation.
""")

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Experience"],
    df["Salary"]
)

plt.title("Experience vs Salary")
plt.xlabel("Experience")
plt.ylabel("Salary")

plt.grid()

plt.show()


# ====================================================================
# 7. HISTOGRAM
# ====================================================================

print("\n" + "=" * 70)
print("7. HISTOGRAM")
print("=" * 70)

print("""
WHAT?
-----
A histogram shows the distribution of a numerical variable.

WHY?
----
It helps us understand:

    - Where values are concentrated
    - Spread
    - Shape of distribution
    - Possible skewness
    - Possible unusual values

WHEN?
------
Use it for ONE numerical variable.

Examples:

    Salary distribution
    Age distribution
    Exam marks
    House prices

BINS
-----
A histogram divides values into intervals called bins.

Example:

    bins=5

means the data is divided into approximately 5 intervals.

Too few bins:
    Important patterns may disappear.

Too many bins:
    Plot may become noisy.
""")

plt.figure(figsize=(8, 5))

plt.hist(
    df["Salary"],
    bins=6
)

plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")

plt.show()


# ====================================================================
# 8. BOX PLOT
# ====================================================================

print("\n" + "=" * 70)
print("8. BOX PLOT")
print("=" * 70)

print("""
WHAT?
-----
A box plot summarizes the distribution of numerical data.

It helps show:

    - Median
    - Quartiles
    - Spread
    - Possible outliers

IMPORTANT TERMS:

Q1
--
25th percentile

Median
------
50th percentile

Q3
--
75th percentile

IQR
---
Interquartile Range

    IQR = Q3 - Q1

Common outlier rule:

    Lower limit = Q1 - 1.5 * IQR
    Upper limit = Q3 + 1.5 * IQR

WHY?
----
Very useful during data cleaning and EDA.

REAL-WORLD EXAMPLE:
-------------------
Suppose most employee salaries are between
4 LPA and 10 LPA but one salary is 50 LPA.

A box plot can help us identify that unusual value.

IMPORTANT:
----------
An outlier is NOT automatically an error.

Always investigate why it exists.
""")

plt.figure(figsize=(6, 5))

plt.boxplot(
    df["Salary"]
)

plt.title("Salary Box Plot")
plt.ylabel("Salary")

plt.show()


# ====================================================================
# 9. SEABORN COUNT PLOT
# ====================================================================

print("\n" + "=" * 70)
print("9. SEABORN COUNT PLOT")
print("=" * 70)

print("""
WHAT?
-----
countplot() counts how many observations belong to each category.

WHEN?
------
Useful for categorical variables.

Examples:

    Number of students by department
    Number of customers by city
    Number of samples per class

VERY IMPORTANT IN ML
--------------------
For classification problems, checking class counts helps
identify class imbalance.

Example:

    Class 0 -> 900 samples
    Class 1 -> 100 samples

This is an imbalanced dataset.
""")

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Department"
)

plt.title("Employees by Department")

plt.xticks(rotation=15)

plt.show()


# ====================================================================
# 10. SEABORN BARPLOT
# ====================================================================

print("\n" + "=" * 70)
print("10. SEABORN BARPLOT")
print("=" * 70)

print("""
WHAT?
-----
Seaborn's barplot compares an aggregated numerical value
across categories.

By default, it calculates an estimate of the mean.

Example:

    Department -> Average Salary

DIFFERENCE:

countplot:
    Counts observations.

barplot:
    Shows an aggregated numerical value.

Example:

countplot:
    Engineering -> 5 employees

barplot:
    Engineering -> Average salary = 7.1 LPA
""")

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="Department",
    y="Salary"
)

plt.title("Average Salary by Department")

plt.xticks(rotation=15)

plt.show()


# ====================================================================
# 11. SEABORN SCATTERPLOT
# ====================================================================

print("\n" + "=" * 70)
print("11. SEABORN SCATTERPLOT")
print("=" * 70)

print("""
Seaborn makes scatter plots more powerful.

Important parameters:

hue
---
Changes color based on a categorical variable.

size
----
Changes point size based on a variable.

style
-----
Changes marker style based on a category.

This allows us to visualize multiple dimensions.
""")

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Experience",
    y="Salary",
    hue="Department",
    size="Performance"
)

plt.title("Experience vs Salary")

plt.show()


# ====================================================================
# 12. SEABORN HISTPLOT
# ====================================================================

print("\n" + "=" * 70)
print("12. SEABORN HISTPLOT")
print("=" * 70)

print("""
Seaborn provides histplot().

It can also display KDE.

KDE
---
Kernel Density Estimation.

It provides a smooth estimate of the distribution.

Example:

    sns.histplot(
        data=df,
        x="Salary",
        kde=True
    )
""")

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Salary",
    bins=6,
    kde=True
)

plt.title("Salary Distribution")

plt.show()


# ====================================================================
# 13. SEABORN BOXPLOT
# ====================================================================

print("\n" + "=" * 70)
print("13. SEABORN BOXPLOT")
print("=" * 70)

print("""
One major advantage of Seaborn is easy grouping.

Instead of plotting all salaries together,
we can compare salary distributions between departments.
""")

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Department",
    y="Salary"
)

plt.title("Salary Distribution by Department")

plt.xticks(rotation=15)

plt.show()


# ====================================================================
# 14. CORRELATION
# ====================================================================

print("\n" + "=" * 70)
print("14. CORRELATION")
print("=" * 70)

print("""
WHAT?
-----
Correlation measures the strength and direction of a relationship
between numerical variables.

Typical Pearson correlation range:

    +1 -> strong positive linear relationship
     0 -> little/no linear relationship
    -1 -> strong negative linear relationship

Example:

If Experience and Salary have correlation +0.90,
they have a strong positive linear relationship.

IMPORTANT:
----------
Correlation does NOT prove causation.

Example:

Ice cream sales and swimming accidents may both increase
during summer.

That does not mean ice cream causes swimming accidents.
A third factor, such as hot weather, may influence both.
""")

numeric_columns = [
    "Experience",
    "Salary",
    "Performance",
    "Projects"
]

correlation = df[numeric_columns].corr()

print("\nCorrelation Matrix:")
print(correlation)


# ====================================================================
# 15. HEATMAP
# ====================================================================

print("\n" + "=" * 70)
print("15. SEABORN HEATMAP")
print("=" * 70)

print("""
WHAT?
-----
A heatmap represents numerical values using colors.

WHY?
----
It is especially useful for correlation matrices.

Instead of reading:

    0.85
    -0.21
    0.67

we can visually identify strong and weak relationships.

IMPORTANT PARAMETERS:

annot=True
----------
Displays values inside cells.

fmt=".2f"
---------
Displays values to 2 decimal places.
""")

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()


# ====================================================================
# 16. PAIRPLOT
# ====================================================================

print("\n" + "=" * 70)
print("16. SEABORN PAIRPLOT")
print("=" * 70)

print("""
WHAT?
-----
pairplot() creates multiple plots showing relationships between
several numerical variables.

WHY?
----
Useful for quick exploratory analysis.

It can help identify:

    - Relationships
    - Clusters
    - Distributions
    - Possible correlations

IMPORTANT:
----------
Don't use pairplot with dozens of columns.

It can become extremely crowded.

Use it with a small set of important features.
""")

sns.pairplot(
    df[
        [
            "Experience",
            "Salary",
            "Performance",
            "Projects"
        ]
    ]
)

plt.show()


# ====================================================================
# 17. SUBPLOTS
# ====================================================================

print("\n" + "=" * 70)
print("17. SUBPLOTS")
print("=" * 70)

print("""
WHAT?
-----
Subplots allow multiple charts inside one figure.

Example:

    2 rows × 2 columns

creates 4 plotting areas.

Useful when you want to compare multiple visualizations
without opening many separate windows.
""")

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 8)
)

axes[0, 0].scatter(
    df["Experience"],
    df["Salary"]
)
axes[0, 0].set_title("Experience vs Salary")

axes[0, 1].hist(
    df["Salary"],
    bins=6
)
axes[0, 1].set_title("Salary Distribution")

axes[1, 0].boxplot(
    df["Salary"]
)
axes[1, 0].set_title("Salary Box Plot")

axes[1, 1].scatter(
    df["Performance"],
    df["Salary"]
)
axes[1, 1].set_title("Performance vs Salary")

plt.tight_layout()

plt.show()


# ====================================================================
# 18. CUSTOMIZATION
# ====================================================================

print("\n" + "=" * 70)
print("18. IMPORTANT CUSTOMIZATION")
print("=" * 70)

print("""
Important Matplotlib functions:

plt.figure(figsize=(8, 5))
    -> controls figure size

plt.title()
    -> chart title

plt.xlabel()
    -> X-axis label

plt.ylabel()
    -> Y-axis label

plt.legend()
    -> displays legend

plt.grid()
    -> adds grid

plt.xlim()
    -> controls X-axis range

plt.ylim()
    -> controls Y-axis range

plt.xticks()
    -> controls X-axis tick labels

plt.yticks()
    -> controls Y-axis tick labels

plt.tight_layout()
    -> adjusts spacing

plt.savefig()
    -> saves chart

plt.show()
    -> displays chart
""")

plt.figure(figsize=(8, 5))

plt.bar(
    department_salary.index,
    department_salary.values
)

plt.title("Average Department Salary")
plt.xlabel("Department")
plt.ylabel("Average Salary")

plt.xticks(rotation=15)

plt.grid(axis="y")

plt.tight_layout()

plt.show()


# ====================================================================
# 19. SAVING VISUALIZATIONS
# ====================================================================

print("\n" + "=" * 70)
print("19. SAVING A PLOT")
print("=" * 70)

print("""
Use:

    plt.savefig("filename.png")

Common formats:

    PNG
    JPG
    PDF
    SVG

DPI
---
DPI controls image resolution.

Example:

    plt.savefig(
        "chart.png",
        dpi=300
    )

For reports and presentations, higher resolution is useful.
""")

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Projects",
    y="Performance"
)

plt.title("Projects vs Performance")

plt.tight_layout()

plt.savefig(
    "projects_vs_performance.png",
    dpi=300
)

plt.show()


# ====================================================================
# 20. SEABORN THEMES
# ====================================================================

print("\n" + "=" * 70)
print("20. SEABORN THEMES")
print("=" * 70)

print("""
Seaborn provides themes that make charts easier to style.

Example:

    sns.set_theme()

Common idea:

    sns.set_style("whitegrid")

Use styles to improve readability.

Do not use styling simply for decoration.
The visualization should remain easy to understand.
""")

sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="Department",
    y="Performance"
)

plt.title("Average Performance by Department")

plt.xticks(rotation=15)

plt.show()


# ====================================================================
# 21. EDA — EXPLORATORY DATA ANALYSIS
# ====================================================================

print("\n" + "=" * 70)
print("21. EDA — EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("""
EDA means Exploratory Data Analysis.

The goal is to understand the dataset before building
a machine-learning model.

Typical EDA:

1. Understand dataset shape
2. Understand columns
3. Check data types
4. Check missing values
5. Check duplicates
6. Look at statistics
7. Analyze distributions
8. Find outliers
9. Compare categories
10. Study relationships
11. Check correlations
12. Form useful hypotheses

Visualization is an important part of EDA.
""")

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nStatistics:")
print(df.describe())

print("\nDepartment counts:")
print(df["Department"].value_counts())

print("\nAverage salary by department:")
print(
    df.groupby("Department")["Salary"]
    .mean()
)

print("\nCorrelation:")
print(
    df[
        [
            "Experience",
            "Salary",
            "Performance",
            "Projects"
        ]
    ].corr()
)


# ====================================================================
# 22. MACHINE LEARNING USE
# ====================================================================

print("\n" + "=" * 70)
print("22. MATPLOTLIB + SEABORN IN MACHINE LEARNING")
print("=" * 70)

print("""
IMPORTANT:

Matplotlib and Seaborn are NOT machine-learning algorithms.

They are visualization tools.

Their major ML use is understanding data and model behavior.


BEFORE TRAINING
---------------

Use visualizations to:

    - Understand feature distributions
    - Find outliers
    - Check class imbalance
    - Identify relationships
    - Inspect correlations
    - Compare groups


DURING MODEL DEVELOPMENT
------------------------

You can visualize:

    - Training loss
    - Validation loss
    - Learning curves
    - Predictions
    - Residuals
    - Feature relationships


AFTER TRAINING
--------------

You can visualize:

    - Confusion matrix
    - ROC curve
    - Precision-recall curve
    - Actual vs predicted values
    - Residual plots
    - Feature importance


EXAMPLE ML WORKFLOW
-------------------

Dataset
   ↓
Pandas
   ↓
Cleaning
   ↓
EDA
   ↓
Matplotlib / Seaborn
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
ML Algorithm
   ↓
Evaluation
   ↓
Visualization
""")


# ====================================================================
# 23. ACTUAL VS PREDICTED
# ====================================================================

print("\n" + "=" * 70)
print("23. ML EXAMPLE — ACTUAL VS PREDICTED")
print("=" * 70)

print("""
Suppose a regression model predicts employee salaries.

We can compare:

    Actual salary
        vs
    Predicted salary

Points close to the diagonal line indicate predictions
close to actual values.
""")

actual = np.array([
    450000,
    600000,
    700000,
    800000,
    1000000
])

predicted = np.array([
    470000,
    590000,
    680000,
    830000,
    960000
])

plt.figure(figsize=(8, 5))

plt.scatter(
    actual,
    predicted
)

minimum = min(
    actual.min(),
    predicted.min()
)

maximum = max(
    actual.max(),
    predicted.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.title("Actual vs Predicted Salary")
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")

plt.grid()

plt.show()


# ====================================================================
# 24. WHICH PLOT SHOULD I USE?
# ====================================================================

print("\n" + "=" * 70)
print("24. WHICH PLOT SHOULD I USE?")
print("=" * 70)

print("""
Question                         Best choice
------------------------------------------------

Want to see a trend?             Line plot

Compare categories?              Bar chart

Count categories?                Countplot

See numerical distribution?      Histogram

Find outliers/spread?            Box plot

Relationship between X and Y?    Scatter plot

Correlation matrix?              Heatmap

Explore many numerical vars?     Pairplot


MEMORY TRICK:

"What changed?"
    -> Line

"Which category is bigger?"
    -> Bar

"How many?"
    -> Countplot

"How is the data distributed?"
    -> Histogram

"Are there outliers?"
    -> Boxplot

"Are X and Y related?"
    -> Scatter

"How do variables correlate?"
    -> Heatmap
""")


# ====================================================================
# 25. IMPORTANT FUNCTIONS
# ====================================================================

print("\n" + "=" * 70)
print("25. IMPORTANT FUNCTIONS TO REMEMBER")
print("=" * 70)

print("""
MATPLOTLIB
----------

Basic:
    plt.figure()
    plt.subplots()
    plt.show()

Charts:
    plt.plot()
    plt.bar()
    plt.barh()
    plt.scatter()
    plt.hist()
    plt.boxplot()

Labels:
    plt.title()
    plt.xlabel()
    plt.ylabel()
    plt.legend()

Formatting:
    plt.grid()
    plt.xlim()
    plt.ylim()
    plt.xticks()
    plt.yticks()
    plt.tight_layout()

Saving:
    plt.savefig()


SEABORN
-------

    sns.set_theme()
    sns.set_style()

    sns.countplot()
    sns.barplot()
    sns.scatterplot()
    sns.histplot()
    sns.boxplot()
    sns.heatmap()
    sns.pairplot()


IMPORTANT:
----------

Do NOT try to memorize every visualization function.

Understand:

    What question am I asking?

Then choose the appropriate plot.
""")


# ====================================================================
# 26. COMMON MISTAKES
# ====================================================================

print("\n" + "=" * 70)
print("26. COMMON MISTAKES")
print("=" * 70)

print("""
1. USING THE WRONG CHART
------------------------
Choose the visualization based on the question.

2. NO AXIS LABELS
------------------
The viewer should understand what X and Y represent.

3. MISLEADING AXES
------------------
Changing axis limits can make differences appear larger
or smaller than they actually are.

4. TOO MANY BINS
----------------
A histogram can become noisy.

5. TOO MANY COLORS
------------------
Colors should communicate information.

6. CONFUSING CORRELATION WITH CAUSATION
---------------------------------------
Correlation does not prove that one variable causes another.

7. DELETING ALL OUTLIERS
------------------------
An outlier may be:
    - genuine data
    - measurement error
    - rare event
    - important business case

Investigate before removing.

8. VISUALIZING DIRTY DATA
-------------------------
Bad data can lead to misleading charts.

9. OVERLOADED PAIRPLOT
----------------------
Too many columns create unreadable visualizations.

10. FORGETTING TIGHT_LAYOUT()
-----------------------------
Labels can get cut off.

11. FORGETTING TO SAVE BEFORE SHOW
----------------------------------
In scripts, it is safer to call savefig()
before show().

12. USING DECORATION INSTEAD OF INFORMATION
--------------------------------------------
A chart should communicate something useful.
""")


# ====================================================================
# 27. INTERVIEW QUESTIONS
# ====================================================================

print("\n" + "=" * 70)
print("27. IMPORTANT INTERVIEW QUESTIONS + ANSWERS")
print("=" * 70)

print("""
Q1. What is Matplotlib?
-----------------------
Answer:
Matplotlib is a Python visualization library used to create
different types of charts and graphs.

---------------------------------------------------------------

Q2. What is Seaborn?
---------------------
Answer:
Seaborn is a statistical visualization library built on top
of Matplotlib. It provides a higher-level interface for
creating statistical graphics.

---------------------------------------------------------------

Q3. Matplotlib vs Seaborn?
---------------------------
Answer:
Matplotlib provides detailed control and customization.
Seaborn provides easier statistical visualization and works
particularly well with Pandas DataFrames.

---------------------------------------------------------------

Q4. What is a Figure?
---------------------
Answer:
A Figure is the complete canvas containing one or more Axes.

---------------------------------------------------------------

Q5. What is an Axes?
--------------------
Answer:
An Axes is the actual plotting area where data is displayed.
A Figure can contain multiple Axes.

---------------------------------------------------------------

Q6. When would you use a line plot?
------------------------------------
Answer:
For trends and changes across ordered observations,
especially time-series data.

---------------------------------------------------------------

Q7. When would you use a bar chart?
------------------------------------
Answer:
To compare numerical values across discrete categories.

---------------------------------------------------------------

Q8. When would you use a scatter plot?
---------------------------------------
Answer:
To examine the relationship between two numerical variables.

---------------------------------------------------------------

Q9. What is a histogram?
------------------------
Answer:
A histogram shows the distribution of a numerical variable
by dividing values into intervals called bins.

---------------------------------------------------------------

Q10. Histogram vs Bar Chart?
----------------------------
Answer:
A histogram represents the distribution of numerical data,
while a bar chart compares separate categorical values.

---------------------------------------------------------------

Q11. What does a box plot show?
-------------------------------
Answer:
A box plot shows the median, quartiles, spread and possible
outliers of a numerical variable.

---------------------------------------------------------------

Q12. What is an outlier?
-------------------------
Answer:
An outlier is an observation that is unusually far from
the majority of observations.

---------------------------------------------------------------

Q13. Should all outliers be removed?
------------------------------------
Answer:
No. First investigate why the outlier exists. It may represent
a valid and important observation.

---------------------------------------------------------------

Q14. What is correlation?
-------------------------
Answer:
Correlation measures the strength and direction of association
between variables, commonly their linear relationship.

---------------------------------------------------------------

Q15. Does correlation imply causation?
--------------------------------------
Answer:
No. Correlation does not prove that one variable causes another.

---------------------------------------------------------------

Q16. What is a heatmap?
-----------------------
Answer:
A heatmap uses colors to represent numerical values.
It is commonly used to visualize correlation matrices.

---------------------------------------------------------------

Q17. What is KDE?
-----------------
Answer:
KDE stands for Kernel Density Estimation. It provides a
smooth estimate of a probability distribution.

---------------------------------------------------------------

Q18. What is EDA?
-----------------
Answer:
EDA stands for Exploratory Data Analysis. It is the process
of understanding a dataset using statistics, summaries and
visualizations before deeper analysis or modeling.

---------------------------------------------------------------

Q19. Why is visualization important in ML?
-------------------------------------------
Answer:
It helps identify distributions, outliers, relationships,
class imbalance and data-quality problems before and during
model development.

---------------------------------------------------------------

Q20. Can Matplotlib build an ML model?
---------------------------------------
Answer:
No. Matplotlib is a visualization library, not an ML algorithm.

---------------------------------------------------------------

Q21. Can Seaborn replace Pandas?
--------------------------------
Answer:
No. Pandas is primarily used for data manipulation and analysis,
while Seaborn is primarily used for visualization.

---------------------------------------------------------------

Q22. What does plt.show() do?
------------------------------
Answer:
It displays the current figure.

---------------------------------------------------------------

Q23. What does plt.savefig() do?
--------------------------------
Answer:
It saves the current figure to a file.

---------------------------------------------------------------

Q24. Why is figsize used?
-------------------------
Answer:
It controls the dimensions of the figure and improves readability.

---------------------------------------------------------------

Q25. Why use plt.tight_layout()?
--------------------------------
Answer:
It automatically adjusts spacing between plot elements so
labels and titles do not overlap or get cut off.

---------------------------------------------------------------

Q26. What is hue in Seaborn?
----------------------------
Answer:
hue maps a categorical variable to different colors, allowing
different groups to be visually distinguished.

---------------------------------------------------------------

Q27. What is the difference between countplot and barplot?
------------------------------------------------------------
Answer:
countplot counts observations in each category.
barplot displays an aggregated numerical value, such as mean,
for each category.

---------------------------------------------------------------

Q28. What are bins in a histogram?
----------------------------------
Answer:
Bins are intervals used to group numerical values in a histogram.

---------------------------------------------------------------

Q29. What is a pairplot?
------------------------
Answer:
A pairplot creates multiple plots to show relationships and
distributions among several numerical variables.

---------------------------------------------------------------

Q30. Why use Seaborn in EDA?
----------------------------
Answer:
Seaborn provides convenient statistical plots and integrates
well with Pandas DataFrames, making common EDA tasks easier.
""")


# ====================================================================
# 28. PRACTICE EXERCISES
# ====================================================================

print("\n" + "=" * 70)
print("28. PRACTICE EXERCISES")
print("=" * 70)

print("""
LEVEL 1 — BASICS
----------------

1. Create a line plot for monthly sales.

2. Create a bar chart showing average salary
   by department.

3. Create a histogram of salary.

4. Create a box plot of salary.

5. Create a countplot of departments.


LEVEL 2 — RELATIONSHIPS
-----------------------

6. Create a scatter plot:
       Experience vs Salary

7. Create a scatter plot:
       Performance vs Salary

8. Use hue to display departments.

9. Create a heatmap using numerical columns.

10. Create a pairplot using:
       Experience
       Salary
       Performance


LEVEL 3 — EDA
-------------

11. Which department has the highest average salary?

12. Which employee has the highest salary?

13. Is Experience related to Salary?

14. Is Performance related to Salary?

15. Are there possible salary outliers?

16. Which department has the highest average performance?

17. Which department has the most employees?

18. Write 5 observations from the visualizations.


LEVEL 4 — ML
------------

19. Create actual and predicted values.

20. Plot actual vs predicted values.

21. Create a training-loss vs epoch line plot.

22. Create a validation-loss vs epoch line plot.

23. Plot residuals for a regression model.

24. Visualize a confusion matrix.

IMPORTANT:
----------
Do not only create charts.

For every chart, answer:

    What does this chart tell me?

That is the actual purpose of EDA.
""")


# ====================================================================
# 29. MINI PROJECT
# ====================================================================

print("\n" + "=" * 70)
print("29. MINI PROJECT — EMPLOYEE EDA")
print("=" * 70)

print("""
PROJECT GOAL
------------

Analyze employee data using Pandas, Matplotlib and Seaborn.

BUSINESS QUESTIONS
------------------

1. Which department has the highest average salary?

2. Which department has the highest performance?

3. Does experience appear related to salary?

4. Does performance appear related to salary?

5. Are there salary outliers?

6. Which department has the most employees?

7. Which variables have strong correlations?

REQUIRED VISUALIZATIONS
-----------------------

1. Bar chart:
   Average salary by department

2. Bar chart:
   Average performance by department

3. Countplot:
   Employees by department

4. Histogram:
   Salary distribution

5. Boxplot:
   Salary by department

6. Scatter:
   Experience vs Salary

7. Scatter:
   Performance vs Salary

8. Heatmap:
   Correlation matrix

9. Pairplot:
   Important numerical variables

FINAL TASK
----------

Write at least 5 conclusions from your analysis.

Example:

"Engineering employees have a higher average salary
than HR employees."

Do not simply describe the chart.
Explain what the finding means.
""")


# ====================================================================
# 30. MASTERY CHECKLIST
# ====================================================================

print("\n" + "=" * 70)
print("30. MASTERY CHECKLIST")
print("=" * 70)

"""
CONCEPTS
--------
[ ] What is Matplotlib?
[ ] What is Seaborn?
[ ] Why visualization is important
[ ] When to use visualization
[ ] Where visualization is used
[ ] Figure vs Axes
[ ] EDA


MATPLOTLIB
----------
[ ] plt.figure()
[ ] plt.subplots()
[ ] plt.plot()
[ ] plt.bar()
[ ] plt.scatter()
[ ] plt.hist()
[ ] plt.boxplot()
[ ] plt.title()
[ ] plt.xlabel()
[ ] plt.ylabel()
[ ] plt.legend()
[ ] plt.grid()
[ ] plt.tight_layout()
[ ] plt.savefig()
[ ] plt.show()


SEABORN
-------
[ ] sns.set_theme()
[ ] sns.countplot()
[ ] sns.barplot()
[ ] sns.scatterplot()
[ ] sns.histplot()
[ ] sns.boxplot()
[ ] sns.heatmap()
[ ] sns.pairplot()
[ ] hue
[ ] KDE


DATA UNDERSTANDING
------------------
[ ] Distribution
[ ] Outliers
[ ] Median
[ ] Quartiles
[ ] IQR
[ ] Correlation
[ ] Positive correlation
[ ] Negative correlation
[ ] Class imbalance


MACHINE LEARNING
----------------
[ ] Pre-model EDA
[ ] Feature distributions
[ ] Outlier detection
[ ] Class distribution
[ ] Correlation analysis
[ ] Training curves
[ ] Actual vs predicted
[ ] Confusion matrix visualization


INTERVIEW
---------
[ ] Explain each major plot
[ ] Choose the correct plot for a question
[ ] Explain histogram vs bar chart
[ ] Explain correlation vs causation
[ ] Explain Matplotlib vs Seaborn
[ ] Explain EDA
[ ] Explain outliers
[ ] Explain heatmaps


FINAL RULE
----------

Do not memorize charts blindly.

First ask:

    "What question am I trying to answer?"

Then choose the visualization.

======================================================================
END OF MATPLOTLIB + SEABORN
======================================================================

NEXT:
    Jupyter Notebook
    +
    Data Cleaning
    +
    Complete EDA Workflow

After that we can move into the Mathematics required for ML.
======================================================================
"""