"""
NUMPY - QUICK LEARNING MODULE
=============================

How to run:
1. Install NumPy:
       python -m pip install numpy

2. Run:
       python numpy_learning.py

Windows alternative:
       py numpy_learning.py

This file covers the NumPy concepts you actually need
before moving to Pandas.
"""

import numpy as np


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# 1. WHAT IS NUMPY?
# ============================================================

def basics():
    section("1. WHAT IS NUMPY?")

    print("""
NumPy = Numerical Python

It is used for fast numerical computation.

Main uses:
- Data Science
- Machine Learning
- Deep Learning
- Scientific computing
- Data analysis

The main object in NumPy is called an ndarray
(N-dimensional array).
""")

    arr = np.array([10, 20, 30, 40])

    print("Example:")
    print("Array:", arr)


# ============================================================
# 2. CREATING ARRAYS
# ============================================================

def creating_arrays():
    section("2. CREATING ARRAYS")

    arr = np.array([1, 2, 3, 4])
    zeros = np.zeros(5)
    ones = np.ones(5)
    numbers = np.arange(0, 10, 2)
    line = np.linspace(0, 1, 5)

    print("np.array():", arr)
    print("np.zeros():", zeros)
    print("np.ones():", ones)
    print("np.arange():", numbers)
    print("np.linspace():", line)

    print("""
Remember:

arange(start, stop, step)
linspace(start, stop, number_of_values)
""")


# ============================================================
# 3. ARRAY PROPERTIES
# ============================================================

def properties():
    section("3. ARRAY PROPERTIES")

    arr = np.array([
        [10, 20, 30],
        [40, 50, 60]
    ])

    print("Array:")
    print(arr)

    print("\nshape:", arr.shape)
    print("ndim :", arr.ndim)
    print("size :", arr.size)
    print("dtype:", arr.dtype)

    print("""
Important:

shape → dimensions
ndim  → number of dimensions
size  → total number of elements
dtype → data type
""")


# ============================================================
# 4. INDEXING AND SLICING
# ============================================================

def indexing_slicing():
    section("4. INDEXING AND SLICING")

    arr = np.array([10, 20, 30, 40, 50])

    print("Array:", arr)

    print("arr[0]   =", arr[0])
    print("arr[-1]  =", arr[-1])
    print("arr[1:4] =", arr[1:4])

    matrix = np.array([
        [10, 20, 30],
        [40, 50, 60]
    ])

    print("\nMatrix:")
    print(matrix)

    print("\nmatrix[0, 1] =", matrix[0, 1])
    print("First column =", matrix[:, 0])
    print("Second row   =", matrix[1, :])


# ============================================================
# 5. VECTORIZED OPERATIONS
# ============================================================

def vectorization():
    section("5. VECTORIZED OPERATIONS")

    arr = np.array([1, 2, 3, 4])

    print("Array:", arr)
    print("arr * 2:", arr * 2)
    print("arr + 10:", arr + 10)
    print("arr ** 2:", arr ** 2)

    print("""
NumPy allows operations on the entire array
without manually writing a loop.

This is called vectorization.
""")


# ============================================================
# 6. AGGREGATION
# ============================================================

def aggregation():
    section("6. AGGREGATION FUNCTIONS")

    marks = np.array([80, 70, 90, 60, 85])

    print("Marks:", marks)

    print("Sum   :", np.sum(marks))
    print("Mean  :", np.mean(marks))
    print("Median:", np.median(marks))
    print("Min   :", np.min(marks))
    print("Max   :", np.max(marks))
    print("Std   :", np.std(marks))


# ============================================================
# 7. AXIS
# ============================================================

def axis():
    section("7. AXIS")

    marks = np.array([
        [80, 70, 90],
        [60, 85, 75],
        [90, 95, 88]
    ])

    print("Marks:")
    print(marks)

    print("\nColumn-wise mean (axis=0):")
    print(np.mean(marks, axis=0))

    print("\nRow-wise mean (axis=1):")
    print(np.mean(marks, axis=1))

    print("""
Remember:

axis=0 → column-wise
axis=1 → row-wise
""")


# ============================================================
# 8. FILTERING
# ============================================================

def filtering():
    section("8. BOOLEAN FILTERING")

    marks = np.array([40, 55, 70, 85, 95])

    print("Marks:", marks)

    print("Marks > 60:", marks[marks > 60])

    print("Marks between 50 and 90:")
    print(marks[(marks >= 50) & (marks <= 90)])


# ============================================================
# 9. NP.WHERE
# ============================================================

def where_example():
    section("9. NP.WHERE")

    marks = np.array([40, 70, 30, 90, 55])

    result = np.where(
        marks >= 50,
        "Pass",
        "Fail"
    )

    print("Marks :", marks)
    print("Result:", result)


# ============================================================
# 10. RESHAPE
# ============================================================

def reshape_example():
    section("10. RESHAPING")

    arr = np.arange(1, 7)

    print("Original:")
    print(arr)

    reshaped = arr.reshape(2, 3)

    print("\nReshaped:")
    print(reshaped)

    print("""
Important:

The total number of elements must remain the same.

6 elements → 2 × 3
""")


# ============================================================
# 11. BROADCASTING
# ============================================================

def broadcasting():
    section("11. BROADCASTING")

    marks = np.array([
        [70, 80, 90],
        [60, 75, 85]
    ])

    bonus = np.array([5, 5, 5])

    print("Original marks:")
    print(marks)

    print("\nBonus:")
    print(bonus)

    print("\nAfter bonus:")
    print(marks + bonus)

    print("""
Broadcasting allows NumPy to perform operations
between compatible array shapes.
""")


# ============================================================
# 12. RANDOM
# ============================================================

def random_numbers():
    section("12. RANDOM NUMBERS")

    rng = np.random.default_rng(42)

    print("Random integers:")
    print(rng.integers(1, 100, size=5))

    print("\nRandom decimals:")
    print(rng.random(5))


# ============================================================
# 13. SORTING
# ============================================================

def sorting():
    section("13. SORTING")

    arr = np.array([40, 10, 30, 20])

    print("Original:", arr)
    print("Sorted  :", np.sort(arr))
    print("Indices :", np.argsort(arr))


# ============================================================
# 14. COPY
# ============================================================

def copy_example():
    section("14. COPY")

    a = np.array([10, 20, 30])

    b = a.copy()

    b[0] = 999

    print("Original:", a)
    print("Copy    :", b)

    print("""
copy() creates an independent array.

b = a
would make b refer to the same underlying array.
""")


# ============================================================
# 15. MINI PROJECT
# ============================================================

def student_marks_analyzer():
    section("15. MINI PROJECT - STUDENT MARKS ANALYZER")

    # 5 students × 4 subjects
    marks = np.array([
        [85, 78, 92, 88],
        [70, 82, 75, 80],
        [95, 91, 89, 94],
        [60, 65, 70, 68],
        [88, 90, 84, 86]
    ])

    subjects = [
        "Math",
        "Physics",
        "Computer Science",
        "Data Science"
    ]

    print("Marks:")
    print(marks)

    # Average of each student
    student_average = np.mean(marks, axis=1)

    print("\nAverage of each student:")
    print(student_average)

    # Average of each subject
    subject_average = np.mean(marks, axis=0)

    print("\nAverage of each subject:")

    for subject, average in zip(subjects, subject_average):
        print(f"{subject}: {average:.2f}")

    # Overall average
    print("\nOverall average:", np.mean(marks))

    # Highest mark
    print("Highest mark:", np.max(marks))

    # Lowest mark
    print("Lowest mark:", np.min(marks))

    # Students above 80
    print("\nStudents with average > 80:")
    print(student_average > 80)

    # Pass / Fail
    pass_fail = np.where(
        student_average >= 50,
        "Pass",
        "Fail"
    )

    print("\nPass / Fail:")
    print(pass_fail)

    # Best student
    best_student = np.argmax(student_average) + 1

    print("\nBest student: Student", best_student)

    # Best subject
    best_subject_index = np.argmax(subject_average)

    print("Best subject:", subjects[best_subject_index])


# ============================================================
# 16. INTERVIEW QUESTIONS
# ============================================================

def interview_questions():
    section("16. NUMPY INTERVIEW QUESTIONS")

    questions = [
        "1. What is NumPy?",
        "2. Why is NumPy faster than Python lists?",
        "3. What is an ndarray?",
        "4. What are shape, ndim, size and dtype?",
        "5. What is vectorization?",
        "6. What is broadcasting?",
        "7. Explain axis=0 and axis=1.",
        "8. How do you filter a NumPy array?",
        "9. What does reshape() do?",
        "10. Difference between copy() and assignment?"
    ]

    for question in questions:
        print(question)


# ============================================================
# 17. PRACTICE
# ============================================================

def practice():
    section("17. YOUR PRACTICE")

    print("""
Try these yourself:

1. Create an array:
       [10, 20, 30, 40, 50]

   Find:
   - mean
   - maximum
   - minimum

2. Create:
       [10, 25, 40, 55, 70]

   Print only numbers greater than 30.

3. Create a 3 × 3 matrix.

   Find:
   - row averages
   - column averages

4. Create marks:
       [35, 80, 45, 90, 65]

   Use np.where() to print:
       Pass / Fail

5. Create numbers 1 to 12 and reshape them
   into a 3 × 4 matrix.
""")


# ============================================================
# MAIN
# ============================================================

def main():

    print("""
============================================================
                 NUMPY QUICK COURSE
============================================================

Goal:
Learn the NumPy concepts required for Data Science
and Machine Learning.

After this → Pandas
""")

    basics()
    creating_arrays()
    properties()
    indexing_slicing()
    vectorization()
    aggregation()
    axis()
    filtering()
    where_example()
    reshape_example()
    broadcasting()
    random_numbers()
    sorting()
    copy_example()

    student_marks_analyzer()

    interview_questions()
    practice()

    section("NUMPY COMPLETE")

    print("""
You should now know:

✓ NumPy arrays
✓ Creating arrays
✓ shape / ndim / size / dtype
✓ Indexing and slicing
✓ Vectorization
✓ Aggregation
✓ Axis
✓ Filtering
✓ np.where()
✓ Reshaping
✓ Broadcasting
✓ Random numbers
✓ Sorting
✓ Copying arrays

Next:
                    PANDAS
""")


if __name__ == "__main__":
    main()