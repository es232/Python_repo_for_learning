"""
================================================================================
                       NUMPY MASTERCLASS & ANALYTICS MODULE
================================================================================

1. WHAT IS NUMPY?
-----------------
NumPy (Numerical Python) is the foundational library for scientific computing in
Python. At its core is the `ndarray` (N-dimensional array), an efficient, 
multidimensional array object designed for high-performance array operations.

2. WHY DO WE USE NUMPY INSTEAD OF NATIVE PYTHON LISTS?
------------------------------------------------------
- Contiguous Memory Allocation: Standard Python lists store pointers to objects
  scattered throughout memory. NumPy arrays store data in contiguous, unfragmented
  memory blocks of a single homogeneous data type (e.g., float64, int32).
- C-Engine Speed: Core NumPy routines are written in highly optimized C and Fortran.
- Vectorization & SIMD: Allows mathematical operations on millions of elements 
  simultaneously using Single Instruction, Multiple Data (SIMD) processor instructions,
  eliminating slow Python `for` loops.
- Memory Efficiency: Consumes significantly less memory compared to lists of objects.

3. WHERE IS NUMPY USED IN REAL-WORLD TECH?
------------------------------------------
- Machine Learning & Deep Learning: Underpins frameworks like PyTorch, TensorFlow,
  and Scikit-Learn for tensor manipulation and matrix operations.
- Computer Vision: Digital images are 3D NumPy arrays (Height x Width x Color Channels).
  OpenCV relies directly on NumPy arrays.
- Financial Modeling: Algorithmic trading, Monte Carlo simulations, risk modeling.
- Data Science & Signal Processing: Audio waveform processing, FFT (Fast Fourier Transform),
  and geospatial analysis.

================================================================================
HOW TO RUN THIS SCRIPT:
--------------------------------------------------------------------------------
1. Ensure Python 3.8+ and NumPy are installed:
       pip install numpy

2. Run directly from your terminal:
       python numpy_masterclass.py
================================================================================
"""

import numpy as np


def header(title: str):
    """Utility helper to format console section titles."""
    print("\n" + "=" * 70)
    print(f"   {title.upper()}")
    print("=" * 70)


def module_1_array_creation():
    """Module 1: Array Creation, Shapes, and Inspection."""
    header("Module 1: Array Creation, Shapes, and Data Types")

    # Creating 1D and 2D arrays manually
    vec = np.array([10, 20, 30, 40])
    mat = np.array([[1, 2, 3], [4, 5, 6]])

    print("[+] 1D Array (Vector):\n", vec)
    print("    Shape:", vec.shape, "| Dimensions:", vec.ndim, "| DataType:", vec.dtype)

    print("\n[+] 2D Array (Matrix):\n", mat)
    print("    Shape:", mat.shape, "| Dimensions:", mat.ndim, "| DataType:", mat.dtype)

    # Built-in creation functions
    zeros = np.zeros((2, 4))
    ones = np.ones((3, 3))
    ranged = np.arange(0, 20, 5)  # Start, Stop, Step
    spaced = np.linspace(0, 1, 5)  # 5 linearly spaced points between 0 and 1

    print("\n[+] Zeros (2x4):\n", zeros)
    print("[+] Ones (3x3):\n", ones)
    print("[+] Arange (0 to 20, step 5):", ranged)
    print("[+] Linspace (0 to 1, 5 items):", spaced)


def module_2_vectorization():
    """Module 2: Vectorization and Element-Wise Operations."""
    header("Module 2: Vectorization & Element-Wise Math")

    arr = np.array([1, 2, 3, 4, 5])

    print("[+] Original Array:          ", arr)
    print("[+] Element-wise Addition (+10):", arr + 10)
    print("[+] Element-wise Multiplication (*3):", arr * 3)
    print("[+] Element-wise Exponentiation (**2):", arr ** 2)

    # Array vs Array operations
    b = np.array([10, 20, 30, 40, 50])
    print("\n[+] Vector A:              ", arr)
    print("[+] Vector B:              ", b)
    print("[+] Vector Addition (A + B):", arr + b)


def module_3_indexing_slicing():
    """Module 3: 2D Indexing, Slicing, and Subgrids."""
    header("Module 3: Indexing and Slicing (2D Grids)")

    grid = np.array([
        [10, 20, 30, 40],
        [50, 60, 70, 80],
        [90, 100, 110, 120]
    ])

    print("[+] Full Grid (3x4):\n", grid)
    print("\n[+] Specific Element at Row 1, Col 2 (0-indexed):", grid[1, 2])
    print("[+] Entire First Row (grid[0, :]):               ", grid[0, :])
    print("[+] Entire Second Column (grid[:, 1]):           ", grid[:, 1])
    print("[+] Sub-grid (Top-Left 2x2):\n", grid[0:2, 0:2])


def module_4_boolean_masking():
    """Module 4: Boolean Masking and Data Filtering."""
    header("Module 4: Boolean Masking & Conditional Filtering")

    scores = np.array([45, 88, 92, 31, 75, 60, 99, 42])
    print("[+] All Test Scores:", scores)

    # Step 1: Create a boolean mask condition
    mask = scores >= 70
    print("[+] Boolean Mask (Score >= 70):\n   ", mask)

    # Step 2: Pass mask to array to filter matching values
    passing_scores = scores[mask]
    print("[+] Filtered Passing Scores:", passing_scores)


def mini_project_sensor_analytics():
    """
    REAL-WORLD MINI PROJECT: Weather Sensor Analytics System
    
    Scenario:
    We are monitoring temperature data collected from 4 weather sensors over
    5 consecutive hours. We need to analyze, filter, and normalize the dataset.
    """
    header("Mini-Project: Weather Sensor Analytics System")

    # Seed for reproducibility
    np.random.seed(42)

    # Generate random sensor temperatures between 18.0°C and 35.0°C
    temperatures = np.random.uniform(18.0, 35.0, size=(4, 5)).round(1)

    print("[+] RAW DATASET - Sensor Temperatures Grid (4 Sensors x 5 Hours):")
    print(temperatures)
    print(f"    Grid Shape: {temperatures.shape} (Rows=Sensors, Cols=Hours)")

    # -------------------------------------------------------------------------
    # Task 1: Overall Mean Temperature
    # -------------------------------------------------------------------------
    overall_mean = np.mean(temperatures)
    print(f"\n[Task 1] Overall Mean Temperature: {overall_mean:.2f}°C")

    # -------------------------------------------------------------------------
    # Task 2: Sensor Averages (Across Hours -> along columns, axis=1)
    # -------------------------------------------------------------------------
    sensor_averages = np.mean(temperatures, axis=1)
    print("\n[Task 2] Average Temperature per Sensor:")
    for idx, avg in enumerate(sensor_averages):
        print(f"         - Sensor {idx + 1}: {avg:.2f}°C")

    # -------------------------------------------------------------------------
    # Task 3: Heatwave Check (Filter values > 30.0°C)
    # -------------------------------------------------------------------------
    heatwave_mask = temperatures > 30.0
    heatwave_readings = temperatures[heatwave_mask]
    print(f"\n[Task 3] Heatwave Check (> 30.0°C):")
    print(f"         - Extreme Heat Readings Found: {heatwave_readings}")
    print(f"         - Total Readings Exceeding Threshold: {len(heatwave_readings)}")

    # -------------------------------------------------------------------------
    # Task 4: Normalization (Z-score Scaling: (X - mean) / std)
    # -------------------------------------------------------------------------
    std_dev = np.std(temperatures)
    normalized_temperatures = (temperatures - overall_mean) / std_dev

    print("\n[Task 4] Feature Scaling / Data Normalization:")
    print("         Formula: Z = (X - Mean) / StdDev")
    print("         Overall Standard Deviation:", round(std_dev, 2))
    print("         Normalized Temperature Matrix (Mean = 0, Std = 1):\n")
    print(normalized_temperatures.round(2))


def main():
    """Main execution function running all modules sequentially."""
    print("Starting NumPy Masterclass...")
    module_1_array_creation()
    module_2_vectorization()
    module_3_indexing_slicing()
    module_4_boolean_masking()
    mini_project_sensor_analytics()
    
    header("NumPy Masterclass Completed Successfully")
    print("\nGreat job! You now understand the core mechanics of NumPy.")
    print("Next stop: Pandas for structured tabular data analysis!\n")


if __name__ == "__main__":
    main()