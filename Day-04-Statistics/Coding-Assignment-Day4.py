# Assignment – Day 4

# Coding

# 1. Create a DataFrame with marks for 10 students.

import pandas as pd

students = pd.DataFrame({
    "Student": ["Ali", "Sara", "Ahmed", "Ayesha", "Bilal", "Fatima", "Huzaifa", "Bilal", "Mariya", "Hassan"],
    "Marks": [78, 92, 85, 88, 73, 91, 75, 81, 83, 500]
})

# 2. Calculate:- Mean - Median - Variance - Standard Deviation -- Add one extreme value (for example, 500) -- Observe how the mean and median change.

print("Mean:", students["Marks"].mean())
print("Median:", students["Marks"].median())
print("Variance:", students["Marks"].var())
print("Std Dev:", students["Marks"].std())

# 5. Write your observations.
# The mean is significantly affected by the extreme value (500), while the median remains relatively stable.
