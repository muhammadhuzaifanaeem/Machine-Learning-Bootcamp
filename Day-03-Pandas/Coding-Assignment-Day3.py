# Coding Assignment

# Q1. Create a DataFrame with: Name- Age - City
import pandas as pd

df = pd.DataFrame({
    'Name': ['Huzaifa', 'Ali', 'Talha', 'Aseef', 'Umar'],
    'Age': [21, 24, 20, 19, 16],
    'City': ['Chichawatni', 'Lahore', 'Rahimyar Khan', 'Faisalabad', 'Islamabad']
})

# 2. Display: - Head - Tail - Shape - Columns - Data Types
print("Head:", df.head(2))
print("\nTail:", df.tail(2))
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData Types:", df.dtypes)

# 3. Filter people older than 20.
print(df[df['Age'] > 20])

# 4. Sort by Age.
print(df.sort_values(by='Age'))
print(df.sort_values(by='City'))


# 5. Save the DataFrame as `students.csv`.
df.to_csv('students.csv', index=False)
