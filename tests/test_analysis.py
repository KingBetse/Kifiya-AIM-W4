import pandas as pd
import numpy as np

# Step 1: Create a small mock dataset
data = {
    'Store': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    'Date': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03',
                            '2021-01-01', '2021-01-02', '2021-01-03',
                            '2021-01-01', '2021-01-02', '2021-01-03']),
    'Sales': [200, 210, 215, 300, 290, 5000, 150, 160, 170],  # Includes an outlier (5000)
    'CompetitionDistance': [100, 100, np.nan, 200, 200, np.nan, 300, 300, 300]
}

# Convert to DataFrame
train_data = pd.DataFrame(data)

# Display the original DataFrame
print("Original DataFrame:")
print(train_data)

# Step 2: Handle Missing Data
# Fill missing CompetitionDistance with the mean
train_data['CompetitionDistance'].fillna(train_data['CompetitionDistance'].mean(), inplace=True)

# Step 3: Detect and Handle Outliers (using IQR)
Q1 = train_data['Sales'].quantile(0.25)
Q3 = train_data['Sales'].quantile(0.75)
IQR = Q3 - Q1

# Remove outliers
train_data = train_data[~((train_data['Sales'] < (Q1 - 1.5 * IQR)) | (train_data['Sales'] > (Q3 + 1.5 * IQR)))]

# Display the cleaned DataFrame
print("\nCleaned DataFrame:")
print(train_data)

# Step 4: Analyze Impact of Competitor Distance
# Analyze sales average based on CompetitionDistance
avg_sales_by_distance = train_data.groupby('CompetitionDistance')['Sales'].mean().reset_index()

# Display average sales by competition distance
print("\nAverage Sales by Competition Distance:")
print(avg_sales_by_distance)

# Step 5: Conclusion
for index, row in avg_sales_by_distance.iterrows():
    print(f"At a distance of {row['CompetitionDistance']} meters, the average sales are {row['Sales']:.2f}.")
