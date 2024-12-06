import pandas as pd
from matplotlib import pyplot as plt

# Load the dataset
file_path = r'C:\Users\fethi tech\Downloads\hepatitis (1).csv'

hepatitis_df = pd.read_csv(file_path)



# 1. Drop the feature "Sex" (if it exists, to avoid errors)
if 'sex' in hepatitis_df.columns:
    hepatitis_df_dropped = hepatitis_df.drop(columns=['sex'])
    print("\nDataset after dropping 'Sex' feature:")
else:
    hepatitis_df_dropped = hepatitis_df.copy()
    print("\n'Sex' feature not found in the dataset.")
if 'class' in hepatitis_df_dropped.columns and hepatitis_df_dropped['class'].dtype == 'object':
    hepatitis_df_dropped['class'] = hepatitis_df_dropped['class'].apply(lambda x: 1 if x == 'die' else 0)





hepatitis_df_dropped.select_dtypes(include=['float64', 'int64'])

categorical_columns = hepatitis_df_dropped.select_dtypes(include=['object']).columns
hepatitis_df_dropped = hepatitis_df_dropped.drop(columns=categorical_columns)

print(hepatitis_df_dropped.select_dtypes(include=['float64', 'int64']).shape[1])

print(hepatitis_df_dropped.corr()['class'].abs())
selected_features = hepatitis_df_dropped.corr()['class'].abs()[hepatitis_df_dropped.corr()['class'].abs() > 0.5].index.tolist()

print(selected_features)


