import pandas as pd

# Read the first CSV file, selecting only "Pattern String" and "Pattern Category" columns
df1 = pd.read_csv("dark_patterns_new.csv", usecols=["Pattern String", "Pattern Category"])

# Read the second CSV file, selecting only "Pattern String" and "Pattern Category" columns
df2 = pd.read_csv("dark_patterns.csv", usecols=["Pattern String", "Pattern Category"])

# Concatenate the two dataframes vertically
merged_df = pd.concat([df1, df2], ignore_index=True)

# Write the merged dataframe to a new CSV file
merged_df.to_csv("merged_file.csv", index=False)
