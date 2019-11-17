import pandas as pd
import numpy as np

missing_values = ["n/a", "NA", "-", "setosa", "versicolor", "virginica", "Versicolour"]
df = pd.read_csv("iris_with_errors.csv", na_values=missing_values)

print("Brakujące dane przed poprawą:")
print(df['sepal.length'].isnull().sum())
print(df['sepal.width'].isnull().sum())
print(df['petal.length'].isnull().sum())
print(df['petal.width'].isnull().sum())
print(df['variety'].isnull().sum())

median = df['sepal.length'].median()

# Poprawki
print("Brakujące dane z variety:")
median = df['sepal.length'].median()
df['sepal.length'].fillna(median, inplace=True)
median = df['sepal.width'].median()
df['sepal.width'].fillna(median, inplace=True)
median = df['petal.length'].median()
df['petal.length'].fillna(median, inplace=True)
median = df['petal.width'].median()
df['petal.width'].fillna(median, inplace=True)
median = df['sepal.length'].median()

cnt = 0
for row in df['sepal.length']:
    int(row)
    if df.loc[cnt, 'sepal.length'] > 30 or df.loc[cnt, 'sepal.length'] < 30:
        df.loc[cnt, 'sepal.length'] = median
    cnt += 1

cnt = 0
for row in df['variety']:
    str(row)
    if df.loc[cnt, 'variety'] != "Setosa" and df.loc[cnt, 'variety'] != "Versicolor" and df.loc[
        cnt, 'variety'] != "Virginica":
        if cnt == 0:
            df.loc[cnt, 'variety'] = df.loc[cnt + 1, 'variety']
        else:
            df.loc[cnt, 'variety'] = df.loc[cnt - 1, 'variety']

    cnt += 1
# -------------------------


print("Brakujące dane po poprawie:")
print(df['sepal.length'].isnull().sum())
print(df['sepal.width'].isnull().sum())
print(df['petal.length'].isnull().sum())
print(df['petal.width'].isnull().sum())
print(df['variety'].isnull().sum())
