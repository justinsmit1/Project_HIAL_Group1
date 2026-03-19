import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("trajectory_features.csv")
print(df.columns)
print("\nMean feature values:")
df = df.drop(columns=["clip"])
print(df.groupby("type").mean())

print("\nStd dev:")
print(df.groupby("type").std())




for col in df.columns:
    if col.startswith("f"):
        sns.boxplot(x="type", y=col, data=df)
        plt.title(col)
        plt.show()