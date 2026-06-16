import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/student_exam_performance.csv")

# Select only numeric columns
corr = df.select_dtypes(include='number').corr()

plt.figure(figsize=(12, 8))

plt.imshow(corr, cmap='coolwarm')

plt.colorbar()

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=45,
    ha='right'
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig("images/correlation_matrix.png")

plt.show()