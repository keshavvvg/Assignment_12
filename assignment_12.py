import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. LOAD DATA
df = pd.read_csv("customer_churn.csv")

# 2. DATA CLEANING & MISSING VALUES
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '')

# Identify totalcharges column dynamically
total_col = 'totalcharges' if 'totalcharges' in df.columns else 'total_charges'
monthly_col = 'monthlycharges' if 'monthlycharges' in df.columns else 'monthly_charges'

if total_col in df.columns:
    df[total_col] = pd.to_numeric(df[total_col], errors='coerce')

# Impute missing values (using 'str' to silence Pandas 3+ deprecation warning)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = df.select_dtypes(include=['str', 'category']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode churn target (1/0) and remove duplicates
df['churn_binary'] = np.where(df['churn'].astype(str).str.strip().str.title() == 'Yes', 1, 0)
df = df.drop_duplicates()

# 3. SUMMARY STATISTICS
print("--- Dataset Summary ---")
print(f"Shape: {df.shape}")
print(f"Overall Churn Rate: {df['churn_binary'].mean() * 100:.2f}%\n")

print("--- Numerical Metrics ---")
print(df[num_cols].describe().T[['mean', 'std', '50%', 'max']])

print("\n--- Summary by Churn Status ---")
print(df.groupby('churn').agg(
    avg_tenure=('tenure', 'mean'),
    avg_monthly_charges=(monthly_col, 'mean'),
    customer_count=('churn_binary', 'count')
).reset_index())

# 4. VISUALIZATIONS (Pure Matplotlib)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. Churn Count Bar Chart
churn_counts = df['churn'].value_counts()
axes[0, 0].bar(churn_counts.index, churn_counts.values, color=['#2ca02c', '#d62728'])
axes[0, 0].set_title('Churn Count')

# 2. Monthly Charges Histogram by Churn
for churn_val, color in zip(df['churn'].unique(), ['#2ca02c', '#d62728']):
    subset = df[df['churn'] == churn_val]
    axes[0, 1].hist(subset[monthly_col], alpha=0.5, label=f'Churn: {churn_val}', bins=20)
axes[0, 1].set_title('Monthly Charges Distribution')
axes[0, 1].legend()

# 3. Tenure Boxplot by Churn
data_to_plot = [df[df['churn'] == val]['tenure'] for val in df['churn'].unique()]
axes[1, 0].boxplot(data_to_plot, tick_labels=df['churn'].unique())
axes[1, 0].set_title('Tenure by Churn')

# 4. Correlation Heatmap
corr_cols = num_cols + (['churn_binary'] if 'churn_binary' not in num_cols else [])
corr = df[corr_cols].corr()
cax = axes[1, 1].matshow(corr, cmap='coolwarm')
fig.colorbar(cax, ax=axes[1, 1])
axes[1, 1].set_xticks(range(len(corr_cols)))
axes[1, 1].set_yticks(range(len(corr_cols)))
axes[1, 1].set_xticklabels(corr_cols, rotation=45, ha='left')
axes[1, 1].set_yticklabels(corr_cols)
axes[1, 1].set_title('Correlation Matrix', pad=20)

plt.tight_layout()
plt.show()


#OUTPUT:


'''
--- Dataset Summary ---
Shape: (100, 10)
Overall Churn Rate: 40.00%

--- Numerical Metrics ---
                     mean          std       50%      max
age               43.9300    14.411173    43.500    70.00
tenure            36.6100    22.585816    37.000    72.00
monthlycharges    70.2227    28.792519    68.545   119.46
totalcharges    2622.3149  2159.684502  2260.040  8175.24

--- Summary by Churn Status ---
  churn  avg_tenure  avg_monthly_charges  customer_count
0    No      36.400            67.163417              60
1   Yes      36.925            74.811625              40
'''