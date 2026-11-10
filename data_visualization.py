#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder




# In[12]:


df=pd.read_csv("data/cleaned_data.csv")
df.head()


# In[13]:


df.columns.to_list()


# In[14]:


df.info()


# In[15]:


df.isnull().sum()


# In[16]:


# Cell 3: Inspect the dataset

print("Dataset dimensions:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nData types and missing-value overview:")
df.info()

print("\nSummary of numeric columns:")
print(df.describe())


# In[17]:


numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
print("Numeric columns:", numeric_columns)

# Plotting distribution of the first continuous feature
column_to_plot = "Signal Strength (dBm)"

plt.figure(figsize=(8, 4))
plt.hist(df[column_to_plot].dropna(), bins=20, edgecolor="black")
plt.title(f"Distribution of {column_to_plot}")
plt.xlabel(column_to_plot)
plt.ylabel("Frequency")
plt.grid(alpha=0.25)
plt.show()



# In[18]:


feature_for_plot = "Distance to Tower (km)"
target_for_plot = "Signal Strength (dBm)"

plt.figure(figsize=(8, 5))
plt.scatter(df[feature_for_plot], df[target_for_plot], alpha=0.65, color="blue")
plt.title(f"{target_for_plot} vs {feature_for_plot}")
plt.xlabel(feature_for_plot)
plt.ylabel(target_for_plot)
plt.grid(alpha=0.25)
plt.show()


# In[19]:


df = df.drop_duplicates().copy()

# Target for regression (e.g., predicting Signal Strength)
target_column = "Signal Strength (dBm)"

# Select feature columns (excluding Timestamp, ID columns, and target)
feature_columns = [
    "SNR",
    "Call Duration (s)",
    "Environment",
    "Attenuation",
    "Distance to Tower (km)",
    "Call Type",
    "Incoming/Outgoing"
]

print("Target Column:", target_column)
print("Feature Columns:", feature_columns)
print(df[[target_column] + feature_columns].head())


# In[20]:


# Separate features (X) and target (y)
X = df[feature_columns].copy()
y = df[target_column].copy()

# Identify numerical and categorical column groups
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

# Define numerical and categorical transformers
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessors
preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])

print("Input shape (X):", X.shape)
print("Target shape (y):", y.shape)
print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)
print("Rows available for modeling:", len(df))


# In[21]:


import matplotlib.pyplot as plt
import pandas as pd

# 1. Load your dataset first
df = pd.read_csv("train.csv")  # Replace "train.csv" with your actual file path if different

# 2. Plot the boxplot
df.boxplot(
    column='Signal Strength (dBm)', 
    by='Environment', 
    patch_artist=True,
    boxprops=dict(facecolor='skyblue', color='navy'),
    medianprops=dict(color='red', linewidth=2)
)

plt.suptitle('')  # Removes default pandas title
plt.xlabel('Environment Type')
plt.ylabel('Signal Strength (dBm)')
plt.grid(alpha=0.3)
plt.show()


# In[22]:


import matplotlib.pyplot as plt
import numpy as np

# Scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(df['SNR'], df['Signal Strength (dBm)'], alpha=0.6, color='teal', label='Measurement')

# Linear Trendline Calculation
z = np.polyfit(df['SNR'], df['Signal Strength (dBm)'], 1)
p = np.poly1d(z)
plt.plot(df['SNR'].sort_values(), p(df['SNR'].sort_values()), color='darkred', linestyle='--', linewidth=2, label='Linear Fit')

plt.title('Signal Strength vs. Signal-to-Noise Ratio (SNR)')
plt.xlabel('SNR (dB)')
plt.ylabel('Signal Strength (dBm)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# In[24]:


import matplotlib.pyplot as plt

# Helper function to assign standard RF quality bands
def categorize_signal(dbm):
    if  dbm >= -70:
        return 'Excellent (>= -70 dBm)'
    elif dbm >= -85:
        return 'Good (-70 to -85 dBm)'
    elif dbm >= -100:
        return 'Fair (-85 to -100 dBm)'
    else:
        return 'Poor (< -100 dBm)'

df['Quality_Band'] = df['Signal Strength (dBm)'].apply(categorize_signal)
counts = df['Quality_Band'].value_counts().reindex([
    'Excellent (>= -70 dBm)', 'Good (-70 to -85 dBm)', 'Fair (-85 to -100 dBm)', 'Poor (< -100 dBm)'
]).fillna(0)

colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']

plt.figure(figsize=(8, 5))
bars = plt.bar(['Excellent', 'Good', 'Fair', 'Poor'], counts.values, color=colors, edgecolor='black', alpha=0.85)

# Annotate counts on top of bars
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{int(height)}',
 xy=(bar.get_x() + bar.get_width() / 2, height),
xytext=(0, 3),
textcoords="offset points",
 ha='center', va='bottom', fontweight='bold')

plt.title('Cellular Signal Quality Class Distribution')
plt.xlabel('Quality Threshold')
plt.ylabel('Total Samples')
plt.grid(axis='y', alpha=0.3)
plt.show()

