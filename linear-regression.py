#!/usr/bin/env python
# coding: utf-8

# ## Understanding the Dataset

# In[4]:





# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[8]:


df=pd.read_csv("data/train_aiproject.csv")
df


# In[9]:


print("Dimensions of dataset:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())   
print("\nFirst five rows:")
print(df.head())
print("\nData types and missing-value overview:")
print(df.info())
print("\nSummary of numeric columns:")
print(df.describe())


# In[10]:


df=df.drop_duplicates().copy()
target_column="Signal Strength (dBm)"
feature_columns=["SNR","Attenuation","Distance to Tower (km)","Environment"]
missing_columns=[
    col for col in feature_columns+[target_column]
    if col not in df.columns
]
if missing_columns:
    print("Error: These columns do not exist:", missing_columns)
else:
    print("Target:", target_column)
    print("Features:", feature_columns)
    print(df[[target_column] + feature_columns].head())


# In[11]:


df.to_csv("data/cleaned_data.csv",index=False)


# In[12]:


from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
model = df[feature_columns + [target_column]].copy()
model[target_column] = pd.to_numeric(model[target_column], errors="coerce")
model= model.dropna(subset=[target_column])
X = model[feature_columns]
y = model[target_column]
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))])

# Fill missing categories with the most frequent category, then encode them
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))])

# Apply the appropriate preprocessing to each column type
preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)])

print("Input shape:", X.shape)
print("Target shape:", y.shape)
print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)
print("Rows available for modelling:", len(model))


# ## Model Training

# In[13]:


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("Number of Training examples:", len(x_train))
print("Number of Testing examples:", len(x_test))


# In[14]:


model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(x_train, y_train)

print("Model training is complete!")


# ## Model Testing
# 

# In[15]:


y_pred = model.predict(x_test)

print("Predictions generated successfully!")


# In[16]:


results = pd.DataFrame({
    "Actual Signal Strength": np.array(y_test).flatten(),
    "Predicted Signal Strength": y_pred,
    "Error (Actual - Predicted)": y_test.to_numpy() - y_pred
})

print(results.head(10))


# In[17]:


print("Number of test samples:", len(x_test))
print("Number of predictions:", len(y_pred))


# In[22]:


timestamp = "2022-05-15 14:30:00"
snr = 20.5
call_duration = 500
environment = "urban"
attenuation = 8.5
distance_to_tower = 3.5
tower_id = 5
user_id = 25
call_type = "voice"
incoming_outgoing = "incoming"

new_record = pd.DataFrame({
    "Timestamp": [timestamp],
    "SNR": [snr],
    "Call Duration (s)": [call_duration],
    "Environment": [environment],
    "Attenuation": [attenuation],
    "Distance to Tower (km)": [distance_to_tower],
    "Tower ID": [tower_id],
    "User ID": [user_id],
    "Call Type": [call_type],
    "Incoming/Outgoing": [incoming_outgoing]
})

predicted_signal = model.predict(new_record)

print("Predicted Signal Strength (dBm):", predicted_signal[0])


# In[18]:


results.to_csv("testing_results.csv", index=False)

print("Testing results saved successfully!")


# In[19]:


import matplotlib.pyplot as plt 

plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Signal Strength (dBm)")
plt.ylabel("Predicted Signal Strength (dBm)")
plt.title("Actual vs Predicted Signal Strength")
plt.savefig("actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.show()


# ## Error Analysis

# In[20]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

error_metrics = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'RMSE', 'R2 Score'],
    'Value': [mae, mse, rmse, r2]
})

print(error_metrics)

results = pd.DataFrame({
    'Actual Signal Strength': np.array(y_test).flatten(),
    'Predicted Signal Strength': np.array(y_pred).flatten(),
    'Error (Actual - Predicted)': np.array(y_test).flatten() - np.array(y_pred).flatten()
})

print(results.head(10))

