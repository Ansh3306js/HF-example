# Step 1:Import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Step 2: Create sample dataset (you can replace this with your own data)
data = pd.DataFrame({
    'age': [25, 30, np.nan, 35, 40, 28],
    'salary': [50000, 60000, 55000, np.nan, 65000, 70000],
    'city': ['Surat', 'Mumbai', 'Delhi', 'Surat', 'Delhi', 'Mumbai'],
    'purchased': [0, 1, 0, 1, 1, 0]
})

print("Original Data:\n", data)

# Step 3: Split into X (input) and y (output)
X = data.drop('purchased', axis=1)
y = data['purchased']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Define column types
num_cols = ['age', 'salary']
cat_cols = ['city']

# Step 6: Create preprocessing pipelines

# Numerical pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder())
])

# Combine both
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# Step 7: Create full pipeline (preprocessing + model)
model_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', LogisticRegression())
])

# Step 8: Train model
model_pipeline.fit(X_train, y_train)

# Step 9: Predict
y_pred = model_pipeline.predict(X_test)

# Step 10: Evaluate
print("\n--- Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))


print("\nFull Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Step 11: Cross-validation
scores = cross_val_score(model_pipeline, X, y, cv=3)

print("\nCross-validation scores:", scores)
print("Average CV score:", scores.mean())