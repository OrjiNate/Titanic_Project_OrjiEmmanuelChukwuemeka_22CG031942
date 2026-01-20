import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
import joblib

# 1. Load the dataset (Ensure you have train.csv in your model folder)
# You can download it from Kaggle or use a public URL:
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 2. Feature Selection (5 features)
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare']
X = df[features]
y = df['Survived']

# 3. Preprocessing Pipeline
# We must handle missing 'Age' and encode 'Sex'
num_features = ['Age', 'SibSp', 'Fare']
cat_features = ['Sex']

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# 4. Create and Train Model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
print("Titanic Survival Classification Report:")
print(classification_report(y_test, model.predict(X_test)))

# 6. Save Model
joblib.dump(model, 'model/titanic_survival_model.pkl')
print("Model saved as model/titanic_survival_model.pkl")