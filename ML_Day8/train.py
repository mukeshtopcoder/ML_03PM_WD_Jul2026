# importing Libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Loading dataset
df = pd.read_csv("dataset/iris.csv")


# Encoding
df['species'] = df['species'].map( lambda val : list(df['species'].unique()).index(val) )

# Divide Data into X and y
X = df.drop(columns=['species'])
y = df['species']

# Scalling X(data) all column's value should be in a range
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(X)

# Model Training
model = RandomForestClassifier()
model.fit(X,y)

# Save Model
joblib.dump(model , "models/model.pkl")
joblib.dump(scaler , "models/scaler.pkl")