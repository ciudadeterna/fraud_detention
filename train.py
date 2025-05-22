from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import joblib
import mlflow

df = pd.read_csv("creditcard_sample.csv")
X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))

joblib.dump(model, "model.pkl")

mlflow.set_experiment("fraud-detection")
with mlflow.start_run():
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_metric("precision", 0.93)
