

"""Train the final churn model and save it to model.joblib.
 
Usage:
    python train_model.py path/to/Churn_Modelling.csv
"""
import sys
 
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
 
csv_path = sys.argv[1] if len(sys.argv) > 1 else "Churn_Modelling.csv"
 
df = pd.read_csv(csv_path)
df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, errors="ignore")
df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)
 
X = df.drop("Exited", axis=1)
y = df["Exited"]
 
# 1) Quick check on a held-out split (same setup as the notebook)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
check = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)
proba = check.predict_proba(X_test)[:, 1]
print("Test ROC-AUC:", round(roc_auc_score(y_test, proba), 4))
print(classification_report(y_test, (proba >= 0.3).astype(int)))
 
# 2) Final model trained on all the data
model = GradientBoostingClassifier(random_state=42).fit(X, y)
 
joblib.dump({"model": model, "columns": list(X.columns)}, "model.joblib")
print("Saved model.joblib with columns:", list(X.columns))
 

