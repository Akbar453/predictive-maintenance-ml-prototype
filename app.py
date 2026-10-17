"""Synthetic classification baseline. NOT a claimed real-world accuracy."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

def main():
    rng = np.random.default_rng(42)
    n = 1200
    temp = rng.normal(65, 12, n)
    vibration = rng.normal(4, 1.8, n)
    pressure = rng.normal(38, 7, n)
    X = np.column_stack([temp, vibration, pressure])
    risk = (temp > 74).astype(int) + (vibration > 5.5).astype(int) + (pressure < 30).astype(int)
    y = ((risk >= 2) | ((risk == 1) & (rng.random(n) < .07))).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
    model = RandomForestClassifier(n_estimators=100,random_state=42,class_weight="balanced")
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    print("SYNTHETIC DATA ONLY — confusion matrix:\n",confusion_matrix(y_test,pred))
    print(classification_report(y_test,pred,zero_division=0))
    if len(set(y_test)) > 1:
        print("ROC AUC:",round(roc_auc_score(y_test,model.predict_proba(X_test)[:,1]),3))

if __name__ == "__main__":main()
