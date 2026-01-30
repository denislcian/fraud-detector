from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report, confusion_matrix
import joblib 

class FraudModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    def train(self, X, y):
        """Entrena el modelo con los datos balanceados (SMOTE)."""
        self.model.fit(X, y)

    def evaluate(self, X_test, y_test):
        """Evalúa el desempeño enfocándose en el RECALL."""
        y_pred = self.model.predict(X_test)
        print("--- Reporte de Clasificación ---")
        print(classification_report(y_test, y_pred))
        return confusion_matrix(y_test, y_pred)

    def save_model(self, path='models/fraud_model.pkl'):
        """Guarda el modelo entrenado para usarlo en la App."""
        joblib.dump(self.model, path)