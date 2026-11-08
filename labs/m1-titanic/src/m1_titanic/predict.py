import pandas as pd
import joblib


def predict_survival():
    # Charger le modèle
    model = joblib.load("models/titanic_model.pkl")

    # Créer les données d'un passager
    passenger = pd.DataFrame([{
        "Pclass": 3,
        "Sex": "female",
        "Age": 25,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }])

    # Encoder les variables catégorielles
    passenger = pd.get_dummies(
        passenger,
        columns=["Sex", "Embarked"],
        drop_first=True
    )

    # Aligner les colonnes avec celles de l'entraînement
    passenger = passenger.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # Effectuer la prédiction
    prediction = model.predict(passenger)

    if prediction[0] == 1:
        print("Le passager pourrait survivre.")
    else:
        print("Le passager pourrait ne pas survivre.")


if __name__ == "__main__":
    predict_survival()
