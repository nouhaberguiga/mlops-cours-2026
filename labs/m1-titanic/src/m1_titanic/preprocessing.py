import pandas as pd


def preprocess_data(input_path, output_path):
    # Charger le dataset
    df = pd.read_csv(input_path)

    # Supprimer la colonne Cabin
    df = df.drop(columns=["Cabin"])

    # Remplacer les valeurs manquantes de Age par la médiane
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # Remplacer les valeurs manquantes de Embarked par la valeur la plus fréquente
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Sauvegarder le dataset nettoyé
    df.to_csv(output_path, index=False)

    print("Nettoyage terminé !")
    print("Dimensions :", df.shape)
    print("Valeurs manquantes restantes :", df.isnull().sum().sum())


if __name__ == "__main__":
    preprocess_data(
        "data/raw/titanic.csv",
        "data/processed/titanic_clean.csv"
    )
