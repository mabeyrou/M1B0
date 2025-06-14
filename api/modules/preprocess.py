from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def split(X, y, test_size=0.2, random_state=42):
    """
    Divises le set de données en sous-ensembles de test et d'entrainement avec la méthode `train_test_split()` de scikitlearn
    - x_train : entrées du subset d'entrainement
    - y_train : sorties du subset d'entrainement
    - x_test : entrées du subset de test
    - y_test : sorties du subset de test
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def preprocessing(df):
    """
    Fonction pour effectuer le prétraitement des données :
    - Imputation des valeurs manquantes.
    - Standardisation des variables numériques.
    - Encodage des variables catégorielles.
    """
    numerical_cols = ["age", "taille", "poids", "revenu_estime_mois"]
    categorical_cols = ["sexe", "sport_licence", "niveau_etude", "region", "smoker", "nationalité_francaise"]

    # imputer : Pour les valeurs numériques, on complète les valeurs manquantes par la moyenne des valeurs présentes
    # scaler : Normalisation des données
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")), 
        ("scaler", StandardScaler())
    ])

    # imputer : Pour les valeurs catégorielles, on complète les valeurs manquantes par la valeur la plus fréquente
    # encoder : Transformation des catégories au format binaire (chaque catégorie correspond à une nouvelle colonne)
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols)
    ])

    # Prétraitement
    X = df.drop(columns=["nom", "prenom", "montant_pret"])
    y = df["montant_pret"]

    # Fit all transformers, transform the data and concatenate results.
    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor