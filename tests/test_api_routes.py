import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from api.main import app  
from api.schemas import Profile
from api.config import model_path, preprocessor_path 

client = TestClient(app)

@pytest.fixture
def mock_model():
    """Fixture pour un modèle mocké."""
    model = MagicMock()
    model.predict.return_value = np.array([[150000.0]]) 
    return model

@pytest.fixture
def mock_preprocessor():
    """Fixture pour un préprocesseur mocké."""
    preprocessor = MagicMock()
    
    preprocessor.transform.return_value = pd.DataFrame([[0.5, 0.2, 1.0]]) 
    return preprocessor

@pytest.fixture
def sample_profile_data():
    """Données d'exemple valides pour un profil."""
    return {
        "age": 30,
        "taille": 170.0,
        "poids": 70.0,
        "sexe": "H",
        "sport_licence": "oui",
        "niveau_etude": "master",
        "region": "Île-de-France",
        "smoker": "non",
        "nationalité_francaise": "oui",
        "revenu_estime_mois": 3000
    }

@pytest.fixture(autouse=True)
def mock_load_dependencies(mock_model, mock_preprocessor):
    """
    Mock joblib.load pour retourner les mocks du modèle et du préprocesseur
    et patch les variables globales 'model' et 'preprocessor' dans api_routes.
    Cette fixture s'appliquera automatiquement à tous les tests de ce module.
    """
    with patch('joblib.load') as mock_joblib_load, \
         patch('api.api_routes.model', new=mock_model) as mock_global_model, \
         patch('api.api_routes.preprocessor', new=mock_preprocessor) as mock_global_preprocessor:
        
        
        def side_effect_load(path):
            if path == model_path:
                return mock_model
            elif path == preprocessor_path:
                return mock_preprocessor
            raise FileNotFoundError(f"Path {path} not mocked for joblib.load")
        mock_joblib_load.side_effect = side_effect_load
        
        yield mock_joblib_load, mock_global_model, mock_global_preprocessor

def test_health_check():
    """Teste l'endpoint /health."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"is_running": True}

def test_predict_success(sample_profile_data, mock_preprocessor, mock_model):
    """Teste une prédiction réussie."""
    
    response = client.post("/api/predict", json=sample_profile_data)
    
    assert response.status_code == 200
    assert "prediction" in response.json()
    
    assert isinstance(response.json()["prediction"], str) 
    
    mock_preprocessor.transform.assert_called_once()
    mock_model.predict.assert_called_once()


def test_predict_invalid_data():
    """Teste une prédiction avec des données de profil invalides (FastAPI devrait gérer cela)."""
    invalid_profile_data = {"age": "trente"} 
    response = client.post("/api/predict", json=invalid_profile_data)
    assert response.status_code == 422  

@patch('api.api_routes.model_predict') 
def test_predict_processing_error(mock_model_predict_func, sample_profile_data):
    """Teste une erreur durant le traitement de la prédiction (après validation)."""
    mock_model_predict_func.side_effect = Exception("Test processing error")
    
    response = client.post("/api/predict", json=sample_profile_data)
    
    assert response.status_code == 500
    assert "detail" in response.json()
    
    assert "Something went wrong during prediction" in response.json()["detail"]
