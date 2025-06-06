from dotenv import load_dotenv
from loguru import logger
from os import getenv
import requests
import time

load_dotenv()

API_URL = getenv('API_URL')

def check_health():
    time.sleep(1)
    try:
        response = requests.get(url=f'{API_URL}/health', timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as error:
        logger.error(f"API health check failed: {error}")
        return {"is_running": False, "status": "offline"}

def predict(form_data):
    try:    
        response = requests.post(url=f'{API_URL}/predict', json=form_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as error:
        logger.error(f"Erreur HTTP lors de la prédiction : {error}")
        return {"success": False, "message": f"Erreur HTTP lors de la prédiction: {error}"}
    except Exception as error:
        logger.error(f"Erreur lors de la prédiction : {error}")
        return {"success": False, "message": f"Erreur lors de la prédiction: {error}"}
    
def retrain(uploaded_file, num_epochs):
    files_payload = None
    data_payload = {'epochs': num_epochs}

    if uploaded_file is not None:
        files_payload = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        logger.info(f"Préparation de l'envoi du fichier {uploaded_file.name} pour le réentraînement avec {num_epochs} epochs.")
    else:
        logger.info(f"Réentraînement demandé sans nouvelles données, avec {num_epochs} epochs.")

    try:
        logger.debug(f"Data payload: {data_payload}, Files: {files_payload is not None}")
        response = requests.post(url=f'{API_URL}/retrain', files=files_payload, data=data_payload, timeout=3000)
        response.raise_for_status()
        logger.info(f"Réponse de l'API de réentraînement: {response.json()}")
        return response.json()
    except requests.exceptions.RequestException as error:
        logger.error(f"Erreur lors de la requête de réentraînement à l'API: {error}")
        return {"success": False, "message": str(error)}
    except Exception as error:
        logger.error(f"Erreur inattendue lors du réentraînement: {error}")
        return {"success": False, "message": f"Erreur inattendue: {str(error)}"}

