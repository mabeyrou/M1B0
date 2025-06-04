from fastapi import APIRouter, HTTPException, Response
from loguru import logger
from os.path import join
import pandas as pd
import joblib

from .schemas import Profile
from .models.models import model_predict


router = APIRouter(prefix='/api')

model = joblib.load(join('api', 'models','model_2025_06_03.pkl'))
preprocessor = joblib.load(join('api', 'models', 'preprocessor.pkl'))

@router.get('/health')
async def health():
    return {'is_running': True}

@router.post('/predict')
async def predict(profile: Profile):
    processed_profile = preprocessor.transform(pd.DataFrame([profile.model_dump()]))
    prediction = round(model_predict(model, processed_profile)[0],2)
    logger.info(f'prediction: {prediction} avec le profile suivant : {profile}')
    return {'prediction': str(prediction)}

@router.post('/retrain')
async def retrain():
    return {'message': 'retrain'}

