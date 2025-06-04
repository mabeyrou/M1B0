from fastapi import APIRouter, HTTPException, Response
from schemas import Profile
from loguru import logger
import joblib
from os.path import join
from models.models import model_predict
import pandas as pd


router = APIRouter(prefix='/api')

model = joblib.load(join('models','model_2025_06_03.pkl'))
preprocessor = joblib.load(join('models', 'preprocessor.pkl'))

@router.get('/health')
async def health():
    return {'message': 'The server is up and running'}

@router.post('/predict')
async def predict(profile: Profile):
    processed_profile = preprocessor.transform(pd.DataFrame([profile.model_dump()]))
    prediction = round(model_predict(model, processed_profile)[0],2)
    logger.info(f'prediction: {prediction} avec le profile suivant : {profile}')
    return {'prediction': str(prediction)}

@router.post('/retrain')
async def retrain():
    return {'message': 'retrain'}

