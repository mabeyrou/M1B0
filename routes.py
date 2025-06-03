from fastapi import APIRouter, HTTPException
from loguru import logger

router = APIRouter(prefix='/api/webcam')

@router.get('/')
async def root():
    return {'message': 'The webcam object recognition server is up and running'}

@router.get('/health')
async def health():
    return {'message': 'health'}

@router.post('/predict')
async def predict():
    return {'message': 'predict'}

@router.post('/retrain')
async def retrain(data_path):
    return {'message': 'retrain'}

