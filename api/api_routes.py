from fastapi import APIRouter, HTTPException, UploadFile, Form
from loguru import logger
from os.path import join
import pandas as pd
import joblib

from .config import dataset_path, model_path, preprocessor_path
from .schemas import Profile
from .models.models import model_predict, train_model
from .modules.preprocess import split, preprocessing
from .modules.evaluate import evaluate_performance

router = APIRouter(prefix='/api')

try:
    model = joblib.load(model_path)
except Exception as error:
    logger.error(f'Error loading model from {model_path}: {error}')

try:
    preprocessor = joblib.load(preprocessor_path)
except Exception as error:
    logger.error(f'Error loading preprocessor from {preprocessor_path}: {error}')

@router.get('/health')
async def health():
    return {'is_running': True}

@router.post('/predict')
async def predict(profile: Profile):
    try:
        processed_profile = preprocessor.transform(pd.DataFrame([profile.model_dump()]))
        prediction = round(model_predict(model, processed_profile)[0],2)
        logger.info(f'prediction: {prediction} avec le profile suivant : {profile}')
        return {'prediction': str(prediction)}
    except HTTPException as error:
        logger.error(f'Something went wrong during prediction: {error}, with profile: {str(prediction)}')
        raise HTTPException(status_code=500, detail='Something went wrong during prediction: {error}')

@router.post('/retrain')
async def retrain(epochs: int = Form(...), file: UploadFile | None = None):
    logger.debug(f"Starting retraining process with {epochs} epochs (0/5)")
    df = None
    if file:
        logger.info(f"File received: {file.filename}")
        df = pd.read_csv(file.file)
        file.file.close()
        logger.info(f"Uploaded dataset shape: {df.shape}")
    else:
        logger.info("No file received for retraining.")

    retraining_dataset = df if df is not None else pd.read_csv(dataset_path)

    try:
        logger.debug('Starting preprocessing.. (1/5)')
        X, y, _ = preprocessing(retraining_dataset)

        logger.debug('Starting spliting... (2/5)')
        X_train, X_test, y_train, y_test = split(X, y)
    except Exception as error:
        logger.error('Something went wrong during preprocessing: {error}')
        raise HTTPException(status_code=500, detail=f'Something went wrong during preprocessing.')
    
    try:
        logger.debug('Starting training... (3/5)')
        model, _ = train_model(model, X_train, y_train, X_val=X_test, y_val=y_test)
    except Exception as error:
        logger.error('Something went wrong during training: {error}')
        raise HTTPException(status_code=500, detail=f'Something went wrong during training.')
    
    try:
        logger.debug(f'Updating model in memory... (4/5)')
        joblib.dump(model, model_path)
    except Exception as error:
        logger.error('Something went wrong while updating model in memory: {error}')
        raise HTTPException(status_code=500, detail=f'Something went wrong while updating model in memory.')
    
    try:
        logger.debug('Calculating new performances... (5/5)')
        y_pred = model_predict(model, X_test)
        perf = evaluate_performance(y_test, y_pred)
        logger.info(f'New model performances: {perf}')

        return {'success': True, 'message': f'Retraining process was successfull with R² = {perf.get('R²')}.'}
    except Exception as error:
        logger.error('Something went wrong while calculatin new model performances: {error}')
        raise HTTPException(status_code=500, detail=f'Something went wrong while calculatin new model performances.')
