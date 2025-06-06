from dotenv import load_dotenv
from os.path import join
from os import getenv

load_dotenv()

model_filename = getenv('BASE_MODEL_NAME', 'model_2024_08.pkl')
dataset_path = getenv('DATASET_PATH', join('data', 'df_new.csv'))
model_path = getenv('MODEL_PATH', join('api', 'models', model_filename))
preprocessor_path = getenv('PREPROCESSOR_PATH', join('api', 'models', 'preprocessor.pkl'))

