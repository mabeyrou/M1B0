import streamlit as st
from loguru import logger
import requests
import json

logger.remove()

logger.add("logs/dev_streamlit.log",
          rotation="10 MB",
          retention="7 days",
          compression="zip",
          level="TRACE",
          enqueue=True,
          format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

API_URL = 'http://localhost:8000/api'

def predict(form_data):
    response = requests.post(url=f'{API_URL}/predict', json=form_data)

    return response.json()


def main():
    st.header('Prédiction du montant d\'un prêt')

    with st.form('loan_prediction'):
        age = st.number_input(label='Age', min_value=18, max_value=100, step=1, value=30)
        taille = st.number_input(label='Taille', min_value=float(100), max_value=float(220), step=0.1, value=float(170))
        poids = st.number_input(label='Poids', min_value=float(30), max_value=float(300), step=0.1, value=float(60))
        sexe = st.radio(label='Genre', options=['H','F'])
        sport_licence = st.toggle(label='Licence de sport')
        niveau_etude = st.selectbox(label='Niveau d\'étude', options=['bac+2', 'bac', 'master', 'doctorat', 'aucun'])
        region = st.selectbox(label='Région', options=['Normandie', 'Occitanie', 'Bretagne', 'Hauts-de-France', 'Corse',
        'Auvergne-Rhône-Alpes', 'Île-de-France',
        'Provence-Alpes-Côte d\'Azur'])
        smoker = st.toggle(label='Fumeur')
        nationalité_francaise = st.toggle(label='Nationalité française')
        revenu_estime_mois = st.number_input(label='Revenu mensuel (€)', min_value=0, max_value=100000, step=100, value=1700)

        submitted = st.form_submit_button('Envoyer')

        if submitted:
            form_data = {
                'age': age,
                'taille': taille,
                'poids': poids,
                'sexe': sexe,
                'sport_licence': 'oui' if sport_licence else 'non',
                'niveau_etude': niveau_etude,
                'region': region,
                'smoker': 'oui' if smoker else 'non',
                'nationalité_francaise': 'oui' if nationalité_francaise else 'non',
                'revenu_estime_mois': revenu_estime_mois
            }
            result = predict(form_data)
            prediction = result.get('prediction')
            st.metric(label='Résultat',value=f'{prediction}€')
        
        sidebar = st.sidebar
        # sidebar.button=

if __name__ == "__main__":
    main()
