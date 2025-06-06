import streamlit as st
from loguru import logger
import requests

from api_client import predict, check_health, retrain

logger.remove()
logger.add("./streamlit/logs/dev_streamlit.log",
          rotation="10 MB",
          retention="7 days",
          compression="zip",
          level="TRACE",
          enqueue=True,
          format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

def initialize_session_state():
    if 'is_running' not in st.session_state:
        health_status = check_health()
        st.session_state.is_running = health_status.get('is_running', False)
    if 'checking_health' not in st.session_state:
        st.session_state.checking_health = False

def render_prediction_form():
    st.header('Prédiction du montant d\'un prêt')

    @st.dialog("Prédiction du prêt")
    def prediction_dialog(form_data, prediction):
        st.text('Avec ce profil :')
        st.json(form_data)
        st.text('On obtient ce montant de prêt :')
        st.metric(label='Résultat',value=f'{prediction}€')

    with st.form(key='loan_prediction', clear_on_submit=True):
        age = st.number_input(
            label='Age',
            min_value=18,
            max_value=100,
            step=1,
            value=30,
            disabled=not st.session_state.is_running
        )
        taille = st.number_input(
            label='Taille',
            min_value=float(100),
            max_value=float(220),
            step=0.1,
            value=float(170),
            disabled=not st.session_state.is_running
        )
        poids = st.number_input(
            label='Poids',
            min_value=float(30),
            max_value=float(300),
            step=0.1,
            value=float(60),
            disabled=not st.session_state.is_running
        )
        sexe = st.radio(
            label='Genre',
            options=['H','F'],
            disabled=not st.session_state.is_running
        )
        sport_licence = st.toggle(
            label='Licence de sport',
            disabled=not st.session_state.is_running
        )
        niveau_etude = st.selectbox(
            label='Niveau d\'étude',
            options=['bac', 'bac+2', 'master', 'doctorat', 'aucun'],
            disabled=not st.session_state.is_running
        )
        region = st.selectbox(
            label='Région',
            options=['Auvergne-Rhône-Alpes', 'Bretagne', 'Corse',
                     'Hauts-de-France', 'Île-de-France', 'Normandie',
                     'Occitanie', 'Provence-Alpes-Côte d\'Azur'],
            disabled=not st.session_state.is_running
        )
        smoker = st.toggle(
            label='Fumeur', disabled=not st.session_state.is_running)
        nationalité_francaise = st.toggle(
            label='Nationalité française', 
            disabled=not st.session_state.is_running
        )
        revenu_estime_mois = st.number_input(
            label='Revenu mensuel (€)',
            min_value=0,
            max_value=100000,
            step=100,
            value=1700,
            disabled=not st.session_state.is_running
        )

        submitted = st.form_submit_button(label='Envoyer',
                                          disabled=not st.session_state.is_running
                    )

        if submitted and st.session_state.is_running:
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
            try:
                result = predict(form_data)
                if result:
                    prediction = result.get('prediction')
                    prediction_dialog(form_data, prediction)
            except requests.exceptions.RequestException as error:
                st.error(f"Erreur lors de la prédiction : {error}")
        elif submitted and not st.session_state.is_running:
            st.error("L'API est hors ligne. Veuillez réessayer plus tard.")

def render_sidebar():
    with st.sidebar:
        st.subheader('Etat du modèle', divider=True)
        
        col_sidebar1, col_sidebar2 = st.columns([2,1])
        
        with col_sidebar2:
            if st.button(label='Vérifier', key='health_check_button'):
                st.session_state.checking_health = True
        
        with col_sidebar1:
            if st.session_state.checking_health:
                with st.spinner("Vérification..."):
                    health_status_updated = check_health()
                    st.session_state.is_running = health_status_updated.get('is_running', False)
                    st.session_state.checking_health = False
                    st.rerun()
            else:
                if st.session_state.is_running:
                    st.badge(
                        label='En ligne',
                        icon=':material/check:',
                        color='green'
                    )
                else:
                    st.badge(
                        label='Hors ligne',
                        icon=':material/error:',
                        color='red'
                    )
        
        with st.expander('💪 Réentrainer le model'):
            with_new_data = st.checkbox(
                label='Avec de nouvelles données ?',
                key='checkbox_retrain_new_data',
                disabled=not st.session_state.is_running
            )
            
            uploaded_file = None
            if with_new_data:
                uploaded_file = st.file_uploader(
                    label='Importer les données (.csv)',
                    type=['csv'],
                    key='file_uploader_retrain_data',
                    disabled=not st.session_state.is_running
                )

            with st.form('retrain_form', clear_on_submit=True, border=False):
                epochs = st.number_input(
                    label='Nombre d\'epochs',
                    value=50,
                    step=10,
                    min_value=10,
                    max_value=600,
                    key='epochs_retrain_input',
                    disabled=not st.session_state.is_running
                )
                retrain_submited = st.form_submit_button(
                    'Ré-entraîner',
                    disabled=not st.session_state.is_running or (with_new_data and uploaded_file is None)
                )

                if retrain_submited:
                    if with_new_data and uploaded_file is None:
                        st.warning("Veuillez sélectionner un fichier si vous cochez 'Avec de nouvelles données'.")
                    else:
                        with st.spinner("Réentraînement en cours..."):
                            result = retrain(uploaded_file if with_new_data else None, epochs)
                            if result and result.get("success", False):
                                st.success(result.get("message", "Modèle réentraîné avec succès!"))
                            else:
                                st.error(result.get("message", "Échec du réentraînement du modèle."))
                                
def main():
    st.set_page_config(page_title="Prédiction de Prêt")
    initialize_session_state()
    render_prediction_form()
    render_sidebar()
                                
if __name__ == "__main__":
    main()
