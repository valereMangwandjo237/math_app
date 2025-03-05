import streamlit as st
from streamlit_option_menu import option_menu
import requests
import joblib
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


col1, col2 = st.columns([1, 3])  # Colonne 1 pour la navbar (1/4), Colonne 2 pour le contenu (3/4)

with st.sidebar:
    selected = option_menu(
        menu_title = "Main menu",
        options = ["Acceuil", "EDA", "Prediction", "Contact"],
        icons = ["house","graph-up-arrow", "database", "envelope"],
        menu_icon = "cast",
        default_index = 0,
    )  



# Fonction pour envoyer les données à l'API
def send_data_to_api(data):
    response = predict(data)
    return response.json()

def load_model():
    # Charger le modèle
    model = joblib.load('model_classifier_iris.pkl')
    scale = joblib.load("scaler.pkl")

    return model, scale

def predict(data):
    model, scale = load_model()
    try:
        
        # Extraire les caractéristiques
        sepal_length = data['sepal_length']
        sepal_width = data['sepal_width']
        petal_length = data['petal_length']
        petal_width = data['petal_width']
        
        # Créer un tableau NumPy des caractéristiques
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Convertir en DataFrame
        X_train = pd.DataFrame(features, columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'])

        # Normalisation
        X_train_normalized = scale.transform(X_train)
        
        # Faire la prédiction
        prediction = model.predict(X_train_normalized)
        
        # Renvoie la prédiction
        return prediction[0]
        
    except Exception as e:
        return 


def front_iris():
    st.markdown("<h1 style='text-align: center;'>PREDICTION DU TYPE DE FLEUR D'IRIS</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])  # Colonne 1 pour la navbar (1/4), Colonne 2 pour le contenu (3/4)
    # Navbar verticale dans la colonne de gauche


    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.slider("Longueur du sépal", 0.0, 10.0, value=0.0, step=0.1)
        countries = [country.name for country in pycountry.countries]
        # Liste déroulante pour sélectionner un pays
        selected_country = st.selectbox("Choisissez un pays :", countries)

    with col2:
        sepal_width = st.slider("Largeur du sépal", 0.0, 10.0, value=0.0, step=0.1)

    # Deuxième ligne avec deux autres curseurs
    col3, col4 = st.columns(2)

    with col3:
        petal_length = st.slider("Longueur du pétale", 0.0, 10.0, value=0.0, step=0.1)

    with col4:
        petal_width = st.slider("Largeur du pétale", 0.0, 10.0, value=0.0, step=0.1)


    # Bouton pour envoyer les données à l'API
    if st.button("Prédire la fleur...", help="Cliquez pour envoyer les données", type="primary"):
        data = {
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
        }
        
        # Envoyer les données à l'API
        response = predict(data)
        #model, sca = load_model()
        
        # Afficher la réponse de l'API
        st.write("<p style='font-size: 20px; font-weight: bold;'>Votre fleur semble être : <span style='color: #ff4b4b;'>", response , "</span></p>", unsafe_allow_html=True)
        if response == "setosa":
            st.image("images/setosa.jpg", width=300)
        elif response == "virginica":
            st.image("images/verginca.jpg", width=300)
        else:
            st.image("images/versicolor.jpg", width=300)


my_data = "iris.csv"
def explore_data(dataset):
    df = pd.read_csv(os.path.join(dataset), sep=";")
    return df

if selected == "Acceuil":
    st.title("Acceuil")
    texte_a_afficher = """<div style="text-align: justify;">
    Je suis <b>MABOM VALERE</b>, élève professeur à l'Ecole Nomarle Supérieure de Yaounde filière <i>'informatique'</i>.
    Je suis également en fin d'étude à l'Ecole Nationale Supérieure de Yaoundé dans la filiere <i>Intelligence Artificielle</i>.
    J'ai réalisé ce petit projet, pour aider l'utilisateur à prédire les type de données en fonction des differents parametres.
    Vous pouvez me donner des suggestions ou remarques dans la rubrique <b>Contact</b> du menu en dessous.
    Merci...
    </div>"""

    # Ou vous pouvez utiliser st.markdown pour plus de mise en forme
    st.markdown(texte_a_afficher, unsafe_allow_html=True)

if selected == "EDA":
    st.title("EDA")
    # Fonction de chargement du fichier
    @st.cache_data
    def load_data(file):
        if file is not None:
            try:
                if file.name.endswith('.csv'):
                    df = pd.read_csv(file)
                elif file.name.endswith('.xlsx'):
                    df = pd.read_excel(file)
                elif file.name.endswith('.json'):
                    df = pd.read_json(file)
                else:
                    st.error("Format non supporté. Utilisez CSV, Excel ou JSON.")
                    return None
                return df
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier : {e}")
                return None
        return None
    
    # Fonction de nettoyage des données
    def clean_data(df, missing_threshold=0.6):
        if df is None:
            return None
    
        df_cleaned = df.copy()
    
        # Suppression des colonnes avec trop de valeurs manquantes
        df_cleaned = df_cleaned.dropna(thresh=int(missing_threshold * len(df_cleaned)), axis=1)
    
        # Remplacement des valeurs manquantes par la moyenne pour les colonnes numériques
        df_cleaned.fillna(df_cleaned.mean(numeric_only=True), inplace=True)
    
        # Encodage des variables catégorielles
        label_encoders = {}
        for col in df_cleaned.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df_cleaned[col] = le.fit_transform(df_cleaned[col].astype(str))
            label_encoders[col] = le
    
        # Normalisation des valeurs numériques
        scaler = StandardScaler()
        numeric_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
        df_cleaned[numeric_cols] = scaler.fit_transform(df_cleaned[numeric_cols])
    
        return df_cleaned
    
    # Interface Streamlit
    st.title("Prétraitement du MathE Dataset")
    st.write("Téléchargez un fichier de données et effectuez son prétraitement.")
    
    # Upload du fichier
    uploaded_file = st.file_uploader("📂 Importer un fichier (CSV, Excel, JSON)", type=["csv", "xlsx", "json"])
    
    if uploaded_file is not None:
        # Charger et afficher les données
        df = load_data(uploaded_file)
        
        if df is not None:
            st.subheader("🔍 Aperçu des données originales")
            st.dataframe(df.head())
    
            # Affichage des informations générales
            st.write("Informations générales sur le dataset :")
            buffer = df.info(buf=None)
            st.text(buffer)
    
            # Affichage des valeurs manquantes
            st.write("Valeurs manquantes par colonne :")
            missing_values = df.isnull().sum()
            st.write(missing_values[missing_values > 0])
    
            # Nettoyage des données
            st.subheader("⚙ Prétraitement des données")
            cleaned_df = clean_data(df)
    
            if cleaned_df is not None:
                st.write("Données nettoyées et transformées")
                st.dataframe(cleaned_df.head())
    
                # Télécharger le fichier nettoyé
                csv = cleaned_df.to_csv(index=False).encode('utf-8')
                st.download_button(label="Télécharger les données prétraitées",
                                   data=csv,
                                   file_name="MathE_dataset_cleaned.csv",
                                   mime="text/csv")






if selected == "Predictions Iris":
    front_iris()
    
if selected == "Contact":
    st.title("Formulaire de Contact")

    # Créer un formulaire
    with st.form("contact_form"):
        nom = st.text_input("Nom")
        email = st.text_input("Email")
        objet = st.text_input("Objet")
        message = st.text_area("Message")

        # Bouton pour soumettre le formulaire
        submitted = st.form_submit_button("Envoyer", type="primary")

        if submitted:
            # Vérifier si tous les champs sont remplis
            if not nom or not email or not objet or not message:
                st.error("Tous les champs sont obligatoires!")
            else:
                 st.error("Cette fonctionnailté n'est pas encore achévée...")
                # try:
                #     envoyer_email(nom, email, objet, message)
                #     st.success(f"Merci, {nom}, votre message a été envoyé !")
                # except Exception as e:
                #     st.error(f"Une erreur est survenue lors de l'envoi de l'email : {e}")
