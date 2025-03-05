import streamlit as st
from streamlit_option_menu import option_menu
import requests
import joblib
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pycountry


col1, col2 = st.columns([1, 3])  # Colonne 1 pour la navbar (1/4), Colonne 2 pour le contenu (3/4)

with st.sidebar:
    selected = option_menu(
        menu_title = "Main menu",
        options = ["Acceuil", "EDA", "Predictions", "Contact"],
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
    st.markdown("<h1 style='text-align: center;'>PREDICTION DU TYPE DE REPONSE</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])  # Colonne 1 pour la navbar (1/4), Colonne 2 pour le contenu (3/4)
    # Navbar verticale dans la colonne de gauche


    col1, col2 = st.columns(2)

    with col1:
        countries = [country.name for country in pycountry.countries]
        # Liste déroulante pour sélectionner un pays
        selected_country = st.selectbox("Choisissez un pays :", countries)

    with col2:
        question_level = st.radio("Niveau de question :", ("Basic", "Level"))

    # Deuxième ligne avec deux autres curseurs
    col3, col4 = st.columns(2)

    with col3:
        topic = st.selectbox("Choisissez votre topic :", ("Linear Algebra", "Fundamental Mathematics", "Differentiation", "Differential Equations"))

    with col4:
        sub_topic = st.selectbox("Choisissez votre sub_topic :", ("Probability", "Graph Theory", "Complex Numbers", "Elementary Geometry"))
        
    commentaire = st.text_area("Écrivez votre question ici :", "")

    # Bouton pour envoyer les données à l'API
    if st.button("Prédire le type de question...", type="primary"):
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


my_data = "data.csv"
def explore_data(dataset):
    df = pd.read_csv(os.path.join(dataset), sep=";", encoding='ISO-8859-1')
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
    data = explore_data(my_data)
    if st.checkbox("Preview Data"):
        
        if st.button("Head"):
            st.write(data.head())

        if st.button("Sample"):
            st.write(data.sample(5))

        if st.button("Show All Dataset"):
            st.write(data)

    if st.checkbox("Show Columns Names"):
        st.write(data.columns)

    text_dim = st.radio("What Dimensions Do You Want To See?", ("Rows", "Columns", "All"))
    if text_dim == "Rows":
        st.text("Nombre de ligne:")
        st.write(data.shape[0]) 
    elif text_dim == "Columns":
        st.text("Nombre de colonne:")
        st.write(data.shape[1])   
    else:
        st.text("Dimensions:")
        st.write(data.shape) 

    if st.checkbox("Show Summary of Dataset"):
        st.write(data.describe())

    if st.checkbox("percentage of distributions"):
        effectif = data["Type of Answer"].value_counts()
        fig, ax = plt.subplots()
        st.write(plt.pie(effectif, labels=effectif.index, autopct='%1.2f%%'))
        plt.title("Répartition des modalités")
        plt.axis('equal')
        st.pyplot(fig)
    
    if st.checkbox("Visualization"):
        plt.figure(figsize=(15, 9))
        st.write(sns.countplot(x='Question Level', hue='Type of Answer', data=data))
        st.write(sns.countplot(x='Student Country', hue='Type of Answer', data=data))
       #st.pyplot(fig)
        st.pyplot(plt)






if selected == "Predictions":
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
