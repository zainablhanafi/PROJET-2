
import base64
import pandas as pd 
import numpy as np 
import streamlit as st 
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from deep_translator import GoogleTranslator
import matplotlib.pyplot as plt


df_final= pd.read_csv("df_final.csv")
df = df_final.copy()                                            # Copie pour la partie streamlit (utilisateurs)

df_final['Genre'] = df_final['Genre'].fillna('')
df_final['Acteurs'] = df_final['Acteurs'].fillna('')
df_final['Resume'] = df_final['Resume'].fillna('')


# CREATION DES LISTES DU DF_FINAL (avant modifications)

list_acteurs = []                                               # Création d'une liste d'acteur
for list_noms in df_final['Acteurs'] :                          # Première boucle pour décomposer la liste d'acteurs        
    list_decompo = list_noms.split(', ')
    for nom_entier in list_decompo :                            # Deuxième boucle pour séparer le prénom(s) et Nom de chaque acteur
        if nom_entier != 'Unknown' :                            # Condition pour enlever les inconnus
            list_acteurs.append(nom_entier)

list_acteurs = list(set(list_acteurs))                          # Enlève les doublons et conserve le type (liste)
list_acteurs.sort()                                             # Liste par ordre alphabétique

list_genre = ['Action', 'Adventure', 'Documentary', 'Drama', 'Fantasy', 'Animation', 'Comedy', 'Family', 'History']


# TRAITEMENT DES DATA DU DF_FINAL + CONSTRUCTION DU MODEL

df_final['Acteurs'] = df_final['Acteurs'].str.replace(' ', '_', regex=False).str.replace(',', ' ')     # Suppression des espaces entre les prénoms et noms pour la Tokenisation
df_final['Acteurs']  = df_final['Acteurs'].str.lower()



df_final['Realisateur'] = df_final['Realisateur'].str.replace(' ', '_')                                # Suppression des espaces pour la Tokenisation comme pour les acteurs
df_final['Realisateur']  = df_final['Realisateur'].str.lower()

df_final['Genre']  = df_final['Genre'].str.lower()

df_final['Resume'] = df_final['Resume'].str.lower()
df_final['Resume'] = df_final['Resume'].str.replace('[^a-z ]','', regex=True)

col_num = df_final.select_dtypes(include=['number']).columns

preprocess = ColumnTransformer(transformers=[
        ('resume', TfidfVectorizer(stop_words = 'english',ngram_range=(1,2),min_df=2,max_df=0.8), 'Resume'),
        ('genre', make_pipeline(TfidfVectorizer(), FunctionTransformer(lambda x: x * 3)), 'Genre'),
        ('realisateur', make_pipeline(TfidfVectorizer(), FunctionTransformer(lambda x: x * 2)), 'Realisateur'),
        ('acteurs', make_pipeline(TfidfVectorizer(), FunctionTransformer(lambda x: x * 2)), 'Acteurs'),
])

my_pipeline = Pipeline(steps=[
    ('preprocess', preprocess),                                          # Étape 1 : encodage et normalisation de l'ensemble des colonnes
    ('model', NearestNeighbors(n_neighbors=7, metric='cosine'))          # Étape 2 : On donne les données propres au modèle
])

my_pipeline.fit(df_final)

def reco_preferences(acteur=None, genre=None, realisateur=None):

    # transformation des noms de l'acteur et du réalisateur
    if acteur is not None :
        acteur = acteur.replace(" ", "_").lower()
        acteur = (acteur + " ") * 5
    
    if genre is not None: 
        genre = (genre + " ") * 5

    if realisateur is not None :
        realisateur = realisateur.replace(" ", "_").lower()
        realisateur = (realisateur + " ") * 5

    # création du film virtuel
    film = pd.DataFrame({
        'Genre': [genre],
        'Realisateur': [realisateur],
        'Acteurs': [acteur],
        'Resume': [""]})

    # transformation en vecteur
    vecteur = my_pipeline.named_steps['preprocess'].transform(film)

    # recherche des voisins
    distances, indices = my_pipeline.named_steps['model'].kneighbors(vecteur)

    return df.iloc[indices[0]][["Titre", "Annee", "Genre","Acteurs","Realisateur","Affiche","Image", "Note", "Resume",  "Duree"]].sort_values(by='Note', ascending=False)

def reco_titre(titre: str):

    index = df_final[df_final['Titre'].str.lower() == titre.lower()].index[0]                        # Trouve l’index du film

    vecteur = my_pipeline.named_steps['preprocess'].transform(df_final.iloc[[index]])                   # Transform l'index en vecteur pour le modèle
    
    distances, indices = my_pipeline.named_steps['model'].kneighbors(vecteur)                           # Récupère les distances et indices des plus proches voisins

    return df.iloc[indices[0]][["Titre", "Annee", "Genre", "Realisateur", "Acteurs","Affiche","Image", "Note", "Resume", "Duree"]].sort_values(by='Note', ascending=False)      # Retourne les film les plus proches 




def set_background(image_path):                                             
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    st.markdown(f"""
        <style>
            .stApp {{
                background-image: url("data:image/gif;base64,{data}");
                background-size: cover;
                background-attachment: fixed;
            }}
            h1, h2, h3 {{ color: white !important; }}
            p, div, span {{ color: #e0e0e0 !important; }}
        </style>
    """, unsafe_allow_html=True)


set_background("image/download (2).jpg")



with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Films", "Préférences", "Top 5", "KPI"],
        icons=["film", "heart", "fire", "bar-chart"],
        default_index=0,
    )



if selected == "Films":
        st.title("Recommandations par film ˙✧") 
    
        
        titre = st.selectbox(
        "Choisissez un titre :",
        [""] + sorted(df["Titre"].dropna().unique()))


        if titre != "":
            reco = reco_titre(titre)
            st.subheader("Films recommandés")
            
            for ind, row in reco.iterrows():
                
                with st.container():                                                            # Aligne les colonnes entre elles
                    col_image, col_description, col_resume = st.columns([1,2,3])

                    with col_image:
                        BASE_URL = "https://image.tmdb.org/t/p/w500"
                        st.image(BASE_URL + row["Affiche"], use_container_width=True)

                    with col_description:
                        st.write(f"Titre : {row['Titre']}")
                        st.write(f"Année : {row['Annee']}")
                        st.write(row['Genre'])
                        st.write(f"Realisateur : {row['Realisateur']}")
                        st.write()
                    
                    if row['Note'] >= 7.5 :
                        star = '⭐⭐⭐'
                    elif row['Note'] >= 5 :
                        star = '⭐⭐'
                    else : 
                        star = '⭐'

                    with col_resume :
                        st.markdown(f" {star} {row['Note']} /10  |  ⏱ {row['Duree']} min")
                        with st.expander("Voir le résumé"):
                            translator = GoogleTranslator(source='en', target='fr')                     
                            st.write(translator.translate(row['Resume']))                      # Traduction ddu résumé en Français



elif selected == "Préférences":
        st.title("Recommandations par préférences ˙✧")
        

        realisateur = st.selectbox(
        "Choisissez un réalisateur :",
        [""] + sorted(df["Realisateur"].dropna().unique()))

        genre = st.selectbox(
        "Choisissez un genre :",
        [""] + sorted(list_genre))
        
        acteur = st.selectbox(
        "Choisissez un acteur :",
        [""] + sorted(list_acteurs))


        if realisateur != ""  or genre != "" or acteur != "" :
            reco_pref = reco_preferences(acteur, genre, realisateur)

            st.subheader("Films recommandés")
                    
            for ind, row in reco_pref.iterrows():

                with st.container():                                                            # Aligne les colonnes entre elles
                    col_image, col_description, col_resume = st.columns([1,2,3])

                    with col_image:
                        BASE_URL = "https://image.tmdb.org/t/p/w500"
                        if row["Affiche"] != "" and row["Affiche"] != "Unknown":
                            st.image(BASE_URL + row["Affiche"], use_container_width=True)
                        else:
                            st.image("https://via.placeholder.com/200x300?text=No+Image", use_container_width=True)         # +Image -> variable ?

                    with col_description:
                        st.write(f"Titre : {row['Titre']}")
                        st.write(f"Année : {row['Annee']}")
                        st.write(f"Genre : {row['Genre']}")
                        st.write(f"Realisateur : {row['Realisateur']}")
                        st.write()

                    if row['Note'] >= 7.5 :
                        star = '⭐⭐⭐'
                    elif row['Note'] >= 5 :
                        star = '⭐⭐'
                    else : 
                        star = '⭐'

                    with col_resume :
                        st.markdown(f" {star} {row['Note']} /10 | ⏱ {row['Duree']} min")
                        with st.expander("Voir le résumé"):
                            translator = GoogleTranslator(source='en', target='fr')                     
                            st.write(translator.translate(row['Resume']))                      # Traduction ddu résumé en Français



elif selected == "Top 5":
    st.title("Top 5 Films")
    def afficher_top_netflix(dataframe, titre_section):
        st.subheader(titre_section)
        BASE_URL = "https://image.tmdb.org/t/p/w500"
        cols_per_row = 5
        rows = [dataframe.iloc[i:i+cols_per_row] for i in range(0, len(dataframe), cols_per_row)]
    
        for row in rows:
            cols = st.columns(cols_per_row)
        for col, (_, film) in zip(cols, row.iterrows()):
            with col:
                affiche = film.get("Affiche", "")
                if pd.notna(affiche) and affiche != "" and affiche != "Unknown":
                    st.image(BASE_URL + affiche, use_container_width=True)  
                else:
                    st.image("https://via.placeholder.com/200x300?text=No+Image", use_container_width=True)  
                
                titre_film = film.get("Titre Original") or film.get("Titre", "")
                st.markdown(f"**{titre_film}**")
                st.caption(f" ᯓ★ {film['Note']} | {int(film['Nb_votes'])} vote | ⏱ {int(film['Duree'])} min")
    st.markdown("---")


    top_comedies = (df[df["Genre"].str.contains("Comedy", na=False) & (df["Nb_votes"] > 1000)].sort_values("Note", ascending=False).head(10))
    afficher_top_netflix(top_comedies, "Top 5 Comédies")

    top_animation = (df[df["Genre"].str.contains("Animation", na=False) & (df["Nb_votes"] > 1000)].sort_values("Note", ascending=False).head(10))
    afficher_top_netflix(top_animation, "Top 5 Animations")

    top_Drama = (df[df["Genre"].str.contains("Drama", na=False) & (df["Nb_votes"] > 1000)].sort_values("Note", ascending=False).head(10))
    afficher_top_netflix(top_Drama, "Top 5 Drama")

    top_Adventure = (df[df["Genre"].str.contains("Adventure", na=False) & (df["Nb_votes"] > 1000)].sort_values("Note", ascending=False).head(10))
    afficher_top_netflix(top_Adventure, "Top 5 Aventure")

    top_Action = (df[df["Genre"].str.contains("Action", na=False) & (df["Nb_votes"] > 1000)].sort_values("Note", ascending=False).head(10))
    afficher_top_netflix(top_Action, "Top 5 Action")

    top_famille = (df[df["Genre"].str.contains("Family", na=False) & (df["Nb_votes"] > 1000)].sort_values("Note", ascending=False).head(10))
    afficher_top_netflix(top_famille, "Top 5 Famille")

    top_doc_histoire = (df[(df["Genre"].str.contains("Documentary,History", na=False)) & (df["Nb_votes"] > 1000)].sort_values("Note", ascending=False).head(10))
    afficher_top_netflix(top_doc_histoire, "Top 5 Documentaires Historiques")

    top_doc_music = (df[(df["Genre"].str.contains("Documentary,Music", na=False)) & (df["Nb_votes"] > 1000)].sort_values("Note", ascending=False).head(10))
    afficher_top_netflix(top_doc_music, "Top 5 Documentaires Musicaux")

    top_doc_sport = (df[(df["Genre"].str.contains("Documentary,Sport", na=False)) & (df["Nb_votes"] > 1000)].sort_values("Note", ascending=False).head(10))
    afficher_top_netflix(top_doc_sport, "Top 5 Documentaires Sportifs")

   
elif selected == "KPI":
        st.title("Indicateurs Clés de Performance")
    
        options = {
        "Top 10 films qui ont généré le plus de recette": "image/canva1.png",
        "Top 10 films des plus gros budgets": "image/canva2.png",
        "Films les mieux notés ": "image/canva3.png",
        "Répartition des films par années de sortie": "image/canva4.png",
        "Films les plus populaires": "image/canva5.png",
    }
    
        choix = st.selectbox("Sélectionnez un KPI :", list(options.keys()))
        st.image(options[choix], use_column_width=True)
