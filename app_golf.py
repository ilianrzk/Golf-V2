import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Configuration de la page
st.set_page_config(page_title="GolfShot Pro 2.0", layout="wide")

# --- GESTION DES DONNÉES (Sauvegarde/Chargement) ---
if 'coups' not in st.session_state:
    st.session_state['coups'] = []

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# --- SIDEBAR : SAUVEGARDE & PROFIL ---
st.sidebar.title("💾 Ma Progression")

# Import/Export des données (CRUCIAL pour ne rien perdre)
st.sidebar.markdown("### 1. Sauvegarder/Charger")
uploaded_file = st.sidebar.file_uploader("Charger une ancienne partie (CSV)", type="csv")
if uploaded_file is not None:
    try:
        df_loaded = pd.read_csv(uploaded_file)
        st.session_state['coups'] = df_loaded.to_dict('records')
        st.sidebar.success("Historique chargé !")
    except:
        st.sidebar.error("Erreur de fichier")

if st.session_state['coups']:
    df_export = pd.DataFrame(st.session_state['coups'])
    csv = convert_df(df_export)
    st.sidebar.download_button(
        label="📥 Télécharger mes données (Sauvegarde)",
        data=csv,
        file_name='mon_historique_golf.csv',
        mime='text/csv',
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Paramètres")
# Profil Simplifié
target_index = st.sidebar.number_input("Mon Index Cible", value=18)

# Bouton Démo
if st.sidebar.button("Générer Données Démo (Test)"):
    # Génération de données fictives avec dates
    clubs = ['Driver', 'Fer 7', 'Wedge']
    directions = ['Centre', 'Gauche', 'Droite']
    dates = [datetime.date.today() - datetime.timedelta(days=x) for x in range(10)]
    
    data = []
    for _ in range(50):
        c = np.random.choice(clubs)
        d = np.random.choice(dates)
        dist_base = {'Driver': 210, 'Fer 7': 135, 'Wedge': 75}[c]
        var = np.random.randint(-15, 25)
        
        coup = {
            'date': str(d),
            'club': c,
            'distance': dist_base + var,
            'direction': np.random.choice(directions, p=[0.5, 0.25, 0.25]),
            'longueur': np.random.choice(['Ok', 'Court', 'Long']),
            'lie': 'Fairway'
        }
        data.append(coup)
    st.session_state['coups'] = data
    st.sidebar.success("Données de démo chargées !")

if st.sidebar.button("🗑️ Tout effacer"):
    st.session_state['coups'] = []

# --- INTERFACE PRINCIPALE ---
st.title("🏌️‍♂️ GolfShot Pro 2.0")

tab1, tab2, tab3, tab4 = st.tabs(["📝 Saisie", "📏 Étalonnage (Distances)", "📊 Dispersion", "📈 Évolution"])

# --- ONGLET 1 : SAISIE ---
with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        date_coup = st.date_input("Date", datetime.date.today())
        club = st.selectbox("Club", ["Driver", "Bois 3", "Hybride", "Fer 5", "Fer 6", "Fer 7", "Fer 8", "Fer 9", "PW", "GW", "SW", "LW", "Putter"])
    with col2:
        distance = st.number_input("Distance Totale (m)", min_value=0, value=130)
        lie = st.selectbox("Lie", ["Tee", "Fairway", "Rough", "Bunker"])
    with col3:
        direction = st.radio("Direction", ["Gauche", "Centre", "Droite"], horizontal=True)
        longueur = st.radio("Profondeur", ["Court", "Ok", "Long"], horizontal=True)
        
    if st.button("Enregistrer le coup", type="primary"):
        nouveau_coup = {
            'date': str(date_coup),
            'club': club,
            'distance': distance,
            'direction': direction,
            'longueur': longueur,
            'lie': lie
        }
        st.session_state['coups'].append(nouveau_coup)
        st.success(f"Coup enregistré : {club} - {distance}m")

# --- ONGLET 2 : ÉTALONNAGE (NOUVEAU) ---
with tab2:
    st.header("📏 La vérité sur vos distances")
    st.markdown("Ce graphique montre vos distances réelles. La ligne orange est votre médiane (votre coup 'standard'). Les points sont les coups isolés.")
    
    if st.session_state['coups']:
        df = pd.DataFrame(st.session_state['coups'])
        
        # On trie l'ordre des clubs pour que l'affichage soit logique
        ordre_clubs = ["Driver", "Bois 3", "Hybride", "Fer 5", "Fer 6", "Fer 7", "Fer 8", "Fer 9", "PW", "GW", "SW", "LW", "Putter"]
        df['club'] = pd.Categorical(df['club'], categories=ordre_clubs, ordered=True)
        df = df.sort_values('club')

        # Graphique Boxplot avec Matplotlib
        fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
        
        clubs_presents = df['club'].unique()
        data_to_plot = [df[df['club'] == c]['distance'] for c in clubs_presents]
        
        ax_dist.boxplot(data_to_plot, labels=clubs_presents, patch_artist=True, boxprops=dict(facecolor="lightblue"))
        ax_dist.set_title("Distribution de vos distances par club")
        ax_dist.set_ylabel("Distance (mètres)")
        ax_dist.grid(True, linestyle='--', alpha=0.6)
        
        st.pyplot(fig_dist)
        
        # Tableau récapitulatif
        st.write("### Tableau des Moyennes")
        stats = df.groupby('club', observed=True)['distance'].agg(['count', 'mean', 'max']).round(1)
        stats.columns = ['Nb Coups', 'Moyenne', 'Maximum']
        st.dataframe(stats)
    else:
        st.info("Entrez des données pour voir vos distances.")

# --- ONGLET 3 : DISPERSION (CLASSIQUE) ---
with tab3:
    if st.session_state['coups']:
        df = pd.DataFrame(st.session_state['coups'])
        st.header("🎯 Analyse de précision")
        club_filter = st.selectbox("Filtrer par club", df['club'].unique())
        subset = df[df['club'] == club_filter]
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Coordonnées simulées pour affichage
        def get_coords(row):
            x, y = 0, 0
            # Ajout de bruit aléatoire pour éviter la superposition
            noise_x = np.random.normal(0, 0.08)
            noise_y = np.random.normal(0, 0.08)
            
            if row['direction'] == 'Gauche': x = -1
            elif row['direction'] == 'Droite': x = 1
            
            if row['longueur'] == 'Court': y = -1
            elif row['longueur'] == 'Long': y = 1
            
            return x + noise_x, y + noise_y

        coords = subset.apply(get_coords, axis=1, result_type='expand')
        
        # Cible
        ax.add_patch(plt.Circle((0, 0), 0.3, color='green', alpha=0.3))
        ax.scatter(coords[0], coords[1], alpha=0.7, c='blue', s=100)
        
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.axhline(0, color='grey', linestyle='--')
        ax.axvline(0, color='grey', linestyle='--')
        ax.set_xticks([-1, 0, 1], ['Gauche', 'Centre', 'Droite'])
        ax.set_yticks([-1, 0, 1], ['Court', 'Ok', 'Long'])
        ax.set_title(f"Dispersion : {club_filter}")
        
        st.pyplot(fig)
        
        # Coaching IA simple
        miss_left = len(subset[subset['direction']=='Gauche'])
        miss_right = len(subset[subset['direction']=='Droite'])
        total = len(subset)
        if total > 3:
            if miss_right > miss_left:
                st.warning(f"⚠️ Tendance au Slice avec le {club_filter}. {int(miss_right/total*100)}% des balles à droite.")
            elif miss_left > miss_right:
                st.warning(f"⚠️ Tendance au Hook avec le {club_filter}. {int(miss_left/total*100)}% des balles à gauche.")

# --- ONGLET 4 : ÉVOLUTION (