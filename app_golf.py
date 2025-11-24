import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Configuration de la page
st.set_page_config(page_title="GolfShot Pro 3.1", layout="wide")

# --- GESTION DES DONNÉES ---
if 'coups' not in st.session_state:
    st.session_state['coups'] = []

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# --- SIDEBAR : SAUVEGARDE & OPTIONS ---
st.sidebar.title("💾 Menu")

# Sauvegarde / Chargement
uploaded_file = st.sidebar.file_uploader("Charger historique (CSV)", type="csv")
if uploaded_file is not None:
    try:
        df_loaded = pd.read_csv(uploaded_file)
        st.session_state['coups'] = df_loaded.to_dict('records')
        st.sidebar.success("Données chargées !")
    except:
        st.sidebar.error("Erreur de fichier")

if st.session_state['coups']:
    df_export = pd.DataFrame(st.session_state['coups'])
    csv = convert_df(df_export)
    st.sidebar.download_button(
        label="📥 Télécharger mes données (Sauvegarde)",
        data=csv,
        file_name='mon_golf_v3_1.csv',
        mime='text/csv',
    )

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Tout effacer"):
    st.session_state['coups'] = []

# --- INTERFACE PRINCIPALE ---
st.title("🏌️‍♂️ GolfShot Pro 3.1")

tab1, tab2, tab3 = st.tabs(["⛳ Saisie Parcours", "📊 Statistiques", "🧠 Coaching Putting"])

# --- ONGLET 1 : SAISIE PARCOURS ---
with tab1:
    st.markdown("### Nouveau Coup")
    
    # 1. CONTEXTE DU TROU
    col_ctx1, col_ctx2 = st.columns(2)
    with col_ctx1:
        par_trou = st.selectbox("Par du trou", [3, 4, 5], index=1)
    with col_ctx2:
        # --- LISTE DES CLUBS MISE À JOUR ---
        liste_clubs = [
            "Driver", "Bois 5", "Hybride", 
            "Fer 4", "Fer 5", "Fer 6", "Fer 7", "Fer 8", "Fer 9", 
            "PW", "50°", "55°", "60°", 
            "Putter"
        ]
        club = st.selectbox("Club joué", liste_clubs)

    st.markdown("---")

    # 2. LOGIQUE DYNAMIQUE (PUTTER vs AUTRES)
    if club == "Putter":
        st.info("🟢 Mode Putting Activé")
        col_p1, col_p2, col_p3 = st.columns(3)
        
        with col_p1:
            # Distance avec virgules (float)
            dist_putt = st.number_input("Distance Putt (m)", min_value=0.0, value=1.5, step=0.1, format="%.1f")
        
        with col_p2:
            pente_lat = st.selectbox("Pente Latérale", ["Plat", "Gauche-Droite", "Droite-Gauche"])
            denivele = st.selectbox("Dénivelé", ["Plat", "Montée", "Descente"])
            
        with col_p3:
            resultat_putt = st.radio("Résultat", ["Dans le trou", "Raté - Court", "Raté - Long", "Raté - Gauche", "Raté - Droite"])
            
        if st.button("Enregistrer le Putt", type="primary"):
            coup = {
                'date': str(datetime.date.today()),
                'par_trou': par_trou,
                'club': club,
                'distance': dist_putt,
                'lie': 'Green',
                'pente': pente_lat,
                'denivele': denivele,
                'resultat': resultat_putt,
                'type_coup': 'Putt'
            }
            st.session_state['coups'].append(coup)
            st.success(f"Putt de {dist_putt}m enregistré !")

    else:
        # Interface pour les coups normaux (Driver, Fers, Wedges, Bois)
        col1, col2, col3 = st.columns(3)
        with col1:
            distance = st.number_input("Distance (m)", min_value=0, value=100)
            lie = st.selectbox("Lie", ["Tee", "Fairway", "Rough", "Bunker", "Rough Epais"])
        
        with col2:
            direction = st.radio("Direction", ["Gauche", "Centre", "Droite"], horizontal=True)
            longueur = st.radio("Profondeur", ["Court", "Ok", "Long"], horizontal=True)
            
        with col3:
            st.write("Validation")
            if st.button("Enregistrer le Coup", type="primary"):
                coup = {
                    'date': str(datetime.date.today()),
                    'par_trou': par_trou,
                    'club': club,
                    'distance': distance,
                    'lie': lie,
                    'direction': direction,
                    'longueur': longueur,
                    'resultat': f"{direction}/{longueur}",
                    'type_coup': 'Jeu Long'
                }
                st.session_state['coups'].append(coup)
                st.success(f"Coup de {club} enregistré !")

    # Affichage de l'historique récent
    if st.session_state['coups']:
        st.markdown("### 📝 5 derniers coups saisis")
        df_show = pd.DataFrame(st.session_state['coups'])
        cols_to_show = ['club', 'distance', 'resultat', 'pente', 'denivele']
        for c in cols_to_show:
            if c not in df_show.columns:
                df_show[c] = ""
        st.dataframe(df_show[cols_to_show].tail(5))

# --- ONGLET 2 : STATISTIQUES GÉNÉRALES ---
with tab2:
    if st.session_state['coups']:
        df = pd.DataFrame(st.session_state['coups'])
        df_long = df[df['type_coup'] == 'Jeu Long']
        
        if not df_long.empty:
            st.header("Dispersion (Jeu Long)")
            
            # On s'assure que les clubs s'affichent dans le bon ordre dans le filtre
            clubs_presents = df_long['club'].unique()
            # Petit tri manuel pour l'esthétique si possible, sinon ordre d'apparition
            club_filter = st.selectbox("Voir Club", clubs_presents)
            
            subset = df_long[df_long['club'] == club_filter]
            
            if not subset.empty:
                fig, ax = plt.subplots(figsize=(6, 4))
                # Simulation coordonnées
                def get_coords(row):
                    x, y = 0, 0
                    noise = np.random.normal(0, 0.1)
                    if row['direction'] == 'Gauche': x = -1
                    if row['direction'] == 'Droite': x = 1
                    if row['longueur'] == 'Court': y = -1
                    if row['longueur'] == 'Long': y = 1
                    return x+noise, y+noise
                
                coords = subset.apply(get_coords, axis=1, result_type='expand')
                ax.scatter(coords[0], coords[1], c='blue', alpha=0.6)
                ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
                ax.axhline(0, c='gray', ls='--'); ax.axvline(0, c='gray', ls='--')
                ax.set_title(f"Dispersion : {club_filter}")
                st.pyplot(fig)
        else:
            st.info("Enregistrez des coups de fer/bois pour voir la dispersion.")

# --- ONGLET 3 : COACHING PUTTING ---
with tab3:
    st.header("🧠 Analyse du Putting")
    
    if st.session_state['coups']:
        df = pd.DataFrame(st.session_state['coups'])
        df_putt = df[df['type_coup'] == 'Putt']
        
        if not df_putt.empty:
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("Réussite par Pente")
                if 'pente' in df_putt.columns:
                    res_pente = df_putt.groupby('pente')['resultat'].apply(lambda x: (x == 'Dans le trou').mean() * 100)
                    st.bar_chart(res_pente)
            
            with col_b:
                st.subheader("Réussite par Dénivelé")
                if 'denivele' in df_putt.columns:
                    res_deniv = df_putt.groupby('denivele')['resultat'].apply(lambda x: (x == 'Dans le trou').mean() * 100)
                    st.bar_chart(res_deniv)
            
            st.markdown("---")
            st.write("### Détail des putts ratés")
            missed = df_putt[df_putt['resultat'] != 'Dans le trou']
            if not missed.empty:
                st.write(missed[['distance', 'pente', 'denivele', 'resultat']])
            else:
                st.success("Aucun putt raté pour l'instant !")
        else:
            st.info("Aucune donnée de putting enregistrée.")
    else:
        st.info("Commencez par saisir des données.")
