import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
import datetime

# Configuration de la page
st.set_page_config(page_title="GolfShot 10.0 Pro Dashboard", layout="wide")

# --- GESTION DES DONNÉES ---
if 'coups' not in st.session_state:
    st.session_state['coups'] = []

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

CLUBS_ORDER = [
    "Driver", "Bois 5", "Hybride", 
    "Fer 3", "Fer 5", "Fer 6", "Fer 7", "Fer 8", "Fer 9", 
    "PW", "50°", "55°", "60°", 
    "Putter"
]

DIST_REF = {
    "Driver": 220, "Bois 5": 200, "Hybride": 180, "Fer 3": 170,
    "Fer 5": 160, "Fer 6": 150, "Fer 7": 140, "Fer 8": 130, "Fer 9": 120,
    "PW": 110, "50°": 100, "55°": 90, "60°": 80, "Putter": 3
}

# --- SIDEBAR : GÉNÉRATEUR INTELLIGENT ---
st.sidebar.title("⚙️ Data Lab")

if st.sidebar.button("Générer Dataset 10.0 (Test)"):
    new_data = []
    dates = [datetime.date.today() - datetime.timedelta(days=x) for x in range(30)]
    
    for _ in range(300):
        club = np.random.choice(["Driver", "Fer 7", "PW", "55°"])
        mode = "Parcours"
        
        # Gestion des amplitudes
        if club in ["PW", "55°"]:
            ampli = np.random.choice(["Plein", "3/4", "1/2"], p=[0.5, 0.3, 0.2])
        else:
            ampli = "Plein"

        # Gestion des Lies
        if club == "Driver": lie = "Tee"
        else: lie = np.random.choice(["Fairway", "Rough", "Bunker", "Tee"], p=[0.6, 0.3, 0.05, 0.05])

        # Distance Cible
        dist_target = DIST_REF[club]
        if ampli == "3/4": dist_target *= 0.85
        if ampli == "1/2": dist_target *= 0.60
        
        # Réalité (Impact du Lie)
        penalty = 0.90 if lie == "Rough" else (0.70 if lie == "Bunker" else 1.0)
        dist_real = np.random.normal(dist_target * penalty, 8)

        # Erreurs directionnelles pour la Heatmap
        # Création de patterns (ex: souvent court à droite)
        delta_dist = dist_real - dist_target
        if delta_dist < -5: err_long = "Court"
        elif delta_dist > 5: err_long = "Long"
        else: err_long = "Bonne Longueur"
        
        direction = np.random.choice(["Gauche", "Centre", "Droite"], p=[0.3, 0.4, 0.3])

        new_data.append({
            'date': str(np.random.choice(dates)), 'mode': mode, 'club': club,
            'strat_dist': int(dist_target), 'amplitude': ampli, # Intention
            'distance': round(dist_real, 1), 'lie': lie,        # Réalité
            'direction': direction, 'delta_dist': delta_dist,
            'err_longueur': err_long,
            'type_coup': 'Jeu Long'
        })
    st.session_state['coups'].extend(new_data)
    st.sidebar.success("300 Coups générés !")

if st.session_state['coups']:
    df_ex = pd.DataFrame(st.session_state['coups'])
    st.sidebar.download_button("📥 Export CSV", convert_df(df_ex), "golf_v10.csv", "text/csv")
    
if st.sidebar.button("🗑️ Reset"): st.session_state['coups'] = []

# --- INTERFACE PRINCIPALE ---
st.title("🏌️‍♂️ GolfShot 10.0 : Pro Dashboard")

tab_saisie, tab_analyse = st.tabs(["🧠 Saisie & Contexte", "🔬 Analyse Granulaire"])

# --- ONGLET 1 : SAISIE ---
with tab_saisie:
    col_m, col_c = st.columns(2)
    with col_m: mode = st.radio("Mode", ["Parcours ⛳", "Practice 🚜"], horizontal=True)
    with col_c: club = st.selectbox("Club en main", CLUBS_ORDER)
    
    st.markdown("---")
    col_int, col_real = st.columns(2)

    # --- 1. INTENTION (Avec Amplitude) ---
    with col_int:
        st.subheader("1️⃣ INTENTION")
        if club == "Putter":
            obj_dist = st.number_input("Dist. Cible", 0.0, 30.0, 3.0)
            amplitude = "Plein"
        else:
            obj_dist = st.number_input("Dist. Cible", 0, 350, DIST_REF.get(club, 100))
            # NOUVEAUTÉ : AMPLITUDE
            amplitude = st.radio("Amplitude Swing", ["Plein", "3/4", "1/2"], horizontal=True)

    # --- 2. RÉALITÉ (Avec Lie) ---
    with col_real:
        st.subheader("2️⃣ RÉALITÉ")
        if club == "Putter":
            res_putt = st.radio("Résultat", ["Dans le trou", "Raté"])
            dist_real = obj_dist if res_putt == "Dans le trou" else st.number_input("Dist. Réelle", 0.0, 30.0, obj_dist)
            lie = "Green"
            direction = "Centre"
        else:
            dist_real = st.number_input("Distance Réelle (m)", 0, 350, int(obj_dist))
            
            # NOUVEAUTÉ : LIE (Situation)
            lie = st.selectbox("Situation (Lie)", ["Tee", "Fairway", "Rough", "Bunker", "Rough Épais"])
            
            direction = st.radio("Direction", ["Gauche", "Centre", "Droite"], horizontal=True)

    st.markdown("---")
    if st.button("💾 Enregistrer le Coup", type="primary", use_container_width=True):
        # Calcul auto de l'erreur longueur pour l'analyse
        delta = dist_real - obj_dist
        if delta < -5: err_L = "Court"
        elif delta > 5: err_L = "Long"
        else: err_L = "Bonne Longueur"

        st.session_state['coups'].append({
            'date': str(datetime.date.today()), 'mode': mode, 'club': club,
            'strat_dist': obj_dist, 'amplitude': amplitude, 
            'distance': dist_real, 'lie': lie,
            'direction': direction, 'delta_dist': delta, 'err_longueur': err_L,
            'type_coup': 'Putt' if club == "Putter" else 'Jeu Long'
        })
        st.success("Données sauvegardées !")

# --- ONGLET 2 : ANALYSE ---
with tab_analyse:
    if not st.session_state['coups']:
        st.info("Générez des données de test ou saisissez des coups.")
    else:
        df = pd.DataFrame(st.session_state['coups'])
        df_long = df[df['type_coup'] == 'Jeu Long']
        
        if not df_long.empty:
            # FILTRE GLOBAL D'AMPLITUDE
            st.markdown("### 🎚️ Filtre d'Analyse")
            f_ampli = st.selectbox("Quelle amplitude analyser ?", ["Plein", "3/4", "1/2", "Tout"], index=0)
            
            if f_ampli != "Tout":
                df_viz = df_long[df_long['amplitude'] == f_ampli]
            else:
                df_viz = df_long

            st.markdown("---")

            # --- ANALYSE 1 : DISTANCE MOYENNE PAR LIE ---
            st.header(f"📏 Distances par Lie ({f_ampli})")
            
            # Création du tableau croisé (Pivot Table)
            # Lignes = Clubs, Colonnes = Lie, Valeurs = Distance Moyenne
            pivot_lie = pd.pivot_table(
                df_viz, 
                values='distance', 
                index='club', 
                columns='lie', 
                aggfunc='mean'
            ).round(1)
            
            # Tri des clubs pour l'affichage
            pivot_lie = pivot_lie.reindex([c for c in CLUBS_ORDER if c in pivot_lie.index])
            
            # Affichage avec Heatmap colorée
            st.dataframe(pivot_lie.style.background_gradient(cmap="YlGnBu", axis=None).format("{:.1f}m"), use_container_width=True)
            st.caption("Ce tableau montre vos distances moyennes réelles. Comparez vos résultats sur 'Tee' vs 'Rough'.")

            st.markdown("---")

            # --- ANALYSE 2 : LA HEATMAP DES RATÉS (NOUVEAUTÉ) ---
            st.header("🔥 Heatmap des Ratés (Cone of Error)")
            col_heat, col_txt = st.columns([2, 1])
            
            with col_heat:
                # Préparation des données pour la Heatmap
                # On croise "Direction" (Gauche/Centre/Droite) avec "Erreur Longueur" (Court/Bon/Long)
                
                # Ordre logique pour le graphique
                y_order = ["Long", "Bonne Longueur", "Court"]
                x_order = ["Gauche", "Centre", "Droite"]
                
                # Comptage
                heatmap_data = df_viz.groupby(['err_longueur', 'direction']).size().unstack(fill_value=0)
                
                # On s'assure que toutes les colonnes/lignes existent même si 0
                heatmap_data = heatmap_data.reindex(index=y_order, columns=x_order, fill_value=0)
                
                fig, ax = plt.subplots()
                sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="Reds", cbar=False, linewidths=1, linecolor='black', ax=ax)
                ax.set_title(f"Distribution des Impacts ({f_ampli})")
                ax.set_ylabel("")
                ax.set_xlabel("")
                st.pyplot(fig)
            
            with col_txt:
                st.markdown("#### 💡 Analyse")
                # Analyse automatique
                total_shots = heatmap_data.sum().sum()
                miss_short_right = heatmap_data.loc['Court', 'Droite']
                miss_long_left = heatmap_data.loc['Long', 'Gauche']
                
                if miss_short_right > total_shots * 0.15:
                    st.warning("⚠️ **Tendance 'Push/Slice faible'** : Beaucoup de balles courtes à droite.")
                    st.write("C'est le raté classique de l'amateur (face ouverte, manque de puissance).")
                
                elif miss_long_left > total_shots * 0.15:
                    st.warning("⚠️ **Tendance 'Pull/Hook puissant'** : Balles longues à gauche.")
                    st.write("Signe d'un bon joueur qui referme trop les mains.")
                    
                st.info("Le but est d'avoir la case centrale ('Bonne Longueur' / 'Centre') la plus foncée possible.")
