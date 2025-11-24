import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
import datetime

# Configuration de la page
st.set_page_config(page_title="GolfShot 11.0 Fusion Ultimate", layout="wide")

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

# --- SIDEBAR : GÉNÉRATEUR FUSIONNÉ (V9 + V10) ---
st.sidebar.title("⚙️ Data Lab")

if st.sidebar.button("Générer Dataset V11 (Fusion)"):
    new_data = []
    dates = [datetime.date.today() - datetime.timedelta(days=x) for x in range(30)]
    
    for _ in range(300):
        # Paramètres de base
        club = np.random.choice(["Driver", "Fer 7", "PW", "55°"])
        mode = np.random.choice(["Practice", "Parcours"], p=[0.3, 0.7])
        
        # --- LOGIQUE V10 (Amplitude & Lie) ---
        if club in ["PW", "55°"]:
            ampli = np.random.choice(["Plein", "3/4", "1/2"], p=[0.5, 0.3, 0.2])
        else:
            ampli = "Plein"

        if mode == "Practice":
            lie = "Tapis/Tee"
        else:
            if club == "Driver": lie = "Tee"
            else: lie = np.random.choice(["Fairway", "Rough", "Bunker"], p=[0.6, 0.3, 0.1])

        # --- LOGIQUE V9 (Effets & Stratégie) ---
        strat_effet = np.random.choice(["Tout droit", "Fade", "Draw"], p=[0.7, 0.15, 0.15])
        
        # Calcul Distance Cible (Intention)
        dist_target = DIST_REF[club]
        if ampli == "3/4": dist_target *= 0.85
        if ampli == "1/2": dist_target *= 0.60
        
        # Simulation Réalité
        # Pénalité Lie (V10)
        penalty_lie = 0.90 if lie == "Rough" else (0.70 if lie == "Bunker" else 1.0)
        # Variance Practice vs Parcours (V9)
        std_dev = 5 if mode == "Practice" else 12
        
        dist_real = np.random.normal(dist_target * penalty_lie, std_dev)
        
        # Réussite Effet (V9)
        if np.random.random() < 0.6: real_effet = strat_effet
        else: real_effet = "Raté"

        # Erreurs Directionnelles & Heatmap Logic
        delta = dist_real - dist_target
        if delta < -5: err_L = "Court"
        elif delta > 5: err_L = "Long"
        else: err_L = "Bonne Longueur"
        
        # Score Latéral (V9 0-5)
        lat_score = abs(np.random.normal(0, 1 if mode=="Practice" else 2))
        lat_score = min(5, int(lat_score))
        direction = "Centre" if lat_score == 0 else np.random.choice(["Gauche", "Droite"])

        new_data.append({
            'date': str(np.random.choice(dates)), 'mode': mode, 'club': club,
            # Intention (V9+V10)
            'strat_dist': int(dist_target), 'strat_effet': strat_effet, 'amplitude': ampli,
            # Réalité (V9+V10)
            'distance': round(dist_real, 1), 'lie': lie, 'real_effet': real_effet,
            'direction': direction, 'score_lateral': lat_score,
            'delta_dist': delta, 'err_longueur': err_L, 'contact': np.random.choice(["Bon", "Gratte"]),
            'type_coup': 'Jeu Long'
        })
    st.session_state['coups'].extend(new_data)
    st.sidebar.success("300 Coups complets (V9+V10) générés !")

if st.session_state['coups']:
    df_ex = pd.DataFrame(st.session_state['coups'])
    st.sidebar.download_button("📥 Export CSV", convert_df(df_ex), "golf_v11.csv", "text/csv")
    
if st.sidebar.button("🗑️ Reset"): st.session_state['coups'] = []

# --- INTERFACE PRINCIPALE ---
st.title("🏌️‍♂️ GolfShot 11.0 : Fusion Ultimate")

tab_saisie, tab_tech, tab_strat, tab_fail = st.tabs([
    "🧠 Saisie Complète", 
    "📏 Calibration (V10)", 
    "🎯 Dispersion (V9)", 
    "🔥 Analyse Ratés (Mix)"
])

# --- ONGLET 1 : SAISIE FUSIONNÉE ---
with tab_saisie:
    col_m, col_c = st.columns(2)
    with col_m: mode = st.radio("Mode", ["Parcours ⛳", "Practice 🚜"], horizontal=True)
    with col_c: club = st.selectbox("Club en main", CLUBS_ORDER)
    
    st.markdown("---")
    col_int, col_real = st.columns(2)

    # --- 1. INTENTION (V9 Strat + V10 Amplitude) ---
    with col_int:
        st.subheader("1️⃣ INTENTION")
        if club == "Putter":
            obj_dist = st.number_input("Dist. Cible", 0.0, 30.0, 3.0)
            strat_effet = "Aucun"
            amplitude = "Plein"
        else:
            obj_dist = st.number_input("Dist. Cible", 0, 350, DIST_REF.get(club, 100))
            # V10 : Amplitude
            amplitude = st.radio("Amplitude Swing", ["Plein", "3/4", "1/2"], horizontal=True)
            # V9 : Effet Annoncé
            strat_effet = st.selectbox("Effet Annoncé", ["Tout droit", "Fade", "Draw", "Balle basse"])

    # --- 2. RÉALITÉ (V9 Latéral + V10 Lie) ---
    with col_real:
        st.subheader("2️⃣ RÉALITÉ")
        if club == "Putter":
            res_putt = st.radio("Résultat", ["Dans le trou", "Raté"])
            dist_real = obj_dist if res_putt == "Dans le trou" else st.number_input("Dist. Réelle", 0.0, 30.0, obj_dist)
            lie = "Green"
            direction = "Centre"
            score_lat = 0
            real_effet = "Aucun"
            contact = "Bon"
        else:
            dist_real = st.number_input("Distance Réelle (m)", 0, 350, int(obj_dist))
            
            # V10 : Lie
            lie = st.selectbox("Situation (Lie)", ["Tee", "Fairway", "Rough", "Bunker"])
            # V9 : Effet Réalisé & Contact
            c_eff, c_cont = st.columns(2)
            with c_eff: real_effet = st.selectbox("Effet Réalisé", ["Tout droit", "Fade", "Draw", "Raté"])
            with c_cont: contact = st.selectbox("Contact", ["Bon", "Gratte", "Top"])
            
            # V9 : Dispersion Latérale Fine
            c_dir, c_sco = st.columns(2)
            with c_dir: direction = st.radio("Direction", ["Gauche", "Centre", "Droite"], horizontal=True)
            with c_sco: score_lat = st.number_input("Écart (0-5)", 0, 5, 0) if direction != "Centre" else 0

    st.markdown("---")
    if st.button("💾 Enregistrer (V11)", type="primary", use_container_width=True):
        # Calcul auto delta (V9/V10)
        delta = dist_real - obj_dist
        if delta < -5: err_L = "Court"
        elif delta > 5: err_L = "Long"
        else: err_L = "Bonne Longueur"

        st.session_state['coups'].append({
            'date': str(datetime.date.today()), 'mode': mode.split()[0], 'club': club,
            'strat_dist': obj_dist, 'amplitude': amplitude, 'strat_effet': strat_effet,
            'distance': dist_real, 'lie': lie, 'real_effet': real_effet, 'contact': contact,
            'direction': direction, 'score_lateral': score_lat,
            'delta_dist': delta, 'err_longueur': err_L,
            'type_coup': 'Putt' if club == "Putter" else 'Jeu Long'
        })
        st.success("Données sauvegardées !")

# --- ANALYSES ---
if st.session_state['coups']:
    df = pd.DataFrame(st.session_state['coups'])
    df_long = df[df['type_coup'] == 'Jeu Long']
else:
    df_long = pd.DataFrame()

# --- ONGLET 2 : CALIBRATION (V10 Focus) ---
with tab_tech:
    if not df_long.empty:
        st.header("📏 Distances & Calibration (V10)")
        
        # Filtre Amplitude
        f_ampli = st.selectbox("Filtrer par Amplitude", ["Plein", "3/4", "1/2", "Tout"], index=0)
        df_viz = df_long if f_ampli == "Tout" else df_long[df_long['amplitude'] == f_ampli]
        
        # Tableau Croisé Lie vs Club
        pivot_lie = pd.pivot_table(df_viz, values='distance', index='club', columns='lie', aggfunc='mean').round(1)
        # Tri logique
        valid_clubs = [c for c in CLUBS_ORDER if c in pivot_lie.index]
        pivot_lie = pivot_lie.reindex(valid_clubs)
        
        st.dataframe(pivot_lie.style.background_gradient(cmap="YlGnBu", axis=None).format("{:.1f}m"), use_container_width=True)
        st.caption(f"Distances moyennes pour les coups : {f_ampli}")

# --- ONGLET 3 : DISPERSION (V9 Focus) ---
with tab_strat:
    if not df_long.empty:
        st.header("🎯 Dispersion & Ellipses (V9)")
        
        col_list, col_graph = st.columns([1, 3])
        with col_list:
            club_graph = st.selectbox("Choisir Club", df_long['club'].unique())
        
        with col_graph:
            subset = df_long[df_long['club'] == club_graph]
            if len(subset) > 3:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Coordonnées (V9 Logic)
                def get_x(row):
                    x = row['score_lateral'] * 5 # 1pt = 5m approx
                    if row['direction'] == 'Gauche': return -x
                    if row['direction'] == 'Droite': return x
                    return 0 + np.random.normal(0,1) # Jitter
                
                subset['x_viz'] = subset.apply(get_x, axis=1)
                
                # Fonction Ellipse
                def draw_ellipse(x, y, ax, color, label):
                    if len(x) < 3: return
                    cov = np.cov(x, y)
                    lambda_, v = np.linalg.eig(cov)
                    lambda_ = np.sqrt(lambda_)
                    ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                                  width=lambda_[0]*4, height=lambda_[1]*4,
                                  angle=np.rad2deg(np.arccos(v[0, 0])), 
                                  edgecolor=color, facecolor=color, alpha=0.2, label=label)
                    ax.add_artist(ell)
                    ax.scatter(x, y, c=color, s=30, alpha=0.6)

                # Séparation Practice / Parcours
                draw_ellipse(subset[subset['mode']=='Practice']['x_viz'], subset[subset['mode']=='Practice']['distance'], ax, 'blue', 'Practice')
                draw_ellipse(subset[subset['mode']=='Parcours']['x_viz'], subset[subset['mode']=='Parcours']['distance'], ax, 'red', 'Parcours')
                
                # Cible Moyenne
                ax.scatter([0], [subset['strat_dist'].mean()], c='green', marker='*', s=200, label='Cible')
                
                ax.set_xlabel("Gauche <---> Droite")
                ax.set_ylabel("Distance")
                ax.legend()
                ax.autoscale()
                st.pyplot(fig)
                

[Image of box plot chart explanation]

                st.info("Ellipse Rouge (Parcours) vs Bleue (Practice). Une ellipse rouge 'large' indique un problème de face de club sous pression.")

# --- ONGLET 4 : RATÉS & GAP (Mix V9/V10) ---
with tab_fail:
    if not df_long.empty:
        st.header("🔥 Analyse des Ratés & Gaps")
        
        c1, c2 = st.columns(2)
        
        # 1. HEATMAP (V10)
        with c1:
            st.subheader("1. La Heatmap des Ratés")
            heatmap_data = df_long.groupby(['err_longueur', 'direction']).size().unstack(fill_value=0)
            # Reindex pour ordre visuel logique
            y_ord = ["Long", "Bonne Longueur", "Court"]
            x_ord = ["Gauche", "Centre", "Droite"]
            heatmap_data = heatmap_data.reindex(index=y_ord, columns=x_ord, fill_value=0)
            
            fig_h, ax_h = plt.subplots()
            sns.heatmap(heatmap_data, annot=True, fmt='d', cmap="Reds", cbar=False, ax=ax_h)
            st.pyplot(fig_h)
        
        # 2. GAP ANALYSIS (V9)
        with c2:
            st.subheader("2. Gap Intention vs Réalité")
            fig_g, ax_g = plt.subplots()
            sns.histplot(df_long['delta_dist'], kde=True, ax=ax_g, color="purple")
            ax_g.axvline(0, color='green', ls='--', label='Cible')
            ax_g.axvline(df_long['delta_dist'].mean(), color='red', label='Moyenne')
            ax_g.set_title("Ecart de distance (Mètres)")
            ax_g.legend()
            st.pyplot(fig_g)
            
            bias = df_long['delta_dist'].mean()
            st.metric("Biais moyen", f"{bias:.1f}m", help="Si négatif, vous jouez trop court.")

        # 3. EFFETS (V9)
        st.divider()
        st.subheader("3. Maîtrise des Effets (Fade/Draw)")
        df_eff = df_long[df_long['strat_effet'].isin(['Fade', 'Draw'])]
        if not df_eff.empty:
            df_eff['reussite'] = (df_eff['strat_effet'] == df_eff['real_effet'])
            st.bar_chart(df_eff.groupby('strat_effet')['reussite'].mean() * 100)
            st.caption("% de réussite quand l'effet est annoncé.")
        else:
            st.info("Aucun effet (Fade/Draw) tenté.")
