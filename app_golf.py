import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse, Circle
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="GolfShot 15.0 Dual Vision", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;}
    h4 {color: #1565C0;}
</style>
""", unsafe_allow_html=True)

# --- DONNÉES & CONSTANTES ---
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

SHOT_TYPES = [
    "Départ (Tee Shot)", 
    "Attaque de Green", 
    "Lay-up / Sécurité", 
    "Approche (<50m)", 
    "Sortie de Bunker", 
    "Recovery"
]

PUTT_RESULTS = ["Dans le trou", "Raté - Court", "Raté - Long", "Raté - Gauche", "Raté - Droite"]

DIST_REF = {
    "Driver": 220, "Bois 5": 200, "Hybride": 180, "Fer 3": 170,
    "Fer 5": 160, "Fer 6": 150, "Fer 7": 140, "Fer 8": 130, "Fer 9": 120,
    "PW": 110, "50°": 100, "55°": 90, "60°": 80, "Putter": 3
}

# --- BARRE LATÉRALE : DATA LAB ---
st.sidebar.title("⚙️ Gestion Données")

# 1. IMPORT CSV (NOUVEAU)
uploaded_file = st.sidebar.file_uploader("📂 Importer Historique (CSV)", type="csv")
if uploaded_file is not None:
    try:
        df_loaded = pd.read_csv(uploaded_file)
        # On ajoute les données importées à celles existantes
        current_data = pd.DataFrame(st.session_state['coups'])
        combined = pd.concat([current_data, df_loaded], ignore_index=True).drop_duplicates()
        st.session_state['coups'] = combined.to_dict('records')
        st.sidebar.success(f"{len(df_loaded)} coups importés avec succès !")
    except Exception as e:
        st.sidebar.error(f"Erreur lors de l'import : {e}")

st.sidebar.markdown("---")

# 2. GÉNÉRATEUR V15
if st.sidebar.button("Générer Données Test V15"):
    new_data = []
    dates = [datetime.date.today() - datetime.timedelta(days=x) for x in range(60)]
    
    for _ in range(400):
        is_putt = np.random.choice([True, False], p=[0.25, 0.75])
        mode = np.random.choice(["Practice", "Parcours"], p=[0.4, 0.6])
        current_date = str(np.random.choice(dates))

        if is_putt:
            # Génération Putt
            club = "Putter"
            obj_dist = np.random.exponential(4)
            if obj_dist < 0.5: obj_dist = 0.5
            success_prob = max(0.1, 1 - (obj_dist / 6))
            
            if np.random.random() < success_prob:
                res_putt = "Dans le trou"
                dist_real = obj_dist
            else:
                res_putt = np.random.choice(PUTT_RESULTS[1:], p=[0.4, 0.2, 0.2, 0.2])
                dist_real = obj_dist

            new_data.append({
                'date': current_date, 'mode': mode, 'club': club,
                'strat_dist': round(obj_dist, 1), 'distance': round(dist_real, 1),
                'resultat_putt': res_putt, 'type_coup': 'Putt',
                'pente': np.random.choice(["Plat", "G-D", "D-G"]),
                'amplitude': 'Plein', 'lie': 'Green', 'strat_type': 'Putt'
            })
            
        else:
            # Génération Jeu Long
            club = np.random.choice(["Driver", "Fer 7", "PW", "55°"])
            
            # Type de Coup cohérent
            if club == "Driver": shot_type = "Départ (Tee Shot)"
            elif club == "55°": shot_type = np.random.choice(["Approche (<50m)", "Sortie de Bunker"], p=[0.7, 0.3])
            else: shot_type = "Attaque de Green"

            lie = "Tee" if shot_type == "Départ (Tee Shot)" else ("Bunker" if "Bunker" in shot_type else "Fairway")
            ampli = "Plein" if "Approche" not in shot_type else "1/2"
            
            # Physique du coup
            dist_target = DIST_REF[club]
            if ampli == "1/2": dist_target *= 0.5
            
            # Variance Practice vs Parcours
            std_dev = 5 if mode == "Practice" else 15
            dist_real = np.random.normal(dist_target, std_dev)
            
            # Dispersion Latérale
            # Practice = plus précis (0-2), Parcours = moins précis (0-5)
            max_lat = 2 if mode == "Practice" else 5
            lat_score = min(5, int(abs(np.random.normal(0, 1.5 if mode=="Practice" else 2.5))))
            direction = "Centre" if lat_score == 0 else np.random.choice(["Gauche", "Droite"])
            
            delta = dist_real - dist_target
            err_L = "Court" if delta < -5 else ("Long" if delta > 5 else "Bonne Longueur")
            
            new_data.append({
                'date': current_date, 'mode': mode, 'club': club,
                'strat_dist': int(dist_target), 'strat_type': shot_type, 'amplitude': ampli,
                'distance': round(dist_real, 1), 'lie': lie, 'direction': direction, 
                'score_lateral': lat_score, 'delta_dist': delta, 'err_longueur': err_L,
                'type_coup': 'Jeu Long', 'resultat_putt': "N/A"
            })
            
    st.session_state['coups'].extend(new_data)
    st.sidebar.success("Données V15 générées !")

# 3. EXPORT CSV
if st.session_state['coups']:
    df_ex = pd.DataFrame(st.session_state['coups'])
    st.sidebar.download_button("📥 Sauvegarder CSV", convert_df(df_ex), "golf_v15.csv", "text/csv")
    
if st.sidebar.button("🗑️ Reset Tout"): st.session_state['coups'] = []

# --- INTERFACE ---
st.title("🏌️‍♂️ GolfShot 15.0 : Dual Vision")

tab_saisie, tab_dual, tab_sac, tab_putt = st.tabs([
    "📝 Saisie", 
    "🧬 Analyse Club (Dispersion)", 
    "🎒 Bag Mapping",
    "🟢 Analyse Putting"
])

# --- FONCTION HELPER POUR GRAPHES ---
def plot_dispersion_ellipse(ax, data, title, color_main):
    """Trace un graphe de dispersion avec ellipse pour un dataset donné"""
    if data.empty:
        ax.text(0.5, 0.5, "Pas de données", ha='center', va='center')
        return

    # Simulation X (Latéral) basé sur le score 0-5
    def get_x(row):
        x = row['score_lateral'] * 5 # 1 point = 5m approx
        if row['direction'] == 'Gauche': return -x
        if row['direction'] == 'Droite': return x
        return 0 + np.random.normal(0, 1) # Jitter
    
    data = data.copy()
    data['x_viz'] = data.apply(get_x, axis=1)
    
    # Scatter plot
    ax.scatter(data['x_viz'], data['distance'], c=color_main, alpha=0.6, s=60, edgecolors='white')
    
    # Cible
    target = data['strat_dist'].mean()
    ax.scatter([0], [target], c='green', marker='*', s=150, label='Cible')
    
    # Ellipse
    if len(data) > 3:
        cov = np.cov(data['x_viz'], data['distance'])
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        ell = Ellipse(xy=(data['x_viz'].mean(), data['distance'].mean()),
                      width=lambda_[0]*4, height=lambda_[1]*4,
                      angle=np.rad2deg(np.arccos(v[0, 0])), 
                      edgecolor=color_main, facecolor=color_main, alpha=0.15, linewidth=2)
        ax.add_artist(ell)
        
    ax.set_title(title)
    ax.set_xlabel("Gauche (m) <---> Droite (m)")
    ax.set_ylabel("Distance (m)")
    ax.grid(True, alpha=0.3)
    # Fixer les limites pour comparaison honnête si possible, sinon auto
    # ax.set_xlim(-30, 30) 

# --- ONGLET 1 : SAISIE (V14 Standard) ---
with tab_saisie:
    col_m, col_c = st.columns(2)
    with col_m: mode = st.radio("Mode", ["Parcours ⛳", "Practice 🚜"], horizontal=True)
    with col_c: club = st.selectbox("Club en main", CLUBS_ORDER)
    
    st.markdown("---")
    col_int, col_real = st.columns(2)

    with col_int:
        st.subheader("1️⃣ STRATÉGIE")
        if club == "Putter":
            obj_dist = st.number_input("Dist. Cible (m)", 0.0, 30.0, 3.0, 0.1)
        else:
            shot_type = st.selectbox("Type de Coup", SHOT_TYPES)
            obj_dist = st.number_input("Dist. Cible", 0, 350, DIST_REF.get(club, 100))
            amplitude = st.radio("Amplitude", ["Plein", "3/4", "1/2"], horizontal=True)
            strat_effet = st.selectbox("Effet Voulu", ["Tout droit", "Fade", "Draw"])

    with col_real:
        st.subheader("2️⃣ RÉSULTAT")
        if club == "Putter":
            res_putt = st.selectbox("Résultat du Putt", PUTT_RESULTS)
            dist_real = obj_dist 
            lie, score_lat, direction, amplitude = "Green", 0, "Centre", "Plein"
            shot_type = "Putt"
        else:
            dist_real = st.number_input("Distance Réelle (m)", 0, 350, int(obj_dist))
            lie = st.selectbox("Situation (Lie)", ["Tee", "Fairway", "Rough", "Bunker"])
            c_dir, c_sco = st.columns(2)
            with c_dir: direction = st.radio("Direction", ["Gauche", "Centre", "Droite"], horizontal=True)
            with c_sco: score_lat = st.number_input("Écart (0-5)", 0, 5, 0) if direction != "Centre" else 0
            res_putt = "N/A"

    if st.button("💾 Enregistrer Coup", type="primary", use_container_width=True):
        delta = dist_real - obj_dist
        err_L = "Court" if delta < -5 else ("Long" if delta > 5 else "Bonne Longueur")
        st.session_state['coups'].append({
            'date': str(datetime.date.today()), 'mode': mode.split()[0], 'club': club,
            'strat_dist': obj_dist, 'strat_type': shot_type, 'amplitude': amplitude,
            'distance': dist_real, 'lie': lie, 'resultat_putt': res_putt,
            'direction': direction, 'score_lateral': score_lat,
            'delta_dist': delta, 'err_longueur': err_L,
            'type_coup': 'Putt' if club == "Putter" else 'Jeu Long'
        })
        st.success("Sauvegardé !")

# --- DATA LOAD ---
if st.session_state['coups']:
    df = pd.DataFrame(st.session_state['coups'])
    df_long = df[df['type_coup'] == 'Jeu Long']
    df_putt = df[df['type_coup'] == 'Putt']
else:
    df_long = pd.DataFrame()
    df_putt = pd.DataFrame()

# --- ONGLET 2 : ANALYSE DUAL VISION (NOUVEAU) ---
with tab_dual:
    if not df_long.empty:
        st.header("🧬 Club DNA : Dispersion Comparée")
        
        # Sélecteur
        sel_club = st.selectbox("Choisir Club pour Analyse", df_long['club'].unique())
        df_c = df_long[df_long['club'] == sel_club]
        
        # --- SECTION 1 : PRACTICE vs PARCOURS ---
        st.subheader("1. Réalité (Parcours) vs Entraînement (Practice)")
        col_prac, col_parc = st.columns(2)
        
        df_practice = df_c[df_c['mode'] == 'Practice']
        df_parcours = df_c[df_c['mode'] == 'Parcours']
        
        with col_prac:
            fig1, ax1 = plt.subplots(figsize=(5, 5))
            plot_dispersion_ellipse(ax1, df_practice, "Practice (Labo)", "blue")
            st.pyplot(fig1)
            if not df_practice.empty:
                st.metric("Dispersion Practice", f"± {df_practice['distance'].std():.1f}m")

        with col_parc:
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            plot_dispersion_ellipse(ax2, df_parcours, "Parcours (Réalité)", "red")
            st.pyplot(fig2)
            if not df_parcours.empty:
                st.metric("Dispersion Parcours", f"± {df_parcours['distance'].std():.1f}m")
        
        st.markdown("---")
        
        # --- SECTION 2 : PAR TYPE DE COUP ---
        st.subheader("2. Analyse par Stratégie de Coup")
        
        types_dispo = df_c['strat_type'].unique()
        if len(types_dispo) > 0:
            sel_type = st.selectbox("Filtrer par Type de Coup", types_dispo)
            df_strat = df_c[df_c['strat_type'] == sel_type]
            
            col_s1, col_s2 = st.columns([2, 1])
            with col_s1:
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                # On colore par Mode pour voir la différence dans ce contexte
                colors = {'Practice': 'blue', 'Parcours': 'red'}
                
                # Plot manuel pour gérer les couleurs
                for m in df_strat['mode'].unique():
                    subset = df_strat[df_strat['mode'] == m]
                    plot_dispersion_ellipse(ax3, subset, f"Dispersion : {sel_type}", colors[m])
                
                # Légende manuelle
                from matplotlib.lines import Line2D
                custom_lines = [Line2D([0], [0], color='blue', lw=4),
                                Line2D([0], [0], color='red', lw=4)]
                ax3.legend(custom_lines, ['Practice', 'Parcours'])
                
                st.pyplot(fig3)
            
            with col_s2:
                st.write(f"**Stats : {sel_type}**")
                st.write(f"Coups analysés : {len(df_strat)}")
                st.write(f"Moyenne : {df_strat['distance'].mean():.1f}m")
                st.write(f"Ecart Type : {df_strat['distance'].std():.1f}m")

        else:
            st.info("Pas de types de coups définis pour ce club.")

    else:
        st.info("En attente de données de jeu long.")

# --- ONGLET 3 : BAG MAPPING ---
with tab_sac:
    if not df_long.empty:
        st.header("🎒 Étalonnage du Sac")
        f_type = st.selectbox("Filtrer par situation", SHOT_TYPES, index=1)
        df_sac = df_long[df_long['strat_type'] == f_type].copy()
        
        if not df_sac.empty:
            df_sac['club'] = pd.Categorical(df_sac['club'], categories=CLUBS_ORDER, ordered=True)
            df_sac = df_sac.sort_values('club')
            
            fig_bag, ax_bag = plt.subplots(figsize=(12, 5))
            sns.boxplot(x='club', y='distance', data=df_sac, ax=ax_bag, palette="viridis")
            ax_bag.grid(True, axis='y', alpha=0.3)
            ax_bag.set_title(f"Distances pour : {f_type}")
            st.pyplot(fig_bag)
            
            stats = df_sac.groupby('club', observed=True)['distance'].agg(['count', 'mean', 'max']).round(1)
            st.dataframe(stats.style.background_gradient(cmap="Blues"), use_container_width=True)
        else:
            st.warning("Pas de données.")
    else:
        st.info("Attente de données...")

# --- ONGLET 4 : PUTTING ---
with tab_putt:
    if not df_putt.empty:
        st.header("🟢 Putting Intelligence")
        col_p1, col_p2 = st.columns([2, 1])
        
        with col_p1:
            st.subheader("La Boussole des Ratés")
            def get_putt_coords(row):
                if row['resultat_putt'] == "Dans le trou": return 0, 0
                d = 1
                if "Court" in row['resultat_putt']: return 0, -d
                if "Long" in row['resultat_putt']: return 0, d
                if "Gauche" in row['resultat_putt']: return -d, 0
                if "Droite" in row['resultat_putt']: return d, 0
                return 0, 0

            coords = df_putt.apply(get_putt_coords, axis=1, result_type='expand')
            df_putt['x'] = coords[0] + np.random.normal(0, 0.1, len(df_putt))
            df_putt['y'] = coords[1] + np.random.normal(0, 0.1, len(df_putt))
            
            fig_p, ax_p = plt.subplots(figsize=(6, 6))
            ax_p.add_patch(Circle((0,0), 0.2, color='green', alpha=0.3))
            colors = {"Dans le trou": "green", "Raté - Court": "orange", "Raté - Long": "red", "Raté - Gauche": "blue", "Raté - Droite": "purple"}
            sns.scatterplot(data=df_putt, x='x', y='y', hue='resultat_putt', palette=colors, s=100, ax=ax_p)
            ax_p.set_xlim(-2, 2); ax_p.set_ylim(-2, 2)
            st.pyplot(fig_p)
            
        with col_p2:
            st.subheader("Stats")
            bins = [0, 1.5, 3, 6, 20]
            labels = ["0-1.5m", "1.5-3m", "3-6m", "+6m"]
            df_putt['Zone'] = pd.cut(df_putt['strat_dist'], bins=bins, labels=labels)
            stats_zone = df_putt.groupby('Zone', observed=False)['resultat_putt'].apply(lambda x: (x == "Dans le trou").mean() * 100).round(1)
            st.dataframe(stats_zone.to_frame("% Réussite").style.background_gradient(cmap="Greens"))
