import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse, Circle
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="GolfShot 14.0 Precision", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;}
    h4 {color: #2E7D32;}
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
    "Recovery (Sortie de forêt)"
]

PUTT_RESULTS = ["Dans le trou", "Raté - Court", "Raté - Long", "Raté - Gauche", "Raté - Droite"]

DIST_REF = {
    "Driver": 220, "Bois 5": 200, "Hybride": 180, "Fer 3": 170,
    "Fer 5": 160, "Fer 6": 150, "Fer 7": 140, "Fer 8": 130, "Fer 9": 120,
    "PW": 110, "50°": 100, "55°": 90, "60°": 80, "Putter": 3
}

# --- GÉNÉRATEUR V14 ---
st.sidebar.title("⚙️ Data Lab V14")

if st.sidebar.button("Générer Dataset 'Precision'"):
    new_data = []
    dates = [datetime.date.today() - datetime.timedelta(days=x) for x in range(60)]
    
    for _ in range(500):
        # 30% Putts, 70% Jeu Long
        is_putt = np.random.choice([True, False], p=[0.3, 0.7])
        mode = np.random.choice(["Practice", "Parcours"], p=[0.2, 0.8])
        current_date = str(np.random.choice(dates))

        if is_putt:
            club = "Putter"
            # Distance Putt
            obj_dist = np.random.exponential(4) # Beaucoup de putts courts
            if obj_dist < 0.5: obj_dist = 0.5
            
            # Logique réussite (plus c'est loin, plus on rate)
            success_prob = max(0.1, 1 - (obj_dist / 7))
            
            if np.random.random() < success_prob:
                res_putt = "Dans le trou"
                dist_real = obj_dist
            else:
                # Type de raté
                res_putt = np.random.choice(
                    ["Raté - Court", "Raté - Long", "Raté - Gauche", "Raté - Droite"],
                    p=[0.4, 0.2, 0.2, 0.2] # On rate souvent court chez les amateurs
                )
                dist_real = obj_dist # On garde la distance cible pour l'analyse

            new_data.append({
                'date': current_date, 'mode': mode, 'club': club,
                'strat_dist': round(obj_dist, 1), 'distance': round(dist_real, 1),
                'resultat_putt': res_putt, 'type_coup': 'Putt',
                'pente': np.random.choice(["Plat", "G-D", "D-G"]),
                'amplitude': 'Plein', 'lie': 'Green'
            })
            
        else:
            # JEU LONG
            club = np.random.choice(["Driver", "Fer 7", "PW", "55°"])
            
            # Type de Coup Logique
            if club == "Driver": 
                shot_type = "Départ (Tee Shot)"
                lie = "Tee"
            elif club == "55°":
                shot_type = np.random.choice(["Approche (<50m)", "Sortie de Bunker", "Attaque de Green"], p=[0.5, 0.2, 0.3])
                lie = "Bunker" if "Bunker" in shot_type else "Fairway"
            else:
                shot_type = np.random.choice(["Attaque de Green", "Lay-up / Sécurité"], p=[0.8, 0.2])
                lie = np.random.choice(["Fairway", "Rough"], p=[0.7, 0.3])

            ampli = "Plein" if "Approche" not in shot_type else "1/2"
            
            # Calculs Distance
            dist_target = DIST_REF[club]
            if ampli == "1/2": dist_target *= 0.5
            
            dist_real = np.random.normal(dist_target, 10)
            
            # Erreurs
            delta = dist_real - dist_target
            err_L = "Court" if delta < -5 else ("Long" if delta > 5 else "Bonne Longueur")
            
            lat_score = min(5, int(abs(np.random.normal(0, 2))))
            direction = "Centre" if lat_score == 0 else np.random.choice(["Gauche", "Droite"])
            
            new_data.append({
                'date': current_date, 'mode': mode, 'club': club,
                'strat_dist': int(dist_target), 'strat_type': shot_type, # Nouveau champ
                'amplitude': ampli, 'distance': round(dist_real, 1), 
                'lie': lie, 'direction': direction, 'score_lateral': lat_score,
                'delta_dist': delta, 'err_longueur': err_L,
                'type_coup': 'Jeu Long', 'resultat_putt': "N/A"
            })
            
    st.session_state['coups'].extend(new_data)
    st.sidebar.success("500 Coups générés !")

if st.session_state['coups']:
    df_ex = pd.DataFrame(st.session_state['coups'])
    st.sidebar.download_button("📥 Export CSV", convert_df(df_ex), "golf_v14.csv", "text/csv")
    
if st.sidebar.button("🗑️ Reset"): st.session_state['coups'] = []

# --- INTERFACE ---
st.title("🏌️‍♂️ GolfShot 14.0 : Precision Strategy")

tab_saisie, tab_sac, tab_dna, tab_putt = st.tabs([
    "📝 Saisie", 
    "🎒 Bag Mapping", 
    "🧬 Club DNA",
    "🟢 Analyse Putting"
])

# --- ONGLET 1 : SAISIE ---
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
            # Pas d'effet au putt dans cette version simple
        else:
            # NOUVEAU : TYPE DE COUP
            shot_type = st.selectbox("Type de Coup", SHOT_TYPES)
            obj_dist = st.number_input("Dist. Cible", 0, 350, DIST_REF.get(club, 100))
            amplitude = st.radio("Amplitude", ["Plein", "3/4", "1/2"], horizontal=True)
            strat_effet = st.selectbox("Effet Voulu", ["Tout droit", "Fade", "Draw"])

    with col_real:
        st.subheader("2️⃣ RÉSULTAT")
        if club == "Putter":
            # NOUVEAU : RÉSULTAT DÉTAILLÉ
            res_putt = st.selectbox("Résultat du Putt", PUTT_RESULTS)
            pente = st.selectbox("Pente", ["Plat", "G-D", "D-G"])
            
            # Variables fictives pour homogénéité
            dist_real = obj_dist 
            lie = "Green"
            score_lat = 0
            direction = "Centre"
            shot_type = "Putt"
            amplitude = "Plein"
        else:
            dist_real = st.number_input("Distance Réelle (m)", 0, 350, int(obj_dist))
            lie = st.selectbox("Situation (Lie)", ["Tee", "Fairway", "Rough", "Bunker"])
            contact = st.selectbox("Contact", ["Bon", "Gratte", "Top", "Pointe"])
            
            c_dir, c_sco = st.columns(2)
            with c_dir: direction = st.radio("Direction", ["Gauche", "Centre", "Droite"], horizontal=True)
            with c_sco: score_lat = st.number_input("Écart (0-5)", 0, 5, 0) if direction != "Centre" else 0
            res_putt = "N/A"

    st.markdown("---")
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
        st.success("Données sauvegardées !")

# --- DATA PREP ---
if st.session_state['coups']:
    df = pd.DataFrame(st.session_state['coups'])
    df_long = df[df['type_coup'] == 'Jeu Long']
    df_putt = df[df['type_coup'] == 'Putt']
else:
    df_long = pd.DataFrame()
    df_putt = pd.DataFrame()

# --- ONGLET 2 : BAG MAPPING (Gapping) ---
with tab_sac:
    if not df_long.empty:
        st.header("🎒 Étalonnage par Type de Coup")
        
        # Filtre Type de Coup
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
            
            # Tableau Data
            stats = df_sac.groupby('club', observed=True)['distance'].agg(['count', 'mean', 'max']).round(1)
            stats.columns = ['Coups', 'Moyenne', 'Max']
            st.dataframe(stats.style.background_gradient(cmap="Blues"), use_container_width=True)
        else:
            st.warning(f"Aucune donnée pour '{f_type}'.")
    else:
        st.info("Attente de données...")

# --- ONGLET 3 : CLUB DNA ---
with tab_dna:
    if not df_long.empty:
        sel_club = st.selectbox("Club", df_long['club'].unique())
        df_c = df_long[df_long['club'] == sel_club]
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📊 Dispersion")
            fig, ax = plt.subplots()
            # Simulation X
            x_vals = [(-r['score_lateral']*5 if r['direction']=='Gauche' else r['score_lateral']*5) + np.random.normal(0,1) for i,r in df_c.iterrows()]
            sns.scatterplot(x=x_vals, y=df_c['distance'], hue=df_c['lie'], style=df_c['strat_type'], s=100, ax=ax)
            ax.set_title(f"Dispersion {sel_club}")
            st.pyplot(fig)
        
        with c2:
            st.subheader("📋 Stats par Lie")
            piv = df_c.groupby('lie')['distance'].agg(['mean', 'count']).round(1)
            st.dataframe(piv, use_container_width=True)

# --- ONGLET 4 : ANALYSE PUTTING (NOUVEAU) ---
with tab_putt:
    if not df_putt.empty:
        st.header("🟢 Putting Intelligence")
        
        col_p1, col_p2 = st.columns([2, 1])
        
        with col_p1:
            st.subheader("🎯 La Boussole des Ratés")
            # On veut visualiser où vont les balles ratées
            # On crée des coordonnées simulées pour la visualisation
            def get_putt_coords(row):
                if row['resultat_putt'] == "Dans le trou": return 0, 0
                dist = 1 # Unité arbitraire pour l'écart
                if "Court" in row['resultat_putt']: return 0, -dist
                if "Long" in row['resultat_putt']: return 0, dist
                if "Gauche" in row['resultat_putt']: return -dist, 0
                if "Droite" in row['resultat_putt']: return dist, 0
                return 0, 0

            coords = df_putt.apply(get_putt_coords, axis=1, result_type='expand')
            df_putt['x_putt'] = coords[0] + np.random.normal(0, 0.1, len(df_putt)) # Jitter
            df_putt['y_putt'] = coords[1] + np.random.normal(0, 0.1, len(df_putt))
            
            fig_p, ax_p = plt.subplots(figsize=(6, 6))
            
            # Zones
            ax_p.add_patch(Circle((0,0), 0.2, color='green', alpha=0.3, label='Trou'))
            ax_p.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax_p.axvline(0, color='gray', linestyle='--', alpha=0.5)
            
            # Points
            colors = {"Dans le trou": "green", "Raté - Court": "orange", "Raté - Long": "red", "Raté - Gauche": "blue", "Raté - Droite": "purple"}
            sns.scatterplot(data=df_putt, x='x_putt', y='y_putt', hue='resultat_putt', palette=colors, s=100, ax=ax_p)
            
            ax_p.set_xlim(-2, 2); ax_p.set_ylim(-2, 2)
            ax_p.set_xlabel("Gauche <---> Droite"); ax_p.set_ylabel("Court <---> Long")
            ax_p.set_xticks([]); ax_p.set_yticks([]) # Enlever les chiffres
            ax_p.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            st.pyplot(fig_p)
            
        with col_p2:
            st.subheader("📉 Stats de Réussite")
            
            # Buckets de distance
            bins = [0, 1.5, 3, 6, 20]
            labels = ["0-1.5m", "1.5-3m", "3-6m", "+6m"]
            df_putt['Zone'] = pd.cut(df_putt['strat_dist'], bins=bins, labels=labels)
            
            # Taux de réussite
            stats_zone = df_putt.groupby('Zone', observed=False)['resultat_putt'].apply(lambda x: (x == "Dans le trou").mean() * 100).round(1)
            
            st.write("**% de Réussite par Zone**")
            st.dataframe(stats_zone.to_frame(name="% Réussite").style.background_gradient(cmap="Greens"), use_container_width=True)
            
            st.write("**Type d'erreur dominant**")
            misses = df_putt[df_putt['resultat_putt'] != "Dans le trou"]
            if not misses.empty:
                counts = misses['resultat_putt'].value_counts()
                st.bar_chart(counts)
            else:
                st.success("Aucun raté enregistré !")

    else:
        st.info("En attente de données de Putting (Utilisez le générateur V14).")
