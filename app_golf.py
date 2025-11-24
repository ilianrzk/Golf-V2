import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse, Circle
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Golf Deep Data", layout="wide")

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

PUTT_RESULTS = [
    "Dans le trou", 
    "Court", "Long", "Gauche", "Droite",
    "Court-Gauche", "Court-Droite", "Long-Gauche", "Long-Droite"
]

DIST_REF = {
    "Driver": 220, "Bois 5": 200, "Hybride": 180, "Fer 3": 170,
    "Fer 5": 160, "Fer 6": 150, "Fer 7": 140, "Fer 8": 130, "Fer 9": 120,
    "PW": 110, "50°": 100, "55°": 90, "60°": 80, "Putter": 3
}

# --- BARRE LATÉRALE : DATA LAB ---
st.sidebar.title("⚙️ Gestion Données")

# 1. IMPORT CSV
uploaded_file = st.sidebar.file_uploader("📂 Importer Historique (CSV)", type="csv")
if uploaded_file is not None:
    try:
        df_loaded = pd.read_csv(uploaded_file)
        current_data = pd.DataFrame(st.session_state['coups'])
        combined = pd.concat([current_data, df_loaded], ignore_index=True).drop_duplicates()
        st.session_state['coups'] = combined.to_dict('records')
        st.sidebar.success(f"{len(df_loaded)} coups importés !")
    except Exception as e:
        st.sidebar.error(f"Erreur import : {e}")

st.sidebar.markdown("---")

# 2. GÉNÉRATEUR V19
if st.sidebar.button("Générer Données Test V19"):
    new_data = []
    dates = [datetime.date.today() - datetime.timedelta(days=x) for x in range(60)]
    
    for _ in range(500):
        is_putt = np.random.choice([True, False], p=[0.3, 0.7])
        mode = np.random.choice(["Practice", "Parcours"], p=[0.4, 0.6])
        current_date = str(np.random.choice(dates))

        # Simulation Par du trou
        par_trou = np.random.choice([3, 4, 5], p=[0.2, 0.5, 0.3]) if mode == "Parcours" else 0

        if is_putt:
            club = "Putter"
            obj_dist = np.random.exponential(4)
            if obj_dist < 0.5: obj_dist = 0.5
            success_prob = max(0.1, 1 - (obj_dist / 6))
            res_putt = "Dans le trou" if np.random.random() < success_prob else np.random.choice(PUTT_RESULTS[1:])
            
            new_data.append({
                'date': current_date, 'mode': mode, 'club': club, 'par_trou': par_trou,
                'strat_dist': round(obj_dist, 1), 'distance': round(obj_dist, 1),
                'resultat_putt': res_putt, 'type_coup': 'Putt',
                'pente': np.random.choice(["Plat", "G-D", "D-G"]),
                'amplitude': 'Plein', 'lie': 'Green', 'strat_type': 'Putt',
                'real_effet': 'N/A', 'strat_effet': 'N/A'
            })
            
        else:
            club = np.random.choice(["Driver", "Fer 7", "PW", "55°"])
            
            if mode == "Practice":
                shot_type = "Practice"
                lie = "Tee" if club == "Driver" else "Tapis"
            else:
                if club == "Driver": shot_type = "Départ (Tee Shot)"
                elif club == "55°": shot_type = np.random.choice(["Approche (<50m)", "Sortie de Bunker"], p=[0.7, 0.3])
                else: shot_type = "Attaque de Green"
                lie = "Tee" if shot_type == "Départ (Tee Shot)" else ("Bunker" if "Bunker" in shot_type else "Fairway")

            ampli = "Plein" if shot_type != "Approche (<50m)" else "1/2"
            
            dist_target = DIST_REF[club]
            if ampli == "1/2": dist_target *= 0.5
            
            std_dev = 5 if mode == "Practice" else 15
            dist_real = np.random.normal(dist_target, std_dev)
            
            lat_score = min(5, int(abs(np.random.normal(0, 1.5 if mode=="Practice" else 2.5))))
            direction = "Centre" if lat_score == 0 else np.random.choice(["Gauche", "Droite"])
            
            # Simulation Effets
            strat_eff = np.random.choice(["Tout droit", "Fade", "Draw"], p=[0.7, 0.15, 0.15])
            if np.random.random() > 0.6: real_eff = strat_eff # Réussite
            else: real_eff = "Raté"
            
            delta = dist_real - dist_target
            err_L = "Court" if delta < -5 else ("Long" if delta > 5 else "Bonne Longueur")
            
            new_data.append({
                'date': current_date, 'mode': mode, 'club': club, 'par_trou': par_trou,
                'strat_dist': int(dist_target), 'strat_type': shot_type, 'amplitude': ampli,
                'distance': round(dist_real, 1), 'lie': lie, 'direction': direction, 
                'score_lateral': lat_score, 'delta_dist': delta, 'err_longueur': err_L,
                'type_coup': 'Jeu Long', 'resultat_putt': "N/A",
                'strat_effet': strat_eff, 'real_effet': real_eff
            })
            
    st.session_state['coups'].extend(new_data)
    st.sidebar.success("Données V19 générées !")

# 3. EXPORT CSV
if st.session_state['coups']:
    df_ex = pd.DataFrame(st.session_state['coups'])
    st.sidebar.download_button("📥 Sauvegarder CSV", convert_df(df_ex), "golf_v19.csv", "text/csv")
    
if st.sidebar.button("🗑️ Reset Tout"): st.session_state['coups'] = []

# --- INTERFACE ---
st.title("🏌️‍♂️ GolfShot 19.0 : Deep Data")

tab_saisie, tab_dual, tab_sac, tab_putt = st.tabs([
    "📝 Saisie", 
    "🧬 Analyse Club & Effets", 
    "🎒 Bag Mapping",
    "🟢 Analyse Putting 360"
])

# --- HELPER GRAPH ---
def plot_dispersion_ellipse(ax, data, title, color_main):
    if data.empty:
        ax.text(0.5, 0.5, "Pas de données", ha='center', va='center')
        return

    def get_x(row):
        x = row['score_lateral'] * 5 
        if row['direction'] == 'Gauche': return -x
        if row['direction'] == 'Droite': return x
        return 0 + np.random.normal(0, 1)
    
    data = data.copy()
    data['x_viz'] = data.apply(get_x, axis=1)
    
    ax.scatter(data['x_viz'], data['distance'], c=color_main, alpha=0.6, s=60, edgecolors='white')
    target = data['strat_dist'].mean()
    if target > 0:
        ax.scatter([0], [target], c='green', marker='*', s=150, label='Cible')
    
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

# --- ONGLET 1 : SAISIE ---
with tab_saisie:
    col_m, col_c = st.columns(2)
    with col_m: mode = st.radio("Mode", ["Parcours ⛳", "Practice 🚜"], horizontal=True)
    with col_c: club = st.selectbox("Club en main", CLUBS_ORDER)
    
    if "Parcours" in mode:
        par_trou = st.selectbox("Par du trou", [3, 4, 5])
    else:
        par_trou = 0

    st.markdown("---")
    col_int, col_real = st.columns(2)

    with col_int:
        st.subheader("1️⃣ STRATÉGIE")
        if club == "Putter":
            obj_dist = st.number_input("Dist. Cible (m)", 0.0, 30.0, 3.0, 0.1)
            strat_effet = "N/A"
        else:
            if "Practice" in mode:
                shot_type = "Practice"
            else:
                shot_type = st.selectbox("Type de Coup", SHOT_TYPES)
            
            ask_target_dist = True
            if "Parcours" in mode and shot_type == "Départ (Tee Shot)" and par_trou in [4, 5]:
                ask_target_dist = False
                st.info("🎯 Objectif : Distance Max / Fairway")
                obj_dist = 0
            
            if ask_target_dist:
                obj_dist = st.number_input("Dist. Cible", 0, 350, DIST_REF.get(club, 100))
            
            amplitude = st.radio("Amplitude", ["Plein", "3/4", "1/2"], horizontal=True)
            strat_effet = st.selectbox("Effet Voulu", ["Tout droit", "Fade", "Draw", "Balle Basse"])

    with col_real:
        st.subheader("2️⃣ RÉSULTAT")
        if club == "Putter":
            res_putt = st.selectbox("Résultat du Putt", PUTT_RESULTS)
            dist_real = obj_dist 
            lie, score_lat, direction, amplitude = "Green", 0, "Centre", "Plein"
            shot_type = "Putt"
            real_effet = "N/A"
        else:
            dist_real = st.number_input("Distance Réelle (m)", 0, 350, int(obj_dist) if obj_dist > 0 else 200)
            
            if not ask_target_dist:
                obj_dist = dist_real

            lie = st.selectbox("Situation (Lie)", ["Tee", "Fairway", "Rough", "Bunker"])
            
            c_eff, c_dummy = st.columns(2)
            with c_eff:
                real_effet = st.selectbox("Effet Réalisé (Vol de balle)", ["Tout droit", "Fade", "Draw", "Push (Droite)", "Pull (Gauche)", "Hook", "Slice", "Top", "Gratte"])
            
            c_dir, c_sco = st.columns(2)
            with c_dir: direction = st.radio("Direction", ["Gauche", "Centre", "Droite"], horizontal=True)
            with c_sco: score_lat = st.number_input("Écart (0-5)", 0, 5, 0) if direction != "Centre" else 0
            res_putt = "N/A"

    if st.button("💾 Enregistrer Coup", type="primary", use_container_width=True):
        delta = dist_real - obj_dist
        err_L = "Court" if delta < -5 else ("Long" if delta > 5 else "Bonne Longueur")
        st.session_state['coups'].append({
            'date': str(datetime.date.today()), 'mode': mode.split()[0], 'club': club, 'par_trou': par_trou,
            'strat_dist': obj_dist, 'strat_type': shot_type, 'amplitude': amplitude, 'strat_effet': strat_effet,
            'distance': dist_real, 'lie': lie, 'resultat_putt': res_putt,
            'direction': direction, 'score_lateral': score_lat, 'real_effet': real_effet,
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

# --- ONGLET 2 : ANALYSE DUAL VISION + EFFETS ---
with tab_dual:
    if not df_long.empty:
        st.header("🧬 Club DNA")
        
        sel_club = st.selectbox("Choisir Club pour Analyse", df_long['club'].unique())
        df_c = df_long[df_long['club'] == sel_club]
        
        # 1. DISPERSION
        col_prac, col_parc = st.columns(2)
        df_practice = df_c[df_c['mode'] == 'Practice']
        df_parcours = df_c[df_c['mode'] == 'Parcours']
        
        def calc_lat_disp(d):
            if d.empty: return 0
            lats = []
            for _, r in d.iterrows():
                val = r['score_lateral'] * 5
                if r['direction'] == 'Gauche': val = -val
                if r['direction'] == 'Centre': val = 0
                lats.append(val)
            return np.std(lats)

        with col_prac:
            fig1, ax1 = plt.subplots(figsize=(5, 5))
            plot_dispersion_ellipse(ax1, df_practice, "Practice (Labo)", "blue")
            st.pyplot(fig1)
            if not df_practice.empty:
                # MODIFICATION : AJOUT DISPERSION PROFONDEUR
                st.metric("Dispersion Latérale", f"± {calc_lat_disp(df_practice):.1f}m")
                st.metric("Dispersion Profondeur", f"± {df_practice['distance'].std():.1f}m")

        with col_parc:
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            plot_dispersion_ellipse(ax2, df_parcours, "Parcours (Réalité)", "red")
            st.pyplot(fig2)
            if not df_parcours.empty:
                # MODIFICATION : AJOUT DISPERSION PROFONDEUR
                st.metric("Dispersion Latérale", f"± {calc_lat_disp(df_parcours):.1f}m")
                st.metric("Dispersion Profondeur", f"± {df_parcours['distance'].std():.1f}m")
        
        st.markdown("---")
        
        # 2. MAÎTRISE DES EFFETS (TABLEAU)
        st.subheader("🎨 Maîtrise des Effets")
        df_effets = df_c[df_c['strat_effet'].isin(["Fade", "Draw", "Tout droit", "Balle Basse"])]
        
        if not df_effets.empty:
            df_effets['Reussite'] = df_effets.apply(lambda x: 1 if x['strat_effet'] in x['real_effet'] else 0, axis=1)
            
            # MODIFICATION : REMPLACEMENT GRAPH PAR TABLEAU
            summary_effets = df_effets.groupby('strat_effet').agg(
                Tentatives=('strat_effet', 'count'),
                Reussites=('Reussite', 'sum'),
                Taux=('Reussite', 'mean')
            )
            # Mise en forme
            summary_effets['Taux'] = (summary_effets['Taux'] * 100).round(1)
            summary_effets.columns = ['Tentatives', 'Réussites', '% Réussite']
            
            st.dataframe(
                summary_effets.style.background_gradient(cmap="Greens", subset=['% Réussite'])
                                    .format("{:.1f}%", subset=['% Réussite']),
                use_container_width=True
            )
            st.caption("Ce tableau vous indique précisément la fiabilité de vos effets annoncés.")
            
        else:
            st.info("Aucun effet spécifique annoncé pour ce club.")

    else:
        st.info("En attente de données de jeu long.")

# --- ONGLET 3 : BAG MAPPING ---
with tab_sac:
    if not df_long.empty:
        st.header("🎒 Étalonnage du Sac")
        options_filtre = ["Tous les coups"] + [t for t in SHOT_TYPES]
        f_type = st.selectbox("Filtrer par situation", options_filtre, index=0)
        
        if f_type == "Tous les coups":
            df_sac = df_long.copy()
        else:
            df_sac = df_long[df_long['strat_type'] == f_type].copy()
        
        if not df_sac.empty:
            df_sac['club'] = pd.Categorical(df_sac['club'], categories=CLUBS_ORDER, ordered=True)
            df_sac = df_sac.sort_values('club')
            
            fig_bag, ax_bag = plt.subplots(figsize=(12, 5))
            sns.boxplot(x='club', y='distance', data=df_sac, ax=ax_bag, palette="viridis")
            ax_bag.grid(True, axis='y', alpha=0.3)
            ax_bag.set_title(f"Distances : {f_type}")
            st.pyplot(fig_bag)
            
            stats = df_sac.groupby('club', observed=True)['distance'].agg(['count', 'mean', 'max', 'std']).round(1)
            stats.columns = ['Coups', 'Moyenne', 'Max', 'Écart Type (±m)']
            
            st.dataframe(
                stats.style.background_gradient(cmap="Blues", subset=['Moyenne'])
                           .background_gradient(cmap="Reds_r", subset=['Écart Type (±m)']), 
                use_container_width=True
            )
            st.info("💡 **Écart Type** : Plus ce chiffre est bas, plus vous êtes régulier avec ce club.")
        else:
            st.warning("Pas de données pour ce filtre.")
    else:
        st.info("Attente de données...")

# --- ONGLET 4 : PUTTING ---
with tab_putt:
    if not df_putt.empty:
        st.header("🟢 Putting Intelligence")
        col_p1, col_p2 = st.columns([1, 1])
        
        with col_p1:
            st.subheader("Boussole des Ratés")
            def get_putt_coords(row):
                if row['resultat_putt'] == "Dans le trou": return 0, 0
                d = 1
                r = row['resultat_putt']
                x, y = 0, 0
                if "Gauche" in r: x = -d
                if "Droite" in r: x = d
                if "Court" in r: y = -d
                if "Long" in r: y = d
                return x, y

            coords = df_putt.apply(get_putt_coords, axis=1, result_type='expand')
            df_putt['x'] = coords[0] + np.random.normal(0, 0.15, len(df_putt))
            df_putt['y'] = coords[1] + np.random.normal(0, 0.15, len(df_putt))
            
            fig_p, ax_p = plt.subplots(figsize=(6, 6))
            ax_p.add_patch(Circle((0,0), 0.2, color='green', alpha=0.3))
            
            sns.scatterplot(data=df_putt, x='x', y='y', hue='resultat_putt', s=100, ax=ax_p)
            ax_p.set_xlim(-2, 2); ax_p.set_ylim(-2, 2)
            ax_p.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            st.pyplot(fig_p)
            
        with col_p2:
            st.subheader("Analyse de Réussite")
            
            bins = [0, 1.5, 3, 6, 20]
            labels = ["0-1.5m", "1.5-3m", "3-6m", "+6m"]
            df_putt['Zone'] = pd.cut(df_putt['strat_dist'], bins=bins, labels=labels)
            
            counts = df_putt['resultat_putt'].value_counts(normalize=True) * 100
            df_stats_putt = pd.DataFrame(counts).reset_index()
            df_stats_putt.columns = ['Résultat', '%']
            
            fig_hist, ax_hist = plt.subplots(figsize=(5, 3))
            sns.barplot(data=df_stats_putt, x='Résultat', y='%', ax=ax_hist, palette="viridis")
            ax_hist.set_xticklabels(ax_hist.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(fig_hist)
            
            st.markdown("---")
            st.write("**Détail par Distance**")
            df_putt['Is_Made'] = df_putt['resultat_putt'] == "Dans le trou"
            
            stats_table = df_putt.groupby('Zone', observed=False).agg(
                Total=('resultat_putt', 'count'),
                Reussis=('Is_Made', 'sum')
            )
            stats_table['% Réussite'] = (stats_table['Reussis'] / stats_table['Total'] * 100).round(1)
            st.dataframe(stats_table.style.background_gradient(cmap="Greens", subset=['% Réussite']), use_container_width=True)
