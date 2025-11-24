import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse, Circle
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="GolfShot 21.0 Pro Performance", layout="wide")

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
if 'parties' not in st.session_state: # Nouveau pour le Score Global
    st.session_state['parties'] = []

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

# 2. GÉNÉRATEUR V21 (MISE A JOUR TEMPORELLE + PROXIMITÉ)
if st.sidebar.button("Générer Données Test V21"):
    new_data = []
    # Génération sur 90 jours pour voir l'évolution
    dates = [datetime.date.today() - datetime.timedelta(days=x) for x in range(90)]
    dates.sort() # Du plus ancien au plus récent
    
    for i in range(600):
        # Simulation d'amélioration progressive (le joueur devient meilleur avec le temps)
        progress = i / 600 # 0 au début, 1 à la fin
        
        is_putt = np.random.choice([True, False], p=[0.3, 0.7])
        mode = np.random.choice(["Practice", "Parcours", "Combine Test"], p=[0.3, 0.5, 0.2])
        current_date = str(dates[i % 90])

        par_trou = np.random.choice([3, 4, 5], p=[0.2, 0.5, 0.3]) if mode == "Parcours" else 0

        if is_putt:
            club = "Putter"
            obj_dist = np.random.exponential(4)
            if obj_dist < 0.5: obj_dist = 0.5
            # Réussite s'améliore avec le temps
            base_prob = max(0.1, 1 - (obj_dist / 6))
            success_prob = base_prob + (0.1 * progress) 
            
            res_putt = "Dans le trou" if np.random.random() < success_prob else np.random.choice(PUTT_RESULTS[1:])
            
            new_data.append({
                'date': current_date, 'mode': mode, 'club': club, 'par_trou': par_trou,
                'strat_dist': round(obj_dist, 1), 'distance': round(obj_dist, 1),
                'resultat_putt': res_putt, 'type_coup': 'Putt',
                'pente': np.random.choice(["Plat", "G-D", "D-G"]),
                'amplitude': 'Plein', 'lie': 'Green', 'strat_type': 'Putt',
                'real_effet': 'N/A', 'strat_effet': 'N/A', 'dist_remain': 0
            })
            
        else:
            club = np.random.choice(["Driver", "Fer 7", "PW", "55°"])
            
            if mode == "Practice": shot_type = "Practice"
            elif mode == "Combine Test": shot_type = "Test Précision"
            else:
                if club == "Driver": shot_type = "Départ (Tee Shot)"
                elif club == "55°": shot_type = np.random.choice(["Approche (<50m)", "Sortie de Bunker"], p=[0.7, 0.3])
                else: shot_type = "Attaque de Green"
            
            lie = "Tee" if club == "Driver" else "Fairway"
            if shot_type == "Sortie de Bunker": lie = "Bunker"
            
            ampli = "Plein" if shot_type != "Approche (<50m)" else "1/2"
            dist_target = DIST_REF[club]
            if ampli == "1/2": dist_target *= 0.5
            
            # Dispersion s'améliore avec le temps (Progress)
            std_dev = (15 - (5 * progress)) if mode == "Parcours" else (8 - (3 * progress))
            dist_real = np.random.normal(dist_target, std_dev)
            
            # Score Latéral s'améliore
            base_lat = 2.5 if mode == "Parcours" else 1.5
            lat_score = min(5, int(abs(np.random.normal(0, base_lat - (0.5 * progress)))))
            direction = "Centre" if lat_score == 0 else np.random.choice(["Gauche", "Droite"])
            
            # Proximité (Short Game) - Nouveau V21
            dist_remain = 0
            if club in ["PW", "50°", "55°", "60°"] and dist_target < 100:
                # Plus le coup est bon, plus dist_remain est petit
                dist_remain = abs(dist_target - dist_real) + (lat_score * 2) + np.random.random()
                dist_remain = round(dist_remain, 1)

            delta = dist_real - dist_target
            err_L = "Court" if delta < -5 else ("Long" if delta > 5 else "Bonne Longueur")
            
            new_data.append({
                'date': current_date, 'mode': mode, 'club': club, 'par_trou': par_trou,
                'strat_dist': int(dist_target), 'strat_type': shot_type, 'amplitude': ampli,
                'distance': round(dist_real, 1), 'lie': lie, 'direction': direction, 
                'score_lateral': lat_score, 'delta_dist': delta, 'err_longueur': err_L,
                'type_coup': 'Jeu Long', 'resultat_putt': "N/A",
                'strat_effet': "Tout droit", 'real_effet': "Tout droit",
                'dist_remain': dist_remain # Nouveau V21
            })
            
    st.session_state['coups'].extend(new_data)
    st.sidebar.success("Données V21 (avec Historique) générées !")

# 3. EXPORT CSV
if st.session_state['coups']:
    df_ex = pd.DataFrame(st.session_state['coups'])
    st.sidebar.download_button("📥 Sauvegarder CSV", convert_df(df_ex), "golf_v21.csv", "text/csv")
    
if st.sidebar.button("🗑️ Reset Tout"): st.session_state['coups'] = []

# --- INTERFACE ---
st.title("🏌️‍♂️ GolfShot 21.0 : Pro Performance")

tab_saisie, tab_evol, tab_short, tab_dna, tab_sac = st.tabs([
    "📝 Saisie & Score", 
    "📈 Progression (Temps)",
    "🏆 Short Game & Combine",
    "🧬 Analyse Club", 
    "🎒 Bag Mapping"
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
    if target > 0: ax.scatter([0], [target], c='green', marker='*', s=150, label='Cible')
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

# --- ONGLET 1 : SAISIE (AVEC MODES ET SCORE) ---
with tab_saisie:
    # NOUVEAU : AJOUT DU MODE "COMBINE TEST"
    col_m, col_c = st.columns(2)
    with col_m: mode = st.radio("Mode", ["Parcours ⛳", "Practice 🚜", "Combine Test 🏆"], horizontal=True)
    with col_c: club = st.selectbox("Club en main", CLUBS_ORDER)
    
    if "Parcours" in mode:
        par_trou = st.selectbox("Par du trou", [3, 4, 5])
    else:
        par_trou = 0

    st.markdown("---")
    
    # --- SAISIE COUP STANDARD ---
    col_int, col_real = st.columns(2)
    with col_int:
        st.subheader("1️⃣ STRATÉGIE")
        if club == "Putter":
            obj_dist = st.number_input("Dist. Cible (m)", 0.0, 30.0, 3.0, 0.1)
            strat_effet = "N/A"
        else:
            if mode == "Practice": shot_type = "Practice"
            elif mode == "Combine Test": shot_type = "Test Précision"
            else: shot_type = st.selectbox("Type de Coup", SHOT_TYPES)
            
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
            dist_remain = 0
        else:
            dist_real = st.number_input("Distance Réelle (m)", 0, 350, int(obj_dist) if obj_dist > 0 else 200)
            if not ask_target_dist: obj_dist = dist_real
            lie = st.selectbox("Situation (Lie)", ["Tee", "Fairway", "Rough", "Bunker"])
            real_effet = st.selectbox("Effet Réalisé", ["Tout droit", "Fade", "Draw", "Push (Droite)", "Pull (Gauche)", "Hook", "Slice", "Top", "Gratte"])
            
            c_dir, c_sco = st.columns(2)
            with c_dir: direction = st.radio("Direction", ["Gauche", "Centre", "Droite"], horizontal=True)
            with c_sco: score_lat = st.number_input("Écart (0-5)", 0, 5, 0) if direction != "Centre" else 0
            
            # NOUVEAU V21 : PROXIMITÉ AU DRAPEAU (POUR LES APPROCHES)
            dist_remain = 0
            if shot_type in ["Approche (<50m)", "Sortie de Bunker", "Test Précision"] or club in ["50°", "55°", "60°", "PW"]:
                st.markdown("#### 📏 Proximité")
                dist_remain = st.number_input("Distance restante au trou (m)", 0.0, 50.0, 0.0)
            
            res_putt = "N/A"

    if st.button("💾 Enregistrer Coup", type="primary", use_container_width=True):
        delta = dist_real - obj_dist
        err_L = "Court" if delta < -5 else ("Long" if delta > 5 else "Bonne Longueur")
        
        # Calcul Points Combine Test (Simplifié)
        points_test = 0
        if mode == "Combine Test":
            if score_lat == 0: points_test += 50
            elif score_lat == 1: points_test += 30
            elif score_lat == 2: points_test += 10
            
            if abs(delta) < 5: points_test += 50
            elif abs(delta) < 10: points_test += 25
            
            st.toast(f"🏆 Score du coup : {points_test}/100")

        st.session_state['coups'].append({
            'date': str(datetime.date.today()), 'mode': mode.split()[0], 'club': club, 'par_trou': par_trou,
            'strat_dist': obj_dist, 'strat_type': shot_type, 'amplitude': amplitude, 'strat_effet': strat_effet,
            'distance': dist_real, 'lie': lie, 'resultat_putt': res_putt,
            'direction': direction, 'score_lateral': score_lat, 'real_effet': real_effet,
            'delta_dist': delta, 'err_longueur': err_L, 'dist_remain': dist_remain, 'points_test': points_test,
            'type_coup': 'Putt' if club == "Putter" else 'Jeu Long'
        })
        st.success("Sauvegardé !")

    # NOUVEAU V21 : SAISIE FIN DE PARTIE (STROKES GAINED SIMPLIFIÉ)
    if "Parcours" in mode:
        with st.expander("📝 Fin de Partie (Score & Stats)"):
            c_s1, c_s2, c_s3 = st.columns(3)
            with c_s1: score_final = st.number_input("Score Final (+/-)", -20, 150, 0, help="Par rapport au Par (ex: +12)")
            with c_s2: nb_putts = st.number_input("Total Putts", 18, 50, 32)
            with c_s3: penaltys = st.number_input("Pénalités", 0, 20, 0)
            if st.button("Enregistrer Partie"):
                st.session_state['parties'].append({
                    'date': str(datetime.date.today()), 'score': score_final, 'putts': nb_putts, 'penalties': penaltys
                })
                st.success("Partie archivée !")

# --- DATA LOAD ---
if st.session_state['coups']:
    df = pd.DataFrame(st.session_state['coups'])
    df_long = df[df['type_coup'] == 'Jeu Long']
    df_putt = df[df['type_coup'] == 'Putt']
else:
    df_long = pd.DataFrame()
    df_putt = pd.DataFrame()

# --- ONGLET 2 : PROGRESSION TEMPORELLE (NOUVEAU V21) ---
with tab_evol:
    if not df_long.empty:
        st.header("📈 Évolution de la Performance")
        
        # Préparation des dates
        df_long['date_dt'] = pd.to_datetime(df_long['date'])
        df_long = df_long.sort_values('date_dt')
        
        sel_club_evo = st.selectbox("Club à suivre", df_long['club'].unique())
        df_evo = df_long[df_long['club'] == sel_club_evo]
        
        if len(df_evo) > 5:
            col_e1, col_e2 = st.columns(2)
            
            with col_e1:
                st.subheader("Puissance (Distance)")
                # Moyenne Glissante
                df_evo['Moyenne Mobile'] = df_evo['distance'].rolling(window=5).mean()
                
                fig_ev1, ax_ev1 = plt.subplots(figsize=(6, 4))
                ax_ev1.plot(df_evo['date_dt'], df_evo['distance'], 'o', alpha=0.3, color='gray')
                ax_ev1.plot(df_evo['date_dt'], df_evo['Moyenne Mobile'], color='blue', linewidth=3, label='Tendance')
                ax_ev1.set_title(f"Distance Moyenne : {sel_club_evo}")
                ax_ev1.grid(True, alpha=0.3)
                st.pyplot(fig_ev1)
                
            with col_e2:
                st.subheader("Précision (Score Latéral)")
                # Score Latéral (plus bas = mieux)
                df_evo['Precision Mobile'] = df_evo['score_lateral'].rolling(window=5).mean()
                
                fig_ev2, ax_ev2 = plt.subplots(figsize=(6, 4))
                ax_ev2.plot(df_evo['date_dt'], df_evo['score_lateral'], 'o', alpha=0.3, color='red')
                ax_ev2.plot(df_evo['date_dt'], df_evo['Precision Mobile'], color='green', linewidth=3, label='Tendance')
                ax_ev2.set_title(f"Écarts Latéraux (0=Parfait)")
                ax_ev2.set_ylim(0, 5)
                ax_ev2.grid(True, alpha=0.3)
                st.pyplot(fig_ev2)
                st.caption("Si la courbe verte descend, votre précision s'améliore.")
        else:
            st.warning("Pas assez de données temporelles pour tracer une courbe (min 5 coups).")
            
        # Suivi du Score (si parties enregistrées)
        if st.session_state['parties']:
            st.subheader("Évolution du Score & Handicap")
            df_part = pd.DataFrame(st.session_state['parties'])
            df_part['date_dt'] = pd.to_datetime(df_part['date'])
            st.line_chart(df_part.set_index('date_dt')[['score', 'putts']])
    else:
        st.info("Générez des données pour voir l'évolution.")

# --- ONGLET 3 : SHORT GAME & COMBINE (NOUVEAU V21) ---
with tab_short:
    c_short, c_combine = st.columns(2)
    
    with c_short:
        st.header("🏆 Short Game (Scrambling)")
        if 'dist_remain' in df_long.columns:
            # On filtre les coups d'approche
            df_sg = df_long[(df_long['dist_remain'] > 0) & (df_long['dist_remain'] < 50)]
            
            if not df_sg.empty:
                avg_prox = df_sg['dist_remain'].mean()
                best_prox = df_sg['dist_remain'].min()
                
                st.metric("Proximité Moyenne au drapeau", f"{avg_prox:.2f} m")
                st.metric("Meilleure Approche", f"{best_prox:.2f} m")
                
                st.write("**Répartition par Club**")
                prox_by_club = df_sg.groupby('club')['dist_remain'].mean().sort_values()
                st.bar_chart(prox_by_club)
                st.caption("Plus la barre est basse, plus vous mettez la balle près du trou.")
            else:
                st.info("Aucune donnée de proximité (Saisissez 'Distance restante' lors des approches).")
                
    with c_combine:
        st.header("🥇 Combine Test")
        df_test = df_long[df_long['mode'] == 'Combine']
        
        if 'points_test' in df_long.columns:
            # On prend les coups qui ont des points > 0
            df_points = df_long[df_long['points_test'] > 0]
            if not df_points.empty:
                avg_score = df_points['points_test'].mean()
                last_score = df_points.iloc[-1]['points_test']
                
                st.metric("Score Moyen au Test", f"{avg_score:.0f} / 100")
                st.metric("Dernier coup noté", f"{last_score:.0f} / 100")
                
                st.write("**Derniers Tests**")
                st.dataframe(df_points[['date', 'club', 'distance', 'points_test']].tail(10))
            else:
                st.info("Lancez le mode 'Combine Test' et enregistrez des coups pour voir votre score.")

    # STROKES GAINED SIMPLIFIÉ
    st.markdown("---")
    st.header("📊 Bilan Partie (Strokes Gained Simplifié)")
    if st.session_state['parties']:
        last_round = st.session_state['parties'][-1]
        
        col_sg1, col_sg2, col_sg3 = st.columns(3)
        col_sg1.metric("Dernier Score", f"{last_round['score']}")
        
        # Logique simplifiée SG
        # Base amateur : 36 putts. Base pro : 29 putts.
        sg_putt = 34 - last_round['putts'] # Comparaison vs Scratch 34 putts
        col_sg2.metric("SG Putting", f"{sg_putt:+.1f}", help="Positif = Vous avez gagné des points au putting")
        
        # Perte balles
        sg_pen = -(last_round['penalties'] * 1) # 1 coup perdu par pénalité approx
        col_sg3.metric("Pénalités (Coups perdus)", f"{sg_pen}", help="Impact direct sur le score")
        
    else:
        st.info("Enregistrez une fin de partie dans l'onglet Saisie.")


# --- ONGLET 4 : ANALYSE CLUB (V20) ---
with tab_dna:
    if not df_long.empty:
        st.header("🧬 Club DNA")
        sel_club = st.selectbox("Choisir Club pour Analyse", df_long['club'].unique())
        df_c = df_long[df_long['club'] == sel_club]
        
        col_prac, col_parc = st.columns(2)
        df_practice = df_c[df_c['mode'] == 'Practice']
        df_parcours = df_c[df_c['mode'] == 'Parcours']
        
        with col_prac:
            fig1, ax1 = plt.subplots(figsize=(5, 5))
            plot_dispersion_ellipse(ax1, df_practice, "Practice (Labo)", "blue")
            st.pyplot(fig1)
            if not df_practice.empty:
                st.metric("Score Latéral Moyen", f"{df_practice['score_lateral'].mean():.1f} / 5")
                st.metric("Dispersion Profondeur", f"± {df_practice['distance'].std():.1f}m")

        with col_parc:
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            plot_dispersion_ellipse(ax2, df_parcours, "Parcours (Réalité)", "red")
            st.pyplot(fig2)
            if not df_parcours.empty:
                st.metric("Score Latéral Moyen", f"{df_parcours['score_lateral'].mean():.1f} / 5")
                st.metric("Dispersion Profondeur", f"± {df_parcours['distance'].std():.1f}m")
        
        st.markdown("---")
        st.subheader("🎨 Maîtrise des Effets")
        df_effets = df_c[df_c['strat_effet'].isin(["Fade", "Draw", "Tout droit", "Balle Basse"])]
        if not df_effets.empty:
            df_effets['Reussite'] = df_effets.apply(lambda x: 1 if x['strat_effet'] in x['real_effet'] else 0, axis=1)
            summary_effets = df_effets.groupby('strat_effet').agg(Tentatives=('strat_effet', 'count'), Reussites=('Reussite', 'sum'), Taux=('Reussite', 'mean'))
            summary_effets['Taux'] = (summary_effets['Taux'] * 100).round(1)
            st.dataframe(summary_effets.style.background_gradient(cmap="Greens", subset=['Taux']).format("{:.1f}%", subset=['Taux']), use_container_width=True)
        else:
            st.info("Aucun effet spécifique annoncé.")
    else:
        st.info("En attente de données.")

# --- ONGLET 5 : BAG MAPPING (V20) ---
with tab_sac:
    if not df_long.empty:
        st.header("🎒 Étalonnage du Sac")
        options_filtre = ["Tous les coups"] + [t for t in SHOT_TYPES]
        f_type = st.selectbox("Filtrer par situation", options_filtre, index=0)
        
        if f_type == "Tous les coups": df_sac = df_long.copy()
        else: df_sac = df_long[df_long['strat_type'] == f_type].copy()
        
        if not df_sac.empty:
            df_sac['club'] = pd.Categorical(df_sac['club'], categories=CLUBS_ORDER, ordered=True)
            df_sac = df_sac.sort_values('club')
            
            fig_bag, ax_bag = plt.subplots(figsize=(12, 5))
            sns.boxplot(x='club', y='distance', data=df_sac, ax=ax_bag, palette="viridis")
            ax_bag.grid(True, axis='y', alpha=0.3)
            st.pyplot(fig_bag)
            
            stats = df_sac.groupby('club', observed=True)['distance'].agg(['count', 'mean', 'max', 'std']).round(1)
            stats.columns = ['Coups', 'Moyenne', 'Max', 'Écart Type (±m)']
            st.dataframe(stats.style.background_gradient(cmap="Blues", subset=['Moyenne']).background_gradient(cmap="Reds_r", subset=['Écart Type (±m)']), use_container_width=True)
        else:
            st.warning("Pas de données.")
            
