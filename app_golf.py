import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="GolfShot 12.0 Club DNA", layout="wide")

# --- CSS POUR DASHBOARD ---
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;}
    h3 {border-bottom: 2px solid #4CAF50; padding-bottom: 10px;}
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

DIST_REF = {
    "Driver": 220, "Bois 5": 200, "Hybride": 180, "Fer 3": 170,
    "Fer 5": 160, "Fer 6": 150, "Fer 7": 140, "Fer 8": 130, "Fer 9": 120,
    "PW": 110, "50°": 100, "55°": 90, "60°": 80, "Putter": 3
}

# --- GÉNÉRATEUR V12 (ULTRA COMPLET) ---
st.sidebar.title("⚙️ Data Lab V12")

if st.sidebar.button("Générer Dataset 'Club DNA'"):
    new_data = []
    # On simule sur 60 jours pour avoir une évolution temporelle
    start_date = datetime.date.today() - datetime.timedelta(days=60)
    
    for i in range(400):
        # Date progressive
        current_date = start_date + datetime.timedelta(days=np.random.randint(0, 60))
        
        club = np.random.choice(["Driver", "Fer 7", "PW", "55°"])
        mode = np.random.choice(["Practice", "Parcours"], p=[0.4, 0.6])
        
        # Logique V10/V11
        if club in ["PW", "55°"]:
            ampli = np.random.choice(["Plein", "3/4", "1/2"], p=[0.5, 0.3, 0.2])
        else:
            ampli = "Plein"

        if mode == "Practice":
            lie = "Tapis/Tee"
        else:
            if club == "Driver": lie = "Tee"
            else: lie = np.random.choice(["Fairway", "Rough", "Bunker"], p=[0.6, 0.3, 0.1])

        # Calcul Distance
        dist_target = DIST_REF[club]
        if ampli == "3/4": dist_target *= 0.85
        if ampli == "1/2": dist_target *= 0.60
        
        # Simulation Réalité & Evolution (Le joueur progresse un peu)
        progress_factor = 1 + (i / 8000) # Légère augmentation dist
        penalty_lie = 0.85 if lie == "Rough" else (0.70 if lie == "Bunker" else 1.0)
        std_dev = 6 if mode == "Practice" else 14
        
        dist_real = np.random.normal(dist_target * penalty_lie * progress_factor, std_dev)
        
        # Erreurs
        delta = dist_real - dist_target
        if delta < -6: err_L = "Court"
        elif delta > 6: err_L = "Long"
        else: err_L = "Bonne Longueur"
        
        # Latéral
        lat_score = abs(np.random.normal(0, 2))
        lat_score = min(5, int(lat_score))
        direction = "Centre" if lat_score == 0 else np.random.choice(["Gauche", "Droite"])
        
        # Contact
        contact = np.random.choice(["Bon", "Gratte", "Top", "Pointe"], p=[0.6, 0.2, 0.1, 0.1])

        new_data.append({
            'date': str(current_date), 'mode': mode, 'club': club,
            'strat_dist': int(dist_target), 'amplitude': ampli, 
            'distance': round(dist_real, 1), 'lie': lie, 'contact': contact,
            'direction': direction, 'score_lateral': lat_score,
            'delta_dist': delta, 'err_longueur': err_L,
            'type_coup': 'Jeu Long'
        })
    st.session_state['coups'].extend(new_data)
    st.sidebar.success("400 Coups 'DNA' générés !")

if st.session_state['coups']:
    df_ex = pd.DataFrame(st.session_state['coups'])
    st.sidebar.download_button("📥 Export CSV", convert_df(df_ex), "golf_v12.csv", "text/csv")
    
if st.sidebar.button("🗑️ Reset"): st.session_state['coups'] = []

# --- INTERFACE ---
st.title("🏌️‍♂️ GolfShot 12.0 : Club DNA")

tab_saisie, tab_dna, tab_sac = st.tabs(["📝 Saisie Rapide", "🧬 Analyse Club 360°", "🎒 Mapping du Sac"])

# --- ONGLET 1 : SAISIE (STANDARD V11) ---
with tab_saisie:
    col_m, col_c = st.columns(2)
    with col_m: mode = st.radio("Mode", ["Parcours ⛳", "Practice 🚜"], horizontal=True)
    with col_c: club = st.selectbox("Club en main", CLUBS_ORDER)
    
    st.markdown("---")
    col_int, col_real = st.columns(2)

    with col_int:
        st.subheader("1️⃣ INTENTION")
        if club == "Putter":
            obj_dist = st.number_input("Dist. Cible", 0.0, 30.0, 3.0)
            amplitude = "Plein"
        else:
            obj_dist = st.number_input("Dist. Cible", 0, 350, DIST_REF.get(club, 100))
            amplitude = st.radio("Amplitude", ["Plein", "3/4", "1/2"], horizontal=True)

    with col_real:
        st.subheader("2️⃣ RÉALITÉ")
        if club == "Putter":
            res_putt = st.radio("Résultat", ["Dans le trou", "Raté"])
            dist_real = obj_dist if res_putt == "Dans le trou" else st.number_input("Dist. Réelle", 0.0, 30.0, obj_dist)
            lie = "Green"
            direction = "Centre"
            score_lat = 0
            contact = "Bon"
        else:
            dist_real = st.number_input("Distance Réelle (m)", 0, 350, int(obj_dist))
            lie = st.selectbox("Situation (Lie)", ["Tee", "Fairway", "Rough", "Bunker"])
            c_cont, c_dummy = st.columns(2)
            with c_cont: contact = st.selectbox("Contact", ["Bon", "Gratte", "Top", "Pointe", "Talon"])
            
            c_dir, c_sco = st.columns(2)
            with c_dir: direction = st.radio("Direction", ["Gauche", "Centre", "Droite"], horizontal=True)
            with c_sco: score_lat = st.number_input("Écart (0-5)", 0, 5, 0) if direction != "Centre" else 0

    st.markdown("---")
    if st.button("💾 Enregistrer", type="primary", use_container_width=True):
        delta = dist_real - obj_dist
        if delta < -5: err_L = "Court"
        elif delta > 5: err_L = "Long"
        else: err_L = "Bonne Longueur"

        st.session_state['coups'].append({
            'date': str(datetime.date.today()), 'mode': mode.split()[0], 'club': club,
            'strat_dist': obj_dist, 'amplitude': amplitude,
            'distance': dist_real, 'lie': lie, 'contact': contact,
            'direction': direction, 'score_lateral': score_lat,
            'delta_dist': delta, 'err_longueur': err_L,
            'type_coup': 'Putt' if club == "Putter" else 'Jeu Long'
        })
        st.success("Coup enregistré !")

# --- ONGLET 2 : CLUB DNA (LE COEUR DU SYSTÈME) ---
with tab_dna:
    if st.session_state['coups']:
        df = pd.DataFrame(st.session_state['coups'])
        df_long = df[df['type_coup'] == 'Jeu Long']
        
        # --- SÉLECTEUR GLOBAL ---
        st.markdown("### 🔎 Inspecteur de Club")
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            selected_club = st.selectbox("Sélectionnez le club à analyser", df_long['club'].unique())
        with col_sel2:
            selected_ampli = st.selectbox("Amplitude", ["Plein", "3/4", "1/2", "Tout"], index=0)

        # Filtrage des données
        df_club = df_long[df_long['club'] == selected_club]
        if selected_ampli != "Tout":
            df_club = df_club[df_club['amplitude'] == selected_ampli]
        
        if not df_club.empty:
            
            # --- LIGNE 1 : KPI CARTE D'IDENTITÉ ---
            st.markdown(f"### 🆔 Carte d'Identité : {selected_club} ({selected_ampli})")
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            
            avg_dist = df_club['distance'].mean()
            max_dist = df_club['distance'].max()
            std_dev = df_club['distance'].std() # Ecart Type
            contact_rate = len(df_club[df_club['contact']=='Bon']) / len(df_club) * 100
            
            kpi1.metric("Distance Moyenne", f"{avg_dist:.1f}m")
            kpi2.metric("Record (Max)", f"{max_dist:.1f}m")
            kpi3.metric("Dispersion (Sigma)", f"± {std_dev:.1f}m", help="Plus ce chiffre est bas, plus vous êtes régulier.")
            kpi4.metric("Qualité de Contact", f"{contact_rate:.0f}%", help="% de coups déclarés 'Bon Contact'")
            
            st.markdown("---")

            # --- LIGNE 2 : ANALYSE DISTANCE & LIE (GAUCHE) / DISPERSION (DROITE) ---
            col_L, col_R = st.columns([1, 1])
            
            with col_L:
                st.subheader("📏 Consistance & Lies")
                
                # Graph 1 : Histogramme de Distribution (KDE)
                fig_dist, ax_dist = plt.subplots(figsize=(6, 3))
                sns.histplot(df_club['distance'], kde=True, ax=ax_dist, color="teal")
                ax_dist.axvline(avg_dist, color='red', linestyle='--')
                ax_dist.set_title("Distribution des distances (Régularité)")
                st.pyplot(fig_dist)
                st.caption("Une courbe étroite et haute = Très régulier. Une courbe plate = Distances aléatoires.")
                
                # Graph 2 : Boxplot par Lie
                st.write("**Impact du Lie (Tee vs Fairway vs Rough)**")
                if len(df_club['lie'].unique()) > 1:
                    fig_lie, ax_lie = plt.subplots(figsize=(6, 3))
                    sns.boxplot(x='lie', y='distance', data=df_club, ax=ax_lie, palette="Set2")
                    st.pyplot(fig_lie)
                else:
                    st.info("Pas assez de lies différents pour comparer.")

            with col_R:
                st.subheader("🎯 Précision & Ratés")
                
                # Graph 3 : Dispersion XY (Ellipses)
                fig_scat, ax_scat = plt.subplots(figsize=(6, 4))
                # Coordonnées
                def get_x(row):
                    x = row['score_lateral'] * 5 
                    if row['direction'] == 'Gauche': return -x
                    if row['direction'] == 'Droite': return x
                    return 0 + np.random.normal(0,1)
                
                df_club['x_viz'] = df_club.apply(get_x, axis=1)
                
                sns.scatterplot(data=df_club, x='x_viz', y='distance', hue='mode', style='contact', s=100, ax=ax_scat)
                # Ellipse globale
                if len(df_club) > 3:
                    cov = np.cov(df_club['x_viz'], df_club['distance'])
                    lambda_, v = np.linalg.eig(cov)
                    lambda_ = np.sqrt(lambda_)
                    ell = Ellipse(xy=(df_club['x_viz'].mean(), df_club['distance'].mean()),
                                  width=lambda_[0]*4, height=lambda_[1]*4,
                                  angle=np.rad2deg(np.arccos(v[0, 0])), 
                                  edgecolor='green', facecolor='none', linestyle='--', label='Zone 95%')
                    ax_scat.add_artist(ell)
                
                ax_scat.set_title("Dispersion Latérale vs Distance")
                st.pyplot(fig_scat)
                
                # Graph 4 : Heatmap Simplifiée
                st.write("**Zone de Ratés (Heatmap)**")
                heat_data = df_club.groupby(['err_longueur', 'direction']).size().unstack(fill_value=0)
                y_ord = ["Long", "Bonne Longueur", "Court"]
                x_ord = ["Gauche", "Centre", "Droite"]
                heat_data = heat_data.reindex(index=y_ord, columns=x_ord, fill_value=0)
                fig_h, ax_h = plt.subplots(figsize=(5, 3))
                sns.heatmap(heat_data, annot=True, fmt='d', cmap="Reds", cbar=False, ax=ax_h)
                st.pyplot(fig_h)

            st.markdown("---")

            # --- LIGNE 3 : ÉVOLUTION & TECHNIQUE ---
            col_evo, col_tech = st.columns([2, 1])
            
            with col_evo:
                st.subheader("📈 Évolution Temporelle")
                # Conversion date pour tri
                df_club['date_dt'] = pd.to_datetime(df_club['date'])
                df_sorted = df_club.sort_values('date_dt')
                
                # Moyenne glissante sur 5 coups
                df_sorted['Moyenne Mobile'] = df_sorted['distance'].rolling(window=5).mean()
                
                fig_line, ax_line = plt.subplots(figsize=(8, 3))
                ax_line.plot(df_sorted['date_dt'], df_sorted['distance'], 'o', alpha=0.3, label="Coup brut")
                ax_line.plot(df_sorted['date_dt'], df_sorted['Moyenne Mobile'], color='red', linewidth=2, label="Tendance")
                ax_line.set_title("Progression de la distance")
                ax_line.legend()
                st.pyplot(fig_line)
            
            with col_tech:
                st.subheader("⚙️ Qualité de Frappe")
                cont_counts = df_club['contact'].value_counts()
                fig_pie, ax_pie = plt.subplots(figsize=(3, 3))
                ax_pie.pie(cont_counts, labels=cont_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
                st.pyplot(fig_pie)
        
        else:
            st.warning(f"Pas de données pour {selected_club} en {selected_ampli}.")
    else:
        st.info("Importez ou générez des données.")

# --- ONGLET 3 : MAPPING DU SAC (BONUS) ---
with tab_sac:
    if st.session_state['coups']:
        st.header("🎒 Bag Mapping (Gapping)")
        st.markdown("Ce graphique montre l'étalonnage de tout votre sac pour repérer les trous de distance.")
        
        df = pd.DataFrame(st.session_state['coups'])
        df_full = df[(df['type_coup']=='Jeu Long') & (df['amplitude']=='Plein')]
        
        if not df_full.empty:
            # Tri des clubs
            df_full['club'] = pd.Categorical(df_full['club'], categories=CLUBS_ORDER, ordered=True)
            df_full = df_full.sort_values('club')
            
            fig_bag, ax_bag = plt.subplots(figsize=(12, 6))
            sns.boxplot(x='club', y='distance', data=df_full, ax=ax_bag, palette="viridis")
            sns.swarmplot(x='club', y='distance', data=df_full, color=".25", size=3, alpha=0.5, ax=ax_bag)
            
            ax_bag.grid(True, axis='y', linestyle='--', alpha=0.5)
            ax_bag.set_title("Étalonnage complet (Pleins Coups)")
            st.pyplot(fig_bag)
            
            st.info("💡 Astuce : Si les boîtes de deux clubs se chevauchent trop (ex: Fer 8 et Fer 7), c'est que vous n'avez pas besoin des deux clubs, ou que votre technique doit être ajustée.")
    else:
        st.info("En attente de données...")
