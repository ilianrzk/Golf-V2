import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse, Circle
import datetime
import sqlite3
from fpdf import FPDF
import io

# --- CONFIGURATION ---
st.set_page_config(page_title="GolfShot 33.0 Pro Data", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;}
    h4 {color: #1565C0;}
    .stButton>button {width: 100%;}
    .caddie-box {
        border: 2px solid #2E7D32; padding: 15px; border-radius: 10px; 
        background-color: #E8F5E9; color: #1B5E20 !important; 
        text-align: center; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. GESTION BASE DE DONNÉES (SQLITE) ---
def init_db():
    conn = sqlite3.connect('golf_database.db', check_same_thread=False)
    c = conn.cursor()
    # Table Coups
    c.execute('''CREATE TABLE IF NOT EXISTS coups
                 (date TEXT, mode TEXT, club TEXT, strat_dist REAL, distance REAL, 
                  score_lateral REAL, direction TEXT, type_coup TEXT, resultat_putt TEXT,
                  delta_dist REAL, points_test REAL, err_longueur TEXT, lie TEXT,
                  strat_effet TEXT, real_effet TEXT, amplitude TEXT, contact TEXT,
                  dist_remain REAL, strat_type TEXT, par_trou INTEGER)''')
    # Table Parties
    c.execute('''CREATE TABLE IF NOT EXISTS parties
                 (date TEXT, score INTEGER, putts INTEGER)''')
    conn.commit()
    return conn

conn = init_db()

def add_coup_to_db(data):
    c = conn.cursor()
    c.execute('''INSERT INTO coups VALUES 
                 (:date, :mode, :club, :strat_dist, :distance, :score_lateral, :direction,
                  :type_coup, :resultat_putt, :delta_dist, :points_test, :err_longueur,
                  :lie, :strat_effet, :real_effet, :amplitude, :contact, :dist_remain,
                  :strat_type, :par_trou)''', data)
    conn.commit()

def load_coups_from_db():
    return pd.read_sql("SELECT * FROM coups", conn)

def add_partie_to_db(data):
    c = conn.cursor()
    c.execute("INSERT INTO parties VALUES (:date, :score, :putts)", data)
    conn.commit()

def load_parties_from_db():
    return pd.read_sql("SELECT * FROM parties", conn)

# --- 2. GÉNÉRATEUR PDF ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'GolfShot - Rapport de Performance', 0, 1, 'C')
        self.ln(5)

def create_pdf_report(df_coups, df_parties):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)
    
    # Titre
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f"Date du rapport : {datetime.date.today()}", 0, 1)
    pdf.ln(5)
    
    # 1. Résumé Global
    pdf.set_fill_color(200, 220, 255)
    pdf.cell(0, 10, "1. STATISTIQUES GLOBALES", 1, 1, 'L', 1)
    pdf.set_font('Arial', '', 11)
    
    if not df_coups.empty:
        nb_coups = len(df_coups)
        nb_practice = len(df_coups[df_coups['mode'] == 'Practice'])
        pdf.cell(0, 8, f"Total coups analysés : {nb_coups}", 0, 1)
        pdf.cell(0, 8, f"Dont Practice : {nb_practice}", 0, 1)
        
        # Stats Driver
        df_drive = df_coups[df_coups['club'] == 'Driver']
        if not df_drive.empty:
            avg_drive = df_drive['distance'].mean()
            pdf.cell(0, 8, f"Moyenne Driver : {avg_drive:.1f}m", 0, 1)
    
    if not df_parties.empty:
        last_score = df_parties.iloc[-1]['score']
        avg_score = df_parties['score'].mean()
        pdf.cell(0, 8, f"Dernier Score : {last_score}", 0, 1)
        pdf.cell(0, 8, f"Score Moyen : {avg_score:.1f}", 0, 1)
    
    pdf.ln(5)
    
    # 2. Points à travailler
    pdf.set_font('Arial', 'B', 12)
    pdf.set_fill_color(255, 200, 200)
    pdf.cell(0, 10, "2. FOCUS TECHNIQUE (A TRAVAILLER)", 1, 1, 'L', 1)
    pdf.set_font('Arial', '', 11)
    
    if not df_coups.empty:
        # Analyse des ratés
        misses = df_coups[df_coups['err_longueur'] != 'Bonne Longueur']
        if not misses.empty:
            top_miss = misses['err_longueur'].mode()[0]
            pdf.cell(0, 8, f"- Tendance Longueur : {top_miss}", 0, 1)
        
        # Analyse Putting
        df_putt = df_coups[df_coups['type_coup'] == 'Putt']
        if not df_putt.empty:
            miss_putts = df_putt[df_putt['resultat_putt'] != "Dans le trou"]
            if not miss_putts.empty:
                top_putt_miss = miss_putts['resultat_putt'].mode()[0]
                pdf.cell(0, 8, f"- Putting : Attention aux {top_putt_miss}", 0, 1)

    return pdf.output(dest='S').encode('latin-1', 'replace')

# --- CONSTANTES ---
CLUBS_ORDER = ["Driver", "Bois 5", "Hybride", "Fer 3", "Fer 5", "Fer 6", "Fer 7", "Fer 8", "Fer 9", "PW", "50°", "55°", "60°", "Putter"]
SHOT_TYPES = ["Départ (Tee Shot)", "Attaque de Green", "Lay-up / Sécurité", "Approche (<50m)", "Sortie de Bunker", "Recovery"]
PUTT_RESULTS = ["Dans le trou", "Raté - Court", "Raté - Long", "Raté - Gauche", "Raté - Droite", "Raté - Court/Gauche", "Raté - Court/Droite", "Raté - Long/Gauche", "Raté - Long/Droite"]
DIST_REF = {"Driver": 220, "Bois 5": 200, "Hybride": 180, "Fer 3": 170, "Fer 5": 160, "Fer 6": 150, "Fer 7": 140, "Fer 8": 130, "Fer 9": 120, "PW": 110, "50°": 100, "55°": 90, "60°": 80, "Putter": 3}

# --- ETAT SESSION (CHARGEMENT DEPUIS DB) ---
if 'combine_state' not in st.session_state: st.session_state['combine_state'] = None
if 'current_card' not in st.session_state:
    st.session_state['current_card'] = pd.DataFrame({'Trou': range(1, 19), 'Par': [4]*18, 'Score': [0]*18, 'Putts': [0]*18})
if 'current_hole' not in st.session_state: st.session_state['current_hole'] = 1
if 'shots_on_current_hole' not in st.session_state: st.session_state['shots_on_current_hole'] = 0
if 'putts_on_current_hole' not in st.session_state: st.session_state['putts_on_current_hole'] = 0

# Chargement initial des données depuis la DB
st.session_state['coups'] = load_coups_from_db().to_dict('records')
st.session_state['parties'] = load_parties_from_db().to_dict('records')

# ==================================================
# BARRE LATÉRALE
# ==================================================
st.sidebar.title("⚙️ Data Lab")

# 1. FILTRE TEMPOREL
st.sidebar.header("📅 Filtre Temporel")
default_start = datetime.date(datetime.date.today().year, 1, 1)
filter_start = st.sidebar.date_input("Du", default_start)
filter_end = st.sidebar.date_input("Au", datetime.date.today())

# DataFrame Filtré pour affichage
df_analysis = pd.DataFrame()
if st.session_state['coups']:
    df_raw = pd.DataFrame(st.session_state['coups'])
    df_raw['date_dt'] = pd.to_datetime(df_raw['date']).dt.date
    df_analysis = df_raw[(df_raw['date_dt'] >= filter_start) & (df_raw['date_dt'] <= filter_end)]
    st.sidebar.caption(f"{len(df_analysis)} coups sur la période.")

# 2. EXPORT RAPPORT PDF (NOUVEAU)
st.sidebar.markdown("---")
st.sidebar.header("📄 Rapports")
if st.sidebar.button("Générer Rapport Coach (PDF)"):
    if not df_analysis.empty:
        pdf_bytes = create_pdf_report(df_analysis, pd.DataFrame(st.session_state['parties']))
        st.sidebar.download_button(
            label="📥 Télécharger le PDF",
            data=pdf_bytes,
            file_name="golf_report.pdf",
            mime="application/pdf"
        )
    else:
        st.sidebar.error("Pas de données sur cette période.")

st.sidebar.markdown("---")

# 3. SMART CADDIE
st.sidebar.header("🤖 Smart Caddie")
with st.sidebar.expander("Assistant", expanded=True):
    cad_dist = st.number_input("Distance (m)", 50, 250, 135, step=5)
    cad_lie = st.selectbox("Lie", ["Tee", "Fairway", "Rough", "Bunker"])
    
    if not df_analysis.empty:
        df_caddie = df_analysis[df_analysis['type_coup'] == 'Jeu Long']
        df_lie = df_caddie[df_caddie['lie'] == cad_lie]
        if len(df_lie) < 5: df_lie = df_caddie 
        if not df_lie.empty:    
            stats = df_lie.groupby('club')['distance'].mean().reset_index()
            stats['diff'] = abs(stats['distance'] - cad_dist)
            best_match = stats.nsmallest(1, 'diff')
            if not best_match.empty:
                rec = best_match.iloc[0]
                st.markdown(f"<div class='caddie-box'>💡 {rec['club']}<br><small>Moy: {rec['distance']:.1f}m</small></div>", unsafe_allow_html=True)
            else: st.warning("?")
        else: st.warning("Manque de données")
    else: st.warning("Données requises")

st.sidebar.markdown("---")
# BOUTON GENERATEUR (Pour peupler la DB au début)
if st.sidebar.button("Injecter Données Démo (V33)"):
    # Génération simple pour test
    dates = [datetime.date.today() - datetime.timedelta(days=x) for x in range(30)]
    for _ in range(50):
        club = "Fer 7"
        entry = {
            'date': str(np.random.choice(dates)), 'mode': 'Practice', 'club': club,
            'strat_dist': 140, 'distance': 140 + np.random.randint(-10,10), 'score_lateral': 1, 
            'direction': 'Centre', 'type_coup': 'Jeu Long', 'resultat_putt': 'N/A', 
            'delta_dist': 0, 'points_test': 0, 'err_longueur': 'Ok', 'lie': 'Fairway',
            'strat_effet': 'N/A', 'real_effet': 'N/A', 'amplitude': 'Plein', 'contact': 'Bon',
            'dist_remain': 0, 'strat_type': 'Practice', 'par_trou': 0
        }
        add_coup_to_db(entry)
    st.rerun()

# --- INTERFACE ---
st.title("🏌️‍♂️ GolfShot 33.0 : Pro Data & Reporting")

tab_parcours, tab_practice, tab_combine, tab_dna, tab_sac, tab_putt = st.tabs([
    "⛳ Parcours", "🚜 Practice", "🏆 Combine", "🧬 Club DNA", "🎒 Mapping", "🟢 Putting"
])

# HELPER GRAPH
def plot_dispersion_analysis(ax, data, title, color):
    if data.empty or len(data) < 2: return
    def get_x(row):
        x = row['score_lateral'] * 5 
        if row['direction'] == 'Gauche': return -x
        if row['direction'] == 'Droite': return x
        return np.random.normal(0, 1)
    data = data.copy()
    data['x_viz'] = data.apply(get_x, axis=1)
    ax.scatter(data['x_viz'], data['distance'], c=color, alpha=0.6, s=60, edgecolors='white')
    target = data['strat_dist'].mean()
    if target > 0: ax.scatter([0], [target], c='green', marker='*', s=150, label='Cible')
    if len(data) > 3:
        try:
            cov = np.cov(data['x_viz'], data['distance'])
            lambda_, v = np.linalg.eig(cov)
            ell = Ellipse(xy=(data['x_viz'].mean(), data['distance'].mean()), width=np.sqrt(lambda_[0])*4, height=np.sqrt(lambda_[1])*4, angle=np.rad2deg(np.arccos(v[0, 0])), edgecolor=color, facecolor=color, alpha=0.15, linewidth=2)
            ax.add_artist(ell)
        except: pass
    ax.set_title(title)
    ax.set_xlabel("Gauche <---> Droite"); ax.set_ylabel("Distance"); ax.grid(True, alpha=0.3)

# ==================================================
# ONGLET 1 : PARCOURS (SAVE TO DB)
# ==================================================
with tab_parcours:
    c1, c2 = st.columns(2)
    with c1:
        st.header(f"Trou {st.session_state['current_hole']}")
        club = st.selectbox("Club", CLUBS_ORDER, key="p_c")
        
        if club == "Putter":
            obj_dist = st.number_input("Dist Cible", 0.0, 30.0, 2.0, 0.1)
            dist_real = st.number_input("Dist Réelle", 0.0, 30.0, obj_dist, 0.1)
            res_putt = st.selectbox("Résultat", PUTT_RESULTS)
            typ = "Putt"
            dir_ = "Centre"; lat = 0
        else:
            obj_dist = st.number_input("Cible", 0, 300, 150)
            dist_real = st.number_input("Réel", 0, 300, 150)
            dir_ = st.radio("Dir", ["Gauche", "Centre", "Droite"], horizontal=True)
            lat = st.slider("Ecart", 0, 5, 0)
            res_putt = "N/A"
            typ = "Jeu Long"

        if st.button("Valider Coup"):
            coup_data = {
                'date': str(datetime.date.today()), 'mode': 'Parcours', 'club': club,
                'strat_dist': obj_dist, 'distance': dist_real, 'direction': dir_,
                'score_lateral': lat, 'real_effet': 'N/A', 'strat_effet': 'N/A',
                'type_coup': typ, 'delta_dist': dist_real-obj_dist, 'resultat_putt': res_putt,
                'err_longueur': 'Ok', 'strat_type': 'Parcours', 'par_trou': 4,
                'points_test': 0, 'lie': 'Fairway', 'amplitude': 'Plein', 'contact': 'Bon', 'dist_remain': 0
            }
            add_coup_to_db(coup_data) # SAVE DB
            st.session_state['coups'].append(coup_data) # Update session for immediate view
            st.success("Sauvegardé en Base de Données !")

# ==================================================
# ONGLET 2 : PRACTICE (SAVE TO DB)
# ==================================================
with tab_practice:
    st.header("Practice")
    if st.button("Enregistrer Test Practice"):
        # Exemple simplifié pour practice
        entry = {
            'date': str(datetime.date.today()), 'mode': 'Practice', 'club': 'Fer 7',
            'strat_dist': 150, 'distance': 150, 'direction': 'Centre', 'score_lateral': 0,
            'type_coup': 'Jeu Long', 'resultat_putt': 'N/A', 'delta_dist': 0,
            'points_test': 0, 'err_longueur': 'Ok', 'lie': 'Tapis', 'strat_type': 'Practice',
            'strat_effet': 'N/A', 'real_effet': 'N/A', 'amplitude': 'Plein', 'contact': 'Bon',
            'dist_remain': 0, 'par_trou': 0
        }
        add_coup_to_db(entry)
        st.session_state['coups'].append(entry)
        st.success("Coup Practice Sauvegardé !")

# ==================================================
# ANALYSES (UTILISENT df_analysis FILTRÉ)
# ==================================================
with tab_combine:
    st.header("🏆 Combine Analytics")
    if not df_analysis.empty:
        df_c = df_analysis[df_analysis['mode'] == 'Combine']
        if not df_c.empty:
            st.metric("Score Moyen", f"{df_c['points_test'].mean():.0f}/100")
        else: st.info("Pas de données Combine sur cette période.")

with tab_dna:
    st.header("🧬 Club DNA")
    if not df_analysis.empty:
        df_l = df_analysis[df_analysis['type_coup'] == 'Jeu Long']
        if not df_l.empty:
            sel = st.selectbox("Club", df_l['club'].unique())
            sub = df_l[df_l['club'] == sel]
            col1, col2 = st.columns(2)
            with col1: 
                fig, ax = plt.subplots()
                plot_dispersion_analysis(ax, sub[sub['mode']=='Practice'], "Practice", "blue")
                st.pyplot(fig)
            with col2:
                fig, ax = plt.subplots()
                plot_dispersion_analysis(ax, sub[sub['mode']=='Parcours'], "Parcours", "red")
                st.pyplot(fig)

with tab_sac:
    st.header("🎒 Mapping")
    if not df_analysis.empty:
        df_s = df_analysis[df_analysis['type_coup'] == 'Jeu Long']
        if not df_s.empty:
            df_s['club'] = pd.Categorical(df_s['club'], categories=CLUBS_ORDER, ordered=True)
            fig, ax = plt.subplots(figsize=(10,5))
            sns.boxplot(data=df_s, x='club', y='distance', ax=ax)
            st.pyplot(fig)
            stats = df_s.groupby('club')['distance'].agg(['count', 'mean', 'max', 'std']).round(1)
            st.dataframe(stats.style.background_gradient(cmap="Blues"), use_container_width=True)

with tab_putt:
    st.header("🟢 Putting")
    if not df_analysis.empty:
        df_p = df_analysis[df_analysis['type_coup'] == 'Putt']
        if not df_p.empty:
            df_p['Zone'] = pd.cut(df_p['strat_dist'], [0,2,5,10,30], labels=["0-2m","2-5m","5-10m","+10m"])
            piv = df_p.groupby('Zone', observed=False).apply(lambda x: (x['resultat_putt']=="Dans le trou").mean()*100)
            st.dataframe(piv.to_frame("%").style.background_gradient(cmap="RdYlGn"), use_container_width=True)
