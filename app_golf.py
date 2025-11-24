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
st.set_page_config(page_title="GolfShot 36.0 Complete DNA", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6; 
        padding: 10px; 
        border-radius: 5px; 
        margin-bottom: 10px;
    }
    h4 {color: #1565C0;}
    .stButton>button {width: 100%;}
    
    .caddie-box {
        border: 2px solid #2E7D32; 
        padding: 15px; 
        border-radius: 10px; 
        background-color: #E8F5E9; 
        color: #1B5E20 !important;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- FONCTION UTILITAIRE ---
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# --- 1. GESTION BASE DE DONNÉES ---
def init_db():
    conn = sqlite3.connect('golf_database.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS coups
                 (date TEXT, mode TEXT, club TEXT, strat_dist REAL, distance REAL, 
                  score_lateral REAL, direction TEXT, type_coup TEXT, resultat_putt TEXT,
                  delta_dist REAL, points_test REAL, err_longueur TEXT, lie TEXT,
                  strat_effet TEXT, real_effet TEXT, amplitude TEXT, contact TEXT,
                  dist_remain REAL, strat_type TEXT, par_trou INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS parties
                 (date TEXT, score INTEGER, putts INTEGER)''')
    conn.commit()
    return conn

conn = init_db()

def add_coup_to_db(data):
    keys = ['date', 'mode', 'club', 'strat_dist', 'distance', 'score_lateral', 'direction', 
            'type_coup', 'resultat_putt', 'delta_dist', 'points_test', 'err_longueur', 
            'lie', 'strat_effet', 'real_effet', 'amplitude', 'contact', 
            'dist_remain', 'strat_type', 'par_trou']
    for k in keys:
        if k not in data: data[k] = None
    c = conn.cursor()
    placeholders = ','.join([':' + k for k in keys])
    c.execute(f'''INSERT INTO coups VALUES ({placeholders})''', data)
    conn.commit()

def load_coups_from_db():
    try: return pd.read_sql("SELECT * FROM coups", conn)
    except: return pd.DataFrame()

def add_partie_to_db(data):
    c = conn.cursor()
    c.execute("INSERT INTO parties VALUES (:date, :score, :putts)", data)
    conn.commit()

def load_parties_from_db():
    try: return pd.read_sql("SELECT * FROM parties", conn)
    except: return pd.DataFrame()

# --- 2. GÉNÉRATEUR PDF PRO ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'GolfShot Analytics - Rapport Technique', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f"Genere le {datetime.date.today()}", 0, 1, 'C')
        self.line(10, 30, 200, 30)
        self.ln(10)

    def chapter_title(self, label):
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 10, label, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, body)
        self.ln()

def create_pro_pdf(df_coups, df_parties):
    pdf = PDF()
    pdf.add_page()
    
    pdf.chapter_title("1. RESUME DE PERFORMANCE")
    nb_total = len(df_coups)
    if not df_parties.empty:
        avg_score = df_parties['score'].mean()
        avg_putts = df_parties['putts'].mean()
        best_score = df_parties['score'].min()
        txt_score = f"Score Moyen: {avg_score:.1f} | Meilleur: {best_score} | Moy. Putts: {avg_putts:.1f}"
    else:
        txt_score = "Aucune partie complete enregistree."
    pdf.chapter_body(f"Volume de jeu analyse : {nb_total} coups.\n{txt_score}")
    
    pdf.chapter_title("2. TABLEAU D'ETALONNAGE (GAPPING)")
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(30, 8, 'Club', 1); pdf.cell(30, 8, 'Moyenne', 1); pdf.cell(30, 8, 'Max', 1); pdf.cell(30, 8, 'Dispersion', 1); pdf.cell(40, 8, 'Efficacite', 1); pdf.ln()
    
    pdf.set_font('Arial', '', 10)
    df_long = df_coups[df_coups['type_coup'] == 'Jeu Long']
    club_order_pdf = ["Driver", "Bois 5", "Hybride", "Fer 3", "Fer 5", "Fer 6", "Fer 7", "Fer 8", "Fer 9", "PW", "50°", "55°", "60°"]
    
    for club in club_order_pdf:
        stats = df_long[df_long['club'] == club]
        if not stats.empty and len(stats) > 1:
            avg = stats['distance'].mean()
            maxx = stats['distance'].max()
            std = stats['distance'].std()
            eff = (avg / maxx * 100) if maxx > 0 else 0
            pdf.cell(30, 8, str(club), 1); pdf.cell(30, 8, f"{avg:.1f}m", 1); pdf.cell(30, 8, f"{maxx:.1f}m", 1); pdf.cell(30, 8, f"+/- {std:.1f}", 1); pdf.cell(40, 8, f"{eff:.0f}%", 1); pdf.ln()
    
    pdf.ln(5)
    pdf.chapter_title("3. DIAGNOSTIC")
    pdf.set_font('Arial', '', 11)
    if not df_long.empty:
        miss_L = df_long[df_long['err_longueur'] != 'Bonne Longueur']
        if not miss_L.empty:
            try:
                top_L = miss_L['err_longueur'].mode()[0]
                pdf.cell(0, 8, f">> Tendance Longueur : {top_L}", 0, 1)
            except: pass
    return pdf.output(dest='S').encode('latin-1', 'replace')

# --- ETAT (CHARGEMENT DB) ---
if 'coups' not in st.session_state or not st.session_state['coups']:
    st.session_state['coups'] = load_coups_from_db().to_dict('records')
if 'parties' not in st.session_state or not st.session_state['parties']:
    st.session_state['parties'] = load_parties_from_db().to_dict('records')

if 'combine_state' not in st.session_state: st.session_state['combine_state'] = None

if 'current_card' not in st.session_state:
    st.session_state['current_card'] = pd.DataFrame({
        'Trou': range(1, 19), 'Par': [4]*18, 'Score': [0]*18, 'Putts': [0]*18
    })
if 'current_hole' not in st.session_state: st.session_state['current_hole'] = 1
if 'shots_on_current_hole' not in st.session_state: st.session_state['shots_on_current_hole'] = 0
if 'putts_on_current_hole' not in st.session_state: st.session_state['putts_on_current_hole'] = 0

CLUBS_ORDER = ["Driver", "Bois 5", "Hybride", "Fer 3", "Fer 5", "Fer 6", "Fer 7", "Fer 8", "Fer 9", "PW", "50°", "55°", "60°", "Putter"]
SHOT_TYPES = ["Départ (Tee Shot)", "Attaque de Green", "Lay-up / Sécurité", "Approche (<50m)", "Sortie de Bunker", "Recovery"]
PUTT_RESULTS = ["Dans le trou", "Raté - Court", "Raté - Long", "Raté - Gauche", "Raté - Droite", "Raté - Court/Gauche", "Raté - Court/Droite", "Raté - Long/Gauche", "Raté - Long/Droite"]
DIST_REF = {"Driver": 220, "Bois 5": 200, "Hybride": 180, "Fer 3": 170, "Fer 5": 160, "Fer 6": 150, "Fer 7": 140, "Fer 8": 130, "Fer 9": 120, "PW": 110, "50°": 100, "55°": 90, "60°": 80, "Putter": 3}

# ==================================================
# BARRE LATÉRALE
# ==================================================
st.sidebar.title("⚙️ Data Lab")

# 1. IMPORT
uploaded_file = st.sidebar.file_uploader("📂 Importer CSV", type="csv")
if uploaded_file is not None:
    try:
        df_loaded = pd.read_csv(uploaded_file)
        for _, row in df_loaded.iterrows():
            add_coup_to_db(row.to_dict())
        st.session_state['coups'] = load_coups_from_db().to_dict('records')
        st.sidebar.success("Importé !")
        st.rerun()
    except Exception as e: st.sidebar.error(f"Erreur : {e}")

st.sidebar.markdown("---")

# 2. FILTRE TEMPOREL
st.sidebar.header("📅 Filtre Temporel")
default_start = datetime.date(datetime.date.today().year, 1, 1)
filter_start = st.sidebar.date_input("Du", default_start)
filter_end = st.sidebar.date_input("Au", datetime.date.today())

df_analysis = pd.DataFrame()
if st.session_state['coups']:
    df_raw = pd.DataFrame(st.session_state['coups'])
    df_raw['date_dt'] = pd.to_datetime(df_raw['date']).dt.date
    df_analysis = df_raw[(df_raw['date_dt'] >= filter_start) & (df_raw['date_dt'] <= filter_end)]
    st.sidebar.caption(f"{len(df_analysis)} coups analysés.")

# 3. RAPPORT PDF
if st.sidebar.button("📄 Générer Rapport PDF"):
    if not df_analysis.empty:
        try:
            pdf_bytes = create_pro_pdf(df_analysis, pd.DataFrame(st.session_state['parties']))
            st.sidebar.download_button("📥 Télécharger PDF", pdf_bytes, "Rapport_Coach_Golf.pdf", "application/pdf")
        except Exception as e: st.sidebar.error(f"Erreur PDF : {e}")
    else: st.sidebar.error("Pas de données.")

st.sidebar.markdown("---")

# 4. SMART CADDIE
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
            best = stats.nsmallest(1, 'diff')
            if not best.empty:
                rec = best.iloc[0]
                st.markdown(f"<div class='caddie-box'>💡 {rec['club']}<br><small>Moy: {rec['distance']:.1f}m</small></div>", unsafe_allow_html=True)
            else: st.warning("?")
        else: st.warning("Données insuf.")
    else: st.warning("Données requises.")

st.sidebar.markdown("---")
if st.sidebar.button("Injecter Données Test V36"):
    new_data = []
    dates = [datetime.date.today() - datetime.timedelta(days=x) for x in range(60)]
    for _ in range(200):
        mode = np.random.choice(["Parcours", "Practice", "Combine"], p=[0.6, 0.3, 0.1])
        club = np.random.choice(["Driver", "Fer 7", "Putter"])
        entry = {
            'date': str(np.random.choice(dates)), 'mode': mode, 'club': club,
            'strat_dist': 0, 'distance': 0, 'score_lateral': 0, 'direction': 'Centre',
            'strat_type': 'Entraînement', 'resultat_putt': 'N/A', 'delta_dist': 0, 
            'points_test': 0, 'err_longueur': 'Ok', 'lie': 'Fairway',
            'strat_effet': 'N/A', 'real_effet': 'N/A', 'amplitude': 'Plein', 'contact': 'Bon',
            'dist_remain': 0, 'par_trou': 4
        }
        if club == "Putter":
            d = round(np.random.uniform(0.5, 15.0), 1)
            entry.update({'strat_dist': d, 'distance': d, 'resultat_putt': np.random.choice(PUTT_RESULTS), 'type_coup': 'Putt', 'lie': 'Green'})
        else:
            t = DIST_REF[club]
            r = np.random.normal(t, 10)
            entry.update({'strat_dist': t, 'distance': r, 'score_lateral': np.random.randint(0,3), 'type_coup': 'Jeu Long'})
        add_coup_to_db(entry)
    st.session_state['coups'] = load_coups_from_db().to_dict('records')
    st.sidebar.success("Données injectées !")
    st.rerun()

if st.session_state['coups']:
    df_ex = pd.DataFrame(st.session_state['coups'])
    st.sidebar.download_button("📥 Backup CSV", convert_df(df_ex), "golf_v36_backup.csv", "text/csv")

if st.sidebar.button("⚠️ Vider DB"):
    conn.execute("DELETE FROM coups"); conn.execute("DELETE FROM parties"); conn.commit()
    st.session_state['coups'] = []; st.session_state['parties'] = []; st.rerun()

# --- INTERFACE ---
st.title("🏌️‍♂️ GolfShot 36.0 : Complete DNA")

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
    ax.set_title(title); ax.set_xlabel("Gauche <---> Droite"); ax.set_ylabel("Distance"); ax.grid(True, alpha=0.3)

# ==================================================
# ONGLET 1 : PARCOURS
# ==================================================
with tab_parcours:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.header(f"Trou {st.session_state['current_hole']}")
        idx_hole = st.session_state['current_hole'] - 1
        current_par = st.session_state['current_card'].at[idx_hole, 'Par']
        st.info(f"Par {current_par} | Coups : {st.session_state['shots_on_current_hole']}")

        date_coup = st.date_input("Date", datetime.date.today())
        club = st.selectbox("Club", CLUBS_ORDER, key="p_c")
        st.markdown("---")
        c_int, c_real = st.columns(2)
        
        with c_int:
            st.subheader("Intention")
            if club == "Putter":
                obj_dist = st.number_input("Cible (m)", 0.0, 30.0, 1.5, 0.1, format="%.1f")
                strat_effet = "N/A"; shot_type = "Putt"
            else:
                shot_type = st.selectbox("Type", SHOT_TYPES, key="p_t")
                if shot_type == "Départ (Tee Shot)" and current_par > 3:
                    obj_dist = 0.0; st.caption("🚀 Max")
                else: obj_dist = st.number_input("Cible (m)", 0, 350, 150)
                strat_effet = st.selectbox("Effet Voulu", ["Tout droit", "Fade", "Draw", "Balle Basse"])

        with c_real:
            st.subheader("Réalité")
            if club == "Putter":
                dist_real = st.number_input("Réel (m)", 0.0, 30.0, obj_dist, 0.1, format="%.1f")
                res_putt = st.selectbox("Résultat", PUTT_RESULTS)
                dir_, lat, real_effet = "Centre", 0, "N/A"
            else:
                dist_real = st.number_input("Réel (m)", 0, 350, int(obj_dist) if obj_dist>0 else 200)
                real_effet = st.selectbox("Effet Réalisé", ["Tout droit", "Fade", "Draw", "Raté"])
                c_d1, c_d2 = st.columns(2)
                with c_d1: dir_ = st.radio("Axe", ["Gauche", "Centre", "Droite"], horizontal=True)
                with c_d2: lat = st.slider("Ecart", 0, 5, 0)
                res_putt = "N/A"
                lie = st.selectbox("Lie", ["Fairway", "Rough", "Tee", "Bunker"])

        if st.button("Valider Coup (+1 Score)"):
            delta = dist_real - obj_dist if obj_dist > 0 else 0
            data = {
                'date': str(date_coup), 'mode': 'Parcours', 'club': club,
                'strat_dist': obj_dist, 'distance': dist_real, 'direction': dir_,
                'score_lateral': lat, 'real_effet': real_effet, 'strat_effet': strat_effet,
                'type_coup': 'Putt' if club=='Putter' else 'Jeu Long', 'delta_dist': delta, 
                'resultat_putt': res_putt, 'err_longueur': 'Ok', 'strat_type': shot_type, 
                'par_trou': current_par, 'lie': lie if club != 'Putter' else 'Green',
                'points_test': 0, 'amplitude': 'Plein', 'contact': 'Bon', 'dist_remain': 0
            }
            add_coup_to_db(data)
            st.session_state['coups'].append(data)
            
            st.session_state['shots_on_current_hole'] += 1
            if club == "Putter": st.session_state['putts_on_current_hole'] += 1
            idx = st.session_state['current_hole'] - 1
            st.session_state['current_card'].at[idx, 'Score'] = st.session_state['shots_on_current_hole']
            st.session_state['current_card'].at[idx, 'Putts'] = st.session_state['putts_on_current_hole']
            
            if res_putt == "Dans le trou":
                st.balloons()
                st.session_state['shots_on_current_hole'] = 0
                st.session_state['putts_on_current_hole'] = 0
                if st.session_state['current_hole'] < 18: st.session_state['current_hole'] += 1
                st.rerun()
            else: st.success("Sauvegardé !")

        c_p, c_n = st.columns(2)
        if c_p.button("<< Préc"):
             if st.session_state['current_hole'] > 1:
                st.session_state['current_hole'] -= 1
                st.rerun()
        if c_n.button("Suiv >>"):
            st.session_state['shots_on_current_hole'] = 0
            st.session_state['putts_on_current_hole'] = 0
            if st.session_state['current_hole'] < 18:
                st.session_state['current_hole'] += 1
                st.rerun()

    with c2:
        st.header("📋 Carte")
        edited_df = st.data_editor(st.session_state['current_card'], hide_index=True, use_container_width=True)
        st.session_state['current_card'] = edited_df
        played = edited_df[edited_df['Score'] > 0]
        tot = played['Score'].sum()
        if st.button("💾 Sauvegarder Partie"):
            p_data = {'date': str(datetime.date.today()), 'score': int(tot), 'putts': int(played['Putts'].sum())}
            add_partie_to_db(p_data)
            st.session_state['parties'].append(p_data)
            st.success("Partie archivée !")

# ==================================================
# ONGLET 2 : PRACTICE
# ==================================================
with tab_practice:
    st.header("🚜 Practice")
    date_prac = st.date_input("Date", datetime.date.today(), key="d_pr")
    c1, c2, c3 = st.columns(3)
    with c1: 
        cp = st.selectbox("Club", CLUBS_ORDER, key="pr_c")
        cont = st.selectbox("Contact", ["Bon", "Gratte", "Top"], key="pr_co")
    with c2: 
        op = st.number_input("Cible", 0, 300, 150, key="pr_o")
        dp = st.number_input("Réel", 0, 300, 150, key="pr_r")
        se = st.selectbox("Effet V", ["Tout droit", "Fade", "Draw"], key="pr_ev")
    with c3:
        dr = st.radio("Dir", ["Gauche", "Centre", "Droite"], key="pr_d")
        lt = st.slider("Dispersion", 0, 5, 0, key="pr_l")
        re = st.selectbox("Effet R", ["Tout droit", "Fade", "Draw", "Raté"], key="pr_er")

    if st.button("Enregistrer Practice"):
        data = {
            'date': str(date_prac), 'mode': 'Practice', 'club': cp,
            'strat_dist': op, 'distance': dp, 'direction': dr, 'score_lateral': lt,
            'type_coup': 'Jeu Long', 'resultat_putt': 'N/A', 'delta_dist': dp-op,
            'contact': cont, 'strat_effet': se, 'real_effet': re,
            'err_longueur': 'Ok', 'lie': 'Tapis', 'strat_type': 'Practice', 
            'par_trou': 0, 'points_test': 0, 'amplitude': 'Plein', 'dist_remain': 0
        }
        add_coup_to_db(data)
        st.session_state['coups'].append(data)
        st.success("Sauvegardé !")
        st.rerun()

# ==================================================
# ONGLET 3 : COMBINE
# ==================================================
with tab_combine:
    st.header("🏆 Combine")
    if st.button("🎲 Lancer"):
        cands = [c for c in CLUBS_ORDER if c != "Putter"]
        sels = np.random.choice(cands, 3, replace=False)
        targs = [{'club': c, 'target': DIST_REF[c] + np.random.randint(-5, 6)} for c in sels]
        st.session_state['combine_state'] = {'clubs_info': targs, 'current_club_idx': 0, 'current_shot': 1, 'score_total': 0}
        st.rerun()

    stt = st.session_state['combine_state']
    if stt and stt['current_club_idx'] < 3:
        inf = stt['clubs_info'][stt['current_club_idx']]
        st.info(f"Club : {inf['club']} | Cible : {inf['target']}m | Balle {stt['current_shot']}/5")
        c1, c2 = st.columns(2)
        with c1: dc = st.number_input("Distance", 0, 350, inf['target'], key="cd")
        with c2: lc = st.slider("Dispersion", 0, 5, 0, key="cl")
        if st.button("Valider"):
            pts = max(0, 50 - (abs(dc - inf['target'])*2)) + max(0, 50 - (lc*10))
            data = {
                'date': str(datetime.date.today()), 'mode': 'Combine', 'club': inf['club'],
                'strat_dist': inf['target'], 'distance': dc, 'score_lateral': lc,
                'direction': 'Centre' if lc==0 else 'Gauche', 'type_coup': 'Jeu Long',
                'points_test': pts, 'resultat_putt': 'N/A', 'delta_dist': dc-inf['target'],
                'lie': 'Practice', 'strat_type': 'Combine', 'par_trou': 0, 
                'strat_effet': 'N/A', 'real_effet': 'N/A', 'amplitude': 'Plein', 'contact': 'Bon', 'dist_remain': 0, 'err_longueur': 'Ok'
            }
            add_coup_to_db(data)
            st.session_state['coups'].append(data)
            stt['score_total'] += pts
            if stt['current_shot'] < 5: stt['current_shot'] += 1
            else: 
                stt['current_shot'] = 1
                stt['current_club_idx'] += 1
            st.rerun()
    elif stt:
        st.success(f"Score : {int(stt['score_total']/15)}/100")
        if st.button("Fermer"):
            st.session_state['combine_state'] = None
            st.rerun()
            
    st.markdown("---")
    if not df_analysis.empty:
        df_c = df_analysis[df_analysis['mode'] == 'Combine']
        if not df_c.empty:
            st.metric("Score Moyen", f"{df_c['points_test'].mean():.0f}/100")

# ==================================================
# ANALYSES
# ==================================================
with tab_dna:
    st.header("🧬 Club DNA")
    if not df_analysis.empty:
        df_l = df_analysis[df_analysis['type_coup'] == 'Jeu Long']
        if not df_l.empty:
            sel = st.selectbox("Club", df_l['club'].unique())
            sub = df_l[df_l['club'] == sel]
            c1, c2 = st.columns(2)
            with c1: 
                fig, ax = plt.subplots()
                plot_dispersion_analysis(ax, sub[sub['mode']=='Practice'], "Practice", "blue")
                st.pyplot(fig)
                # --- CORRECTION V36 : METRIQUES PRACTICE ---
                if not sub[sub['mode']=='Practice'].empty:
                    st.metric("Dispersion Profondeur", f"± {sub[sub['mode']=='Practice']['distance'].std():.1f}m")
                    st.metric("Score Latéral Moyen", f"{sub[sub['mode']=='Practice']['score_lateral'].mean():.1f} / 5")

            with c2:
                fig, ax = plt.subplots()
                plot_dispersion_analysis(ax, sub[sub['mode']=='Parcours'], "Parcours", "red")
                st.pyplot(fig)
                # --- CORRECTION V36 : METRIQUES PARCOURS ---
                if not sub[sub['mode']=='Parcours'].empty:
                    st.metric("Dispersion Profondeur", f"± {sub[sub['mode']=='Parcours']['distance'].std():.1f}m")
                    st.metric("Score Latéral Moyen", f"{sub[sub['mode']=='Parcours']['score_lateral'].mean():.1f} / 5")
            
            st.subheader("Effets")
            df_eff = sub[sub['strat_effet'].isin(["Fade", "Draw", "Tout droit"])]
            if not df_eff.empty:
                df_eff['OK'] = df_eff.apply(lambda x: 1 if x['strat_effet'] in x['real_effet'] else 0, axis=1)
                res = df_eff.groupby('strat_effet')['OK'].mean() * 100
                st.dataframe(res.to_frame("% Réussite").style.background_gradient(cmap="Greens"), use_container_width=True)

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
            
            c1, c2 = st.columns(2)
            with c1:
                def get_pc(r):
                    res = r['resultat_putt']
                    if "Dans" in res: return 0,0
                    x, y = 0, 0
                    if "Gauche" in res: x = -1
                    if "Droite" in res: x = 1
                    if "Court" in res: y = -1
                    if "Long" in res: y = 1
                    return x + np.random.normal(0,0.1), y + np.random.normal(0,0.1)
                coords = df_p.apply(get_pc, axis=1, result_type='expand')
                fig, ax = plt.subplots()
                ax.scatter(coords[0], coords[1], alpha=0.5, s=100, c='purple')
                ax.axhline(0, c='gray'); ax.axvline(0, c='gray')
                st.pyplot(fig)
            with c2:
                misses = df_p[df_p['resultat_putt'] != "Dans le trou"]
                if not misses.empty: st.bar_chart(misses['resultat_putt'].value_counts())
