import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from fpdf import FPDF
import datetime
import plotly.express as px
import plotly.graph_objects as go
import io

# --- CONFIGURATION ---
st.set_page_config(page_title="GolfShot 48.0 Visual Fix", layout="wide")

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
                 (date TEXT, score INTEGER, putts INTEGER, course_name TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS courses
                 (name TEXT PRIMARY KEY, pars TEXT)''')
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
    try: c.execute("INSERT INTO parties VALUES (:date, :score, :putts, :course_name)", data)
    except: c.execute("INSERT INTO parties (date, score, putts) VALUES (:date, :score, :putts)", data)
    conn.commit()

def load_parties_from_db():
    try: return pd.read_sql("SELECT * FROM parties", conn)
    except: return pd.DataFrame()

def save_course(name, pars_list):
    c = conn.cursor()
    pars_str = ",".join(map(str, pars_list))
    c.execute("INSERT OR REPLACE INTO courses VALUES (?, ?)", (name, pars_str))
    conn.commit()

def get_courses_list():
    try:
        df = pd.read_sql("SELECT name FROM courses", conn)
        return df['name'].tolist()
    except: return []

def get_course_pars(name):
    c = conn.cursor()
    c.execute("SELECT pars FROM courses WHERE name=?", (name,))
    res = c.fetchone()
    if res: return [int(x) for x in res[0].split(',')]
    return [4]*18

# --- 2. GÉNÉRATEUR PDF PRO ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'GolfShot Analytics - Rapport Coach', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f"Genere le {datetime.date.today()}", 0, 1, 'C')
        self.line(10, 30, 200, 30)
        self.ln(10)

    def chapter_title(self, label):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(220, 220, 220)
        self.cell(0, 10, label, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()

def create_pro_pdf(df_coups, df_parties):
    pdf = PDF()
    pdf.add_page()
    
    # 1. SYNTHESE
    pdf.chapter_title("1. SYNTHESE")
    if not df_parties.empty:
        last = df_parties.iloc[-1]
        pdf.chapter_body(f"Dernier Score: {last['score']} | Putts: {last['putts']}")
    pdf.chapter_body(f"Volume Total: {len(df_coups)} coups.")
    
    # 2. TABLEAU GAPPING
    pdf.chapter_title("2. ETALONNAGE CLUB")
    pdf.set_font('Arial', 'B', 8)
    cols = [30, 25, 25, 25, 35]
    headers = ['Club', 'Moy (m)', 'Max (m)', 'Disp Prof', 'Disp Lat']
    for i, h in enumerate(headers): pdf.cell(cols[i], 6, h, 1, 0, 'C')
    pdf.ln()
    
    pdf.set_font('Arial', '', 8)
    df_long = df_coups[df_coups['type_coup'] == 'Jeu Long']
    club_list = ["Driver", "Bois 5", "Hybride", "Fer 3", "Fer 5", "Fer 6", "Fer 7", "Fer 8", "Fer 9", "PW", "50°", "55°", "60°"]
    
    for c in club_list:
        sub = df_long[df_long['club'] == c]
        if not sub.empty and len(sub)>1:
            avg = sub['distance'].mean()
            maxx = sub['distance'].max()
            std = sub['distance'].std()
            lat = sub['score_lateral'].mean()
            pdf.cell(cols[0], 6, c, 1, 0, 'C')
            pdf.cell(cols[1], 6, f"{avg:.1f}", 1, 0, 'C')
            pdf.cell(cols[2], 6, f"{maxx:.1f}", 1, 0, 'C')
            pdf.cell(cols[3], 6, f"+/- {std:.1f}", 1, 0, 'C')
            pdf.cell(cols[4], 6, f"{lat:.1f}/5", 1, 0, 'C')
            pdf.ln()
            
    # 3. PUTTING
    pdf.ln(5)
    pdf.chapter_title("3. PUTTING")
    df_p = df_coups[df_coups['type_coup'] == 'Putt']
    if not df_p.empty:
        misses = df_p[df_p['resultat_putt'] != "Dans le trou"]
        if not misses.empty:
            top = misses['resultat_putt'].mode()[0]
            pdf.chapter_body(f"Faute principale : {top}")
        else:
            pdf.chapter_body("Aucun rate enregistré.")

    return pdf.output(dest='S').encode('latin-1', 'replace')

# --- INIT ETATS ---
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
# HELPER GRAPH PLOTLY (RESTAURÉ V45)
# ==================================================
def plot_interactive_dispersion(data, title, color_hex):
    if data.empty: return go.Figure()
    
    # 1. Simulation X (Latéral)
    def get_x(row):
        x = row['score_lateral'] * 5 
        if row['direction'] == 'Gauche': x = -x
        elif row['direction'] == 'Droite': x = x
        else: x = 0
        return x + np.random.normal(0, 1) # Jitter
    
    data = data.copy()
    data['x_viz'] = data.apply(get_x, axis=1)
    
    fig = go.Figure()
    
    # Points
    fig.add_trace(go.Scatter(
        x=data['x_viz'], y=data['distance'],
        mode='markers',
        marker=dict(size=12, color=color_hex, line=dict(width=1, color='white'), opacity=0.8),
        text=data['date'],
        name='Coups',
        hovertemplate='Dist: %{y:.1f}m<br>Lat: %{x:.1f}m<br>Date: %{text}'
    ))
    
    # Cible
    mean_target = data['strat_dist'].mean()
    if mean_target > 0:
        fig.add_trace(go.Scatter(x=[0], y=[mean_target], mode='markers', marker=dict(symbol='star', size=18, color='green'), name='Cible'))
        
    # Ellipse
    if len(data) > 3:
        try:
            cov = np.cov(data['x_viz'], data['distance'])
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals = vals[order]
            vecs = vecs[:,order]
            theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
            w, h = 2 * 2 * np.sqrt(vals) # 2 std dev
            
            t = np.linspace(0, 2*np.pi, 100)
            ell_x = w/2 * np.cos(t)
            ell_y = h/2 * np.sin(t)
            R = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta))], 
                          [np.sin(np.radians(theta)), np.cos(np.radians(theta))]])
            ell_r = np.dot(R, np.array([ell_x, ell_y]))
            
            fig.add_trace(go.Scatter(
                x=ell_r[0,:] + data['x_viz'].mean(),
                y=ell_r[1,:] + data['distance'].mean(),
                mode='lines',
                line=dict(color=color_hex, dash='dot', width=2),
                name='Zone 95%'
            ))
        except: pass
        
    fig.update_layout(
        title=title,
        xaxis_title="Gauche (m) <---> Droite (m)",
        yaxis_title="Profondeur (m)",
        template="plotly_white",
        height=450,
        hovermode="closest",
        showlegend=True
    )
    return fig

# ==================================================
# BARRE LATÉRALE
# ==================================================
st.sidebar.title("⚙️ Data Lab")

uploaded_file = st.sidebar.file_uploader("📂 Importer CSV", type="csv")
if uploaded_file is not None:
    try:
        df_loaded = pd.read_csv(uploaded_file)
        for _, row in df_loaded.iterrows(): add_coup_to_db(row.to_dict())
        st.session_state['coups'] = load_coups_from_db().to_dict('records')
        st.sidebar.success("Importé !")
        st.rerun()
    except: pass

st.sidebar.markdown("---")
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

if st.sidebar.button("📄 Générer Rapport PDF"):
    if not df_analysis.empty:
        try:
            pdf_bytes = create_pro_pdf(df_analysis, pd.DataFrame(st.session_state['parties']))
            st.sidebar.download_button("📥 Télécharger PDF", pdf_bytes, "Rapport_Coach.pdf", "application/pdf")
        except Exception as e: st.sidebar.error(f"Erreur PDF : {e}")
    else: st.sidebar.error("Pas de données.")

st.sidebar.markdown("---")
st.sidebar.header("🤖 Smart Caddie")
with st.sidebar.expander("Assistant", expanded=True):
    cad_dist = st.number_input("Distance (m)", 50, 250, 135, step=5)
    cad_lie = st.selectbox("Lie", ["Tee", "Fairway", "Rough", "Bunker"])
    if not df_analysis.empty:
        df_c = df_analysis[df_analysis['type_coup'] == 'Jeu Long']
        df_l = df_c[df_c['lie'] == cad_lie]
        if len(df_l) < 5: df_l = df_c
        if not df_l.empty:
            stats = df_l.groupby('club')['distance'].mean().reset_index()
            stats['diff'] = abs(stats['distance'] - cad_dist)
            best = stats.nsmallest(1, 'diff')
            if not best.empty:
                rec = best.iloc[0]
                st.markdown(f"<div class='caddie-box'>💡 {rec['club']}<br><small>{rec['distance']:.1f}m</small></div>", unsafe_allow_html=True)

st.sidebar.markdown("---")
if st.sidebar.button("Injecter Données V48"):
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
            t = DIST_REF.get(club, 100)
            r = np.random.normal(t, 10)
            entry.update({'strat_dist': t, 'distance': r, 'score_lateral': np.random.randint(0,3), 'type_coup': 'Jeu Long'})
            if mode == 'Combine': entry['points_test'] = np.random.randint(50, 100)
        add_coup_to_db(entry)
    st.session_state['coups'] = load_coups_from_db().to_dict('records')
    st.success("Données injectées !")
    st.rerun()

if st.session_state['coups']:
    df_ex = pd.DataFrame(st.session_state['coups'])
    st.sidebar.download_button("📥 Backup CSV", convert_df(df_ex), "golf_v48_backup.csv", "text/csv")

if st.sidebar.button("⚠️ Vider DB"):
    conn.execute("DELETE FROM coups"); conn.execute("DELETE FROM parties"); conn.commit()
    st.session_state['coups'] = []; st.session_state['parties'] = []; st.rerun()

# --- INTERFACE ---
st.title("🏌️‍♂️ GolfShot 48.0 : Visual Fix")

tab_parcours, tab_practice, tab_combine, tab_dna, tab_sac, tab_putt = st.tabs([
    "⛳ Parcours", "🚜 Practice", "🏆 Combine", "🧬 Club DNA", "🎒 Mapping", "🟢 Putting"
])

# ==================================================
# ONGLET 1 : PARCOURS
# ==================================================
with tab_parcours:
    with st.expander("🗺️ Gestion des Parcours"):
        c_load, c_new = st.columns(2)
        with c_load:
            courses_avail = get_courses_list()
            if courses_avail:
                selected_course = st.selectbox("Charger un parcours", courses_avail)
                if st.button("Charger la carte"):
                    pars = get_course_pars(selected_course)
                    st.session_state['current_card']['Par'] = pars
                    st.success(f"Parcours {selected_course} chargé !")
            else: st.info("Aucun parcours enregistré.")
        with c_new:
            new_name = st.text_input("Nom nouveau parcours")
            default_pars_str = "4,4,3,4,5,4,3,4,5,4,4,3,4,5,4,3,4,5"
            new_pars_str = st.text_area("Pars (séparés par virgule)", default_pars_str)
            if st.button("Sauvegarder Parcours"):
                try:
                    p_list = [int(x.strip()) for x in new_pars_str.split(',')]
                    if len(p_list) == 18:
                        save_course(new_name, p_list)
                        st.success("Parcours sauvegardé !")
                        st.rerun()
                    else: st.error("Il faut 18 chiffres.")
                except: st.error("Format invalide.")

    st.markdown("---")
    c1, c2 = st.columns([1, 1])
    with c1:
        idx = st.session_state['current_hole'] - 1
        current_par = st.session_state['current_card'].at[idx, 'Par']
        st.header(f"Trou {st.session_state['current_hole']} (Par {current_par})")
        
        club = st.selectbox("Club", CLUBS_ORDER, key="p_c")
        c_i, c_r = st.columns(2)
        with c_i:
            if club == "Putter":
                obj = st.number_input("Cible", 0.0, 30.0, 1.5, 0.1)
                se = "N/A"; stype="Putt"
            else:
                stype = st.selectbox("Type", SHOT_TYPES)
                obj = st.number_input("Cible", 0, 350, 150) if stype != "Départ (Tee Shot)" else 0
                se = st.selectbox("Effet V", ["Tout droit", "Fade", "Draw"])
        with c_r:
            if club=="Putter":
                real = st.number_input("Réel", 0.0, 30.0, obj, 0.1)
                res = st.selectbox("Résultat", PUTT_RESULTS)
                dr="Centre"; lt=0; re="N/A"; lie="Green"
            else:
                real = st.number_input("Réel", 0, 350, int(obj) if obj>0 else 200)
                re = st.selectbox("Effet R", ["Tout droit", "Fade", "Draw", "Raté"])
                dr = st.radio("Dir", ["G", "C", "D"], horizontal=True)
                lt = st.slider("Ecart", 0, 5, 0)
                res="N/A"
                lie = st.selectbox("Lie", ["Tee", "Fairway", "Rough", "Bunker"])

        if st.button("Valider (+1)"):
            d = {
                'date': str(datetime.date.today()), 'mode': 'Parcours', 'club': club,
                'strat_dist': obj, 'distance': real, 'direction': dr, 'score_lateral': lt,
                'real_effet': re, 'strat_effet': se, 'type_coup': 'Putt' if club=='Putter' else 'Jeu Long',
                'delta_dist': real-obj, 'resultat_putt': res, 'err_longueur': 'Ok',
                'strat_type': stype, 'par_trou': current_par, 'lie': lie,
                'points_test': 0, 'amplitude': 'Plein', 'contact': 'Bon', 'dist_remain': 0
            }
            add_coup_to_db(d)
            st.session_state['coups'].append(d) # UPDATE SESSION INSTANT
            st.session_state['shots_on_current_hole'] += 1
            if club == "Putter": st.session_state['putts_on_current_hole'] += 1
            st.session_state['current_card'].at[idx, 'Score'] = st.session_state['shots_on_current_hole']
            st.session_state['current_card'].at[idx, 'Putts'] = st.session_state['putts_on_current_hole']
            if res == "Dans le trou":
                st.balloons()
                st.session_state['shots_on_current_hole'] = 0
                st.session_state['putts_on_current_hole'] = 0
                if st.session_state['current_hole'] < 18: st.session_state['current_hole'] += 1
            st.rerun()

        if st.button("Trou Suivant >>"):
            st.session_state['current_hole'] = min(18, st.session_state['current_hole']+1)
            st.rerun()

    with c2:
        st.subheader("Carte")
        edited = st.data_editor(st.session_state['current_card'], hide_index=True)
        st.session_state['current_card'] = edited
        if st.button("Sauvegarder Partie"):
            tot = edited[edited['Score']>0]['Score'].sum()
            putts = edited[edited['Score']>0]['Putts'].sum()
            add_partie_to_db({'date': str(datetime.date.today()), 'score': int(tot), 'putts': int(putts), 'course_name': 'Inconnu'})
            st.success("Archivé !")

# ==================================================
# ONGLET 2 : PRACTICE
# ==================================================
with tab_practice:
    st.header("🚜 Practice")
    date_prac = st.date_input("Date", datetime.date.today(), key="dp")
    c1, c2, c3 = st.columns(3)
    with c1: 
        cp = st.selectbox("Club", CLUBS_ORDER, key="pc")
        cont = st.selectbox("Contact", ["Bon", "Gratte", "Top"], key="pco")
    with c2: 
        op = st.number_input("Cible", 0, 300, 150, key="po")
        dp = st.number_input("Réel", 0, 300, 150, key="pr")
        se = st.selectbox("Effet V", ["Tout droit", "Fade", "Draw"], key="pev")
    with c3:
        dr = st.radio("Dir", ["G", "C", "D"], key="pd")
        lt = st.slider("Dispersion", 0, 5, 0, key="pl")
        re = st.selectbox("Effet R", ["Tout droit", "Fade", "Draw", "Raté"], key="per")

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
        st.success("OK")
        st.rerun()

# ==================================================
# ONGLET 3 : COMBINE
# ==================================================
with tab_combine:
    st.header("🏆 Combine")
    with st.expander("🎮 Zone de Jeu", expanded=True):
        if st.button("🎲 Lancer"):
            cands = [c for c in CLUBS_ORDER if c != "Putter"]
            sels = np.random.choice(cands, 3, replace=False)
            targs = [{'club': c, 'target': DIST_REF[c]} for c in sels]
            st.session_state['combine_state'] = {'clubs_info': targs, 'current_club_idx': 0, 'current_shot': 1, 'score_total': 0}
            st.rerun()

        stt = st.session_state['combine_state']
        if stt and stt['current_club_idx'] < 3:
            inf = stt['clubs_info'][stt['current_club_idx']]
            st.info(f"Club: {inf['club']} | Cible: {inf['target']}m | Balle {stt['current_shot']}/5")
            c1, c2 = st.columns(2)
            with c1: dc = st.number_input("Dist", 0, 350, inf['target'], key="cd")
            with c2: lc = st.slider("Disp", 0, 5, 0, key="cl")
            if st.button("Valider"):
                pts = max(0, 50 - abs(dc - inf['target'])*2) + max(0, 50 - lc*10)
                data = {
                    'date': str(datetime.date.today()), 'mode': 'Combine', 'club': inf['club'],
                    'strat_dist': inf['target'], 'distance': dc, 'score_lateral': lc, 
                    'direction': 'C', 'type_coup': 'Jeu Long', 'points_test': pts, 'resultat_putt': 'N/A', 'delta_dist': dc-inf['target'],
                    'lie': 'Practice', 'strat_type': 'Combine', 'par_trou': 0, 'strat_effet': 'N/A', 'real_effet': 'N/A', 'amplitude': 'Plein', 'contact': 'Bon', 'dist_remain': 0, 'err_longueur': 'Ok'
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
            st.success(f"Score Final : {int(stt['score_total']/15)}/100")
            if st.button("Fermer"):
                st.session_state['combine_state'] = None; st.rerun()
    
    st.markdown("---")
    st.subheader("📊 Stats Combine")
    if not df_analysis.empty:
        df_c = df_analysis[df_analysis['mode'] == 'Combine']
        if not df_c.empty:
            st.metric("Score Moyen Global", f"{df_c['points_test'].mean():.0f}/100")
            sl = st.selectbox("Club Combine", df_c['club'].unique(), key='sc')
            subset = df_c[df_c['club'] == sl]
            
            c_a1, c_a2 = st.columns(2)
            with c_a1:
                # PLOTLY CHART
                st.plotly_chart(plot_interactive_dispersion(subset, f"Dispersion : {sl}", "#FFA500"), use_container_width=True)
            with c_a2:
                st.metric("Précision Prof.", f"± {subset['distance'].std():.1f}m")
                st.metric("Précision Lat.", f"{subset['score_lateral'].mean():.1f}/5")
                st.write("Comparaison Practice :")
                data_p = df_analysis[(df_analysis['mode'] == 'Practice') & (df_analysis['club'] == sl)]
                # PLOTLY CHART
                st.plotly_chart(plot_interactive_dispersion(data_p, "Ref Practice", "#2196F3"), use_container_width=True)
        else: st.info("Pas de données.")

# ==================================================
# ONGLET 4 : CLUB DNA (SPLIT)
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
                # PLOTLY
                st.plotly_chart(plot_interactive_dispersion(sub[sub['mode']=='Practice'], "Practice (Labo)", "#2196F3"), use_container_width=True)
                if not sub[sub['mode']=='Practice'].empty:
                    st.metric("Disp. Prof.", f"± {sub[sub['mode']=='Practice']['distance'].std():.1f}m")
                    st.metric("Disp. Lat.", f"{sub[sub['mode']=='Practice']['score_lateral'].mean():.1f}/5")
            with c2:
                # PLOTLY
                st.plotly_chart(plot_interactive_dispersion(sub[sub['mode']=='Parcours'], "Parcours (Réalité)", "#D32F2F"), use_container_width=True)
                if not sub[sub['mode']=='Parcours'].empty:
                    st.metric("Disp. Prof.", f"± {sub[sub['mode']=='Parcours']['distance'].std():.1f}m")
                    st.metric("Disp. Lat.", f"{sub[sub['mode']=='Parcours']['score_lateral'].mean():.1f}/5")
            
            st.subheader("Effets")
            df_eff = sub[sub['strat_effet'].isin(["Fade", "Draw"])]
            if not df_eff.empty:
                df_eff['OK'] = df_eff.apply(lambda x: 1 if x['strat_effet'] in x['real_effet'] else 0, axis=1)
                res = df_eff.groupby('strat_effet')['OK'].mean()*100
                st.dataframe(res.to_frame("% Success"), use_container_width=True)

# ==================================================
# ONGLET 5 : MAPPING
# ==================================================
with tab_sac:
    st.header("🎒 Mapping")
    if not df_analysis.empty:
        df_s = df_analysis[df_analysis['type_coup'] == 'Jeu Long']
        if not df_s.empty:
            df_s['club'] = pd.Categorical(df_s['club'], categories=CLUBS_ORDER, ordered=True)
            df_s = df_s.sort_values('club')
            # PLOTLY BOX
            fig = px.box(df_s, x='club', y='distance', color='club', title="Étalonnage Interactif")
            st.plotly_chart(fig, use_container_width=True)
            stats = df_s.groupby('club', observed=True)['distance'].agg(['count', 'mean', 'max', 'std']).round(1)
            st.dataframe(stats.style.background_gradient(cmap="Blues"), use_container_width=True)

# ==================================================
# ONGLET 6 : PUTTING
# ==================================================
with tab_putt:
    st.header("🟢 Putting")
    if not df_analysis.empty:
        df_p = df_analysis[df_analysis['type_coup'] == 'Putt'].copy()
        if not df_p.empty:
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
                    return x + np.random.normal(0,0.15), y + np.random.normal(0,0.15)
                coords = df_p.apply(get_pc, axis=1, result_type='expand')
                df_p['x'] = coords[0]; df_p['y'] = coords[1]
                # PLOTLY SCATTER
                fig = px.scatter(df_p, x='x', y='y', color='resultat_putt', title="Boussole Interactive")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                df_p['Zone'] = pd.cut(df_p['strat_dist'], [0,2,5,10,30], labels=["0-2m","2-5m","5-10m","+10m"])
                piv = df_p.groupby('Zone', observed=False).apply(lambda x: (x['resultat_putt']=="Dans le trou").mean()*100)
                st.dataframe(piv.to_frame("%").style.background_gradient(cmap="RdYlGn"), use_container_width=True)
                
                st.subheader("Histogramme des Ratés")
                misses = df_p[df_p['resultat_putt'] != "Dans le trou"]
                if not misses.empty:
                    cnt = misses['resultat_putt'].value_counts()
                    # PLOTLY BAR
                    fig_m = px.bar(x=cnt.index, y=cnt.values, color=cnt.values, color_continuous_scale='Reds')
                    st.plotly_chart(fig_m, use_container_width=True)
