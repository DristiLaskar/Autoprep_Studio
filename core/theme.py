"""Colour palette + CSS shared by the Streamlit UI and the Plotly charts."""

PALETTE = ["#7C5CFF", "#00D4B4", "#FF6B9D", "#FFB547", "#4DA3FF", "#9BE564", "#FF7A59", "#C77DFF",
           "#2EC4B6", "#F15BB5"]
TEXT = "#E8ECF8"
MUTED = "#9AA4C7"
GRID = "rgba(255,255,255,0.08)"

# dark navy -> violet -> teal -> yellow (sequential)
SEQ = [[0.0, "#1B1F4B"], [0.35, "#5B4BDB"], [0.7, "#00D4B4"], [1.0, "#F9F871"]]
# pink <- neutral -> teal (diverging)
DIV = [[0.0, "#FF6B9D"], [0.5, "#1A2040"], [1.0, "#00D4B4"]]
# green -> amber -> red (risk)
RISK = [[0.0, "#2EC4B6"], [0.5, "#FFB547"], [1.0, "#FF4D6D"]]

CSS = """
<style>
:root { --accent:#7C5CFF; --accent2:#00D4B4; --pink:#FF6B9D; --line:rgba(255,255,255,.09); }
.block-container { padding-top: 1.4rem; max-width: 1320px; }
header[data-testid="stHeader"] { background: transparent; }

.hero { background: linear-gradient(120deg, rgba(124,92,255,.38), rgba(0,212,180,.26));
        border: 1px solid var(--line); border-radius: 22px; padding: 26px 32px; margin-bottom: 18px; }
.hero h1 { margin: 0; font-size: 2.15rem; font-weight: 800; letter-spacing: -.5px; }
.hero p { margin: .45rem 0 0; opacity: .88; font-size: 1.02rem; }

.pill { display:inline-block; padding: 3px 12px; border-radius: 999px; font-size: .78rem; font-weight: 600;
        margin: 2px 6px 2px 0; background: rgba(124,92,255,.22); border: 1px solid rgba(124,92,255,.55); }
.pill.teal { background: rgba(0,212,180,.16); border-color: rgba(0,212,180,.55); }
.pill.pink { background: rgba(255,107,157,.16); border-color: rgba(255,107,157,.55); }
.pill.amber { background: rgba(255,181,71,.16); border-color: rgba(255,181,71,.55); }

div[data-testid="stMetric"] { background: linear-gradient(145deg, #171F3E, #101733); border: 1px solid var(--line);
        border-left: 4px solid var(--accent); padding: 14px 18px; border-radius: 14px; }
div[data-testid="stMetricValue"] { font-weight: 800; }
div[data-testid="stMetricLabel"] p { opacity: .8; }

.stTabs [data-baseweb="tab-list"] { gap: 6px; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { background: rgba(255,255,255,.04); border-radius: 12px 12px 0 0; padding: 9px 16px; height: auto; }
.stTabs [aria-selected="true"] { background: linear-gradient(120deg, rgba(124,92,255,.40), rgba(0,212,180,.28)); }

.stButton > button, .stDownloadButton > button { background: linear-gradient(120deg, #7C5CFF, #00D4B4); color: #fff; border: 0;
        border-radius: 12px; font-weight: 700; padding: .6rem 1.2rem; transition: all .15s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { filter: brightness(1.12); transform: translateY(-1px); color:#fff; border:0; }

.card { background: linear-gradient(145deg, #171F3E, #101733); border: 1px solid var(--line); border-radius: 16px;
        padding: 16px 20px; margin-bottom: 12px; }
.card h4 { margin: 0 0 6px 0; }
.big-download { background: linear-gradient(120deg, rgba(124,92,255,.30), rgba(0,212,180,.22)); border: 1px solid var(--line);
        border-radius: 18px; padding: 18px 24px; margin: 8px 0 18px 0; }
.stage { display:flex; align-items:center; gap:10px; padding: 6px 0; }
.stage .dot { width: 12px; height: 12px; border-radius: 50%; background: linear-gradient(120deg,#7C5CFF,#00D4B4); }
.small { opacity: .75; font-size: .85rem; }
</style>
"""
