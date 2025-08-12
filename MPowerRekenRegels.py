import base64
import io

import dash
from dash import html, dcc, dash_table
import plotly.graph_objects as go
import pandas as pd
from dash.dependencies import Input, Output, State

# =========================
# INITIËLE DATA
# =========================
regels_data = [
    {"Prioriteit": 2, "Op1": "<", "Grens1": 60,  "Tijd1": 18, "Minuut1": "15'", "Op2": ">", "Grens2": -240000, "Tijd2": 18, "Minuut2": "15'", "Op3": "<", "Grens3": 250, "Tijd3": 1, "Minuut3": "1'", "Actie": "Stop behouden - 15'",  "Vermogen": -2000},
    {"Prioriteit": 3, "Op1": "<", "Grens1": 30,  "Tijd1": 7,  "Minuut1": "1'",  "Op2": ">", "Grens2": 60000,   "Tijd2": 7,  "Minuut2": "1'",  "Op3": ">", "Grens3": 500, "Tijd3": 5, "Minuut3": "1'", "Actie": "Stop (snel) - 1'",   "Vermogen": -2000},
    {"Prioriteit": 4, "Op1": "<", "Grens1": 30,  "Tijd1": 18, "Minuut1": "15'", "Op2": ">", "Grens2": 120000,  "Tijd2": 18, "Minuut2": "15'", "Op3": ">", "Grens3": 800, "Tijd3": 1, "Minuut3": "1'", "Actie": "Stop - 15'",           "Vermogen": -2000},
    {"Prioriteit": 5, "Op1": "<", "Grens1": 50,  "Tijd1": 5,  "Minuut1": "1'",  "Op2": ">", "Grens2": 30000,   "Tijd2": 5,  "Minuut2": "1'",  "Op3": ">", "Grens3": 800, "Tijd3": 1, "Minuut3": "1'", "Actie": "Down - 1'",           "Vermogen": -600},
    {"Prioriteit": 6, "Op1": ">", "Grens1": 150, "Tijd1": 5,  "Minuut1": "1'",  "Op2": "<", "Grens2": 30000,   "Tijd2": 5,  "Minuut2": "1'",  "Op3": ">", "Grens3": 250, "Tijd3": 1, "Minuut3": "1'", "Actie": "Up - 1'",             "Vermogen": 2000},
    {"Prioriteit": 7, "Op1": ">", "Grens1": 70,  "Tijd1": 18, "Minuut1": "15'", "Op2": "<", "Grens2": 50000,   "Tijd2": 18, "Minuut2": "15'", "Op3": ">", "Grens3": 500, "Tijd3": 1, "Minuut3": "1'", "Actie": "Start behouden - 15'", "Vermogen": 1400},
    {"Prioriteit": 8, "Op1": ">", "Grens1": 120, "Tijd1": 7,  "Minuut1": "1'",  "Op2": "<", "Grens2": -160000, "Tijd2": 7,  "Minuut2": "1'",  "Op3": "<", "Grens3": 250, "Tijd3": 1, "Minuut3": "1'", "Actie": "Start (snel) - 1'",   "Vermogen": 1400},
    {"Prioriteit": 9, "Op1": ">", "Grens1": 120, "Tijd1": 18, "Minuut1": "15'", "Op2": "<", "Grens2": -160000, "Tijd2": 18, "Minuut2": "15'", "Op3": "<", "Grens3": 250, "Tijd3": 1, "Minuut3": "1'", "Actie": "Start - 15'",          "Vermogen": 1400}
]
df_rules_init = pd.DataFrame(regels_data).sort_values("Prioriteit").reset_index(drop=True)

# =========================
# STIJL & HELPERS
# =========================
FONT = {'fontFamily': 'Calibri', 'fontSize': '14px'}

# Kolombreedtes / uitlijning
LABEL_WIDTH = 180
COND_SUBCOL_WIDTH = 80
META_PRI_WIDTH = COND_SUBCOL_WIDTH * 4

RED_HDR = "#fde7e7"
BLUE_HDR = "#e7f0ff"

YRANGE_IMBAL = (-250, 250)          # €/MWh
YRANGE_SI    = (-250000, 250000)    # kWh
YRANGE_PWR   = (0, 1000)            # kW

RANGES = {
    "Imbal_Grens": (-500.0, 500.0),
    "SI_Grens": (-400000.0, 400000.0),
    "Power_Grens": (0.0, 2000.0),
    "Tijd": (0.0, 60.0),
    "Vermogen": (-4000.0, 4000.0),
}
VALID_TYPES = {"1'", "15'"}
VALID_OPS = {"<", ">"}

# CSS: dropdowns boven grafieken + overflow zichtbaar
CSS_RULES = [
    {"selector": ".dash-spreadsheet-container .dash-spreadsheet-inner", "rule": "overflow: visible !important;"},
    {"selector": ".dash-spreadsheet td div", "rule": "overflow: visible;"},
    {"selector": ".dash-table-container", "rule": "overflow: visible;"},
    {"selector": ".dash-cell div.dash-dropdown", "rule": "position: relative; z-index: 2000;"},
    {"selector": ".dash-cell div.dash-dropdown > div", "rule": "position: relative; z-index: 2000;"},
    {"selector": ".Select-menu-outer", "rule": "z-index: 2100;"},
    {"selector": ".Select-menu", "rule": "z-index: 2100;"},
]

def _norm_op(op: str) -> str:
    s = (op or "").strip()
    return "<" if s in ("<","<=") else ">"

def _norm_minute_str(s: str) -> str:
    return (s or "").strip().replace("′", "'")

def _is_up(actie: str) -> bool:
    return actie in ["Up", "Start behouden", "Start (snel)", "Start"]

def _bar_color(actie: str, is_15: bool) -> str:
    if _is_up(actie): return "#3d6dcc" if not is_15 else "#a9c1f5"
    return "#d64545" if not is_15 else "#f0a8a8"

def _clamp(v, lo, hi): return max(lo, min(hi, v))
def _num(x, default=0.0):
    v = pd.to_numeric(x, errors="coerce")
    return float(v) if pd.notna(v) else float(default)

def _segment(op: str, grens: float, lo: float, hi: float):
    op = _norm_op(op)
    if op == "<": top, base = _clamp(grens, lo, hi), lo
    else:         base, top = _clamp(grens, lo, hi), hi
    return base, top

# =========================
# TABELLEN (META + CONDITIES)
# =========================
def build_meta_df(df_rules: pd.DataFrame) -> pd.DataFrame:
    df = df_rules.sort_values("Prioriteit").reset_index(drop=True)
    cols = [str(p) for p in df["Prioriteit"]]
    data = {"Label": ["Actie", "Vermogen", "Prioriteit"]}
    for c in cols: data[c] = [None]*3
    for i, c in enumerate(cols):
        data[c][0] = df.loc[i, "Actie"]
        data[c][1] = df.loc[i, "Vermogen"]
        data[c][2] = int(df.loc[i, "Prioriteit"])
    return pd.DataFrame(data)

def build_cond_columns(prioriteiten: list[int]):
    # volgorde: Type, Op, Grens, Tijd
    cols = [{"name": "Label", "id": "Label"}]
    for p in prioriteiten:
        parent = str(p)
        cols += [
            {"name": [parent, "Type"],  "id": f"{parent}||Type",  "presentation": "dropdown"},
            {"name": [parent, "Op"],    "id": f"{parent}||Op",    "presentation": "dropdown"},
            {"name": [parent, "Grens"], "id": f"{parent}||Grens"},
            {"name": [parent, "Tijd"],  "id": f"{parent}||Tijd"},
        ]
    return cols

def rules_to_cond_df(df_rules: pd.DataFrame) -> pd.DataFrame:
    df = df_rules.sort_values("Prioriteit").reset_index(drop=True)
    pri = [int(p) for p in df["Prioriteit"]]
    rows = {"Label": ["Imbal (€/MWh)", "SI (kWh)", "Powersetpoint (kW)"]}

    for p in pri:
        for fld in ["Type","Op","Grens","Tijd"]:
            rows[f"{p}||{fld}"] = [None, None, None]

    for i, p in enumerate(pri):
        rows[f"{p}||Type"] [0] = df.loc[i, "Minuut1"]
        rows[f"{p}||Op"]   [0] = df.loc[i, "Op1"]
        rows[f"{p}||Grens"][0] = df.loc[i, "Grens1"]
        rows[f"{p}||Tijd"] [0] = df.loc[i, "Tijd1"]

        rows[f"{p}||Type"] [1] = df.loc[i, "Minuut2"]
        rows[f"{p}||Op"]   [1] = df.loc[i, "Op2"]
        rows[f"{p}||Grens"][1] = df.loc[i, "Grens2"]
        rows[f"{p}||Tijd"] [1] = df.loc[i, "Tijd2"]

        rows[f"{p}||Type"] [2] = df.loc[i, "Minuut3"]
        rows[f"{p}||Op"]   [2] = df.loc[i, "Op3"]
        rows[f"{p}||Grens"][2] = df.loc[i, "Grens3"]
        rows[f"{p}||Tijd"] [2] = df.loc[i, "Tijd3"]

    return pd.DataFrame(rows)

def cond_to_rules(df_meta: pd.DataFrame, df_cond: pd.DataFrame) -> pd.DataFrame:
    meta = df_meta.copy(); meta.columns = [str(c) for c in meta.columns]
    cond = df_cond.copy(); cond.columns = [str(c) for c in cond.columns]
    pri = []
    for c in meta.columns:
        if c == "Label": continue
        try: pri.append(int(c))
        except: pass
    pri.sort()

    meta_map = {col: {} for col in [c for c in meta.columns if c!="Label"]}
    for _, r in meta.iterrows():
        lab = str(r["Label"]).strip()
        for col in meta_map: meta_map[col][lab] = r[col]

    row_map = {}
    for _, r in cond.iterrows():
        rowlabel = str(r["Label"])
        row_map[rowlabel] = r.to_dict()

    im, si, pw = "Imbal (€/MWh)", "SI (kWh)", "Powersetpoint (kW)"
    recs = []
    for p in pri:
        col = str(p)
        recs.append({
            "Prioriteit": int(meta_map[col].get("Prioriteit", p)),
            "Actie":      str(meta_map[col].get("Actie","")),
            "Vermogen":   _num(meta_map[col].get("Vermogen"), 0),

            "Minuut1": _norm_minute_str(str(row_map.get(im, {}).get(f"{col}||Type",""))),
            "Op1":     _norm_op(row_map.get(im, {}).get(f"{col}||Op", "<")),
            "Grens1":  _num(row_map.get(im, {}).get(f"{col}||Grens"), 0),
            "Tijd1":   _num(row_map.get(im, {}).get(f"{col}||Tijd"), 0),

            "Minuut2": _norm_minute_str(str(row_map.get(si, {}).get(f"{col}||Type",""))),
            "Op2":     _norm_op(row_map.get(si, {}).get(f"{col}||Op", ">")),
            "Grens2":  _num(row_map.get(si, {}).get(f"{col}||Grens"), 0),
            "Tijd2":   _num(row_map.get(si, {}).get(f"{col}||Tijd"), 0),

            "Minuut3": _norm_minute_str(str(row_map.get(pw, {}).get(f"{col}||Type",""))),
            "Op3":     _norm_op(row_map.get(pw, {}).get(f"{col}||Op", "<")),
            "Grens3":  _num(row_map.get(pw, {}).get(f"{col}||Grens"), 0),
            "Tijd3":   _num(row_map.get(pw, {}).get(f"{col}||Tijd"), 0),
        })
    return pd.DataFrame(recs).sort_values("Prioriteit").reset_index(drop=True)

# =========================
# VALIDATIE
# =========================
def validate_rules(df_rules: pd.DataFrame):
    df = df_rules.copy()
    notes = []

    lo, hi = RANGES["Vermogen"]
    cv = df["Vermogen"].clip(lo, hi)
    if (cv != df["Vermogen"]).any(): notes.append(f"Vermogen buiten bereik [{lo}, {hi}] gecorrigeerd.")
    df["Vermogen"] = cv

    for k, default in [("Op1","<"),("Op2",">"),("Op3","<")]:
        invalid = ~df[k].isin(VALID_OPS)
        if invalid.any():
            df.loc[invalid, k] = default
            notes.append(f"Operatoren in {k} gecorrigeerd naar '{default}'.")

    for k in ["Minuut1","Minuut2","Minuut3"]:
        invalid = ~df[k].isin(VALID_TYPES)
        if invalid.any():
            df.loc[invalid, k] = "1'"
            notes.append(f"Type in {k} ingesteld op 1'.")

    tlo, thi = RANGES["Tijd"]
    for k in ["Tijd1","Tijd2","Tijd3"]:
        clip = df[k].clip(tlo, thi)
        if (clip != df[k]).any(): notes.append(f"Tijd in {k} buiten bereik [{tlo}, {thi}] gecorrigeerd.")
        df[k] = clip

    for k, rn in [("Grens1","Imbal_Grens"),("Grens2","SI_Grens"),("Grens3","Power_Grens")]:
        lo, hi = RANGES[rn]
        clip = df[k].clip(lo, hi)
        if (clip != df[k]).any(): notes.append(f"{k} buiten bereik [{lo}, {hi}] gecorrigeerd.")
        df[k] = clip

    return df, notes

# =========================
# GRAFIEKEN (grid midden tussen staven + custom labels)
# =========================
def _figure_conditioneel(df_in: pd.DataFrame, which: str) -> go.Figure:
    df_ = df_in.sort_values("Prioriteit").reset_index(drop=True)

    if which == "Imbal":
        lo, hi = YRANGE_IMBAL
        ops, grenzen, minuten, tijden = df_["Op1"], df_["Grens1"], df_["Minuut1"], df_["Tijd1"]
        ytitle, title, unit = "€/MWh", "Imbal Pr (conditioneel gebied)", "€/MWh"
    elif which == "SI":
        lo, hi = YRANGE_SI
        ops, grenzen, minuten, tijden = df_["Op2"], df_["Grens2"], df_["Minuut2"], df_["Tijd2"]
        ytitle, title, unit = "kWh", "SI (conditioneel gebied)", "kWh"
    else:
        lo, hi = YRANGE_PWR
        ops, grenzen, minuten, tijden = df_["Op3"], df_["Grens3"], df_["Minuut3"], df_["Tijd3"]
        ytitle, title, unit = "kW", "Powersetpoint (conditioneel gebied)", "kW"

    n = len(df_)
    xvals  = list(range(1, n+1))         # numeriek
    xtexts = [str(p) for p in df_["Prioriteit"]]

    bases, heights, colors = [0]*n, [0]*n, [None]*n
    dmin, dmax = lo, hi

    for i in range(n):
        op_i    = str(ops.iloc[i])
        grens_i = _num(grenzen.iloc[i], 0)
        tijd_i  = _num(tijden.iloc[i], 0)
        base, top = _segment(op_i, grens_i, lo, hi)
        bases[i]   = base
        heights[i] = top - base
        colors[i]  = _bar_color(df_["Actie"].iloc[i], _norm_minute_str(minuten.iloc[i]) == "15'")
        dmin = min(dmin, grens_i)
        dmax = max(dmax, grens_i)

    fig = go.Figure(go.Bar(
        x=xvals, y=heights, base=bases, marker_color=colors, marker_line_width=0, cliponaxis=False,
        hovertemplate="<b>Prioriteit %{customdata[3]}</b><br>"
                      "Actie: %{customdata[0]}<br>"
                      "Voorwaarde: %{customdata[1]} "+unit+"<br>"
                      "Tijd: %{customdata[2]} min<extra></extra>",
        customdata=[
            [df_["Actie"].iloc[i],
             f"{_norm_op(str(ops.iloc[i]))} {_num(grenzen.iloc[i],0)}",
             int(_num(tijden.iloc[i],0)),
             int(df_["Prioriteit"].iloc[i])]
            for i in range(n)
        ],
        name="Voorwaarde"
    ))

    # Annotaties (min)
    for i in range(n):
        if heights[i] <= 0: 
            continue
        y_mid = bases[i] + heights[i] / 2.0
        tijd_val = int(_num(tijden.iloc[i], 0))
        fig.add_annotation(
            x=xvals[i], y=y_mid, text=f"{tijd_val} min",
            showarrow=False,
            font=dict(family="Calibri", size=13, color="#111"),
            align="center",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.25)", borderwidth=1, borderpad=3,
            xanchor="center", yanchor="middle"
        )

    margin = 0.05 * (dmax - dmin or 1.0)
    auto_lo, auto_hi = min(lo, dmin - margin), max(hi, dmax + margin)
    gridcolor = "#d9d9d9"

    # ---- Layout: grid tussen staven, ticklabels uit ----
    fig.update_layout(
        title=title, height=300,
        margin=dict(l=50, r=20, t=42, b=60),   # extra bottom voor custom labels
        template="none",
        font=dict(family="Calibri", size=14),
        yaxis=dict(range=[auto_lo, auto_hi], title=ytitle, zeroline=True, zerolinecolor="#bbbbbb",
                   showgrid=True, gridcolor=gridcolor, gridwidth=1, showline=True, linecolor="#cccccc"),
        xaxis=dict(
            title=None,            # we plaatsen eigen labels
            showticklabels=False,  # verberg standaard labels
            showgrid=True, gridcolor=gridcolor, gridwidth=1,
            tick0=0.5, dtick=1,    # ✅ grid midden tussen staven (0.5, 1.5, ...)
            showline=True, linecolor="#cccccc"
        ),
        bargap=0.22,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0)
    )

    # ---- Custom x-as labels (Prioriteit) onder de as via yref='paper' ----
    for i, lbl in enumerate(xtexts):
        fig.add_annotation(
            x=xvals[i],
            y=-0.12,
            xref="x", yref="paper",
            text=lbl,
            showarrow=False,
            font=dict(family="Calibri", size=13, color="#333"),
            align="center"
        )

    fig.add_annotation(
        x=(xvals[0] + xvals[-1]) / 2 if xvals else 0,
        y=-0.22, xref="x", yref="paper",
        text="Prioriteit",
        showarrow=False,
        font=dict(family="Calibri", size=13, color="#555")
    )

    return fig

# =========================
# DASH APP
# =========================
app = dash.Dash(__name__)
app.title = "Matrix – export/import/print (8054)"

# Stores & Download
store_import = dcc.Store(id="store-import", data=None)
download_csv = dcc.Download(id="download-csv")

df_meta_init = build_meta_df(df_rules_init)
df_cond_init = rules_to_cond_df(df_rules_init)
prioriteiten = [int(c) for c in df_meta_init.columns if c != "Label"]; prioriteiten.sort()

meta_columns = [{"name": "Label", "id": "Label"}] + [{"name": str(p), "id": str(p)} for p in prioriteiten]
cond_columns = build_cond_columns(prioriteiten)

# Dropdowns
operator_opts = [{"label": "<", "value": "<"}, {"label": ">", "value": ">"}]
minuut_opts   = [{"label": "1'", "value": "1'"}, {"label": "15'", "value": "15'"}]
actie_opts    = [{"label": s, "value": s} for s in ["Stop behouden","Stop (snel)","Stop","Down","Up","Start behouden","Start (snel)","Start"]]

def meta_dropdowns(pri):
    cond = []
    for p in pri:
        cond.append({"if": {"column_id": str(p), "filter_query": '{Label} = "Actie"'}, "options": actie_opts})
    return cond

def cond_dropdowns(pri):
    cond = []
    for p in pri:
        cond.append({"if": {"column_id": f"{p}||Op"},   "options": operator_opts})
        cond.append({"if": {"column_id": f"{p}||Type"}, "options": minuut_opts})
    return cond

def header_styles(pri):
    styles = [{"if": {"column_id": "Label"}, "width": f"{LABEL_WIDTH}px"}]
    for p in pri:
        col_meta = str(p)
        shade = RED_HDR if 2 <= p <= 5 else BLUE_HDR
        styles.append({"if": {"column_id": col_meta}, "backgroundColor": shade, "width": f"{META_PRI_WIDTH}px"})
        for sub in ["Type","Op","Grens","Tijd"]:
            styles.append({"if": {"column_id": f"{p}||{sub}"}, "backgroundColor": shade, "width": f"{COND_SUBCOL_WIDTH}px"})
    return styles

def cond_alignments(pri):
    styles = [{"if": {"column_id": "Label"}, "textAlign": "left"}]
    for p in pri:
        styles.append({"if": {"column_id": f"{p}||Op"}, "textAlign": "right"})
    # dropdown-cellen Type/Op lichtgrijs
    for p in pri:
        for sub in ["Type", "Op"]:
            styles.append({"if": {"column_id": f"{p}||{sub}"}, "backgroundColor": "#f2f2f2"})
    return styles

app.layout = html.Div(
    [
        store_import,
        download_csv,
        html.H3("Rekenregels – Matrix (export/import/print) ↔ Grafieken", style={'margin':'12px 8px', **FONT}),
        html.Div([
            html.Button("Reset", id="btn-reset", n_clicks=0, style={"marginRight":"8px"}),
            html.Button("Export CSV", id="btn-export", n_clicks=0, style={"marginRight":"8px"}),
            dcc.Upload(
                id="upload-csv",
                children=html.Div(["📥 Sleep CSV hier of ", html.A("klik om te kiezen")]),
                multiple=False,
                accept=".csv,text/csv",
                style={
                    "display": "inline-block", "padding": "6px 10px", "border": "1px dashed #999",
                    "borderRadius": "6px", "background": "#fafafa", "cursor": "pointer", "marginRight": "8px"
                }
            ),
            html.Button("Afdrukken", id="btn-print", n_clicks=0),
        ], style={"marginBottom":"8px"}),
        html.H4("Meta", style={**FONT, 'fontWeight':'bold', 'marginTop':'8px'}),
        dash_table.DataTable(
            id="meta-table",
            columns=meta_columns,
            data=df_meta_init.to_dict("records"),
            editable=True,
            style_table={'overflowX': 'auto', 'overflowY': 'visible'},
            css=CSS_RULES,
            style_cell={
                'fontFamily':'Calibri','fontSize':'14px','padding':'6px','textAlign':'center',
                'backgroundColor': '#fdfdfd', 'border': '1px solid #ddd', 'zIndex': 10
            },
            style_header={'fontWeight':'bold'},
            style_header_conditional=header_styles(prioriteiten),
            style_data_conditional=[
                {"if": {"column_id": "Label"}, "textAlign": "left"},
                {"if": {"filter_query": '{Label} = "Actie" or {Label} = "Vermogen"'}, "fontWeight": "bold"},
                *[
                    {"if": {"column_id": str(p), "filter_query": '{Label} = "Vermogen" and {'+str(p)+'} < 0'}, "color": "red"}
                    for p in prioriteiten
                ]
            ],
            dropdown_conditional=meta_dropdowns(prioriteiten),
            fill_width=False
        ),
        html.H4("Condities", style={**FONT, 'fontWeight':'bold', 'marginTop':'16px'}),
        dash_table.DataTable(
            id="cond-table",
            columns=cond_columns,
            data=df_cond_init.to_dict("records"),
            editable=True,
            merge_duplicate_headers=True,
            style_table={'overflowX': 'auto', 'overflowY': 'visible'},
            css=CSS_RULES,
            style_cell={
                'fontFamily':'Calibri','fontSize':'14px','padding':'6px','textAlign':'center',
                'backgroundColor': '#fdfdfd', 'border': '1px solid #ddd', 'zIndex': 10
            },
            style_header={'fontWeight':'bold'},
            style_header_conditional=header_styles(prioriteiten),
            style_data_conditional=cond_alignments(prioriteiten),
            dropdown_conditional=cond_dropdowns(prioriteiten),
            fill_width=False
        ),
        html.Div(id="graphs-container", style={'marginTop':'16px', 'position':'relative', 'zIndex': 0}),
        html.Div(id="validation-report", style={'marginTop':'8px','fontFamily':'Calibri','fontSize':'13px','color':'#7a4b00'}),
        html.Hr(),
        html.H4("Debug: eerste rekenregel na mapping"),
        html.Pre(id="debug-first-rule", style={"whiteSpace": "pre-wrap", "fontFamily": "monospace", "background": "#f7f7f7", "padding": "8px"})
    ],
    style={'background':'#fafafa','padding':'8px'}
)

# =========================
# CLIENTSIDE PRINT
# =========================
app.clientside_callback(
    """
    function(n){
        if(n && n > 0){ window.print(); }
        return 0; // reset n_clicks
    }
    """,
    Output("btn-print", "n_clicks"),
    Input("btn-print", "n_clicks")
)

# =========================
# IMPORT (UPLOAD) → parse CSV naar store
# =========================
@app.callback(
    Output("store-import", "data"),
    Input("upload-csv", "contents"),
    State("upload-csv", "filename"),
    prevent_initial_call=True
)
def parse_upload(contents, filename):
    if not contents:
        return None
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df_up = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    except Exception:
        # fallback latin-1
        df_up = pd.read_csv(io.StringIO(decoded.decode("latin-1")))
    # Verwacht schema: Prioriteit, Actie, Vermogen, Op1, Grens1, Tijd1, Minuut1, Op2, Grens2, Tijd2, Minuut2, Op3, Grens3, Tijd3, Minuut3
    needed = ["Prioriteit","Actie","Vermogen",
              "Op1","Grens1","Tijd1","Minuut1",
              "Op2","Grens2","Tijd2","Minuut2",
              "Op3","Grens3","Tijd3","Minuut3"]
    missing = [c for c in needed if c not in df_up.columns]
    if missing:
        return None
    # enforce types
    df_up = df_up[needed].copy()
    df_up["Prioriteit"] = pd.to_numeric(df_up["Prioriteit"], errors="coerce").astype("Int64")
    df_up["Vermogen"]   = pd.to_numeric(df_up["Vermogen"], errors="coerce")
    for k in ["Grens1","Grens2","Grens3","Tijd1","Tijd2","Tijd3"]:
        df_up[k] = pd.to_numeric(df_up[k], errors="coerce")
    return df_up.sort_values("Prioriteit").to_dict("records")

# =========================
# EXPORT CSV
# =========================
@app.callback(
    Output("download-csv", "data"),
    Input("btn-export", "n_clicks"),
    State("meta-table", "data"),
    State("cond-table", "data"),
    prevent_initial_call=True
)
def on_export(n, meta_rows, cond_rows):
    # bouw regels uit de tabellen
    df_meta = pd.DataFrame(meta_rows); df_meta.columns = [str(c) for c in df_meta.columns]
    df_cond = pd.DataFrame(cond_rows); df_cond.columns = [str(c) for c in df_cond.columns]
    df_rules = cond_to_rules(df_meta, df_cond)
    df_rules_valid, _ = validate_rules(df_rules)
    csv_str = df_rules_valid.to_csv(index=False)
    return dict(content=csv_str, filename="rekenregels.csv", type="text/csv")

# =========================
# HOOFD-CALLBACK (reset / edits / import)
# =========================
@app.callback(
    Output("meta-table", "data"),
    Output("cond-table", "data"),
    Output("graphs-container", "children"),
    Output("validation-report", "children"),
    Output("debug-first-rule", "children"),
    Input("btn-reset", "n_clicks"),
    Input("meta-table", "data"),
    Input("cond-table", "data"),
    Input("store-import", "data"),       # <- import via store
    prevent_initial_call=False
)
def on_any_change(n_reset, meta_rows, cond_rows, imported_rows):
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    # 1) Reset
    if triggered.startswith("btn-reset"):
        meta_data = build_meta_df(df_rules_init).to_dict("records")
        cond_data = rules_to_cond_df(df_rules_init).to_dict("records")

    # 2) Import
    elif triggered.startswith("store-import") and imported_rows:
        df_imp = pd.DataFrame(imported_rows)
        # valideren & doorzetten naar de twee tabellen
        df_imp_valid, _ = validate_rules(df_imp)
        meta_data = build_meta_df(df_imp_valid).to_dict("records")
        cond_data = rules_to_cond_df(df_imp_valid).to_dict("records")

    # 3) Gewone edit
    else:
        meta_data = meta_rows
        cond_data = cond_rows

    # Bouw grafieken uit huidige tabellen
    df_meta = pd.DataFrame(meta_data); df_meta.columns = [str(c) for c in df_meta.columns]
    df_cond = pd.DataFrame(cond_data); df_cond.columns = [str(c) for c in df_cond.columns]

    # prioriteiten
    pri = []
    for c in df_meta.columns:
        if c == "Label": continue
        try: pri.append(int(c))
        except: pass
    pri.sort()

    # kolommen verzekeren
    needed_cond_cols = [c["id"] for c in build_cond_columns(pri)]
    for c in needed_cond_cols:
        if c not in df_cond.columns: df_cond[c] = None
    for p in pri:
        if str(p) not in df_meta.columns: df_meta[str(p)] = None

    # Matrix → rules
    df_rules = cond_to_rules(df_meta, df_cond)

    # Validatie (server-side) + rapport
    df_rules_valid, notes = validate_rules(df_rules)

    # Debug
    first_rule_str = "(geen regels)"
    if len(df_rules_valid) > 0:
        first = df_rules_valid.iloc[0].to_dict()
        first_rule_str = "\n".join(f"{k}: {v}" for k, v in first.items())

    # Grafieken
    graphs = html.Div([
        dcc.Graph(figure=_figure_conditioneel(df_rules_valid, "Imbal")),
        dcc.Graph(figure=_figure_conditioneel(df_rules_valid, "SI")),
        dcc.Graph(figure=_figure_conditioneel(df_rules_valid, "Power")),
    ])

    # Rapport
    report = "✔️ Geen correcties nodig." if not notes else "⚠️ " + "  •  ".join(notes)

    return df_meta.to_dict("records"), df_cond.to_dict("records"), graphs, report, first_rule_str

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050)
