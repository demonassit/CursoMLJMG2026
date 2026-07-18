
from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import joblib
import json

# ── Cargar modelos al arrancar la app (una sola vez) ─────────
modelo_lr   = joblib.load("modelo_clasificador_lr.pkl")
modelo_perm = joblib.load("modelo_permanencia.pkl")
modelo_sal  = joblib.load("modelo_salario.pkl")

with open("features_clasificador.json") as f: import json; feats_clf = json.load(f)
with open("features_permanencia.json")  as f: feats_perm = json.load(f)
with open("features_salario.json")      as f: feats_sal  = json.load(f)

# ── CSS ──────────────────────────────────────────────────────
css_lines = [
    "    body { background:#F4F6F9; font-family:Segoe UI,Arial,sans-serif; }",
    "    .navbar { background:#1F3864 !important; }",
    "    .navbar-brand, .nav-link { color:white !important; font-weight:500; }",
    "    .pred-box { background:white; border-radius:10px; padding:24px;",
    "               box-shadow:0 2px 8px rgba(0,0,0,.1); text-align:center; }",
    "    .pred-valor { font-size:2.4em; font-weight:bold; }",
    "    .pred-label { font-size:.9em; color:#666; margin-top:6px; }",
]
css = ui.tags.style(chr(10).join(css_lines))

# ── UI ───────────────────────────────────────────────────────
app_ui = ui.page_navbar(css,

    # PESTANA 1 -- CLASIFICADOR
    ui.nav_panel("Predictor de renuncia",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("Perfil del empleado"),
                ui.input_slider("age",       "Edad",              18, 60, 35),
                ui.input_slider("jobsat",    "Satisfaccion (1-4)", 1,  4,  3),
                ui.input_slider("envsat",    "Sat. Ambiente (1-4)",1,  4,  3),
                ui.input_slider("wlb",       "Bal. Vida-Trabajo",  1,  4,  3),
                ui.input_slider("jobinv",    "Involucramiento",    1,  4,  3),
                ui.input_slider("stock",     "Stock Options",      0,  3,  0),
                ui.input_slider("years_co",  "Anos en empresa",    0, 40,  2),
                ui.input_slider("income",    "Salario mensual",  1000,20000,5000,step=500),
                ui.input_slider("prev_cos",  "Empresas previas",  0,  9,  1),
                ui.input_slider("total_yrs", "Anos experiencia",  0, 40,  5),
                ui.input_slider("yrs_mgr",   "Anos con manager",  0, 17,  2),
                ui.input_select("overtime",  "Horas extra",
                                {"No":"No","Yes":"Si (Yes)"}),
                ui.input_select("marital",   "Estado civil",
                                {"Single":"Soltero","Married":"Casado","Divorced":"Divorciado"}),
                ui.input_select("travel",    "Viajes de trabajo",
                                {"Non-Travel":"No viaja",
                                 "Travel_Rarely":"Raramente",
                                 "Travel_Frequently":"Frecuente"}),
            ),
            ui.output_ui("pred_renuncia"),
            ui.output_ui("comparacion_score"),
        )
    ),

    # PESTANA 2 -- PERMANENCIA
    ui.nav_panel("Permanencia estimada",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("Perfil del empleado"),
                ui.input_slider("p_age",     "Edad",              18, 60, 30),
                ui.input_slider("p_joblvl",  "Nivel de puesto",    1,  5,  2),
                ui.input_slider("p_jobsat",  "Satisfaccion (1-4)", 1,  4,  3),
                ui.input_slider("p_wlb",     "Bal. Vida-Trabajo",  1,  4,  3),
                ui.input_slider("p_income",  "Salario mensual", 1000,20000,4000,step=500),
                ui.input_slider("p_numcos",  "Empresas previas",   0,  9,  1),
                ui.input_slider("p_totalyrs","Anos experiencia",   0, 40,  5),
            ),
            ui.output_ui("pred_permanencia"),
        )
    ),

    # PESTANA 3 -- SALARIO
    ui.nav_panel("Salario estimado",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("Perfil del empleado"),
                ui.input_slider("s_joblvl",  "Nivel de puesto",    1,  5,  2),
                ui.input_slider("s_totalyrs","Anos experiencia",   0, 40,  5),
                ui.input_slider("s_educ",    "Educacion (1-5)",    1,  5,  3),
                ui.input_select("s_jobrole", "Puesto",
                    {"0":"Healthcare Rep","1":"Human Resources",
                     "2":"Laboratory Tech","3":"Manager",
                     "4":"Manufacturing Dir","5":"Research Dir",
                     "6":"Research Scientist","7":"Sales Exec","8":"Sales Rep"}),
            ),
            ui.output_ui("pred_salario"),
        )
    ),

    title="TechNova Solutions -- Predictor de RH con ML",
)

# ── SERVER ───────────────────────────────────────────────────
def server(input, output, session):

    @render.ui
    def pred_renuncia():
        le_ot = {"No":0,"Yes":1}
        le_ms = {"Single":2,"Married":1,"Divorced":0}
        le_bt = {"Non-Travel":0,"Travel_Frequently":1,"Travel_Rarely":2}

        fila = pd.DataFrame([[
            input.age(), input.jobsat(), input.envsat(), input.wlb(),
            input.jobinv(), input.stock(), input.years_co(), input.income(),
            input.prev_cos(), input.total_yrs(), input.yrs_mgr(),
            le_ot[input.overtime()], le_ms[input.marital()], le_bt[input.travel()]
        ]], columns=feats_clf)

        prob  = modelo_lr.predict_proba(fila)[0][1]
        pred  = modelo_lr.predict(fila)[0]

        if prob >= 0.70:   color = "#C62828"; nivel = "RIESGO ALTO"
        elif prob >= 0.40: color = "#F57F17"; nivel = "RIESGO MEDIO"
        else:              color = "#2E7D32"; nivel = "RIESGO BAJO"

        # Score manual para comparar
        ot_val  = 100 if input.overtime()=="Yes" else 0
        js_val  = (4 - input.jobsat()) / 3 * 100
        yr_val  = max(0, (3 - input.years_co()) / 3 * 100)
        wlb_val = (4 - input.wlb()) / 3 * 100
        es_val  = (4 - input.envsat()) / 3 * 100
        score   = (ot_val*0.30 + js_val*0.25 + yr_val*0.20 +
                   wlb_val*0.15 + es_val*0.10)

        return ui.div(
            ui.row(
                ui.column(6, ui.div(
                    ui.div(f"{prob*100:.1f} %", class_="pred-valor",
                           style=f"color:{color};"),
                    ui.div(f"Probabilidad de renuncia -- {nivel}", class_="pred-label"),
                    class_="pred-box")),
                ui.column(6, ui.div(
                    ui.div(f"{score:.1f}/100", class_="pred-valor",
                           style="color:#1A5F8A;"),
                    ui.div("Score manual (metodo S5)", class_="pred-label"),
                    class_="pred-box")),
            )
        )

    @render.ui
    def comparacion_score():
        return ui.div(
            ui.p("Columna izquierda: modelo ML (aprendio de los datos historicos). "
                 "Columna derecha: score manual construido en S5 con reglas del analista. "
                 "¿Coinciden? ¿Cual confias mas?",
                 style="color:#666;font-size:13px;margin-top:12px;")
        )

    @render.ui
    def pred_permanencia():
        fila = pd.DataFrame([[
            input.p_age(), input.p_joblvl(), input.p_jobsat(),
            input.p_wlb(), input.p_income(), input.p_numcos(), input.p_totalyrs()
        ]], columns=feats_perm)
        anos = max(0, modelo_perm.predict(fila)[0])
        if anos >= 5:   color = "#2E7D32"
        elif anos >= 2: color = "#F57F17"
        else:           color = "#C62828"
        return ui.div(ui.div(
            ui.div(f"{anos:.1f} anos", class_="pred-valor", style=f"color:{color};"),
            ui.div("Permanencia estimada en la empresa", class_="pred-label"),
            class_="pred-box"), style="max-width:400px;margin:40px auto;")

    @render.ui
    def pred_salario():
        fila = pd.DataFrame([[
            int(input.s_joblvl()), int(input.s_totalyrs()),
            int(input.s_educ()), int(input.s_jobrole())
        ]], columns=feats_sal)
        sal = max(0, modelo_sal.predict(fila)[0])
        return ui.div(ui.div(
            ui.div(f"${sal:,.0f}", class_="pred-valor", style="color:#6A1B9A;"),
            ui.div("Salario mensual estimado (USD)", class_="pred-label"),
            class_="pred-box"), style="max-width:400px;margin:40px auto;")

app = App(app_ui, server)
