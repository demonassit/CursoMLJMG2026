
from shiny import App, ui, render
import pandas as pd
import numpy as np
import joblib
import json

# Cargar modelos al arrancar
km_model  = joblib.load("modelo_kmeans.pkl")
scaler    = joblib.load("scaler_kmeans.pkl")

with open("features_kmeans.json") as f:
    feats = json.load(f)
with open("nombres_clusters.json") as f:
    nombres = json.load(f)

perfiles_info = {
    "0": {"color":"#2E7D32","icono":"Estable","accion":"Monitoreo periodico -- mantener condiciones actuales."},
    "1": {"color":"#C62828","icono":"En riesgo","accion":"Revision inmediata -- conversacion de retencion prioritaria."},
    "2": {"color":"#F57F17","icono":"Transicion","accion":"Seguimiento mensual -- revisar carga laboral y plan de carrera."},
}

css_lines = [
    "    body { background:#F4F6F9; font-family:Segoe UI,Arial,sans-serif; }",
    "    .navbar { background:#1F3864 !important; }",
    "    .navbar-brand,.nav-link { color:white !important; font-weight:500; }",
    "    .seg-box { background:white; border-radius:12px; padding:28px;",
    "               box-shadow:0 2px 10px rgba(0,0,0,.12); text-align:center;",
    "               max-width:500px; margin:30px auto; }",
    "    .seg-perfil { font-size:2em; font-weight:bold; margin-bottom:6px; }",
    "    .seg-label  { font-size:.95em; color:#555; }",
    "    .seg-accion { background:#F4F6F9; border-radius:8px; padding:14px;",
    "                  margin-top:16px; font-size:14px; text-align:left; }",
]
css = ui.tags.style(chr(10).join(css_lines))

app_ui = ui.page_navbar(
    ui.nav_panel("Segmentador de empleados",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("Ingresa el perfil del empleado"),
                ui.input_slider("jobsat", "Satisfaccion laboral (1-4)", 1, 4, 3),
                ui.input_slider("wlb",    "Balance vida-trabajo (1-4)", 1, 4, 3),
                ui.input_slider("envsat", "Sat. del ambiente (1-4)",    1, 4, 3),
                ui.input_slider("jobinv", "Involucramiento (1-4)",      1, 4, 3),
                ui.input_slider("income", "Salario mensual (USD)",
                                1000, 20000, 5000, step=500),
                ui.input_slider("years",  "Anos en la empresa",  0, 40,  3),
                ui.input_slider("age",    "Edad",               18, 60, 30),
                ui.hr(),
                ui.p("El modelo K-Means asignara al empleado al perfil que mas "
                     "se parezca a su segmento natural.",
                     style="font-size:12px;color:#888;"),
            ),
            ui.output_ui("resultado_segmento"),
            ui.output_ui("comparacion_clusters"),
        )
    ),
    ui.nav_panel("Perfil de los 3 segmentos",
        ui.output_ui("descripcion_clusters"),
    ),
    title="TechNova Solutions -- Segmentador de Empleados K-Means",
    header=css,
)

def server(input, output, session):

    @render.ui
    def resultado_segmento():
        fila = pd.DataFrame([[
            input.jobsat(), input.wlb(), input.envsat(), input.jobinv(),
            input.income(), input.years(), input.age()
        ]], columns=feats)
        fila_scaled = scaler.transform(fila)
        cluster_id  = int(km_model.predict(fila_scaled)[0])

        nombre  = nombres.get(str(cluster_id), f"Cluster {cluster_id}")
        info    = perfiles_info.get(str(cluster_id), {})
        color   = info.get("color", "#333")
        accion  = info.get("accion", "Sin recomendacion disponible.")
        icono   = info.get("icono", "")

        # Distancias a los 3 centroides para mostrar certeza
        distancias = km_model.transform(fila_scaled)[0]
        dist_min   = distancias.min()
        certeza    = max(0, (1 - dist_min / distancias.max()) * 100)

        return ui.div(
            ui.div(
                ui.div(nombre, class_="seg-perfil", style=f"color:{color};"),
                ui.div(f"Certeza de asignacion: {certeza:.0f} %", class_="seg-label"),
                ui.div(
                    ui.strong("Recomendacion de accion:"),
                    ui.br(),
                    accion,
                    class_="seg-accion",
                    style=f"border-left:4px solid {color};",
                ),
                class_="seg-box",
                style=f"border-top:6px solid {color};",
            )
        )

    @render.ui
    def comparacion_clusters():
        return ui.div(
            ui.p("Modifica los controles del sidebar para ver como cambia el segmento. "
                 "Intenta un empleado con satisfaccion=1, anos=0 -- "
                 "¿a que perfil lo asigna el modelo?",
                 style="color:#666;font-size:13px;margin-top:8px;max-width:600px;")
        )

    @render.ui
    def descripcion_clusters():
        items = []
        colores = {"0":"#2E7D32","1":"#C62828","2":"#F57F17"}
        descripciones = {
            "0": ("Empleado estable",
                  "Alta satisfaccion laboral y con el ambiente. "
                  "Buen balance vida-trabajo. Lleva varios anos en la empresa. "
                  "Salario por encima del promedio.",
                  "Bajo riesgo de rotacion. Accion: mantener condiciones y "
                  "ofrecer oportunidades de crecimiento para retener a largo plazo."),
            "1": ("Empleado en riesgo",
                  "Baja satisfaccion laboral. Posiblemente hace horas extra. "
                  "Pocos anos en la empresa. Salario por debajo del promedio del nivel.",
                  "Alto riesgo de rotacion. Accion: conversacion de retencion "
                  "inmediata. Revisar condiciones de trabajo y compensacion."),
            "2": ("Empleado de transicion",
                  "Satisfaccion media. Recien llegado o en cambio de rol. "
                  "Aun no consolida pertenencia a la empresa.",
                  "Riesgo medio. Accion: seguimiento mensual, asignar mentor, "
                  "revisar plan de carrera en los primeros 3 anos."),
        }
        for k, (nombre, desc, accion) in descripciones.items():
            color = colores[k]
            items.append(ui.div(
                ui.h4(nombre, style=f"color:{color};border-left:4px solid {color};"
                                    "padding-left:10px;"),
                ui.p(desc),
                ui.div(ui.strong("Accion recomendada: "), accion,
                       style=f"background:#F4F6F9;padding:10px;border-radius:6px;"
                             f"border-left:3px solid {color};font-size:14px;"),
                style="margin-bottom:24px;"
            ))
        return ui.div(*items, style="max-width:700px;margin:20px auto;")

app = App(app_ui, server)
