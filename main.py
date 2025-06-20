from fastapi import FastAPI, Query
from typing import List
from homologador import buscar_similares_hibrido, get_descriptions, load_json
from pathlib import Path

app = FastAPI()

# Cargar catálogos al iniciar
path_base = Path(".")
chubb = get_descriptions(load_json(path_base / "chubb-data.json"), "chubb")
hdi = get_descriptions(load_json(path_base / "hdi-data.json"), "hdi")
mapfre = get_descriptions(load_json(path_base / "mapfre-data.json"), "mapfre")
catalogo_total = chubb + hdi + mapfre

@app.get("/homologar")
def homologar(
    year: int = Query(...),
    make: str = Query(...),
    submake: str = Query(...),
    model: str = Query(...),
    version: str = Query(...)
):
    resultados = buscar_similares_hibrido((
        make.upper(),
        submake.upper(),
        str(year),
        model.upper(),
        version.upper()
    ), catalogo_total)

    # Formatear resultado con modelString, aseguradora y compañía
    salida = []
    for aseguradora, modelos in resultados.items():
        for modelo in modelos:
            salida.append({
                "modelString": modelo["description"],
                "aseguradora": aseguradora,
                "insuranceCompanyId": modelo["insuranceCompanyId"]
            })

    return salida
