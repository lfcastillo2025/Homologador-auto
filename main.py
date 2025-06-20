from fastapi import FastAPI, Query
from pydantic import BaseModel
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

class HomologacionRequest(BaseModel):
    marca: str
    submarca: str = ""
    año: int
    modelo: str
    version: str

@app.post("/homologar")
def homologar(data: HomologacionRequest):
    resultados = buscar_similares_hibrido((
        data.marca.upper(),
        data.submarca.upper(),
        str(data.año),
        data.modelo.upper(),
        data.version.upper()
    ), catalogo_total)

    return resultados
