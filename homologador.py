import json
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz

# Cargar modelo SBERT una sola vez
model = SentenceTransformer('all-MiniLM-L6-v2')

# Cargar JSON
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Preparar catálogo
def get_descriptions(data, source_name):
    result = []
    for item in data:
        make_str = item["make"]["makeString"].strip().upper()

        # Submarca: si viene vacía, asumimos que es igual a la marca
        submake_raw = item["make"].get("submake", "").strip().upper()
        submake_final = submake_raw if submake_raw else make_str

        try:
            description = json.loads(item["extraData"])["description"] if item["extraData"] != "null" else item["model"]["modelString"]
        except:
            description = item["model"]["modelString"]

        result.append({
            "source": source_name,
            "insuranceCompanyId": item["insuranceCompanyId"],
            "make": make_str,
            "submake": submake_final,
            "year": item["year"],
            "typeId": item.get("typeId", "").strip().upper(),
            "makeId": item["make"].get("makeId", ""),
            "description": description.strip().upper()
        })
    return result

# Entrada del usuario
def get_user_input():
    marca = input("Marca: ").strip().upper()
    submarca = input("Submarca (puede estar vacía): ").strip().upper()
    year = input("Año: ").strip()
    modelo = input("Modelo: ").strip().upper()
    version = input("Versión/Descripción: ").strip().upper()
    return marca, submarca, year, modelo, version

# Buscar mejores coincidencias híbridas
def buscar_similares_hibrido(input_data, catalogos):
    marca, submarca, year, modelo, version = input_data
    resultados_por_aseguradora = defaultdict(list)
    input_full = f"{modelo} {version}".strip().upper()
    input_embedding = model.encode(input_full, convert_to_tensor=True)

    for item in catalogos:
        if item["make"] != marca or (submarca and item["submake"] != submarca) or str(item["year"]) != year:
            continue

        type_id = item["typeId"]
        desc = item["description"]

        # Filtro estricto: el modelo debe coincidir exactamente
        if not (type_id == modelo or desc.startswith(modelo + " ")):
            continue

        # SBERT score
        desc_embedding = model.encode(desc, convert_to_tensor=True)
        sbert_score = util.cos_sim(input_embedding, desc_embedding).item() * 100
        
        # Fuzz score
        fuzz_score = fuzz.token_sort_ratio(input_full, desc)

        # Score combinado
        combined_score = round((sbert_score + fuzz_score) / 2, 2)

        resultados_por_aseguradora[item["source"]].append({
            "score": combined_score,
            "sbert": round(sbert_score, 2),
            "fuzz": round(fuzz_score, 2),
            "description": desc,
            "year": item["year"],
            "typeId": type_id,
            "makeId": item["makeId"]
        })

    # Top 10 por aseguradora
    top_por_aseguradora = {}
    for aseguradora, matches in resultados_por_aseguradora.items():
        top_por_aseguradora[aseguradora] = sorted(matches, key=lambda x: x["score"], reverse=True)[:10]

    return top_por_aseguradora

# MAIN
if __name__ == "__main__":
    path_base = Path(".")
    chubb = get_descriptions(load_json(path_base / "chubb-data.json"), "chubb")
    hdi = get_descriptions(load_json(path_base / "hdi-data.json"), "hdi")
    mapfre = get_descriptions(load_json(path_base / "mapfre-data.json"), "mapfre")

    print(f"CHUBB: {len(chubb)} registros")
    print(f"HDI: {len(hdi)} registros")
    print(f"MAPFRE: {len(mapfre)} registros")

    catalogo_total = chubb + hdi + mapfre

    marca, submarca, año, modelo, version = get_user_input()
    resultados = buscar_similares_hibrido((marca, submarca, año, modelo, version), catalogo_total)

    print("\n🔍 Top 10 coincidencias por aseguradora (SBERT + Fuzz):")
    for aseguradora, matches in resultados.items():
        print(f"\n🛡️ Aseguradora: {aseguradora.upper()}")
        for match in matches:
            print(f"  🟢 Score combinado: {match['score']}%  | SBERT: {match['sbert']}%  | Fuzz: {match['fuzz']}%")
            print(f"  🔸 Año: {match['year']}")
            print(f"  🧾 TypeID: {match['typeId']}  |  MakeID: {match['makeId']}")
            print(f"  📄 Descripción: {match['description']}\n")
