import json
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz

# Cargar modelo SBERT una sola vez
model = SentenceTransformer('all-MiniLM-L6-v2')

# Cargar grupos de marcas desde JSON
def load_brand_groups(file_path="brand_aliases.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    alias_to_group = {}
    for group_leader, aliases in data.items():
        group_set = set(alias.upper() for alias in aliases)
        for alias in aliases:
            alias_to_group[alias.upper()] = group_set
    return alias_to_group

brand_groups = load_brand_groups()

# Cargar JSON
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Preparar catálogo
def get_descriptions(data, source_name):
    result = []
    for item in data:
        make_str = item["make"]["makeString"].strip().upper()
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

# Buscar mejores coincidencias híbridas
def buscar_similares_hibrido(input_data, catalogos):
    marca, submarca, year, modelo, version = input_data
    resultados_por_aseguradora = defaultdict(list)
    input_full = f"{modelo} {version}".strip().upper()
    input_embedding = model.encode(input_full, convert_to_tensor=True)

    # Obtener grupos desde el alias
    grupo_marca = brand_groups.get(marca.upper(), set([marca.upper()]))
    grupo_submarca = brand_groups.get(submarca.upper(), set([submarca.upper()]))

    print(f"[INFO] Marca recibida: {marca} → Grupo detectado: {grupo_marca}")
    print(f"[INFO] Submarca recibida: {submarca} → Grupo detectado: {grupo_submarca}")

    for item in catalogos:
        if item["make"] not in grupo_marca:
            print(f"[DEBUG] ❌ {item['make']} no está en grupo de {marca}")
            continue

        if item["source"].lower() != "mapfre":
            if item["submake"] not in grupo_submarca:
                print(f"[DEBUG] ❌ Submarca {item['submake']} no está en grupo de {submarca}")
                continue

        if str(item["year"]) != year:
            continue

        type_id = item["typeId"]
        desc = item["description"]

        if not (type_id == modelo or desc.startswith(modelo + " ")):
            continue

        # SBERT + Fuzz
        desc_embedding = model.encode(desc, convert_to_tensor=True)
        sbert_score = util.cos_sim(input_embedding, desc_embedding).item() * 100
        fuzz_score = fuzz.token_sort_ratio(input_full, desc)
        combined_score = round((sbert_score + fuzz_score) / 2, 2)

        resultados_por_aseguradora[item["source"]].append({
            "score": combined_score,
            "sbert": round(sbert_score, 2),
            "fuzz": round(fuzz_score, 2),
            "description": desc,
            "year": item["year"],
            "typeId": type_id,
            "makeId": item["makeId"],
            "insuranceCompanyId": item["insuranceCompanyId"]
        })

    # Top 10 por aseguradora
    top_por_aseguradora = {}
    for aseguradora, matches in resultados_por_aseguradora.items():
        top_por_aseguradora[aseguradora] = sorted(matches, key=lambda x: x["score"], reverse=True)[:10]

    return top_por_aseguradora
