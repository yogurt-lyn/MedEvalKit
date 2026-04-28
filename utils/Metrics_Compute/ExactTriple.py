import re
from typing import List, Dict

def map_P_en(text: str) -> str:
    t = text.lower()

    if "nrds" in t:
        return "NRDS"

    elif any(k in t for k in [
        "pneumonia",
        "inflammation",   
        "inflammatory",
        "inflamed",  
        "infection",
        "neonatal pneumonia"      
    ]):
        return "neonatal pneumonia"

    elif any(k in t for k in [
        "bronchopulmonary dysplasia",
        "bpd",
        "hypoplasia",              
        "hypoplastic",
        "dysplasia"
    ]):
        return "bronchopulmonary dysplasia"

    elif any(k in t for k in [
        "consolidation", "consolidations","consolidate","consolidated"
    ]):
        return "lung consolidation"
    

    elif any(k in t for k in [
        "pneumothorax"
    ]):
        return "pneumothorax"

    elif ("texture" in t or "marking" in t) and not any(k in t for k in [
        "pneumonia", "inflammation", "inflamed", "consolidation",
        "infiltrate", "exudate", "exudation", "exudative", "opacity",
        "bronchopulmonary dysplasia", "bpd", "hypoplasia", "nrds","trachea dysplasia"
    ]):
        return "lung texture abnormal"

    elif any(k in t for k in [
        "infiltrate", "infiltrates", "infiltration",
        "opacity", "opacities",
        "exudate", "exudation", "exudative",  # 新增 exudative
        "hazy", "patchy", "diffuse", "scattered",
        "infection","exudes",
        "wet lung","exuding"
    ]):
        return "pulmonary exudation"

    return "other"

def map_D_en(text: str) -> str:
    t = text.lower()
    if any(re.search(rf"\b{kw}\b", t) for kw in ["similar",
        "similar to","not significantly changed","no improvement","no obvious","not obvious","no significant change","no significant","not significantly"
        ]):
        return "similar" 
    elif any(re.search(rf"\b{kw}\b", t) for kw in [
        "better",
        "decreased",
        "lighter",
        "less obvious",
        "improve",
        "improved",
        "improvement",
        "absorbed",
        "little than",
        "little exudation than",
        "smaller",
        "lower than",
        "less",
        "absorbs",
        "less obvious",
        "reduced",
        "absorb",
        "absorption with improvement",
    ]) or re.search(r"(?<!\bno\s)(?<!\bnot\s)\b(absorption|absorbed)\b", t):
        return "improved"
    elif any(re.search(rf"\b{kw}\b", t) for kw in ["advanced",
        "increased",
        "added",
        "progressive",
        "worse than",
        "progressed",
        "more obvious",
        "progressing",
        "progressed",
        "Increased",
        "new than",
        "more than",
        "progress",
        "exudate slightly more",]):
        return "progress"
    else:
        return ""

def map_S_en(text: str) -> str:
    t = text.lower()


    if any(
        re.search(rf"\b{kw}\b", t) and not (
            re.search(rf"(?:no|not|n't)\s+(?:\w+\s+){{0,3}}\b{kw}\b", t) or
            re.search(rf"\b{kw}\b(?:\s+\w+){{0,3}}\s+(?:no|not|n't)", t)
        )
        for kw in ["significantly", "significant", "obviously"]
        ):
        return "severe"
    elif any(re.search(rf"\b{kw}\b", t) for kw in ["slightly","slight"]):
        return "mild"
    elif any(re.search(rf"\b{kw}\b", t) for kw in ["some","somewhat","roughly", "partially","basically"]):
        return "moderate"
    else:
        return ""



def extract_triplets(response: str) -> List[Dict[str, str]]:
    triplets = []

    concl_match = re.search(
        r'Image conclusion[:\s]*\n(.*?)(\n\s*\n|\n\s*Disease diagnosis|$)',
        response,
        re.DOTALL | re.IGNORECASE
    )
    if not concl_match:
        return triplets, ""

    conclusion = concl_match.group(1).strip()

    line1_match = re.search(r'1\.\s*(.*?)(\n\s*2\.|$)', conclusion, re.DOTALL)
    if not line1_match:
        return triplets, ""

    line1 = line1_match.group(1).strip()

    clauses = re.split(r'[;.\n]+', line1)
    clauses = [c.strip() for c in clauses if c.strip()]

    temporal_kws = {
        "previous", "prior", "compared", "than",
        "before", "better", "worse", "similar", "unchanged","added"
    }

    candidate_clause = None
    D = ""
    S = ""

    for clause in clauses:
        clause_lower = clause.lower()
        if any(kw in clause_lower for kw in temporal_kws):
            D_candidate = map_D_en(clause)
            if D_candidate:
                candidate_clause = clause
                break

    if not candidate_clause:
        return triplets, ""

    # P = map_P_en(candidate_clause)
    cut_pos = None
    candidate_lower = candidate_clause.lower()
    for kw in temporal_kws:
        pos = candidate_lower.find(kw)
        if pos != -1:
            cut_pos = pos if cut_pos is None else min(cut_pos, pos)

    P_text = candidate_clause[:cut_pos].strip() if cut_pos is not None else candidate_clause
    P = map_P_en(P_text)


    sub_clauses = [sc.strip() for sc in re.split(r',', candidate_clause) if sc.strip()]

    for sc in sub_clauses:
        sc_lower = sc.lower()
        if not any(kw in sc_lower for kw in temporal_kws):
            continue

        if not D:
            D = map_D_en(sc)
        if not S:
            S = map_S_en(sc)

        if D and S:
            break

    triplets.append({
        "P": P,
        "D": D,
        "S": S
    })
    # print(triplets.shape)

    return triplets
