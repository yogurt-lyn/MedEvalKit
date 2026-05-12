from utils import (
    Medbullets_op4,
    Medbullets_op5,
    MedXpertQA,
    MMMU,
    OmniMedVQA,
    PATH_VQA,
    PMC_VQA,
    SLAKE,
    TemporalNeoCXR,
    NeoCXR,
    NeoCXR_EV,
    SuperGPQA,
    VQA_RAD,
    HealthBench,
    PubMedQA,
    MedMCQA,
    MedQA_USMLE,
    MMLU,
    CMB,
    CMExam,
    MedQA_MCMLE,
    CMMLU,
    IU_XRAY,
    CheXpert_Plus,
    MIMIC_CXR,
    MIMIC_CXRfrontal,
    MedFrameQA,
    Radrestruct,
    )


def _dataset_ctor(cls, name: str):
    """Avoid TypeError: None(...) when optional_import failed (e.g. missing nltk)."""
    if cls is None:
        raise ImportError(
            f'Dataset class "{name}" failed to import (got None). '
            "See stderr lines starting with [utils._optional_import]. "
            "Typical fix: use a MedEvalKit-ready conda env (e.g. DQllava / qwencomp), or "
            "`pip install nltk tenacity` in the active environment."
        )
    return cls


#eval_MedQA_USMLE
def prepare_benchmark(model,eval_dataset,eval_dataset_path,eval_output_path):
    # Hallu,Geometry3k
    supported_dataset = ["MMMU-Medical-test","NeoCXR","NeoCXR_EV","MMMU-Medical-val","PATH_VQA","PMC_VQA","VQA_RAD","SLAKE","MedQA_USMLE","MedMCQA","PubMedQA","OmniMedVQA","Medbullets_op4","Medbullets_op5","MedXpertQA-Text","MedXpertQA-MM","SuperGPQA""HealthBench","IU_XRAY","CheXpert_Plus","MIMIC_CXR","MIMIC_CXRfrontal","CMB","CMExam","CMMLU","MedQA_MCMLE","MedFrameQA"]

    if eval_dataset in ["MMMU-Medical-test", "MMMU-Medical-val"]:
        if eval_dataset_path:
            eval_dataset_path = eval_dataset_path.replace(eval_dataset,"MMMU")
        _ , subset , split = eval_dataset.split("-")
        dataset = _dataset_ctor(MMMU, "MMMU")(model, eval_dataset_path, eval_output_path, split, subset)

    elif eval_dataset == "PATH_VQA":
        dataset = _dataset_ctor(PATH_VQA, "PATH_VQA")(model, eval_dataset_path, eval_output_path)

    elif eval_dataset == "NeoCXR":
        dataset = _dataset_ctor(NeoCXR, "NeoCXR")(model, eval_dataset_path, eval_output_path)

    elif eval_dataset == "NeoCXR_EV":
        dataset = _dataset_ctor(NeoCXR_EV, "NeoCXR_EV")(model, eval_dataset_path, eval_output_path)

    elif eval_dataset == "PMC_VQA":
        dataset = _dataset_ctor(PMC_VQA, "PMC_VQA")(model, eval_dataset_path, eval_output_path)

    elif eval_dataset == "VQA_RAD":
        dataset = _dataset_ctor(VQA_RAD, "VQA_RAD")(model, eval_dataset_path, eval_output_path)

    elif eval_dataset == "SLAKE":
        dataset = _dataset_ctor(SLAKE, "SLAKE")(model, eval_dataset_path, eval_output_path)

    elif eval_dataset == "OmniMedVQA":
        dataset = _dataset_ctor(OmniMedVQA, "OmniMedVQA")(model, eval_dataset_path, eval_output_path)

    elif eval_dataset == "Medbullets_op4":
        dataset = _dataset_ctor(Medbullets_op4, "Medbullets_op4")(model, eval_dataset_path, eval_output_path)

    elif eval_dataset == "Medbullets_op5":
        dataset = _dataset_ctor(Medbullets_op5, "Medbullets_op5")(model, eval_dataset_path, eval_output_path)

    elif eval_dataset == "TemporalNeoCXR":
        dataset = _dataset_ctor(TemporalNeoCXR, "TemporalNeoCXR")(model, eval_dataset_path, eval_output_path)

    elif eval_dataset in ["MedXpertQA-Text","MedXpertQA-MM"]:
        if eval_dataset_path:
            eval_dataset_path = eval_dataset_path.replace(eval_dataset,"MedXpertQA")
        _,split = eval_dataset.split("-")
        dataset = _dataset_ctor(MedXpertQA, "MedXpertQA")(model, eval_dataset_path, eval_output_path, split)

    elif eval_dataset == "SuperGPQA":
        dataset = _dataset_ctor(SuperGPQA, "SuperGPQA")(model, eval_dataset_path, eval_output_path)
    
    elif eval_dataset == "HealthBench":
        dataset = _dataset_ctor(HealthBench, "HealthBench")(model, eval_dataset_path, eval_output_path)
    
    elif eval_dataset == "PubMedQA":
        dataset = _dataset_ctor(PubMedQA, "PubMedQA")(model, eval_dataset_path, eval_output_path)

    elif eval_dataset == "MedMCQA":
        dataset = _dataset_ctor(MedMCQA, "MedMCQA")(model, eval_dataset_path, eval_output_path)

    elif eval_dataset == "MedQA_USMLE":
        dataset = _dataset_ctor(MedQA_USMLE, "MedQA_USMLE")(model, eval_dataset_path, eval_output_path)
    
    elif eval_dataset in ["MMLU-medical","MMLU-all"]:
        if eval_dataset_path:
            eval_dataset_path = eval_dataset_path.replace(eval_dataset,"MMLU")
        _,subject = eval_dataset.split("-")
        dataset = _dataset_ctor(MMLU, "MMLU")(model, eval_dataset_path, eval_output_path, subject)
    
    elif eval_dataset == "CMB":
        dataset = _dataset_ctor(CMB, "CMB")(model, eval_dataset_path, eval_output_path)
    
    elif eval_dataset == "CMExam":
        dataset = _dataset_ctor(CMExam, "CMExam")(model, eval_dataset_path, eval_output_path)

    elif eval_dataset == "MedQA_MCMLE":
        dataset = _dataset_ctor(MedQA_MCMLE, "MedQA_MCMLE")(model, eval_dataset_path, eval_output_path)

    elif eval_dataset == "CMMLU":
        dataset = _dataset_ctor(CMMLU, "CMMLU")(model, eval_dataset_path, eval_output_path)
    
    elif eval_dataset == "IU_XRAY":
        dataset = _dataset_ctor(IU_XRAY, "IU_XRAY")(model, eval_dataset_path, eval_output_path)

    elif eval_dataset == "CheXpert_Plus":
        dataset = _dataset_ctor(CheXpert_Plus, "CheXpert_Plus")(model, eval_dataset_path, eval_output_path)

    elif eval_dataset == "MIMIC_CXR":
        dataset = _dataset_ctor(MIMIC_CXR, "MIMIC_CXR")(model, eval_dataset_path, eval_output_path)

    elif eval_dataset == "MIMIC_CXRfrontal":
        dataset = _dataset_ctor(MIMIC_CXRfrontal, "MIMIC_CXRfrontal")(
            model, eval_dataset_path, eval_output_path
        )

    elif eval_dataset == "MedFrameQA":
        dataset = _dataset_ctor(MedFrameQA, "MedFrameQA")(model, eval_dataset_path, eval_output_path)
    elif eval_dataset == "Radrestruct":
        dataset = _dataset_ctor(Radrestruct, "Radrestruct")(model, eval_dataset_path, eval_output_path)
    else:
        print(f"unknown eval dataset {eval_dataset}, we only support {supported_dataset}")
        dataset = None

    return dataset

if __name__ == '__main__':
    pass
