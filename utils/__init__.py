import os
import sys
import traceback
from importlib import import_module


# 设 OPTIONAL_IMPORT_VERBOSE=0 可关闭异常打印；默认开启，便于在
# benchmarks.py 报 "NoneType object is not callable" 时定位真实原因。
_VERBOSE = os.environ.get("OPTIONAL_IMPORT_VERBOSE", "1").lower() not in ("0", "false", "no")


def _optional_import(module_path, symbol):
    try:
        module = import_module(module_path, package=__name__)
        return getattr(module, symbol)
    except Exception as exc:
        if _VERBOSE:
            print(
                f"[utils._optional_import] failed to import {symbol} from "
                f"{__name__}{module_path}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc()
        return None


Medbullets_op4 = _optional_import(".Medbullets_op4.Medbullets_op4", "Medbullets_op4")
Medbullets_op5 = _optional_import(".Medbullets_op5.Medbullets_op5", "Medbullets_op5")
MedXpertQA = _optional_import(".MedXpertQA.MedXpertQA", "MedXpertQA")
MMMU = _optional_import(".MMMU.MMMU", "MMMU")
OmniMedVQA = _optional_import(".OmniMedVQA.OmniMedVQA", "OmniMedVQA")
PATH_VQA = _optional_import(".PATH_VQA.PATH_VQA", "PATH_VQA")
PMC_VQA = _optional_import(".PMC_VQA.PMC_VQA", "PMC_VQA")
SLAKE = _optional_import(".SLAKE.SLAKE", "SLAKE")
SuperGPQA = _optional_import(".SuperGPQA.SuperGPQA", "SuperGPQA")
VQA_RAD = _optional_import(".VQA_RAD.VQA_RAD", "VQA_RAD")
HealthBench = _optional_import(".HealthBench.HealthBench", "HealthBench")
PubMedQA = _optional_import(".PubMedQA.PubMedQA", "PubMedQA")
MedMCQA = _optional_import(".MedMCQA.MedMCQA", "MedMCQA")
MedQA_USMLE = _optional_import(".MedQA_USMLE.MedQA_USMLE", "MedQA_USMLE")
MMLU = _optional_import(".MMLU.MMLU", "MMLU")
CMB = _optional_import(".CMB.CMB", "CMB")
CMExam = _optional_import(".CMExam.CMExam", "CMExam")
MedQA_MCMLE = _optional_import(".MedQA_MCMLE.MedQA_MCMLE", "MedQA_MCMLE")
CMMLU = _optional_import(".CMMLU.CMMLU", "CMMLU")
IU_XRAY = _optional_import(".IU_XRAY.IU_XRAY", "IU_XRAY")
NeoCXR = _optional_import(".NeoCXR.NeoCXR", "NeoCXR")
NeoCXR_EV = _optional_import(".NeoCXR_EV.NeoCXR_EV", "NeoCXR_EV")
TemporalNeoCXR = _optional_import(".TNeoCXR.TemporalNeoCXR", "TemporalNeoCXR")
CheXpert_Plus = _optional_import(".CheXpert_Plus.CheXpert_Plus", "CheXpert_Plus")
MIMIC_CXR = _optional_import(".MIMIC_CXR.MIMIC_CXR", "MIMIC_CXR")
MIMIC_CXRfrontal = _optional_import(".MIMIC_CXRfrontal.MIMIC_CXRfrontal", "MIMIC_CXRfrontal")
MedFrameQA = _optional_import(".MedFrameQA.MedFrameQA", "MedFrameQA")
Radrestruct = _optional_import(".Radrestruct.Radrestruct", "Radrestruct")