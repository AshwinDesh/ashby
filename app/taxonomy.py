from __future__ import annotations


GROUP_TERMS = dict[str, tuple[str, ...]]


# Radiology term inventories used by feature extraction.
# Matchers in app.features require token/phrase boundaries; avoid adding short
# abbreviations unless they are unambiguous in study descriptions.
MODALITY_PREFIXES: GROUP_TERMS = {
    "mri": ("mri", "mr "),
    "ct": ("ct", "cta"),
    "xr": ("xr", "xray", "x-ray", "radiograph"),
    "mam": ("mam", "mammo", "mammography"),
    "us": ("us", "ultrasound"),
    "nm": ("nm", "nuclear"),
    "echo": ("echo", "tte"),
    "dxa": ("dxa",),
    "vas": ("vas", "vascular", "venous"),
    "pet": ("pet",),
}

BODY_REGION_TERMS: GROUP_TERMS = {
    "head_neuro": ("head", "brain", "skull", "neuro", "cerebral"),
    "neck_spine": ("neck", "cervical", "cervicl", "spine", "lumbar", "thoracic"),
    "chest_cardiac": ("chest", "lung", "cardiac", "heart", "coronary", "echo"),
    "breast": ("breast", "mam", "mammo", "mammography"),
    "abdomen_pelvis": (
        "abdomen",
        "abdominal",
        "pelvis",
        "pelvic",
        "renal",
        "kidney",
        "kidneys",
        "bladder",
        "liver",
    ),
    "upper_extremity": ("shoulder", "humerus", "elbow", "wrist", "hand", "finger"),
    "lower_extremity": ("hip", "knee", "ankle", "foot", "femur", "tibia"),
}

LATERALITY_TERMS: GROUP_TERMS = {
    "left": ("left", "lt"),
    "right": ("right", "rt"),
    "bilateral": ("bilat", "bilateral", "bi"),
}

CONTRAST_TERMS: GROUP_TERMS = {
    "with_contrast": ("with contrast", "with cntrst", "w con", "wo/w con", "w/contrast"),
    "without_contrast": ("without contrast", "without cntrst", "wo con", "w/o contrast", "w/o cntrst"),
}

CLINICAL_FAMILY_TERMS: GROUP_TERMS = {
    "breast_imaging": (
        "mam",
        "mammography",
        "mammo",
        "breast",
        "screen",
        "screening",
        "screener",
        "diagnostic",
        "diag",
        "tomo",
        "cad",
        "stereo",
        "biopsy",
        "bx",
        "localization",
        "ultrasound",
        "target",
    ),
    "cardiac_coronary": (
        "echo",
        "tte",
        "myo",
        "myocardial",
        "perfusion",
        "spect",
        "stress",
        "coronary",
        "cardiac",
        "calcium",
        "calc",
    ),
    "vascular_angio": (
        "angio",
        "angiogram",
        "cta",
        "carotid",
        "arterial",
        "venous",
        "vascular",
        "doppler",
    ),
    "pet_oncology": ("pet", "piflu", "skullthigh", "skull", "thigh", "f18"),
    "chest_lung": ("chest", "lung", "lungs", "pulmonary", "thorax", "ribs", "rib"),
    "ribs_thoracic_spine": ("rib", "ribs", "thoracic", "t-spine", "tspine", "sternum"),
    "spine": ("spine", "spinal", "cervical", "cervicl", "thoracic", "lumbar"),
    "head_neck_neuro": (
        "head",
        "brain",
        "stroke",
        "cerebral",
        "neck",
        "carotid",
        "cervical",
        "cervicl",
    ),
    "abdomen_pelvis": (
        "abdomen",
        "abdominal",
        "abd",
        "pelvis",
        "pelvic",
        "pel",
        "enterography",
        "urogram",
    ),
    "renal_urinary": ("renal", "kidney", "kidneys", "bladder", "urogram", "hematuria", "uro"),
    "hepatobiliary": ("liver", "hepatic", "biliary", "cholangiography", "gallbladder"),
    "lower_extremity": ("leg", "hip", "femur", "knee", "ankle", "foot", "pelvis"),
}

CLINICALLY_RELATED_FAMILY_PAIRS = {
    frozenset(("breast_imaging",)),
    frozenset(("cardiac_coronary", "vascular_angio")),
    frozenset(("cardiac_coronary", "chest_lung")),
    frozenset(("pet_oncology", "chest_lung")),
    frozenset(("pet_oncology", "abdomen_pelvis")),
    frozenset(("pet_oncology", "breast_imaging")),
    frozenset(("chest_lung", "ribs_thoracic_spine")),
    frozenset(("chest_lung", "spine")),
    frozenset(("ribs_thoracic_spine", "spine")),
    frozenset(("head_neck_neuro", "vascular_angio")),
    frozenset(("head_neck_neuro", "spine")),
    frozenset(("abdomen_pelvis", "renal_urinary")),
    frozenset(("abdomen_pelvis", "hepatobiliary")),
    frozenset(("abdomen_pelvis", "lower_extremity")),
    frozenset(("renal_urinary", "hepatobiliary")),
    frozenset(("vascular_angio", "lower_extremity")),
}
