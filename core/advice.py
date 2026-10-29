"""General crop-protection guidance per condition.

This is generic integrated-pest-management guidance, NOT a prescription.
Chemical choices, doses and legal limits differ by crop, state and season, so
the UI always tells the farmer to confirm with a local agriculture office
(e.g. a Krishi Vigyan Kendra) before spraying.
"""

# condition keyword -> (cause type, what it is, actions)
_RULES = [
    ("healthy", "healthy", "No visible disease symptoms on this leaf.",
     ["Keep scouting every few days, especially after rain or humid spells.",
      "Maintain balanced fertiliser and even irrigation to keep plants vigorous."]),
    ("Apple_scab", "fungal", "Olive-brown scabby spots on leaves and fruit, favoured by cool wet spring weather.",
     ["Rake and destroy fallen leaves in autumn.", "Prune for airflow so foliage dries quickly.",
      "Use a protectant fungicide on the schedule advised locally."]),
    ("Black_rot", "fungal", "Dark rotting spots with a ring pattern on leaves and fruit; overwinters in mummified fruit and cankers.",
     ["Remove mummified fruit and infected wood.", "Open the canopy for airflow and drying.",
      "Time fungicide sprays to early growth stages as advised locally."]),
    ("Cedar_apple_rust", "fungal", "Bright yellow-orange leaf spots; the fungus alternates between apple and juniper/cedar.",
     ["Remove nearby juniper/cedar hosts where practical.", "Plant resistant varieties.", "Apply protectant fungicide from pink bud as advised locally."]),
    ("Powdery_mildew", "fungal", "White powdery fungal growth on leaf surfaces; thrives in warm days, cool humid nights.",
     ["Improve spacing and airflow.", "Avoid excess nitrogen.", "Use sulphur-based or other mildew fungicides as advised locally."]),
    ("Cercospora_leaf_spot", "fungal", "Rectangular grey-tan lesions between leaf veins (grey leaf spot) in humid weather.",
     ["Rotate away from maize for a season or more.", "Bury or manage crop residue.", "Choose tolerant hybrids; fungicide only if pressure is high."]),
    ("Common_rust", "fungal", "Small brick-red pustules on both leaf surfaces; spreads by wind-blown spores.",
     ["Plant resistant hybrids.", "Scout early; severe cases at an early stage may justify a fungicide."]),
    ("Northern_Leaf_Blight", "fungal", "Long cigar-shaped grey-green lesions that can merge and blight the leaf.",
     ["Rotate crops and bury residue.", "Plant resistant hybrids.", "Consider fungicide if lesions reach upper leaves early."]),
    ("Esca", "fungal", "Interveinal 'tiger-stripe' leaf scorch and wood decay (black measles).",
     ["There is no cure: remove and burn badly affected vines or wood.", "Protect pruning wounds and disinfect tools."]),
    ("Leaf_blight", "fungal", "Irregular brown leaf blotches that grow and merge, often late in the season.",
     ["Remove infected leaves.", "Manage the canopy for airflow.", "Apply recommended fungicide if spread continues."]),
    ("Haunglongbing", "bacterial", "Citrus greening: blotchy yellow leaves and misshapen bitter fruit, spread by the Asian citrus psyllid. Incurable.",
     ["Remove infected trees as advised by authorities and report suspected cases.", "Control psyllids.", "Plant only certified disease-free nursery stock."]),
    ("Bacterial_spot", "bacterial", "Small dark water-soaked spots, often with a yellow halo; spreads in splashing rain and on infected seed.",
     ["Use clean seed or transplants and rotate crops.", "Avoid overhead irrigation and handling wet plants.",
      "Copper-based sprays slow spread; follow local advice."]),
    ("Early_blight", "fungal", "Brown spots with concentric 'target' rings, usually on older leaves first.",
     ["Remove lower infected leaves; mulch to stop soil splash.", "Rotate crops and keep plants well fed.", "Apply recommended fungicide if it keeps spreading."]),
    ("Late_blight", "oomycete", "Water-soaked grey-green patches that turn brown, with white growth underneath in humid weather. Can destroy a field in days.",
     ["Act quickly: remove and destroy infected plants.", "Avoid wetting foliage; improve drainage and airflow.",
      "Use protectant/systemic fungicides on the local schedule and certified seed."]),
    ("Leaf_Mold", "fungal", "Pale yellow patches on upper leaf surface with olive-grey mould underneath; typical in humid greenhouses.",
     ["Lower humidity and ventilate.", "Space plants and remove lower leaves.", "Choose resistant varieties."]),
    ("Septoria_leaf_spot", "fungal", "Many small round spots with dark borders and grey centres, starting on lower leaves.",
     ["Remove infected leaves and mulch.", "Rotate for at least two seasons.", "Fungicide if pressure continues."]),
    ("Spider_mites", "pest", "Tiny mites cause fine speckling, bronzing and webbing, worst in hot dry weather.",
     ["Check leaf undersides for mites and webbing.", "Rinse plants and reduce dust and drought stress.",
      "Use predatory mites or a recommended miticide if numbers build up."]),
    ("Target_Spot", "fungal", "Brown spots with rings and yellow halos on leaves and fruit.",
     ["Remove infected debris.", "Space plants for airflow.", "Rotate crops; fungicide per local advice."]),
    ("Yellow_Leaf_Curl_Virus", "viral", "Upward-curling, yellowing, stunted leaves; spread by whiteflies. No cure.",
     ["Control whiteflies (nets, reflective mulch, recommended insecticides).", "Remove and destroy infected plants.", "Plant tolerant varieties."]),
    ("mosaic_virus", "viral", "Mottled light and dark green leaves, distortion; spreads on hands, tools and seed. No cure.",
     ["Remove infected plants and do not compost them.", "Wash hands and disinfect tools.", "Use certified seed and resistant varieties."]),
    ("Leaf_scorch", "fungal", "Purple-red blotches that dry into scorched brown patches on leaves.",
     ["Remove old infected leaves after harvest.", "Avoid overhead irrigation.", "Use resistant cultivars and fungicide per local advice."]),
]

TYPE_LABEL = {
    "healthy": "Healthy", "fungal": "Fungal disease", "bacterial": "Bacterial disease",
    "viral": "Viral disease", "oomycete": "Water-mould disease", "pest": "Pest damage",
}


def split_label(label: str):
    """'Tomato___Late_blight' -> ('Tomato', 'Late blight')"""
    crop, _, cond = label.partition("___")
    crop = crop.replace(",_", ", ").replace("_", " ").replace("(including sour)", "").strip()
    crop = crop.replace("Pepper, bell", "Bell pepper").replace("Corn (maize)", "Maize (corn)")
    cond = cond.replace("_", " ").strip()
    cond = {"Cercospora leaf spot Gray leaf spot": "Grey leaf spot (Cercospora)",
            "Spider mites Two-spotted spider mite": "Spider mites (two-spotted)",
            "Haunglongbing (Citrus greening)": "Citrus greening (Huanglongbing)",
            "Esca (Black Measles)": "Esca (black measles)",
            "Leaf blight (Isariopsis Leaf Spot)": "Leaf blight (Isariopsis leaf spot)",
            "Tomato Yellow Leaf Curl Virus": "Yellow leaf curl virus",
            "Tomato mosaic virus": "Mosaic virus"}.get(cond, cond)
    return crop, cond[:1].upper() + cond[1:]


def lookup(label: str) -> dict:
    crop, cond = split_label(label)
    for key, kind, what, actions in _RULES:
        if key.lower() in label.lower():
            return {"crop": crop, "condition": cond, "kind": kind, "kind_label": TYPE_LABEL[kind],
                    "summary": what, "actions": actions, "healthy": kind == "healthy"}
    return {"crop": crop, "condition": cond, "kind": "unknown", "kind_label": "Unclassified",
            "summary": "", "actions": [], "healthy": False}
