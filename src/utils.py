import json

def load_knowledge_base(path='data/car_knowledge.json'):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

FALLBACK_INFO = {
    'reliability_score': 'N/A',
    'common_issues': ['Data not yet available for this model.'],
    'avg_repair_cost_annual': 'N/A',
    'what_to_check': ['Consult a certified mechanic before purchasing.'],
    'typical_price_range': 'N/A',
    'verdict': 'No data available for this model yet.',
    'reddit_communities': [],
    'real_experiences': ['No owner reports collected for this model yet.'],
}

def get_car_info(class_name, kb):
    """Look up a class name in the knowledge base.

    Returns the stored profile, or a fallback when the model is unknown.
    Older knowledge-base entries may predate the reddit/experience fields,
    so missing keys are backfilled from FALLBACK_INFO.
    """
    info = kb.get(class_name)
    if info is None:
        return dict(FALLBACK_INFO)
    return {**FALLBACK_INFO, **info}
