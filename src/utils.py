import json

def load_knowledge_base(path='data/car_knowledge.json'):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return {}

def get_car_info(class_name, kb):
    return kb.get(class_name, {
        'reliability_score': 'N/A',
        'common_issues': ['Data not yet available for this model.'],
        'avg_repair_cost_annual': 'N/A',
        'what_to_check': ['Consult a certified mechanic before purchasing.'],
        'typical_price_range': 'N/A',
        'verdict': 'No data available for this model yet.'
    })
