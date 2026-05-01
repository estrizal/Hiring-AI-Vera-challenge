from layer1_ranker import extract_facts

def make_merchant(mid, langs, name, owner, cat, city):
    identity = {'languages': langs, 'owner_first_name': owner, 'name': name, 'locality': 'l', 'city': city, 'verified': True}
    return {'merchant_id': mid, 'category_slug': cat, 'identity': identity,
            'performance': {}, 'offers': [], 'subscription': {}, 'customer_aggregate': {}, 'signals': []}

meera_m = make_merchant('m_001', ['en', 'hi'], 'Bright Smiles', 'Meera', 'dentists', 'Delhi')
f = extract_facts(meera_m, {}, {'id': 't', 'kind': 'competitor_opened', 'payload': {}}, None)
print('Dr. Meera is_hindi:', f['is_hindi'], '(expected False)')

karthik_m = make_merchant('m_007', ['en', 'hi', 'kn'], 'PowerHouse', 'Karthik', 'gyms', 'Bangalore')
f2 = extract_facts(karthik_m, {}, {'id': 't2', 'kind': 'customer_lapsed_hard', 'payload': {}}, None)
print('Karthik  is_hindi:', f2['is_hindi'], '(expected False)')

# Merchants where Hindi IS primary (e.g. a Jaipur pharmacy with hi-first)
jaipur_m = make_merchant('m_x', ['hi', 'en'], 'Ramesh Medicals', 'Ramesh', 'pharmacies', 'Jaipur')
f3 = extract_facts(jaipur_m, {}, {'id': 't3', 'kind': 'supply_alert', 'payload': {}}, None)
print('Jaipur   is_hindi:', f3['is_hindi'], '(expected True)')

from layer3_composer import _build_customer_brief
brief = _build_customer_brief({
    'trigger_kind': 'customer_lapsed_hard',
    'customer_id': 'c_010_rashmi',
    'trigger_payload': {'days_since_last_visit': 57, 'previous_membership_months': 5, 'previous_focus': 'weight_loss'}
}, 'c_010_rashmi')
print()
print('=== CUSTOMER BRIEF ===')
print(brief)

# Merchant-scope: brief should be empty
brief_merchant = _build_customer_brief({'trigger_kind': 'competitor_opened', 'trigger_payload': {}}, None)
print()
print('Merchant-scope brief empty:', repr(brief_merchant) == repr(''))
