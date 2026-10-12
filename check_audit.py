import json
with open('reports/data_audit.json') as f:
    audit = json.load(f)
print("Checksum:", audit.get('checksum_sha256'))
print("Rows:", audit.get('rows'))
print("Missing:", sum(audit.get('missing_values', {}).values()))
