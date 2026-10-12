"""Import a historical .xls facility directory; never turn facilities into donors.

Usage: python tools/import_blood_banks.py PATH_TO_XLS
Requires xlrd (read-only legacy Excel reader).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def convert(source, output):
    import xlrd
    sheet = xlrd.open_workbook(str(source)).sheet_by_index(0)
    headers = sheet.row_values(0)
    facilities = []
    for index in range(1, sheet.nrows):
        row = dict(zip(headers, sheet.row_values(index)))
        lat, lon = row.get("latitude"), row.get("longitude")
        valid = (type(lat) in (int, float) and type(lon) in (int, float)
                 and math.isfinite(lat) and math.isfinite(lon)
                 and -90 <= lat <= 90 and -180 <= lon <= 180 and (lat != 0 or lon != 0))
        facilities.append({"facility_id": f"BB2015_{index:04d}", "name": str(row.get("h_name", "")),
                           **{key: str(row.get(key, "")) for key in ("state", "city", "district", "address", "pincode")},
                           "latitude": lat if valid else None, "longitude": lon if valid else None,
                           "coordinates_verified": False, "source_date": "2015-09"})
    payload = {"source_file": source.name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "source_date_basis": "Provided filename; not independently verified", "facilities": facilities,
               "records": len(facilities), "usable_coordinate_pairs": sum(f["latitude"] is not None for f in facilities)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "facilities"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/blood_bank_directory.json"))
    args = parser.parse_args()
    convert(args.source, args.output)
