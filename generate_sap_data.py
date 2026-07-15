"""
SAP SD Module - Dummy Data Generator v3
Mirrors real SAP table structures: KNA1, KNVV, VBAK, VBAP, LIKP, LIPS, VBRK, VBRP
Added: cost price, margins, fiscal years FY22-FY25, area managers, regions
Enables all 6 benchmark questions across 4 fiscal years
"""

import json
import random
import os
from datetime import datetime, timedelta
from faker import Faker

fake = Faker("en_IN")
random.seed(42)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
NUM_CUSTOMERS  = 30
NUM_ORDERS     = 240   # 60 per FY: FY22, FY23, FY24, FY25
NUM_DELIVERIES = 160
NUM_BILLINGS   = 140

# Indian Fiscal Year date ranges (Apr 1 - Mar 31)
FISCAL_YEARS = {
    "FY2022-23": (datetime(2022, 4, 1), datetime(2023, 3, 31)),
    "FY2023-24": (datetime(2023, 4, 1), datetime(2024, 3, 31)),
    "FY2024-25": (datetime(2024, 4, 1), datetime(2025, 3, 31)),
    "FY2025-26": (datetime(2025, 4, 1), datetime(2026, 3, 15)),  # capped before today
}
FY_LABELS = list(FISCAL_YEARS.keys())

# ─── MASTER DATA ─────────────────────────────────────────────────────────────
SALES_ORGS      = ["1000", "2000", "3000"]
DIST_CHANNELS   = ["10", "20", "30"]
DIVISIONS       = ["01", "02", "03"]
PLANTS          = ["1001", "1002", "2001", "3001"]
SHIPPING_POINTS = ["SP01", "SP02", "SP03"]
PAYMENT_TERMS   = ["Z001", "Z002", "Z030", "Z060"]
INCOTERMS       = ["CIF", "FOB", "EXW", "DAP"]

AREA_MANAGERS = {
    "AM-001": {"name": "Rajesh Sharma",  "region": "North", "territory": "Delhi-NCR"},
    "AM-002": {"name": "Priya Mehta",    "region": "West",  "territory": "Mumbai"},
    "AM-003": {"name": "Suresh Nair",    "region": "South", "territory": "Bangalore"},
    "AM-004": {"name": "Anita Gupta",    "region": "East",  "territory": "Kolkata"},
    "AM-005": {"name": "Vikram Patel",   "region": "West",  "territory": "Ahmedabad"},
    "AM-006": {"name": "Deepika Reddy",  "region": "South", "territory": "Hyderabad"},
}

CITIES_BY_REGION = {
    "North": ["Delhi", "Jaipur", "Chandigarh", "Lucknow"],
    "West":  ["Mumbai", "Pune", "Ahmedabad", "Surat"],
    "South": ["Bangalore", "Chennai", "Hyderabad", "Kochi"],
    "East":  ["Kolkata", "Bhubaneswar", "Patna"],
}

# Materials with sale price AND cost price for margin calculation
MATERIALS = [
    {"matnr": "MAT-1001", "maktx": "Industrial Motor 5HP",        "meins": "EA",  "price": 15000,  "cost": 9500},
    {"matnr": "MAT-1002", "maktx": "Control Panel Unit",          "meins": "EA",  "price": 45000,  "cost": 31500},
    {"matnr": "MAT-1003", "maktx": "Hydraulic Pump 20L",          "meins": "EA",  "price": 28000,  "cost": 19600},
    {"matnr": "MAT-1004", "maktx": "PLC Controller Siemens S7",   "meins": "EA",  "price": 62000,  "cost": 46500},
    {"matnr": "MAT-1005", "maktx": "Steel Pipe 50mm x 6m",        "meins": "MTR", "price": 850,    "cost": 680},
    {"matnr": "MAT-1006", "maktx": "Bearing SKF 6205",            "meins": "EA",  "price": 320,    "cost": 210},
    {"matnr": "MAT-1007", "maktx": "Electrical Cable 4mm 100m",   "meins": "RL",  "price": 4200,   "cost": 3150},
    {"matnr": "MAT-1008", "maktx": "Compressor 10HP Atlas Copco", "meins": "EA",  "price": 95000,  "cost": 71250},
    {"matnr": "MAT-1009", "maktx": "Valve Gate 2 inch SS",        "meins": "EA",  "price": 1800,   "cost": 1260},
    {"matnr": "MAT-1010", "maktx": "Heat Exchanger Shell Tube",   "meins": "EA",  "price": 120000, "cost": 96000},
    {"matnr": "MAT-1011", "maktx": "Sensor Pressure 0-10 bar",    "meins": "EA",  "price": 2400,   "cost": 1680},
    {"matnr": "MAT-1012", "maktx": "Conveyor Belt 500mm x 10m",   "meins": "EA",  "price": 18000,  "cost": 13500},
]

ORDER_STATUSES    = ["A", "B", "C"]
DELIVERY_STATUSES = ["A", "B", "C"]
BILLING_STATUSES  = ["A", "B", "C"]
DOCUMENT_TYPES    = ["OR", "ZOR", "RE", "CR", "DR"]

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def get_fiscal_year(dt):
    """Indian FY: Apr-Mar. Returns e.g. FY2022-23"""
    if dt.month >= 4:
        return f"FY{dt.year}-{str(dt.year+1)[-2:]}"
    else:
        return f"FY{dt.year-1}-{str(dt.year)[-2:]}"

def rand_date_in_fy(fy_label):
    start, end = FISCAL_YEARS[fy_label]
    end = min(end, datetime.now() - timedelta(days=1))
    return start + (end - start) * random.random()

def rand_date(start_days_ago=365, end_days_ago=0):
    start = datetime.now() - timedelta(days=start_days_ago)
    end   = datetime.now() - timedelta(days=end_days_ago)
    return start + (end - start) * random.random()

def fmt_date(dt): return dt.strftime("%Y-%m-%d")
def fmt_ts(dt):   return dt.strftime("%Y-%m-%d %H:%M:%S")

# ─── KNA1: Customer Master ────────────────────────────────────────────────────
def generate_kna1(n):
    customers = []
    am_list   = list(AREA_MANAGERS.keys())
    for i in range(1, n + 1):
        am_code = random.choice(am_list)
        am_info = AREA_MANAGERS[am_code]
        region  = am_info["region"]
        city    = random.choice(CITIES_BY_REGION[region])
        customers.append({
            "KUNNR":       f"C{str(i).zfill(6)}",
            "NAME1":       fake.company(),
            "NAME2":       fake.company_suffix(),
            "STRAS":       fake.street_address(),
            "ORT01":       city,
            "PSTLZ":       fake.postcode(),
            "LAND1":       "IN",
            "TELF1":       fake.phone_number(),
            "STCEG":       f"GST{fake.numerify('##AAAAA####A#Z#')}",
            "ERDAT":       fmt_date(rand_date(1800, 730)),
            "KTOKD":       "Z001",
            "AREA_MGR_ID": am_code,
            "AREA_MGR":    am_info["name"],
            "REGION":      region,
            "TERRITORY":   am_info["territory"],
        })
    return customers

# ─── KNVV: Customer Sales Data ───────────────────────────────────────────────
def generate_knvv(kna1_records):
    return [{
        "KUNNR":  c["KUNNR"],
        "VKORG":  random.choice(SALES_ORGS),
        "VTWEG":  random.choice(DIST_CHANNELS),
        "SPART":  random.choice(DIVISIONS),
        "BZIRK":  f"ZN{random.randint(1,5):02d}",
        "KDGRP":  random.choice(["01","02","03"]),
        "ZTERM":  random.choice(PAYMENT_TERMS),
        "INCO1":  random.choice(INCOTERMS),
        "WAERS":  "INR",
        "PLTYP":  "PR00",
    } for c in kna1_records]

# ─── VBAK: Sales Order Header ────────────────────────────────────────────────
def generate_vbak(n, customers):
    orders      = []
    orders_per_fy = n // len(FY_LABELS)   # 60 per FY
    order_id    = 1

    for fy_label in FY_LABELS:
        for _ in range(orders_per_fy):
            cust     = random.choice(customers)
            created  = rand_date_in_fy(fy_label)
            req_date = created + timedelta(days=random.randint(7, 60))
            orders.append({
                "VBELN":       f"SO{str(order_id).zfill(8)}",
                "ERDAT":       fmt_date(created),
                "ERZET":       fmt_ts(created),
                "AUART":       random.choice(DOCUMENT_TYPES),
                "KUNNR":       cust["KUNNR"],
                "NAME1":       cust["NAME1"],
                "VKORG":       random.choice(SALES_ORGS),
                "VTWEG":       random.choice(DIST_CHANNELS),
                "SPART":       random.choice(DIVISIONS),
                "WAERS":       "INR",
                "NETWR":       0,
                "MWSBP":       0,
                "COST_TOTAL":  0,
                "MARGIN_PCT":  0,
                "VDATU":       fmt_date(req_date),
                "ZTERM":       random.choice(PAYMENT_TERMS),
                "INCO1":       random.choice(INCOTERMS),
                "GBSTK":       random.choice(ORDER_STATUSES),
                "LIFSK":       "" if random.random() > 0.1 else "Z1",
                "FAKSK":       "" if random.random() > 0.1 else "Z1",
                "FISCAL_YEAR": fy_label,
                "AREA_MGR":    cust.get("AREA_MGR", ""),
                "AREA_MGR_ID": cust.get("AREA_MGR_ID", ""),
                "REGION":      cust.get("REGION", ""),
            })
            order_id += 1

    return orders

# ─── VBAP: Sales Order Items ──────────────────────────────────────────────────
def generate_vbap(orders):
    items = []
    for order in orders:
        chosen     = random.sample(MATERIALS, random.randint(1, 5))
        total_net  = 0
        total_cost = 0
        for idx, mat in enumerate(chosen, start=10):
            qty        = random.randint(1, 50)
            sale_price = mat["price"] * random.uniform(0.9, 1.15)
            cost_price = mat["cost"]  * random.uniform(0.95, 1.05)
            net        = round(qty * sale_price, 2)
            cost_total = round(qty * cost_price, 2)
            margin_pct = round((sale_price - cost_price) / sale_price * 100, 2)
            total_net  += net
            total_cost += cost_total
            items.append({
                "VBELN":       order["VBELN"],
                "POSNR":       str(idx * 10).zfill(6),
                "MATNR":       mat["matnr"],
                "ARKTX":       mat["maktx"],
                "KWMENG":      qty,
                "MEINS":       mat["meins"],
                "NETPR":       round(sale_price, 2),
                "NETWR":       net,
                "COST_PRICE":  round(cost_price, 2),
                "COST_TOTAL":  cost_total,
                "MARGIN_PCT":  margin_pct,
                "WERKS":       random.choice(PLANTS),
                "LGORT":       f"000{random.randint(1,4)}",
                "PSTYV":       "TAN",
                "ABGRU":       "",
                "LFSTA":       random.choice(DELIVERY_STATUSES),
                "FKSTA":       random.choice(BILLING_STATUSES),
                "FISCAL_YEAR": order["FISCAL_YEAR"],
                "KUNNR":       order["KUNNR"],
                "NAME1":       order["NAME1"],
                "AREA_MGR":    order["AREA_MGR"],
                "REGION":      order["REGION"],
            })
        order["NETWR"]      = round(total_net, 2)
        order["MWSBP"]      = round(total_net * 0.18, 2)
        order["COST_TOTAL"] = round(total_cost, 2)
        order["MARGIN_PCT"] = round((total_net - total_cost) / total_net * 100, 2) if total_net > 0 else 0
    return items

# ─── LIKP: Delivery Header ───────────────────────────────────────────────────
def generate_likp(n, orders):
    deliveries   = []
    valid_orders = [o for o in orders if o["GBSTK"] in ["B", "C"]]
    sample       = random.sample(valid_orders, min(n, len(valid_orders)))
    for i, order in enumerate(sample, start=1):
        ship_date = datetime.strptime(order["ERDAT"], "%Y-%m-%d") + timedelta(days=random.randint(3, 14))
        deliveries.append({
            "VBELN":       f"DL{str(i).zfill(8)}",
            "ERDAT":       fmt_date(ship_date),
            "VSTEL":       random.choice(SHIPPING_POINTS),
            "KUNNR":       order["KUNNR"],
            "NAME1":       order["NAME1"],
            "LFART":       "LF",
            "WADAT":       fmt_date(ship_date),
            "WADAT_IST":   fmt_date(ship_date + timedelta(days=random.randint(0, 3))),
            "LFGSX":       random.choice(DELIVERY_STATUSES),
            "VBELN_VK":    order["VBELN"],
            "TRAID":       f"TRK-{fake.numerify('####')}",
            "BTGEW":       round(random.uniform(10, 5000), 2),
            "GEWEI":       "KG",
            "FISCAL_YEAR": order["FISCAL_YEAR"],
            "AREA_MGR":    order["AREA_MGR"],
            "REGION":      order["REGION"],
        })
    return deliveries

# ─── LIPS: Delivery Items ─────────────────────────────────────────────────────
def generate_lips(deliveries, vbap_items):
    items         = []
    vbap_by_order = {}
    for item in vbap_items:
        vbap_by_order.setdefault(item["VBELN"], []).append(item)
    for dlv in deliveries:
        for idx, item in enumerate(vbap_by_order.get(dlv["VBELN_VK"], []), start=10):
            items.append({
                "VBELN":       dlv["VBELN"],
                "POSNR":       str(idx * 10).zfill(6),
                "MATNR":       item["MATNR"],
                "ARKTX":       item["ARKTX"],
                "LFIMG":       item["KWMENG"],
                "VRKME":       item["MEINS"],
                "WERKS":       item["WERKS"],
                "LGORT":       item["LGORT"],
                "VBELN_VK":    dlv["VBELN_VK"],
                "POSNR_VK":    item["POSNR"],
                "FISCAL_YEAR": item["FISCAL_YEAR"],
            })
    return items

# ─── VBRK: Billing Header ────────────────────────────────────────────────────
def generate_vbrk(n, deliveries, customers_dict):
    billings = []
    sample   = random.sample(deliveries, min(n, len(deliveries)))
    for i, dlv in enumerate(sample, start=1):
        bill_date = datetime.strptime(dlv["WADAT_IST"], "%Y-%m-%d") + timedelta(days=random.randint(1, 7))
        net       = round(random.uniform(5000, 500000), 2)
        cust      = customers_dict.get(dlv["KUNNR"], {})
        billings.append({
            "VBELN":       f"BI{str(i).zfill(8)}",
            "ERDAT":       fmt_date(bill_date),
            "FKART":       "F2",
            "KUNNR":       dlv["KUNNR"],
            "NAME1":       dlv["NAME1"],
            "WAERS":       "INR",
            "NETWR":       net,
            "MWSBP":       round(net * 0.18, 2),
            "FKSTO":       "X" if random.random() < 0.05 else "",
            "RFBSK":       random.choice(["A", "B", "C"]),
            "ZTERM":       random.choice(PAYMENT_TERMS),
            "VBELN_VL":    dlv["VBELN"],
            "ZUONR":       f"REF-{fake.numerify('######')}",
            "FISCAL_YEAR": dlv["FISCAL_YEAR"],
            "AREA_MGR":    cust.get("AREA_MGR", ""),
            "AREA_MGR_ID": cust.get("AREA_MGR_ID", ""),
            "REGION":      cust.get("REGION", ""),
        })
    return billings

# ─── VBRP: Billing Items ──────────────────────────────────────────────────────
def generate_vbrp(billings):
    items = []
    for bill in billings:
        chosen = random.sample(MATERIALS, random.randint(1, 4))
        for idx, mat in enumerate(chosen, start=10):
            qty        = random.randint(1, 30)
            sale_price = mat["price"] * random.uniform(0.9, 1.15)
            cost_price = mat["cost"]  * random.uniform(0.95, 1.05)
            net        = round(qty * sale_price, 2)
            cost_total = round(qty * cost_price, 2)
            margin_pct = round((sale_price - cost_price) / sale_price * 100, 2)
            items.append({
                "VBELN":       bill["VBELN"],
                "POSNR":       str(idx * 10).zfill(6),
                "MATNR":       mat["matnr"],
                "ARKTX":       mat["maktx"],
                "FKIMG":       qty,
                "VRKME":       mat["meins"],
                "NETWR":       net,
                "MWSBP":       round(net * 0.18, 2),
                "NETPR":       round(sale_price, 2),
                "COST_PRICE":  round(cost_price, 2),
                "COST_TOTAL":  cost_total,
                "MARGIN_PCT":  margin_pct,
                "FISCAL_YEAR": bill["FISCAL_YEAR"],
                "KUNNR":       bill["KUNNR"],
                "NAME1":       bill["NAME1"],
                "AREA_MGR":    bill["AREA_MGR"],
                "REGION":      bill["REGION"],
            })
    return items

# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("Generating SAP SD dummy data v3 (FY22-FY25)...")

    kna1           = generate_kna1(NUM_CUSTOMERS)
    knvv           = generate_knvv(kna1)
    customers_dict = {c["KUNNR"]: c for c in kna1}

    vbak = generate_vbak(NUM_ORDERS, kna1)
    vbap = generate_vbap(vbak)
    likp = generate_likp(NUM_DELIVERIES, vbak)
    lips = generate_lips(likp, vbap)
    vbrk = generate_vbrk(NUM_BILLINGS, likp, customers_dict)
    vbrp = generate_vbrp(vbrk)

    tables = {
        "KNA1": kna1, "KNVV": knvv,
        "VBAK": vbak, "VBAP": vbap,
        "LIKP": likp, "LIPS": lips,
        "VBRK": vbrk, "VBRP": vbrp,
    }

    os.makedirs("data", exist_ok=True)
    for name, records in tables.items():
        fname = f"data/{name}.json"
        with open(fname, "w") as f:
            json.dump(records, f, indent=2, default=str)
        print(f"  {name:6s} → {fname:22s} ({len(records):4d} records)")

    print(f"\nFiscal Year breakdown (Sales Orders):")
    for fy in FY_LABELS:
        count = sum(1 for o in vbak if o["FISCAL_YEAR"] == fy)
        total = sum(o["NETWR"] for o in vbak if o["FISCAL_YEAR"] == fy)
        print(f"  {fy}: {count} orders | ₹{total:,.0f} total revenue")

    print(f"\nArea Manager breakdown (Sales Orders):")
    for am_id, am_info in AREA_MANAGERS.items():
        count = sum(1 for o in vbak if o["AREA_MGR_ID"] == am_id)
        total = sum(o["NETWR"] for o in vbak if o["AREA_MGR_ID"] == am_id)
        print(f"  {am_info['name']:20s} ({am_info['region']:5s}): {count} orders | ₹{total:,.0f}")

    print(f"\nSummary:")
    print(f"  Customers  : {len(kna1)} (6 area managers, 4 regions)")
    print(f"  Orders     : {len(vbak)} headers, {len(vbap)} items (FY22-FY25)")
    print(f"  Deliveries : {len(likp)} headers, {len(lips)} items")
    print(f"  Billings   : {len(vbrk)} headers, {len(vbrp)} items")
    print(f"\nDone. All files saved to data/")

if __name__ == "__main__":
    main()