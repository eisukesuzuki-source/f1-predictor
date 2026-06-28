from __future__ import annotations

# Circuit database keyed by FastF1 schedule `Location` field
# Fields per circuit:
#   name         : display name
#   lat, lon     : GPS coordinates for weather API
#   altitude_m   : metres above sea level
#   type         : "permanent" or "street"
#   overtaking   : 1 (very hard) … 5 (easy)
#   tire_deg     : 1 (low deg) … 5 (high deg)
#   pu_sensitivity : 1 … 5 (how much engine power matters)
#   wet_sensitivity: 1 … 5
#   drs_zones    : number of DRS zones

CIRCUIT_DB: dict[str, dict] = {
    # ── Bahrain ──────────────────────────────────────────────────────────────
    "Sakhir": {
        "name": "バーレーン国際サーキット",
        "lat": 26.0325,
        "lon": 50.5106,
        "altitude_m": 7,
        "type": "permanent",
        "overtaking": 4,
        "tire_deg": 4,
        "pu_sensitivity": 3,
        "wet_sensitivity": 2,
        "drs_zones": 3,
    },
    # ── Saudi Arabia ─────────────────────────────────────────────────────────
    "Jeddah": {
        "name": "ジェッダ市街地サーキット",
        "lat": 21.6319,
        "lon": 39.1044,
        "altitude_m": 15,
        "type": "street",
        "overtaking": 2,
        "tire_deg": 2,
        "pu_sensitivity": 4,
        "wet_sensitivity": 3,
        "drs_zones": 3,
    },
    # ── Australia ────────────────────────────────────────────────────────────
    "Melbourne": {
        "name": "アルバートパーク・サーキット",
        "lat": -37.8497,
        "lon": 144.9680,
        "altitude_m": 15,
        "type": "street",
        "overtaking": 2,
        "tire_deg": 3,
        "pu_sensitivity": 3,
        "wet_sensitivity": 4,
        "drs_zones": 4,
    },
    # ── Japan ────────────────────────────────────────────────────────────────
    "Suzuka": {
        "name": "�木サーキット",
        "lat": 34.8431,
        "lon": 136.5407,
        "altitude_m": 40,
        "type": "permanent",
        "overtaking": 2,
        "tire_deg": 3,
        "pu_sensitivity": 3,
        "wet_sensitivity": 5,
        "drs_zones": 2,
    },
    # ── China ────────────────────────────────────────────────────────────────
    "Shanghai": {
        "name": "上海国際サーキット",
        "lat": 31.3389,
        "lon": 121.2197,
        "altitude_m": 5,
        "type": "permanent",
        "overtaking": 4,
        "tire_deg": 3,
        "pu_sensitivity": 3,
        "wet_sensitivity": 3,
        "drs_zones": 2,
    },
    # ── Miami ─────────────────────────────────────────────────────────────────
    "Miami": {
        "name": "マイアミ・インターナショナル・オートドローム",
        "lat": 25.9581,
        "lon": -80.2389,
        "altitude_m": 2,
        "type": "street",
        "overtaking": 3,
        "tire_deg": 3,
        "pu_sensitivity": 3,
        "wet_sensitivity": 3,
        "drs_zones": 3,
    },
    # ── Imola (Emilia Romagna) ────────────────────────────────────────────────
    "Imola": {
        "name": "エンツォ・エ・ディーノ・フェラーリ・サーキット",
        "lat": 44.3439,
        "lon": 11.7167,
        "altitude_m": 35,
        "type": "permanent",
        "overtaking": 1,
        "tire_deg": 3,
        "pu_sensitivity": 3,
        "wet_sensitivity": 4,
        "drs_zones": 2,
    },
    # ── Monaco ───────────────────────────────────────────────────────────────
    "Monte-Carlo": {
        "name": "モナコ市街地サーキット",
        "lat": 43.7347,
        "lon": 7.4206,
        "altitude_m": 7,
        "type": "street",
        "overtaking": 1,
        "tire_deg": 2,
        "pu_sensitivity": 1,
        "wet_sensitivity": 5,
        "drs_zones": 1,
    },
    # ── Canada ───────────────────────────────────────────────────────────────
    "Montreal": {
        "name": "ジル・ヴィルヌーヴ・サーキット",
        "lat": 45.5000,
        "lon": -73.5228,
        "altitude_m": 20,
        "type": "street",
        "overtaking": 3,
        "tire_deg": 3,
        "pu_sensitivity": 4,
        "wet_sensitivity": 4,
        "drs_zones": 2,
    },
    # ── Spain ────────────────────────────────────────────────────────────────
    "Barcelona": {
        "name": "カタロニア・サーキット",
        "lat": 41.5700,
        "lon": 2.2611,
        "altitude_m": 120,
        "type": "permanent",
        "overtaking": 2,
        "tire_deg": 4,
        "pu_sensitivity": 3,
        "wet_sensitivity": 3,
        "drs_zones": 2,
    },
    # ── Austria ──────────────────────────────────────────────────────────────
    "Spielberg": {
        "name": "レッドブル・リンク",
        "lat": 47.2197,
        "lon": 14.7647,
        "altitude_m": 660,
        "type": "permanent",
        "overtaking": 4,
        "tire_deg": 3,
        "pu_sensitivity": 4,
        "wet_sensitivity": 4,
        "drs_zones": 3,
    },
    # ── Great Britain ─────────────────────────────────────────────────────────
    "Silverstone": {
        "name": "シルバーストーン・サーキット",
        "lat": 52.0786,
        "lon": -1.0169,
        "altitude_m": 153,
        "type": "permanent",
        "overtaking": 3,
        "tire_deg": 4,
        "pu_sensitivity": 3,
        "wet_sensitivity": 5,
        "drs_zones": 2,
    },
    # ── Hungary ──────────────────────────────────────────────────────────────
    "Budapest": {
        "name": "ハンガロリンク",
        "lat": 47.5789,
        "lon": 19.2486,
        "altitude_m": 264,
        "type": "permanent",
        "overtaking": 1,
        "tire_deg": 3,
        "pu_sensitivity": 2,
        "wet_sensitivity": 4,
        "drs_zones": 2,
    },
    # ── Belgium ──────────────────────────────────────────────────────────────
    "Spa-Francorchamps": {
        "name": "スパ・フランコルシャン",
        "lat": 50.4372,
        "lon": 5.9714,
        "altitude_m": 401,
        "type": "permanent",
        "overtaking": 4,
        "tire_deg": 3,
        "pu_sensitivity": 5,
        "wet_sensitivity": 5,
        "drs_zones": 2,
    },
    # ── Netherlands ──────────────────────────────────────────────────────────
    "Zandvoort": {
        "name": "ザントフォールト・サーキット",
        "lat": 52.3888,
        "lon": 4.5408,
        "altitude_m": 3,
        "type": "permanent",
        "overtaking": 1,
        "tire_deg": 4,
        "pu_sensitivity": 2,
        "wet_sensitivity": 4,
        "drs_zones": 2,
    },
    # ── Italy (Monza) ─────────────────────────────────────────────────────────
    "Monza": {
        "name": "モンツァ・サーキット",
        "lat": 45.6156,
        "lon": 9.2811,
        "altitude_m": 162,
        "type": "permanent",
        "overtaking": 5,
        "tire_deg": 2,
        "pu_sensitivity": 5,
        "wet_sensitivity": 3,
        "drs_zones": 2,
    },
    # ── Azerbaijan ────────────────────────────────────────────────────────────
    "Baku": {
        "name": "バクー市街地サーキット",
        "lat": 40.3725,
        "lon": 49.8533,
        "altitude_m": -28,
        "type": "street",
        "overtaking": 4,
        "tire_deg": 2,
        "pu_sensitivity": 4,
        "wet_sensitivity": 4,
        "drs_zones": 2,
    },
    # ── Singapore ─────────────────────────────────────────────────────────────
    "Marina Bay": {
        "name": "マリーナベイ市街地サーキット",
        "lat": 1.2914,
        "lon": 103.8640,
        "altitude_m": 15,
        "type": "street",
        "overtaking": 2,
        "tire_deg": 3,
        "pu_sensitivity": 1,
        "wet_sensitivity": 4,
        "drs_zones": 3,
    },
    # ── United States (Austin) ────────────────────────────────────────────────
    "Austin": {
        "name": "サーキット・オブ・ジ・アメリカズ",
        "lat": 30.1328,
        "lon": -97.6411,
        "altitude_m": 150,
        "type": "permanent",
        "overtaking": 4,
        "tire_deg": 4,
        "pu_sensitivity": 3,
        "wet_sensitivity": 3,
        "drs_zones": 2,
    },
    # ── Mexico ────────────────────────────────────────────────────────────────
    "Mexico City": {
        "name": "エルマノス・ロドリゲス・サーキット",
        "lat": 19.4042,
        "lon": -99.0907,
        "altitude_m": 2285,
        "type": "permanent",
        "overtaking": 3,
        "tire_deg": 2,
        "pu_sensitivity": 5,
        "wet_sensitivity": 3,
        "drs_zones": 3,
    },
    # ── Brazil (Sao Paulo) ────────────────────────────────────────────────────
    "Sao Paulo": {
        "name": "インテルラゴス・サーキット",
        "lat": -23.7014,
        "lon": -46.6969,
        "altitude_m": 785,
        "type": "permanent",
        "overtaking": 4,
        "tire_deg": 3,
        "pu_sensitivity": 4,
        "wet_sensitivity": 5,
        "drs_zones": 2,
    },
    # ── Las Vegas ─────────────────────────────────────────────────────────────
    "Las Vegas": {
        "name": "ラスベガス・ストリート・サーキット",
        "lat": 36.1147,
        "lon": -115.1728,
        "altitude_m": 616,
        "type": "street",
        "overtaking": 4,
        "tire_deg": 2,
        "pu_sensitivity": 4,
        "wet_sensitivity": 2,
        "drs_zones": 2,
    },
    # ── Qatar ─────────────────────────────────────────────────────────────────
    "Lusail": {
        "name": "ルサイル国際サーキット",
        "lat": 25.4900,
        "lon": 51.4542,
        "altitude_m": 15,
        "type": "permanent",
        "overtaking": 3,
        "tire_deg": 5,
        "pu_sensitivity": 3,
        "wet_sensitivity": 2,
        "drs_zones": 2,
    },
    # ── Abu Dhabi ─────────────────────────────────────────────────────────────
    "Yas Island": {
        "name": "ヤス・マリーナ・サーキット",
        "lat": 24.4672,
        "lon": 54.6031,
        "altitude_m": 3,
        "type": "permanent",
        "overtaking": 3,
        "tire_deg": 2,
        "pu_sensitivity": 3,
        "wet_sensitivity": 2,
        "drs_zones": 3,
    },
    # ── Madrid (2026) ─────────────────────────────────────────────────────────
    "Madrid": {
        "name": "マドリード・サーキット",
        "lat": 40.4168,
        "lon": -3.7038,
        "altitude_m": 667,
        "type": "street",
        "overtaking": 3,
        "tire_deg": 3,
        "pu_sensitivity": 3,
        "wet_sensitivity": 3,
        "drs_zones": 3,
    },
}

# Alternative location spellings that FastF1 may use
_LOCATION_ALIASES: dict[str, str] = {
    "Sakhir": "Sakhir",
    "Bahrain": "Sakhir",
    "Jeddah": "Jeddah",
    "Saudi Arabia": "Jeddah",
    "Melbourne": "Melbourne",
    "Suzuka": "Suzuka",
    "Japan": "Suzuka",
    "Shanghai": "Shanghai",
    "China": "Shanghai",
    "Miami": "Miami",
    "Imola": "Imola",
    "Monte-Carlo": "Monte-Carlo",
    "Monaco": "Monte-Carlo",
    "Montreal": "Montreal",
    "Canada": "Montreal",
    "Barcelona": "Barcelona",
    "Spain": "Barcelona",
    "Spielberg": "Spielberg",
    "Austria": "Spielberg",
    "Silverstone": "Silverstone",
    "Britain": "Silverstone",
    "Great Britain": "Silverstone",
    "Budapest": "Budapest",
    "Hungary": "Budapest",
    "Spa-Francorchamps": "Spa-Francorchamps",
    "Spa": "Spa-Francorchamps",
    "Belgium": "Spa-Francorchamps",
    "Zandvoort": "Zandvoort",
    "Netherlands": "Zandvoort",
    "Monza": "Monza",
    "Italy": "Monza",
    "Baku": "Baku",
    "Azerbaijan": "Baku",
    "Marina Bay": "Marina Bay",
    "Singapore": "Marina Bay",
    "Austin": "Austin",
    "United States": "Austin",
    "Mexico City": "Mexico City",
    "Mexico": "Mexico City",
    "Sao Paulo": "Sao Paulo",
    "Brazil": "Sao Paulo",
    "Las Vegas": "Las Vegas",
    "Lusail": "Lusail",
    "Qatar": "Lusail",
    "Yas Island": "Yas Island",
    "Abu Dhabi": "Yas Island",
    "Madrid": "Madrid",
    "Spain Street": "Madrid",
}

_DEFAULT_CIRCUIT: dict = {
    "name": "Unknown Circuit",
    "lat": 0.0,
    "lon": 0.0,
    "altitude_m": 50,
    "type": "permanent",
    "overtaking": 3,
    "tire_deg": 3,
    "pu_sensitivity": 3,
    "wet_sensitivity": 3,
    "drs_zones": 2,
}

# PU supplier mapping — keyed by common FastF1 TeamName strings
PU_SUPPLIERS: dict[str, str] = {
    # Mercedes-PU
    "Mercedes": "Mercedes",
    "Mercedes-AMG Petronas F1 Team": "Mercedes",
    "Mercedes AMG": "Mercedes",
    "Williams": "Mercedes",
    "Williams Racing": "Mercedes",
    "Aston Martin": "Mercedes",
    "Aston Martin Aramco F1 Team": "Mercedes",
    # Ferrari-PU
    "Ferrari": "Ferrari",
    "Scuderia Ferrari": "Ferrari",
    "Haas F1 Team": "Ferrari",
    "Haas": "Ferrari",
    "Sauber": "Ferrari",
    "Kick Sauber": "Ferrari",
    "Alfa Romeo": "Ferrari",
    "Alfa Romeo F1 Team Stake": "Ferrari",
    # Red Bull / Honda RBPT
    "Red Bull Racing": "Honda RBPT",
    "Oracle Red Bull Racing": "Honda RBPT",
    "RB": "Honda RBPT",
    "Visa Cash App RB F1 Team": "Honda RBPT",
    "AlphaTauri": "Honda RBPT",
    "Scuderia AlphaTauri": "Honda RBPT",
    # Renault / Alpine
    "Alpine": "Renault",
    "Alpine F1 Team": "Renault",
    "Renault": "Renault",
    # McLaren / Mercedes (from 2021)
    "McLaren": "Mercedes",
    "McLaren F1 Team": "Mercedes",
}


def get_circuit_info(location: str, event_name: str = "") -> dict:
    """Return circuit info dict for the given FastF1 Location / EventName.

    Matching strategy:
    1. Exact key match in CIRCUIT_DB
    2. Alias lookup
    3. Partial case-insensitive match on CIRCUIT_DB keys
    4. Partial case-insensitive match on event_name against CIRCUIT_DB keys
    5. Partial match against alias keys
    6. Return default dict with neutral values
    """
    # 1. Exact key
    if location in CIRCUIT_DB:
        return dict(CIRCUIT_DB[location])

    # 2. Alias exact
    alias_key = _LOCATION_ALIASES.get(location)
    if alias_key and alias_key in CIRCUIT_DB:
        return dict(CIRCUIT_DB[alias_key])

    # 3. Partial match on CIRCUIT_DB keys (location)
    loc_lower = location.lower()
    for key, info in CIRCUIT_DB.items():
        if loc_lower in key.lower() or key.lower() in loc_lower:
            return dict(info)

    # 4. Partial match using event_name against CIRCUIT_DB keys
    if event_name:
        ev_lower = event_name.lower()
        for key, info in CIRCUIT_DB.items():
            if key.lower() in ev_lower or ev_lower in key.lower():
                return dict(info)
        # Also check aliases
        for alias, target in _LOCATION_ALIASES.items():
            if alias.lower() in ev_lower or ev_lower in alias.lower():
                if target in CIRCUIT_DB:
                    return dict(CIRCUIT_DB[target])

    # 5. Partial alias key match on location
    for alias, target in _LOCATION_ALIASES.items():
        if loc_lower in alias.lower() or alias.lower() in loc_lower:
            if target in CIRCUIT_DB:
                return dict(CIRCUIT_DB[target])

    # 6. Default
    return dict(_DEFAULT_CIRCUIT)
