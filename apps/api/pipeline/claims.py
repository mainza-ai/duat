"""
duat/apps/api/pipeline/claims.py

Claim-provenance records: every dataset the product presents carries the
evidential status of its source claims (who claimed, where, review status,
retractions, expert response, physical plausibility). Pipeline provenance
(SHA-256 manifests) means little if the source claim itself is unexamined.
"""
from __future__ import annotations

from typing import Any, Dict, List

ClaimStatus = str  # unverified | contested | retracted-foundation | validated | ongoing

CLAIM_RECORDS: List[Dict[str, Any]] = [
    {
        "id": "khafre-2025-press",
        "title": "Khafre subterranean complex (8 shafts, spiral pathways, cubic chambers)",
        "claimants": ["Corrado Malanga (Univ. Pisa)", "Filippo Biondi (Univ. Strathclyde)", "Armando Mei"],
        "venue": "Press conference / media briefings",
        "venue_status": "press",
        "date": "2025-03-15",
        "status": "contested",
        "parameters": {
            "shafts": {"value": 8, "cited": "8 (all outlets agree; 2 parallel N–S rows of 4)"},
            "shaft_diameter_m": {"value": 11, "cited": "10–12 m diameter (radius 5.0–6.0 m; 5.5 m modeled)"},
            "shaft_depth_m": {"value": 648, "cited": "610–648 m (~2,100 ft); network claimed down to ~2,000 m"},
            "chamber_size_m": {"value": 80, "cited": "80x80x80 m cubes (JPost/Ahram) vs 90 m (NatGeo/TRT) — outlet variance"},
            "multi_level_structures": {"value": 5, "cited": "5 multi-level 'Zed'-like basal structures linked by horizontal pathways"},
            "spiral_pathways": {"value": True, "cited": "described, never imaged publicly"},
            "extra_structures": {"value": "5 multi-level structures, water system, Hall of Records language",
                                 "cited": "press statements only"},
        },
        "sources": [
            {"outlet": "Telegraph", "url": "https://www.telegraph.co.uk/news/2025/03/23/egyptologists-row-over-underground-city-beneath-pyramids", "date": "2025-03-23"},
            {"outlet": "National Geographic", "url": "https://www.nationalgeographic.com/history/article/is-there-a-secret-city-under-the-pyramids-", "date": "2025-04-16"},
            {"outlet": "AFP Fact Check", "url": "https://factcheck.afp.com/doc.afp.com.42LQ7G3", "date": "2025-04-23"},
            {"outlet": "Al-Ahram Weekly", "url": "https://english.ahram.org.eg/NewsContent/50/1207/544236/AlAhram-Weekly/Heritage/Controversy-over-Pyramid-claims.aspx", "date": "2025-04-08"},
            {"outlet": "Euronews", "url": "https://www.euronews.com/2025/03/26/going-underground-experts-clash-over-hidden-city-beneath-egypt-pyramids", "date": "2025-03-26"},
        ],
        "expert_response": "Rejected by Zahi Hawass (no authorized work, no radar inside, no evidence); "
                           "radar specialist L. Conyers ('huge exaggeration'); AFP expert panel ('unfounded').",
        "physical_notes": "Orbital SAR cannot resolve such features at such depths: rock penetration "
                          "is meters, claimed depths 610–2000 m; method unpublished for independent review.",
        "anchors": [
            {"label": "Z1", "feature": "Multi-level basal structure Z1 (NW Zed complex)", "position": [-40.0, -40.0, -22.0],
             "note": "5-level stepped structure near pyramid base above West shaft row."},
            {"label": "Z2", "feature": "Multi-level basal structure Z2 (SW Zed complex)", "position": [-40.0, 40.0, -22.0],
             "note": "5-level stepped structure near pyramid base above West shaft row."},
            {"label": "Z3", "feature": "Multi-level basal structure Z3 (Central Zed complex)", "position": [0.0, 0.0, -20.0],
             "note": "Central 5-level stepped basal structure beneath Khafre core."},
            {"label": "Z4", "feature": "Multi-level basal structure Z4 (NE Zed complex)", "position": [40.0, -40.0, -22.0],
             "note": "5-level stepped structure near pyramid base above East shaft row."},
            {"label": "Z5", "feature": "Multi-level basal structure Z5 (SE Zed complex)", "position": [40.0, 40.0, -22.0],
             "note": "5-level stepped structure near pyramid base above East shaft row."},
            {"label": "W1", "feature": "Shaft W1 (west row)", "position": [-40.0, -60.0, -300.0],
             "note": "Claimed vertical cylinder."},
            {"label": "W2", "feature": "Shaft W2 (west row, modeled)", "position": [-40.0, -20.0, -300.0],
             "note": "The shaft this observatory descends; spiral reading disputed (ramp vs coil)."},
            {"label": "W3", "feature": "Shaft W3 (west row)", "position": [-40.0, 20.0, -300.0], "note": ""},
            {"label": "W4", "feature": "Shaft W4 (west row)", "position": [-40.0, 60.0, -300.0], "note": ""},
            {"label": "E1", "feature": "Shaft E1 (east row)", "position": [40.0, -60.0, -300.0], "note": ""},
            {"label": "E2", "feature": "Shaft E2 (east row)", "position": [40.0, -20.0, -300.0], "note": ""},
            {"label": "E3", "feature": "Shaft E3 (east row)", "position": [40.0, 20.0, -300.0], "note": ""},
            {"label": "E4", "feature": "Shaft E4 (east row)", "position": [40.0, 60.0, -300.0], "note": ""},
            {"label": "C-W", "feature": "West basal chamber (80 m per some outlets, 90 m per others)",
             "position": [-40.0, 0.0, -590.0], "note": "Outlet variance recorded, not resolved."},
            {"label": "C-E", "feature": "East basal chamber", "position": [40.0, 0.0, -590.0], "note": ""},
            {"label": "H2O", "feature": "Claimed water table / aquifer", "position": [0.0, 0.0, -585.0],
             "note": "Press-briefing claim; our Tier-2 boost is a moisture proxy, not water."},
            {"label": "M", "feature": "Upper manifold corridor", "position": [-40.0, 0.0, -45.0], "note": ""},
        ],
    },
    {
        "id": "biondi-malanga-2022",
        "title": "SAR Doppler tomography of the Great Pyramid (foundational method paper)",
        "claimants": ["Filippo Biondi", "Corrado Malanga"],
        "venue": "Remote Sensing 14(20):5231 (2022)",
        "venue_status": "journal",
        "date": "2022-10-19",
        "status": "retracted-foundation",
        "parameters": {},
        "sources": [
            {"outlet": "MDPI retraction notice", "url": "https://www.mdpi.com/2072-4292/18/16/2679", "date": "2026-08-09"},
            {"outlet": "Ahram Online", "url": "https://english.ahram.org.eg/UI/Front/Inner.aspx?NewsContentID=576187", "date": "2026-09-08"},
            {"outlet": "Newsweek", "url": "https://www.newsweek.com/bombshell-great-pyramid-of-giza-study-retracted-12412886", "date": "2026-09-07"},
        ],
        "expert_response": "Retracted for 'serious methodological flaws and statistical errors'; "
                           "review history already questioned validation and measurement axes.",
        "physical_notes": "Any downstream claim citing this method inherits the retraction.",
    },
    {
        "id": "scanpyramids-2017",
        "title": "ScanPyramids Big Void (muography, Khufu)",
        "claimants": ["ScanPyramids collaboration (Morishima et al.)"],
        "venue": "Nature 552:386 (2017)",
        "venue_status": "journal",
        "date": "2017-11-02",
        "status": "validated",
        "parameters": {"void_length_m": {"value": 30, "cited": "≥30 m long, Grand-Gallery-like cross-section"}},
        "sources": [
            {"outlet": "Nature", "url": "https://www.nature.com/articles/nature24647", "date": "2017-11-02"},
        ],
        "expert_response": "Confirmed by three detector technologies and independent analyses.",
        "physical_notes": "Reference benchmark: ~3 m voids resolvable with ~2 yr exposure (Fermilab sim).",
    },
    {
        "id": "scanpyramids-2023",
        "title": "ScanPyramids North-Face Corridor (muography, Khufu)",
        "claimants": ["ScanPyramids collaboration"],
        "venue": "Nature Communications (2023)",
        "venue_status": "journal",
        "date": "2023-03-02",
        "status": "validated",
        "parameters": {"corridor_m": {"value": [9, 2, 2], "cited": "9 m long, ~2×2 m, cm-level precision"}},
        "sources": [
            {"outlet": "Nature Communications", "url": "https://www.nature.com/articles/s41467-023-36351-0", "date": "2023-03-02"},
        ],
        "expert_response": "Peer-reviewed; position/dimensions characterized to centimeters.",
        "physical_notes": "Gold-standard validated void used as our fidelity benchmark.",
    },
    {
        "id": "scidep-2025",
        "title": "ScIDEP muon radiography at Khafre (ongoing)",
        "claimants": ["ScIDEP collaboration"],
        "venue": "arXiv:2503.21959 (2025)",
        "venue_status": "preprint",
        "date": "2025-03-27",
        "status": "ongoing",
        "parameters": {},
        "sources": [
            {"outlet": "arXiv", "url": "https://arxiv.org/abs/2503.21959", "date": "2025-03-27"},
        ],
        "expert_response": "Project overview; detector studies ongoing — no void claims made.",
        "physical_notes": "The legitimate in-progress Khafre scan to watch.",
    },
    {
        "id": "alvarez-1960s",
        "title": "Alvarez Khafre muon survey (historic null result)",
        "claimants": ["Luis Alvarez et al."],
        "venue": "Field survey (1960s)",
        "venue_status": "historic",
        "date": "1969",
        "status": "validated",
        "parameters": {"finding": {"value": "solid rock around known rooms", "cited": "no unknown voids found"}},
        "sources": [
            {"outlet": "IEEE Spectrum (retrospective)", "url": "https://spectrum.ieee.org/muon-imaging-finds-hidden-chamber-in-great-pyramid-of-giza", "date": "2017-11-02"},
        ],
        "expert_response": "Historic null result at Khafre; anchors our bedrock-null prior.",
        "physical_notes": "Older detectors, but a real measurement at this exact pyramid.",
    },
    {
        "id": "ert-2025",
        "title": "Electrical resistivity tomography study (related work)",
        "claimants": ["Scientific Reports authors (Nov 2025)"],
        "venue": "Scientific Reports (2025)",
        "venue_status": "journal",
        "date": "2025-11-01",
        "status": "validated",
        "parameters": {},
        "sources": [],
        "expert_response": "Peer-reviewed ERT work cited in retraction coverage as adjacent legitimate method.",
        "physical_notes": "Listed for completeness; details not independently verified here.",
    },
]

VALID_STATUSES = ("unverified", "contested", "retracted-foundation", "validated", "ongoing")

KHAFRE_DATASET_CLAIM_IDS = ["khafre-2025-press", "biondi-malanga-2022", "alvarez-1960s", "scidep-2025"]


def get_claim(claim_id: str) -> Dict[str, Any] | None:
    for c in CLAIM_RECORDS:
        if c["id"] == claim_id:
            return c
    return None


def list_claims(status: str | None = None) -> List[Dict[str, Any]]:
    if status is None:
        return list(CLAIM_RECORDS)
    return [c for c in CLAIM_RECORDS if c["status"] == status]
