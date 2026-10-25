"""
duat/apps/api/pipeline/prompts.py

Hypothesis Mixer + Instrument Lighting Rigs -> Atlas prompt composer.
Never prompt for "a pretty scene": always pair a domain hypothesis prior
with a physical instrument rig on the same geometry.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

HYPOTHESIS_PRIORS: Dict[str, str] = {
    "megalithic": (
        "Massive ancient subterranean vertical shaft carved with precision-dressed "
        "Aswan granite and weathered nummulitic limestone blocks, stepped helical "
        "spiral ramp, ancient copper chisel marks, mineral salt efflorescence"
    ),
    "karst": (
        "Natural deep phreatic limestone dissolution tube and sinkhole cavern, "
        "wet travertine flowstone curtains, scalloped water erosion channels, "
        "subterranean aquifer reflections at base"
    ),
    "artifact": (
        "Stepped voxelized stone surfaces with geometric acoustic radar interference "
        "fringes and quantized depth facets, raw synthetic aperture tomography structure"
    ),
    "bedrock_null": (
        "Solid unbroken bedrock with no void whatsoever, dense natural limestone "
        "strata with no carving, no chambers and no passages, exactly what the "
        "1960s Alvarez muon survey measured: predicts empty depth returns"
    ),
}

INSTRUMENT_RIGS: Dict[str, Tuple[str, float]] = {
    "borescope_led": (
        "lit solely by a harsh single endoscope LED ring, brutal inverse-square "
        "near-field light falloff, overexposed close stone texture, pitch-black "
        "distant corners, floating dust motes, raw industrial borescope camera footage",
        6.0,
    ),
    "drop_flare": (
        "illuminated by a high-lumen tethered exploration drop-rig and magnesium flare "
        "suspended in the shaft, long dramatic vertical shadows across the spiral ledge, "
        "volumetric dust scatter, high dynamic range",
        40.0,
    ),
    "structured_laser": (
        "illuminated by monochromatic 532nm green autonomous rover LiDAR scanlines "
        "and structured laser crosshairs in pitch darkness, high-contrast rock relief",
        25.0,
    ),
}

ENHANCE_DEFAULT = True


def compose_epistemic_prompt(
    hypothesis: str = "megalithic",
    instrument_rig: str = "drop_flare",
    custom_notes: Optional[str] = None,
) -> Tuple[str, float]:
    prior = HYPOTHESIS_PRIORS.get(hypothesis, HYPOTHESIS_PRIORS["megalithic"])
    rig_desc, recommended_zmax = INSTRUMENT_RIGS.get(instrument_rig, INSTRUMENT_RIGS["drop_flare"])
    prompt = f"{prior}, {rig_desc}."
    if custom_notes:
        prompt = f"{prompt} {custom_notes.strip()}"
    return prompt, recommended_zmax
