/** Typed client for the Duat FastAPI backend + Atlas camera helpers. */

export interface AtlasCamera {
  intrinsics: { fx: number; fy: number; cx: number; cy: number; width: number; height: number };
  extrinsics: { position: [number, number, number]; quaternion: [number, number, number, number]; coordinateSystem: "rub" | "rdf" };
}

export interface EnsembleRequest {
  camera: { width?: number; height?: number; fov_deg?: number; coordinate_system?: "rub" | "rdf";
    pose: { position: [number, number, number]; quaternion: [number, number, number, number] } };
  hypothesis?: "megalithic" | "karst" | "artifact";
  instrument_rig?: "borescope_led" | "drop_flare" | "structured_laser";
  seeds?: number[];
  voxel_size?: number;
  sensor_noise_sigma_m?: number;
  z_min?: number;
  z_max?: number | null;
  mock_mode?: boolean;
}

export interface EnsembleResponse {
  provenance_sha256: string;
  timestamp_utc: number;
  prompt_used: string;
  z_min: number;
  z_max: number;
  voxel_size: number;
  empty_void_pixel_ratio: number;
  mean_ensemble_disagreement: number;
  layers: {
    measured_log_depth_b64: string;
    inferred_reflectivity_b64: string;
    imagined_consensus_rgb_b64: string;
    disagreement_map_b64: string;
  };
  next_best_view: { target_z_m: number; predicted_variance_reduction_pct: number; rationale: string };
}

export async function postEnsemble(baseUrl: string, req: EnsembleRequest): Promise<EnsembleResponse> {
  const res = await fetch(`${baseUrl}/api/chisel/ensemble`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`ensemble failed: ${res.status} ${await res.text()}`);
  return res.json();
}

/** Official inverted log-depth decode (255 = zMin, 0 = empty). Mirrors backend + GLSL. */
export function decodeLogDepthByte(p: number, zMin: number, zMax: number): number {
  if (p <= 0) return 0;
  return zMin * Math.pow(zMax / zMin, 1 - p / 255);
}

/** Build an Atlas nested camera from flat pose + intrinsics. */
export function atlasCamera(
  position: [number, number, number],
  quaternion: [number, number, number, number],
  fx = 914.29, w = 1280, h = 720
): AtlasCamera {
  return {
    intrinsics: { fx, fy: fx, cx: w / 2, cy: h / 2, width: w, height: h },
    extrinsics: { position, quaternion, coordinateSystem: "rub" },
  };
}
