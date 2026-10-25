/** Typed client for the Duat FastAPI backend + Atlas camera helpers. */

export interface AtlasCamera {
  intrinsics: { fx: number; fy: number; cx: number; cy: number; width: number; height: number };
  extrinsics: { position: [number, number, number]; quaternion: [number, number, number, number]; coordinateSystem: "rub" | "rdf" };
}

export type RunMode = "mock" | "live";
export type DatasetKind = "khafre" | "nfc" | "big-void" | "uploaded_mesh";
export type InstrumentRig = "borescope_led" | "drop_flare" | "structured_laser";

const OPERATOR_KEY_STORAGE = "duat_operator_key";

export function getOperatorKey(): string {
  try {
    return localStorage.getItem(OPERATOR_KEY_STORAGE) || (import.meta.env.VITE_OPERATOR_API_KEY as string) || "";
  } catch {
    return (import.meta.env.VITE_OPERATOR_API_KEY as string) || "";
  }
}

export function setOperatorKey(key: string): void {
  try {
    if (key.trim()) localStorage.setItem(OPERATOR_KEY_STORAGE, key.trim());
    else localStorage.removeItem(OPERATOR_KEY_STORAGE);
  } catch {
    // ignore storage errors
  }
}

export function apiHeaders(extra: Record<string, string> = {}): Record<string, string> {
  const headers: Record<string, string> = { ...extra };
  const key = getOperatorKey();
  if (key) headers["X-Duat-Key"] = key;
  return headers;
}

export interface EnsembleRequest {
  camera: { width?: number; height?: number; fov_deg?: number; coordinate_system?: "rub" | "rdf";
    pose: { position: [number, number, number]; quaternion: [number, number, number, number] } };
  dataset?: DatasetKind;
  hypothesis?: "megalithic" | "karst" | "artifact" | "bedrock_null";
  force_geometry_on_null?: boolean;
  instrument_rig?: InstrumentRig;
  seeds?: number[];
  voxel_size?: number;
  sensor_noise_sigma_m?: number;
  z_min?: number;
  z_max?: number | null;
  mode?: RunMode;
  structural_weight?: number;
  posterior_resample?: boolean;
  spiral_style?: "interior_ramp" | "exterior_coil";
  chain_batches?: number;
  chain_targets?: number;
  return_depth?: boolean;
  start_batch?: number;
  resume_source_asset_id?: string | null;
  resume_source_camera?: Record<string, unknown> | null;
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
    headers: apiHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`ensemble failed: ${res.status} ${await res.text()}`);
  return res.json();
}

export interface PricingInfo {
  credits_per_usd: number;
  atlas_chisel_1t_credits: number | null;
  atlas_chisel_32t_credits: number | null;
  note: string;
}

export interface JobRecord {
  id: string; kind: string; status: string; progress: number;
  result?: EnsembleResponse & { batches?: { batch: number; operation_id: string }[] };
  error?: { code: string; detail: string };
}

export async function fetchPricing(baseUrl: string): Promise<PricingInfo> {
  const res = await fetch(`${baseUrl}/api/pricing`, { headers: apiHeaders() });
  if (!res.ok) throw new Error("pricing unavailable");
  return res.json();
}

export async function createJob(baseUrl: string, kind: "ensemble" | "chain", request: EnsembleRequest): Promise<{ id: string; status: string; deduplicated: boolean }> {
  const res = await fetch(`${baseUrl}/api/jobs`, {
    method: "POST",
    headers: apiHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ kind, request }),
  });
  if (!res.ok) throw new Error(`job create failed: ${res.status} ${await res.text()}`);
  return res.json();
}

export async function getJob(baseUrl: string, id: string): Promise<JobRecord> {
  const res = await fetch(`${baseUrl}/api/jobs/${id}`, { headers: apiHeaders() });
  if (!res.ok) throw new Error(`job fetch failed: ${res.status}`);
  return res.json();
}

export async function cancelJob(baseUrl: string, id: string): Promise<JobRecord> {
  const res = await fetch(`${baseUrl}/api/jobs/${id}/cancel`, { method: "POST", headers: apiHeaders() });
  if (!res.ok) throw new Error(`job cancel failed: ${res.status}`);
  return res.json();
}

export async function listJobs(baseUrl: string, limit = 20): Promise<{ jobs: JobRecord[] }> {
  const res = await fetch(`${baseUrl}/api/jobs?limit=${limit}`, { headers: apiHeaders() });
  if (!res.ok) throw new Error(`job list failed: ${res.status}`);
  return res.json();
}

export interface DegradationLevel {
  name: string;
  blur_sigma_m: number;
  quant_m: number;
  vertices: number;
  faces: number;
  rms_displacement_m: number;
}

export interface DegradationResponse {
  base_vertices: number;
  base_faces: number;
  levels: DegradationLevel[];
}

export async function fetchDegradation(baseUrl: string): Promise<DegradationResponse> {
  const res = await fetch(`${baseUrl}/api/khafre/degradation`, { headers: apiHeaders() });
  if (!res.ok) throw new Error(`degradation fetch failed: ${res.status}`);
  return res.json();
}

export interface ReferenceVoidInfo {
  kind: string;
  dims_m: [number, number, number];
  claim_id: string;
  grid_shape: [number, number, number];
  void_fraction: number;
  sensor_uncertainty_m: number;
}

export async function fetchReferenceVoids(baseUrl: string): Promise<{ voids: ReferenceVoidInfo[] }> {
  const res = await fetch(`${baseUrl}/api/khafre/reference-voids`, { headers: apiHeaders() });
  if (!res.ok) throw new Error(`reference voids fetch failed: ${res.status}`);
  return res.json();
}

export interface MeshUploadResponse {
  grid_shape: [number, number, number];
  bounds_min_m: [number, number, number];
  bounds_max_m: [number, number, number];
  void_fraction: number;
  mean_reflectivity: number;
}

export async function uploadMesh(baseUrl: string, file: File): Promise<MeshUploadResponse> {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${baseUrl}/api/khafre/mesh`, {
    method: "POST",
    headers: apiHeaders(),
    body: fd,
  });
  if (!res.ok) throw new Error(`mesh upload failed: ${res.status} ${await res.text()}`);
  return res.json();
}

/** Cost preview: tasks × configured rate, or "unknown" when unset. */
export function estimateCost(tasks: number, pricing: PricingInfo | null, perTarget = 1): string {
  if (!pricing) return `${tasks} task(s) · rate unknown`;
  const rate = perTarget <= 1 ? pricing.atlas_chisel_1t_credits : pricing.atlas_chisel_32t_credits;
  if (rate == null) return `${tasks} task(s) · rate unknown (set ATLAS_CREDITS_PER_CHISEL_*)`;
  const usd = (tasks * rate) / pricing.credits_per_usd;
  return `${tasks} task(s) ≈ ${rate * tasks} credits ($${usd.toFixed(2)})`;
}

/** Download a composite comparison card with the provenance bar baked in. */
export async function exportComparisonCard(
  baseUrl: string,
  req: EnsembleRequest
): Promise<string> {
  const res = await postEnsemble(baseUrl, req);
  const load = (uri: string) =>
    new Promise<HTMLImageElement>((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = uri;
    });
  const [depth, refl, rgb, dis, logo] = await Promise.all([
    load(res.layers.measured_log_depth_b64),
    load(res.layers.inferred_reflectivity_b64),
    load(res.layers.imagined_consensus_rgb_b64),
    load(res.layers.disagreement_map_b64),
    load("/milimo_logo.png").catch(() => null),
  ]);

  const W = 1920, H = 1080;
  const cv = document.createElement("canvas");
  cv.width = W; cv.height = H;
  const g = cv.getContext("2d")!;
  g.fillStyle = "#090A0F";
  g.fillRect(0, 0, W, H);

  // Provenance bar (baked in — anti-vividness-bias)
  g.fillStyle = "#0D0F17";
  g.fillRect(0, 0, W, 96);
  g.fillStyle = "#00F0FF";
  g.fillRect(0, 96, W, 3);
  if (logo) g.drawImage(logo, 28, 20, 56, 56);
  g.fillStyle = "#ffffff";
  g.font = "bold 26px monospace";
  g.fillText("MILIMO DUAT // MODEL PREDICTION — CONDITIONED ON SENSOR GEOMETRY", 100, 45);
  g.fillStyle = "#8b95a7";
  g.font = "20px monospace";
  const stamp = new Date(res.timestamp_utc * 1000).toISOString();
  g.fillText(`sha ${res.provenance_sha256.slice(0, 12)} · ${stamp}`, 100, 74);
  const tier = (x: number, color: string, label: string) => {
    g.fillStyle = color;
    g.beginPath(); g.arc(x, 60, 10, 0, Math.PI * 2); g.fill();
    g.fillStyle = "#cbd2dc"; g.font = "20px monospace";
    g.fillText(label, x + 20, 67);
  };
  tier(1380, "#00F0FF", "T1 Measured");
  tier(1580, "#FFB800", "T2 Inferred");
  tier(1760, "#34D399", "T3 Imagined");

  // Panels
  const panel = (img: HTMLImageElement, x: number, label: string, color: string) => {
    g.fillStyle = color; g.font = "bold 22px monospace";
    g.fillText(label, x, 150);
    g.drawImage(img, x, 165, 600, 338);
    g.strokeStyle = "#2a3342"; g.lineWidth = 2;
    g.strokeRect(x, 165, 600, 338);
  };
  panel(depth, 24, "TIER 1 · MEASURED DEPTH (0 = EMPTY VOID)", "#00F0FF");
  panel(refl, 660, "TIER 2 · DIELECTRIC PROXY", "#FFB800");
  panel(rgb, 1296, "TIER 3 · ATLAS CONSENSUS", "#34D399");
  // Disagreement inset on the consensus panel
  g.drawImage(dis, 1296 + 600 - 190, 165 + 338 - 112, 170, 96);
  g.fillStyle = "#34D399"; g.font = "bold 18px monospace";
  g.fillText("VARIANCE", 1296 + 600 - 190, 165 + 338 - 118);

  // Footer: prompt + params + badge
  g.fillStyle = "#0D0F17";
  g.fillRect(0, 560, W, H - 560);
  g.fillStyle = "#34D399";
  g.fillRect(24, 590, 6, 120);
  g.fillStyle = "#cbd2dc"; g.font = "22px monospace";
  wrapText(g, `PROMPT: ${res.prompt_used}`, 48, 615, W - 96, 30);
  g.fillStyle = "#8b95a7"; g.font = "21px monospace";
  const seeds = (req.seeds ?? []).join(",");
  g.fillText(
    `voxel ${res.voxel_size}m · seeds [${seeds}] · z [${res.z_min}, ${res.z_max}] · empty ${(res.empty_void_pixel_ratio * 100).toFixed(1)}% · disagreement ${(res.mean_ensemble_disagreement * 100).toFixed(1)}% · NBV z=${res.next_best_view.target_z_m}m (−${res.next_best_view.predicted_variance_reduction_pct}%)`,
    48, 740);
  g.fillStyle = "#090A0F";
  g.fillRect(0, H - 90, W, 90);
  g.fillStyle = "#34D399";
  g.fillRect(0, H - 90, W, 4);
  g.fillStyle = "#ffffff"; g.font = "bold 24px monospace";
  g.fillText("MODEL PREDICTION — CONDITIONED ON SENSOR GEOMETRY · NOT GROUND TRUTH", 48, H - 38);
  g.fillStyle = "#8b95a7"; g.font = "20px monospace";
  g.fillText(`provenance ${res.provenance_sha256}`, 48, H - 12);

  const name = `milimo-duat-${res.provenance_sha256.slice(0, 12)}.png`;
  const a = document.createElement("a");
  a.download = name;
  a.href = cv.toDataURL("image/png");
  a.click();
  return name;
}

function wrapText(g: CanvasRenderingContext2D, text: string, x: number, y: number, maxW: number, lh: number) {
  const words = text.split(" ");
  let line = "", yy = y, count = 0;
  for (const w of words) {
    const t = line ? line + " " + w : w;
    if (g.measureText(t).width > maxW && line) {
      g.fillText(line, x, yy); yy += lh; line = w;
      if (++count >= 2) break;
    } else line = t;
  }
  if (count < 3) g.fillText(line, x, yy);
}
export interface DepthRenderResponse {
  log_depth_b64: string;
  reflectivity_b64: string;
  width: number; height: number;
  fx: number; fy: number; cx: number; cy: number;
  z_min: number; z_max: number;
  valid_fraction: number;
}

export async function renderDepth(
  baseUrl: string,
  camera: EnsembleRequest["camera"],
  outW = 160, outH = 90, seed = 7,
  shaftCenter: [number, number] = [-40, -20],
  spiralStyle: "interior_ramp" | "exterior_coil" = "interior_ramp",
  dataset: DatasetKind = "khafre",
  hypothesis: "megalithic" | "karst" | "artifact" | "bedrock_null" = "megalithic"
): Promise<DepthRenderResponse> {
  const res = await fetch(`${baseUrl}/api/khafre/depth`, {
    method: "POST",
    headers: apiHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ camera, out_width: outW, out_height: outH, seed,
      shaft_center_xy: shaftCenter, spiral_style: spiralStyle, dataset, hypothesis }),
  });
  if (!res.ok) throw new Error(`depth render failed: ${res.status}`);
  return res.json();
}

/** Decode a data-URL grayscale PNG into bytes via an offscreen canvas. */
export async function decodeGrayscalePng(uri: string): Promise<{ data: Uint8ClampedArray; width: number; height: number }> {
  const img = await new Promise<HTMLImageElement>((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = reject;
    im.src = uri;
  });
  const cv = document.createElement("canvas");
  cv.width = img.naturalWidth; cv.height = img.naturalHeight;
  const g = cv.getContext("2d", { willReadFrequently: true })!;
  g.drawImage(img, 0, 0);
  const { data, width, height } = g.getImageData(0, 0, cv.width, cv.height);
  return { data, width, height };
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
