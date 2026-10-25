/**
 * duat/apps/web/src/components/EpistemicLensViewport.tsx
 * Interactive 3-Tier Epistemic Stack Shader & Telemetry HUD.
 * GLSL uses the OFFICIAL Atlas inverted log-depth (255 = zMin, 0 = empty).
 */
import React, { useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Canvas, useFrame, useThree, ThreeEvent } from "@react-three/fiber";
import { Layers, Eye, Sparkles, Radar, Sliders, Crosshair, ShieldAlert } from "lucide-react";
import {
  postEnsemble, createJob, getJob, cancelJob, listJobs, fetchPricing, estimateCost,
  exportComparisonCard, decodeGrayscalePng, decodeLogDepthByte, fetchDegradation,
  uploadMesh, apiHeaders,
} from "../lib/api";
import type {
  EnsembleRequest, EnsembleResponse, JobRecord, PricingInfo, DatasetKind,
  InstrumentRig, DegradationResponse, MeshUploadResponse,
} from "../lib/api";

const VERTEX_SHADER = /*glsl*/ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const FRAGMENT_SHADER = /*glsl*/ `
precision highp float;
varying vec2 vUv;
uniform sampler2D uRgbTexture;
uniform sampler2D uDepthTexture;
uniform sampler2D uReflectivityTexture;
uniform sampler2D uDisagreementTexture;
uniform float uMeasuredWeight;
uniform float uInferredWeight;
uniform float uImaginedWeight;
uniform float uShimmerIntensity;
uniform vec2 uMouseUv;
uniform float uLoupeRadius;
uniform int uLoupeMode;
uniform float uZMin;
uniform float uZMax;
uniform float uTime;
const vec3 COLOR_OBSIDIAN = vec3(0.035, 0.039, 0.059);
const vec3 COLOR_MEASURED = vec3(0.0, 0.941, 1.0);
const vec3 COLOR_INFERRED = vec3(1.0, 0.722, 0.0);
const vec3 COLOR_SHIMMER  = vec3(0.204, 0.827, 0.600); // #34D399 emerald (Tier 3: Imagined/variance)

// OFFICIAL Atlas inverted log-depth: 255 = zMin (near), 0 = empty.
float decodePlanarDepth(float rawSample, out bool isEmpty) {
  if (rawSample < (0.5 / 255.0)) { isEmpty = true; return 0.0; }
  isEmpty = false;
  float p = rawSample * 255.0;
  return uZMin * pow(uZMax / uZMin, 1.0 - p / 255.0);
}
vec3 renderTier1Measured(vec2 uv, float planarZ, bool isEmpty) {
  if (isEmpty) {
    float hatch = step(0.92, fract((uv.x + uv.y) * 60.0));
    return COLOR_OBSIDIAN + vec3(0.08, 0.02, 0.04) * hatch;
  }
  float depthNorm = clamp((planarZ - uZMin) / (uZMax - uZMin), 0.0, 1.0);
  float contour = smoothstep(0.88, 0.98, abs(sin(planarZ * 3.14159 / 1.5)));
  vec2 grid = abs(fract(uv * vec2(64.0, 36.0) - 0.5) - 0.5) / fwidth(uv * vec2(64.0, 36.0));
  float gridLine = 1.0 - min(min(grid.x, grid.y), 1.0);
  vec3 baseDepth = mix(COLOR_MEASURED * 0.85, COLOR_OBSIDIAN, depthNorm * 0.8);
  return baseDepth + COLOR_MEASURED * (contour * 0.65 + gridLine * 0.25);
}
vec3 renderTier2Inferred(float refl, bool isEmpty) {
  if (isEmpty) return COLOR_OBSIDIAN;
  vec3 cold = vec3(0.05, 0.22, 0.38);
  vec3 mid  = COLOR_MEASURED * 0.6;
  vec3 hot  = COLOR_INFERRED;
  return refl < 0.5 ? mix(cold, mid, refl * 2.0) : mix(mid, hot, (refl - 0.5) * 2.0);
}
void main() {
  float rawDepth = texture2D(uDepthTexture, vUv).r;
  bool isEmpty = false;
  float planarZ = decodePlanarDepth(rawDepth, isEmpty);
  vec3 rgbImagined = texture2D(uRgbTexture, vUv).rgb;
  float refl = texture2D(uReflectivityTexture, vUv).r;
  float disagreement = texture2D(uDisagreementTexture, vUv).r;
  vec3 tier1 = renderTier1Measured(vUv, planarZ, isEmpty);
  vec3 tier2 = renderTier2Inferred(refl, isEmpty);
  // If isEmpty is true (0-masked depth), show rgbImagined if non-black (e.g. bedrock_null unconditioned run)
  bool hasRgb = dot(rgbImagined, vec3(1.0)) > 0.01;
  vec3 tier3 = (isEmpty && !hasRgb) ? COLOR_OBSIDIAN : rgbImagined;
  float totalW = max(uMeasuredWeight + uInferredWeight + uImaginedWeight, 0.001);
  vec3 composite = (tier1 * uMeasuredWeight + tier2 * uInferredWeight + tier3 * uImaginedWeight) / totalW;
  if ((!isEmpty || hasRgb) && uShimmerIntensity > 0.01 && disagreement > 0.18) {
    float pulse = 0.5 + 0.5 * sin(uTime * 5.0 + (vUv.x * 45.0) + (vUv.y * 45.0));
    float wave = smoothstep(0.18, 0.85, disagreement);
    vec3 shimmerColor = mix(COLOR_SHIMMER, COLOR_INFERRED, pulse * 0.4);
    composite = mix(composite, shimmerColor, wave * uShimmerIntensity * (0.45 + 0.35 * pulse));
  }
  if (uLoupeMode > 0) {
    vec2 aspectCorrect = vec2(16.0 / 9.0, 1.0);
    float dist = length((vUv - uMouseUv) * aspectCorrect);
    if (dist < uLoupeRadius) {
      vec3 loupeView = composite;
      if (uLoupeMode == 1) loupeView = tier1;
      else if (uLoupeMode == 2) loupeView = tier2;
      else if (uLoupeMode == 3) loupeView = mix(COLOR_OBSIDIAN, COLOR_SHIMMER, disagreement);
      float rim = smoothstep(uLoupeRadius - 0.005, uLoupeRadius, dist);
      composite = mix(loupeView, COLOR_MEASURED, rim * 0.9);
    }
  }
  gl_FragColor = vec4(composite, 1.0);
}
`;

type Hypothesis = "megalithic" | "karst" | "artifact" | "bedrock_null";

function useProceduralKhafreTextures(hypothesis: Hypothesis) {
  return useMemo(() => {
    const width = 512, height = 288;
    const depthData = new Uint8Array(width * height * 4);
    const reflData = new Uint8Array(width * height * 4);
    const rgbData = new Uint8Array(width * height * 4);
    const disData = new Uint8Array(width * height * 4);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        const nx = (x / width) * 2.0 - 1.0;
        const ny = (y / height) * 2.0 - 1.0;
        const r = Math.sqrt(nx * nx + ny * ny);
        const angle = Math.atan2(ny, nx);
        if (r < 0.14) { depthData[idx] = 0; depthData[idx + 3] = 255; continue; }
        // Inverted log-depth: bright (255) = near wall, dark = far.
        const spiral = Math.sin(10.0 / Math.max(r, 0.14) + angle * 2.0);
        const logDepthByte = Math.min(255, Math.max(1, Math.round((1.0 - r * 0.68 + spiral * 0.08) * 230)));
        depthData[idx] = depthData[idx + 1] = depthData[idx + 2] = logDepthByte;
        depthData[idx + 3] = 255;
        const reflByte = Math.min(255, Math.max(0, Math.round((0.45 + 0.4 * spiral) * 255)));
        reflData[idx] = reflData[idx + 1] = reflData[idx + 2] = reflByte;
        reflData[idx + 3] = 255;
        const falloff = Math.min(1.0, Math.pow(r, 1.35) * 1.4);
        const masonry = Math.sin(x * 0.25) * Math.sin(y * 0.25);
        if (hypothesis === "megalithic") {
          rgbData[idx] = Math.round((175 + spiral * 35 + masonry * 15) * falloff);
          rgbData[idx + 1] = Math.round((152 + spiral * 28 + masonry * 12) * falloff);
          rgbData[idx + 2] = Math.round((118 + spiral * 20) * falloff);
        } else if (hypothesis === "karst") {
          rgbData[idx] = Math.round((95 + spiral * 25) * falloff);
          rgbData[idx + 1] = Math.round((135 + spiral * 35) * falloff);
          rgbData[idx + 2] = Math.round((145 + spiral * 40) * falloff);
        } else if (hypothesis === "bedrock_null") {
          // Null prior: unbroken bedrock, no dressing, no falloff tricks — dark strata.
          const strata = 0.5 + 0.5 * Math.sin(y * 0.08);
          rgbData[idx] = Math.round((104 + strata * 14) * falloff);
          rgbData[idx + 1] = Math.round((96 + strata * 12) * falloff);
          rgbData[idx + 2] = Math.round((88 + strata * 10) * falloff);
        } else {
          const stepQuant = Math.floor(logDepthByte / 24) * 24;
          rgbData[idx] = Math.round(stepQuant * 0.4);
          rgbData[idx + 1] = Math.round(stepQuant * 0.85);
          rgbData[idx + 2] = stepQuant;
        }
        rgbData[idx + 3] = 255;
        const edgeDisagreement = Math.pow(Math.max(0, Math.cos(10.0 / r + angle * 2.0)), 3);
        const disByte = Math.min(255, Math.round(edgeDisagreement * 240));
        disData[idx] = disData[idx + 1] = disData[idx + 2] = disByte;
        disData[idx + 3] = 255;
      }
    }
    const makeTex = (arr: Uint8Array) => {
      const tex = new THREE.DataTexture(arr as Uint8Array<ArrayBuffer>, width, height, THREE.RGBAFormat);
      tex.needsUpdate = true;
      return tex;
    };

    return { depthTex: makeTex(depthData), reflTex: makeTex(reflData), rgbTex: makeTex(rgbData), disTex: makeTex(disData),
      depthArr: depthData, reflArr: reflData, disArr: disData, w: width, h: height };
  }, [hypothesis]);
}

export interface SampleSource {
  depth: Uint8Array | Uint8ClampedArray; refl: Uint8Array | Uint8ClampedArray; dis: Uint8Array | Uint8ClampedArray;
  w: number; h: number; zMin: number; zMax: number;
}

/** Nearest-pixel sample of the ACTUAL layer bytes (procedural arrays or decoded PNGs). */
export function sampleLayers(src: SampleSource, u: number, v: number): PixelTelemetry {
  const x = Math.min(src.w - 1, Math.max(0, Math.floor(u * src.w)));
  const y = Math.min(src.h - 1, Math.max(0, Math.floor(v * src.h)));
  const i = (y * src.w + x) * 4;
  const db = src.depth[i];
  if (db === 0) return { u: Math.round(u * 1280), v: Math.round((1 - v) * 720), planarZ: 0, reflectivity: 0, disagreement: 0 };
  return {
    u: Math.round(u * 1280), v: Math.round((1 - v) * 720),
    planarZ: Number(decodeLogDepthByte(db, src.zMin, src.zMax).toFixed(2)),
    reflectivity: Number((src.refl[i] / 255).toFixed(2)),
    disagreement: Number((src.dis[i] / 255).toFixed(2)),
  };
}

export type PixelTelemetry = { u: number; v: number; planarZ: number; reflectivity: number; disagreement: number };

export type LayerUris = { rgb: string; depth: string; refl: string; dis: string };
export type RunMode = "mock" | "live-probe" | "live";

interface EpistemicQuadProps {
  measuredWeight: number; inferredWeight: number; imaginedWeight: number;
  shimmerIntensity: number; loupeMode: number; loupeRadius: number;
  layers: LayerUris | null;
  proc: { depthTex: THREE.Texture; reflTex: THREE.Texture; rgbTex: THREE.Texture; disTex: THREE.Texture };
  sampleSrc: SampleSource | null;
  onInspectPixel: (t: PixelTelemetry) => void;
  onPinPixel: (t: PixelTelemetry) => void;
}

/** Fit the 16x9 epistemic quad to any container: zoom = min(w/16, h/9). */
const ZoomFit: React.FC = () => {
  const camera = useThree((s) => s.camera);
  const size = useThree((s) => s.size);
  React.useEffect(() => {
    if (camera instanceof THREE.OrthographicCamera) {
      camera.zoom = Math.min(size.width / 16, size.height / 9);
      camera.updateProjectionMatrix();
    }
  }, [camera, size]);
  return null;
};

const EpistemicShaderQuad: React.FC<EpistemicQuadProps> = (props) => {  const { measuredWeight, inferredWeight, imaginedWeight, shimmerIntensity, loupeMode, loupeRadius, onInspectPixel, onPinPixel, layers, proc, sampleSrc } = props;
  const matRef = useRef<THREE.ShaderMaterial>(null);
  const lastTelemetry = useRef<PixelTelemetry>({ u: 640, v: 360, planarZ: 14.2, reflectivity: 0.72, disagreement: 0.34 });
  const fallback = proc;
  const [live, setLive] = useState<{ rgb: THREE.Texture; depth: THREE.Texture; refl: THREE.Texture; dis: THREE.Texture } | null>(null);

  // Backend layers arrive as data-URI PNGs (mock or live job output); otherwise procedural fallback.
  React.useEffect(() => {
    if (!layers) { setLive(null); return; }
    const loader = new THREE.TextureLoader();
    loader.setCrossOrigin("anonymous");
    const loaded = {
      rgb: loader.load(layers.rgb),
      depth: loader.load(layers.depth),
      refl: loader.load(layers.refl),
      dis: loader.load(layers.dis),
    };
    setLive(loaded);
    return () => {
      loaded.rgb.dispose();
      loaded.depth.dispose();
      loaded.refl.dispose();
      loaded.dis.dispose();
    };
  }, [layers]);

  const tex = live ?? { rgb: fallback.rgbTex, depth: fallback.depthTex, refl: fallback.reflTex, dis: fallback.disTex };
  const uniforms = useMemo(() => ({
    uRgbTexture: { value: tex.rgb }, uDepthTexture: { value: tex.depth },
    uReflectivityTexture: { value: tex.refl }, uDisagreementTexture: { value: tex.dis },
    uMeasuredWeight: { value: measuredWeight }, uInferredWeight: { value: inferredWeight },
    uImaginedWeight: { value: imaginedWeight }, uShimmerIntensity: { value: shimmerIntensity },
    uMouseUv: { value: new THREE.Vector2(0.5, 0.5) }, uLoupeRadius: { value: 0.18 },
    uLoupeMode: { value: loupeMode }, uZMin: { value: 0.5 }, uZMax: { value: 35.0 }, uTime: { value: 0.0 },
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }), []);

  useFrame((state) => {
    if (!matRef.current) return;
    const u = matRef.current.uniforms;
    u.uTime.value = state.clock.elapsedTime;
    u.uRgbTexture.value = tex.rgb; u.uDepthTexture.value = tex.depth;
    u.uReflectivityTexture.value = tex.refl; u.uDisagreementTexture.value = tex.dis;
    u.uMeasuredWeight.value = measuredWeight; u.uInferredWeight.value = inferredWeight;
    u.uImaginedWeight.value = imaginedWeight; u.uShimmerIntensity.value = shimmerIntensity;
    u.uLoupeMode.value = loupeMode; u.uLoupeRadius.value = loupeRadius;
    u.uZMin.value = sampleSrc?.zMin ?? 0.5;
    u.uZMax.value = sampleSrc?.zMax ?? 35.0;
  });

  const handlePointerMove = (e: ThreeEvent<PointerEvent>) => {
    if (!e.uv || !matRef.current) return;
    matRef.current.uniforms.uMouseUv.value.copy(e.uv);
    // Telemetry is decoded from the ACTUAL layer bytes, never synthesized.
    const t = sampleSrc
      ? sampleLayers(sampleSrc, e.uv.x, e.uv.y)
      : { u: Math.round(e.uv.x * 1280), v: Math.round((1.0 - e.uv.y) * 720), planarZ: 0, reflectivity: 0, disagreement: 0 };
    lastTelemetry.current = t;
    onInspectPixel(t);
  };

  return (
    <mesh onPointerMove={handlePointerMove} onClick={() => onPinPixel(lastTelemetry.current)}>
      <planeGeometry args={[16, 9]} />
      <shaderMaterial ref={matRef} vertexShader={VERTEX_SHADER} fragmentShader={FRAGMENT_SHADER} uniforms={uniforms} />
    </mesh>
  );
};

export interface EpistemicLensProps {
  apiBaseUrl: string;
  active?: boolean;
  sharedPose?: { position: [number, number, number]; quaternion: [number, number, number, number]; camIdx?: number };
  onPoseChange?: (pose: { position: [number, number, number]; quaternion: [number, number, number, number]; camIdx?: number }) => void;
}

export const EpistemicLensViewport: React.FC<EpistemicLensProps> = ({ apiBaseUrl, active, sharedPose }) => {
  const [measuredWeight, setMeasuredWeight] = useState(0.15);
  const [inferredWeight, setInferredWeight] = useState(0.10);
  const [imaginedWeight, setImaginedWeight] = useState(0.75);
  const [shimmerIntensity, setShimmerIntensity] = useState(0.65);
  const [loupeMode, setLoupeMode] = useState(1);
  const [loupeRadius, setLoupeRadius] = useState(0.18);
  const [dataset, setDataset] = useState<DatasetKind>("khafre");
  const [hypothesis, setHypothesis] = useState<Hypothesis>("megalithic");
  const [instrumentRig, setInstrumentRig] = useState<InstrumentRig>("drop_flare");
  const [voxelSize, setVoxelSize] = useState(0.25);
  const [sensorNoise, setSensorNoise] = useState(0.18);
  const [spiral, setSpiral] = useState<"interior_ramp" | "exterior_coil">("interior_ramp");
  const [mode, setMode] = useState<RunMode | "live-probe" | "live">("mock");
  const [armingLive, setArmingLive] = useState<null | RunMode>(null);
  const [layers, setLayers] = useState<LayerUris | null>(null);
  const [layersBusy, setLayersBusy] = useState(false);
  const [activeJob, setActiveJob] = useState<{ id: string; progress: number } | null>(null);
  const [pricing, setPricing] = useState<PricingInfo | null>(null);
  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [telemetry, setTelemetry] = useState<PixelTelemetry>({ u: 640, v: 360, planarZ: 14.2, reflectivity: 0.72, disagreement: 0.34 });
  const [pinned, setPinned] = useState<PixelTelemetry | null>(null);
  const [compare, setCompare] = useState<{ status: "idle" | "loading" | "error" | "done"; thumbs: { h: Hypothesis; uri: string }[] }>({ status: "idle", thumbs: [] });
  const [exporting, setExporting] = useState(false);
  const [sampleSrc, setSampleSrc] = useState<SampleSource | null>(null);
  const [slices, setSlices] = useState<{ filename: string; anomaly_b64: string; peak_count: number; peak_positions_norm: number[]; masked_fraction: number }[]>([]);
  const [sliceBusy, setSliceBusy] = useState(false);
  const [sliceError, setSliceError] = useState<string | null>(null);
  const [degradation, setDegradation] = useState<DegradationResponse | null>(null);
  const [meshStats, setMeshStats] = useState<MeshUploadResponse | null>(null);
  const [meshBusy, setMeshBusy] = useState(false);
  const [meshError, setMeshError] = useState<string | null>(null);
  const [lastNBV, setLastNBV] = useState<{ target_z_m: number; predicted_variance_reduction_pct: number } | null>(null);
  const proc = useProceduralKhafreTextures(hypothesis);

  // Procedural fallback sampling: same arrays the shader renders.
  React.useEffect(() => {
    if (layers) return;
    setSampleSrc({ depth: proc.depthArr, refl: proc.reflArr, dis: proc.disArr,
      w: proc.w, h: proc.h, zMin: 0.5, zMax: 35.0 });
  }, [proc, layers]);

  const buildSampleSrc = async (res: EnsembleResponse): Promise<void> => {
    try {
      const [d, r, s] = await Promise.all([
        decodeGrayscalePng(res.layers.measured_log_depth_b64),
        decodeGrayscalePng(res.layers.inferred_reflectivity_b64),
        decodeGrayscalePng(res.layers.disagreement_map_b64)]);
      setSampleSrc({ depth: d.data, refl: r.data, dis: s.data,
        w: d.width, h: d.height, zMin: res.z_min, zMax: res.z_max });
    } catch {
      setSampleSrc(null);
    }
  };

  React.useEffect(() => {
    fetchPricing(apiBaseUrl).then(setPricing).catch(() => setPricing(null));
    listJobs(apiBaseUrl, 8).then((r) => setJobs(r.jobs)).catch(() => {});
    fetchDegradation(apiBaseUrl).then(setDegradation).catch(() => setDegradation(null));
  }, [apiBaseUrl]);

  const activePosition = sharedPose?.position ?? [-38.5, -20.0, -45.0];
  const activeQuaternion = sharedPose?.quaternion ?? [0, 0, 0, 1];
  const activeCamIdx = sharedPose?.camIdx ?? 3;

  const baseReq = (h: Hypothesis, liveMode: "mock" | "live", seeds: number[]): EnsembleRequest => ({
    camera: { pose: { position: activePosition, quaternion: activeQuaternion } },
    dataset,
    hypothesis: h,
    instrument_rig: instrumentRig,
    seeds,
    mode: liveMode,
    voxel_size: voxelSize,
    sensor_noise_sigma_m: sensorNoise,
    spiral_style: spiral,
  });

  const layersFromResponse = (res: EnsembleResponse): LayerUris => ({
    rgb: res.layers.imagined_consensus_rgb_b64, depth: res.layers.measured_log_depth_b64,
    refl: res.layers.inferred_reflectivity_b64, dis: res.layers.disagreement_map_b64,
  });

  const loadLayers = async (m: RunMode | "live-probe" | "live") => {
    setLayersBusy(true);
    try {
      if (m === "mock") {
        const res = await postEnsemble(apiBaseUrl, baseReq(hypothesis, "mock", [7, 42]));
        setLayers(layersFromResponse(res));
        setLastNBV({ target_z_m: res.next_best_view.target_z_m,
          predicted_variance_reduction_pct: res.next_best_view.predicted_variance_reduction_pct });
        await buildSampleSrc(res);
      } else {
        const seeds = m === "live-probe" ? [7] : [7, 42];
        const { id } = await createJob(apiBaseUrl, "ensemble", baseReq(hypothesis, "live", seeds));
        setActiveJob({ id, progress: 0 });
        for (let i = 0; i < 120; i++) {
          await new Promise((r) => setTimeout(r, 3000));
          const job = await getJob(apiBaseUrl, id);
          setActiveJob({ id, progress: job.progress });
          if (job.status === "done" && job.result) {
            const res = job.result as EnsembleResponse;
            setLayers(layersFromResponse(res));
            setLastNBV({ target_z_m: res.next_best_view.target_z_m,
              predicted_variance_reduction_pct: res.next_best_view.predicted_variance_reduction_pct });
            await buildSampleSrc(res);
            break;
          }
          if (job.status === "failed" || job.status === "cancelled") throw new Error(job.error?.detail ?? job.status);
        }
        setActiveJob(null);
      }
    } catch {
      if (m === "mock") { setLayers(null); setSampleSrc(null); }
    } finally {
      setLayersBusy(false);
      listJobs(apiBaseUrl, 8).then((r) => setJobs(r.jobs)).catch(() => {});
    }
  };

  const requestMode = (m: RunMode | "live-probe" | "live") => {
    if (m !== "mock" && mode === "mock") { setArmingLive(m as RunMode); return; } // spend gate
    setArmingLive(null);
    setMode(m);
    void loadLayers(m);
  };

  const runHypothesisCompare = async () => {
    if (!pinned) return;
    setCompare({ status: "loading", thumbs: [] });
    try {
      const hs: Hypothesis[] = ["megalithic", "karst", "artifact", "bedrock_null"];
      const live = mode !== "mock";
      const results = await Promise.all(hs.map(async (h) => {
        const req = baseReq(h, live ? "live" : "mock", live ? [7] : [7, 42]);
        const res = live
          ? await (async () => {
              const { id } = await createJob(apiBaseUrl, "ensemble", req);
              for (let i = 0; i < 120; i++) {
                await new Promise((r) => setTimeout(r, 3000));
                const job = await getJob(apiBaseUrl, id);
                if (job.status === "done" && job.result) return job.result as EnsembleResponse;
                if (job.status === "failed" || job.status === "cancelled") throw new Error(job.error?.detail ?? job.status);
              }
              throw new Error("compare timed out");
            })()
          : await postEnsemble(apiBaseUrl, req);
        return { h, uri: res.layers.imagined_consensus_rgb_b64 };
      }));
      setCompare({ status: "done", thumbs: results });
    } catch {
      setCompare({ status: "error", thumbs: [] });
    }
  };

  const runExport = async () => {
    setExporting(true);
    try {
      await exportComparisonCard(apiBaseUrl, baseReq(hypothesis, mode === "mock" ? "mock" : "live", mode === "live-probe" ? [7] : [7, 42]));
    } catch {
      alert("Export failed — is the backend running?");
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="flex flex-col lg:flex-row w-full min-h-[720px] bg-[#090A0F] text-slate-100 font-sans border border-slate-800 rounded-xl overflow-hidden shadow-2xl">
      <div className="relative flex-1 bg-black flex flex-col justify-start min-h-[480px]">
        <div className="z-10 flex flex-wrap items-center justify-between gap-2 px-4 py-2.5 bg-[#090A0F]/90 backdrop-blur-md border-b border-slate-800 text-xs">
          <div className="flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-[#FFB800]" />
            <span className="font-semibold tracking-wider uppercase text-slate-200">
              MILIMO DUAT // MODEL PREDICTION — CONDITIONED ON {dataset === "khafre" ? "KHAFRE SAR" : dataset === "nfc" ? "SCANPYRAMIDS NFC" : dataset === "big-void" ? "SCANPYRAMIDS BIG VOID" : "UPLOADED MESH"} GEOMETRY
            </span>
          </div>
          <div className="flex items-center gap-4 font-mono text-[11px] text-slate-400">
            <span>TASK: atlasChisel</span>
            <span>VOXEL_GAUGE: {voxelSize.toFixed(2)}m</span>
            <span className="text-slate-500">RIG:{instrumentRig === "drop_flare" ? "FLARE 40m" : instrumentRig === "structured_laser" ? "LASER 25m" : "LED 6m"}</span>
            <span className="text-[#00F0FF]">
              CAM #{String(activeCamIdx + 1).padStart(2, "0")}/32 (Z={activePosition[2].toFixed(1)}m)
            </span>
          </div>
        </div>
        <div className="relative w-full cursor-crosshair overflow-hidden bg-black" style={{ aspectRatio: "16 / 9" }}>
          <Canvas orthographic flat camera={{ position: [0, 0, 10] }} frameloop={active === false ? "never" : "always"} style={{ position: "absolute", inset: 0 }}>
            <ZoomFit />
            <EpistemicShaderQuad measuredWeight={measuredWeight} inferredWeight={inferredWeight}
              imaginedWeight={imaginedWeight} shimmerIntensity={shimmerIntensity} loupeMode={loupeMode}
              loupeRadius={loupeRadius} layers={layers} proc={proc} sampleSrc={sampleSrc}
              onInspectPixel={setTelemetry} onPinPixel={setPinned} />
          </Canvas>
        </div>
        <div className="z-10 grid grid-cols-2 sm:grid-cols-4 gap-2 px-4 py-2.5 bg-[#090A0F]/95 border-t border-slate-800 font-mono text-xs">
          <div className="flex items-center gap-2"><Crosshair className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-slate-400">PIXEL:</span><span className="text-white">({telemetry.u}, {telemetry.v})</span></div>
          <div><span className="text-[#00F0FF]">MEASURED Z: </span>
            <span className="text-white">{telemetry.planarZ === 0 ? "0.0m (EMPTY VOID)" : `${telemetry.planarZ}m`}</span></div>
          <div><span className="text-[#FFB800]">DIELECTRIC: </span><span className="text-white">{telemetry.reflectivity}</span></div>
          <div><span className="text-[#34D399]">ENSEMBLE SPLIT: </span><span className="text-white">{Math.round(telemetry.disagreement * 100)}%</span></div>
        </div>
      </div>
      <div className="w-full lg:w-96 bg-[#0D0F17] border-l border-slate-800 p-5 flex flex-col gap-6 lg:max-h-[calc(100vh-160px)] lg:overflow-y-auto">
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
            <Layers className="w-4 h-4 text-[#00F0FF]" /><span>Epistemic Stack Mixer</span></div>
          <div className="space-y-4 bg-[#090A0F] p-4 rounded-lg border border-slate-800/80">
            {([
              ["Tier 1: Measured (SAR Depth)", measuredWeight, setMeasuredWeight, "#00F0FF"],
              ["Tier 2: Inferred (Dielectric Proxy)", inferredWeight, setInferredWeight, "#FFB800"],
              ["Tier 3: Imagined (Atlas Chisel RGB)", imaginedWeight, setImaginedWeight, "#34d399"],
            ] as const).map(([label, val, set, color]) => (
              <div key={label}>
                <div className="flex justify-between text-xs mb-1">
                  <span style={{ color }} className="font-medium">{label}</span>
                  <span className="font-mono">{Math.round(val * 100)}%</span></div>
                <input type="range" min="0" max="1" step="0.01" value={val}
                  onChange={(e) => set(parseFloat(e.target.value))} className="w-full" style={{ accentColor: color }} />
              </div>
            ))}
            <div className="pt-2 border-t border-slate-800">
              <div className="flex justify-between text-xs mb-1">
                <span className="text-[#34D399] font-medium flex items-center gap-1"><Sparkles className="w-3.5 h-3.5" /> Variance Shimmer</span>
                <span className="font-mono">{Math.round(shimmerIntensity * 100)}%</span></div>
              <input type="range" min="0" max="1" step="0.01" value={shimmerIntensity}
                onChange={(e) => setShimmerIntensity(parseFloat(e.target.value))} className="w-full accent-[#34D399]" />
            </div>
            <div className="pt-2 border-t border-slate-800">
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-300 font-medium">Conditioning dataset</span>
                <span className="font-mono text-[10px] text-slate-500">{dataset === "khafre" ? "contested" : dataset === "uploaded_mesh" ? "custom" : "validated"}</span>
              </div>
              <div className="grid grid-cols-2 gap-1.5 text-[11px] mb-2">
                {([
                  ["khafre", "Khafre Shaft #2"],
                  ["nfc", "ScanPyramids NFC"],
                  ["big-void", "ScanPyramids Void"],
                  ["uploaded_mesh", "Uploaded Mesh"],
                ] as [DatasetKind, string][]).map(([ds, label]) => (
                  <button key={ds} onClick={() => setDataset(ds)}
                    className={`px-2 py-1.5 rounded-md border transition text-left truncate ${dataset === ds
                      ? "bg-slate-800 border-[#00F0FF] text-white font-medium"
                      : "bg-[#090A0F] border-slate-800 text-slate-400 hover:border-slate-700"}`}>{label}</button>
                ))}
              </div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-300 font-medium">Atlas render mode</span>
                <span className="font-mono text-[11px] text-slate-500">{layersBusy ? "working…" : layers ? "layers loaded" : "procedural fallback"}</span>
              </div>
              <div className="grid grid-cols-3 gap-1.5 text-[11px]">
                {([["mock", "Simulated"], ["live-probe", "Live probe"], ["live", "Live full"]] as const).map(([m, label]) => (
                  <button key={m} onClick={() => requestMode(m)}
                    className={`px-2 py-1.5 rounded-md border transition ${mode === m
                      ? "bg-slate-800 border-[#00F0FF] text-white font-medium"
                      : "bg-[#090A0F] border-slate-800 text-slate-400 hover:border-slate-700"}`}>{label}</button>
                ))}
              </div>
              {armingLive && (
                <div className="mt-2 p-2 rounded border border-[#FFB800]/60 bg-[#FFB800]/10 text-[11px] text-slate-200">
                  <div className="mb-1.5">Live Atlas will bill tasks. {estimateCost(armingLive === "live-probe" ? 1 : 2, pricing, 1)}.</div>
                  <div className="flex gap-1.5">
                    <button onClick={() => { const m = armingLive; setArmingLive(null); setMode(m); void loadLayers(m); }}
                      className="px-2 py-1 rounded bg-[#FFB800] text-slate-950 font-semibold">Confirm spend</button>
                    <button onClick={() => setArmingLive(null)} className="px-2 py-1 rounded border border-slate-700">Cancel</button>
                  </div>
                </div>
              )}
              {mode !== "mock" && !armingLive && (
                <div className="mt-2 font-mono text-[11px] text-slate-400">
                  {estimateCost(mode === "live-probe" ? 1 : 2, pricing, 1)} per load
                </div>
              )}
              {activeJob && (
                <div className="mt-2">
                  <div className="flex justify-between text-[11px] font-mono text-slate-400 mb-1">
                    <span>job {activeJob.id.slice(0, 12)}… {Math.round(activeJob.progress * 100)}%</span>
                    <button onClick={() => cancelJob(apiBaseUrl, activeJob.id).catch(() => {}).finally(() => setActiveJob(null))}
                      className="text-[#FF2D55] hover:underline">Cancel</button>
                  </div>
                  <div className="h-1.5 rounded bg-slate-800 overflow-hidden">
                    <div className="h-full bg-[#00F0FF] transition-all" style={{ width: `${Math.round(activeJob.progress * 100)}%` }} />
                  </div>
                </div>
              )}
              {layersBusy && !activeJob && <div className="mt-2 font-mono text-[11px] text-slate-400">Rendering ensemble…</div>}
            </div>
            <div className="pt-2 border-t border-slate-800 flex gap-2">
              <button onClick={runExport} disabled={exporting}
                className="flex-1 px-3 py-2 rounded-md bg-slate-800 border border-slate-700 text-xs font-medium hover:bg-slate-700 transition disabled:opacity-50">
                {exporting ? "Exporting…" : "Export card (current view)"}
              </button>
              <button onClick={() => setLayers(null)}
                className="px-3 py-2 rounded-md border border-slate-800 text-xs text-slate-400 hover:border-slate-700 transition">Reset</button>
            </div>
            {jobs.length > 0 && (
              <details className="pt-2 border-t border-slate-800 text-[11px] font-mono text-slate-400">
                <summary className="cursor-pointer hover:text-slate-200">Recent jobs ({jobs.length})</summary>
                <div className="mt-1 space-y-0.5 max-h-28 overflow-y-auto">
                  {jobs.map((j) => (
                    <div key={j.id} className="flex justify-between gap-2">
                      <span className="truncate">{j.id.slice(0, 12)}… {j.kind}</span>
                      <span className={j.status === "done" ? "text-[#34D399]" : j.status === "failed" ? "text-[#FF2D55]" : "text-[#FFB800]"}>{j.status} {Math.round((j.progress || 0) * 100)}%</span>
                    </div>
                  ))}
                </div>
              </details>
            )}
          </div>
        </div>
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
            <Eye className="w-4 h-4 text-[#FFB800]" /><span>Cursor X-Ray Loupe</span></div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            {[{ id: 1, label: "Measured Wireframe" }, { id: 2, label: "Dielectric Proxy" },
              { id: 3, label: "Ensemble Split" }, { id: 0, label: "Disable Loupe" }].map((mode) => (
              <button key={mode.id} onClick={() => setLoupeMode(mode.id)}
                className={`px-3 py-2 rounded-md border text-left transition ${loupeMode === mode.id
                  ? "bg-slate-800 border-[#00F0FF] text-white font-medium"
                  : "bg-[#090A0F] border-slate-800 text-slate-400 hover:border-slate-700"}`}>{mode.label}</button>
            ))}
          </div>
          <div className="mt-3">
            <div className="flex justify-between text-xs mb-1">
              <span className="text-slate-400">Loupe radius</span>
              <span className="font-mono text-slate-300">{Math.round(loupeRadius * 100)}%</span></div>
            <input type="range" min="0.06" max="0.30" step="0.01" value={loupeRadius}
              onChange={(e) => setLoupeRadius(parseFloat(e.target.value))} className="w-full accent-[#00F0FF]" />
          </div>
        </div>
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
            <Sliders className="w-4 h-4 text-[#34D399]" /><span>Hypothesis Prior (Same Geometry)</span></div>
          <div className="flex flex-col gap-2 text-xs">
            {([
              ["megalithic", "Megalithic Architecture", "Dressed limestone & granite spiral ramp, high-lumen drop flare", "press claim · unverified"],
              ["karst", "Phreatic Karst Dissolution", "Natural limestone sinkhole tube, wet travertine & water table", "geology · plausible"],
              ["artifact", "SAR Side-Lobe Echo Test", "Visualizes stepped voxel quantization & acoustic ghosting", "method artifact"],
              ["bedrock_null", "Unbroken Bedrock (Null)", "No void at all — Alvarez 1960s muon null result (0-mask depth)", "validated null"],
            ] as [Hypothesis, string, string, string][]).map(([key, title, desc, chip]) => (
              <button key={key} onClick={() => setHypothesis(key)}
                className={`p-3 rounded-lg border text-left transition ${hypothesis === key
                  ? "bg-slate-800/90 border-[#34D399] text-white" : "bg-[#090A0F] border-slate-800 text-slate-400 hover:border-slate-700"}`}>
                <div className="flex items-center justify-between gap-2">
                  <div className="font-semibold text-slate-100">{title}</div>
                  <div className="font-mono text-[9px] px-1.5 py-0.5 rounded border border-slate-700 text-slate-400 whitespace-nowrap">{chip}</div>
                </div>
                <div className="text-[11px] text-slate-400 mt-0.5">{desc}</div>
              </button>
            ))}
          </div>
        </div>
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
            <Sliders className="w-4 h-4 text-[#00F0FF]" /><span>Sensor, Rig & Voxel Gauge</span></div>
          <div className="space-y-4 bg-[#090A0F] p-4 rounded-lg border border-slate-800/80">
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-300 font-medium">Instrument lighting rig</span>
                <span className="font-mono text-[10px] text-[#00F0FF]">
                  {instrumentRig === "drop_flare" ? "zMax 40m" : instrumentRig === "structured_laser" ? "zMax 25m" : "zMax 6m"}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-1.5">
                {([
                  ["drop_flare", "Drop flare"],
                  ["structured_laser", "Laser sheet"],
                  ["borescope_led", "Borescope LED"],
                ] as [InstrumentRig, string][]).map(([rig, label]) => (
                  <button key={rig} onClick={() => setInstrumentRig(rig)}
                    className={`px-2 py-1.5 rounded-md border text-[11px] transition ${instrumentRig === rig
                      ? "bg-slate-800 border-[#00F0FF] text-white font-medium"
                      : "bg-[#090A0F] border-slate-800 text-slate-400 hover:border-slate-700"}`}>{label}</button>
                ))}
              </div>
            </div>
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-300 font-medium">Voxel size (scene gauge)</span>
                <span className="font-mono text-[#00F0FF]">{voxelSize.toFixed(2)}m</span></div>
              <input type="range" min="0" max="0.4" step="0.01" value={voxelSize}
                onChange={(e) => setVoxelSize(parseFloat(e.target.value))} className="w-full accent-[#00F0FF]" />
            </div>
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-300 font-medium">Sensor noise σ (posterior)</span>
                <span className="font-mono text-[#FFB800]">{sensorNoise.toFixed(2)}m</span></div>
              <input type="range" min="0" max="1" step="0.01" value={sensorNoise}
                onChange={(e) => setSensorNoise(parseFloat(e.target.value))} className="w-full accent-[#FFB800]" />
            </div>
            <p className="text-[11px] text-slate-500 leading-relaxed">Finer voxels preserve blockout detail; higher σ widens the posterior ensemble spread.</p>
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-300 font-medium">Spiral reading</span>
                <span className="font-mono text-[10px] text-slate-500">{spiral === "interior_ramp" ? "walkway" : "coil"}</span></div>
              <div className="grid grid-cols-2 gap-1.5">
                {(["interior_ramp", "exterior_coil"] as const).map((s) => (
                  <button key={s} onClick={() => setSpiral(s)}
                    title={s === "interior_ramp" ? "Interior walkway shelf (our reading)" : "Exterior wrap (claimants' diagram)"}
                    className={`px-2 py-1.5 rounded-md border text-[11px] transition ${spiral === s ? "bg-slate-800 border-[#00F0FF] text-white font-medium" : "bg-[#090A0F] border-slate-800 text-slate-400"}`}>
                    {s === "interior_ramp" ? "Interior ramp" : "Exterior coil"}</button>
                ))}
              </div>
            </div>
          </div>
        </div>
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
            <Layers className="w-4 h-4 text-[#00F0FF]" /><span>Sensor Degradation & Mesh Ingest</span></div>
          <div className="bg-[#090A0F] p-4 rounded-lg border border-slate-800/80 text-xs space-y-3">
            {degradation && (
              <div className="space-y-1 font-mono text-[10px]">
                <div className="flex justify-between text-slate-400 border-b border-slate-800 pb-1">
                  <span>SENSOR</span><span>VERTS / FACES</span><span>RMS ERROR</span>
                </div>
                {degradation.levels.map((lv) => (
                  <div key={lv.name} className="flex justify-between text-slate-300">
                    <span className="uppercase text-[#00F0FF]">{lv.name}</span>
                    <span>{lv.vertices.toLocaleString()} / {lv.faces.toLocaleString()}</span>
                    <span className="text-[#FFB800]">{lv.rms_displacement_m.toFixed(3)}m</span>
                  </div>
                ))}
              </div>
            )}
            <label className="block px-3 py-2 rounded-md border border-slate-700 text-center cursor-pointer hover:border-[#00F0FF] transition">
              {meshBusy ? "Ingesting SDF…" : "Upload surface mesh (.obj/.ply/.glb/.stl)"}
              <input type="file" accept=".obj,.ply,.glb,.stl,.off" className="hidden"
                onChange={async (e) => {
                  const f = e.target.files?.[0];
                  if (!f) return;
                  setMeshBusy(true); setMeshError(null);
                  try {
                    const stats = await uploadMesh(apiBaseUrl, f);
                    setMeshStats(stats);
                    setDataset("uploaded_mesh");
                  } catch (err) {
                    setMeshError(err instanceof Error ? err.message.slice(0, 140) : "mesh ingest failed");
                  } finally {
                    setMeshBusy(false);
                    e.target.value = "";
                  }
                }} />
            </label>
            {meshError && <div className="font-mono text-[11px] text-[#FF2D55]">{meshError}</div>}
            {meshStats && (
              <div className="font-mono text-[10px] text-[#34D399] space-y-0.5">
                <div>Grid {meshStats.grid_shape.join("×")} · Void {(meshStats.void_fraction * 100).toFixed(1)}%</div>
                <div>Mean reflectivity {meshStats.mean_reflectivity} · Dataset switched to Uploaded Mesh</div>
              </div>
            )}
          </div>
        </div>
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
            <Layers className="w-4 h-4 text-[#FFB800]" /><span>SAR Slice Ingest</span></div>
          <div className="bg-[#090A0F] p-4 rounded-lg border border-slate-800/80 text-xs space-y-2">
            <label className="block px-3 py-2 rounded-md border border-slate-700 text-center cursor-pointer hover:border-[#00F0FF] transition">
              {sliceBusy ? "Decoding…" : "Upload tomography slice(s)"}
              <input type="file" accept="image/*" multiple className="hidden"
                onChange={async (e) => {
                  const files = e.target.files;
                  if (!files || files.length === 0) return;
                  setSliceBusy(true); setSliceError(null);
                  try {
                    const fd = new FormData();
                    for (const f of Array.from(files)) fd.append("files", f);
                    const res = await fetch(`${apiBaseUrl}/api/khafre/slices`, { method: "POST", headers: apiHeaders(), body: fd });
                    if (!res.ok) throw new Error(await res.text());
                    setSlices((await res.json()).slices);
                  } catch (err) {
                    setSliceError(err instanceof Error ? err.message.slice(0, 160) : "decode failed");
                  } finally {
                    setSliceBusy(false);
                    e.target.value = "";
                  }
                }} />
            </label>
            {sliceError && <div className="font-mono text-[11px] text-[#FF2D55]">{sliceError}</div>}
            {slices.map((s) => (
              <div key={s.filename} className="pt-1">
                <div className="flex justify-between font-mono text-[11px] text-slate-400 mb-1">
                  <span className="truncate text-slate-300">{s.filename}</span>
                  <span>{s.peak_count} stripes · masked {Math.round(s.masked_fraction * 100)}%</span>
                </div>
                <img src={s.anomaly_b64} alt={`decoded ${s.filename}`} className="rounded border border-slate-700 w-full" />
                {s.peak_positions_norm.length > 0 && (
                  <div className="font-mono text-[10px] text-slate-500 mt-1">peaks @ {s.peak_positions_norm.map((p) => p.toFixed(2)).join(", ")}</div>
                )}
              </div>
            ))}
            <p className="text-[11px] text-slate-500 leading-relaxed">Jet heatmaps invert through the transfer function; legends/frames are masked, never interpolated.</p>
          </div>
        </div>
        {pinned && (
          <div className="p-3.5 rounded-lg bg-[#090A0F] border border-[#34D399]/50 text-xs">
            <div className="flex items-center justify-between mb-2">
              <span className="font-semibold text-[#34D399]">Pinned Inspection (click viewport to re-pin)</span>
              <button onClick={() => { setPinned(null); setCompare({ status: "idle", thumbs: [] }); }}
                className="text-slate-500 hover:text-white font-mono">✕</button>
            </div>
            <div className="font-mono text-[11px] text-slate-300 space-y-1 mb-3">
              <div>PIXEL ({pinned.u}, {pinned.v}) · Z {pinned.planarZ === 0 ? "0.0m EMPTY" : `${pinned.planarZ}m`}</div>
              <div>DIELECTRIC {pinned.reflectivity} · SPLIT {Math.round(pinned.disagreement * 100)}%</div>
            </div>
            {compare.status === "idle" && (
              <button onClick={runHypothesisCompare}
                className="w-full px-3 py-2 rounded-md bg-[#34D399]/15 border border-[#34D399]/50 text-[#34D399] text-xs font-medium hover:bg-[#34D399]/25 transition">
                Compare all 4 hypotheses here ({mode === "mock" ? "simulated" : "live"})</button>
            )}
            {compare.status === "loading" && <div className="text-slate-400 font-mono text-[11px]">Evaluating 4 hypotheses…</div>}
            {compare.status === "error" && <div className="text-[#FF2D55] font-mono text-[11px]">Backend unreachable — start the API first.</div>}
            {compare.status === "done" && (
              <div className="grid grid-cols-2 gap-2">
                {compare.thumbs.map(({ h, uri }) => (
                  <figure key={h}>
                    <img src={uri} alt={h} className="rounded border border-slate-700 w-full" />
                    <figcaption className="text-center text-[10px] font-mono text-slate-400 mt-1">{h}</figcaption>
                  </figure>
                ))}
              </div>
            )}
          </div>
        )}
        <div className="mt-auto p-3.5 rounded-lg bg-[#34D399]/10 border border-[#34D399]/40 text-xs">
          <div className="flex items-center gap-2 font-semibold text-[#34D399] mb-1">
            <Radar className="w-4 h-4" /><span>Next-Best-View Target Identified</span></div>
          <p className="text-slate-300 leading-relaxed text-[11px]">
            Ensemble disagreement peaks at{" "}
            <span className="font-mono text-white">
              {lastNBV ? `Z = ${lastNBV.target_z_m}m (live ensemble)` : "Z = -63.5m (Spiral Ledge #2, offline placeholder)"}
            </span>.
            A single physical borescope frame here resolves{" "}
            <span className="text-white font-semibold">
              {lastNBV ? `${lastNBV.predicted_variance_reduction_pct}%` : "68% (offline placeholder)"}
            </span>{" "}
            of structural variance.
          </p>
        </div>
      </div>
    </div>
  );
};

export default EpistemicLensViewport;
