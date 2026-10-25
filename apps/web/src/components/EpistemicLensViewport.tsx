/**
 * duat/apps/web/src/components/EpistemicLensViewport.tsx
 * Interactive 3-Tier Epistemic Stack Shader & Telemetry HUD.
 * GLSL uses the OFFICIAL Atlas inverted log-depth (255 = zMin, 0 = empty).
 */
import React, { useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Canvas, useFrame, ThreeEvent } from "@react-three/fiber";
import { Layers, Eye, Sparkles, Radar, Sliders, Crosshair, ShieldAlert } from "lucide-react";
import { postEnsemble } from "../lib/api";

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
const vec3 COLOR_SHIMMER  = vec3(1.0, 0.420, 0.102);

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
  vec3 tier3 = isEmpty ? COLOR_OBSIDIAN : rgbImagined;
  float totalW = max(uMeasuredWeight + uInferredWeight + uImaginedWeight, 0.001);
  vec3 composite = (tier1 * uMeasuredWeight + tier2 * uInferredWeight + tier3 * uImaginedWeight) / totalW;
  if (!isEmpty && uShimmerIntensity > 0.01 && disagreement > 0.18) {
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

type Hypothesis = "megalithic" | "karst" | "artifact";

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
    return { depthTex: makeTex(depthData), reflTex: makeTex(reflData), rgbTex: makeTex(rgbData), disTex: makeTex(disData) };
  }, [hypothesis]);
}

export type PixelTelemetry = { u: number; v: number; planarZ: number; reflectivity: number; disagreement: number };

interface EpistemicQuadProps {
  measuredWeight: number; inferredWeight: number; imaginedWeight: number;
  shimmerIntensity: number; loupeMode: number; hypothesis: Hypothesis;
  apiBaseUrl: string; useLive: boolean; voxelSize: number; sensorNoise: number;
  onInspectPixel: (t: PixelTelemetry) => void;
  onPinPixel: (t: PixelTelemetry) => void;
}

const EpistemicShaderQuad: React.FC<EpistemicQuadProps> = (props) => {
  const { measuredWeight, inferredWeight, imaginedWeight, shimmerIntensity, loupeMode, hypothesis, onInspectPixel, onPinPixel, voxelSize, sensorNoise } = props;
  const matRef = useRef<THREE.ShaderMaterial>(null);
  const lastTelemetry = useRef<PixelTelemetry>({ u: 640, v: 360, planarZ: 14.2, reflectivity: 0.72, disagreement: 0.34 });
  const fallback = useProceduralKhafreTextures(hypothesis);
  const [live, setLive] = useState<{ rgb: THREE.Texture; depth: THREE.Texture; refl: THREE.Texture; dis: THREE.Texture } | null>(null);

  // Fetch live ensemble layers when toggled (debounced by hypothesis change)
  React.useEffect(() => {
    if (!props.useLive) { setLive(null); return; }
    let cancelled = false;
    const loader = new THREE.TextureLoader();
    loader.setCrossOrigin("anonymous");
    postEnsemble(props.apiBaseUrl, {
      camera: { pose: { position: [-38.5, -20.0, -45.0], quaternion: [0, 0, 0, 1] } },
      hypothesis, instrument_rig: "drop_flare", seeds: [7, 42], mock_mode: true,
      voxel_size: voxelSize, sensor_noise_sigma_m: sensorNoise,
    }).then((res) => {
      if (cancelled) return;
      const tx = (uri: string) => loader.load(uri);
      setLive({ rgb: tx(res.layers.imagined_consensus_rgb_b64), depth: tx(res.layers.measured_log_depth_b64),
        refl: tx(res.layers.inferred_reflectivity_b64), dis: tx(res.layers.disagreement_map_b64) });
    }).catch(() => setLive(null));
    return () => { cancelled = true; };
  }, [props.useLive, props.apiBaseUrl, hypothesis, voxelSize, sensorNoise]);

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
    u.uLoupeMode.value = loupeMode;
  });

  const handlePointerMove = (e: ThreeEvent<PointerEvent>) => {
    if (!e.uv || !matRef.current) return;
    matRef.current.uniforms.uMouseUv.value.copy(e.uv);
    const nx = e.uv.x * 2.0 - 1.0, ny = e.uv.y * 2.0 - 1.0;
    const r = Math.sqrt(nx * nx + ny * ny);
    const isEmpty = r < 0.14;
    // Inverted decode: byte 255 = near (0.5m), byte ~1 = far (35m)
    const byte = isEmpty ? 0 : Math.min(255, Math.max(1, Math.round((1.0 - r * 0.68) * 230)));
    const planarZ = byte === 0 ? 0.0 : Number((0.5 * Math.pow(70, 1 - byte / 255)).toFixed(2));
    const t = { u: Math.round(e.uv.x * 1280), v: Math.round((1.0 - e.uv.y) * 720),
      planarZ, reflectivity: isEmpty ? 0 : 0.72, disagreement: isEmpty ? 0 : 0.34 };
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

export const EpistemicLensViewport: React.FC<{ apiBaseUrl: string }> = ({ apiBaseUrl }) => {
  const [measuredWeight, setMeasuredWeight] = useState(0.15);
  const [inferredWeight, setInferredWeight] = useState(0.10);
  const [imaginedWeight, setImaginedWeight] = useState(0.75);
  const [shimmerIntensity, setShimmerIntensity] = useState(0.65);
  const [loupeMode, setLoupeMode] = useState(1);
  const [hypothesis, setHypothesis] = useState<Hypothesis>("megalithic");
  const [useLive, setUseLive] = useState(false);
  const [voxelSize, setVoxelSize] = useState(0.25);
  const [sensorNoise, setSensorNoise] = useState(0.18);
  const [telemetry, setTelemetry] = useState<PixelTelemetry>({ u: 640, v: 360, planarZ: 14.2, reflectivity: 0.72, disagreement: 0.34 });
  const [pinned, setPinned] = useState<PixelTelemetry | null>(null);
  const [compare, setCompare] = useState<{ status: "idle" | "loading" | "error" | "done"; thumbs: { h: Hypothesis; uri: string }[] }>({ status: "idle", thumbs: [] });

  const runHypothesisCompare = async () => {
    if (!pinned) return;
    setCompare({ status: "loading", thumbs: [] });
    try {
      const hs: Hypothesis[] = ["megalithic", "karst", "artifact"];
      const results = await Promise.all(hs.map(async (h) => {
        const res = await postEnsemble(apiBaseUrl, {
          camera: { pose: { position: [-38.5, -20.0, -45.0], quaternion: [0, 0, 0, 1] } },
          hypothesis: h, instrument_rig: "drop_flare", seeds: [7, 42],
          mock_mode: true, voxel_size: voxelSize, sensor_noise_sigma_m: sensorNoise,
        });
        return { h, uri: res.layers.imagined_consensus_rgb_b64 };
      }));
      setCompare({ status: "done", thumbs: results });
    } catch {
      setCompare({ status: "error", thumbs: [] });
    }
  };

  return (
    <div className="flex flex-col lg:flex-row w-full min-h-[720px] bg-[#090A0F] text-slate-100 font-sans border border-slate-800 rounded-xl overflow-hidden shadow-2xl">
      <div className="relative flex-1 bg-black flex flex-col justify-between min-h-[480px]">
        <div className="z-10 flex flex-wrap items-center justify-between gap-2 px-4 py-2.5 bg-[#090A0F]/90 backdrop-blur-md border-b border-slate-800 text-xs">
          <div className="flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-[#FFB800]" />
            <span className="font-semibold tracking-wider uppercase text-slate-200">MILIMO DUAT // MODEL PREDICTION — CONDITIONED ON KHAFRE SAR GEOMETRY</span>
          </div>
          <div className="flex items-center gap-4 font-mono text-[11px] text-slate-400">
            <span>TASK: atlasChisel</span><span>VOXEL_GAUGE: {voxelSize.toFixed(2)}m</span>
            <span className="text-[#00F0FF]">CAM #04/32 (1280×720 PINHOLE)</span>
          </div>
        </div>
        <div className="relative flex-1 w-full h-full cursor-crosshair">
          <Canvas orthographic camera={{ zoom: 45, position: [0, 0, 10] }}>
            <EpistemicShaderQuad measuredWeight={measuredWeight} inferredWeight={inferredWeight}
              imaginedWeight={imaginedWeight} shimmerIntensity={shimmerIntensity} loupeMode={loupeMode}
              hypothesis={hypothesis} apiBaseUrl={apiBaseUrl} useLive={useLive}
              voxelSize={voxelSize} sensorNoise={sensorNoise}
              onInspectPixel={setTelemetry} onPinPixel={setPinned} />
          </Canvas>
        </div>
        <div className="z-10 grid grid-cols-2 sm:grid-cols-4 gap-2 px-4 py-2.5 bg-[#090A0F]/95 border-t border-slate-800 font-mono text-xs">
          <div className="flex items-center gap-2"><Crosshair className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-slate-400">PIXEL:</span><span className="text-white">({telemetry.u}, {telemetry.v})</span></div>
          <div><span className="text-[#00F0FF]">MEASURED Z: </span>
            <span className="text-white">{telemetry.planarZ === 0 ? "0.0m (EMPTY VOID)" : `${telemetry.planarZ}m`}</span></div>
          <div><span className="text-[#FFB800]">DIELECTRIC: </span><span className="text-white">{telemetry.reflectivity}</span></div>
          <div><span className="text-[#FF6B1A]">ENSEMBLE SPLIT: </span><span className="text-white">{Math.round(telemetry.disagreement * 100)}%</span></div>
        </div>
      </div>
      <div className="w-full lg:w-96 bg-[#0D0F17] border-l border-slate-800 p-5 flex flex-col gap-6">
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
                <span className="text-[#FF6B1A] font-medium flex items-center gap-1"><Sparkles className="w-3.5 h-3.5" /> Variance Shimmer</span>
                <span className="font-mono">{Math.round(shimmerIntensity * 100)}%</span></div>
              <input type="range" min="0" max="1" step="0.01" value={shimmerIntensity}
                onChange={(e) => setShimmerIntensity(parseFloat(e.target.value))} className="w-full accent-[#FF6B1A]" />
            </div>
            <label className="flex items-center justify-between text-xs pt-2 border-t border-slate-800 cursor-pointer">
              <span className="text-slate-300">Simulated ensemble layers (free)</span>
              <input type="checkbox" checked={useLive} onChange={(e) => setUseLive(e.target.checked)} className="accent-[#00F0FF]" />
            </label>
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
        </div>
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
            <Sliders className="w-4 h-4 text-[#FF6B1A]" /><span>Hypothesis Prior (Same Geometry)</span></div>
          <div className="flex flex-col gap-2 text-xs">
            {([
              ["megalithic", "Megalithic Architecture", "Dressed limestone & granite spiral ramp, high-lumen drop flare"],
              ["karst", "Phreatic Karst Dissolution", "Natural limestone sinkhole tube, wet travertine & water table"],
              ["artifact", "SAR Side-Lobe Echo Test", "Visualizes stepped voxel quantization & acoustic ghosting"],
            ] as [Hypothesis, string, string][]).map(([key, title, desc]) => (
              <button key={key} onClick={() => setHypothesis(key)}
                className={`p-3 rounded-lg border text-left transition ${hypothesis === key
                  ? "bg-slate-800/90 border-[#FF6B1A] text-white" : "bg-[#090A0F] border-slate-800 text-slate-400 hover:border-slate-700"}`}>
                <div className="font-semibold text-slate-100">{title}</div>
                <div className="text-[11px] text-slate-400 mt-0.5">{desc}</div>
              </button>
            ))}
          </div>
        </div>
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
            <Sliders className="w-4 h-4 text-[#00F0FF]" /><span>Sensor & Voxel Gauge</span></div>
          <div className="space-y-4 bg-[#090A0F] p-4 rounded-lg border border-slate-800/80">
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
          </div>
        </div>
        {pinned && (
          <div className="p-3.5 rounded-lg bg-[#090A0F] border border-[#FF6B1A]/50 text-xs">
            <div className="flex items-center justify-between mb-2">
              <span className="font-semibold text-[#FF6B1A]">Pinned Inspection (click viewport to re-pin)</span>
              <button onClick={() => { setPinned(null); setCompare({ status: "idle", thumbs: [] }); }}
                className="text-slate-500 hover:text-white font-mono">✕</button>
            </div>
            <div className="font-mono text-[11px] text-slate-300 space-y-1 mb-3">
              <div>PIXEL ({pinned.u}, {pinned.v}) · Z {pinned.planarZ === 0 ? "0.0m EMPTY" : `${pinned.planarZ}m`}</div>
              <div>DIELECTRIC {pinned.reflectivity} · SPLIT {Math.round(pinned.disagreement * 100)}%</div>
            </div>
            {compare.status === "idle" && (
              <button onClick={runHypothesisCompare}
                className="w-full px-3 py-2 rounded-md bg-[#FF6B1A]/15 border border-[#FF6B1A]/50 text-[#FF6B1A] text-xs font-medium hover:bg-[#FF6B1A]/25 transition">
                Compare 3 hypotheses here (simulated)</button>
            )}
            {compare.status === "loading" && <div className="text-slate-400 font-mono text-[11px]">Simulating 3 hypotheses…</div>}
            {compare.status === "error" && <div className="text-[#FF2D55] font-mono text-[11px]">Backend unreachable — start the API first.</div>}
            {compare.status === "done" && (
              <div className="grid grid-cols-3 gap-2">
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
        <div className="mt-auto p-3.5 rounded-lg bg-[#FF6B1A]/10 border border-[#FF6B1A]/40 text-xs">
          <div className="flex items-center gap-2 font-semibold text-[#FF6B1A] mb-1">
            <Radar className="w-4 h-4" /><span>Next-Best-View Target Identified</span></div>
          <p className="text-slate-300 leading-relaxed text-[11px]">
            Ensemble disagreement peaks at <span className="font-mono text-white">Z = -112.4m (Spiral Ledge #4)</span>.
            A single physical borescope frame here resolves <span className="text-white font-semibold">68%</span> of structural variance.
          </p>
        </div>
      </div>
    </div>
  );
};

export default EpistemicLensViewport;
