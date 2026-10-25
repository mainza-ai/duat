/**
 * duat/apps/web/src/components/HelicalDescentNavigator.tsx
 * 3D Helical Descent & Posed RGBD Point-Cloud Back-Projection Navigator.
 * Uses OFFICIAL Atlas inverted log-depth (255 = zMin, 0 = empty).
 */
import React, { useMemo, useRef, useState, useEffect } from "react";
import * as THREE from "three";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Line, PerspectiveCamera, useGLTF } from "@react-three/drei";
import { Compass, Video, Cpu, Play, Pause, Layers, Crosshair } from "lucide-react";
import { renderDepth, decodeGrayscalePng, decodeLogDepthByte, createJob, getJob, cancelJob, fetchPricing, estimateCost, apiHeaders } from "../lib/api";
import type { JobRecord, PricingInfo } from "../lib/api";

export interface PosedCameraSpec {
  index: number; width: number; height: number; fx: number; fy: number; cx: number; cy: number;
  position: [number, number, number]; quaternion: [number, number, number, number];
  disagreementScore: number; isNextBestView?: boolean;
}
export type PointCloudColorMode = "relief" | "dielectric" | "depth" | "uncertainty";

export interface FlyTarget { pos: [number, number, number]; look: [number, number, number]; }

export interface ClaimAnchor { label: string; feature: string; position: [number, number, number]; note: string }
export interface ClaimRecord {
  id: string; title: string; status: string; venue: string; date: string;
  claimants: string[]; expert_response: string;
  sources: { outlet: string; url: string; date: string }[];
  anchors?: ClaimAnchor[];
}

/** Numbered claim pins (canvas sprites — no DOM portals, no font downloads).
 *  Backend Z-up anchors mapped into the shaft-#2-centered scene frame. */
const ClaimPins: React.FC<{
  pins: { label: string; pos: [number, number, number] }[];
  selected: string | null;
  onSelect: (label: string) => void;
}> = ({ pins, selected, onSelect }) => {
  const textures = useMemo(() => {
    const map = new Map<string, THREE.CanvasTexture>();
    const mk = (label: string, hot: boolean) => {
      const cv = document.createElement("canvas");
      cv.width = cv.height = 128;
      const g = cv.getContext("2d")!;
      g.beginPath(); g.arc(64, 64, 56, 0, Math.PI * 2);
      g.fillStyle = "rgba(9,10,15,0.92)"; g.fill();
      g.lineWidth = 7; g.strokeStyle = hot ? "#34D399" : "#FFB800"; g.stroke();
      g.fillStyle = "#ffffff"; g.font = "bold 52px monospace";
      g.textAlign = "center"; g.textBaseline = "middle";
      g.fillText(label.length > 3 ? label.slice(0, 3) : label, 64, 68);
      const tx = new THREE.CanvasTexture(cv);
      map.set(label + (hot ? ":h" : ""), tx);
    };
    for (const p of pins) { mk(p.label, false); mk(p.label, true); }
    return map;
  }, [pins]);
  return (
    <group>
      {pins.map((p) => (
        <sprite key={p.label} position={p.pos} scale={[7, 7, 1]}
          onClick={(e) => { e.stopPropagation(); onSelect(p.label); }}>
          <spriteMaterial map={textures.get(p.label + (selected === p.label ? ":h" : "")) ?? undefined}
            depthTest={false} transparent opacity={0.95} />
        </sprite>
      ))}
    </group>
  );
};

function generateHelicalTrajectory(numCameras = 32): PosedCameraSpec[] {
  const width = 1280, height = 720, fovDeg = 70.0;
  const fx = Number(((width / 2.0) / Math.tan((fovDeg * Math.PI) / 360.0)).toFixed(2));
  const cameras: PosedCameraSpec[] = [];
  const orbitRadius = 2.4;
  for (let i = 0; i < numCameras; i++) {
    const t = i / Math.max(numCameras - 1, 1);
    const depthY = -10.0 - t * 90.0;
    const angle = t * Math.PI * 4.0;
    const x = orbitRadius * Math.cos(angle), z = orbitRadius * Math.sin(angle);
    const dummy = new THREE.Object3D();
    dummy.position.set(x, depthY, z);
    dummy.lookAt(-Math.cos(angle + 0.4) * 5.5, depthY - 4.5, -Math.sin(angle + 0.4) * 5.5);
    const splitScore = Number((0.22 + 0.68 * Math.exp(-Math.pow((i - 18) / 3.5, 2))).toFixed(2));
    cameras.push({ index: i, width, height, fx, fy: fx, cx: width / 2, cy: height / 2,
      position: [Number(x.toFixed(3)), Number(depthY.toFixed(3)), Number(z.toFixed(3))],
      quaternion: [Number(dummy.quaternion.x.toFixed(5)), Number(dummy.quaternion.y.toFixed(5)),
        Number(dummy.quaternion.z.toFixed(5)), Number(dummy.quaternion.w.toFixed(5))],
      disagreementScore: splitScore, isNextBestView: i === 18 });
  }
  return cameras;
}

const KhafreShaftCutawayMesh: React.FC<{
  showWireframe: boolean;
  spiral: "interior_ramp" | "exterior_coil";
  heightM?: number;
}> = ({ showWireframe, spiral, heightM = 110 }) => {
  const geometry = useMemo(() => {
    // Plain cylinder prop (matches the shared shaft model: radius 5.5, no wall
    // modulation — the spiral reads via the step instancing below, not paint).
    const geo = new THREE.CylinderGeometry(5.5, 5.5, heightM, 64, Math.min(320, Math.round(heightM * 1.5)), true);
    geo.translate(0, -heightM / 2 - 5, 0);
    const pos = geo.attributes.position;
    const colors = new Float32Array(pos.count * 3);
    for (let i = 0; i < pos.count; i++) {
      const vy = pos.getY(i);
      // Subtle vertical stratigraphy bands (Tier-2 flavor, no fake spiral).
      const band = 0.5 + 0.5 * Math.sin(vy * 0.08);
      colors[i * 3] = 0.10 + band * 0.35;
      colors[i * 3 + 1] = 0.42 + band * 0.18;
      colors[i * 3 + 2] = 0.72 - band * 0.25;
    }
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geo.computeVertexNormals();
    return geo;
  }, [heightM]);

  // Exterior-coil reading (claimants' diagram): helical tube wrapping the
  // shaft at r≈5.8, same 28 m pitch/handedness as the backend coil solid.
  const spanM = heightM - 22;
  const coilGeometry = useMemo(() => {
    const pts: THREE.Vector3[] = [];
    const turns = spanM / 28;
    const segs = Math.min(480, Math.round(turns * 64));
    for (let i = 0; i <= segs; i++) {
      const s = i / segs;
      const theta = -s * turns * Math.PI * 2;
      pts.push(new THREE.Vector3(
        5.8 * Math.cos(theta), -15 - s * spanM, -5.8 * Math.sin(theta)));
    }
    return new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), segs, 0.6, 12, false);
  }, [spanM]);
  const stepsRef = useRef<THREE.InstancedMesh>(null);
  const stepCount = heightM > 200 ? 180 : 48;
  useEffect(() => {
    const mesh = stepsRef.current;
    if (!mesh) return;
    const dummy = new THREE.Object3D();
    for (let i = 0; i < stepCount; i++) {
      const y = -15 - (i / (stepCount - 1)) * spanM;
      const theta = ((((y + 15) % 28) + 28) % 28 / 28) * Math.PI * 2;
      dummy.position.set(4.5 * Math.cos(theta), y, -4.5 * Math.sin(theta));
      dummy.lookAt(0, y, 0);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
  }, [stepCount, spanM]);

  return (
    <group>
      {/* Translucent Inward-Facing Cavity Boundary */}
      <mesh geometry={geometry}>
        <meshStandardMaterial
          vertexColors
          side={THREE.DoubleSide}
          transparent
          opacity={0.24}
          roughness={0.8}
          polygonOffset
          polygonOffsetFactor={1}
          polygonOffsetUnits={1}
        />
      </mesh>

      {/* Helical spiral: interior ramp steps or exterior coil tube (variants) */}
      {spiral === "interior_ramp" ? (
        <instancedMesh ref={stepsRef} args={[undefined, undefined, stepCount]}>
          <boxGeometry args={[2.2, 0.35, 2.0]} />
          <meshStandardMaterial color="#8a7a5f" roughness={0.9} />
        </instancedMesh>
      ) : (
        <mesh geometry={coilGeometry}>
          <meshStandardMaterial color="#7fb2d9" roughness={0.55} metalness={0.25} />
        </mesh>
      )}

      {/* Radar-Cyan Structural Tomography Wireframe */}
      {showWireframe && (
        <mesh geometry={geometry}>
          <meshBasicMaterial
            color="#00F0FF"
            wireframe
            transparent
            opacity={0.18}
          />
        </mesh>
      )}
    </group>
  );
};

const CameraFrustumNode: React.FC<{
  cam: PosedCameraSpec; isSelected: boolean; onSelect: (i: number) => void;
}> = ({ cam, isSelected, onSelect }) => {
  const color = cam.isNextBestView ? "#34D399" : isSelected ? "#00F0FF" : "#475569";
  const frustumPoints = useMemo(() => {
    const scale = isSelected ? 1.4 : 0.75, w = 0.8 * scale, h = 0.45 * scale, z = -1.2 * scale;
    const o = new THREE.Vector3(0, 0, 0);
    const tl = new THREE.Vector3(-w, h, z), tr = new THREE.Vector3(w, h, z);
    const br = new THREE.Vector3(w, -h, z), bl = new THREE.Vector3(-w, -h, z);
    return [o, tl, o, tr, o, br, o, bl, tl, tr, tr, br, br, bl, bl, tl];
  }, [isSelected]);
  return (
    <group position={cam.position} quaternion={cam.quaternion as unknown as THREE.Quaternion}
      onClick={(e) => { e.stopPropagation(); onSelect(cam.index); }}>
      <Line points={frustumPoints} color={color} lineWidth={isSelected || cam.isNextBestView ? 2.2 : 1.0} />
      <mesh><sphereGeometry args={[isSelected ? 0.28 : 0.16, 12, 12]} /><meshBasicMaterial color={color} /></mesh>
    </group>
  );
};

interface DepthPayload {
  depth: Uint8ClampedArray; refl: Uint8ClampedArray;
  width: number; height: number;
  fx: number; fy: number; cx: number; cy: number;
  zMin: number; zMax: number;
}

/**
 * Back-projects MEASURED planar depth into world space. Geometry is always
 * measured; color layers derived from it: relief = camera-space hillshade,
 * uncertainty = depth-gradient magnitude (occlusion proxy).
 */
export function buildPointsGeometry(
  payload: DepthPayload,
  scenePos: [number, number, number],
  sceneQuat: [number, number, number, number],
  colorMode: PointCloudColorMode,
  maxDisplayZ: number
): THREE.BufferGeometry {
  const geo = new THREE.BufferGeometry();
  const { depth, refl, width, height, fx, fy, cx, cy, zMin, zMax } = payload;
  const zGrid = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const b = depth[i * 4];
    zGrid[i] = b > 0 ? decodeLogDepthByte(b, zMin, zMax) : 0;
  }
  const positions: number[] = [], colors: number[] = [];
  const camPos = new THREE.Vector3(...scenePos);
  const camQuat = new THREE.Quaternion(...sceneQuat);
  const v = new THREE.Vector3();
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      if (depth[i * 4] === 0) continue; // strict 0-mask: no surface, no point
      const z = zGrid[i];
      if (z > maxDisplayZ) continue; // display-range cap tames grazing far spray
      v.set(((x - cx) / fx) * z, -((y - cy) / fy) * z, -z).applyQuaternion(camQuat).add(camPos);
      positions.push(v.x, v.y, v.z);
      const rByte = refl[i * 4] / 255;
      if (colorMode === "dielectric") {
        colors.push(rByte, 0.45 + 0.3 * (1.0 - Math.abs(rByte - 0.5)), 1.0 - rByte);
      } else if (colorMode === "depth") {
        const c = Math.abs(Math.sin(z * Math.PI)) > 0.9 ? 1.0 : 0.35;
        colors.push(0.0, 0.94 * c, 1.0 * c);
      } else {
        const zx1 = zGrid[y * width + Math.min(width - 1, x + 1)], zx0 = zGrid[y * width + Math.max(0, x - 1)];
        const zy1 = zGrid[Math.min(height - 1, y + 1) * width + x], zy0 = zGrid[Math.max(0, y - 1) * width + x];
        const gm = Math.hypot((zx1 - zx0) / 2, (zy1 - zy0) / 2);
        if (colorMode === "relief") {
          const sh = (0.35 + 0.65 / Math.sqrt(1 + gm * gm)) * Math.min(1.0, Math.pow(z / zMax + 0.4, 0.6));
          colors.push(0.74 * sh, 0.62 * sh, 0.47 * sh);
        } else {
          const m = Math.min(1, gm / 3);
          colors.push(0.15 + 0.85 * m, 0.05, 0.25 + 0.75 * m);
        }
      }
    }
  }
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  return geo;
}

export interface FrameDepthReq {
  position: [number, number, number]; quaternion: [number, number, number, number];
  shaft: [number, number]; style: "interior_ramp" | "exterior_coil";
}

/** One persisted world frame: fetches its measured depth once, builds points. */
const FramePoints: React.FC<{
  sceneCam: PosedCameraSpec;
  depthReq: FrameDepthReq | null;
  apiBaseUrl?: string;
  colorMode: PointCloudColorMode;
  pointSize: number;
  maxDisplayZ: number;
}> = ({ sceneCam, depthReq, apiBaseUrl, colorMode, pointSize, maxDisplayZ }) => {
  const [payload, setPayload] = useState<DepthPayload | null>(null);
  const reqKey = depthReq ? JSON.stringify(depthReq) : "none";
  useEffect(() => {
    if (!depthReq || !apiBaseUrl) return;
    let cancelled = false;
    renderDepth(
      apiBaseUrl,
      { pose: { position: depthReq.position, quaternion: depthReq.quaternion } },
      160, 90, 7, depthReq.shaft, depthReq.style
    ).then(async (res) => {
      if (cancelled) return;
      const [d, r] = await Promise.all([
        decodeGrayscalePng(res.log_depth_b64), decodeGrayscalePng(res.reflectivity_b64)]);
      if (cancelled) return;
      setPayload({ depth: d.data, refl: r.data, width: res.width, height: res.height,
        fx: res.fx, fy: res.fy, cx: res.cx, cy: res.cy, zMin: res.z_min, zMax: res.z_max });
    }).catch(() => {});
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reqKey]);
  const geometry = useMemo(
    () => (payload ? buildPointsGeometry(payload, sceneCam.position, sceneCam.quaternion, colorMode, maxDisplayZ) : new THREE.BufferGeometry()),
    [payload, sceneCam, colorMode, maxDisplayZ]);
  if (!payload) return null;
  return (<points geometry={geometry}><pointsMaterial size={pointSize} vertexColors sizeAttenuation transparent opacity={0.92} /></points>);
};
const BackProjectedPointCloud: React.FC<{
  sceneCam: PosedCameraSpec;
  depthReq: FrameDepthReq | null;
  apiBaseUrl?: string;
  colorMode: PointCloudColorMode;
  pointSize: number;
  maxDisplayZ: number;
  onSource: (s: "live" | "pending" | "error") => void;
}> = ({ sceneCam, depthReq, apiBaseUrl, colorMode, pointSize, maxDisplayZ, onSource }) => {
  const [payload, setPayload] = useState<DepthPayload | null>(null);
  const reqKey = depthReq ? JSON.stringify(depthReq) : "none";

  useEffect(() => {
    if (!depthReq || !apiBaseUrl) { onSource("error"); return; }
    let cancelled = false;
    onSource("pending");
    renderDepth(
      apiBaseUrl,
      { pose: { position: depthReq.position, quaternion: depthReq.quaternion } },
      160, 90, 7, depthReq.shaft, depthReq.style
    ).then(async (res) => {
      if (cancelled) return;
      const [d, r] = await Promise.all([
        decodeGrayscalePng(res.log_depth_b64), decodeGrayscalePng(res.reflectivity_b64)]);
      if (cancelled) return;
      setPayload({ depth: d.data, refl: r.data, width: res.width, height: res.height,
        fx: res.fx, fy: res.fy, cx: res.cx, cy: res.cy, zMin: res.z_min, zMax: res.z_max });
      onSource("live");
    }).catch(() => { if (!cancelled) onSource("error"); });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reqKey]);

  const pointsGeometry = useMemo(
    () => (payload
      ? buildPointsGeometry(payload, sceneCam.position, sceneCam.quaternion, colorMode, maxDisplayZ)
      : new THREE.BufferGeometry()),
    [payload, colorMode, maxDisplayZ, sceneCam]);

  return (<points geometry={pointsGeometry}><pointsMaterial size={pointSize} vertexColors sizeAttenuation transparent opacity={0.92} /></points>);
};

const CameraRigSync: React.FC<{
  activeCam: PosedCameraSpec; firstPersonMode: boolean;
  flyTo: FlyTarget | null; onArrive: () => void;
}> = ({ activeCam, firstPersonMode, flyTo, onArrive }) => {
  const camera = useThree((s) => s.camera);
  const controls = useThree((s) => s.controls) as unknown as { target: THREE.Vector3; update: () => void } | null;
  const targetPos = useMemo(() => new THREE.Vector3(...activeCam.position), [activeCam]);
  const targetQuat = useMemo(() => new THREE.Quaternion(...activeCam.quaternion), [activeCam]);
  const arriveRef = useRef(onArrive);
  arriveRef.current = onArrive;
  const tmpA = useMemo(() => new THREE.Vector3(), []);
  const tmpB = useMemo(() => new THREE.Vector3(), []);
  useFrame((_, delta) => {
    if (firstPersonMode) {
      camera.position.lerp(targetPos, Math.min(1.0, delta * 8.0));
      camera.quaternion.slerp(targetQuat, Math.min(1.0, delta * 8.0));
      return;
    }
    if (flyTo && controls) {
      const k = Math.min(1.0, delta * 2.5);
      camera.position.lerp(tmpA.set(...flyTo.pos), k);
      controls.target.lerp(tmpB.set(...flyTo.look), k);
      controls.update();
      if (camera.position.distanceToSquared(tmpA.set(...flyTo.pos)) < 0.04) arriveRef.current();
    }
  });
  return null;
};

/** Full claimed-complex overview (pyramid + formation + chambers + water).
 *  Backend Z-up frame mapped like the trajectory: (x, z, −y), origin-centered. */
const ComplexOverview: React.FC<{ apiBaseUrl?: string }> = ({ apiBaseUrl }) => {
  const gltf = useGLTF(`${apiBaseUrl ?? ""}/api/khafre/complex-glb`, true);
  // The exporter writes COLOR_0 but no materials: assign lit, double-sided
  // materials on load (water stays translucent). Never trust exporter defaults.
  React.useEffect(() => {
    gltf.scene.traverse((obj) => {
      const mesh = obj as unknown as { isMesh?: boolean; name?: string; material?: unknown;
        geometry?: THREE.BufferGeometry };
      if (mesh.isMesh && mesh.geometry) {
        // toNonIndexed expands every attribute (position + color); then facets.
        const flat = mesh.geometry.toNonIndexed();
        flat.computeVertexNormals();
        mesh.geometry.dispose();
        mesh.geometry = flat;
        const isWater = (mesh.name ?? "").includes("water");
        mesh.material = new THREE.MeshStandardMaterial({
          vertexColors: true,
          side: THREE.DoubleSide,
          roughness: 0.85,
          metalness: 0.05,
          transparent: isWater,
          opacity: isWater ? 0.5 : 1.0,
        });
      }
    });
  }, [gltf]);
  return (
    <group rotation={[-Math.PI / 2, 0, 0]}>
      <primitive object={gltf.scene} />
    </group>
  );
};

/** Depth ruler overlay: ticks + active-camera marker (depths in meters). */
const DepthRuler: React.FC<{ top: number; bottom: number; step: number; camY: number }> = ({ top, bottom, step, camY }) => {
  const ticks: number[] = [];
  for (let d = top; d >= bottom; d -= step) ticks.push(d);
  const pct = Math.min(100, Math.max(0, ((top - camY) / (top - bottom)) * 100));
  return (
    <div className="absolute right-3 top-16 bottom-24 w-10 pointer-events-none select-none">
      <div className="relative h-full border-r border-slate-700">
        {ticks.map((t) => {
          const p = ((top - t) / (top - bottom)) * 100;
          return (
            <div key={t} className="absolute right-0 flex items-center gap-1 font-mono text-[9px] text-slate-500" style={{ top: `${p}%` }}>
              <span>{t}m</span>
              <span className="w-2 h-px bg-slate-600" />
            </div>
          );
        })}
        <div className="absolute -right-0.5 font-mono text-[9px] text-[#00F0FF]" style={{ top: `${pct}%` }}>◀ {camY.toFixed(0)}m</div>
      </div>
    </div>
  );
};
// Posterior SDF mesh (Z-up frame → Y-up scene with shaft #2 recenter).
const PosteriorMeshOverlay: React.FC<{ apiBaseUrl?: string; seed: number; spiral: "interior_ramp" | "exterior_coil" }> = ({ apiBaseUrl, seed, spiral }) => {
  const gltf = useGLTF(`${apiBaseUrl ?? ""}/api/khafre/mesh-glb?posterior_seed=${seed}&spiral_style=${spiral}`, true);
  React.useEffect(() => {
    gltf.scene.traverse((obj) => {
      const mesh = obj as unknown as { isMesh?: boolean; material?: unknown; geometry?: THREE.BufferGeometry };
      if (mesh.isMesh && mesh.geometry) {
        const flat = mesh.geometry.toNonIndexed();
        flat.computeVertexNormals();
        mesh.geometry.dispose();
        mesh.geometry = flat;
        mesh.material = new THREE.MeshStandardMaterial({
          vertexColors: true,
          side: THREE.DoubleSide,
          roughness: 0.78,
          metalness: 0.08,
          transparent: true,
          opacity: 0.45,
        });
      }
    });
  }, [gltf]);
  return (
    <group rotation={[-Math.PI / 2, 0, 0]} position={[40, 0, -20]}>
      <primitive object={gltf.scene} />
    </group>
  );
};

export interface HelicalDescentProps {
  apiBaseUrl?: string;
  active?: boolean;
  onInspectInLens?: (pose: { position: [number, number, number]; quaternion: [number, number, number, number]; camIdx?: number }) => void;
}

export const HelicalDescentNavigator: React.FC<HelicalDescentProps> = ({ apiBaseUrl, active, onInspectInLens }) => {
  const [cameras, setCameras] = useState<PosedCameraSpec[]>(() => generateHelicalTrajectory(32));
  const [trajSource, setTrajSource] = useState<"procedural" | "backend">("procedural");
  const [selectedIdx, setSelectedIdx] = useState(4);
  const [isPlaying, setIsPlaying] = useState(false);
  const [firstPersonMode, setFirstPersonMode] = useState(false);
  const [showShaftWireframe, setShowShaftWireframe] = useState(true);
  const [shaftSpan, setShaftSpan] = useState<"upper110" | "full648">("upper110");
  const [colorMode, setColorMode] = useState<PointCloudColorMode>("relief");
  const [pointSize, setPointSize] = useState(0.18);
  const [maxDisplayZ, setMaxDisplayZ] = useState(40);
  const [backendCams, setBackendCams] = useState<any[] | null>(null);
  const [depthSrc, setDepthSrc] = useState<"live" | "pending" | "error">("pending");
  const [flyTo, setFlyTo] = useState<FlyTarget | null>(null);
  const [lookTarget, setLookTarget] = useState<[number, number, number]>([0, -64, 0]);
  const [overview, setOverview] = useState(false);
  const [exY, setExY] = useState(1);
  const [chainHyp, setChainHyp] = useState<"megalithic" | "karst" | "artifact" | "bedrock_null">("megalithic");
  const [spiral, setSpiral] = useState<"interior_ramp" | "exterior_coil">("interior_ramp");
  const [accumulate, setAccumulate] = useState(false);
  const [visited, setVisited] = useState<number[]>([]);
  const [frameHidden, setFrameHidden] = useState<Record<number, boolean>>({});
  const [claimRec, setClaimRec] = useState<ClaimRecord | null>(null);
  const [selectedAnchor, setSelectedAnchor] = useState<string | null>(null);

  React.useEffect(() => {
    if (!apiBaseUrl) return;
    fetch(`${apiBaseUrl}/api/claims/khafre-2025-press`, { headers: apiHeaders() })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("claims"))))
      .then((body: ClaimRecord) => setClaimRec(body))
      .catch(() => setClaimRec(null));
  }, [apiBaseUrl]);

  // Claim pins in scene coords (exaggeration-aware; sprite positions scale).
  // Shaft view shows only nearby pins unless full 648 m span or overview is active.
  const pinPositions = useMemo(() => {
    const arr: { label: string; pos: [number, number, number] }[] = [];
    for (const a of claimRec?.anchors ?? []) {
      const [x, y, z] = a.position;
      const pos: [number, number, number] = [x + 40, z * exY, -(y + 20)];
      if (!overview && shaftSpan === "upper110" && pos[1] < -130) continue;
      arr.push({ label: a.label, pos });
    }
    return arr;
  }, [claimRec, exY, overview, shaftSpan]);

  const flyToAnchor = (label: string) => {
    const pin = pinPositions.find((p) => p.label === label);
    if (!pin || exY !== 1) return;
    setOverview(false);
    setSelectedAnchor(label);
    const [ax, ay, az] = pin.pos;
    setFlyTo({ pos: [ax + 15, ay + 10, az + 8], look: [ax, ay, az] });
    setLookTarget([ax, ay, az]);
  };
  const [chainBatches, setChainBatches] = useState(3);
  const [chainJob, setChainJob] = useState<{ id: string; progress: number; status: string; batches?: number } | null>(null);
  const [armingChain, setArmingChain] = useState(false);
  const [pricing, setPricing] = useState<PricingInfo | null>(null);
  const [showPosteriorMesh, setShowPosteriorMesh] = useState(false);
  const [posteriorSeed, setPosteriorSeed] = useState(101);

  React.useEffect(() => {
    if (!apiBaseUrl) return;
    fetchPricing(apiBaseUrl).then(setPricing).catch(() => setPricing(null));
  }, [apiBaseUrl]);

  const runChain = async (live: boolean) => {
    if (!apiBaseUrl) return;
    setChainJob(null);
    try {
      const { id } = await createJob(apiBaseUrl, "chain", {
        camera: { pose: { position: [-38.5, -20.0, -45.0], quaternion: [0, 0, 0, 1] } },
        hypothesis: chainHyp, instrument_rig: "drop_flare", seeds: [7],
        mode: live ? "live" : "mock", chain_batches: chainBatches, chain_targets: 2,
      });
      for (let i = 0; i < 240; i++) {
        await new Promise((r) => setTimeout(r, 3000));
        const job: JobRecord = await getJob(apiBaseUrl, id);
        setChainJob({ id, progress: job.progress, status: job.status,
          batches: (job.result as { batches?: unknown[] } | undefined)?.batches?.length });
        if (job.status === "done" || job.status === "failed" || job.status === "cancelled") break;
      }
    } catch {
      setChainJob({ id: "error", progress: 0, status: "failed" });
    }
  };

  // Prefer the backend's helical trajectory; fall back to procedural.
  // Backend cameras are Z-up world frame around the real shaft center;
  // the scene is Y-up around the origin: p_scene = R_x(-90°) * (p - center).
  useEffect(() => {
    if (!apiBaseUrl) return;
    let cancelled = false;
    const zEnd = shaftSpan === "full648" ? -600 : -105;
    fetch(`${apiBaseUrl}/api/khafre/trajectory?num_cameras=32&z_start_m=-15&z_end_m=${zEnd}&orbit_radius_m=2.4`, { headers: apiHeaders() })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("trajectory"))))
      .then((body) => {
        if (cancelled || !Array.isArray(body.targetCameras)) return;
        const sc: [number, number] = body.shaft_center_xy ?? [-40, -20];
        // R_x(-90°) as XYZW: maps (x, y, z)_zup -> (x, z, -y)_yup.
        const qm: [number, number, number, number] = [-0.70710678, 0, 0, 0.70710678];
        const qmul = (a: number[], b: number[]): [number, number, number, number] => [
          a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
          a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
          a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
          a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
        ];
        const mapped: PosedCameraSpec[] = body.targetCameras.map((c: any, i: number) => {
          const p = c.extrinsics.position as [number, number, number];
          return {
            index: i,
            width: c.intrinsics.width, height: c.intrinsics.height,
            fx: c.intrinsics.fx, fy: c.intrinsics.fy, cx: c.intrinsics.cx, cy: c.intrinsics.cy,
            position: [p[0] - sc[0], p[2], -(p[1] - sc[1])] as [number, number, number],
            quaternion: qmul(qm, c.extrinsics.quaternion),
            disagreementScore: Number((0.22 + 0.68 * Math.exp(-Math.pow((i - 18) / 3.5, 2))).toFixed(2)),
            isNextBestView: i === 18,
          };
        });
        if (mapped.length > 0) { setCameras(mapped); setBackendCams(body.targetCameras); setTrajSource("backend"); }
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [apiBaseUrl, shaftSpan]);
  const activeCam = cameras[selectedIdx] || cameras[0];

  // Depth request for the active camera: backend pose + shaft frame when the
  // backend trajectory is live, otherwise the procedural scene pose with a
  // shaft centered at the origin (the analytic renderer is shaft-relative).
  const depthReqFor = (idx: number): FrameDepthReq | null => {
    const cam = cameras[idx];
    if (!cam) return null;
    const back = backendCams?.[idx];
    if (trajSource === "backend" && back) {
      return { position: back.extrinsics.position, quaternion: back.extrinsics.quaternion,
        shaft: [-40.0, -20.0] as [number, number], style: spiral };
    }
    return { position: cam.position, quaternion: cam.quaternion,
      shaft: [0.0, 0.0] as [number, number], style: spiral };
  };
  const depthReq = useMemo(() => depthReqFor(activeCam.index),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [trajSource, backendCams, activeCam, spiral]);

  // Accumulate visited frames into the persistent world (FIFO cap 12).
  useEffect(() => {
    if (!accumulate) return;
    setVisited((prev) => {
      if (prev.includes(activeCam.index)) return prev;
      const next = [...prev, activeCam.index];
      return next.length > 12 ? next.slice(next.length - 12) : next;
    });
  }, [activeCam.index, accumulate]);

  const jumpToNBV = () => {
    setIsPlaying(false);
    setOverview(false);
    setExY(1);
    setFirstPersonMode(false);
    setSelectedIdx(18);
    const cam = cameras[18] ?? cameras[0];
    const [px, , pz] = cam.position;
    const len = Math.hypot(px, pz) || 1;
    setFlyTo({ pos: [cam.position[0] + (px / len) * 12, cam.position[1] + 6, cam.position[2] + (pz / len) * 12],
      look: [...cam.position] as [number, number, number] });
    setLookTarget([...cam.position] as [number, number, number]);
  };

  const selectCam = (i: number, fly: boolean) => {
    setIsPlaying(false);
    setFlyTo(null);
    setOverview(false);
    setSelectedIdx(i);
    const cam = cameras[i] ?? cameras[0];
    if (fly && !firstPersonMode && exY === 1) {
      const [px, , pz] = cam.position;
      const len = Math.hypot(px, pz) || 1;
      setFlyTo({ pos: [cam.position[0] + (px / len) * 12, cam.position[1] + 6, cam.position[2] + (pz / len) * 12],
        look: [...cam.position] as [number, number, number] });
      setLookTarget([...cam.position] as [number, number, number]);
    } else {
      setLookTarget([0, cam.position[1] - 4, 0]);
    }
  };
  // Orbit target follows the unscaled look target through vertical exaggeration.
  const lookAt: [number, number, number] = [lookTarget[0], lookTarget[1] * exY, lookTarget[2]];

  const toggleOverview = () => {
    setExY(1);
    if (!overview) {
      setIsPlaying(false);
      setFlyTo({ pos: [420, -160, 520], look: [0, -220, 0] });
      setLookTarget([0, -220, 0]);
      setOverview(true);
    } else {
      setOverview(false);
      setFlyTo({ pos: [55, -60, 62], look: [0, activeCam.position[1], 0] });
      setLookTarget([0, activeCam.position[1] - 4, 0]);
    }
  };
  const setExaggeration = (v: number) => {
    setExY(v);
    setFlyTo(null);
    if (v !== 1) setFirstPersonMode(false); // POV/fly need true scale; orbit stays valid
  };
  useEffect(() => {
    if (!isPlaying) return;
    const timer = setInterval(() => setSelectedIdx((p) => (p + 1) % cameras.length), 450);
    return () => clearInterval(timer);
  }, [isPlaying, cameras.length]);
  const splinePoints = useMemo(() => cameras.map((c) => new THREE.Vector3(...c.position)), [cameras]);
  return (
    <div className="flex flex-col lg:flex-row w-full min-h-[720px] bg-[#090A0F] text-slate-100 font-sans border border-slate-800 rounded-xl overflow-hidden shadow-2xl">
      <div className="relative flex-1 min-h-[520px] bg-[#050609] flex flex-col justify-between">
        <div className="z-10 flex flex-wrap items-center justify-between gap-2 px-4 py-2.5 bg-[#090A0F]/90 backdrop-blur-md border-b border-slate-800 text-xs">
          <div className="flex items-center gap-2"><Compass className="w-4 h-4 text-[#00F0FF]" />
            <span className="font-semibold tracking-wider uppercase">MILIMO DUAT // 3D HELICAL DESCENT & RGBD POINT-CLOUD BACK-PROJECTION</span></div>
          <div className="flex items-center gap-3 font-mono text-[11px] text-slate-400">
            <span>SHAFT #2 (WEST ROW)</span>
            <span className="text-slate-500">SRC:{trajSource === "backend" ? "API" : "LOCAL"}</span>
            <span className="text-slate-500">PTS:{depthSrc === "live" ? "MEASURED" : depthSrc === "pending" ? "…" : "OFFLINE"}</span>
            <span className="text-[#00F0FF]">POSE #{activeCam.index + 1}/32 (DEPTH: {activeCam.position[1].toFixed(1)}m){activeCam.isNextBestView ? " · NBV" : ""}</span>
            <span className="hidden xl:flex items-center gap-2 text-slate-500">
              <span><span className="text-slate-400">●</span> all</span>
              <span><span className="text-[#00F0FF]">●</span> selected</span>
              <span><span className="text-[#34D399]">●</span> NBV</span>
            </span></div>
        </div>
        <div className="relative flex-1 w-full h-full">
          {/* flat: no tone mapping — authored tier/vertex colors render exactly (data fidelity). */}
          <Canvas flat frameloop={active === false ? "never" : "always"}>
            <PerspectiveCamera makeDefault fov={55} position={[55, -60, 62]} near={1} far={overview ? 2500 : shaftSpan === "full648" ? 1200 : 400} />
            {!firstPersonMode && (
              <OrbitControls makeDefault target={lookAt} enableDamping dampingFactor={0.08}
                onStart={() => setFlyTo(null)} />
            )}
            <CameraRigSync activeCam={activeCam} firstPersonMode={firstPersonMode}
              flyTo={flyTo} onArrive={() => setFlyTo(null)} />
            <ambientLight intensity={0.85} />
            <hemisphereLight args={["#3a4a63", "#0a0c12", 0.5]} />
            <pointLight position={[0, activeCam.position[1], 0]} intensity={2.2} distance={45} />
            {overview && (<directionalLight position={[300, 200, 150]} intensity={1.6} />)}
            {overview ? (
              <React.Suspense fallback={null}>
                <ComplexOverview apiBaseUrl={apiBaseUrl} />
              </React.Suspense>
            ) : (
              <group scale={[1, exY, 1]}>
                {/* 1. Shaft geometry: analytic prop or real posterior-mesh overlay */}
            {showPosteriorMesh ? (
              <React.Suspense fallback={null}>
                <PosteriorMeshOverlay apiBaseUrl={apiBaseUrl} seed={posteriorSeed} spiral={spiral} />
              </React.Suspense>
            ) : (
              <KhafreShaftCutawayMesh showWireframe={showShaftWireframe} spiral={spiral} heightM={shaftSpan === "full648" ? 610 : 110} />
            )}
                <Line points={splinePoints} color="#00F0FF" lineWidth={1.5} transparent opacity={0.45} />
                {cameras.map((cam) => (<CameraFrustumNode key={cam.index} cam={cam} isSelected={cam.index === selectedIdx} onSelect={(i) => selectCam(i, false)} />))}
                <BackProjectedPointCloud sceneCam={activeCam} depthReq={depthReq} apiBaseUrl={apiBaseUrl}
                  colorMode={colorMode} pointSize={pointSize} maxDisplayZ={maxDisplayZ} onSource={setDepthSrc} />
                {accumulate && visited.filter((i) => i !== activeCam.index && !frameHidden[i]).map((i) => {
                  const cam = cameras[i];
                  if (!cam) return null;
                  return (<FramePoints key={i} sceneCam={cam} depthReq={depthReqFor(i)} apiBaseUrl={apiBaseUrl}
                    colorMode={colorMode} pointSize={pointSize} maxDisplayZ={maxDisplayZ} />);
                })}
              </group>
            )}
            {pinPositions.length > 0 && !firstPersonMode && (
              <ClaimPins pins={pinPositions} selected={selectedAnchor}
                onSelect={(label) => setSelectedAnchor(label)} />
            )}
          </Canvas>
          {overview
            ? <DepthRuler top={150} bottom={-640} step={100} camY={activeCam.position[1]} />
            : shaftSpan === "full648"
              ? <DepthRuler top={0} bottom={-620} step={100} camY={activeCam.position[1]} />
              : <DepthRuler top={0} bottom={-120} step={20} camY={activeCam.position[1]} />}
        </div>
        <div className="z-10 flex flex-wrap items-center gap-4 px-4 py-3 bg-[#090A0F]/95 border-t border-slate-800 text-xs">
          <button onClick={() => setIsPlaying((p) => !p)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-[#00F0FF] text-slate-950 font-semibold hover:bg-[#00F0FF]/90 transition">
            {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            <span>{isPlaying ? "Pause Descent" : "Auto-Descend (32 Cams)"}</span></button>
          <div className="flex-1 flex items-center gap-3">
            <span className="font-mono text-slate-400 whitespace-nowrap">CAM #{selectedIdx + 1}</span>
            <input type="range" min={0} max={cameras.length - 1} value={selectedIdx}
              onChange={(e) => selectCam(parseInt(e.target.value, 10), false)} className="w-full accent-[#00F0FF]" />
            <span className="font-mono text-[#00F0FF] whitespace-nowrap">Z = {activeCam.position[1].toFixed(1)}m</span></div>
          <button onClick={jumpToNBV} title={exY !== 1 ? "Resets vertical scale to 1×, then flies" : "Fly to the NBV pose"}
            className="px-2.5 py-1.5 rounded border border-[#34D399] bg-[#34D399]/15 text-[#34D399] font-mono text-[11px] hover:bg-[#34D399]/25 transition">Jump to NBV (#19){flyTo ? " · flying…" : ""}</button>
        </div>
      </div>
      <div className="w-full lg:w-96 bg-[#0D0F17] border-l border-slate-800 p-5 flex flex-col gap-6 lg:max-h-[calc(100vh-160px)] lg:overflow-y-auto">
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
            <Video className="w-4 h-4 text-[#00F0FF]" /><span>Viewport Camera Mode</span></div>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <button onClick={() => { setOverview(false); setFirstPersonMode(false); }}
              className={`p-2.5 rounded-lg border text-left transition ${!overview && !firstPersonMode ? "bg-slate-800 border-[#00F0FF] text-white font-semibold" : "bg-[#090A0F] border-slate-800 text-slate-400"}`}>
              <div>3D Cutaway Orbit</div><div className="text-[10px] text-slate-400 font-normal mt-0.5">Inspect all 32 frustums externally</div></button>
            <button onClick={() => { setOverview(false); setExY(1); setFirstPersonMode(true); }}
              title="Resets vertical scale to 1× (POV needs true scale)"
              className={`p-2.5 rounded-lg border text-left transition ${!overview && firstPersonMode ? "bg-slate-800 border-[#00F0FF] text-white font-semibold" : "bg-[#090A0F] border-slate-800 text-slate-400"}`}>
              <div>Borescope POV</div><div className="text-[10px] text-slate-400 font-normal mt-0.5">Lock inside active pinhole pose</div></button>
            <button onClick={toggleOverview}
              className={`p-2.5 rounded-lg border text-left transition ${overview ? "bg-slate-800 border-[#34D399] text-white font-semibold" : "bg-[#090A0F] border-slate-800 text-slate-400"}`}>
              <div>Complex Overview</div><div className="text-[10px] text-slate-400 font-normal mt-0.5">Pyramid to water table</div></button>
          </div>
          {!overview && (
          <div className="space-y-2 mt-2 text-xs">
            <div className="flex items-center justify-between gap-2">
              <span className="text-slate-400">Shaft span</span>
              <div className="flex gap-1">
                {([["upper110", "Upper 110m"], ["full648", "Full 648m"]] as const).map(([sp, lbl]) => (
                  <button key={sp} onClick={() => setShaftSpan(sp)}
                    className={`px-2 py-1 rounded border font-mono text-[10px] ${shaftSpan === sp ? "bg-slate-800 border-[#00F0FF] text-white" : "bg-[#090A0F] border-slate-800 text-slate-400"}`}>{lbl}</button>
                ))}
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-slate-400">Vertical</span>
              {[0.5, 1, 2].map((v) => (
                <button key={v} onClick={() => setExaggeration(v)}
                  title={v === 1 ? "True scale" : `Vertical ×${v} (distorted geometry)`}
                  className={`px-2 py-1 rounded border font-mono text-[10px] ${exY === v ? "bg-slate-800 border-[#00F0FF] text-white" : "bg-[#090A0F] border-slate-800 text-slate-400"}`}>×{v}</button>
              ))}
              {exY !== 1 && <span className="font-mono text-[10px] text-[#FFB800]">distorted — POV/fly need 1×</span>}
            </div>
          </div>
          )}
        </div>
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
            <Layers className="w-4 h-4 text-[#FFB800]" /><span>Back-Projected Point Cloud Layer</span></div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            {[["relief", "Hillshade Relief"], ["dielectric", "Tier 2: Dielectric"],
              ["depth", "Tier 1: Iso-Depth"], ["uncertainty", "Occlusion Proxy"]].map(([id, label]) => (
              <button key={id} onClick={() => setColorMode(id as PointCloudColorMode)}
                className={`px-3 py-2 rounded-md border text-left transition ${colorMode === id
                  ? "bg-slate-800 border-[#FFB800] text-white font-medium" : "bg-[#090A0F] border-slate-800 text-slate-400 hover:border-slate-700"}`}>{label}</button>
            ))}
          </div>
          <div className="mt-4 bg-[#090A0F] p-3.5 rounded-lg border border-slate-800 space-y-3 text-xs">
            <div><div className="flex justify-between mb-1"><span className="text-slate-400">Voxel Point Size</span>
              <span className="font-mono text-[#00F0FF]">{pointSize.toFixed(2)}m</span></div>
              <input type="range" min="0.06" max="0.40" step="0.02" value={pointSize}
                onChange={(e) => setPointSize(parseFloat(e.target.value))} className="w-full accent-[#00F0FF]" /></div>
            <div><div className="flex justify-between mb-1"><span className="text-slate-400">Max display range</span>
              <span className="font-mono text-[#00F0FF]">{maxDisplayZ.toFixed(0)}m</span></div>
              <input type="range" min="8" max="40" step="1" value={maxDisplayZ}
                onChange={(e) => setMaxDisplayZ(parseFloat(e.target.value))} className="w-full accent-[#00F0FF]" /></div>
            <label className="flex items-center justify-between cursor-pointer pt-2 border-t border-slate-800">
              <span className="text-slate-300">Show SAR Shaft Wireframe</span>
              <input type="checkbox" checked={showShaftWireframe} onChange={(e) => setShowShaftWireframe(e.target.checked)} className="accent-[#00F0FF]" />
            </label>
            <label className="flex items-center justify-between cursor-pointer pt-2 border-t border-slate-800">
              <span className="text-slate-300">Show Posterior SDF Mesh (GLB)</span>
              <input type="checkbox" checked={showPosteriorMesh} onChange={(e) => setShowPosteriorMesh(e.target.checked)} className="accent-[#34D399]" />
            </label>
            <label className="flex items-center justify-between cursor-pointer pt-2 border-t border-slate-800">
              <span className="text-slate-300">Accumulate visited frames</span>
              <input type="checkbox" checked={accumulate} onChange={(e) => { setAccumulate(e.target.checked); if (!e.target.checked) { setVisited([]); setFrameHidden({}); } }} className="accent-[#34D399]" />
            </label>
            {accumulate && visited.length > 0 && (
              <div className="flex flex-wrap gap-1 pt-1">
                {visited.map((i) => (
                  <button key={i} onClick={() => setFrameHidden((p) => ({ ...p, [i]: !p[i] }))}
                    title={frameHidden[i] ? "show frame" : "hide frame"}
                    className={`px-1.5 py-0.5 rounded border font-mono text-[10px] ${i === activeCam.index ? "border-[#00F0FF] text-white" : frameHidden[i] ? "border-slate-800 text-slate-600 line-through" : "border-slate-700 text-slate-300"}`}>
                    #{i + 1}</button>
                ))}
                <button onClick={() => { setVisited([]); setFrameHidden({}); }}
                  className="px-1.5 py-0.5 rounded border border-slate-800 font-mono text-[10px] text-slate-500 hover:text-white">clear</button>
              </div>
            )}
            <div className="flex items-center justify-between pt-1">
              <span className="text-slate-400 text-xs">Spiral reading</span>
              <div className="flex gap-1">
                {(["interior_ramp", "exterior_coil"] as const).map((s) => (
                  <button key={s} onClick={() => setSpiral(s)} title={s === "interior_ramp" ? "Walkway shelf inside the shaft (our reading)" : "Coil wrapping the shaft exterior (claimants' diagram)"}
                    className={`px-2 py-0.5 rounded border font-mono text-[10px] ${spiral === s ? "bg-slate-800 border-[#34D399] text-white" : "border-slate-800 text-slate-400"}`}>
                    {s === "interior_ramp" ? "ramp" : "coil"}</button>
                ))}
              </div>
            </div>
            {showPosteriorMesh && (
              <div className="flex items-center justify-between pt-1">
                <span className="text-slate-400 text-xs">Posterior seed</span>
                <div className="flex gap-1">
                  {[101, 202, 303].map((s) => (
                    <button key={s} onClick={() => setPosteriorSeed(s)}
                      className={`px-2 py-0.5 rounded border font-mono text-[10px] ${posteriorSeed === s ? "bg-slate-800 border-[#34D399] text-white" : "border-slate-800 text-slate-400"}`}>{s}</button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
            <Crosshair className="w-4 h-4 text-[#00F0FF]" /><span>Active Pinhole Intrinsics & Pose</span></div>
          <div className="bg-[#090A0F] p-3.5 rounded-lg border border-slate-800 font-mono text-[11px] space-y-1.5 text-slate-300">
            <div className="flex justify-between"><span className="text-slate-500">RESOLUTION:</span><span>{activeCam.width} × {activeCam.height} (16:9)</span></div>
            <div className="flex justify-between"><span className="text-slate-500">FOCAL (fx, fy):</span><span>({activeCam.fx}, {activeCam.fy}) px</span></div>
            <div className="flex justify-between"><span className="text-slate-500">POSITION (m):</span><span className="text-[#00F0FF]">[{activeCam.position.map((n) => Number(n.toFixed(2))).join(", ")}]</span></div>
            <div className="flex justify-between"><span className="text-slate-500">QUATERNION:</span><span>[{activeCam.quaternion.map((q) => q.toFixed(2)).join(", ")}]</span></div>
            <div className="flex justify-between pt-1.5 border-t border-slate-800"><span className="text-[#34D399]">ENSEMBLE DISAGREEMENT:</span>
              <span className="text-white font-semibold">{Math.round(activeCam.disagreementScore * 100)}%</span></div>
            {onInspectInLens && (
              <button
                onClick={() => {
                  const back = backendCams?.[activeCam.index];
                  const pos: [number, number, number] = back?.extrinsics?.position ?? [activeCam.position[0] - 40, -20 - activeCam.position[2], activeCam.position[1]];
                  const quat: [number, number, number, number] = back?.extrinsics?.quaternion ?? [0, 0, 0, 1];
                  onInspectInLens({ position: pos, quaternion: quat, camIdx: activeCam.index });
                }}
                className="w-full mt-2 px-3 py-1.5 rounded bg-[#00F0FF]/15 border border-[#00F0FF]/50 text-[#00F0FF] font-sans font-medium text-xs hover:bg-[#00F0FF]/25 transition">
                Inspect Pose in Epistemic Lens →
              </button>
            )}
          </div>
        </div>
        <div className="mt-auto p-3.5 rounded-lg bg-emerald-950/30 border border-emerald-700/40 text-xs">
          <div className="flex items-center gap-2 font-semibold text-emerald-400 mb-1">
            <Cpu className="w-4 h-4" /><span>Chained Descent Job</span></div>
          <p className="text-slate-300 leading-relaxed text-[11px] mb-2">
            Camera #32 (<span className="font-mono text-white">Z = {cameras[cameras.length - 1]?.position[1].toFixed(1)}m</span>) anchors
            the next batch. Runs {chainBatches} batches × 2 targets as a resumable, cancellable job.
          </p>
          <div className="mb-1 text-[11px] font-medium text-slate-400">Hypothesis prior</div>
          <div className="grid grid-cols-4 gap-1.5 mb-3">
            {(["megalithic", "karst", "artifact", "bedrock_null"] as const).map((h) => (
              <button key={h} onClick={() => setChainHyp(h)}
                className={`px-2 py-1.5 rounded-md border font-mono text-[10px] transition ${chainHyp === h
                  ? "bg-slate-800 border-[#34D399] text-white font-medium" : "bg-[#090A0F] border-slate-800 text-slate-400 hover:border-slate-700"}`}>
                {h === "bedrock_null" ? "null" : h.slice(0, 4)}</button>
            ))}
          </div>
          <div className="mb-3">
            <div className="flex justify-between text-[11px] mb-1">
              <span className="text-slate-400">Batches (× 2 targets)</span>
              <span className="font-mono text-white">{chainBatches}</span></div>
            <input type="range" min={1} max={6} step={1} value={chainBatches}
              onChange={(e) => setChainBatches(parseInt(e.target.value, 10))} className="w-full accent-[#34D399]" />
          </div>
          {!armingChain ? (
            <div className="grid grid-cols-2 gap-2">
              <button onClick={() => void runChain(false)}
                className="px-3 py-2 rounded-md bg-slate-800 border border-slate-700 text-xs font-medium hover:bg-slate-700 transition">Run simulated</button>
              <button onClick={() => setArmingChain(true)}
                className="px-3 py-2 rounded-md border border-[#FFB800]/60 text-[#FFB800] text-xs font-medium hover:bg-[#FFB800]/10 transition">Live chain</button>
            </div>
          ) : (
            <div className="p-2 rounded border border-[#FFB800]/60 bg-[#FFB800]/10 text-[11px]">
              <div className="mb-1.5">Bills {chainBatches} live task(s). {estimateCost(chainBatches, pricing, 2)}.</div>
              <div className="flex gap-1.5">
                <button onClick={() => { setArmingChain(false); void runChain(true); }}
                  className="px-2 py-1 rounded bg-[#FFB800] text-slate-950 font-semibold">Confirm spend</button>
                <button onClick={() => setArmingChain(false)} className="px-2 py-1 rounded border border-slate-700">Cancel</button>
              </div>
            </div>
          )}
          {chainJob && (
            <div className="mt-2 font-mono text-[11px] text-slate-300">
              <div className="flex justify-between mb-1">
                <span>job {chainJob.id.slice(0, 12)}… {chainJob.status}{chainJob.batches ? ` · ${chainJob.batches} batches` : ""}</span>
                {(chainJob.status === "pending" || chainJob.status === "running") && (
                  <button onClick={() => apiBaseUrl && cancelJob(apiBaseUrl, chainJob.id).catch(() => {})}
                    className="text-[#FF2D55] hover:underline">Cancel</button>
                )}
              </div>
              <div className="h-1.5 rounded bg-slate-800 overflow-hidden">
                <div className="h-full bg-[#34D399] transition-all" style={{ width: `${Math.round((chainJob.progress || 0) * 100)}%` }} />
              </div>
            </div>
          )}
        </div>
        {claimRec && (
          <div className="p-3.5 rounded-lg bg-[#090A0F] border border-slate-800 text-xs">
            <div className="flex items-center justify-between gap-2 mb-1.5">
              <span className="font-semibold text-slate-200">Feature claims</span>
              <span className="font-mono text-[10px] px-1.5 py-0.5 rounded border border-[#FFB800]/60 text-[#FFB800]">{claimRec.status}</span>
            </div>
            <div className="text-[11px] text-slate-400 mb-2">{claimRec.title} — {claimRec.venue} ({claimRec.date})</div>
            <div className="flex flex-wrap gap-1.5 mb-2">
              {(claimRec.anchors ?? []).map((a) => (
                <button key={a.label} onClick={() => flyToAnchor(a.label)} title={`${a.feature}${a.note ? ` — ${a.note}` : ""}`}
                  className={`px-2 py-1 rounded-md border font-mono text-[10px] transition ${selectedAnchor === a.label ? "bg-slate-800 border-[#34D399] text-white" : "bg-[#0D0F17] border-slate-700 text-slate-300 hover:border-slate-500"}`}>
                  {a.label}</button>
              ))}
            </div>
            {selectedAnchor && (() => {
              const a = (claimRec.anchors ?? []).find((x) => x.label === selectedAnchor);
              if (!a) return null;
              return (
                <div className="pt-2 border-t border-slate-800 text-[11px] text-slate-300 space-y-1">
                  <div><span className="font-mono text-white">{a.label}</span> — {a.feature}</div>
                  {a.note && <div className="text-slate-400">{a.note}</div>}
                  <div className="text-slate-400">{claimRec.expert_response}</div>
                  <div className="flex flex-wrap gap-2 pt-0.5">
                    {claimRec.sources.slice(0, 3).map((s) => (
                      <a key={s.url} href={s.url} target="_blank" rel="noreferrer" className="text-[#00F0FF] hover:underline font-mono text-[10px]">{s.outlet} →</a>
                    ))}
                  </div>
                </div>
              );
            })()}
          </div>
        )}
      </div>
    </div>
  );
};

export default HelicalDescentNavigator;
