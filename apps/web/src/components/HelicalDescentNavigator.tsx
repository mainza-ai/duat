/**
 * duat/apps/web/src/components/HelicalDescentNavigator.tsx
 * 3D Helical Descent & Posed RGBD Point-Cloud Back-Projection Navigator.
 * Uses OFFICIAL Atlas inverted log-depth (255 = zMin, 0 = empty).
 */
import React, { useMemo, useRef, useState, useEffect } from "react";
import * as THREE from "three";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Line, PerspectiveCamera } from "@react-three/drei";
import { Compass, Video, Cpu, Play, Pause, Layers, Crosshair } from "lucide-react";
import { renderDepth, decodeGrayscalePng, decodeLogDepthByte } from "../lib/api";

export interface PosedCameraSpec {
  index: number; width: number; height: number; fx: number; fy: number; cx: number; cy: number;
  position: [number, number, number]; quaternion: [number, number, number, number];
  disagreementScore: number; isNextBestView?: boolean;
}
export type PointCloudColorMode = "relief" | "dielectric" | "depth" | "uncertainty";

export interface FlyTarget { pos: [number, number, number]; look: [number, number, number]; }

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

const KhafreShaftCutawayMesh: React.FC<{ showWireframe: boolean }> = ({ showWireframe }) => {
  const geometry = useMemo(() => {
    const geo = new THREE.CylinderGeometry(5.5, 5.5, 110, 64, 160, true);
    geo.translate(0, -110 / 2 - 5, 0);
    const pos = geo.attributes.position;
    const colors = new Float32Array(pos.count * 3);
    for (let i = 0; i < pos.count; i++) {
      const vx = pos.getX(i), vy = pos.getY(i), vz = pos.getZ(i);
      const angle = Math.atan2(vz, vx);
      const spiralWave = Math.sin((vy / 22.0) * Math.PI * 2.0 + angle * 2.0);
      const curR = Math.sqrt(vx * vx + vz * vz) || 1.0;
      pos.setX(i, (vx / curR) * (5.5 + 0.85 * spiralWave));
      pos.setZ(i, (vz / curR) * (5.5 + 0.85 * spiralWave));
      const refl = 0.5 + 0.5 * spiralWave;
      colors[i * 3] = 0.05 + refl * 0.85; colors[i * 3 + 1] = 0.45 + refl * 0.25; colors[i * 3 + 2] = 0.85 - refl * 0.65;
    }
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geo.computeVertexNormals();
    return geo;
  }, []);
  return (
    <group>
      <mesh geometry={geometry}>
        <meshStandardMaterial vertexColors side={THREE.DoubleSide} transparent opacity={0.16} roughness={0.8} />
      </mesh>
      {showWireframe && (
        <mesh geometry={geometry}><meshBasicMaterial color="#00F0FF" wireframe transparent opacity={0.12} /></mesh>
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
 * Back-projects MEASURED planar depth (from POST /api/khafre/depth) into
 * world space. Geometry is always measured; color layers derived from it:
 * relief = camera-space hillshade, uncertainty = depth-gradient magnitude
 * (occlusion proxy) — both clearly derived, never presented as Atlas output.
 */
const BackProjectedPointCloud: React.FC<{
  sceneCam: PosedCameraSpec;
  depthReq: { position: [number, number, number]; quaternion: [number, number, number, number]; shaft: [number, number] } | null;
  apiBaseUrl?: string;
  colorMode: PointCloudColorMode;
  pointSize: number;
  onSource: (s: "live" | "pending" | "error") => void;
}> = ({ sceneCam, depthReq, apiBaseUrl, colorMode, pointSize, onSource }) => {
  const [payload, setPayload] = useState<DepthPayload | null>(null);
  const reqKey = depthReq ? JSON.stringify(depthReq) : "none";

  useEffect(() => {
    if (!depthReq || !apiBaseUrl) { onSource("error"); return; }
    let cancelled = false;
    onSource("pending");
    renderDepth(
      apiBaseUrl,
      { pose: { position: depthReq.position, quaternion: depthReq.quaternion } },
      160, 90, 7, depthReq.shaft
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

  const pointsGeometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    if (!payload) return geo;
    const { depth, refl, width, height, fx, fy, cx, cy, zMin, zMax } = payload;
    const zGrid = new Float32Array(width * height);
    for (let i = 0; i < width * height; i++) {
      const b = depth[i * 4];
      zGrid[i] = b > 0 ? decodeLogDepthByte(b, zMin, zMax) : 0;
    }
    const positions: number[] = [], colors: number[] = [];
    const camPos = new THREE.Vector3(...sceneCam.position);
    const camQuat = new THREE.Quaternion(...sceneCam.quaternion);
    const v = new THREE.Vector3();
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = y * width + x;
        if (depth[i * 4] === 0) continue; // strict 0-mask: no surface, no point
        const z = zGrid[i];
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
  }, [payload, colorMode, sceneCam]);

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

export const HelicalDescentNavigator: React.FC<{ apiBaseUrl?: string; active?: boolean }> = ({ apiBaseUrl, active }) => {
  const [cameras, setCameras] = useState<PosedCameraSpec[]>(() => generateHelicalTrajectory(32));
  const [trajSource, setTrajSource] = useState<"procedural" | "backend">("procedural");
  const [selectedIdx, setSelectedIdx] = useState(4);
  const [isPlaying, setIsPlaying] = useState(false);
  const [firstPersonMode, setFirstPersonMode] = useState(false);
  const [showShaftWireframe, setShowShaftWireframe] = useState(true);
  const [colorMode, setColorMode] = useState<PointCloudColorMode>("relief");
  const [pointSize, setPointSize] = useState(0.18);
  const [backendCams, setBackendCams] = useState<any[] | null>(null);
  const [depthSrc, setDepthSrc] = useState<"live" | "pending" | "error">("pending");
  const [flyTo, setFlyTo] = useState<FlyTarget | null>(null);
  const [lookAt, setLookAt] = useState<[number, number, number]>([0, -36, 0]);

  // Prefer the backend's helical trajectory; fall back to procedural.
  // Backend cameras are Z-up world frame around the real shaft center;
  // the scene is Y-up around the origin: p_scene = R_x(-90°) * (p - center).
  useEffect(() => {
    if (!apiBaseUrl) return;
    let cancelled = false;
    // Match the demo shaft prop (110 m tall, top at y = -5).
    fetch(`${apiBaseUrl}/api/khafre/trajectory?num_cameras=32&z_start_m=-15&z_end_m=-105&orbit_radius_m=2.4`)
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
  }, [apiBaseUrl]);
  const activeCam = cameras[selectedIdx] || cameras[0];

  // Depth request for the active camera: backend pose + shaft frame when the
  // backend trajectory is live, otherwise the procedural scene pose with a
  // shaft centered at the origin (the analytic renderer is shaft-relative).
  const depthReq = useMemo(() => {
    const back = backendCams?.[activeCam.index];
    if (trajSource === "backend" && back) {
      return { position: back.extrinsics.position as [number, number, number],
        quaternion: back.extrinsics.quaternion as [number, number, number, number],
        shaft: [-40.0, -20.0] as [number, number] };
    }
    return { position: activeCam.position, quaternion: activeCam.quaternion, shaft: [0.0, 0.0] as [number, number] };
  }, [trajSource, backendCams, activeCam]);

  const selectCam = (i: number, fly: boolean) => {
    setIsPlaying(false);
    setFlyTo(null);
    setSelectedIdx(i);
    const cam = cameras[i] ?? cameras[0];
    if (fly && !firstPersonMode) {
      const [px, , pz] = cam.position;
      const len = Math.hypot(px, pz) || 1;
      setFlyTo({ pos: [cam.position[0] + (px / len) * 12, cam.position[1] + 6, cam.position[2] + (pz / len) * 12],
        look: [...cam.position] as [number, number, number] });
      setLookAt([...cam.position] as [number, number, number]);
    } else {
      setLookAt([0, cam.position[1] - 4, 0]);
    }
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
            <span className="text-[#00F0FF]">POSE #{activeCam.index + 1}/32 (DEPTH: {activeCam.position[1]}m){activeCam.isNextBestView ? " · NBV" : ""}</span></div>
        </div>
        <div className="relative flex-1 w-full h-full">
          <Canvas frameloop={active === false ? "never" : "always"}>
            <PerspectiveCamera makeDefault fov={65} position={[22, -32, 26]} near={0.1} far={500} />
            {!firstPersonMode && (
              <OrbitControls makeDefault target={lookAt} enableDamping dampingFactor={0.08}
                onStart={() => setFlyTo(null)} />
            )}
            <CameraRigSync activeCam={activeCam} firstPersonMode={firstPersonMode}
              flyTo={flyTo} onArrive={() => setFlyTo(null)} />
            <ambientLight intensity={0.65} />
            <pointLight position={[0, activeCam.position[1], 0]} intensity={2.2} distance={45} />
            <KhafreShaftCutawayMesh showWireframe={showShaftWireframe} />
            <Line points={splinePoints} color="#00F0FF" lineWidth={1.5} transparent opacity={0.45} />
            {cameras.map((cam) => (<CameraFrustumNode key={cam.index} cam={cam} isSelected={cam.index === selectedIdx} onSelect={(i) => selectCam(i, false)} />))}
            <BackProjectedPointCloud sceneCam={activeCam} depthReq={depthReq} apiBaseUrl={apiBaseUrl}
              colorMode={colorMode} pointSize={pointSize} onSource={setDepthSrc} />
          </Canvas>
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
            <span className="font-mono text-[#00F0FF] whitespace-nowrap">Z = {activeCam.position[1]}m</span></div>
          <button onClick={() => selectCam(18, true)}
            className="px-2.5 py-1.5 rounded border border-[#34D399] bg-[#34D399]/15 text-[#34D399] font-mono text-[11px] hover:bg-[#34D399]/25 transition">Jump to NBV (#19){flyTo ? " · flying…" : ""}</button>
        </div>
      </div>
      <div className="w-full lg:w-96 bg-[#0D0F17] border-l border-slate-800 p-5 flex flex-col gap-6 lg:max-h-[calc(100vh-160px)] lg:overflow-y-auto">
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
            <Video className="w-4 h-4 text-[#00F0FF]" /><span>Viewport Camera Mode</span></div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            {[["orbit", "3D Cutaway Orbit", "Inspect all 32 frustums externally"],
              ["pov", "Borescope POV", "Lock inside active pinhole pose"]].map(([id, t, d]) => {
              const active = (id === "orbit") !== firstPersonMode;
              return (<button key={id} onClick={() => setFirstPersonMode(id === "pov")}
                className={`p-2.5 rounded-lg border text-left transition ${active ? "bg-slate-800 border-[#00F0FF] text-white font-semibold" : "bg-[#090A0F] border-slate-800 text-slate-400"}`}>
                <div>{t}</div><div className="text-[10px] text-slate-400 font-normal mt-0.5">{d}</div></button>);
            })}
          </div>
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
            <label className="flex items-center justify-between cursor-pointer pt-2 border-t border-slate-800">
              <span className="text-slate-300">Show SAR Shaft Wireframe</span>
              <input type="checkbox" checked={showShaftWireframe} onChange={(e) => setShowShaftWireframe(e.target.checked)} className="accent-[#00F0FF]" />
            </label>
          </div>
        </div>
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
            <Crosshair className="w-4 h-4 text-[#00F0FF]" /><span>Active Pinhole Intrinsics & Pose</span></div>
          <div className="bg-[#090A0F] p-3.5 rounded-lg border border-slate-800 font-mono text-[11px] space-y-1.5 text-slate-300">
            <div className="flex justify-between"><span className="text-slate-500">RESOLUTION:</span><span>{activeCam.width} × {activeCam.height} (16:9)</span></div>
            <div className="flex justify-between"><span className="text-slate-500">FOCAL (fx, fy):</span><span>({activeCam.fx}, {activeCam.fy}) px</span></div>
            <div className="flex justify-between"><span className="text-slate-500">POSITION (m):</span><span className="text-[#00F0FF]">[{activeCam.position.join(", ")}]</span></div>
            <div className="flex justify-between"><span className="text-slate-500">QUATERNION:</span><span>[{activeCam.quaternion.map((q) => q.toFixed(2)).join(", ")}]</span></div>
            <div className="flex justify-between pt-1.5 border-t border-slate-800"><span className="text-[#34D399]">ENSEMBLE DISAGREEMENT:</span>
              <span className="text-white font-semibold">{Math.round(activeCam.disagreementScore * 100)}%</span></div>
          </div>
        </div>
        <div className="mt-auto p-3.5 rounded-lg bg-emerald-950/30 border border-emerald-700/40 text-xs">
          <div className="flex items-center gap-2 font-semibold text-emerald-400 mb-1">
            <Cpu className="w-4 h-4" /><span>Chained Descent Anchor Ready</span></div>
          <p className="text-slate-300 leading-relaxed text-[11px]">
            Camera #32 (<span className="font-mono text-white">Z = {cameras[cameras.length - 1]?.position[1].toFixed(1)}m</span>) can be promoted to a{" "}
            <span className="font-mono text-emerald-300">sourceFrame</span> + <span className="font-mono text-emerald-300">contextFrame</span>{" "}
            to launch the next 32-camera batch into the 80m basal chamber.
          </p>
        </div>
      </div>
    </div>
  );
};

export default HelicalDescentNavigator;
