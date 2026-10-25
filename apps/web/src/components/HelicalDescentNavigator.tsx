/**
 * duat/apps/web/src/components/HelicalDescentNavigator.tsx
 * 3D Helical Descent & Posed RGBD Point-Cloud Back-Projection Navigator.
 * Uses OFFICIAL Atlas inverted log-depth (255 = zMin, 0 = empty).
 */
import React, { useMemo, useRef, useState, useEffect } from "react";
import * as THREE from "three";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Line, PerspectiveCamera, Html } from "@react-three/drei";
import { Compass, Video, Cpu, Play, Pause, Layers, Crosshair } from "lucide-react";

export interface PosedCameraSpec {
  index: number; width: number; height: number; fx: number; fy: number; cx: number; cy: number;
  position: [number, number, number]; quaternion: [number, number, number, number];
  disagreementScore: number; isNextBestView?: boolean;
}
export type PointCloudColorMode = "rgb" | "dielectric" | "disagreement" | "depth";

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

const CameraFrustumNode: React.FC<{ cam: PosedCameraSpec; isSelected: boolean; onSelect: (i: number) => void }> = ({ cam, isSelected, onSelect }) => {
  const color = cam.isNextBestView ? "#FF6B1A" : isSelected ? "#00F0FF" : "#475569";
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
      {(isSelected || cam.isNextBestView) && (
        <Html distanceFactor={28} position={[0, 0.6, 0]}>
          <div className={`px-2 py-0.5 rounded text-[10px] font-mono whitespace-nowrap border shadow-lg pointer-events-none ${
            cam.isNextBestView ? "bg-[#FF6B1A]/90 border-orange-300 text-white" : "bg-[#090A0F]/90 border-[#00F0FF] text-[#00F0FF]"}`}>
            {cam.isNextBestView ? `NBV TARGET #${cam.index + 1} (SPLIT ${Math.round(cam.disagreementScore * 100)}%)`
              : `CAM #${cam.index + 1} (Z=${cam.position[1]}m)`}
          </div>
        </Html>
      )}
    </group>
  );
};

const BackProjectedPointCloud: React.FC<{ cam: PosedCameraSpec; colorMode: PointCloudColorMode; zMin: number; zMax: number; pointSize: number }> = ({ cam, colorMode, zMin, zMax, pointSize }) => {
  const pointsGeometry = useMemo(() => {
    const cols = 160, rows = 90;
    const positions: number[] = [], colors: number[] = [];
    const camPos = new THREE.Vector3(...cam.position);
    const camQuat = new THREE.Quaternion(...cam.quaternion);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const u = ((c + 0.5) / cols) * cam.width, v = ((r + 0.5) / rows) * cam.height;
        const nx = (c / cols) * 2.0 - 1.0, ny = (r / rows) * 2.0 - 1.0;
        const radial = Math.sqrt(nx * nx + ny * ny), angle = Math.atan2(ny, nx);
        if (radial < 0.15) continue;
        const spiralRelief = Math.sin(8.0 / Math.max(radial, 0.15) + angle * 2.0 + cam.index * 0.25);
        // Inverted log-depth byte: bright = near.
        const logByte = Math.min(255, Math.max(1, Math.round((1.0 - radial * 0.65 + spiralRelief * 0.09) * 215)));
        // OFFICIAL inverted decode: z = zMin * (zMax/zMin)^(1 - p/255)
        const planarZ = zMin * Math.pow(zMax / zMin, 1.0 - logByte / 255.0);
        const xCam = ((u - cam.cx) / cam.fx) * planarZ;
        const yCam = -((v - cam.cy) / cam.fy) * planarZ;
        const worldPt = new THREE.Vector3(xCam, yCam, -planarZ).applyQuaternion(camQuat).add(camPos);
        positions.push(worldPt.x, worldPt.y, worldPt.z);
        const falloff = Math.min(1.0, Math.pow(radial, 1.2) * 1.35);
        const dielectric = 0.5 + 0.45 * spiralRelief;
        const disagreement = Math.pow(Math.max(0, Math.cos(8.0 / radial + angle * 2.0)), 3);
        if (colorMode === "rgb") colors.push((0.74 + 0.14 * spiralRelief) * falloff, (0.62 + 0.11 * spiralRelief) * falloff, (0.47 + 0.08 * spiralRelief) * falloff);
        else if (colorMode === "dielectric") colors.push(dielectric, 0.45 + 0.3 * (1.0 - Math.abs(dielectric - 0.5)), 1.0 - dielectric);
        else if (colorMode === "disagreement") colors.push(0.15 + 0.85 * disagreement, 0.05, 0.25 + 0.75 * disagreement);
        else { const contour = Math.abs(Math.sin(planarZ * Math.PI)) > 0.9 ? 1.0 : 0.35; colors.push(0.0, 0.94 * contour, 1.0 * contour); }
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    return geo;
  }, [cam, colorMode, zMin, zMax]);
  return (<points geometry={pointsGeometry}><pointsMaterial size={pointSize} vertexColors sizeAttenuation transparent opacity={0.92} /></points>);
};

const CameraRigSync: React.FC<{ activeCam: PosedCameraSpec; firstPersonMode: boolean }> = ({ activeCam, firstPersonMode }) => {
  const { camera } = useThree();
  const targetPos = useMemo(() => new THREE.Vector3(...activeCam.position), [activeCam]);
  const targetQuat = useMemo(() => new THREE.Quaternion(...activeCam.quaternion), [activeCam]);
  useFrame((_, delta) => {
    if (!firstPersonMode) return;
    camera.position.lerp(targetPos, Math.min(1.0, delta * 8.0));
    camera.quaternion.slerp(targetQuat, Math.min(1.0, delta * 8.0));
  });
  return null;
};

export const HelicalDescentNavigator: React.FC<{ apiBaseUrl?: string }> = ({ apiBaseUrl }) => {
  const [cameras, setCameras] = useState<PosedCameraSpec[]>(() => generateHelicalTrajectory(32));
  const [trajSource, setTrajSource] = useState<"procedural" | "backend">("procedural");
  const [selectedIdx, setSelectedIdx] = useState(4);
  const [isPlaying, setIsPlaying] = useState(false);
  const [firstPersonMode, setFirstPersonMode] = useState(false);
  const [showShaftWireframe, setShowShaftWireframe] = useState(true);
  const [colorMode, setColorMode] = useState<PointCloudColorMode>("rgb");
  const [pointSize, setPointSize] = useState(0.18);

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
        if (mapped.length > 0) { setCameras(mapped); setTrajSource("backend"); }
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [apiBaseUrl]);
  const activeCam = cameras[selectedIdx] || cameras[0];
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
            <span className="text-[#00F0FF]">POSE #{activeCam.index + 1}/32 (DEPTH: {activeCam.position[1]}m)</span></div>
        </div>
        <div className="relative flex-1 w-full h-full">
          <Canvas>
            <PerspectiveCamera makeDefault fov={65} position={[22, -32, 26]} near={0.1} far={500} />
            {!firstPersonMode && <OrbitControls target={[0, activeCam.position[1] - 4, 0]} enableDamping dampingFactor={0.08} />}
            <CameraRigSync activeCam={activeCam} firstPersonMode={firstPersonMode} />
            <ambientLight intensity={0.65} />
            <pointLight position={[0, activeCam.position[1], 0]} intensity={2.2} distance={45} />
            <KhafreShaftCutawayMesh showWireframe={showShaftWireframe} />
            <Line points={splinePoints} color="#00F0FF" lineWidth={1.5} transparent opacity={0.45} />
            {cameras.map((cam) => (<CameraFrustumNode key={cam.index} cam={cam} isSelected={cam.index === selectedIdx} onSelect={setSelectedIdx} />))}
            <BackProjectedPointCloud cam={activeCam} colorMode={colorMode} zMin={0.5} zMax={28.0} pointSize={pointSize} />
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
              onChange={(e) => { setIsPlaying(false); setSelectedIdx(parseInt(e.target.value, 10)); }} className="w-full accent-[#00F0FF]" />
            <span className="font-mono text-[#00F0FF] whitespace-nowrap">Z = {activeCam.position[1]}m</span></div>
          <button onClick={() => { setIsPlaying(false); setSelectedIdx(18); }}
            className="px-2.5 py-1.5 rounded border border-[#FF6B1A] bg-[#FF6B1A]/15 text-[#FF6B1A] font-mono text-[11px] hover:bg-[#FF6B1A]/25 transition">Jump to NBV (#19)</button>
        </div>
      </div>
      <div className="w-full lg:w-96 bg-[#0D0F17] border-l border-slate-800 p-5 flex flex-col gap-6">
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
            {[["rgb", "Tier 3: Atlas RGB"], ["dielectric", "Tier 2: Dielectric"],
              ["depth", "Tier 1: Iso-Depth"], ["disagreement", "Ensemble Split"]].map(([id, label]) => (
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
            <div className="flex justify-between pt-1.5 border-t border-slate-800"><span className="text-[#FF6B1A]">ENSEMBLE DISAGREEMENT:</span>
              <span className="text-white font-semibold">{Math.round(activeCam.disagreementScore * 100)}%</span></div>
          </div>
        </div>
        <div className="mt-auto p-3.5 rounded-lg bg-emerald-950/30 border border-emerald-700/40 text-xs">
          <div className="flex items-center gap-2 font-semibold text-emerald-400 mb-1">
            <Cpu className="w-4 h-4" /><span>Chained Descent Anchor Ready</span></div>
          <p className="text-slate-300 leading-relaxed text-[11px]">
            Camera #32 (<span className="font-mono text-white">Z = -100.0m</span>) can be promoted to a{" "}
            <span className="font-mono text-emerald-300">sourceFrame</span> + <span className="font-mono text-emerald-300">contextFrame</span>{" "}
            to launch the next 32-camera batch into the 80m basal chamber.
          </p>
        </div>
      </div>
    </div>
  );
};

export default HelicalDescentNavigator;
