/**
 * duat/apps/web/src/App.tsx — Master Application Shell for Project Duat.
 */
import React, { useState, useEffect } from "react";
import { Layers, Compass, Server, ShieldCheck, Download } from "lucide-react";
import { EpistemicLensViewport } from "./components/EpistemicLensViewport";
import { HelicalDescentNavigator } from "./components/HelicalDescentNavigator";
import { exportComparisonCard } from "./lib/api";

export function App() {
  const [activeTab, setActiveTab] = useState<"observatory" | "helical">("observatory");
  const [apiStatus, setApiStatus] = useState<"checking" | "live" | "mock" | "offline">("checking");
  const [exporting, setExporting] = useState(false);
  const [apiBaseUrl] = useState<string>("http://127.0.0.1:8001"); // TEMP local-test: :8000 occupied

  useEffect(() => {
    const checkApi = async () => {
      try {
        const res = await fetch(`${apiBaseUrl}/api/health`);
        if (!res.ok) { setApiStatus("offline"); return; }
        // The backend reports its true Atlas mode — surface it instead of hedging.
        const health = await res.json();
        setApiStatus(health.worldlabs_mock_mode ? "mock" : "live");
      } catch { setApiStatus("offline"); }
    };
    checkApi();
    const interval = setInterval(checkApi, 10000);
    return () => clearInterval(interval);
  }, [apiBaseUrl]);

  return (
    <div className="min-h-screen bg-[#050609] text-slate-100 flex flex-col font-sans selection:bg-[#00F0FF] selection:text-black">
      <header className="sticky top-0 z-50 flex flex-wrap items-center justify-between gap-4 px-6 py-3.5 bg-[#090A0F]/95 backdrop-blur-md border-b border-slate-800">
        <div className="flex items-center gap-3">
          <img src="/milimo_logo.png" alt="Milimo Duat" className="w-8 h-8 rounded-lg shadow-lg shadow-cyan-500/20" />
          <div>
            <h1 className="text-base font-bold tracking-wider uppercase bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
              MILIMO DUAT // Subterranean Epistemic Observatory</h1>
            <p className="text-[11px] text-slate-400 font-mono">World Labs atlasChisel World-Model & Sensor Ingestion Engine</p>
          </div>
        </div>
        <div className="flex items-center gap-1 bg-[#0D0F17] p-1 rounded-lg border border-slate-800">
          <button onClick={() => setActiveTab("observatory")}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-md text-xs font-medium transition ${activeTab === "observatory" ? "bg-slate-800 text-[#00F0FF] shadow" : "text-slate-400 hover:text-white"}`}>
            <Layers className="w-3.5 h-3.5" /><span>Epistemic Lens & Hypotheses</span></button>
          <button onClick={() => setActiveTab("helical")}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-md text-xs font-medium transition ${activeTab === "helical" ? "bg-slate-800 text-[#00F0FF] shadow" : "text-slate-400 hover:text-white"}`}>
            <Compass className="w-3.5 h-3.5" /><span>3D Helical Descent & Point-Cloud</span></button>
        </div>
        <div className="flex items-center gap-3">
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-mono ${
            apiStatus === "live" ? "bg-cyan-950/30 border-cyan-500/40 text-[#00F0FF]"
            : apiStatus === "mock" ? "bg-slate-800/80 border-slate-600 text-slate-200"
            : apiStatus === "checking" ? "bg-slate-800/50 border-slate-700 text-slate-400"
            : "bg-[#FF2D55]/10 border-[#FF2D55]/40 text-[#FF2D55]"}`}>
            <Server className="w-3.5 h-3.5" />
            <span>API: {apiStatus === "live" ? "Live · Atlas"
              : apiStatus === "mock" ? "Mock · simulated"
              : apiStatus === "checking" ? "Connecting…" : "Offline"}</span>
          </div>
          <button
            disabled={exporting || apiStatus === "offline" || apiStatus === "checking"}
            title={apiStatus === "offline" ? "Backend unreachable" : "Download PNG comparison card with baked-in provenance bar"}
            onClick={async () => {
              setExporting(true);
              try {
                const name = await exportComparisonCard(apiBaseUrl, {
                  camera: { pose: { position: [-38.5, -20.0, -45.0], quaternion: [0, 0, 0, 1] } },
                  hypothesis: "megalithic", instrument_rig: "drop_flare",
                  seeds: [7, 42], voxel_size: 0.25, mock_mode: true,
                });
                alert(`Exported ${name}`);
              } catch {
                alert("Export failed — is the backend running?");
              } finally {
                setExporting(false);
              }
            }}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-800 text-xs font-medium text-slate-200 hover:bg-slate-700 transition shadow disabled:opacity-50">
            <Download className="w-3.5 h-3.5 text-[#00F0FF]" /><span>{exporting ? "Exporting…" : "Export Card"}</span></button>
        </div>
      </header>
      <main className="flex-1 p-6 max-w-[1750px] w-full mx-auto flex flex-col">
        {activeTab === "observatory" ? (
          <div className="flex-1 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <div className="text-xs text-slate-400 font-mono">FLAGSHIP DATASET: <span className="text-white">Khafre Pyramid Subterranean SAR Anomalies (8 Shafts + Basal Chambers)</span></div>
              <div className="text-xs text-[#FF6B1A] font-mono flex items-center gap-1.5"><ShieldCheck className="w-3.5 h-3.5" /><span>Zero-Vividness-Bias Provenance Layer Active</span></div>
            </div>
            <EpistemicLensViewport apiBaseUrl={apiBaseUrl} />
          </div>
        ) : (
          <div className="flex-1 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <div className="text-xs text-slate-400 font-mono">TRAJECTORY: <span className="text-white">32 Descending Pinhole Cameras along Shaft #2 Helical Wall</span></div>
              <div className="text-xs text-[#00F0FF] font-mono">Live Planar-Depth RGBD Back-Projection Active</div>
            </div>
            <HelicalDescentNavigator apiBaseUrl={apiBaseUrl} />
          </div>
        )}
      </main>
      <footer className="px-6 py-3 bg-[#090A0F] border-t border-slate-800 text-center text-xs text-slate-500 font-mono">
        Project Milimo Duat // Built for open-ended wonder, epistemic rigor, and World Labs Marble 2 / Atlas.
      </footer>
    </div>
  );
}

export default App;
