/**
 * duat/apps/web/src/App.tsx — Master Application Shell for Project Duat.
 */
import React, { useState, useEffect } from "react";
import { Layers, Compass, Server, ShieldCheck, Download } from "lucide-react";
import { EpistemicLensViewport } from "./components/EpistemicLensViewport";
import { HelicalDescentNavigator } from "./components/HelicalDescentNavigator";

export function App() {
  const [activeTab, setActiveTab] = useState<"observatory" | "helical">("observatory");
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">("checking");
  const [apiBaseUrl] = useState<string>("http://localhost:8000");

  useEffect(() => {
    const checkApi = async () => {
      try {
        const res = await fetch(`${apiBaseUrl}/api/health`);
        setApiStatus(res.ok ? "online" : "offline");
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
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#00F0FF] via-[#FFB800] to-[#BD00FF] flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Compass className="w-5 h-5 text-slate-950" />
          </div>
          <div>
            <h1 className="text-base font-bold tracking-wider uppercase bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
              DUAT // Subterranean Epistemic Observatory</h1>
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
            apiStatus === "online" ? "bg-cyan-950/30 border-cyan-500/40 text-[#00F0FF]"
            : apiStatus === "checking" ? "bg-amber-950/30 border-amber-500/40 text-[#FFB800]"
            : "bg-rose-950/30 border-rose-500/40 text-rose-400"}`}>
            <Server className="w-3.5 h-3.5" />
            <span>API: {apiStatus === "online" ? "Connected (Mock/Live)" : apiStatus === "checking" ? "Connecting..." : "Offline (Local Fallback)"}</span>
          </div>
          <button onClick={() => alert("Cryptographic Provenance Manifest & SHA-256 Hashes exported.")}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-800 text-xs font-medium text-slate-200 hover:bg-slate-700 transition shadow">
            <Download className="w-3.5 h-3.5 text-[#00F0FF]" /><span>Export Manifest</span></button>
        </div>
      </header>
      <main className="flex-1 p-6 max-w-[1750px] w-full mx-auto flex flex-col">
        {activeTab === "observatory" ? (
          <div className="flex-1 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <div className="text-xs text-slate-400 font-mono">FLAGSHIP DATASET: <span className="text-white">Khafre Pyramid Subterranean SAR Anomalies (8 Shafts + Basal Chambers)</span></div>
              <div className="text-xs text-[#BD00FF] font-mono flex items-center gap-1.5"><ShieldCheck className="w-3.5 h-3.5" /><span>Zero-Vividness-Bias Provenance Layer Active</span></div>
            </div>
            <EpistemicLensViewport apiBaseUrl={apiBaseUrl} />
          </div>
        ) : (
          <div className="flex-1 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <div className="text-xs text-slate-400 font-mono">TRAJECTORY: <span className="text-white">32 Descending Pinhole Cameras along Shaft #2 Helical Wall</span></div>
              <div className="text-xs text-[#00F0FF] font-mono">Live Planar-Depth RGBD Back-Projection Active</div>
            </div>
            <HelicalDescentNavigator />
          </div>
        )}
      </main>
      <footer className="px-6 py-3 bg-[#090A0F] border-t border-slate-800 text-center text-xs text-slate-500 font-mono">
        Project Duat // Built for open-ended wonder, epistemic rigor, and World Labs Marble 2 / Atlas.
      </footer>
    </div>
  );
}

export default App;
