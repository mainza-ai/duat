// Milimo Duat visual + console gate. Drives headless Chrome over CDP against
// a RUNNING build (dev or prod), screenshots both viewports + post-NBV state,
// and fails non-zero on unexpected console/page errors.
//
// Usage:
//   SHOOT_BASE=http://localhost:5174 node scripts/visual-verify.mjs   # dev
//   SHOOT_BASE=http://localhost:5175 node scripts/visual-verify.mjs   # prod
// Requires: headless Chrome with --remote-debugging-port=9222, backend up.
// Known-benign patterns (React dev StrictMode remount race) are allowlisted;
// everything else fails the gate.
const BASE = process.env.SHOOT_BASE || "http://localhost:5174";
const CDP = "http://127.0.0.1:9222/json/list";
const out = (n) => `/Users/mck/Desktop/duat/shots/${n}.png`;

const BENIGN = [
  "Attempted to synchronously unmount a root", // React 19 dev StrictMode double-mount; absent from prod builds
];

let targets = await (await fetch(CDP)).json();
// Close stale app tabs from earlier runs so we drive exactly one fresh page.
for (const t of targets.filter((t) => t.url.startsWith(BASE))) {
  await fetch(`http://127.0.0.1:9222/json/close/${t.id}`).catch(() => {});
}
await new Promise((r) => setTimeout(r, 1000));
await fetch("http://127.0.0.1:9222/json/new?" + BASE + "/", { method: "PUT" });
await new Promise((r) => setTimeout(r, 2000));
targets = await (await fetch(CDP)).json();
const page = targets.find((t) => t.url.startsWith(BASE));
if (!page) throw new Error("could not open app tab");
const ws = new WebSocket(page.webSocketDebuggerUrl, { maxPayload: 256 * 1024 * 1024 });
let id = 0;
const pending = new Map();
const errors = [];
ws.onopen = async () => {
  const send = (method, params = {}) => new Promise((res) => {
    const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params }));
  });
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data.toString());
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); return; }
    if (m.method === "Runtime.consoleAPICalled" && ["error", "warning"].includes(m.params.type))
      errors.push(`[${m.params.type}] ` + (m.params.args || []).map((a) => a.value ?? a.description ?? "").join(" ").slice(0, 300));
    if (m.method === "Runtime.exceptionThrown")
      errors.push("[exception] " + (m.params.exceptionDetails.text || "") + " " + (m.params.exceptionDetails.exception?.description || "").slice(0, 300));
    if (m.method === "Log.entryAdded" && (m.params.entry.level === "error" || m.params.entry.level === "verbose"))
      errors.push("[log." + m.params.entry.level + "] " + m.params.entry.text.slice(0, 300));
  };
  await send("Runtime.enable"); await send("Page.enable"); await send("Log.enable");
  const shot = async (name) => {
    const r = await send("Page.captureScreenshot", { format: "png" });
    await (await import("fs")).promises.writeFile(out(name), Buffer.from(r.data, "base64"));
    console.log("saved", out(name));
  };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const textPresent = async (needle) => {
    const r = await send("Runtime.evaluate", { expression: `document.body.innerText.includes(${JSON.stringify(needle)})` });
    return r.result.value === true;
  };
  const failures = [];
  const check = (name, ok) => { console.log((ok ? "PASS " : "FAIL ") + name); if (!ok) failures.push(name); };
  await sleep(6000);
  await shot("lens");
  check("lens viewport banner", await textPresent("MODEL PREDICTION"));
  await send("Runtime.evaluate", { expression: `[...document.querySelectorAll('button')].find(b=>b.textContent.includes('Helical')).click()` });
  await sleep(8000);
  await shot("helical");
  check("helical viewport banner", await textPresent("HELICAL DESCENT"));
  check("backend trajectory wired", await textPresent("SRC:API"));
  await send("Runtime.evaluate", { expression: `[...document.querySelectorAll('button')].find(b=>b.textContent.includes('NBV')).click()` });
  await sleep(4000);
  await shot("nbv");
  check("NBV selected", await textPresent("POSE #19/32"));
  const real = errors.filter((e) => !BENIGN.some((b) => e.includes(b)));
  console.log("console errors (non-benign):", real.length ? "\n" + real.join("\n") : "(none)");
  if (real.length) failures.push(`${real.length} console error(s)`);
  ws.close();
  if (failures.length) { console.error("GATE FAILED:", failures.join("; ")); process.exit(1); }
  console.log("GATE PASSED");
  process.exit(0);
};
ws.onerror = (e) => { console.error("ws error", e.message); process.exit(1); };
