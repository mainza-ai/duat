const r = await (await fetch("http://127.0.0.1:9222/json/list")).json();
const page = r.find((t) => t.url.startsWith("http://localhost:5174"));
const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
ws.onopen = () => {
  const send = (m, p = {}) =>
    new Promise((res) => {
      const i = ++id;
      pending.set(m.id, res);
      ws.send(JSON.stringify({ id: i, method: m, params: p }));
    });
  ws.onmessage = async (ev) => {
    const m = JSON.parse(ev.data.toString());
    if (m.id && pending.has(m.id)) {
      pending.get(m.id)(m.result);
      pending.delete(m.id);
    }
  };
  (async () => {
    const ex =
      "(()=>{const out=[];[...document.querySelectorAll('canvas')].forEach((e,i)=>{const q=e.getBoundingClientRect();" +
      "out.push(i+':'+Math.round(q.x)+','+Math.round(q.y)+' '+Math.round(q.width)+'x'+Math.round(q.height)));});" +
      "return JSON.stringify({rects:out,scrollY:Math.round(window.scrollY),ih:window.innerHeight});})()";
    console.log((await send("Runtime.evaluate", { expression: ex })).result.value);
    const shot = await send("Page.captureScreenshot", { format: "png" });
    const fs = await import("fs");
    await fs.promises.writeFile("/Users/mck/Desktop/duat/shots/measure.png", Buffer.from(shot.data, "base64"));
    console.log("saved measure");
    ws.close();
    process.exit(0);
  })();
};
