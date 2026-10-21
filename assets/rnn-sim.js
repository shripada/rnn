// Reusable scalar RNN simulator: h_t = tanh(w_x * x_t + w_h * h_{t-1} + b), h_0 = 0.
// Markup: <div class="panel" data-rnn-sim data-wx="1" data-wh="0.5" data-b="0" data-seq="1,0,0,0,0,0"></div>
// Optional: data-lock="b" hides sliders by name (comma-separated: wx,wh,b).
// Each update dispatches a "rnn:update" CustomEvent on the container with
// detail = { x: [...], h: [...], wx, wh, b } so lessons can attach challenges.
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs = {}, parent) => {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  };
  const fmt = (v) => (v >= 0 ? " " : "") + v.toFixed(2);

  function run(x, wx, wh, b) {
    const h = []; let prev = 0;
    for (const xt of x) { prev = Math.tanh(wx * xt + wh * prev + b); h.push(prev); }
    return h;
  }

  function build(root) {
    const params = {
      wx: parseFloat(root.dataset.wx ?? 1),
      wh: parseFloat(root.dataset.wh ?? 0.5),
      b: parseFloat(root.dataset.b ?? 0),
    };
    const initialSeq = (root.dataset.seq ?? "1,0,0,0,0,0").split(",").map(Number);
    const locked = new Set((root.dataset.lock ?? "").split(",").filter(Boolean));
    let x = initialSeq.slice();

    const sliders = [
      ["wx", "w<sub>x</sub> — weight on the input", -2, 2, "x"],
      ["wh", "w<sub>h</sub> — weight on the previous hidden state", -2, 2, "h"],
      ["b", "b — bias", -1, 1, ""],
    ];
    sliders.forEach(([key, label, min, max, cls]) => {
      if (locked.has(key)) return;
      const lab = document.createElement("label");
      lab.innerHTML = `<span class="${cls}" style="min-width:16rem">${label}</span>`;
      const input = Object.assign(document.createElement("input"), { type: "range", min, max, step: 0.05, value: params[key] });
      const val = Object.assign(document.createElement("span"), { className: "val", textContent: fmt(params[key]) });
      input.addEventListener("input", () => { params[key] = parseFloat(input.value); val.textContent = fmt(params[key]); render(); });
      lab.append(input, val);
      root.appendChild(lab);
    });

    const seqRow = document.createElement("div");
    seqRow.className = "row";
    seqRow.innerHTML = `<span class="x">Input sequence x<sub>1</sub>…x<sub>${x.length}</sub>:</span>`;
    const seqInputs = x.map((v, i) => {
      const inp = Object.assign(document.createElement("input"), { className: "seq", type: "number", step: 0.5, value: v, ariaLabel: `x${i + 1}` });
      inp.addEventListener("input", () => { x[i] = parseFloat(inp.value) || 0; render(); });
      seqRow.appendChild(inp);
      return inp;
    });
    const reset = Object.assign(document.createElement("button"), { className: "small", textContent: "reset", type: "button" });
    reset.addEventListener("click", () => { x = initialSeq.slice(); seqInputs.forEach((s, i) => (s.value = x[i])); render(); });
    seqRow.appendChild(reset);
    root.appendChild(seqRow);

    const svg = el("svg", { viewBox: "0 0 640 230", role: "img", "aria-label": "Unrolled scalar RNN showing hidden state at each step" });
    svg.style.marginTop = ".8rem";
    root.appendChild(svg);

    function render() {
      const h = run(x, params.wx, params.wh, params.b);
      svg.replaceChildren();
      const T = x.length, colW = 600 / T, left = 30;
      const style = getComputedStyle(document.documentElement);
      const cAcc = style.getPropertyValue("--accent").trim();
      const cIn = style.getPropertyValue("--input").trim();
      const cRule = style.getPropertyValue("--rule").trim();
      const cMuted = style.getPropertyValue("--muted").trim();
      const cFg = style.getPropertyValue("--fg").trim();

      const hMid = 80, hAmp = 55;   // hidden-state bar chart band
      const cellY = 160;            // cell row
      el("line", { x1: left - 10, x2: 630, y1: hMid, y2: hMid, stroke: cRule }, svg);
      el("text", { x: 0, y: hMid - hAmp + 4, fill: cMuted, "font-size": 11, "font-family": "monospace" }, svg).textContent = "+1";
      el("text", { x: 0, y: hMid + hAmp + 4, fill: cMuted, "font-size": 11, "font-family": "monospace" }, svg).textContent = "−1";

      for (let t = 0; t < T; t++) {
        const cx = left + colW * t + colW / 2;
        // hidden-state bar
        const barH = Math.abs(h[t]) * hAmp;
        el("rect", { x: cx - 14, y: h[t] >= 0 ? hMid - barH : hMid, width: 28, height: Math.max(barH, 0.5), fill: cAcc, rx: 2 }, svg);
        el("text", { x: cx, y: h[t] >= 0 ? hMid - barH - 6 : hMid + barH + 14, "text-anchor": "middle", fill: cFg, "font-size": 12, "font-family": "monospace" }, svg).textContent = h[t].toFixed(2);
        // RNN cell
        el("rect", { x: cx - 22, y: cellY - 16, width: 44, height: 32, rx: 6, fill: "none", stroke: cAcc, "stroke-width": 1.5 }, svg);
        el("text", { x: cx, y: cellY + 5, "text-anchor": "middle", fill: cAcc, "font-size": 13, "font-style": "italic" }, svg).textContent = `h${t + 1}`;
        // input below
        el("text", { x: cx, y: 222, "text-anchor": "middle", fill: cIn, "font-size": 12, "font-family": "monospace" }, svg).textContent = `x=${x[t]}`;
        el("line", { x1: cx, x2: cx, y1: 206, y2: cellY + 18, stroke: cIn, "stroke-width": 1.2 }, svg);
        // recurrent arrow to next cell
        if (t < T - 1) {
          const x1 = cx + 22, x2 = cx + colW - 24;
          el("line", { x1, x2, y1: cellY, y2: cellY, stroke: cAcc, "stroke-width": 1.2 }, svg);
          el("path", { d: `M${x2} ${cellY} l-6 -4 v8 z`, fill: cAcc }, svg);
        }
      }
      root.dispatchEvent(new CustomEvent("rnn:update", { detail: { x: x.slice(), h, ...params } }));
    }

    // Re-render when the colour scheme flips so SVG colours follow the theme.
    matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", render);
    root.rnnRender = render;
    requestAnimationFrame(render);
  }

  function init() { document.querySelectorAll("[data-rnn-sim]").forEach(build); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
