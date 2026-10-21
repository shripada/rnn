// Reusable gradient-flow simulator for a scalar RNN trained with BPTT.
// The loss sits at the last step T. The signal reaching step t is f^(T−t),
// where f = w_h × tanh′ is the per-step multiplier (Lesson 3).
// Markup: <div class="panel" data-grad-sim data-wh="0.9" data-slope="0.7" data-t="10" data-clip></div>
//   data-clip (optional) shows a gradient-norm-clipping control.
// Each update dispatches "grad:update" with detail = { g, raw, f, T, wh, slope, clip, threshold }.
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const mk = (p, tag, a = {}, text) => { const n = document.createElementNS(NS, tag); for (const k in a) n.setAttribute(k, a[k]); if (text != null) n.textContent = text; p.appendChild(n); return n; };
  const sci = (v) => (v === 0 ? "0" : v >= 0.01 && v < 1000 ? v.toPrecision(2) : v.toExponential(1).replace("e", "×10^").replace("+", ""));

  function build(root) {
    const st = {
      wh: parseFloat(root.dataset.wh ?? 0.9),
      slope: parseFloat(root.dataset.slope ?? 0.7),
      T: parseInt(root.dataset.t ?? 10, 10),
      clip: false,
      threshold: 5,
    };

    const controls = [
      ["wh", "w<sub>h</sub>: recurrent weight", 0, 2, 0.01, "h"],
      ["slope", "tanh′: slope of tanh (1 = unsaturated)", 0.05, 1, 0.01, ""],
      ["T", "T: steps between input and loss", 2, 50, 1, ""],
    ];
    if (root.hasAttribute("data-clip")) controls.push(["threshold", "clipping threshold θ", 1, 50, 1, ""]);

    const inputs = {};
    controls.forEach(([key, label, min, max, step, cls]) => {
      const lab = document.createElement("label");
      lab.innerHTML = `<span class="${cls}" style="min-width:16rem">${label}</span>`;
      const input = Object.assign(document.createElement("input"), { type: "range", min, max, step, value: st[key] });
      const val = Object.assign(document.createElement("span"), { className: "val", textContent: st[key] });
      input.addEventListener("input", () => { st[key] = parseFloat(input.value); val.textContent = input.value; render(); });
      lab.append(input, val);
      root.appendChild(lab);
      inputs[key] = lab;
    });
    if (root.hasAttribute("data-clip")) {
      const lab = document.createElement("label");
      const box = Object.assign(document.createElement("input"), { type: "checkbox" });
      box.addEventListener("change", () => { st.clip = box.checked; render(); });
      lab.append(box, Object.assign(document.createElement("span"), { textContent: "turn on gradient clipping" }));
      root.insertBefore(lab, inputs.threshold);
    }

    const readout = Object.assign(document.createElement("p"), { className: "mono" });
    readout.style.cssText = "font-size:.85rem;margin:.6rem 0 0";
    root.appendChild(readout);
    const svg = mk(root, "svg", { viewBox: "0 0 640 240", role: "img", "aria-label": "Log-scale bar chart of the gradient signal reaching each time step" });
    svg.style.cssText = "width:100%;height:auto;display:block;margin-top:.4rem";

    function render() {
      const f = st.wh * st.slope, T = st.T;
      const raw = Array.from({ length: T }, (_, i) => Math.pow(f, T - 1 - i)); // step i+1
      let g = raw.slice(), clipped = false;
      if (st.clip) {
        const norm = Math.sqrt(raw.reduce((s, v) => s + v * v, 0));
        if (norm > st.threshold) { g = raw.map((v) => (v * st.threshold) / norm); clipped = true; }
      }

      svg.replaceChildren();
      const cs = getComputedStyle(document.documentElement);
      const c = (n) => cs.getPropertyValue(n).trim();
      const left = 46, right = 630, top = 14, bottom = 200, lo = -6, hi = 6;
      const y = (v) => { const l = Math.max(lo, Math.min(hi, Math.log10(Math.max(v, 1e-12)))); return bottom - ((l - lo) / (hi - lo)) * (bottom - top); };
      [-6, -3, 0, 3, 6].forEach((p) => {
        mk(svg, "line", { x1: left, x2: right, y1: y(10 ** p), y2: y(10 ** p), stroke: c("--rule"), "stroke-dasharray": p === 0 ? "0" : "2 3" });
        mk(svg, "text", { x: left - 6, y: y(10 ** p) + 4, "text-anchor": "end", fill: c("--muted"), "font-size": 11, "font-family": "monospace" }, p === 0 ? "1" : `1e${p}`);
      });
      const colW = (right - left) / T, bw = Math.max(3, Math.min(28, colW * 0.7));
      g.forEach((v, i) => {
        const cx = left + colW * i + colW / 2, yv = y(v), y1 = y(1);
        const under = Math.log10(Math.max(v, 1e-12)) < lo;
        mk(svg, "rect", { x: cx - bw / 2, y: Math.min(yv, y1), width: bw, height: Math.max(1, Math.abs(yv - y1)), fill: under ? c("--muted") : c("--accent"), opacity: under ? 0.4 : 0.9, rx: 1.5 });
        if (T <= 20 || i === 0 || i === T - 1 || (i + 1) % 10 === 0)
          mk(svg, "text", { x: cx, y: 222, "text-anchor": "middle", fill: c("--muted"), "font-size": 10, "font-family": "monospace" }, i + 1);
      });
      mk(svg, "text", { x: (left + right) / 2, y: 237, "text-anchor": "middle", fill: c("--muted"), "font-size": 11 }, "time step t (loss is at the last step) →");

      readout.innerHTML = `per-step factor f = w<sub>h</sub> × tanh′ = <strong>${f.toFixed(3)}</strong> · signal reaching step 1 = f<sup>${T - 1}</sup> = <strong>${sci(g[0])}</strong>${clipped ? " (clipped)" : ""}`;
      root.dispatchEvent(new CustomEvent("grad:update", { detail: { g, raw, f, T, wh: st.wh, slope: st.slope, clip: st.clip, clipped, threshold: st.threshold } }));
    }

    matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", render);
    requestAnimationFrame(render);
  }

  function init() { document.querySelectorAll("[data-grad-sim]").forEach(build); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
