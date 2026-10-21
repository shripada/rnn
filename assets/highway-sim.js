// Reusable "gradient highway" chart: the vanilla RNN path vs a gated cell's direct path.
// The loss sits at the last step T. The signal reaching step t, k = T − t steps back, is
//   vanilla RNN:  (w_h × tanh′)^k                    (Lesson 3)
//   gated path:   g^k, g = forget gate f (LSTM) or update gate z (GRU), held constant here
// Markup: <div class="panel" data-highway-sim data-gate="f" data-gate-name="forget gate f"
//              data-cell-name="LSTM cell path" data-g="0.97" data-t="50"></div>
// Each update dispatches "highway:update" with detail = { T, wh, slope, frnn, g, rnn, gated }
//   (rnn / gated = signal reaching step 1).
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const mk = (p, tag, a = {}, text) => { const n = document.createElementNS(NS, tag); for (const k in a) n.setAttribute(k, a[k]); if (text != null) n.textContent = text; p.appendChild(n); return n; };
  const sci = (v) => (v === 0 ? "0" : v >= 0.01 && v < 1000 ? v.toPrecision(2) : v.toExponential(1).replace("e", "×10^").replace("+", ""));

  function build(root) {
    const gateName = root.dataset.gateName ?? "forget gate f";
    const cellName = root.dataset.cellName ?? "LSTM cell path";
    const st = { wh: 0.9, slope: 0.7, g: parseFloat(root.dataset.g ?? 0.97), T: parseInt(root.dataset.t ?? 50, 10) };

    [
      ["wh", "RNN: w<sub>h</sub>", 0, 1, 0.01, "h"],
      ["slope", "RNN: tanh′", 0.05, 1, 0.01, ""],
      ["g", `${cellName.split(" ")[0]}: ${gateName}`, 0, 1, 0.005, "h"],
      ["T", "T: steps between input and loss", 2, 100, 1, ""],
    ].forEach(([key, label, min, max, step, cls]) => {
      const lab = document.createElement("label");
      lab.innerHTML = `<span class="${cls}" style="min-width:13rem">${label}</span>`;
      const input = Object.assign(document.createElement("input"), { type: "range", min, max, step, value: st[key] });
      const val = Object.assign(document.createElement("span"), { className: "val", textContent: st[key] });
      input.addEventListener("input", () => { st[key] = parseFloat(input.value); val.textContent = input.value; render(); });
      lab.append(input, val);
      root.appendChild(lab);
    });

    const readout = Object.assign(document.createElement("p"), { className: "mono" });
    readout.style.cssText = "font-size:.85rem;margin:.6rem 0 0";
    root.appendChild(readout);
    const svg = mk(root, "svg", { viewBox: "0 0 640 250", role: "img", "aria-label": `Log-scale chart of gradient reaching each step: vanilla RNN vs ${cellName}` });
    svg.style.cssText = "width:100%;height:auto;display:block;margin-top:.4rem";

    function render() {
      const T = st.T, frnn = st.wh * st.slope;
      const rnn = Array.from({ length: T }, (_, i) => Math.pow(frnn, T - 1 - i));
      const gated = Array.from({ length: T }, (_, i) => Math.pow(st.g, T - 1 - i));

      svg.replaceChildren();
      const cs = getComputedStyle(document.documentElement), c = (n) => cs.getPropertyValue(n).trim();
      const left = 46, right = 630, top = 30, bottom = 205, lo = -9, hi = 0;
      const y = (v) => { const l = Math.max(lo, Math.min(hi, Math.log10(Math.max(v, 1e-12)))); return bottom - ((l - lo) / (hi - lo)) * (bottom - top); };
      const x = (i) => left + ((right - left) * (T === 1 ? 0.5 : i / (T - 1)));
      [-9, -6, -3, 0].forEach((p) => {
        mk(svg, "line", { x1: left, x2: right, y1: y(10 ** p), y2: y(10 ** p), stroke: c("--rule"), "stroke-dasharray": p === 0 ? "0" : "2 3" });
        mk(svg, "text", { x: left - 6, y: y(10 ** p) + 4, "text-anchor": "end", fill: c("--muted"), "font-size": 11, "font-family": "monospace" }, p === 0 ? "1" : `1e${p}`);
      });
      const line = (vals, color, dash) => mk(svg, "polyline", { points: vals.map((v, i) => `${x(i)},${y(v)}`).join(" "), fill: "none", stroke: color, "stroke-width": 2.5, "stroke-dasharray": dash, "stroke-linejoin": "round" });
      line(rnn, c("--muted"), "5 4");
      line(gated, c("--accent"), "0");
      [1, Math.round(T / 2), T].forEach((t) => mk(svg, "text", { x: x(t - 1), y: 222, "text-anchor": "middle", fill: c("--muted"), "font-size": 10, "font-family": "monospace" }, t));
      mk(svg, "text", { x: (left + right) / 2, y: 242, "text-anchor": "middle", fill: c("--muted"), "font-size": 11 }, "time step t (loss is at the last step) →");
      // legend
      mk(svg, "line", { x1: left + 4, x2: left + 30, y1: 12, y2: 12, stroke: c("--accent"), "stroke-width": 2.5 });
      mk(svg, "text", { x: left + 36, y: 16, fill: c("--fg"), "font-size": 12 }, cellName);
      mk(svg, "line", { x1: left + 230, x2: left + 256, y1: 12, y2: 12, stroke: c("--muted"), "stroke-width": 2.5, "stroke-dasharray": "5 4" });
      mk(svg, "text", { x: left + 262, y: 16, fill: c("--fg"), "font-size": 12 }, "vanilla RNN");

      readout.innerHTML = `signal reaching step 1 · vanilla RNN: (${frnn.toFixed(3)})<sup>${T - 1}</sup> = <strong>${sci(rnn[0])}</strong> · ${cellName}: (${st.g.toFixed(3)})<sup>${T - 1}</sup> = <strong class="h">${sci(gated[0])}</strong>`;
      root.dispatchEvent(new CustomEvent("highway:update", { detail: { T, wh: st.wh, slope: st.slope, frnn, g: st.g, rnn: rnn[0], gated: gated[0] } }));
    }

    matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", render);
    requestAnimationFrame(render);
  }

  function init() { document.querySelectorAll("[data-highway-sim]").forEach(build); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
