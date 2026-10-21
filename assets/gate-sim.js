// Reusable scalar gated-cell simulator: you set every gate by hand, one value per step.
// Markup: <div class="panel" data-gate-sim="lstm" data-seq="2,0,0,1,0,0,0,0"></div>
//   data-gate-sim = "lstm" | "gru"
// Click a gate cell to cycle it 0 → 0.5 → 1. Inputs x_t are editable.
// Simplified scalar cells (the real ones compute each gate from x_t and h_{t−1}):
//   LSTM: c̃ = tanh(x),           c = f·c_prev + i·c̃,   h = o·tanh(c)
//   GRU:  h̃ = tanh(x + r·h_prev), h = z·h_prev + (1 − z)·h̃   (z = 1 keeps the old state)
// Each update dispatches "gate:update" with detail = { x, gates, cand, c, h }.
(function () {
  const CELLS = {
    lstm: {
      gates: [["f", "forget gate: how much old c to keep"], ["i", "input gate: how much new candidate to write"], ["o", "output gate: how much of c to show as h"]],
      outs: [["cand", "c̃", "candidate = tanh(x)"], ["c", "c", "cell state"], ["h", "h", "hidden state (the output)"]],
      run(x, g) {
        const cand = [], c = [], h = []; let cp = 0;
        x.forEach((xt, t) => {
          const ct = Math.tanh(xt);
          cp = g.f[t] * cp + g.i[t] * ct;
          cand.push(ct); c.push(cp); h.push(g.o[t] * Math.tanh(cp));
        });
        return { cand, c, h };
      },
    },
    gru: {
      gates: [["z", "update gate: 1 = keep old h, 0 = take the candidate"], ["r", "reset gate: how much old h the candidate may see"]],
      outs: [["cand", "h̃", "candidate = tanh(x + r·h_prev)"], ["h", "h", "hidden state"]],
      run(x, g) {
        const cand = [], h = []; let hp = 0;
        x.forEach((xt, t) => {
          const ht = Math.tanh(xt + g.r[t] * hp);
          hp = g.z[t] * hp + (1 - g.z[t]) * ht;
          cand.push(ht); h.push(hp);
        });
        return { cand, h };
      },
    },
  };
  const LEVELS = [0, 0.5, 1];
  const short = (v) => { const s = Math.abs(v) < 0.005 ? "0" : v.toFixed(2).replace(/^(-?)0\./, "$1."); return s.replace(/\.?0+$/, "") || "0"; };

  function build(root) {
    const spec = CELLS[root.dataset.gateSim];
    const initialSeq = (root.dataset.seq ?? "2,0,0,0,0,0,0,0").split(",").map(Number);
    const T = initialSeq.length;
    let x = initialSeq.slice();
    const fresh = () => Object.fromEntries(spec.gates.map(([k]) => [k, Array(T).fill(0.5)]));
    let gates = fresh();

    const grid = document.createElement("div");
    grid.className = "gate-grid";
    grid.style.gridTemplateColumns = `2.4rem repeat(${T}, minmax(0, 1fr))`;
    root.appendChild(grid);
    const cell = (cls, text) => { const d = document.createElement("div"); d.className = cls; if (text != null) d.textContent = text; grid.appendChild(d); return d; };

    // header row: step numbers
    cell("gg-label", "t");
    for (let t = 0; t < T; t++) cell("gg-head", t + 1);
    // inputs
    cell("gg-label x", "x");
    const xInputs = x.map((v, t) => {
      const inp = Object.assign(document.createElement("input"), { type: "number", step: 0.5, min: -3, max: 3, value: v, className: "gg-x", ariaLabel: `x${t + 1}` });
      inp.addEventListener("input", () => { x[t] = parseFloat(inp.value) || 0; render(); });
      grid.appendChild(inp);
      return inp;
    });
    // gate rows: click to cycle
    const gateBtns = {};
    spec.gates.forEach(([k, desc]) => {
      cell("gg-label", k).title = desc;
      gateBtns[k] = Array.from({ length: T }, (_, t) => {
        const b = Object.assign(document.createElement("button"), { type: "button", className: "gg-gate", ariaLabel: `${k} at step ${t + 1}` });
        b.addEventListener("click", () => { gates[k][t] = LEVELS[(LEVELS.indexOf(gates[k][t]) + 1) % LEVELS.length]; render(); });
        grid.appendChild(b);
        return b;
      });
    });
    // output rows
    const outCells = {};
    spec.outs.forEach(([k, sym, desc]) => {
      const lab = cell("gg-label" + (k === "cand" ? "" : " h"), sym);
      lab.title = desc;
      outCells[k] = Array.from({ length: T }, () => cell("gg-out" + (k === "cand" ? " cand" : "")));
    });

    const legend = document.createElement("p");
    legend.className = "gg-legend";
    legend.innerHTML = spec.gates.map(([k, d]) => `<strong>${k}</strong>: ${d}`).join(" · ")
      + "<br>" + spec.outs.map(([, s, d]) => `<strong>${s}</strong>: ${d}`).join(" · ")
      + "<br>Click a gate to cycle it 0 → 0.5 → 1. You are playing the part of the learned gate weights.";
    const reset = Object.assign(document.createElement("button"), { type: "button", className: "small", textContent: "reset" });
    reset.addEventListener("click", () => { x = initialSeq.slice(); gates = fresh(); xInputs.forEach((s, t) => (s.value = x[t])); render(); });
    legend.append(" ", reset);
    root.appendChild(legend);

    function render() {
      const out = spec.run(x, gates);
      spec.gates.forEach(([k]) => gateBtns[k].forEach((b, t) => {
        const v = gates[k][t];
        b.textContent = short(v);
        b.style.background = `color-mix(in srgb, var(--accent) ${Math.round(v * 70)}%, var(--bg))`;
        b.style.color = v === 1 ? "var(--bg)" : "var(--fg)";
      }));
      for (const k in outCells) outCells[k].forEach((d, t) => {
        const v = out[k][t];
        d.textContent = short(v);
        d.style.setProperty("--fill", `${Math.min(100, Math.abs(v) * 50)}%`);
        d.classList.toggle("neg", v < 0);
      });
      root.dispatchEvent(new CustomEvent("gate:update", { detail: { x: x.slice(), gates, ...out } }));
    }
    render();
  }

  function init() { document.querySelectorAll("[data-gate-sim]").forEach(build); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
