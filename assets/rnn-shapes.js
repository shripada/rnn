// Reusable unrolled-RNN "task shape" diagrams.
//   Static:  <figure data-rnn-shape="many-to-one"></figure>
//   Gallery: <div data-rnn-shapes></div>   (tabs for every shape + caption)
// Exposes window.RNNShapes = { SHAPES, render(el, key) } for other components (e.g. drill.js).
(function () {
  const NS = "http://www.w3.org/2000/svg";

  // in/out: which steps receive an input / emit an output that we use.
  // feedback: the step's input is the previous step's output (generation).
  // split: index where the decoder starts (encoder–decoder only).
  const SHAPES = {
    "one-to-one": {
      label: "One-to-one", T: 1, in: [1], out: [1], rec: false,
      caption: "No recurrence at all: fixed-size input → fixed-size output. A plain feed-forward net, shown for contrast. Example: image classification.",
    },
    "one-to-many": {
      label: "One-to-many", T: 4, in: [1, 0, 0, 0], out: [1, 1, 1, 1], feedback: true,
      caption: "One input, a sequence out. Each generated output is fed back in as the next input (dashed). Example: image captioning: an image in, a sentence out.",
    },
    "many-to-one": {
      label: "Many-to-one", T: 4, in: [1, 1, 1, 1], out: [0, 0, 0, 1],
      caption: "A sequence in, one output. Only the final hidden state, the summary of the whole sequence, feeds the prediction. Example: sentiment analysis of a sentence.",
    },
    "synced": {
      label: "Many-to-many (synced)", T: 4, in: [1, 1, 1, 1], out: [1, 1, 1, 1],
      caption: "One output per input, aligned step by step, so lengths always match. Example: labelling every frame of a video, or tagging every word in a sentence.",
    },
    "seq2seq": {
      label: "Encoder–decoder (seq2seq)", T: 7, in: [1, 1, 1, 0, 0, 0, 0], out: [0, 0, 0, 1, 1, 1, 1], split: 3, feedback: true,
      caption: "Read the whole input first (encoder), squeeze it into a fixed-size context vector c, then generate the output token by token (decoder) until <eos>. Lengths and word order may differ. Example: machine translation.",
    },
  };

  const mk = (parent, tag, attrs = {}, text) => {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    parent.appendChild(n);
    return n;
  };
  const sub = (i) => String(i).split("").map((d) => "₀₁₂₃₄₅₆₇₈₉"[d]).join("");

  let uid = 0;
  function render(container, key) {
    const s = SHAPES[key];
    const gap = 78, left = 60;
    const W = Math.max(420, left + gap * s.T + 30);
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} 215`);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", `${s.label} RNN, unrolled over ${s.T} step${s.T > 1 ? "s" : ""}`);
    svg.style.maxWidth = `${W * 1.15}px`;
    const id = `rs${uid++}`;
    const defs = mk(svg, "defs");
    [["h", "d-hf"], ["x", "d-xf"], ["y", "d-yf"], ["m", "d-m"]].forEach(([n, cls]) => {
      const m = mk(defs, "marker", { id: `${id}${n}`, viewBox: "0 0 8 8", refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: "auto" });
      mk(m, "path", { d: "M0 0 L8 4 L0 8 z", class: cls });
    });

    const cy = 105, tokIn = 196, tokOut = 22;
    let decStep = 0, encStep = 0;
    for (let t = 0; t < s.T; t++) {
      const cx = left + gap * t + gap / 2;
      const isDec = s.split != null && t >= s.split;
      mk(svg, "rect", { x: cx - 22, y: cy - 16, width: 44, height: 32, rx: 6, class: "d-h", "stroke-width": 1.6, "stroke-dasharray": isDec ? "0" : "0" });
      mk(svg, "text", { x: cx, y: cy + 5, "text-anchor": "middle", class: "d-hf", "font-style": "italic", "font-size": 13 }, `h${sub(t + 1)}`);

      // input
      if (s.in[t]) {
        encStep++;
        mk(svg, "line", { x1: cx, y1: tokIn - 16, x2: cx, y2: cy + 19, class: "d-x", "stroke-width": 1.4, "marker-end": `url(#${id}x)` });
        mk(svg, "text", { x: cx, y: tokIn, "text-anchor": "middle", class: "d-xf", "font-style": "italic", "font-size": 13 }, `x${sub(encStep)}`);
      } else if (s.feedback) {
        // input is the previous output (or <bos> at the first decoder step)
        mk(svg, "line", { x1: cx, y1: tokIn - 16, x2: cx, y2: cy + 19, class: "d-m", stroke: "currentColor", "stroke-width": 1.2, "stroke-dasharray": "3 3", "marker-end": `url(#${id}m)`, style: "stroke: var(--muted)" });
        const fromPrevOut = t > 0 && s.out[t - 1];
        if (fromPrevOut) {
          const px = cx - gap;
          mk(svg, "path", { d: `M${px + 10} ${tokOut + 6} C ${px + 45} ${tokOut + 6}, ${cx - 30} ${tokIn - 10}, ${cx - 4} ${tokIn - 14}`, fill: "none", "stroke-width": 1.1, "stroke-dasharray": "3 3", style: "stroke: var(--muted)" });
        } else {
          mk(svg, "text", { x: cx, y: tokIn, "text-anchor": "middle", class: "d-m", "font-size": 11 }, "<bos>");
        }
      }

      // output
      if (s.out[t]) {
        decStep++;
        mk(svg, "line", { x1: cx, y1: cy - 17, x2: cx, y2: tokOut + 12, class: "d-y", "stroke-width": 1.4, "marker-end": `url(#${id}y)` });
        const last = t === s.T - 1 && s.split != null;
        mk(svg, "text", { x: cx, y: tokOut + 4, "text-anchor": "middle", class: "d-yf", "font-style": last ? "normal" : "italic", "font-size": last ? 11 : 13 }, last ? "<eos>" : `y${sub(decStep)}`);
      }

      // recurrent arrow
      if (s.rec !== false && t < s.T - 1) {
        const boundary = s.split != null && t === s.split - 1;
        mk(svg, "line", { x1: cx + 23, y1: cy, x2: cx + gap - 25, y2: cy, class: "d-h", "stroke-width": boundary ? 3 : 1.4, "marker-end": `url(#${id}h)` });
        if (boundary) mk(svg, "text", { x: cx + gap / 2, y: cy - 8, "text-anchor": "middle", class: "d-hf", "font-size": 13, "font-weight": 600 }, "c");
      }
    }

    // encoder / decoder brackets
    if (s.split != null) {
      const x0 = left + 4, x1 = left + gap * s.split - 4, x2 = left + gap * s.T - 4;
      [[x0, x1, "encoder"], [x1 + 8, x2, "decoder"]].forEach(([a, b, txt]) => {
        mk(svg, "path", { d: `M${a} ${cy + 30} v5 H${b} v-5`, fill: "none", "stroke-width": 1, style: "stroke: var(--rule)" });
        mk(svg, "text", { x: (a + b) / 2, y: cy + 48, "text-anchor": "middle", class: "d-m", "font-size": 11, "letter-spacing": "1" }, txt);
      });
    }

    container.replaceChildren(svg);
    return svg;
  }

  function gallery(root) {
    const tabs = document.createElement("div");
    tabs.className = "tabs";
    tabs.setAttribute("role", "tablist");
    const fig = document.createElement("figure");
    const art = document.createElement("div");
    const cap = document.createElement("figcaption");
    fig.append(art, cap);
    const select = (key) => {
      render(art, key);
      cap.textContent = SHAPES[key].caption;
      tabs.querySelectorAll("button").forEach((b) => b.setAttribute("aria-selected", b.dataset.key === key));
    };
    Object.entries(SHAPES).forEach(([key, s]) => {
      const b = Object.assign(document.createElement("button"), { type: "button", textContent: s.label });
      b.dataset.key = key;
      b.setAttribute("role", "tab");
      b.addEventListener("click", () => select(key));
      tabs.appendChild(b);
    });
    root.append(tabs, fig);
    select(root.dataset.start || "one-to-one");
  }

  function init() {
    document.querySelectorAll("[data-rnn-shape]").forEach((el) => {
      const holder = document.createElement("div");
      el.prepend(holder);
      render(holder, el.dataset.rnnShape);
    });
    document.querySelectorAll("[data-rnn-shapes]").forEach(gallery);
  }
  window.RNNShapes = { SHAPES, render };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
