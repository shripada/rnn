// Reusable classification drill: one fixed set of options, many shuffled cards,
// immediate feedback, and a "retry the ones I missed" loop.
// Markup:
// <div class="drill" data-options='[["key","Label"], ...]'>
//   <div class="card" data-answer="key">Prompt text<div class="why">Explanation</div></div> ...
// </div>
// Optional hook: if window.RNNShapes exists, the correct shape's diagram is drawn in the feedback.
(function () {
  const shuffle = (a) => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };

  function build(root) {
    const options = JSON.parse(root.dataset.options);
    const cards = [...root.querySelectorAll(".card")].map((c) => ({
      prompt: c.firstChild.textContent.trim(),
      answer: c.dataset.answer,
      why: c.querySelector(".why")?.innerHTML ?? "",
    }));
    root.replaceChildren();

    const status = Object.assign(document.createElement("p"), { className: "drill-status" });
    const prompt = Object.assign(document.createElement("p"), { className: "drill-prompt" });
    const opts = Object.assign(document.createElement("div"), { className: "drill-opts" });
    const feedback = Object.assign(document.createElement("div"), { className: "drill-feedback" });
    const next = Object.assign(document.createElement("button"), { type: "button", className: "drill-next", textContent: "Next →" });
    root.append(status, prompt, opts, feedback, next);

    let queue = shuffle(cards.slice()), missed = [], i = 0, firstTryRight = 0, round = 1;

    function show() {
      if (i >= queue.length) return finish();
      const card = queue[i];
      status.textContent = `${round > 1 ? "Retry round · " : ""}Card ${i + 1} of ${queue.length}`;
      prompt.textContent = card.prompt;
      feedback.replaceChildren(); feedback.className = "drill-feedback";
      next.style.display = "none";
      opts.replaceChildren(...options.map(([key, label]) => {
        const b = Object.assign(document.createElement("button"), { type: "button", className: "opt", textContent: label });
        b.addEventListener("click", () => answer(card, key, b));
        return b;
      }));
    }

    function answer(card, key, btn) {
      const ok = key === card.answer;
      [...opts.children].forEach((b, j) => {
        b.disabled = true;
        if (options[j][0] === card.answer) b.classList.add("correct");
      });
      if (!ok) { btn.classList.add("wrong"); missed.push(card); } else if (round === 1) firstTryRight++;
      feedback.className = "drill-feedback " + (ok ? "good" : "bad");
      feedback.innerHTML = `<strong>${ok ? "Right." : "Not quite."}</strong> ${card.why}`;
      if (window.RNNShapes?.SHAPES[card.answer]) {
        const holder = document.createElement("div");
        feedback.appendChild(holder);
        window.RNNShapes.render(holder, card.answer);
      }
      next.style.display = "inline-block";
      next.focus();
    }

    function finish() {
      if (missed.length) {
        status.textContent = `${round === 1 ? `First pass: ${firstTryRight} of ${cards.length} right.` : "Still some to fix."} Retrying the ${missed.length} you missed.`;
        queue = shuffle(missed); missed = []; i = 0; round++;
        prompt.textContent = ""; opts.replaceChildren(); feedback.replaceChildren();
        next.textContent = "Start retry →"; next.style.display = "inline-block";
        next.onclick = () => { next.textContent = "Next →"; next.onclick = null; show(); };
        return;
      }
      status.textContent = `Done. ${firstTryRight} of ${cards.length} right on the first pass.`;
      prompt.textContent = ""; opts.replaceChildren(); feedback.replaceChildren(); next.style.display = "none";
    }

    next.addEventListener("click", () => { if (!next.onclick) { i++; show(); } });
    show();
  }

  function init() { document.querySelectorAll(".drill").forEach(build); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
