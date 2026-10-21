// Reusable auto-checked challenge list for any simulator that emits a CustomEvent.
// Markup:
// <div class="panel" id="my-challenges">
//   <div class="challenge" data-c="key"><span class="tick">○</span><span>Text…</span></div>
// </div>
// JS:  Challenges.attach(simEl, "rnn:update", listEl, { key: (detail) => boolean, ... })
// A challenge ticks once its check passes and stays ticked.
window.Challenges = {
  attach(sim, eventName, list, checks) {
    sim.addEventListener(eventName, (e) => {
      list.querySelectorAll(".challenge").forEach((c) => {
        if (c.classList.contains("done")) return;
        const check = checks[c.dataset.c];
        if (check && check(e.detail)) {
          c.classList.add("done");
          c.querySelector(".tick").textContent = "✓";
        }
      });
    });
  },
};
