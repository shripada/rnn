// Reusable multiple-choice quiz with immediate feedback.
// Markup:
// <div class="quiz">
//   <div class="q">Question?</div>
//   <button class="opt" data-correct>Right answer</button>
//   <button class="opt">Wrong answer</button> ...
//   <div class="why">Explanation shown after answering.</div>
// </div>
// Optional: <p class="quiz-score" data-quiz-score></p> shows a running score.
(function () {
  function shuffle(nodes) {
    for (let i = nodes.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [nodes[i], nodes[j]] = [nodes[j], nodes[i]];
    }
    return nodes;
  }

  function init() {
    const quizzes = [...document.querySelectorAll(".quiz")];
    const scoreEl = document.querySelector("[data-quiz-score]");
    let answered = 0, correct = 0;

    function updateScore() {
      if (!scoreEl) return;
      scoreEl.textContent = answered === quizzes.length
        ? `You got ${correct} of ${quizzes.length} on the first try.`
        : `${answered} of ${quizzes.length} answered.`;
    }

    quizzes.forEach((quiz) => {
      const opts = [...quiz.querySelectorAll("button.opt")];
      const why = quiz.querySelector(".why");
      // Shuffle so position never hints at the answer.
      shuffle(opts).forEach((o) => quiz.insertBefore(o, why));

      opts.forEach((opt) => {
        opt.type = "button";
        opt.addEventListener("click", () => {
          if (quiz.classList.contains("answered")) return;
          quiz.classList.add("answered");
          const ok = opt.hasAttribute("data-correct");
          answered++; if (ok) correct++;
          opts.forEach((o) => {
            o.disabled = true;
            if (o.hasAttribute("data-correct")) o.classList.add("correct");
          });
          if (!ok) opt.classList.add("wrong");
          updateScore();
        });
      });
    });
    updateScore();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
