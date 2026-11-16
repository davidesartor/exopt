const data = JSON.parse(document.getElementById("review-data").textContent);
const darkColors = JSON.parse(document.getElementById("dark-colors").textContent);
const swap = new RegExp(Object.keys(darkColors).map((c) => c.replace(/[()|.]/g, "\\$&")).join("|"), "g");
const config = { displaylogo: false, responsive: true };

const theme = () => (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
const themed = (figure) => JSON.parse(theme() === "dark" ? figure.replace(swap, (c) => darkColors[c]) : figure);
const draw = (id, figure) => { const f = themed(figure); Plotly.react(id, f.data, f.layout, config); };

let shown = data.best;  // position of the trial shown
function showTrial(position) {
  shown = Math.min(Math.max(position, 0), data.trials.length - 1);
  const trial = data.trials[shown];
  document.getElementById("summary").textContent = trial.summary;
  for (const name of Object.keys(trial.figures)) draw("trial-" + name, trial.figures[name]);
}
function drawAll() {
  for (const name of Object.keys(data.optimization)) draw(name, data.optimization[name]);
  if (data.trials.length) showTrial(shown);
}

matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawAll);
drawAll();

// a click on any trial marker or line shows that trial
const positions = new Map(data.trials.map((trial, position) => [trial.index, position]));
for (const name of Object.keys(data.optimization)) {
  document.getElementById(name).on("plotly_click", (event) => {
    const custom = event.points[0].customdata;
    const index = Number(Array.isArray(custom) ? custom[0] : custom);
    if (positions.has(index)) showTrial(positions.get(index));
  });
}
