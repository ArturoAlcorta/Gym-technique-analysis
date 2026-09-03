const STATUS_CLASSES = {
  ok: "text-green-400",
  warn: "text-amber-400",
  bad: "text-red-400",
  info: "text-white",
  na: "text-gray-600",
};

const SEVERITY_CLASSES = {
  major: "bg-red-500/10 border-red-500/30 text-red-300",
  moderate: "bg-amber-500/10 border-amber-500/30 text-amber-300",
};

function esc(text) {
  return String(text == null ? "" : text).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function scoreClass(score) {
  if (score == null) return "text-gray-500";
  if (score >= 80) return "text-green-400";
  if (score >= 60) return "text-amber-400";
  return "text-red-400";
}

function secs(value) {
  return value == null ? "—" : `${value.toFixed(2)} s`;
}

function chip(label, value) {
  return `<span class="text-xs text-gray-400 bg-dark border border-gray-700 rounded-full px-2.5 py-1">
            ${esc(label)} <span class="text-gray-200">${esc(value)}</span></span>`;
}

function metricCell(metric) {
  const cls = STATUS_CLASSES[metric.status] || STATUS_CLASSES.info;
  const hint = metric.hint ? ` title="${esc(metric.hint)}"` : "";
  return `
    <div class="bg-dark border border-gray-700 rounded-lg px-3 py-2"${hint}>
      <p class="text-[11px] uppercase tracking-wide text-gray-500 truncate">${esc(metric.label)}</p>
      <p class="text-sm font-medium ${cls} mt-0.5">${esc(metric.display)}</p>
    </div>`;
}

function comparisonBlock(cmp) {
  if (!cmp) return "";
  const joints = cmp.by_joint.map((j) => `
    <div class="flex items-center gap-2">
      <span class="text-xs text-gray-400 w-20 shrink-0">${esc(j.label)}</span>
      <div class="flex-1 h-1.5 bg-dark rounded-full overflow-hidden border border-gray-700">
        <div class="h-full bg-primary" style="width: ${Math.max(0, Math.min(100, j.score))}%"></div>
      </div>
      <span class="text-xs ${scoreClass(j.score)} w-10 text-right">${j.score.toFixed(0)}</span>
    </div>`).join("");

  return `
    <div class="mt-3 border-t border-gray-700 pt-3">
      <div class="flex items-baseline justify-between gap-3 mb-2">
        <p class="text-xs text-gray-500">
          Closest reference: <span class="text-gray-300">${esc(cmp.best_reference || "—")}</span>
          <span class="text-gray-600">· mean over ${cmp.n_references} refs
          ${cmp.mean_score == null ? "—" : cmp.mean_score.toFixed(1)}</span>
        </p>
        <p class="text-sm font-bold ${scoreClass(cmp.best_score)}">
          ${cmp.best_score == null ? "—" : cmp.best_score.toFixed(1)}<span class="text-gray-600 text-xs">/100</span>
        </p>
      </div>
      <div class="space-y-1.5">${joints}</div>
    </div>`;
}

function faultsBlock(faults) {
  if (!faults.length) return "";
  return `<div class="mt-3 space-y-2">${faults.map((f) => `
    <p class="text-xs border rounded-lg px-3 py-2 ${SEVERITY_CLASSES[f.severity] || SEVERITY_CLASSES.moderate}">
      ${esc(f.cue)}
    </p>`).join("")}</div>`;
}

function repCard(rep) {
  const cmp = rep.comparison;
  return `
    <section class="bg-card border border-gray-700 rounded-xl p-4">
      <div class="flex flex-wrap items-center gap-2 mb-3">
        <h3 class="text-white font-bold mr-1">Rep ${rep.rep_number}</h3>
        ${chip("total", secs(rep.timing.total_s))}
        ${chip("ecc", secs(rep.timing.eccentric_s))}
        ${chip("con", secs(rep.timing.concentric_s))}
        ${rep.score == null ? "" : `<span class="ml-auto text-right leading-tight">
             <span class="text-lg font-bold ${scoreClass(rep.score)}">${rep.score.toFixed(1)}</span>
             <span class="text-gray-600 text-xs">/100</span>
             <span class="block text-[10px] text-gray-600">${esc(scoreBreakdown(rep.pattern_score, rep.metric_score))}</span>
           </span>`}
      </div>
      <div class="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-4 gap-2">
        ${rep.metrics.map(metricCell).join("")}
      </div>
      ${comparisonBlock(cmp)}
      ${faultsBlock(rep.faults)}
    </section>`;
}

function scoreBreakdown(pattern, metric) {
  // Which halves actually exist tells the reader what the number is made of:
  // both (the 50/50 split), metrics only (no comparison run), or pattern only.
  const parts = [];
  if (pattern != null) parts.push(`pattern ${pattern.toFixed(1)}`);
  if (metric != null) parts.push(`metrics ${metric.toFixed(1)}`);
  if (parts.length < 2) return parts.length ? `${parts[0]} only` : "";
  return `${parts.join(" + ")} , 50/50`;
}

function summaryHeader(report) {
  const s = report.summary;
  const chips = [
    chip("reps", report.total_reps),
    chip("avg tempo", secs(s.avg_total_s)),
    chip("ecc", secs(s.avg_eccentric_s)),
    chip("con", secs(s.avg_concentric_s)),
    chip("cues", s.n_faults),
  ].join("");

  const note = report.compare
    ? `Half the score is the movement pattern: each rep matched with DTW against
       ${report.reference.n} reference reps (${esc(report.reference.angles.join(", "))}).`
    : `No comparison was run, so the score is the relational half only — measured
       from this video against the reference range. Re-run with “Compare with
       references” to add the movement-pattern half.`;

  return `
    <div class="flex flex-col sm:flex-row gap-4 items-stretch mb-5">
      <div class="shrink-0 bg-dark border border-gray-700 rounded-2xl px-6 py-4 flex flex-col items-center justify-center min-w-[10rem]">
        <p class="text-5xl font-bold leading-none ${scoreClass(s.score)}">
          ${s.score == null ? "—" : s.score.toFixed(1)}
        </p>
        <p class="text-[11px] uppercase tracking-wider text-gray-500 mt-2">Technique score</p>
        <p class="text-[11px] text-gray-600 mt-0.5">${esc(scoreBreakdown(s.pattern_score, s.metric_score))}</p>
      </div>
      <div class="flex-1 min-w-0 flex flex-col justify-center gap-2">
        <div class="flex flex-wrap gap-2">${chips}</div>
        <p class="text-gray-600 text-xs">${note}</p>
      </div>
    </div>`;
}

async function renderReps(analysisId, containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;
  const res = await fetch(`/analyses/${analysisId}/results`);
  if (!res.ok) {
    container.innerHTML = `<p class="text-gray-500 text-sm">No results available for this set.</p>`;
    return;
  }
  const report = await res.json();
  if (!report.reps.length) {
    container.innerHTML = `<p class="text-amber-400 text-sm bg-amber-500/10 border border-amber-500/30 rounded-lg px-4 py-3">
      No complete reps were detected in this video. Check that the whole body stays in frame.</p>`;
    return;
  }
  container.innerHTML = summaryHeader(report) +
    `<div class="space-y-3">${report.reps.map(repCard).join("")}</div>`;
}

// Rendered after htmx settles the swap so the container has its final size.
function mountReps(root) {
  root.querySelectorAll("[data-render-reps]").forEach((el) => {
    if (el.dataset.rendered) return;
    el.dataset.rendered = "1";
    renderReps(el.dataset.renderReps, el.id);
  });
}

document.addEventListener("htmx:afterSettle", (event) => mountReps(event.target));
document.addEventListener("DOMContentLoaded", () => mountReps(document));
