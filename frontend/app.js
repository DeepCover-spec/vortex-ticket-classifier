/**
 * VORTEX demo client.
 *
 * Calls same-origin routes only. The API key never enters this file.
 * Metrics are read from /static/metrics.json. Missing figures are hidden.
 *
 * Recognized metrics.json fields, all optional:
 * accuracy, macro_f1, categories, per_class_f1,
 * confusion_matrix, confusion_labels,
 * model_type, features, training_approach, evaluation_metric
 */
(function () {
  document.documentElement.classList.add("js");
  const BATCH_LIMIT = 100;
  const GAUGE_LENGTH = 326.73;

  const SAMPLES = {
    password: "I cannot reset my password and I am locked out of my account.",
    billing: "I was charged twice for the same ride.",
    delivery: "My food order was supposed to arrive an hour ago and it still has not arrived.",
    hardware: "The app closes every time I open my order history.",
  };

  const BATCH_SAMPLES = [
    "I was charged twice for the same ride.",
    "My food order has not arrived and the map still shows the driver far away.",
    "I left my bag in the car after the trip.",
    "The app closes every time I open my order history.",
    "The driver was rude and I felt unsafe during the ride.",
  ];

  const state = {
    rows: [],
    sortKey: "index",
    sortDir: 1,
    metrics: null,
    status: null,
    bandFilter: "all",
    motion: true,
    canvasFrame: 0,
  };

  const $ = (id) => document.getElementById(id);

  document.addEventListener("DOMContentLoaded", init);

  function init() {
    bindNavigation();
    bindMotion();
    bindPointer();
    bindReveal();
    bindClassify();
    bindBatch();
    refreshStatus();
    loadMetrics();
    window.setInterval(refreshStatus, 20000);
  }

  function bindNavigation() {
    const toggle = document.querySelector(".nav-toggle");
    const nav = $("site-nav");
    toggle.addEventListener("click", () => {
      const open = nav.classList.toggle("is-open");
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
    nav.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", () => {
        nav.classList.remove("is-open");
        toggle.setAttribute("aria-expanded", "false");
      });
    });

    const links = [...nav.querySelectorAll("a")];
    const sections = links
      .map((link) => document.querySelector(link.getAttribute("href")))
      .filter(Boolean);
    if ("IntersectionObserver" in window) {
      const observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (!entry.isIntersecting) return;
            links.forEach((link) => {
              link.classList.toggle("is-active", link.getAttribute("href") === `#${entry.target.id}`);
            });
          });
        },
        { rootMargin: "-40% 0px -50% 0px", threshold: 0.01 }
      );
      sections.forEach((section) => observer.observe(section));
    }
  }

  function bindClassify() {
    const form = $("classify-form");
    const field = $("ticket-text");
    field.addEventListener("input", () => {
      $("ticket-count").textContent = `${field.value.length} characters`;
      clearError("classify-error");
    });
    field.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        form.requestSubmit();
      }
    });
    $("ticket-clear").addEventListener("click", () => {
      field.value = "";
      field.dispatchEvent(new Event("input"));
      showResultState("empty");
    });
    document.querySelectorAll(".sample-chip").forEach((chip) => {
      chip.addEventListener("click", () => fillTicket(SAMPLES[chip.dataset.sample] || "", chip));
    });
    $("ticket-channel").addEventListener("change", syncSubject);
    syncSubject();
    form.addEventListener("submit", onClassify);
    $("copy-result").addEventListener("click", copyResult);
    $("new-ticket").addEventListener("click", resetTicket);
  }

  function resetTicket() {
    const field = $("ticket-text");
    field.value = "";
    field.dispatchEvent(new Event("input"));
    showResultState("empty");
    clearError("classify-error");
    field.focus();
  }

  async function copyResult() {
    const button = $("copy-result");
    const summary = [
      `Ticket: ${$("result-ticket-id").textContent}`,
      `Category: ${$("result-category").textContent}`,
      `Secondary category: ${$("result-secondary").textContent}`,
      `Team: ${$("result-team").textContent}`,
      `Urgent: ${$("result-urgent").textContent}`,
      `Confidence: ${$("confidence-value").textContent}`,
      `Model: ${$("result-version").textContent}`,
    ].join("\n");
    try {
      await navigator.clipboard.writeText(summary);
      button.textContent = "Copied";
    } catch (_error) {
      button.textContent = "Copy unavailable";
    }
    window.setTimeout(() => {
      button.textContent = "Copy result";
    }, 1400);
  }

  function fillTicket(text, chip) {
    const field = $("ticket-text");
    field.value = text;
    field.dispatchEvent(new Event("input"));
    field.focus();
    chip.classList.add("is-pressed");
    field.classList.add("is-filled");
    window.setTimeout(() => {
      chip.classList.remove("is-pressed");
      field.classList.remove("is-filled");
    }, 280);
  }

  function bindBatch() {
    const field = $("batch-text");
    field.addEventListener("input", updateBatchCount);
    field.addEventListener("scroll", () => {
      $("batch-gutter").scrollTop = field.scrollTop;
    });
    $("batch-clear").addEventListener("click", () => {
      field.value = "";
      updateBatchCount();
      clearError("batch-error");
      $("batch-results").hidden = true;
      state.rows = [];
    });
    $("batch-example").addEventListener("click", () => {
      field.value = BATCH_SAMPLES.join("\n");
      field.dispatchEvent(new Event("input"));
      field.focus();
    });
    $("batch-form").addEventListener("submit", onBatch);
    document.querySelectorAll(".filters button").forEach((button) => {
      button.addEventListener("click", () => {
        state.bandFilter = button.dataset.filter;
        document.querySelectorAll(".filters button").forEach((item) => {
          item.classList.toggle("is-on", item === button);
        });
        renderBatchTable();
      });
    });
    document.querySelectorAll("#batch-table th button").forEach((button) => {
      button.addEventListener("click", () => {
        const key = button.dataset.sort;
        if (state.sortKey === key) state.sortDir *= -1;
        else {
          state.sortKey = key;
          state.sortDir = key === "ticket" || key === "category" || key === "team" ? 1 : -1;
        }
        renderBatchTable();
      });
    });
    updateBatchCount();
  }

  async function onClassify(event) {
    event.preventDefault();
    const text = $("ticket-text").value.trim();
    if (!text) {
      showError("classify-error", "Enter a support ticket before classifying.");
      $("ticket-text").focus();
      return;
    }
    const button = $("classify-button");
    setBusy(button, true, "Classifying");
    clearError("classify-error");
    showResultState("loading");
    const steps = runPipeline();
    try {
      const payload = await postJson("/ui/predict", ticketRequest(text));
      await steps;
      renderPrediction(payload);
    } catch (error) {
      await steps;
      showResultState("empty");
      showError("classify-error", error.message);
    } finally {
      setBusy(button, false, "Classify Ticket");
    }
  }

  async function onBatch(event) {
    event.preventDefault();
    const tickets = ticketLines($("batch-text").value);
    if (!tickets.length) {
      showError("batch-error", "Paste at least one ticket. Put each ticket on its own line.");
      return;
    }
    if (tickets.length > BATCH_LIMIT) {
      showError(
        "batch-error",
        `Synchronous batch classification accepts 1 to ${BATCH_LIMIT} tickets. You have ${tickets.length}.`
      );
      return;
    }
    const button = $("batch-button");
    setBusy(button, true, "Classifying");
    clearError("batch-error");
    try {
      const payload = await postJson("/ui/batch", {
        tickets,
        channel: $("batch-channel").value,
      });
      const predictions = payload && Array.isArray(payload.predictions) ? payload.predictions : null;
      if (!predictions) {
        throw new Error("The classifier returned a response the demo could not read.");
      }
      state.rows = predictions.map((row, index) => ({ ...row, index: index + 1 }));
      state.bandFilter = "all";
      document.querySelectorAll(".filters button").forEach((item) => {
        item.classList.toggle("is-on", item.dataset.filter === "all");
      });
      state.sortKey = "index";
      state.sortDir = 1;
      renderBatchTable();
      $("batch-results").hidden = false;
      $("batch-results").scrollIntoView({ block: "nearest" });
    } catch (error) {
      showError("batch-error", error.message);
    } finally {
      setBusy(button, false, "Classify Batch");
    }
  }

  function renderPrediction(payload) {
    if (!payload || typeof payload.category !== "string" || !payload.category) {
      showResultState("empty");
      showError("classify-error", "The classifier returned a response the demo could not read.");
      return;
    }
    showResultState("body");
    const panel = document.querySelector(".result-panel");
    panel.classList.remove("is-spring");
    void panel.offsetWidth;
    panel.classList.add("is-spring");
    $("result-category").textContent = formatLabel(payload.category);
    $("result-secondary").textContent = payload.secondary_category ? formatLabel(payload.secondary_category) : "None";
    $("result-ticket-id").textContent = payload.ticket_id || "Not returned";
    $("result-version").textContent = payload.model_version || "Not returned";
    $("result-urgent").textContent = payload.is_urgent === true ? "Urgent" : payload.is_urgent === false ? "Not urgent" : "Not returned";
    const teamNote = $("result-team-note");
    const routed = $("routed-badge");
    if (payload.team) {
      $("result-team").textContent = payload.team;
      teamNote.hidden = true;
      routed.hidden = false;
    } else {
      $("result-team").textContent = "Not provided";
      teamNote.hidden = false;
      teamNote.textContent = "The hosted API did not return a team.";
      routed.hidden = true;
    }
    paintGauge(payload.confidence);
  }

  function syncSubject() {
    $("subject-wrap").hidden = $("ticket-channel").value !== "email";
  }

  function ticketRequest(text) {
    const channel = $("ticket-channel").value;
    const body = { text, channel };
    if (channel === "email") {
      const subject = $("ticket-subject").value.trim();
      if (subject) body.subject = subject;
    }
    return body;
  }

  function formatLabel(value) {
    return String(value).replaceAll("_", " ");
  }

  function runPipeline() {
    const steps = [...document.querySelectorAll("#pipeline li")];
    steps.forEach((step) => step.classList.remove("is-active", "is-done"));
    let index = 0;
    return new Promise((resolve) => {
      const tick = () => {
        steps.forEach((step, stepIndex) => {
          step.classList.toggle("is-done", stepIndex < index);
          step.classList.toggle("is-active", stepIndex === index);
        });
        index += 1;
        if (index > steps.length) resolve();
        else window.setTimeout(tick, 150);
      };
      tick();
    });
  }

  function paintGauge(raw) {
    const fraction = asFraction(raw);
    const gauge = $("confidence-gauge");
    const circle = $("gauge-value");
    if (fraction === null) {
      gauge.dataset.band = "empty";
      circle.style.strokeDashoffset = String(GAUGE_LENGTH);
      $("confidence-value").textContent = "—";
      $("confidence-band").textContent = "Confidence was not returned";
      return;
    }
    const band = fraction >= 0.9 ? "high" : fraction >= 0.7 ? "moderate" : "low";
    const label = band === "high" ? "Strong confidence" : band === "moderate" ? "Moderate confidence" : "Lower confidence";
    gauge.dataset.band = band;
    circle.style.transition = "none";
    circle.style.strokeDashoffset = String(GAUGE_LENGTH);
    window.requestAnimationFrame(() => {
      circle.style.transition = "";
      circle.style.strokeDashoffset = String(GAUGE_LENGTH * (1 - fraction));
    });
    $("confidence-value").textContent = formatPercent(fraction);
    $("confidence-band").textContent = label;
  }

  function renderBatchTable() {
    const rows = visibleRows().sort(compareRows);
    const body = $("batch-body");
    body.replaceChildren();
    if (!rows.length) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 7;
      td.textContent = "No tickets in this confidence band.";
      tr.appendChild(td);
      body.appendChild(tr);
    } else {
      rows.forEach((row) => body.appendChild(buildRow(row)));
    }
    $("batch-summary").textContent = `${state.rows.length} ticket${state.rows.length === 1 ? "" : "s"} classified`;
    const average = averageConfidence(state.rows);
    $("batch-average").textContent = average === null ? "" : `Average confidence: ${formatPercent(average)}`;
    renderDistribution(state.rows);
  }

  function buildRow(row) {
    const tr = document.createElement("tr");
    tr.appendChild(cell(String(row.index)));
    tr.appendChild(ticketCell(row));
    tr.appendChild(badgeCell(row.category ? formatLabel(row.category) : "Not provided", "badge"));
    tr.appendChild(badgeCell(row.secondary_category ? formatLabel(row.secondary_category) : "None", "badge"));
    tr.appendChild(badgeCell(row.team || "Not provided", "badge badge-team"));
    tr.appendChild(badgeCell(row.is_urgent === true ? "Urgent" : row.is_urgent === false ? "Not urgent" : "Not returned", "badge"));
    tr.appendChild(confidenceCell(row.confidence));
    return tr;
  }

  function ticketCell(row) {
    const text = row.ticket || "";
    const td = document.createElement("td");
    td.className = "ticket-cell";
    const preview = document.createElement("div");
    preview.className = "ticket-preview";
    preview.textContent = text || "—";
    td.appendChild(preview);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "expand";
    button.textContent = "Response";
    const payload = document.createElement("pre");
    payload.className = "payload";
    payload.hidden = true;
    payload.textContent = JSON.stringify({
      ticket_id: row.ticket_id ?? null,
      ticket: text,
      category: row.category ?? null,
      secondary_category: row.secondary_category ?? null,
      team: row.team ?? null,
      is_urgent: row.is_urgent ?? null,
      confidence: row.confidence ?? null,
      model_version: row.model_version ?? null,
    }, null, 2);
    button.addEventListener("click", () => {
      const open = td.classList.toggle("is-open");
      payload.hidden = !open;
      button.textContent = open ? "Hide response" : "Response";
    });
    td.appendChild(button);
    td.appendChild(payload);
    return td;
  }

  function badgeCell(text, className) {
    const td = cell("");
    const badge = document.createElement("span");
    badge.className = className;
    badge.textContent = text;
    td.appendChild(badge);
    return td;
  }

  function confidenceCell(raw) {
    const td = cell("");
    const fraction = asFraction(raw);
    const badge = document.createElement("span");
    if (fraction === null) {
      badge.className = "badge";
      badge.textContent = "Not returned";
    } else {
      const band = fraction >= 0.9 ? "high" : fraction >= 0.7 ? "moderate" : "low";
      const name = band === "high" ? "High" : band === "moderate" ? "Moderate" : "Low";
      badge.className = `badge badge-${band}`;
      badge.textContent = `${formatPercent(fraction)} · ${name}`;
    }
    td.appendChild(badge);
    return td;
  }

  function cell(text) {
    const td = document.createElement("td");
    td.textContent = text;
    return td;
  }

  function compareRows(a, b) {
    const key = state.sortKey;
    const dir = state.sortDir;
    if (key === "confidence") {
      const left = asFraction(a.confidence);
      const right = asFraction(b.confidence);
      if (left === null && right === null) return a.index - b.index;
      if (left === null) return 1;
      if (right === null) return -1;
      return (left - right) * dir;
    }
    if (key === "index") return (a.index - b.index) * dir;
    const left = String(a[key] || "");
    const right = String(b[key] || "");
    return left.localeCompare(right) * dir;
  }

  function updateBatchCount() {
    syncGutter();
    const count = ticketLines($("batch-text").value).length;
    $("batch-count").textContent = `Tickets detected: ${count}`;
    const limit = $("batch-limit");
    if (count > BATCH_LIMIT) {
      limit.textContent = `Limit is ${BATCH_LIMIT} tickets`;
      limit.classList.add("is-over");
    } else {
      limit.textContent = count ? `Limit ${BATCH_LIMIT}` : "";
      limit.classList.remove("is-over");
    }
  }

  async function refreshStatus() {
    const pill = $("system-status");
    const label = $("system-status-label");
    try {
      const health = await getJson("/health");
      const online = health && health.status === "ok";
      const loading = health && health.status === "loading";
      const version = health && typeof health.model_version === "string" ? health.model_version : "";
      state.status = { online, ready: online, loading, detail: version ? `Model ${version}` : "" };
      pill.classList.remove("is-online", "is-warn", "is-offline");
      if (online) {
        pill.classList.add("is-online");
        label.textContent = "System Online";
      } else if (loading) {
        pill.classList.add("is-warn");
        label.textContent = "Model loading";
      } else {
        pill.classList.add("is-offline");
        label.textContent = "Service Unavailable";
      }
    } catch (error) {
      state.status = { online: false, ready: false, loading: false, detail: error.message };
      pill.classList.remove("is-online", "is-warn");
      pill.classList.add("is-offline");
      label.textContent = "System Offline";
    }
    renderOverview();
  }

  async function loadMetrics() {
    const root = $("analytics-body");
    root.replaceChildren(messagePanel("Loading evaluation metrics."));
    try {
      const response = await fetch("/static/metrics.json", { cache: "no-store" });
      if (!response.ok) throw new Error("missing");
      const data = await response.json();
      state.metrics = data && typeof data === "object" && !Array.isArray(data) ? data : {};
    } catch (_error) {
      state.metrics = null;
    }
    renderOverview();
    renderAnalytics();
  }

  function renderOverview() {
    const grid = $("overview-cards");
    grid.replaceChildren();
    const status = state.status;
    if (status) {
      grid.appendChild(metricCard(
        "System status",
        status.online ? "Online" : "Offline",
        status.online ? "Health check responded" : "Health check failed"
      ));
      grid.appendChild(metricCard(
        "Classifier",
        status.ready ? "Ready" : status.loading ? "Loading" : "Unavailable",
        status.detail || "Hosted classification API"
      ));
    }
    const metrics = state.metrics || {};
    const accuracy = asFraction(firstValue(metrics, ["accuracy"]));
    const macro = asFraction(firstValue(metrics, ["macro_f1", "macroF1", "macro_f1_score", "f1_macro"]));
    if (accuracy !== null) {
      grid.appendChild(metricCard("Model accuracy", formatPercent(accuracy), "Evaluation"));
    }
    if (macro !== null) {
      grid.appendChild(metricCard("Macro F1", formatPercent(macro), "Evaluation"));
    }
    const categories = categoryList(metrics);
    if (categories) {
      grid.appendChild(metricCard("Supported categories", String(categories.length), "From the evaluation file"));
    }
  }

  function renderAnalytics() {
    const root = $("analytics-body");
    root.replaceChildren();
    if (state.metrics === null) {
      root.appendChild(messagePanel("Evaluation metrics could not be loaded."));
      return;
    }
    const metrics = state.metrics;
    const accuracy = asFraction(firstValue(metrics, ["accuracy"]));
    const macro = asFraction(firstValue(metrics, ["macro_f1", "macroF1", "macro_f1_score", "f1_macro"]));
    const perClass = objectValue(metrics, ["per_class_f1", "perClassF1", "class_f1"]);
    const matrix = metrics.confusion_matrix || metrics.confusionMatrix;
    const labels = categoryList(metrics);
    const info = modelInfo(metrics);

    if (accuracy === null && macro === null && !perClass && !matrix && !info.length && !labels) {
      root.appendChild(messagePanel(
        "Evaluation metrics have not been published yet. Accuracy, per-class F1, and the confusion matrix will appear here when the model evaluation is added."
      ));
      return;
    }

    if (accuracy !== null || macro !== null || labels) {
      const grid = document.createElement("div");
      grid.className = "card-grid";
      if (accuracy !== null) grid.appendChild(metricCard("Model accuracy", formatPercent(accuracy), "Held-out evaluation"));
      if (macro !== null) grid.appendChild(metricCard("Macro F1", formatPercent(macro), "Held-out evaluation"));
      if (labels) grid.appendChild(metricCard("Categories", String(labels.length), labels.join(", ")));
      root.appendChild(grid);
    }

    if (perClass) {
      root.appendChild(f1Chart(perClass));
    }
    if (Array.isArray(matrix)) {
      root.appendChild(confusionMatrix(matrix, labels || matrixLabels(matrix)));
    }
    if (info.length) {
      root.appendChild(infoPanel(info));
    }
  }

  function f1Chart(scores) {
    const panel = document.createElement("section");
    panel.className = "panel";
    const title = document.createElement("h3");
    title.textContent = "Per-class F1";
    panel.appendChild(title);
    Object.keys(scores).forEach((name) => {
      const fraction = asFraction(scores[name]);
      if (fraction === null) return;
      const row = document.createElement("div");
      row.className = "chart-row";
      const label = document.createElement("div");
      label.className = "chart-label";
      const nameEl = document.createElement("span");
      nameEl.textContent = name;
      const valueEl = document.createElement("span");
      valueEl.textContent = formatPercent(fraction);
      label.append(nameEl, valueEl);
      const track = document.createElement("div");
      track.className = "bar-track";
      const fill = document.createElement("div");
      fill.className = "bar-fill";
      fill.style.width = "0%";
      window.requestAnimationFrame(() => {
        fill.style.width = `${Math.max(0, Math.min(100, fraction * 100))}%`;
      });
      track.appendChild(fill);
      row.append(label, track);
      panel.appendChild(row);
    });
    return panel;
  }

  function confusionMatrix(matrix, labels) {
    const panel = document.createElement("section");
    panel.className = "panel";
    const title = document.createElement("h3");
    title.textContent = "Confusion matrix";
    const note = document.createElement("p");
    note.className = "meta-note";
    note.textContent = "Rows are the actual class. Columns are the predicted class.";
    panel.append(title, note);

    const scroll = document.createElement("div");
    scroll.className = "matrix-scroll";
    const table = document.createElement("table");
    table.className = "matrix";
    const head = document.createElement("tr");
    head.appendChild(document.createElement("th"));
    const columnLabels = labels || [];
    columnLabels.forEach((label) => {
      const th = document.createElement("th");
      th.className = "axis";
      th.textContent = label;
      head.appendChild(th);
    });
    const thead = document.createElement("thead");
    thead.appendChild(head);
    table.appendChild(thead);

    let max = 0;
    matrix.forEach((row) => {
      if (!Array.isArray(row)) return;
      row.forEach((value) => {
        const number = Number(value);
        if (Number.isFinite(number)) max = Math.max(max, number);
      });
    });

    const tbody = document.createElement("tbody");
    matrix.forEach((row, rowIndex) => {
      if (!Array.isArray(row)) return;
      const tr = document.createElement("tr");
      const th = document.createElement("th");
      th.className = "axis";
      th.textContent = columnLabels[rowIndex] || `Class ${rowIndex + 1}`;
      tr.appendChild(th);
      const rowTotal = row.reduce((sum, value) => {
        const number = Number(value);
        return sum + (Number.isFinite(number) ? number : 0);
      }, 0);
      row.forEach((value, columnIndex) => {
        const number = Number(value);
        const td = document.createElement("td");
        const safe = Number.isFinite(number) ? number : 0;
        const t = max === 0 ? 0 : safe / max;
        td.className = "is-hot";
        td.tabIndex = 0;
        td.textContent = Number.isFinite(number) ? String(number) : "—";
        td.style.background = mixColor(t);
        td.style.color = t > 0.55 ? "#f8fafc" : "#dbe7ff";
        const actual = columnLabels[rowIndex] || `Class ${rowIndex + 1}`;
        const predicted = columnLabels[columnIndex] || `Class ${columnIndex + 1}`;
        const share = rowTotal === 0 ? 0 : (safe / rowTotal) * 100;
        const tip = `True label: ${actual}\nPredicted label: ${predicted}\nCount: ${Number.isFinite(number) ? number : "—"}\nRow share: ${share.toFixed(1)}%`;
        const show = (event) => showMatrixTip(tip, event.currentTarget);
        const hide = () => { $("matrix-tip").hidden = true; };
        td.addEventListener("mouseenter", show);
        td.addEventListener("focus", show);
        td.addEventListener("mouseleave", hide);
        td.addEventListener("blur", hide);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    scroll.appendChild(table);
    panel.appendChild(scroll);

    const legend = document.createElement("div");
    legend.className = "legend";
    const scale = document.createElement("span");
    scale.className = "legend-scale";
    const caption = document.createElement("span");
    caption.textContent = "Lower count to higher count";
    legend.append(scale, caption);
    panel.appendChild(legend);
    return panel;
  }

  function infoPanel(items) {
    const panel = document.createElement("section");
    panel.className = "panel";
    const title = document.createElement("h3");
    title.textContent = "Model information";
    const list = document.createElement("dl");
    list.className = "info-list";
    items.forEach((item) => {
      const wrap = document.createElement("div");
      const dt = document.createElement("dt");
      dt.textContent = item.label;
      const dd = document.createElement("dd");
      dd.textContent = item.value;
      wrap.append(dt, dd);
      list.appendChild(wrap);
    });
    panel.append(title, list);
    return panel;
  }

  function modelInfo(metrics) {
    const fields = [
      ["model_type", "Model type"],
      ["model", "Model type"],
      ["features", "Features"],
      ["training_approach", "Training approach"],
      ["evaluation_metric", "Evaluation metric"],
    ];
    const seen = new Set();
    const items = [];
    fields.forEach(([key, label]) => {
      const value = metrics[key];
      if (typeof value !== "string" || !value.trim() || seen.has(label)) return;
      seen.add(label);
      items.push({ label, value: value.trim() });
    });
    return items;
  }

  function messagePanel(text) {
    const panel = document.createElement("div");
    panel.className = "panel empty-state";
    const p = document.createElement("p");
    p.textContent = text;
    panel.appendChild(p);
    return panel;
  }

  function metricCard(label, value, note) {
    const card = document.createElement("article");
    card.className = "metric-card reveal";
    const labelEl = document.createElement("p");
    labelEl.className = "metric-label";
    labelEl.textContent = label;
    const valueEl = document.createElement("p");
    valueEl.className = "metric-value";
    valueEl.textContent = value;
    const noteEl = document.createElement("p");
    noteEl.className = "metric-note";
    noteEl.textContent = note;
    card.append(labelEl, valueEl, noteEl);
    return card;
  }

  function categoryList(metrics) {
    if (Array.isArray(metrics.categories) && metrics.categories.length) {
      return metrics.categories.map(String);
    }
    if (Array.isArray(metrics.confusion_labels) && metrics.confusion_labels.length) {
      return metrics.confusion_labels.map(String);
    }
    const perClass = objectValue(metrics, ["per_class_f1", "perClassF1", "class_f1"]);
    if (perClass) return Object.keys(perClass);
    return null;
  }

  function matrixLabels(matrix) {
    return matrix.map((_, index) => `Class ${index + 1}`);
  }

  function objectValue(source, keys) {
    for (const key of keys) {
      const value = source[key];
      if (value && typeof value === "object" && !Array.isArray(value)) return value;
    }
    return null;
  }

  function firstValue(source, keys) {
    for (const key of keys) {
      if (source[key] !== undefined && source[key] !== null && source[key] !== "") return source[key];
    }
    return null;
  }

  async function postJson(url, body) {
    let response;
    try {
      response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch (_error) {
      throw new Error("VORTEX couldn't connect to the classification service. Please try again.");
    }
    return readResponse(response);
  }

  async function getJson(url) {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) throw new Error("VORTEX couldn't connect to the classification service. Please try again.");
    return response.json();
  }

  async function readResponse(response) {
    let data = null;
    try {
      data = await response.json();
    } catch (_error) {
      data = null;
    }
    if (!response.ok) {
      throw new Error(friendlyError(response.status, data));
    }
    if (!data || typeof data !== "object") {
      throw new Error("The classifier returned a response the demo could not read.");
    }
    return data;
  }

  function friendlyError(status, data) {
    let message = "";
    if (data && typeof data.error === "string") message = data.error;
    else if (data && data.error && typeof data.error.message === "string") message = data.error.message;
    if (message && !/traceback|api_key|exception/i.test(message)) return message;
    if (status === 400 || status === 422) return "The request was not valid.";
    if (status === 401 || status === 403) return "The classifier refused the request.";
    return "VORTEX couldn't connect to the classification service. Please try again.";
  }

  function visibleRows() {
    return state.rows.filter((row) => {
      if (state.bandFilter === "all") return true;
      const fraction = asFraction(row.confidence);
      if (fraction === null) return false;
      if (state.bandFilter === "high") return fraction >= 0.9;
      if (state.bandFilter === "moderate") return fraction >= 0.7 && fraction < 0.9;
      return fraction < 0.7;
    });
  }

  function bindPointer() {
    const fine = window.matchMedia("(hover: hover) and (pointer: fine)").matches;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!fine || reduced) return;
    document.body.classList.add("has-pointer");
    const glow = $("cursor-glow");
    const heroVisual = document.querySelector(".hero-visual");
    let frame = 0;
    let pointerX = 0;
    let pointerY = 0;
    window.addEventListener("pointermove", (event) => {
      pointerX = event.clientX;
      pointerY = event.clientY;
      if (frame) return;
      frame = window.requestAnimationFrame(() => {
        frame = 0;
        glow.style.transform = `translate3d(${pointerX}px, ${pointerY}px, 0)`;
        if (!heroVisual) return;
        const rect = heroVisual.getBoundingClientRect();
        const dx = Math.max(-8, Math.min(8, (pointerX - (rect.left + rect.width / 2)) / 45));
        const dy = Math.max(-6, Math.min(6, (pointerY - (rect.top + rect.height / 2)) / 45));
        heroVisual.style.transform = `translate(${dx}px, ${dy}px)`;
      });
    });
  }

  function bindReveal() {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const sections = [...document.querySelectorAll(".reveal-on-scroll")];
    if (reduced || !("IntersectionObserver" in window)) {
      sections.forEach((section) => section.classList.add("is-in"));
      return;
    }
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-in");
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12 });
    sections.forEach((section) => observer.observe(section));
  }

  function ticketLines(value) {
    return value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  }

  function asFraction(value) {
    if (value === null || value === undefined || value === "") return null;
    const number = Number(value);
    if (!Number.isFinite(number)) return null;
    if (number > 1 && number <= 100) return number / 100;
    if (number >= 0 && number <= 1) return number;
    return null;
  }

  function formatPercent(fraction) {
    return `${(fraction * 100).toFixed(1)}%`;
  }

  function averageConfidence(rows) {
    const values = rows.map((row) => asFraction(row.confidence)).filter((value) => value !== null);
    if (!values.length) return null;
    return values.reduce((sum, value) => sum + value, 0) / values.length;
  }

  function mixColor(t) {
    const from = [17, 24, 45];
    const to = [139, 92, 246];
    const channel = (index) => Math.round(from[index] + (to[index] - from[index]) * t);
    return `rgb(${channel(0)}, ${channel(1)}, ${channel(2)})`;
  }

  function showMatrixTip(text, anchor) {
    const tip = $("matrix-tip");
    tip.replaceChildren();
    text.split("\n").forEach((line, index) => {
      const row = document.createElement(index === 0 ? "strong" : "div");
      row.textContent = line;
      tip.appendChild(row);
    });
    const rect = anchor.getBoundingClientRect();
    tip.hidden = false;
    const width = tip.offsetWidth;
    const left = Math.min(window.innerWidth - width - 8, Math.max(8, rect.left));
    tip.style.left = `${left}px`;
    tip.style.top = `${Math.max(8, rect.top - tip.offsetHeight - 8)}px`;
  }

  function renderDistribution(rows) {
    const host = $("batch-distribution");
    const counts = { high: 0, moderate: 0, low: 0 };
    rows.forEach((row) => {
      const fraction = asFraction(row.confidence);
      if (fraction === null) return;
      if (fraction >= 0.9) counts.high += 1;
      else if (fraction >= 0.7) counts.moderate += 1;
      else counts.low += 1;
    });
    const total = counts.high + counts.moderate + counts.low;
    host.replaceChildren();
    if (!total) {
      host.hidden = true;
      return;
    }
    host.hidden = false;
    const bar = document.createElement("div");
    bar.className = "dist-bar";
    bar.append(segment("dist-high", counts.high, total), segment("dist-mid", counts.moderate, total), segment("dist-low", counts.low, total));
    const legend = document.createElement("div");
    legend.className = "dist-legend";
    legend.append(
      legendItem(`${counts.high} high`),
      legendItem(`${counts.moderate} moderate`),
      legendItem(`${counts.low} lower`),
    );
    host.append(bar, legend);
  }

  function segment(className, count, total) {
    const span = document.createElement("span");
    span.className = className;
    span.style.width = `${(count / total) * 100}%`;
    return span;
  }

  function legendItem(text) {
    const span = document.createElement("span");
    span.textContent = text;
    return span;
  }

  function syncGutter() {
    const field = $("batch-text");
    const gutter = $("batch-gutter");
    const lines = Math.max(field.value.split(/\r?\n/).length, 1);
    gutter.textContent = Array.from({ length: lines }, (_, index) => String(index + 1)).join("\n");
  }

  function bindMotion() {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    state.motion = !reduced;
    const toggle = $("motion-toggle");
    const canvas = $("vortex-canvas");
    const video = $("ambient-video");
    const context = canvas.getContext("2d");
    const particles = createParticles(48);

    function resize() {
      const bounds = canvas.parentElement.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(bounds.width));
      canvas.height = Math.max(1, Math.floor(bounds.height));
    }

    function draw(now) {
      state.canvasFrame = window.requestAnimationFrame(draw);
      if (!state.motion) return;
      const width = canvas.width;
      const height = canvas.height;
      context.clearRect(0, 0, width, height);
      const cx = width * 0.72;
      const cy = height * 0.42;
      particles.forEach((particle, index) => {
        particle.angle += particle.speed;
        const radius = particle.radius + Math.sin(now / 900 + index) * 8;
        const x = cx + Math.cos(particle.angle) * radius;
        const y = cy + Math.sin(particle.angle) * radius * 0.42;
        particle.x = x;
        particle.y = y;
      });
      context.lineWidth = 1;
      for (let i = 0; i < particles.length; i += 1) {
        for (let step = 1; step <= 4; step += 1) {
          const other = particles[(i + step) % particles.length];
          const dx = particles[i].x - other.x;
          const dy = particles[i].y - other.y;
          const distance = Math.hypot(dx, dy);
          if (distance > 90) continue;
          context.strokeStyle = `rgba(167, 139, 250, ${0.28 * (1 - distance / 90)})`;
          context.beginPath();
          context.moveTo(particles[i].x, particles[i].y);
          context.lineTo(other.x, other.y);
          context.stroke();
        }
      }
      particles.forEach((particle) => {
        context.fillStyle = particle.violet ? "#A78BFA" : "#22D3EE";
        context.beginPath();
        context.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
        context.fill();
      });
    }

    let canvasStarted = false;

    function startCanvas() {
      video.hidden = true;
      canvas.hidden = false;
      if (canvasStarted) return;
      canvasStarted = true;
      resize();
      window.addEventListener("resize", resize);
      draw(0);
    }

    function applyMotion() {
      document.body.classList.toggle("motion-off", !state.motion);
      toggle.setAttribute("aria-pressed", state.motion ? "true" : "false");
      toggle.textContent = state.motion ? "Motion on" : "Motion off";
      video.muted = true;
      video.volume = 0;
      if (video.hidden) return;
      if (state.motion) video.play().catch(startCanvas);
      else video.pause();
    }

    video.muted = true;
    video.volume = 0;
    video.addEventListener("error", startCanvas);
    applyMotion();
    toggle.addEventListener("click", () => {
      state.motion = !state.motion;
      applyMotion();
    });
  }

  function createParticles(count) {
    return Array.from({ length: count }, (_, index) => ({
      angle: (index / count) * Math.PI * 2,
      speed: 0.002 + (index % 5) * 0.0006,
      radius: 40 + (index % 9) * 18,
      size: index % 7 === 0 ? 2.2 : 1.3,
      violet: index % 3 === 0,
      x: 0,
      y: 0,
    }));
  }

  function showResultState(mode) {
    $("result-empty").hidden = mode !== "empty";
    $("result-loading").hidden = mode !== "loading";
    $("result-body").hidden = mode !== "body";
  }

  function showError(id, message) {
    const el = $(id);
    el.hidden = false;
    el.textContent = message;
  }

  function clearError(id) {
    const el = $(id);
    el.hidden = true;
    el.textContent = "";
  }

  function setBusy(button, busy, label) {
    button.disabled = busy;
    button.classList.toggle("is-loading", busy);
    button.replaceChildren();
    if (busy) {
      const spinner = document.createElement("span");
      spinner.className = "spinner";
      spinner.setAttribute("aria-hidden", "true");
      button.appendChild(spinner);
    }
    button.appendChild(document.createTextNode(label));
    button.setAttribute("aria-busy", busy ? "true" : "false");
  }
})();
