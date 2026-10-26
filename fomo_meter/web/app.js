/* FOMO Meter dashboard — vanilla JS, no external dependencies. */
(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const REFRESH_MS = 120_000;
  const STATE = {
    ticker: "BTCUSD",
    status: null,
    scan: null,
    history: [],
    loading: false,
    briefLoading: false,
    analyzeLoading: false,
    useModel: true,
  };

  const SOURCE_ICONS = {
    stocktwits: "icon-st",
    news: "icon-news",
    yahoo: "icon-wire",
    reddit: "icon-forum",
    x: "icon-x",
  };
  const STATUS_TEXT = {
    LIVE: "LIVE",
    PARTIAL: "PARTIAL",
    STALE: "STALE",
    NO_DATA: "NO DATA",
    NEEDS_AUTH: "BUTUH TOKEN",
    ERROR: "ERROR",
  };
  const GATE_ICONS = {
    PASS: "icon-check",
    WARN: "icon-warn",
    FAIL: "icon-fail",
    SKIP: "icon-skip",
  };
  // The desktop shell appends #desktop to the local URL. Browser mode keeps the
  // native browser chrome and never shows fake window controls.
  const IS_DESKTOP = window.location.hash === "#desktop";
  const WINDOW_PREFIX = "FOMO_WINDOW|";
  let fpsMonitorStarted = false;

  function startFpsMonitor() {
    if (fpsMonitorStarted) return;
    fpsMonitorStarted = true;
    const meter = $("#fpsMeter");
    const value = $("#fpsValue");
    if (!meter || !value) return;

    let frames = 0;
    let last = performance.now();
    const tick = (now) => {
      frames += 1;
      const elapsed = now - last;
      if (elapsed >= 500) {
        const fps = Math.max(0, Math.round((frames * 1000) / elapsed));
        value.textContent = String(fps);
        meter.classList.toggle("warn", fps < 50 && fps >= 30);
        meter.classList.toggle("bad", fps < 30);
        frames = 0;
        last = now;
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) {
        frames = 0;
        last = performance.now();
      }
    });
  }

  function sendWindowCommand(action) {
    if (!IS_DESKTOP) return;
    console.log(`${WINDOW_PREFIX}${action}`);
  }

  function setMaximized(maximized) {
    const button = $("#winMax");
    const use = button?.querySelector("use");
    if (use) {
      use.setAttribute(
        "href",
        `/assets/sprites.svg#${maximized ? "icon-restore" : "icon-maximize"}`,
      );
    }
    if (button) {
      const label = maximized ? "Pulihkan jendela" : "Perbesar jendela";
      button.title = label;
      button.setAttribute("aria-label", label);
    }
  }

  window.fomoSetMaximized = setMaximized;

  function setupDesktopShell() {
    if (!IS_DESKTOP) return;
    document.body.classList.add("desktop-shell");
    const controls = $("#windowControls");
    if (controls) controls.hidden = false;

    $("#winMin")?.addEventListener("click", () => sendWindowCommand("minimize"));
    $("#winMax")?.addEventListener("click", () => sendWindowCommand("toggle_max"));
    $("#winClose")?.addEventListener("click", () => sendWindowCommand("close"));

    const topbar = $("#appTopbar");
    if (!topbar) return;
    const drag = {
      pending: false,
      active: false,
      pointerId: null,
      startX: 0,
      startY: 0,
      lastSent: 0,
      frame: 0,
    };

    const isInteractive = (target) =>
      target instanceof Element &&
      Boolean(target.closest("button, a, input, textarea, select, option, label, .ticker-tabs, .window-controls"));

    const finishDrag = () => {
      if (!drag.pending && !drag.active) return;
      drag.pending = false;
      drag.active = false;
      document.body.classList.remove("window-dragging");
      if (drag.frame) cancelAnimationFrame(drag.frame);
      drag.frame = 0;
      if (drag.pointerId !== null) {
        try {
          topbar.releasePointerCapture(drag.pointerId);
        } catch {
          /* pointer capture may already be released by the OS */
        }
      }
      drag.pointerId = null;
      sendWindowCommand("drag_end");
    };

    topbar.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || isInteractive(event.target)) return;
      event.preventDefault();
      drag.pending = true;
      drag.active = false;
      drag.pointerId = event.pointerId;
      drag.startX = event.screenX;
      drag.startY = event.screenY;
      drag.lastSent = 0;
      topbar.setPointerCapture?.(event.pointerId);
    });

    topbar.addEventListener("pointermove", (event) => {
      if (!drag.pending && !drag.active) return;
      if (drag.pending) {
        const distance = Math.hypot(
          event.screenX - drag.startX,
          event.screenY - drag.startY,
        );
        if (distance < 5) return;
        drag.pending = false;
        drag.active = true;
        document.body.classList.add("window-dragging");
        sendWindowCommand("drag_start");
      }
      if (drag.frame) return;
      drag.frame = requestAnimationFrame(() => {
        drag.frame = 0;
        const now = performance.now();
        if (now - drag.lastSent < 24) return;
        drag.lastSent = now;
        sendWindowCommand("drag_move");
      });
    });

    topbar.addEventListener("pointerup", finishDrag);
    topbar.addEventListener("pointercancel", finishDrag);
    topbar.addEventListener("lostpointercapture", finishDrag);
    window.addEventListener("blur", finishDrag);
    topbar.addEventListener("dblclick", (event) => {
      if (isInteractive(event.target)) return;
      sendWindowCommand("toggle_max");
    });
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function safeUrl(value) {
    if (!value) return null;
    try {
      const parsed = new URL(String(value), window.location.origin);
      if (!["http:", "https:"].includes(parsed.protocol)) return null;
      return parsed.href;
    } catch {
      return null;
    }
  }

  function icon(id, className = "") {
    return `<svg class="${className}" aria-hidden="true"><use href="/assets/sprites.svg#${id}"></use></svg>`;
  }

  function statusClass(value) {
    const status = String(value || "").toUpperCase();
    if (status === "LIVE") return "live";
    if (status === "PARTIAL" || status === "STALE") return "partial";
    if (status === "ERROR" || status === "FAIL") return "error";
    if (status === "NEEDS_AUTH" || status === "WARN") return "warn";
    return "wait";
  }

  function stanceClass(value) {
    const stance = String(value || "").toUpperCase();
    if (stance === "BUY") return "buy";
    if (stance === "SELL") return "sell";
    return "wait";
  }

  function stanceText(value) {
    const stance = String(value || "WAIT").toUpperCase();
    if (stance === "CONFLICT") return "CONFLICT";
    if (stance === "NO_DATA") return "NO DATA";
    return stance;
  }

  function formatNumber(value, digits = 0) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
    return new Intl.NumberFormat("en-US", {
      maximumFractionDigits: digits,
      minimumFractionDigits: 0,
    }).format(Number(value));
  }

  function formatMoney(value) {
    if (value === null || value === undefined) return "—";
    const number = Number(value);
    const digits = Math.abs(number) < 10 ? 4 : (Math.abs(number) < 100 ? 2 : 0);
    return `$${formatNumber(number, digits)}`;
  }

  function formatPct(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
    const number = Number(value);
    return `${number >= 0 ? "+" : ""}${number.toFixed(2)}%`;
  }

  function formatTime(timestamp) {
    if (!timestamp) return "—";
    return new Date(Number(timestamp) * 1000).toLocaleTimeString("id-ID", {
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    });
  }

  function timeAgo(timestamp) {
    if (!timestamp) return "—";
    const seconds = Math.max(0, Math.round(Date.now() / 1000 - Number(timestamp)));
    if (seconds < 15) return "baru saja";
    if (seconds < 60) return `${seconds} detik lalu`;
    const minutes = Math.round(seconds / 60);
    if (minutes < 60) return `${minutes} menit lalu`;
    const hours = Math.round(minutes / 60);
    if (hours < 24) return `${hours} jam lalu`;
    return `${Math.round(hours / 24)} hari lalu`;
  }

  async function api(path, options = {}) {
    const response = await fetch(path, {
      ...options,
      headers: {
        Accept: "application/json",
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...(options.headers || {}),
      },
    });
    const raw = await response.text();
    let data = {};
    try {
      data = raw ? JSON.parse(raw) : {};
    } catch {
      throw new Error("Respons server bukan JSON. Periksa log aplikasi.");
    }
    if (!response.ok) {
      const error = new Error(data.error || `HTTP ${response.status}`);
      error.kind = data.kind;
      error.status = response.status;
      throw error;
    }
    return data;
  }

  function showToast(message, type = "") {
    const stack = $("#toastStack");
    if (!stack) return;
    const toast = document.createElement("div");
    toast.className = `toast ${type}`.trim();
    toast.textContent = message;
    stack.appendChild(toast);
    window.setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(8px)";
      window.setTimeout(() => toast.remove(), 220);
    }, 4200);
  }

  function setBusy(button, busy, idleLabel, busyLabel) {
    if (!button) return;
    button.disabled = busy;
    button.classList.toggle("loading", busy);
    const textNode = Array.from(button.childNodes).find(
      (node) => node.nodeType === Node.TEXT_NODE && node.nodeValue.trim()
    );
    if (textNode) {
      textNode.nodeValue = busy ? ` ${busyLabel}` : ` ${idleLabel}`;
    }
  }

  /* ----------------------------------------------------------------- tickers */
  function renderTickers(tickers) {
    const nav = $("#tickerTabs");
    if (!nav) return;
    nav.innerHTML = tickers.map((ticker) => `
      <button class="ticker-tab ${ticker.id === STATE.ticker ? "active" : ""}" data-ticker="${escapeHtml(ticker.id)}">
        ${escapeHtml(ticker.label)}<small>${escapeHtml(ticker.short)}</small>
      </button>
    `).join("");
    $$(".ticker-tab", nav).forEach((button) => {
      button.addEventListener("click", () => {
        const next = button.dataset.ticker;
        if (!next || next === STATE.ticker) return;
        STATE.ticker = next;
        STATE.brief = null;
        $$(".ticker-tab", nav).forEach((item) => item.classList.toggle("active", item === button));
        $("#briefResult").hidden = true;
        loadScan(false);
      });
    });
  }

  /* -------------------------------------------------------------------- scan */
  function setLoadingState() {
    const pill = $("#feedPill");
    if (pill) {
      pill.textContent = "MEMUAT";
      pill.className = "status-pill wait";
    }
    const refresh = $("#refreshBtn");
    if (refresh) refresh.classList.add("spinning");
  }

  function clearLoadingState() {
    const refresh = $("#refreshBtn");
    if (refresh) refresh.classList.remove("spinning");
  }

  async function loadScan(force = false) {
    if (STATE.loading) return;
    STATE.loading = true;
    setLoadingState();
    try {
      const scan = await api(`/api/scan?ticker=${encodeURIComponent(STATE.ticker)}&force=${force ? "1" : "0"}`);
      STATE.scan = scan;
      renderScan(scan);
    } catch (error) {
      const pill = $("#feedPill");
      if (pill) {
        pill.textContent = "ERROR";
        pill.className = "status-pill error";
      }
      showToast(error.message || "Gagal memuat scan.", "error");
      renderScanError(error.message || "Gagal memuat scan.");
    } finally {
      STATE.loading = false;
      clearLoadingState();
    }
  }

  function renderScanError(message) {
    const grid = $("#sourcesGrid");
    if (grid) {
      grid.innerHTML = `<div class="source-card state-error">
        <div class="source-head"><div class="source-icon">${icon("icon-warn")}</div>
        <div class="source-title"><strong>Tidak ada payload</strong><small>SCAN GAGAL</small></div></div>
        <p class="source-reason">${escapeHtml(message)}. Tidak ada data pengganti yang dibuat.</p>
      </div>`;
    }
    const gates = $("#gatesList");
    if (gates) gates.innerHTML = `<div class="gate-item fail"><span class="gate-icon">${icon("icon-fail")}</span><span class="gate-copy"><strong>Gate tidak dapat dijalankan</strong><span>Periksa koneksi atau log aplikasi.</span></span></div>`;
  }

  function updateRadarNodes(sources) {
    const map = new Map((sources || []).map((source) => [source.id, source]));
    $$("[data-source]").forEach((node) => {
      const source = map.get(node.dataset.source);
      node.classList.remove("live", "error", "auth", "no-data");
      if (!source) {
        node.classList.add("no-data");
        node.title = "Sumber belum tersedia";
        return;
      }
      const status = String(source.status || "").toUpperCase();
      if (status === "LIVE") node.classList.add("live");
      else if (status === "ERROR") node.classList.add("error");
      else if (status === "NEEDS_AUTH") node.classList.add("auth");
      else node.classList.add("no-data");
      node.title = `${source.name || node.dataset.source}: ${STATUS_TEXT[status] || status}`;
    });
  }

  function renderGauge(scan) {
    const stance = stanceText(scan.consensus);
    const intensity = Math.max(0, Math.min(100, Number(scan.intensity) || 0));
    const fill = $("#gaugeFill");
    const needle = $("#gaugeNeedle");
    if (fill) fill.style.strokeDashoffset = "0";
    if (needle) {
      let angle = 0;
      if (stance === "BUY") angle = ((intensity - 50) / 50) * 90;
      if (stance === "SELL") angle = -((intensity - 50) / 50) * 90;
      needle.style.transform = `rotate(${angle}deg)`;
    }
    const stanceEl = $("#heroStance");
    if (stanceEl) {
      stanceEl.textContent = stance;
      stanceEl.className = stance === "BUY" ? "buy-text" : (stance === "SELL" ? "sell-text" : "wait-text");
    }
    const intEl = $("#heroIntensity");
    if (intEl) intEl.textContent = `intensitas ${intensity}`;
    const radarStance = $("#radarStance");
    if (radarStance) {
      radarStance.textContent = stance;
      radarStance.className = stance === "BUY" ? "buy-text" : (stance === "SELL" ? "sell-text" : "wait-text");
    }
    const radarConf = $("#radarConfidence");
    if (radarConf) radarConf.textContent = `${Math.round((Number(scan.confidence) || 0) * 100)}% keyakinan`;
  }

  function renderScan(scan) {
    const label = scan.label || scan.ticker || "—";
    $("#heroTitle").textContent = label;
    const pill = $("#feedPill");
    if (pill) {
      const status = String(scan.feed_status || "NO_DATA").toUpperCase();
      pill.textContent = STATUS_TEXT[status] || status;
      pill.className = `status-pill ${statusClass(status)}`;
    }
    $("#lastUpdated").textContent = scan.cached
      ? `cache · ${timeAgo(scan.timestamp)}`
      : `diperbarui ${timeAgo(scan.timestamp)}`;
    $("#conflictBanner").hidden = !scan.conflict_detected;

    renderGauge(scan);
    $("#metricAnalyzed").textContent = formatNumber(scan.analyzed_posts);
    $("#metricLive").textContent = formatNumber(scan.live_sources);
    $("#metricTotal").textContent = formatNumber(scan.total_posts);
    $("#metricLatency").textContent = `${formatNumber(scan.latency_ms)} ms`;
    $("#heroConfidence").textContent = `${Math.round((Number(scan.confidence) || 0) * 100)}%`;
    $("#confidenceBar").style.width = `${Math.max(0, Math.min(100, (Number(scan.confidence) || 0) * 100))}%`;
    const analyzed = Number(scan.analyzed_posts) || 0;
    const live = Number(scan.live_sources) || 0;
    let note = "Konsensus hanya dibuat dari sumber live yang lolos gate.";
    if (scan.conflict_detected) note = "BUY dan SELL bertemu. Arah tidak dipaksa; radar menandai konflik.";
    else if (!live) note = "Tidak ada sumber live. Tidak ada konsensus pengganti.";
    else if (analyzed < 5) note = "Sampel masih kecil; baca hasil sebagai indikasi awal, bukan kepastian.";
    $("#verdictNote").textContent = note;
    $("#radarCaption").textContent = live
      ? `${analyzed} post dianalisis dari ${live} sumber live · ${scan.latency_ms} ms`
      : "Tidak ada sumber live pada scan ini.";

    updateRadarNodes(scan.sources || []);
    renderSources(scan.sources || []);
    renderGates(scan.gates || []);
    renderMarket(scan.market);
    if (STATE.status?.slm) renderModelStatus(STATE.status.slm);
  }

  function renderSources(sources) {
    const grid = $("#sourcesGrid");
    if (!grid) return;
    if (!sources.length) {
      grid.innerHTML = `<div class="source-card state-error"><div class="source-head"><div class="source-icon">${icon("icon-warn")}</div><div class="source-title"><strong>Tidak ada sumber</strong><small>NO DATA</small></div></div><p class="source-reason">Tidak ada adapter sumber yang mengembalikan payload.</p></div>`;
      return;
    }
    grid.innerHTML = sources.map((source) => {
      const status = String(source.status || "NO_DATA").toUpperCase();
      const stance = stanceText(source.stance);
      const iconId = SOURCE_ICONS[source.id] || "icon-pulse";
      const stateClass = status === "ERROR" ? "state-error" : (status === "NEEDS_AUTH" ? "state-auth" : "");
      const evidence = source.evidence
        ? `<p class="source-evidence">“${escapeHtml(source.evidence)}”</p>`
        : "";
      const reason = source.reason
        ? `<p class="source-reason">${escapeHtml(source.reason)}</p>`
        : "";
      const keywords = (source.keywords || []).length
        ? `<div class="tag-row">${source.keywords.slice(0, 5).map((term) => `<span class="tag">${escapeHtml(term)}</span>`).join("")}</div>`
        : "";
      const url = safeUrl(source.evidence_url);
      const link = url
        ? `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${icon("icon-external")} bukti</a>`
        : "";
      const provenance = safeUrl(source.provenance);
      const provenanceLink = provenance
        ? `<a href="${escapeHtml(provenance)}" target="_blank" rel="noopener noreferrer">${icon("icon-external")} sumber</a>`
        : "";
      return `<article class="source-card ${stateClass}">
        <div class="source-head">
          <div class="source-icon">${icon(iconId)}</div>
          <div class="source-title">
            <strong>${escapeHtml(source.name)}</strong>
            <small>${escapeHtml(source.kind || "sumber")}</small>
          </div>
          <span class="status-pill ${statusClass(status)} source-status">${STATUS_TEXT[status] || status}</span>
        </div>
        <div class="source-body">
          <span class="source-stance ${stanceClass(stance)}">${escapeHtml(stance)}</span>
          <span class="source-score"><strong>${Math.round((Number(source.confidence) || 0) * 100)}%</strong>confidence</span>
        </div>
        <div class="source-stats">
          <span><b>${formatNumber(source.analyzed_count)}</b> dianalisis</span>
          <span><b>${formatNumber(source.sample_count)}</b> sampel</span>
          <span><b>${formatNumber(source.bullish_count)}</b> bull</span>
          <span><b>${formatNumber(source.bearish_count)}</b> bear</span>
        </div>
        ${evidence}
        ${keywords}
        ${reason}
        <div class="source-foot">
          <span>${source.latest_post_at ? `post ${timeAgo(source.latest_post_at)}` : "tanpa timestamp"}</span>
          <span>${link} ${provenanceLink}</span>
        </div>
      </article>`;
    }).join("");
  }

  function renderGates(gates) {
    const list = $("#gatesList");
    if (!list) return;
    if (!gates.length) {
      list.innerHTML = `<div class="gate-item skip"><span class="gate-icon">${icon("icon-skip")}</span><span class="gate-copy"><strong>Belum ada gate</strong><span>Scan belum dijalankan.</span></span></div>`;
      $("#gateSummary").textContent = "—";
      return;
    }
    const counts = { PASS: 0, WARN: 0, FAIL: 0, SKIP: 0 };
    list.innerHTML = gates.map((gate) => {
      const state = String(gate.state || "SKIP").toUpperCase();
      counts[state] = (counts[state] || 0) + 1;
      return `<div class="gate-item ${state.toLowerCase()}" title="${escapeHtml(gate.detail || "")}">
        <span class="gate-icon">${icon(GATE_ICONS[state] || "icon-skip")}</span>
        <span class="gate-copy">
          <strong>${escapeHtml(gate.id)} · ${escapeHtml(gate.name)}</strong>
          <span>${escapeHtml(gate.detail || "")}</span>
        </span>
      </div>`;
    }).join("");
    $("#gateSummary").textContent = `${counts.PASS} pass · ${counts.WARN} warn · ${counts.FAIL} fail`;
  }

  function renderMarket(market) {
    const price = $("#marketPrice");
    const change = $("#marketChange");
    const source = $("#marketSource");
    const updated = $("#marketUpdated");
    const fgBlock = $("#fearGreedBlock");
    const fgValue = $("#fgValue");
    const fgBar = $("#fgBar");
    const fgLabel = $("#fgLabel");
    const error = $("#marketError");
    if (!market) {
      price.textContent = "—";
      change.textContent = "—";
      change.className = "change-chip neutral";
      source.textContent = "Konteks pasar tidak tersedia";
      updated.textContent = "—";
      fgBlock.hidden = true;
      error.hidden = true;
      return;
    }
    price.textContent = formatMoney(market.price);
    const delta = Number(market.change_pct);
    const deltaClass = Number.isFinite(delta) ? (delta > 0 ? "up" : (delta < 0 ? "down" : "neutral")) : "neutral";
    change.className = `change-chip ${deltaClass}`;
    change.innerHTML = Number.isFinite(delta)
      ? `${icon(delta >= 0 ? "icon-arrow-up" : "icon-arrow-down")} ${escapeHtml(formatPct(delta))}`
      : "—";
    source.textContent = market.source ? `Sumber: ${market.source}` : "Sumber pasar tidak tersedia";
    updated.textContent = market.fetched_at ? timeAgo(market.fetched_at) : "—";
    if (market.fear_greed !== null && market.fear_greed !== undefined) {
      const value = Math.max(0, Math.min(100, Number(market.fear_greed)));
      fgBlock.hidden = false;
      fgValue.textContent = `${value} · ${market.fear_greed_label || ""}`.trim();
      fgBar.style.width = `${value}%`;
      fgLabel.textContent = "Hanya untuk konteks; tidak memengaruhi sentimen.";
    } else {
      fgBlock.hidden = true;
    }
    if (market.error) {
      error.hidden = false;
      error.textContent = market.error;
    } else {
      error.hidden = true;
    }
  }

  function renderModelStatus(slm) {
    const available = Boolean(slm?.available);
    const badge = $("#modelBadge");
    const badgeText = $("#modelBadgeText");
    const status = $("#slmStatus");
    const accuracy = $("#slmAccuracy");
    const samples = $("#slmSamples");
    const arch = $("#slmArch");
    const button = $("#briefBtn");
    if (badge && badgeText) {
      badge.classList.toggle("online", available);
      badge.classList.toggle("offline", !available);
      badgeText.textContent = available ? "Sentiment SLM · lokal" : "Sentiment SLM · offline";
    }
    if (status) {
      status.textContent = available
        ? "Aktif · bobot lokal dimuat · tanpa API eksternal."
        : (slm?.error || "Model belum tersedia. Jalankan tools/train_slm.py.");
    }
    if (accuracy) accuracy.textContent = available ? `${Math.round((Number(slm.accuracy) || 0) * 1000) / 10}%` : "—";
    if (samples) samples.textContent = available ? `${formatNumber(slm.samples)} contoh` : "—";
    if (arch) arch.textContent = slm?.architecture || "—";
    if (button) button.disabled = !available || STATE.briefLoading;
  }

  /* -------------------------------------------------------------------- brief */
  async function requestBrief() {
    if (STATE.briefLoading) return;
    const button = $("#briefBtn");
    const result = $("#briefResult");
    STATE.briefLoading = true;
    setBusy(button, true, "Buat Brief", "Menyusun brief…");
    result.hidden = true;
    try {
      const brief = await api("/api/brief", {
        method: "POST",
        body: JSON.stringify({ ticker: STATE.ticker }),
      });
      renderBrief(brief);
      loadHistory();
    } catch (error) {
      showToast(error.message || "Brief Sentiment SLM gagal dibuat.", error.kind === "NO_LIVE_DATA" ? "warn" : "error");
      result.hidden = false;
      result.innerHTML = `<p class="error-note">${escapeHtml(error.message || "Brief Sentiment SLM gagal dibuat.")}</p>`;
    } finally {
      STATE.briefLoading = false;
      setBusy(button, false, "Buat Brief", "Menyusun brief…");
    }
  }

  function renderBrief(brief) {
    const result = $("#briefResult");
    if (!result) return;
    const stance = stanceText(brief.stance);
    const probabilities = brief.probabilities || {};
    const probs = ["BUY", "SELL", "WAIT"].map((label) => {
      const value = Math.max(0, Math.min(1, Number(probabilities[label]) || 0));
      return `<div class="prob-row"><span>${label}</span><span class="prob-track"><span style="width:${(value * 100).toFixed(1)}%"></span></span><b>${(value * 100).toFixed(0)}%</b></div>`;
    }).join("");
    const drivers = (brief.key_drivers || []).slice(0, 8).map((term) => `<span class="tag">${escapeHtml(term)}</span>`).join("");
    const evidenceUrl = safeUrl(brief.evidence_url);
    const evidence = brief.evidence_quote
      ? `<div class="brief-evidence">“${escapeHtml(brief.evidence_quote)}”${evidenceUrl ? `<br><a href="${escapeHtml(evidenceUrl)}" target="_blank" rel="noopener noreferrer">${icon("icon-external")} bukti asli</a>` : ""}</div>`
      : "";
    result.hidden = false;
    result.innerHTML = `
      <div class="result-top">
        <span class="result-stance ${stanceClass(stance)}">${escapeHtml(stance)}</span>
        <span class="result-sub">intensitas ${formatNumber(brief.intensity)} · ${Math.round((Number(brief.confidence) || 0) * 100)}% keyakinan</span>
      </div>
      <p class="result-reason">${escapeHtml(brief.reasoning)}</p>
      <div class="prob-bars">${probs}</div>
      ${drivers ? `<div class="tag-row">${drivers}</div>` : ""}
      ${evidence}
      <div class="brief-meta"><span>${escapeHtml(brief.model)} · ${formatNumber(brief.sample_size)} post</span><span>${formatNumber(brief.latency_ms)} ms</span></div>
      <p class="tiny-note" style="margin:8px 0 0">${escapeHtml(brief.disclaimer || "")}</p>
    `;
  }

  /* ----------------------------------------------------------------- analyzer */
  async function analyzeText() {
    if (STATE.analyzeLoading) return;
    const input = $("#analysisInput");
    const text = (input?.value || "").trim();
    if (!text) {
      showToast("Tempel teks terlebih dahulu.", "warn");
      input?.focus();
      return;
    }
    const button = $("#analyzeBtn");
    const result = $("#analysisResult");
    STATE.analyzeLoading = true;
    STATE.useModel = Boolean($("#useModel")?.checked);
    setBusy(button, true, "Analisis teks", "Menganalisis…");
    result.hidden = true;
    try {
      const analysis = await api("/api/analyze", {
        method: "POST",
        body: JSON.stringify({ text, use_model: STATE.useModel }),
      });
      renderAnalysis(analysis);
      loadHistory();
    } catch (error) {
      showToast(error.message || "Analisis gagal.", "error");
      result.hidden = false;
      result.innerHTML = `<p class="error-note">${escapeHtml(error.message || "Analisis gagal.")}</p>`;
    } finally {
      STATE.analyzeLoading = false;
      setBusy(button, false, "Analisis teks", "Menganalisis…");
    }
  }

  function renderAnalysis(analysis) {
    const result = $("#analysisResult");
    if (!result) return;
    result.hidden = false;
    if (!analysis.is_relevant) {
      result.innerHTML = `
        <div class="result-top"><span class="result-stance wait">TOLAK</span><span class="result-sub">gate relevansi</span></div>
        <p class="result-reason">${escapeHtml(analysis.reason || "Teks tidak relevan secara finansial.")}</p>
      `;
      return;
    }
    const stance = stanceText(analysis.stance);
    const probabilities = analysis.probabilities || {};
    const hasProbabilities = Object.keys(probabilities).length > 0;
    const probs = hasProbabilities
      ? `<div class="prob-bars">${["BUY", "SELL", "WAIT"].map((label) => {
          const value = Math.max(0, Math.min(1, Number(probabilities[label]) || 0));
          return `<div class="prob-row"><span>${label}</span><span class="prob-track"><span style="width:${(value * 100).toFixed(1)}%"></span></span><b>${(value * 100).toFixed(0)}%</b></div>`;
        }).join("")}</div>`
      : "";
    const terms = (analysis.matched_terms || []).map((term) => `<span class="tag">${escapeHtml(term)}</span>`).join("");
    const gates = (analysis.gates || []).map((gate) => {
      const state = String(gate.state || "SKIP").toUpperCase();
      return `<span class="mini-gate ${state.toLowerCase()}">${icon(GATE_ICONS[state] || "icon-skip")}${escapeHtml(gate.id)}</span>`;
    }).join("");
    const engine = analysis.engine === "sentiment_slm" ? "Sentiment SLM" : "Lexicon";
    const modelError = analysis.model_error ? `<p class="error-note">${escapeHtml(analysis.model_error)}</p>` : "";
    const reason = analysis.reasoning || analysis.reason || "";
    result.innerHTML = `
      <div class="result-top">
        <span class="result-stance ${stanceClass(stance)}">${escapeHtml(stance)}</span>
        <span class="result-sub">${escapeHtml(engine)} · intensitas ${formatNumber(analysis.intensity)} · ${Math.round((Number(analysis.confidence) || 0) * 100)}% keyakinan</span>
      </div>
      <p class="result-reason">${escapeHtml(reason)}</p>
      ${probs}
      ${terms ? `<div class="tag-row">${terms}</div>` : ""}
      ${gates ? `<div class="mini-gates">${gates}</div>` : ""}
      ${modelError}
    `;
  }

  /* ------------------------------------------------------------------ history */
  async function loadHistory() {
    try {
      const data = await api("/api/history?limit=20");
      STATE.history = data.entries || [];
      renderHistory();
    } catch {
      // History is a convenience; a failure must not break the radar.
    }
  }

  function renderHistory() {
    const list = $("#historyList");
    if (!list) return;
    if (!STATE.history.length) {
      list.innerHTML = `<p class="empty-state">Belum ada analisis.</p>`;
      return;
    }
    list.innerHTML = STATE.history.map((entry) => {
      const stance = stanceText(entry.stance);
      const isBrief = entry.kind === "brief";
      const isRejected = entry.kind === "rejected";
      const title = isRejected
        ? "Ditolak gate relevansi"
        : (isBrief ? `${entry.ticker || "—"} · Brief Sentiment SLM` : "Analisis teks");
      const detail = entry.text || "—";
      const displayStance = isRejected ? "TOLAK" : stance;
      return `<div class="history-item">
        <span class="history-icon">${icon(isBrief ? "icon-spark" : (isRejected ? "icon-shield" : "icon-pulse"))}</span>
        <span class="history-copy">
          <strong>${escapeHtml(title)} <span class="history-stance ${stanceClass(displayStance)}">${escapeHtml(displayStance)}</span></strong>
          <small>${escapeHtml(detail)}</small>
        </span>
        <span class="muted">${timeAgo(entry.timestamp)}</span>
      </div>`;
    }).join("");
  }

  /* --------------------------------------------------------------------- init */
  async function loadStatus() {
    try {
      const status = await api("/api/status");
      STATE.status = status;
      renderTickers(status.tickers || []);
      if (status.slm) renderModelStatus(status.slm);
      if (status.disclaimer) $("#disclaimer").textContent = status.disclaimer;
    } catch (error) {
      showToast(error.message || "Status aplikasi gagal dimuat.", "error");
      renderTickers([
        { id: "BTCUSD", label: "Bitcoin", short: "BTC" },
        { id: "ETHUSD", label: "Ethereum", short: "ETH" },
        { id: "XAUUSD", label: "Gold", short: "XAU" },
        { id: "SPY", label: "S&P 500", short: "SPY" },
        { id: "EURUSD", label: "EUR / USD", short: "EUR" },
      ]);
    }
  }

  function bindEvents() {
    $("#refreshBtn")?.addEventListener("click", () => loadScan(true));
    $("#briefBtn")?.addEventListener("click", requestBrief);
    $("#analyzeBtn")?.addEventListener("click", analyzeText);
    $("#analysisInput")?.addEventListener("input", (event) => {
      $("#charCount").textContent = `${event.target.value.length} / 4000`;
    });
    $("#analysisInput")?.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        analyzeText();
      }
    });
    window.setInterval(() => {
      if (!document.hidden) loadScan(false);
    }, REFRESH_MS);
    window.setInterval(() => {
      const clock = $("#footerClock");
      if (clock) clock.textContent = new Date().toLocaleTimeString("id-ID");
    }, 30_000);
    const clock = $("#footerClock");
    if (clock) clock.textContent = new Date().toLocaleTimeString("id-ID");
  }

  async function init() {
    bindEvents();
    setupDesktopShell();
    startFpsMonitor();
    await loadStatus();
    await Promise.allSettled([loadScan(false), loadHistory()]);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
