<script>
  import Landing from "./Landing.svelte";
  import Chat from "./Chat.svelte";
  import ThemeToggle from "./ThemeToggle.svelte";

  // --- Router (pushState + $state toggle, no library) ---
  const isApp = () => /^\/app\/?$/.test(location.pathname);
  let route = $state(isApp() ? "app" : "home");
  addEventListener("popstate", () => { route = isApp() ? "app" : "home"; });
  /** @param {string} p */
  function go(p) {
    history.pushState({}, "", p);
    route = p === "/app" ? "app" : "home";
    scrollTo(0, 0);
  }

  // Built app is served by Flask on the same address; in dev, talk to Flask directly.
  const API = import.meta.env.VITE_API_URL ?? (import.meta.env.DEV ? "http://127.0.0.1:5000" : "");

  /** @type {File | null} */
  let file = $state(null);
  let preview = $state("");
  let loading = $state(false);
  let error = $state("");
  /** @type {any} */
  let result = $state(null);
  let view = $state("heatmap");          // heatmap | original
  let zoom = $state(1);
  let dragging = $state(false);
  let name = $state("");
  let pid = $state("");
  let symptoms = $state("");
  /** @type {string | number | null} */
  let age = $state("");
  let engine = $state("template");
  /** @type {HTMLDialogElement | undefined} */
  let dlg = $state();
  let incName = $state(false);
  let fname = $state("");
  let building = $state(false);
  let dlMsg = $state("");
  let runId = 0;                          // ignores answers from superseded requests

  // --- Chat panel width: drag the left edge, arrow keys nudge, double-click resets ---
  const CHAT_DEF = 380, CHAT_MIN = 300, CHAT_MAX = 640;
  let chatW = $state(CHAT_DEF);
  try {
    const v = Number(localStorage.getItem("radigen-chat-w"));
    if (v >= CHAT_MIN && v <= CHAT_MAX) chatW = v;
  } catch { /* storage unavailable: keep the default */ }
  const saveW = () => { try { localStorage.setItem("radigen-chat-w", String(chatW)); } catch { /* ignore */ } };
  /** Keeps room for the left card (290px) and a usable centre column. */
  const maxW = () => Math.max(CHAT_MIN, Math.min(CHAT_MAX, innerWidth - 290 - 420 - 60));
  /** @param {number} w */
  const clampW = (w) => Math.min(maxW(), Math.max(CHAT_MIN, w));
  /** @param {PointerEvent} e */
  function startResize(e) {
    e.preventDefault();
    const x0 = e.clientX, w0 = chatW;
    /** @param {PointerEvent} m */
    const move = (m) => { chatW = clampW(w0 + (x0 - m.clientX)); };
    const up = () => { removeEventListener("pointermove", move); removeEventListener("pointerup", up); saveW(); };
    addEventListener("pointermove", move);
    addEventListener("pointerup", up);
  }
  /** @param {KeyboardEvent} e */
  function nudge(e) {
    if (e.key === "ArrowLeft") chatW = clampW(chatW + 24);
    else if (e.key === "ArrowRight") chatW = clampW(chatW - 24);
    else if (e.key === "Home" || e.key === "Enter") chatW = CHAT_DEF;
    else return;
    e.preventDefault(); saveW();
  }
  function resetW() { chatW = CHAT_DEF; saveW(); }

  const today = new Date().toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
  /** @param {number} x */
  const pct = (x) => (x * 100).toFixed(1) + "%";
  const RING = 2 * Math.PI * 40;

  const top = $derived(result?.top ?? null);
  const ranked = $derived(/** @type {{ name: string, probability: number }[]} */ ((result?.ranked ?? []).filter((/** @type {any} */ c) => c.name !== "No Finding")));
  const mode = $derived(!top ? "original" : view);
  const ageNum = $derived(age === "" || age == null ? null : Number(age));
  const pediatric = $derived(ageNum !== null && ageNum < 18);

  function freeUrl() { if (preview.startsWith("blob:")) URL.revokeObjectURL(preview); }

  /** @param {File | null | undefined} f */
  function setFile(f) {
    if (!f || loading) return;
    freeUrl();
    file = f;
    preview = URL.createObjectURL(f);
    result = null; error = ""; zoom = 1; view = "heatmap";
  }
  /** @param {any} e */
  function onPick(e) { setFile(e.target.files[0]); e.target.value = ""; }
  /** @param {any} e */
  function onDrop(e) {
    e.preventDefault(); dragging = false;
    if (route === "app") setFile(e.dataTransfer?.files[0]);
  }

  async function analyze() {
    if (!file) return;
    const id = ++runId;
    loading = true; error = ""; result = null;
    const body = new FormData();
    body.append("image", file);
    body.append("pid", pid); body.append("age", String(age ?? "")); body.append("symptoms", symptoms);
    try {
      const res = await fetch(`${API}/api/analyze`, { method: "POST", body });
      const data = await res.json();
      if (id !== runId) return;
      if (!res.ok) throw new Error(data.error ?? "Request failed");
      result = data;
      view = "heatmap";
    } catch (e) {
      if (id !== runId) return;
      const msg = e instanceof Error ? e.message : String(e);
      error = msg === "Failed to fetch"
        ? "Can't reach the analysis server. Start it and try again."
        : msg;
    } finally {
      if (id === runId) loading = false;
    }
  }

  function reset() {
    runId++; loading = false; freeUrl();
    file = null; preview = ""; result = null; error = ""; zoom = 1; dlMsg = "";
    name = ""; age = ""; pid = ""; symptoms = "";
  }

  // ---------- PDF report ----------
  const tag = () => pid.trim().replace(/[^\w-]/g, "") || "scan";
  const stamp = () => new Date().toISOString().slice(0, 10).replaceAll("-", "");
  const defaultName = () => `radigen-report-${tag()}-${stamp()}`;
  const cleanName = () => fname.trim().replace(/[\\/:*?"<>|]/g, "").replace(/\.pdf$/i, "") || defaultName();

  function openDownload() { dlMsg = ""; fname = defaultName(); dlg?.showModal(); }

  /** @param {string} src @returns {Promise<HTMLImageElement>} */
  async function loadImg(src) { const i = new Image(); i.src = src; await i.decode(); return i; }

  /** @param {boolean} highlight @returns {Promise<string>} */
  async function snapshot(highlight) {
    const S = 640;
    const c = document.createElement("canvas");
    c.width = S; c.height = S;
    const x = c.getContext("2d");
    if (!x) return "";
    x.drawImage(await loadImg(highlight && top?.heatmap ? top.heatmap : preview), 0, 0, S, S);
    return c.toDataURL("image/jpeg", 0.88);
  }

  /** @returns {Promise<Blob>} */
  async function buildPdf() {
    const { jsPDF } = await import("jspdf");
    const doc = new jsPDF({ unit: "mm", format: "a4" });
    const L = 16, W = 178, PT = 0.3528;
    let y = 16;
    /** @param {number} h */
    const room = (h) => { if (y + h > 274) { doc.addPage(); y = 16; } };
    /** Draws wrapped lines justified to width w; the last line of a paragraph stays left-aligned.
     *  @param {string[]} lines @param {number} x @param {number} y0 @param {number} w */
    const justify = (lines, x, y0, w) => {
      const lh = doc.getFontSize() * PT * doc.getLineHeightFactor();
      const space = doc.getTextWidth(" ");
      lines.forEach((ln, i) => {
        const yy = y0 + i * lh;
        const words = ln.trim().split(/\s+/);
        if (i === lines.length - 1 || words.length < 2) { doc.text(ln, x, yy); return; }
        const gap = (w - words.reduce((t, wd) => t + doc.getTextWidth(wd), 0)) / (words.length - 1);
        if (gap > space * 4) { doc.text(ln, x, yy); return; }          // avoid huge gaps
        let cx = x;
        for (const wd of words) { doc.text(wd, cx, yy); cx += doc.getTextWidth(wd) + gap; }
      });
    };
    /** @param {string} s @param {number} [size] @param {"normal" | "bold"} [style] @param {number[]} [rgb] */
    const write = (s, size = 10, style = "normal", rgb = [35, 35, 50]) => {
      doc.setFont("helvetica", style); doc.setFontSize(size); doc.setTextColor(rgb[0], rgb[1], rgb[2]);
      const lines = doc.splitTextToSize(s, W);
      const h = lines.length * size * PT * 1.4 + 1.4;
      room(h);
      justify(lines, L, y + size * PT, W);
      y += h;
    };
    /** @param {string} s */
    const heading = (s) => { y += 3; room(24); doc.setDrawColor(215, 212, 232); doc.line(L, y, L + W, y); y += 4; write(s, 11, "bold", [70, 50, 160]); };
    /** @param {string} k @param {string | undefined} v */
    const row = (k, v) => {
      if (!v) return;
      doc.setFont("helvetica", "normal"); doc.setFontSize(10);
      const lines = doc.splitTextToSize(v, W - 32);
      const h = lines.length * 4.9 + 0.8;
      room(h);
      doc.setTextColor(110, 110, 130); doc.text(k, L, y + 3.5);
      doc.setTextColor(35, 35, 50); justify(lines, L + 32, y + 3.5, W - 32);
      y += h;
    };

    write("Radigen AI", 20, "bold", [70, 50, 160]);
    doc.setFont("helvetica", "normal"); doc.setFontSize(10); doc.setTextColor(110, 110, 130);
    doc.text(new Date().toLocaleString(), L + W, 22, { align: "right" });
    write("Chest X-ray analysis report", 11, "normal", [90, 90, 110]);

    if ((incName && name.trim()) || pid.trim() || ageNum !== null || symptoms.trim()) {
      heading("Patient");
      if (incName) row("Name", name.trim());
      row("Patient ID", pid.trim());
      row("Age", ageNum !== null ? `${ageNum} years` : "");
      row("Symptoms", symptoms.trim());
    }

    heading("Result");
    if (top) {
      write("Most likely finding: " + top.name, 13, "bold");
      write("Model score: " + pct(top.probability) + (result.close_to ? `. Close to ${result.close_to}.` : ""), 10);
      const rest = ranked.slice(0, 3);
      if (rest.length) write("Other scores: " + rest.map((c) => `${c.name} ${pct(c.probability)}`).join("   ·   "), 9.5, "normal", [90, 90, 110]);
    } else {
      write("No clear finding on this image.", 12, "bold");
    }

    if (preview) {
      const IM = 58, gx = L + (W - (IM * 2 + 8)) / 2;
      room(IM + 12); y += 2;
      const o = await snapshot(false);
      if (o) doc.addImage(o, "JPEG", gx, y, IM, IM);
      if (top?.heatmap) { const h = await snapshot(true); if (h) doc.addImage(h, "JPEG", gx + IM + 8, y, IM, IM); }
      y += IM + 1;
      doc.setFont("helvetica", "normal"); doc.setFontSize(8.5); doc.setTextColor(110, 110, 130);
      doc.text("Original", gx, y + 2.5);
      if (top?.heatmap) doc.text("Heatmap: " + top.name, gx + IM + 8, y + 2.5);
      y += 6;
    }

    const r = result.report;
    if (r) {
      heading("Report");
      if (r.context) write(r.context, 9, "normal", [110, 110, 130]);
      write("Summary", 9.5, "bold"); write(r.summary, 9.5);
      write("Findings", 9.5, "bold"); write(r.findings, 9.5);
      write("Impression", 9.5, "bold"); write(r.impression, 9.5);
      if (r.meaning) { write("What this means", 9.5, "bold"); write(r.meaning, 9.5); }
    }

    const cmp = result.comparison;
    if (pid.trim() && cmp?.returning) {
      heading("Compared with the last scan");
      row("Last scan", cmp.when);
      row("Previous finding", cmp.prev_top ? `${cmp.prev_top} (${pct(cmp.prev_score)})` : "No finding flagged");
      row("This scan", cmp.now_top ? `${cmp.now_top} (${pct(cmp.now_score)})` : "No finding flagged");
      write(cmp.changed ? "The leading finding has changed since the last scan." : "The leading finding is the same as in the last scan.", 9.5);
    }

    if (r?.next) { heading("Next step"); write(r.next, 9.5); }
    y += 4;
    write("About this report. Generated by Radigen AI, a research screening model. It is decision support, not a diagnosis, and does not replace review by a qualified radiologist or physician.", 8, "normal", [120, 120, 140]);

    const pages = doc.getNumberOfPages();
    for (let i = 1; i <= pages; i++) {
      doc.setPage(i);
      doc.setFont("helvetica", "normal"); doc.setFontSize(8); doc.setTextColor(130, 130, 150);
      doc.text("Radigen AI · Research use only", L, 288);
      doc.text(`Page ${i} of ${pages}`, L + W, 288, { align: "right" });
    }
    return doc.output("blob");
  }

  /** @param {boolean} saveAs */
  async function deliver(saveAs) {
    if (!result || building) return;
    building = true; dlMsg = "";
    try {
      const blob = await buildPdf();
      const fn = cleanName() + ".pdf";
      const w = /** @type {any} */ (window);
      if (saveAs && w.showSaveFilePicker) {
        try {
          const h = await w.showSaveFilePicker({ suggestedName: fn, types: [{ description: "PDF document", accept: { "application/pdf": [".pdf"] } }] });
          const st = await h.createWritable(); await st.write(blob); await st.close();
        } catch (e) {
          if (/** @type {any} */ (e)?.name === "AbortError") return;     // closed the Save as window
          throw e;
        }
      } else {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url; a.download = fn; a.click();
        setTimeout(() => URL.revokeObjectURL(url), 1500);
      }
      dlg?.close();
    } catch {
      dlMsg = "Couldn't create the PDF. Try again.";
    } finally {
      building = false;
    }
  }
</script>

<svelte:window
  ondragover={(e) => { if (route === "app") { e.preventDefault(); dragging = true; } }}
  ondragleave={(e) => { if (!e.relatedTarget) dragging = false; }}
  ondrop={onDrop}
/>

<svelte:head>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin="anonymous" />
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Sora:wght@500;600;700&family=Geist+Mono:wght@400;500&display=swap" />
</svelte:head>

{#if route === "home"}
  <Landing {go} />
{:else}
<div class="page">
  <header class="glass top">
    <div class="brand" role="link" tabindex="0" onclick={() => go("/")} onkeydown={(e) => e.key === "Enter" && go("/")}>
      <img class="logo" src="/logo.png" alt="Radigen AI" />
      <div><strong>Radigen AI</strong><small>Chest X-ray analysis</small></div>
    </div>
    <div class="patient">
      <label>Patient name
        <input type="text" placeholder="Optional" maxlength="60" bind:value={name} />
      </label>
      <label class="idf">Patient ID
        <input type="text" placeholder="For history" maxlength="40" bind:value={pid} />
      </label>
      <label class="agef">Age
        <input type="number" min="0" max="120" placeholder="Years" bind:value={age} />
      </label>
    </div>
    <div class="actions">
      <label class="btn ghost" class:off={loading}>
        Upload X-ray
        <input type="file" accept="image/png,image/jpeg" onchange={onPick} disabled={loading} hidden />
      </label>
      <button class="btn primary" onclick={analyze} disabled={!file || loading}>{loading ? "Analyzing…" : "Analyze"}</button>
      {#if preview}<button class="btn ghost" onclick={reset}>Clear</button>{/if}
      <button class="btn ghost" onclick={openDownload} disabled={!result} title={result ? "Download a PDF report" : "Analyze an X-ray first"}>Download report</button>
      <ThemeToggle />
    </div>
  </header>

  <main class="grid" style="--chat-w: {chatW}px">
    <!-- LEFT: symptoms, report, patient history in one long card -->
    <aside class="glass card left">
      <section>
        <h2>Symptoms</h2>
        <input class="sym" type="text" placeholder="Cough, fever, chest pain" aria-label="Symptoms" maxlength="200" bind:value={symptoms} />
        <div class="fine mono dt">{today}</div>
        {#if pediatric}
          <p class="note">Patient is under 18. The model was trained mainly on adult chest X-rays, so results may be less reliable.</p>
        {/if}
      </section>

      <section>
        <h2>Report</h2>
        <div class="eng">
          <button class:on={engine === "template"} onclick={() => (engine = "template")}>Template</button>
          <button class:on={engine === "biogpt"} onclick={() => (engine = "biogpt")}>BioGPT + LoRA</button>
        </div>
        {#if result?.report}
          {@const r = result.report}
          {#if r.context}<p class="fine">{r.context}</p>{/if}
          <p class="rp f"><strong>Summary.</strong> {r.summary}</p>
          <p class="rp i"><strong>Impression.</strong> {r.impression}</p>
          <p class="fine tmpl">Share this report with your doctor.</p>
        {:else}
          <p class="muted">A structured report appears here after analysis.</p>
        {/if}
      </section>

      <section>
        <h2>Patient history</h2>
        {#if !result}
          <p class="muted">Enter a patient ID before analyzing to compare with the last scan on record.</p>
        {:else if !pid.trim()}
          <p class="muted">No patient ID entered, so no history was checked.</p>
        {:else if !result.comparison?.returning}
          <p class="muted">No previous visits on record for this ID.</p>
        {:else}
          {@const c = result.comparison}
          <div class="fine">Last scan: {c.when}</div>
          <div class="crow"><span>Previous finding</span><b>{c.prev_top ?? "None flagged"}</b></div>
          <div class="crow"><span>This scan</span><b>{c.now_top ?? "None flagged"}</b></div>
          <p class="fine"><span class="badge" class:chg={c.changed}>{c.changed ? "Finding changed" : "Same finding"}</span></p>
          {#if c.earlier?.length}
            <div class="fine earlier">
              {#each c.earlier as e}<span>{e.when.split(",")[0]}: {e.top ?? "none flagged"}</span>{/each}
            </div>
          {/if}
        {/if}
      </section>
    </aside>

    <!-- CENTRE: X-ray, then the result -->
    <div class="center">
      <section class="glass hero">
        {#if !preview}
          <div class="drop" class:over={dragging}>
            <div class="drop-icon">+</div>
            <p>Drop a chest X-ray anywhere on this page</p>
            <small>PNG or JPG. Frontal views only (PA or AP), or use Upload X-ray above.</small>
          </div>
        {:else}
          <div class="stage">
            <i class="c tl"></i><i class="c tr"></i><i class="c bl"></i><i class="c br"></i>
            <div class="viewport">
              <div class="zoomer" style="transform: scale({zoom})">
                <img class="base" src={mode === "heatmap" && top?.heatmap ? top.heatmap : preview} alt="Chest X-ray" />
              </div>
            </div>
            {#if loading}<div class="scan"><span>Analyzing…</span></div>{/if}
            <div class="zoom">
              <button onclick={() => (zoom = Math.min(zoom + 0.25, 3))} aria-label="Zoom in">+</button>
              <button onclick={() => (zoom = Math.max(zoom - 0.25, 1))} aria-label="Zoom out">−</button>
              <button onclick={() => (zoom = 1)} aria-label="Reset zoom">⟲</button>
            </div>
            {#if top}
              <div class="seg">
                <button class:on={mode === "original"} onclick={() => (view = "original")}>Original</button>
                <button class:on={mode === "heatmap"} onclick={() => (view = "heatmap")}>Heatmap</button>
              </div>
            {/if}
            {#if top && mode !== "original"}
              <span class="tag">Showing: {top.name} · {pct(top.probability)}</span>
            {:else if result && !top}
              <span class="tag">No finding flagged</span>
            {/if}
          </div>
        {/if}
        {#if error}<p class="err">{error}</p>{/if}
      </section>

      <section class="glass card result">
        {#if top}
          <div class="rbig">
            <div class="ring">
              <svg viewBox="0 0 96 96" width="96" height="96" aria-hidden="true">
                <circle cx="48" cy="48" r="40" fill="none" stroke="var(--track)" stroke-width="9" />
                <circle cx="48" cy="48" r="40" fill="none" stroke="var(--ring)" stroke-width="9" stroke-linecap="round"
                  stroke-dasharray="{top.probability * RING} {RING}" transform="rotate(-90 48 48)" />
              </svg>
              <div class="rnum">{pct(top.probability)}</div>
            </div>
            <div class="rtxt">
              <span class="eyebrow">Most likely finding</span>
              <h3>{top.name}</h3>
              {#if result.close_to}<p class="close">Close to {result.close_to}</p>{/if}
            </div>
          </div>
        {:else if result}
          <div class="rbig">
            <div class="rtxt">
              <span class="eyebrow">Result</span>
              <h3>No finding flagged</h3>
              <p class="close">This does not mean the patient is healthy.</p>
            </div>
          </div>
        {:else}
          <div class="rbig">
            <div class="ring"><svg viewBox="0 0 96 96" width="96" height="96" aria-hidden="true"><circle cx="48" cy="48" r="40" fill="none" stroke="var(--track)" stroke-width="9" /></svg><div class="rnum">—</div></div>
            <div class="rtxt"><span class="eyebrow">Most likely finding</span><h3 class="dim">Awaiting an X-ray</h3><p class="close">Results appear here after you analyze an image.</p></div>
          </div>
        {/if}
        <div class="others">
          <span class="eyebrow">{top ? "Next most likely" : "Highest scores"}</span>
          <div class="tiles">
            {#each result ? ranked.slice(0, 3) : Array(3).fill(null) as c}
              <div class="tile" class:ghost={!c}>
                <span>{c ? c.name : "—"}</span><b>{c ? pct(c.probability) : "—"}</b>
              </div>
            {/each}
          </div>
        </div>
      </section>
    </div>

    <!-- RIGHT: chat dock -->
    <aside class="glass card chat">
      <!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
      <div class="grip" role="separator" aria-orientation="vertical" aria-label="Resize chat panel" aria-valuemin={CHAT_MIN} aria-valuemax={CHAT_MAX} aria-valuenow={chatW}
        tabindex="0" title="Drag to resize. Double-click to reset." onpointerdown={startResize} ondblclick={resetW} onkeydown={nudge}></div>
      <h2>Ask Radigen</h2><Chat api={API} finding={top?.name ?? null} scan={top ? { name: top.name, score: top.probability } : null} />
    </aside>
  </main>

  <footer class="foot">Research use only. Not a diagnosis. A qualified radiologist or physician must review every result.</footer>

  <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
  <dialog bind:this={dlg} class="dlg" aria-labelledby="dl-title" onclick={(e) => { if (e.target === dlg) dlg?.close(); }}>
    <div class="dbox">
      <div class="prev" aria-hidden="true">
        <div class="paper">
          <div class="pt"><b>Radigen AI</b><span>{today}</span></div>
          {#if incName && name.trim()}<div class="pn">{name.trim()}</div>{/if}
          <div class="pf">{top ? top.name : "No clear finding"}{#if top} <em>{pct(top.probability)}</em>{/if}</div>
          <div class="pi"><span style="background-image:url({preview})"></span><span style="background-image:url({top?.heatmap || preview})"></span></div>
          <i></i><i></i><i class="s"></i><i></i><i class="s"></i>
          <div class="pg">Page 1</div>
        </div>
      </div>
      <div class="dr">
        <h2 id="dl-title">Before you download</h2>
        <ul class="priv">
          <li><svg viewBox="0 0 24 24" aria-hidden="true"><rect x="4" y="5" width="16" height="11" rx="2"/><path d="M9 20h6M12 16v4"/></svg>Created in your browser.</li>
          <li><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l7 3v5c0 5-3 8-7 10-4-2-7-5-7-10V6z"/></svg>Not stored on our servers.</li>
          <li><svg viewBox="0 0 24 24" aria-hidden="true"><rect x="5" y="10" width="14" height="10" rx="2"/><path d="M8 10V7a4 4 0 018 0v3"/></svg>May contain patient details. Keep the file somewhere safe.</li>
        </ul>
        {#if name.trim()}
          <label class="sw"><span>Include patient name</span><input type="checkbox" bind:checked={incName} /></label>
        {/if}
        <label class="fn"><span>File name</span>
          <span class="fnb"><input type="text" bind:value={fname} maxlength="80" spellcheck="false" /><em>.pdf</em></span>
        </label>
        {#if dlMsg}<p class="note">{dlMsg}</p>{/if}
        <div class="drow">
          <button class="btn ghost" onclick={() => dlg?.close()}>Cancel</button>
          <button class="btn ghost" onclick={() => deliver(true)} disabled={building}>Save as…</button>
          <button class="btn primary" onclick={() => deliver(false)} disabled={building}>{building ? "Creating…" : "Download PDF"}</button>
        </div>
      </div>
    </div>
  </dialog>
</div>
{/if}

<style>
  :global(body) {
    margin: 0;
    background:
      radial-gradient(900px 600px at 8% -8%, rgba(45, 110, 255, 0.38), transparent 60%),
      radial-gradient(800px 600px at 100% 100%, rgba(110, 60, 220, 0.26), transparent 60%),
      linear-gradient(rgba(120, 170, 255, 0.045) 1px, transparent 1px) 0 0 / 44px 44px,
      linear-gradient(90deg, rgba(120, 170, 255, 0.045) 1px, transparent 1px) 0 0 / 44px 44px,
      #060c1c;
    background-attachment: fixed;
    color: #e8f0ff;
    font-family: Inter, "Segoe UI", ui-sans-serif, system-ui, -apple-system, Roboto, sans-serif;
    font-variant-numeric: tabular-nums;
  }

  /* ---- colour tokens: dark (default) ---- */
  .page {
    --head: Sora, Inter, sans-serif; --num: "Geist Mono", ui-monospace, monospace;
    --ink: #e8f0ff; --mut: #9db0d0; --ph: #7a8eb2;
    --acc: #4da3ff; --danger: #ff6b7a; --ok: #6ee7a8; --ring: #ff6b7a; --me: rgba(77, 163, 255, 0.22); --meink: #e8f0ff;
    --line: rgba(170, 205, 255, 0.18);
    --panel: linear-gradient(145deg, rgba(255, 255, 255, 0.11), rgba(255, 255, 255, 0.03));
    --shadow: 0 22px 50px rgba(0, 6, 28, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.2);
    --field: rgba(255, 255, 255, 0.07); --tile: rgba(255, 255, 255, 0.06); --track: rgba(255, 255, 255, 0.12);
    --soft: rgba(77, 163, 255, 0.22);
    --btn: linear-gradient(145deg, #4f9dff, #2b66ec); --btnink: #fff;
    --note: #ffd9a8; --notebg: rgba(255, 170, 70, 0.1);
    --dlg: #0c1530;
  }

  /* ---- light: near-white page, white panels, soft shadow for depth ---- */
  :global(:root[data-theme="light"]) :global(body) {
    background:
      radial-gradient(520px 380px at 12% 8%, rgba(139, 108, 255, 0.14), transparent 70%),
      radial-gradient(560px 420px at 88% 92%, rgba(169, 139, 255, 0.16), transparent 70%),
      radial-gradient(420px 320px at 55% 45%, rgba(110, 150, 255, 0.06), transparent 70%),
      linear-gradient(rgba(91, 52, 230, 0.03) 1px, transparent 1px) 0 0 / 44px 44px,
      linear-gradient(90deg, rgba(91, 52, 230, 0.03) 1px, transparent 1px) 0 0 / 44px 44px,
      #ffffff;
    background-attachment: fixed;
    color: #150f2e;
  }
  :global(:root[data-theme="light"]) .page {
    --ink: #150f2e; --mut: #40395f; --ph: #5e5880;
    --acc: #5b34e6; --danger: #b81f37; --ok: #117a45; --ring: #5b34e6;
    --line: rgba(91, 52, 230, 0.20);
    --panel: linear-gradient(145deg, rgba(139, 108, 255, 0.11), rgba(91, 52, 230, 0.04));
    --shadow: 0 12px 32px rgba(60, 30, 160, 0.13), inset 0 1px 0 rgba(255, 255, 255, 0.85);
    --field: rgba(255, 255, 255, 0.85); --tile: rgba(255, 255, 255, 0.65); --track: rgba(91, 52, 230, 0.20);
    --soft: rgba(110, 80, 230, 0.14);
    --me: linear-gradient(145deg, #8b6cff, #5b34e6); --meink: #fff;
    --btn: linear-gradient(145deg, #8b6cff, #5b34e6); --btnink: #fff;
    --note: #5f3700; --notebg: rgba(255, 170, 70, 0.22);
    --dlg: rgba(255, 255, 255, 0.98);
  }
  :global(:root[data-theme="light"]) .btn.primary { box-shadow: 0 8px 22px rgba(91, 52, 230, 0.34), inset 0 1px 0 rgba(255, 255, 255, 0.45); }
  :global(:root[data-theme="light"]) .viewport { box-shadow: 0 0 0 1px rgba(91, 52, 230, 0.35), 0 10px 30px rgba(91, 52, 230, 0.16); border-color: transparent; }
  :global(:root[data-theme="light"]) .dlg { box-shadow: 0 24px 60px rgba(60, 30, 160, 0.35); }

  :global(button:focus-visible), :global(input:focus-visible), :global(label.btn:focus-within) { outline: 2px solid var(--acc); outline-offset: 2px; }
  .mono, .rnum, .tile b, .tag, .crow b { font-family: var(--num); }
  h2, h3, .brand strong { font-family: var(--head); }

  /* one-screen shell: nothing scrolls on a normal desktop window */
  .page { height: 100dvh; padding: 14px 16px 8px; max-width: 1680px; margin: 0 auto; box-sizing: border-box; display: flex; flex-direction: column; gap: 12px; color: var(--ink); }
  .glass { background: var(--panel); border: 1px solid var(--line); border-radius: 22px; backdrop-filter: blur(24px) saturate(165%); -webkit-backdrop-filter: blur(24px) saturate(165%); box-shadow: var(--shadow); }

  .top { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; padding: 8px 14px 8px 10px; border-radius: 28px; }
  .brand { display: flex; align-items: center; gap: 12px; cursor: pointer; }
  .brand strong { display: block; font-size: 1rem; }
  .brand small { color: var(--mut); font-size: 0.72rem; }
  .logo { width: 44px; height: 44px; border-radius: 50%; object-fit: cover; display: block; flex: none; box-shadow: 0 3px 12px rgba(0, 0, 0, .28); }
  .patient { display: flex; gap: 10px; margin-left: 6px; }
  .patient label { display: grid; gap: 2px; font-size: 0.68rem; color: var(--mut); }
  .patient input, .sym { font: inherit; font-size: 0.86rem; color: var(--ink); padding: 6px 12px; border-radius: 12px; background: var(--field); border: 1px solid var(--line); }
  .patient input { width: 170px; }
  .patient .agef input { width: 80px; }
  .patient .idf input { width: 120px; }
  .patient input::placeholder, .sym::placeholder { color: var(--ph); }
  .actions { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-left: auto; }
  .btn { font: inherit; font-size: 0.86rem; padding: 7px 16px; border-radius: 999px; cursor: pointer; border: 1px solid var(--line); color: var(--ink); background: var(--field); transition: background .15s; }
  .btn:hover:not(:disabled):not(.off) { border-color: var(--acc); }
  .btn.primary { background: var(--btn); border-color: transparent; color: var(--btnink); }
  .btn:disabled, .btn.off { opacity: .5; cursor: not-allowed; }

  /* three columns: patient card | x-ray + result | chat dock */
  .grid { flex: 1; min-height: 0; display: grid; gap: 14px; grid-template-columns: minmax(250px, 290px) minmax(0, 1fr) minmax(min(330px, var(--chat-w, 380px)), var(--chat-w, 380px)); }
  .left { grid-column: 1; display: flex; flex-direction: column; gap: 14px; overflow: hidden; }
  .left section + section { border-top: 1px solid var(--line); padding-top: 14px; }
  .center { grid-column: 2; display: grid; grid-template-rows: minmax(0, 1fr) auto; gap: 14px; min-height: 0; }
  .chat { grid-column: 3; display: flex; flex-direction: column; overflow: hidden; position: relative; }
  .grip { position: absolute; left: 0; top: 0; bottom: 0; width: 14px; cursor: col-resize; touch-action: none; z-index: 3; border-radius: 22px 0 0 22px; }
  .grip::after { content: ""; position: absolute; left: 4px; top: 50%; width: 4px; height: 44px; margin-top: -22px; border-radius: 4px; background: var(--line); transition: background .15s, height .15s, margin-top .15s; }
  .grip:hover::after, .grip:focus-visible::after { background: var(--acc); height: 64px; margin-top: -32px; }
  .grip:focus-visible { outline: none; }
  .card { padding: 16px 18px; min-height: 0; }
  h2 { margin: 0 0 10px; font-size: 0.95rem; font-weight: 600; color: var(--ink); }
  h3 { margin: 2px 0 4px; font-size: 1.55rem; font-weight: 500; line-height: 1.1; letter-spacing: -.02em; }
  h3.dim { color: var(--mut); font-size: 1.3rem; }
  p { line-height: 1.5; font-size: 0.88rem; margin: 0 0 10px; }
  .muted { color: var(--mut); }
  .fine { font-size: 0.76rem; color: var(--mut); }
  .eyebrow { font-size: .72rem; color: var(--mut); text-transform: uppercase; letter-spacing: .08em; }
  .note { font-size: .82rem; color: var(--note); background: var(--notebg); border-left: 2px solid #ffb04d; padding: 8px 11px; border-radius: 6px; }
  .sym { width: 100%; box-sizing: border-box; font-size: .84rem; margin-bottom: 6px; }
  .eng { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }
  .eng button { font: inherit; font-size: .74rem; color: var(--ink); background: var(--tile); border: 1px solid var(--line); border-radius: 999px; padding: 4px 11px; cursor: pointer; }
  .eng button.on { border-color: var(--acc); background: var(--soft); }
  .crow { display: flex; justify-content: space-between; gap: 8px; font-size: .84rem; padding: 5px 0; border-bottom: 1px solid var(--line); }
  .crow span { color: var(--mut); }
  .badge { display: inline-block; font-size: .72rem; padding: 2px 9px; border-radius: 999px; background: var(--tile); color: var(--mut); }
  .badge.chg { background: var(--soft); color: var(--ink); }
  .earlier { display: grid; gap: 2px; }
  .left p, .left .fine, .note, .priv li { text-align: justify; hyphens: auto; -webkit-hyphens: auto; }

  /* x-ray: always fits the space it is given */
  .hero { position: relative; padding: 14px; container-type: size; display: grid; grid-template-columns: minmax(0, 1fr); grid-template-rows: minmax(0, 1fr); place-items: center; min-height: 0; }
  .drop { width: 100%; height: 100%; box-sizing: border-box; display: grid; place-content: center; text-align: center; gap: 6px; padding: 24px; border: 1.5px dashed var(--line); border-radius: 16px; color: var(--mut); transition: border-color .15s, background .15s; }
  .drop.over { border-color: var(--acc); background: var(--soft); }
  .drop p { margin: 0; color: var(--ink); font-size: 1.05rem; }
  .drop-icon { font-size: 2.6rem; font-weight: 200; color: var(--acc); }
  .stage { position: relative; width: min(100cqw, 100cqh); height: min(100cqw, 100cqh); }
  .viewport { position: absolute; inset: 0; overflow: hidden; border-radius: 14px; background: #03060c; border: 1px solid var(--line); }
  .zoomer { position: absolute; inset: 0; transition: transform .2s; transform-origin: center; }
  .zoomer img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: fill; display: block; }
  .c { position: absolute; width: 22px; height: 22px; border: 2px solid var(--acc); opacity: .8; pointer-events: none; z-index: 2; }
  .c.tl { top: -6px; left: -6px; border-right: 0; border-bottom: 0; border-top-left-radius: 10px; }
  .c.tr { top: -6px; right: -6px; border-left: 0; border-bottom: 0; border-top-right-radius: 10px; }
  .c.bl { bottom: -6px; left: -6px; border-right: 0; border-top: 0; border-bottom-left-radius: 10px; }
  .c.br { bottom: -6px; right: -6px; border-left: 0; border-top: 0; border-bottom-right-radius: 10px; }
  .zoom { position: absolute; top: 14px; left: 14px; display: flex; flex-direction: column; gap: 6px; }
  .zoom button { width: 34px; height: 34px; border-radius: 12px; cursor: pointer; font-size: 1rem; color: #e8f0ff; background: rgba(6, 12, 28, .7); border: 1px solid rgba(170, 205, 255, .25); }
  .zoom button:hover { border-color: #6ee7f9; }
  .seg { position: absolute; top: 14px; right: 14px; display: flex; padding: 3px; border-radius: 999px; background: rgba(6, 12, 28, .7); border: 1px solid rgba(170, 205, 255, .25); }
  .seg button { font: inherit; font-size: .78rem; padding: 5px 12px; border: 0; border-radius: 999px; background: transparent; color: #a9bbd9; cursor: pointer; }
  .seg button.on { background: rgba(77, 163, 255, .32); color: #fff; }
  .tag { position: absolute; left: 14px; bottom: 14px; font-size: .78rem; color: #ffd0d5; background: rgba(40, 14, 22, .78); border: 1px solid rgba(255, 140, 150, .5); border-radius: 999px; padding: 4px 12px; }
  .scan { position: absolute; inset: 0; display: grid; place-content: center; border-radius: 14px; background: rgba(6, 12, 28, .55); color: #e8f0ff; overflow: hidden; z-index: 1; }
  .scan::after { content: ""; position: absolute; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, transparent, #6ee7f9, transparent); box-shadow: 0 0 14px #6ee7f9; animation: sweep 1.4s linear infinite; }
  @keyframes sweep { from { top: 0; } to { top: 100%; } }
  .err { position: absolute; left: 14px; right: 14px; bottom: 6px; margin: 0; text-align: center; color: var(--danger); }

  /* result card */
  .result { display: grid; grid-template-columns: minmax(200px, auto) minmax(0, 1fr); gap: 22px; align-items: center; container-type: inline-size; }
  .rbig { display: flex; align-items: center; gap: 16px; }
  .ring { position: relative; width: 96px; height: 96px; flex: none; }
  .ring svg { width: 100%; height: 100%; }
  .rnum { position: absolute; inset: 0; display: grid; place-content: center; font-size: 1.1rem; font-weight: 500; letter-spacing: -.02em; }
  .close { margin: 0; font-size: .82rem; color: var(--mut); }
  .tiles { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin-top: 8px; }
  .tile { display: flex; flex-direction: column; gap: 2px; padding: 8px 10px; border-radius: 12px; background: var(--tile); border: 1px solid var(--line); min-width: 0; }
  .tile span { font-size: .76rem; color: var(--mut); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .tile b { font-size: 1rem; font-weight: 500; letter-spacing: -.02em; }
  .tile.ghost { opacity: .45; }
  @container (max-width: 520px) { .tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); } }

  .foot { font-size: .7rem; color: var(--mut); text-align: center; line-height: 1.2; }

  /* download dialog */
  .dlg { border: 1px solid var(--line); border-radius: 24px; padding: 0; background: var(--dlg); color: var(--ink); width: min(700px, calc(100vw - 32px)); box-shadow: 0 30px 80px rgba(0, 0, 0, .5); animation: pop .2s ease-out; backdrop-filter: blur(24px); }
  .dlg::backdrop { background: rgba(4, 8, 20, .55); backdrop-filter: blur(5px); }
  .dbox { padding: 24px; display: grid; grid-template-columns: 210px minmax(0, 1fr); gap: 26px; }
  .prev { display: grid; place-items: start center; padding: 14px; border-radius: 16px; background: var(--tile); border: 1px solid var(--line); }
  .paper { width: 100%; aspect-ratio: 1 / 1.414; background: #fff; border-radius: 4px; padding: 10px; box-sizing: border-box; display: flex; flex-direction: column; gap: 5px; box-shadow: 0 10px 26px rgba(0, 0, 0, .3); color: #231f3a; }
  .paper .pt { display: flex; justify-content: space-between; align-items: baseline; font-size: 7px; color: #6a6486; } .paper .pt b { font-family: var(--head); font-size: 9px; color: #4632a8; }
  .paper .pn { font-size: 7px; color: #231f3a; } .paper .pf { font: 600 8px var(--head); } .paper .pf em { font: normal 7px var(--num); color: #c4243d; margin-left: 4px; }
  .paper .pi { display: flex; gap: 4px; } .paper .pi span { flex: 1; aspect-ratio: 1; border-radius: 2px; background: #111 center / cover; }
  .paper i { display: block; height: 3px; border-radius: 2px; background: #dcd8ec; } .paper i.s { width: 60%; }
  .paper .pg { margin-top: auto; font-size: 6px; color: #8a86a4; text-align: right; }
  .dr { display: flex; flex-direction: column; gap: 12px; min-width: 0; }
  .dr h2 { font-size: 1.25rem; margin: 0; }
  .priv { list-style: none; margin: 0; padding: 0; display: grid; gap: 9px; }
  .priv li { display: flex; gap: 10px; align-items: center; font-size: .86rem; color: var(--mut); line-height: 1.35; }
  .priv svg { width: 18px; height: 18px; flex: none; fill: none; stroke: var(--acc); stroke-width: 1.7; stroke-linecap: round; stroke-linejoin: round; }
  .sw { display: flex; justify-content: space-between; align-items: center; font-size: .9rem; cursor: pointer; padding: 8px 12px; border-radius: 12px; background: var(--tile); border: 1px solid var(--line); }
  .sw input { appearance: none; -webkit-appearance: none; width: 38px; height: 22px; border-radius: 999px; background: var(--track); position: relative; cursor: pointer; transition: background .15s; margin: 0; }
  .sw input::after { content: ""; position: absolute; top: 3px; left: 3px; width: 16px; height: 16px; border-radius: 50%; background: #fff; transition: transform .15s; box-shadow: 0 1px 3px rgba(0, 0, 0, .35); }
  .sw input:checked { background: var(--acc); } .sw input:checked::after { transform: translateX(16px); }
  .fn { display: grid; gap: 4px; font-size: .74rem; color: var(--mut); }
  .fnb { display: flex; align-items: center; border-radius: 12px; background: var(--field); border: 1px solid var(--line); padding-right: 12px; }
  .fnb input { flex: 1; min-width: 0; font: inherit; font-size: .88rem; color: var(--ink); background: none; border: 0; padding: 9px 4px 9px 12px; outline: none; }
  .fnb em { font-style: normal; font-size: .88rem; color: var(--mut); }
  .drow { display: flex; justify-content: flex-end; gap: 8px; margin-top: auto; flex-wrap: wrap; }
  @keyframes pop { from { opacity: 0; transform: translateY(10px) scale(.97); } to { opacity: 1; transform: none; } }
  @media (max-width: 640px) { .dbox { grid-template-columns: 1fr; } .prev { display: none; } }

  /* laptop-height windows: tighter spacing so everything still fits without scrolling */
  @media (max-height: 800px) {
    .page { padding: 10px 14px 6px; gap: 8px; }
    .grid, .center, .left { gap: 10px; }
    .left section + section { padding-top: 10px; }
    .card { padding: 12px 14px; }
    .top { padding: 6px 12px 6px 8px; }
    h2 { margin-bottom: 6px; }
    h3 { font-size: 1.35rem; }
    p { font-size: .8rem; line-height: 1.4; margin-bottom: 6px; }
    .rp { display: -webkit-box; -webkit-box-orient: vertical; overflow: hidden; }
    .rp.f { -webkit-line-clamp: 4; line-clamp: 4; }
    .rp.i { -webkit-line-clamp: 3; line-clamp: 3; }
    .dt, .tmpl, .earlier { display: none; }
    .ring { width: 72px; height: 72px; }
    .rnum { font-size: .95rem; }
    .result { gap: 14px; }
    .tiles { gap: 6px; margin-top: 6px; }
    .tile { padding: 5px 9px; }
    .tile span { font-size: .72rem; }
    .tile b { font-size: .92rem; }
  }
  /* very short windows fall back to a normal page that scrolls */
  @media (max-height: 600px) {
    .page { height: auto; min-height: 100dvh; }
    .grid { min-height: 820px; }
    .left { overflow: visible; }
  }
  @media (max-width: 1000px) {
    .page { height: auto; }
    .grid { grid-template-columns: 1fr; min-height: 0; }
    .left, .center, .chat { grid-column: auto; }
    .grip { display: none; }
    .hero { height: min(100vw, 560px); }
    .chat { min-height: 420px; }
    .result { grid-template-columns: 1fr; }
  }
  @media (prefers-reduced-motion: reduce) {
    .scan::after, .dlg { animation: none; }
    .zoomer { transition: none; }
  }
</style>