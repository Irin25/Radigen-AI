<script>
  import Globe from "./Globe.svelte";
  import ThemeToggle from "./ThemeToggle.svelte";
  let { go } = $props();

  // Held-out test AUROC (combined_test_results.txt): [class, NIH, CheXpert]
  /** @type {[string, number, number][]} */
  const perf = [["Atelectasis",.8222,.7251],["Cardiomegaly",.9153,.8778],["Consolidation",.8533,.7445],["Edema",.9203,.8578],["Pleural Effusion",.8885,.8888],["Pneumonia",.7487,.7592],["Pneumothorax",.8719,.8728],["No Finding",.8326,.8832]];
  /** @param {1 | 2} i */
  const mean = (i) => (perf.reduce((s, r) => s + r[i], 0) / perf.length).toFixed(3);

  const stats = [
    { to: 8, dec: 0, l: "findings scored on every scan" },
    { to: 3, dec: 0, l: "public datasets used in the project" },
    { to: 3, dec: 0, l: "connected pipelines behind each result" },
    { to: +mean(1), dec: 3, l: "mean AUROC on NIH ChestX-ray14" },
  ];

  const stages = [
    { n: "01", t: "Vision pipeline", s: ["Preprocessing to 224 px grayscale", "Swin-Tiny classifier, 8 classes", "Grad-CAM++ heatmap of the top finding"] },
    { n: "02", t: "Memory and retrieval", s: ["Patient lookup across prior visits", "FAISS + LangChain search of a reference library", "Findings, age and symptoms as context"] },
    { n: "03", t: "Generation", s: ["Structured report built from the findings", "Follow-up comparison with the last scan", "Chatbot that answers from the reference library"] },
  ];

  const findings = [
    { t: "Atelectasis", d: "Partial or complete collapse of lung tissue, so part of the lung looks denser." },
    { t: "Cardiomegaly", d: "An enlarged heart silhouette on the frontal view." },
    { t: "Consolidation", d: "Air spaces filled with fluid, pus or cells, giving a solid-looking patch." },
    { t: "Edema", d: "Excess fluid in the lungs, often linked to heart failure." },
    { t: "Pleural Effusion", d: "Fluid collecting between the lung and the chest wall." },
    { t: "Pneumonia", d: "Infection of the lung tissue, often seen as a hazy opacity." },
    { t: "Pneumothorax", d: "Air in the space around the lung, which can partly collapse it." },
    { t: "No Finding", d: "None of the other seven findings were flagged on the scan." },
  ];

  const reading = [
    { t: "The highlighted region", d: "A Grad-CAM++ heatmap marks the part of the image that most influenced the top finding. It shows where the model looked, not proof that the finding is present there." },
    { t: "Scores and thresholds", d: "Each finding gets a score from 0 to 100% and has its own cut-off, chosen during validation. Findings differ in how easy they are to spot, so one shared cut-off would be unfair to some of them." },
    { t: "Comparison with the last visit", d: "Enter a patient ID and the analyzer compares this scan's top finding with the last scan on record for that ID. Only the ID, age and scores are kept, never names or images." },
  ];

  const limits = [
    "Frontal chest radiographs only. Other views and other body parts are out of scope.",
    "Built on public datasets (NIH ChestX-ray14, CheXpert and MIMIC-CXR) and not validated on any hospital's own patients.",
    "The analyzer names one most likely finding, but several findings can be present at once.",
  ];


  const faqs = [
    { q: "What does Radigen AI do?", a: "It screens a frontal chest X-ray for eight findings, shows where it looked with a heatmap, drafts a structured report, and compares with the patient's last scan." },
    { q: "Which X-rays work?", a: "Frontal chest views (PA or AP) as PNG or JPG." },
    { q: "How do I read the result?", a: "The most likely finding is shown with its score. The other scores add context, and the heatmap marks the region that drove it." },
    { q: "Is my image stored?", a: "No. The image is processed in memory and discarded. History keeps only the ID, age, scores and top finding, never a name." },
    { q: "Can it replace a radiologist?", a: "No. It supports review, and every result still needs a qualified clinician." },
    { q: "How is it trained?", a: "On NIH ChestX-ray14 and CheXpert, with MIMIC-CXR used for report writing." },
  ];



  const crew = ["Chandna S", "Haniya S", "Irin Sujith John", "Jayram S Iyer", "Neil G Manoj"];

  /**
   * Counts a number up once when it scrolls into view.
   * @param {HTMLElement} node
   * @param {{ to: number, dec: number }} p
   */
  function countUp(node, p) {
    /** @param {number} v */
    const show = (v) => { node.textContent = v.toFixed(p.dec); };
    if (matchMedia("(prefers-reduced-motion: reduce)").matches || !("IntersectionObserver" in window)) { show(p.to); return; }
    show(0);
    let raf = 0;
    const io = new IntersectionObserver(([e]) => {
      if (!e.isIntersecting) return;
      io.disconnect();
      const t0 = performance.now();
      /** @param {number} t */
      const step = (t) => {
        const k = Math.min(1, (t - t0) / 1200);
        show(p.to * (1 - Math.pow(1 - k, 3)));
        if (k < 1) raf = requestAnimationFrame(step);
      };
      raf = requestAnimationFrame(step);
    }, { threshold: 0.4 });
    io.observe(node);
    return { destroy() { io.disconnect(); cancelAnimationFrame(raf); } };
  }

  /**
   * Fades an element in the first time it scrolls into view.
   * @param {HTMLElement} node
   * @param {number} [delay]
   */
  function reveal(node, delay = 0) {
    node.classList.add("rv");
    node.style.setProperty("--d", delay + "ms");
    if (matchMedia("(prefers-reduced-motion: reduce)").matches || !("IntersectionObserver" in window)) { node.classList.add("in"); return; }
    const io = new IntersectionObserver((es) => {
      for (const e of es) if (e.isIntersecting) { node.classList.add("in"); io.disconnect(); }
    }, { threshold: 0.12, rootMargin: "0px 0px -6% 0px" });
    io.observe(node);
    return { destroy() { io.disconnect(); } };
  }
</script>

<div class="lp">
  <header class="nav glass">
    <div class="brand"><img class="logo" src="/logo.png" alt="" />Radigen AI</div>
    <nav class="links"><a href="#pipeline">How it works</a><a href="#findings">Findings</a><a href="#perf">Performance</a><a href="#faq">FAQ</a></nav>
    <div class="acts"><ThemeToggle /><button class="btn sm" onclick={() => go("/app")}>Open analyzer</button></div>
  </header>

  <section class="hero">
    <div>
      <span class="tag glass">Research prototype</span>
      <h1>AI chest X-ray analysis, <em>explained.</em></h1>
      <p class="lead">Screening support for frontal chest radiographs: eight findings, visual localisation and a structured report, with comparison against the patient's prior study.</p>
      <div class="cta"><button class="btn" onclick={() => go("/app")}>Open the analyzer →</button><a class="btn ghost" href="#pipeline">How it works</a></div>
      <div class="facts"><span><b>8</b>findings</span><span><b>Per class</b>thresholds</span><span><b>Grad-CAM++</b>heatmaps</span></div>
    </div>
    <div class="stage"><Globe /></div>
  </section>

  <section class="band" use:reveal>
    <div class="stats glass">
      {#each stats as s}
        <div class="stat"><b use:countUp={s}>{s.to.toFixed(s.dec)}</b><span>{s.l}</span></div>
      {/each}
    </div>
  </section>

  <section class="sec" id="pipeline">
    <p class="eyebrow">How it works</p>
    <h2>One scan, three pipelines.</h2>
    <div class="cols">
      {#each stages as s, i}
        <div class="card glass" use:reveal={i * 90}><span class="n">{s.n}</span><h3>{s.t}</h3><ul>{#each s.s as x}<li>{x}</li>{/each}</ul></div>
      {/each}
    </div>
  </section>

  <section class="sec" id="findings">
    <p class="eyebrow">Findings</p>
    <h2>Eight findings, one scan.</h2>
    <div class="fgrid">
      {#each findings as f, i}
        <div class="fcard glass" use:reveal={(i % 4) * 70}><h3>{f.t}</h3><p>{f.d}</p></div>
      {/each}
    </div>
  </section>

  <section class="sec">
    <p class="eyebrow">Reading the output</p>
    <h2>What the analyzer shows you.</h2>
    <div class="cols">
      {#each reading as r, i}
        <div class="card glass" use:reveal={i * 90}><h3>{r.t}</h3><p class="body">{r.d}</p></div>
      {/each}
    </div>
  </section>

  <section class="sec" id="perf">
    <p class="eyebrow">Performance</p>
    <h2>Held-out test AUROC.</h2>
    <div class="legend"><span><i class="d nih"></i>NIH, mean {mean(1)}</span><span><i class="d chx"></i>CheXpert, mean {mean(2)}</span></div>
    <div class="perf glass" use:reveal>
      {#each perf as [n, a, b]}
        <div class="pr"><span>{n}</span>
          <div class="pb"><i class="nih" style="width:{(a - .5) * 200}%"></i></div><em>{a.toFixed(3)}</em>
          <div class="pb"><i class="chx" style="width:{(b - .5) * 200}%"></i></div><em>{b.toFixed(3)}</em>
        </div>
      {/each}
    </div>
  </section>

  <section class="sec" id="limits">
    <p class="eyebrow">Limitations</p>
    <h2>What it can't do.</h2>
    <ul class="lim" use:reveal>{#each limits as l}<li>{l}</li>{/each}</ul>
  </section>

  <section class="sec" id="faq">
    <div class="faqw">
      <div class="faqh" use:reveal>
        <p class="eyebrow">FAQ</p>
        <h2>Common questions.</h2>
        <p class="sub2">Short answers about what the analyzer does and doesn't do.</p>
        <button class="lnk2" onclick={() => go("/app")}>Still have a question? Ask Radigen →</button>
      </div>
      <div class="fl" use:reveal={100}>
        {#each faqs as f}<details name="faq"><summary>{f.q}</summary><p>{f.a}</p></details>{/each}
      </div>
    </div>
  </section>

  <section class="sec cta2">
    <h2>See what a second look returns.</h2>
    <button class="btn" onclick={() => go("/app")}>Open the analyzer →</button>
  </section>

  <footer class="foot">
    <div class="fcols">
      <div>
        <div class="brand"><img class="logo" src="/logo.png" alt="" />Radigen AI</div>
        <p class="fine">Explainable chest X-ray analysis. Built on NIH ChestX-ray14, CheXpert and MIMIC-CXR.</p>
      </div>
      <nav class="flinks">
        <a href="#pipeline">How it works</a><a href="#findings">Findings</a><a href="#perf">Performance</a><a href="#faq">FAQ</a>
        <button class="lnk" onclick={() => go("/app")}>Open analyzer</button>
      </nav>
    </div>
    <p class="fine disc">Not a medical device and not a diagnosis. A qualified radiologist or physician must review every result. Please don't upload identifiable patient data.</p>
    <p class="fine">© 2026 Radigen AI</p>
  </footer>
  <p class="made">Group 9 · {crew.join(" · ")} · B.Tech main project, St. Thomas Institute for Science &amp; Technology</p>
</div>

<style>
  :global(html) { scroll-padding-top: 16px; }
  @media (prefers-reduced-motion: no-preference) { :global(html) { scroll-behavior: smooth; } }
  .lp { --ink: #e8f1ff; --mut: #9db0d0; --acc: #3d9bff; --acc2: #7fe3ff; --em: #7fe3ff; --line: rgba(140,200,255,.18); --glass: rgba(255,255,255,.06); --shadow: 0 18px 40px rgba(0,8,40,.65); --glow: rgba(0,8,40,.6); --btnink: #04101e; --track: rgba(255,255,255,.08); min-height: 100vh; color: var(--ink); font-family: Inter, system-ui, sans-serif; position: relative; overflow-x: clip; }
  .lp::before { content: ""; position: fixed; inset: 0; z-index: -1; background: radial-gradient(55% 55% at 78% 30%, rgba(30,100,255,.32), transparent 70%), radial-gradient(42% 45% at 6% 92%, rgba(90,50,210,.24), transparent 70%), #050a18; }
  .glass { background: var(--glass); border: 1px solid var(--line); backdrop-filter: blur(18px) saturate(160%); -webkit-backdrop-filter: blur(18px) saturate(160%); box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,.12); }
  .nav { position: relative; z-index: 5; margin: 12px 4vw 0; display: flex; align-items: center; gap: 1.4rem; padding: .5rem .6rem; border-radius: 999px; }
  .brand { display: flex; align-items: center; gap: .6rem; font-weight: 600; }
  .logo { width: 38px; height: 38px; border-radius: 50%; object-fit: cover; display: block; flex: none; }
  .links { display: flex; gap: 1.3rem; margin-left: auto; font-size: .9rem; }
  .links a { color: var(--mut); text-decoration: none; } .links a:hover { color: var(--ink); }
  .btn { display: inline-flex; gap: .6rem; align-items: center; border: 0; border-radius: 14px; padding: .95rem 1.5rem; font: 500 .98rem Inter, sans-serif; color: var(--btnink); text-decoration: none; cursor: pointer; background: linear-gradient(160deg, var(--acc2), var(--acc)); box-shadow: inset 0 1px 0 rgba(255,255,255,.5), 0 12px 28px var(--glow); }
  .btn.sm { padding: .55rem 1.1rem; font-size: .85rem; border-radius: 999px; }
  .btn.ghost { color: var(--ink); background: var(--glass); border: 1px solid var(--line); box-shadow: none; }
  .hero { display: grid; grid-template-columns: minmax(0,1fr) minmax(0,1.05fr); align-items: center; gap: 2rem; padding: 1.5rem 5vw 3rem; min-height: calc(100dvh - 70px); }
  .tag { display: inline-flex; align-items: center; gap: .5rem; padding: .35rem .8rem; border-radius: 999px; font: 500 .78rem "Geist Mono", ui-monospace, monospace; color: var(--mut); }
  .tag::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: var(--acc); }
  h1 { margin: 1.3rem 0 1.2rem; font: 600 clamp(2.3rem,4.5vw,4.1rem)/1.08 Sora, Inter, sans-serif; letter-spacing: -.03em; }
  h1 em { color: var(--em); }
  .lead { max-width: 31rem; color: var(--mut); font-size: 1.1rem; line-height: 1.6; margin: 0 0 2rem; }
  .cta { display: flex; gap: .8rem; flex-wrap: wrap; }
  .facts { display: flex; gap: 1.6rem; flex-wrap: wrap; margin-top: 2.4rem; font: .78rem "Geist Mono", ui-monospace, monospace; color: var(--mut); }
  .facts b { display: block; font: 600 1.3rem Sora, Inter, sans-serif; color: var(--ink); margin-bottom: .1rem; }
  .stage { position: relative; width: min(100%, 76vh, 680px); justify-self: center; }

  .band { padding: 0 5vw; }
  .stats { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); border-radius: 26px; padding: 1.6rem 1rem; }
  .stat { text-align: center; padding: 0 .8rem; }
  .stat + .stat { border-left: 1px solid var(--line); }
  .stat b { display: block; font: 600 clamp(1.9rem,3.4vw,2.8rem)/1 Sora, Inter, sans-serif; color: var(--em); }
  .stat span { display: block; margin-top: .5rem; font-size: .85rem; color: var(--mut); line-height: 1.4; }

  .sec { padding: 4.5rem 5vw 1rem; }
  .eyebrow { font: 500 .75rem "Geist Mono", monospace; color: var(--em); letter-spacing: .14em; text-transform: uppercase; margin: 0 0 .6rem; }
  h2 { font: 600 clamp(1.6rem,2.8vw,2.3rem)/1.15 Sora, Inter, sans-serif; letter-spacing: -.02em; margin: 0 0 1.6rem; }
  .cols { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 1rem; }
  .card { padding: 1.5rem; border-radius: 22px; }
  .card .n { font: .75rem "Geist Mono", monospace; color: var(--em); }
  .card h3 { font: 600 1.2rem Sora, Inter, sans-serif; margin: .8rem 0; }
  .card ul { margin: 0; padding: 0; list-style: none; display: grid; gap: .6rem; color: var(--mut); line-height: 1.4; }
  .card li::before { content: "→ "; color: var(--acc); }
  .card .body { margin: 0; color: var(--mut); line-height: 1.6; font-size: .95rem; }
  .fgrid { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 1rem; }
  .fcard { padding: 1.2rem 1.3rem; border-radius: 20px; }
  .fcard h3 { font: 600 1.05rem Sora, Inter, sans-serif; margin: 0 0 .5rem; }
  .fcard p { margin: 0; color: var(--mut); font-size: .88rem; line-height: 1.55; }

  .legend { display: flex; gap: 1.2rem; flex-wrap: wrap; color: var(--mut); font-size: .85rem; margin: -.8rem 0 1rem; }
  .d { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: .4rem; }
  .perf { padding: 1.2rem 1.4rem; border-radius: 22px; display: grid; gap: .7rem; overflow-x: auto; }
  .pr { display: grid; grid-template-columns: 9rem 1fr 3.2rem 1fr 3.2rem; align-items: center; gap: .7rem; font-size: .85rem; min-width: 520px; }
  .pb { height: 6px; border-radius: 3px; background: var(--track); }
  .pb i { display: block; height: 100%; border-radius: 3px; }
  .nih { background: var(--acc); } .chx { background: #a98bff; }
  .pr em { font: normal .75rem "Geist Mono", monospace; color: var(--mut); }

  .lim { margin: 0; padding: 0; list-style: none; display: grid; gap: .9rem; color: var(--mut); line-height: 1.55; font-size: .95rem; }
  .lim li { padding-left: 1.2rem; position: relative; }
  .lim li::before { content: ""; position: absolute; left: 0; top: .6em; width: 6px; height: 6px; border-radius: 50%; background: var(--acc); }
  .faqw { display: grid; grid-template-columns: minmax(0,.8fr) minmax(0,1.4fr); gap: 4rem; align-items: start; }
  .sub2 { color: var(--mut); margin: -.8rem 0 1.2rem; line-height: 1.6; }
  .lnk2 { background: none; border: 0; padding: 0; color: var(--em); font: 500 .95rem Inter, sans-serif; cursor: pointer; }
  .fl details { border-top: 1px solid var(--line); padding: 1.05rem 0; }
  .fl details:last-child { border-bottom: 1px solid var(--line); }
  .fl summary { cursor: pointer; list-style: none; display: flex; justify-content: space-between; gap: 1rem; font: 500 1.02rem Sora, Inter, sans-serif; }
  .fl summary::-webkit-details-marker { display: none; }
  .fl summary::after { content: "+"; color: var(--em); font-size: 1.4rem; line-height: 1; transition: transform .25s; }
  .fl details[open] summary::after { transform: rotate(45deg); }
  .fl p { margin: .7rem 2rem 0 0; color: var(--mut); line-height: 1.65; font-size: .94rem; }

  .made { margin: 0 5vw; padding: 0 0 1.6rem; text-align: center; font-size: .68rem; color: var(--mut); opacity: .75; line-height: 1.6; }

  .cta2 { text-align: center; padding-bottom: 3rem; }
  .foot { margin: 2rem 5vw 0; padding: 2rem 0 2.2rem; border-top: 1px solid var(--line); display: grid; gap: .8rem; }
  .fcols { display: flex; justify-content: space-between; gap: 2rem; flex-wrap: wrap; }
  .fine { color: var(--mut); font-size: .8rem; line-height: 1.6; max-width: 46rem; margin: .5rem 0 0; }
  .disc { margin: 0; }
  .flinks { display: flex; gap: 1.3rem; flex-wrap: wrap; align-items: flex-start; font-size: .9rem; }
  .flinks a, .lnk { color: var(--mut); text-decoration: none; background: none; border: 0; padding: 0; font: inherit; cursor: pointer; }
  .flinks a:hover, .lnk:hover { color: var(--ink); }
  .acts { display: flex; gap: .6rem; align-items: center; }

  :global(.lp .rv) { opacity: 0; transform: translateY(18px); transition: opacity .7s ease var(--d, 0ms), transform .7s ease var(--d, 0ms); }
  :global(.lp .rv.in) { opacity: 1; transform: none; }

  /* ---- light: same glass as dark, tinted purple ---- */
  :global(:root[data-theme="light"]) .lp { --ink: #150f2e; --mut: #40395f; --acc: #5b34e6; --acc2: #a98bff; --em: #5b34e6; --line: rgba(91,52,230,.18); --glass: linear-gradient(150deg, rgba(139,108,255,.09), rgba(91,52,230,.04)); --shadow: 0 14px 36px rgba(60,30,160,.10); --glow: rgba(91,52,230,.3); --btnink: #fff; --track: rgba(91,52,230,.18); }
  :global(:root[data-theme="light"]) .lp::before { background: radial-gradient(900px 640px at 3% -10%, rgba(139,108,255,.30), transparent 62%), radial-gradient(820px 620px at 102% 108%, rgba(169,139,255,.36), transparent 62%), radial-gradient(520px 380px at 80% -4%, rgba(110,150,255,.14), transparent 60%), #f6f3ff; }
  :global(:root[data-theme="light"]) .lp .glass { box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,.7); }

  /* justified body text */
  .lead, .card ul, .card .body, .fcard p, .lim, .sub2, .fl p, .fine { text-align: justify; hyphens: auto; -webkit-hyphens: auto; }

  @media (max-width: 900px) {
    .hero, .cols, .faqw { grid-template-columns: 1fr; gap: 2rem; }
    .fgrid { grid-template-columns: repeat(2, minmax(0,1fr)); }
    .stats { grid-template-columns: repeat(2, minmax(0,1fr)); row-gap: 1.4rem; }
    .stat:nth-child(3) { border-left: 0; }
    .links { display: none; } .acts { margin-left: auto; }
  }
  @media (prefers-reduced-motion: reduce) { :global(.lp .rv) { opacity: 1; transform: none; transition: none; } }
</style>