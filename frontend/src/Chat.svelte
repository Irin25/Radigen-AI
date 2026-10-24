<script>
  import { tick } from "svelte";
  let { api, finding = null, scan = null } = $props();
  const HI = "Ask me about this scan, a finding, the heatmap or the model's limits. I answer from the knowledge base.";
  /** @type {{ me: boolean, t: string, src?: string[], err?: boolean }[]} */
  let msgs = $state([{ me: false, t: HI }]);
  let q = $state("");
  let busy = $state(false);
  /** @type {HTMLDivElement | undefined} */
  let box = $state();
  const hints = $derived(finding
    ? [`What is ${finding}?`, `What does ${finding} look like on X-ray?`, "How do I read the heatmap?"]
    : ["How do I read the heatmap?", "What are the model's limitations?", "What is cardiomegaly?"]);

  /** @param {string} [text] */
  async function send(text = q) {
    text = text.trim();
    if (!text || busy) return;
    const history = msgs.slice(1).filter((m) => !m.err).slice(-8).map((m) => ({ role: m.me ? "user" : "assistant", content: m.t }));
    msgs.push({ me: true, t: text });
    q = ""; busy = true;
    await tick();
    box?.scrollTo(0, box.scrollHeight);
    try {
      const res = await fetch(`${api}/api/chat`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: text, history, scan }) });
      const d = await res.json();
      msgs.push(res.ok ? { me: false, t: d.answer, src: d.sources } : { me: false, t: d.error ?? "Request failed", err: true });
    } catch {
      msgs.push({ me: false, t: "Can't reach the analysis server.", err: true });
    } finally {
      busy = false;
      await tick();
      box?.scrollTo(0, box.scrollHeight);
    }
  }
  function reset() {
    msgs = [{ me: false, t: HI }];
    q = "";
  }
</script>

<div class="wrap">
  <div class="log" bind:this={box} aria-live="polite">
    {#each msgs as m}
      <div class="m" class:me={m.me} class:err={m.err}>
        <p>{m.t}</p>
        {#if m.src?.length}<small>{m.src.join(" · ")}</small>{/if}
      </div>
    {/each}
    {#if busy}<div class="m"><p>Thinking…</p></div>{/if}
  </div>
  <div class="hints">{#each hints as h}<button onclick={() => send(h)} disabled={busy}>{h}</button>{/each}<button class="new" onclick={reset} disabled={busy || msgs.length < 2}>New chat</button></div>
  <div class="in">
    <input placeholder="Ask a follow-up…" maxlength="300" aria-label="Ask a question" bind:value={q} onkeydown={(e) => e.key === "Enter" && send()} />
    <button class="send" onclick={() => send()} disabled={busy || !q.trim()}>Send</button>
  </div>
  <p class="priv">Please don't type patient names or other identifying details.</p>
</div>

<style>
  .wrap { display: flex; flex-direction: column; gap: 10px; min-height: 0; flex: 1; }
  .log { flex: 1; min-height: 0; overflow-y: auto; scrollbar-width: none; display: flex; flex-direction: column; gap: 8px; }
  .log::-webkit-scrollbar { display: none; }
  .m { max-width: 94%; padding: 9px 12px; border-radius: 14px 14px 14px 4px; background: var(--tile); border: 1px solid var(--line); color: var(--ink); font-size: .85rem; line-height: 1.45; }
  .m p { margin: 0; }
  .m small { display: block; margin-top: 6px; color: var(--mut); font-size: .72rem; }
  .me { align-self: flex-end; border-radius: 14px 14px 4px 14px; background: var(--me, var(--soft)); color: var(--meink, var(--ink)); border-color: transparent; }
  .err { color: var(--danger); }
  .hints { display: flex; flex-wrap: wrap; gap: 6px; }
  .hints button { font: inherit; font-size: .74rem; color: var(--ink); background: var(--tile); border: 1px solid var(--line); border-radius: 999px; padding: 5px 11px; cursor: pointer; text-align: left; }
  .hints button:hover:not(:disabled) { border-color: var(--acc); }
  .in { display: flex; gap: 6px; }
  .in input { flex: 1; min-width: 0; font: inherit; font-size: .85rem; color: var(--ink); padding: 8px 12px; border-radius: 12px; background: var(--field); border: 1px solid var(--line); }
  .in input::placeholder { color: var(--ph); }
  .send { font: inherit; font-size: .85rem; padding: 8px 14px; border-radius: 12px; border: 0; cursor: pointer; color: var(--btnink); background: var(--btn); }
  .priv { margin: 0; font-size: .68rem; color: var(--mut); line-height: 1.3; }
  button:disabled { opacity: .5; cursor: not-allowed; }
  .hints .new { margin-left: auto; color: var(--mut); }
</style>