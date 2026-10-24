<script>
  let theme = $state(document.documentElement.dataset.theme === "light" ? "light" : "dark");
  function set(t) {
    theme = t;
    document.documentElement.dataset.theme = t;
    try { localStorage.setItem("rg", t); } catch {}
  }
</script>

<div class="theme" role="group" aria-label="Theme">
  <button aria-pressed={theme === "light"} onclick={() => set("light")}>Light</button>
  <button aria-pressed={theme === "dark"} onclick={() => set("dark")}>Dark</button>
</div>

<style>
  .theme { display: flex; padding: 3px; border-radius: 999px; border: 1px solid var(--line, rgba(140,200,255,.2)); background: rgba(128,128,170,.12); }
  button { font: 500 .8rem Geist, system-ui, sans-serif; border: 0; border-radius: 999px; padding: .35rem .85rem; background: none; color: var(--muted, var(--mut, #8ea4c8)); cursor: pointer; }
  button[aria-pressed="true"] { background: linear-gradient(150deg, #7fe3ff, #3d9bff); color: #04101e; }
  :global(:root[data-theme="light"]) button[aria-pressed="true"] { background: linear-gradient(150deg, #a98bff, #6a3df0); color: #fff; }
</style>
