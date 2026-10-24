<script>
  // @ts-nocheck
  import { onMount } from "svelte";
  let cv = $state();

  onMount(() => {
    const ctx = cv.getContext("2d");
    const rng = (a) => () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    const curve = (s, segs) => { const o = [s]; let p = s; for (const g of segs) { for (let k = 1; k <= 7; k++) { const t = k / 7, u = 1 - t; o.push([u*u*u*p[0] + 3*u*u*t*g[0] + 3*u*t*t*g[2] + t*t*t*g[4], u*u*u*p[1] + 3*u*u*t*g[1] + 3*u*t*t*g[3] + t*t*t*g[5]]); } p = [g[4], g[5]]; } o.pop(); return o; };
    const inside = (P, x, y) => { let c = false; for (let i = 0, j = P.length - 1; i < P.length; j = i++) { const a = P[i], b = P[j]; if ((a[1] > y) !== (b[1] > y) && x < (b[0]-a[0]) * (y-a[1]) / (b[1]-a[1]) + a[0]) c = !c; } return c; };
    function delaunay(P) {
      const n = P.length, V = P.concat([[-9,-9],[9,-9],[0,9]]);
      const mk = (a,b,c) => { const A=V[a],B=V[b],C=V[c],d=2*(A[0]*(B[1]-C[1])+B[0]*(C[1]-A[1])+C[0]*(A[1]-B[1])),a2=A[0]*A[0]+A[1]*A[1],b2=B[0]*B[0]+B[1]*B[1],c2=C[0]*C[0]+C[1]*C[1],x=(a2*(B[1]-C[1])+b2*(C[1]-A[1])+c2*(A[1]-B[1]))/d,y=(a2*(C[0]-B[0])+b2*(A[0]-C[0])+c2*(B[0]-A[0]))/d; return {v:[a,b,c],x,y,r:(A[0]-x)**2+(A[1]-y)**2}; };
      let T = [mk(n,n+1,n+2)];
      for (let i = 0; i < n; i++) { const [px,py] = V[i], keep = [], E = new Map();
        for (const t of T) { if ((px-t.x)**2+(py-t.y)**2 < t.r) { for (let k = 0; k < 3; k++) { const a=t.v[k],b=t.v[(k+1)%3],key=a<b?a+"_"+b:b+"_"+a; E.set(key, E.has(key)?null:[a,b]); } } else keep.push(t); }
        T = keep; for (const e of E.values()) if (e) T.push(mk(e[0],e[1],i)); }
      return T.filter(t => t.v.every(i => i < n)).map(t => t.v);
    }
    function mesh(poly, step, seed, hot) {
      const r = rng(seed), P = poly.map(p => [p[0]+(r()-.5)*.004, p[1]+(r()-.5)*.004]);
      const xs = poly.map(p=>p[0]), ys = poly.map(p=>p[1]), x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
      for (let y = y0+step/2; y < y1; y += step) for (let x = x0+step/2; x < x1; x += step) {
        const px = x+(r()-.5)*step*.8, py = y+(r()-.5)*step*.8;
        if (inside(poly,px,py) && poly.every(q => Math.hypot(q[0]-px,q[1]-py) > step*.42)) P.push([px,py]); }
      const byB = [[],[],[],[]], seen = new Set(), E = [];
      for (const v of delaunay(P)) {
        const cx = (P[v[0]][0]+P[v[1]][0]+P[v[2]][0])/3, cy = (P[v[0]][1]+P[v[1]][1]+P[v[2]][1])/3;
        if (!inside(poly,cx,cy)) continue;
        byB[Math.min(3, Math.floor((r()*.55+(1-(cx+.8)/1.6)*.45)*4))].push(v);
        for (let k = 0; k < 3; k++) { const a=v[k], b=v[(k+1)%3], key=a<b?a*1e4+b:b*1e4+a; if (!seen.has(key)) { seen.add(key); E.push([a,b]); } } }
      const path = new Path2D(); poly.forEach((p,i) => i ? path.lineTo(p[0],p[1]) : path.moveTo(p[0],p[1])); path.closePath();
      return { P, byB, E, path, hot, sz: P.map(() => .6+r()*.9) };
    }
    const LEFT = curve([-.2,-.6],[[-.4,-.62,-.58,-.3,-.68,.05],[-.78,.38,-.74,.62,-.52,.68],[-.3,.72,-.12,.62,-.09,.42],[-.07,.2,-.14,0,-.12,-.22],[-.11,-.42,-.13,-.55,-.2,-.6]]);
    const RIGHT = curve([.2,-.6],[[.4,-.62,.58,-.3,.68,.05],[.78,.38,.74,.62,.52,.68],[.34,.72,.18,.66,.12,.5],[.06,.4,.14,.26,.09,.12],[.07,-.1,.13,-.3,.12,-.42],[.11,-.5,.14,-.56,.2,-.6]]);
    const TRACH = [[-.05,-.98],[.05,-.98],[.05,-.34],[.15,-.12],[.11,-.09],[0,-.25],[-.11,-.09],[-.15,-.12],[-.05,-.34]];
    const meshes = [mesh(LEFT,.085,7,false), mesh(RIGHT,.085,19,true), mesh(TRACH,.05,3,false)];

    const TH = {
      dark:  { s0: "#fff", s1: "rgba(140,225,255,.95)", s2: "rgba(60,160,255,.28)", f: "60,150,255", fa: 1.5, e: "rgba(90,190,255,.28)", b: "61,155,255", r: "127,227,255", l: "110,190,255", n0: "#d6f6ff", n1: "#2a7bff", bd: "#bff2ff", bg: "#7fe3ff" },
      light: { s0: "#fff", s1: "rgba(140,100,255,.95)", s2: "rgba(140,100,255,.25)", f: "140,100,255", fa: 1.1, e: "rgba(120,80,255,.32)", b: "106,61,240", r: "106,61,240", l: "106,61,240", n0: "#e4d9ff", n1: "#5a2fe0", bd: "#6a3df0", bg: "#a98bff" },
    };
    let C = TH.dark;
    // The mesh lung is drawn once into an offscreen canvas, then the globe rotates around it.
    function renderLung(R, dpr) {
      const L = Math.ceil(2*R*dpr), u = R*.56, c = document.createElement("canvas"); c.width = c.height = L;
      const g = c.getContext("2d"); g.setTransform(u*dpr,0,0,u*dpr,L/2,L/2+.14*u*dpr);
      const sp = document.createElement("canvas"); sp.width = sp.height = 64;
      const s = sp.getContext("2d"), gr = s.createRadialGradient(32,32,0,32,32,32);
      gr.addColorStop(0,C.s0); gr.addColorStop(.16,C.s1); gr.addColorStop(.45,C.s2); gr.addColorStop(1,"rgba(0,0,0,0)");
      s.fillStyle = gr; s.fillRect(0,0,64,64);
      for (const m of meshes) {
        const P = m.P;
        for (let b = 0; b < 4; b++) { g.beginPath(); for (const v of m.byB[b]) { g.moveTo(...P[v[0]]); g.lineTo(...P[v[1]]); g.lineTo(...P[v[2]]); g.closePath(); } g.fillStyle = `rgba(${C.f},${(.04+.05*b)*C.fa})`; g.fill(); }
        if (m.hot) { g.save(); g.clip(m.path); const h = g.createRadialGradient(.42,.3,0,.42,.3,.2); h.addColorStop(0,"rgba(255,70,95,.85)"); h.addColorStop(.45,"rgba(255,170,70,.4)"); h.addColorStop(1,"rgba(255,210,90,0)"); g.fillStyle = h; g.fillRect(.1,0,.7,.7); g.restore(); }
        g.lineWidth = 1.1/u; g.strokeStyle = C.e; g.beginPath(); for (const e of m.E) { g.moveTo(...P[e[0]]); g.lineTo(...P[e[1]]); } g.stroke();
        for (let i = 0; i < P.length; i++) { const r = .022*m.sz[i]; g.globalAlpha = .35+.65*((i*7)%10)/10; g.drawImage(sp, P[i][0]-r*2.4, P[i][1]-r*2.4, r*4.8, r*4.8); }
        g.globalAlpha = 1;
      }
      return c;
    }

    // Globe: fibonacci node network + graticule + two orbit rings
    const N = 46, nodes = [], edges = [], seen = new Set(), ga = Math.PI*(3-Math.sqrt(5));
    for (let i = 0; i < N; i++) { const y = 1-(2*(i+.5))/N, r = Math.sqrt(1-y*y), a = i*ga; nodes.push([r*Math.cos(a), y, r*Math.sin(a), [1.8,2.6,2.6,3.6,4.6][(i*7)%5]]); }
    nodes.forEach((p,i) => { nodes.map((q,j) => [j,(p[0]-q[0])**2+(p[1]-q[1])**2+(p[2]-q[2])**2]).filter(d => d[0] !== i).sort((a,b) => a[1]-b[1]).slice(0,3).forEach(([j]) => { const k = i<j ? i+"-"+j : j+"-"+i; if (!seen.has(k)) { seen.add(k); edges.push([i,j]); } }); });
    const lines = [];
    for (const lat of [-60,-30,0,30,60]) { const f = lat*Math.PI/180; lines.push(Array.from({length:73}, (_,k) => { const t = k/72*6.2832; return [Math.cos(f)*Math.cos(t), Math.sin(f), Math.cos(f)*Math.sin(t)]; })); }
    for (let m = 0; m < 12; m++) { const l = m/12*6.2832; lines.push(Array.from({length:37}, (_,k) => { const f = (k/36*2-1)*1.5708; return [Math.cos(f)*Math.cos(l), Math.sin(f), Math.cos(f)*Math.sin(l)]; })); }
    const ring = (a,b) => Array.from({length:97}, (_,k) => { const t = k/96*6.2832, x = 1.17*Math.cos(t), z = 1.17*Math.sin(t), y1 = -z*Math.sin(a), z1 = z*Math.cos(a); return [x*Math.cos(b)-y1*Math.sin(b), x*Math.sin(b)+y1*Math.cos(b), z1]; });
    const rings = [ring(1.1,.5), ring(-.8,-.7)];

    let lung = null, W = 0, dpr = 1, vis = true, raf = 0;
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

    function draw(t) {
      const R = W*.4, c = W/2, ang = .6+t*.00018, tilt = .42+.08*Math.sin(t/4000), ca = Math.cos(ang), sa = Math.sin(ang), ct = Math.cos(tilt), st = Math.sin(tilt);
      const tr = (p) => { const X = p[0]*ca+p[2]*sa; let Z = -p[0]*sa+p[2]*ca; const Y = p[1]*ct-Z*st; Z = p[1]*st+Z*ct; return { x: c+X*R, y: c+Y*R, z: Z }; };
      const P = nodes.map(n => ({ ...tr(n), s: n[3] })), G = lines.map(l => l.map(tr)), RG = rings.map(l => l.map(tr));
      ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,W,W);
      const body = ctx.createRadialGradient(c,c,R*.5,c,c,R); body.addColorStop(0,`rgba(${C.b},0)`); body.addColorStop(1,`rgba(${C.b},.13)`);
      ctx.fillStyle = body; ctx.beginPath(); ctx.arc(c,c,R,0,7); ctx.fill();
      ctx.strokeStyle = `rgba(${C.r},.3)`; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(c,c,R,0,7); ctx.stroke();
      const half = (f) => {
        ctx.lineWidth = 1;
        const run = (pl, col, a0, a1) => { for (let k = 1; k < pl.length; k++) { const a = pl[k-1], b = pl[k], z = (a.z+b.z)/2; if (z >= 0 !== f) continue; ctx.strokeStyle = `rgba(${col},${a0+a1*(z+1)/2})`; ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke(); } };
        G.forEach(pl => run(pl,C.l,.04,.1));
        RG.forEach(pl => run(pl,C.r,.2,.5));
        for (const [i,j] of edges) { const a = P[i], b = P[j], z = (a.z+b.z)/2; if (z >= 0 !== f) continue; ctx.strokeStyle = `rgba(${C.l},${.1+.3*(z+1)/2})`; ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke(); }
        for (const p of P) { if (p.z >= 0 !== f) continue; const k = (p.z+1)/2, r = p.s*(.55+.8*k)*(W/900+.5); ctx.globalAlpha = .3+.7*k; const g = ctx.createRadialGradient(p.x-r*.3,p.y-r*.3,r*.1,p.x,p.y,r); g.addColorStop(0,C.n0); g.addColorStop(1,C.n1); ctx.fillStyle = g; ctx.beginPath(); ctx.arc(p.x,p.y,r,0,7); ctx.fill(); }
        ctx.globalAlpha = 1;
        RG.forEach((pl,i) => { const p = pl[Math.floor((t/45+i*48)%96)]; if (p.z >= 0 !== f) return; ctx.fillStyle = C.bd; ctx.shadowColor = C.bg; ctx.shadowBlur = 12; ctx.beginPath(); ctx.arc(p.x,p.y,3,0,7); ctx.fill(); ctx.shadowBlur = 0; });
      };
      half(false);
      if (lung) ctx.drawImage(lung, c-R, c-R, 2*R, 2*R);
      half(true);
    }

    const fit = () => { C = TH[document.documentElement.dataset.theme === "light" ? "light" : "dark"]; const r = cv.getBoundingClientRect(); dpr = Math.min(devicePixelRatio || 1, 2); W = r.width; cv.width = cv.height = W*dpr; lung = W > 0 ? renderLung(W*.4, dpr) : null; if (reduce) draw(0); };
    fit();
    const ro = new ResizeObserver(fit); ro.observe(cv);
    const io = new IntersectionObserver((e) => (vis = e[0].isIntersecting)); io.observe(cv);
    const mo = new MutationObserver(fit); mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    const loop = (t) => { if (vis && W > 0) draw(t); raf = requestAnimationFrame(loop); };
    if (!reduce) raf = requestAnimationFrame(loop);
    return () => { cancelAnimationFrame(raf); ro.disconnect(); io.disconnect(); mo.disconnect(); };
  });
</script>

<canvas bind:this={cv} aria-hidden="true"></canvas>

<style>
  canvas { display: block; width: 100%; aspect-ratio: 1; }
</style>
