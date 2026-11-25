// Lapilli: the Monitor's five sections (Overview, Requests, Slots, Hardware, Logs) and the Experts tab, for lapilli.js.
// Loaded just before lapilli.js; it only defines the Moniteur object: lapilli.js calls Moniteur.init() once its helpers
// ($, esc, nf, morceaux, ic...) exist, then Moniteur.dessiner(m) / Moniteur.dessinerExperts(m) with each /metrics reading.
// The expert map and the slots' figures come from the engine's console lines (STATS), as in upstream Basalt's dashboard
// (jesdga95, MIT license). Nothing is sent anywhere but to this server.
"use strict";

const Moniteur = (() => {
  const SOUS = ["apercu", "requetes", "creneaux", "materiel", "journal"];
  const NS = "http://www.w3.org/2000/svg";
  let sous = "apercu";
  let dernier = null;                                   // the last /metrics, for a redraw (a resize, a section opened)
  // the engine copies a share of the RAM experts it needs to GPU 0 over PCIe (the Flash-Next's pcie_frac): then the RAM
  // home is not only the CPU's, and a response's offloaded share (DONE) is GPU 1 and PCIe together
  const copiesPcie = (eng) => +((eng || (dernier && dernier.engine) || {}).pcie_frac) > 0;
  const creneaux = {rangs: null, enVol: false, erreur: null, quand: 0};
  const journal = {lignes: [], dernier: 0, enVol: false, suivre: true, filtre: "tout", quand: 0, erreur: null, rendus: 0};
  const carte = {mode: "vivant", geo: null, survol: -1, m: null, couleurs: null, image: false};

  // ------------------------------------------------------------------ small helpers
  const css = (nom) => getComputedStyle(document.documentElement).getPropertyValue(nom).trim();
  const pct = (x, d = 0) => (x == null || !Number.isFinite(+x) ? "–" : `${nf(x * 100, d)}%`);
  const pctFin = (x) => (x == null ? "–" : x > 0 && x < 0.005 ? "< 1%" : x < 1 && x > 0.995 ? "> 99%" : pct(x));
  const parS = (n) => (n == null ? "–" : n >= 10000 ? `${nf(n / 1000, 1)}k` : nf(n));
  const mo = (mib) => (mib == null ? "–" : go(mib * 1048576));
  const debit = (mbs) => (mbs == null ? "–" : mbs >= 1000 ? `${nf(mbs / 1024, 2)} GB/s` : `${nf(mbs, mbs < 10 ? 1 : 0)} MB/s`);
  const secondes = (s) => (s == null ? "–" : s < 10 ? `${nf(s, 2)} s` : s < 100 ? `${nf(s, 1)} s` : `${nf(s)} s`);
  const numero = (id) => { const m = /(\d+)$/.exec(String(id || "")); return m ? String(+m[1]) : String(id || "–"); };
  const POINTS = {"/v1/chat/completions": "Chat", "/v1/messages": "Messages", "/v1/responses": "Responses", "/v1/completions": "Completions"};
  const pointDe = (p) => (p ? POINTS[p] || p : "–");
  const FINS = {stop: ["badge--fini", "Done"], length: ["badge--arretee", "Max length"], cancelled: ["badge--arretee", "Stopped"],
                cancel: ["badge--arretee", "Stopped"], disconnect: ["badge--arretee", "Stopped"], error: ["badge--erreur", "Error"],
                tool_calls: ["badge--fini", "Tool call"]};
  // a response's reading speed, only for a message long enough to measure it (at least 256 tokens read)
  function lectureDe(r) {
    const lus = r.prompt_tokens == null ? 0 : Math.max(0, r.prompt_tokens - (r.reused || 0));
    return lus >= 256 && r.prompt_ms >= 20 ? lus / (r.prompt_ms / 1000) : null;
  }
  function el(nom, attrs, parent, ns) {
    const e = ns ? document.createElementNS(NS, nom) : document.createElement(nom);
    for (const [k, v] of Object.entries(attrs || {})) if (v != null) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  }
  const svg = (nom, attrs, parent) => el(nom, attrs, parent, true);
  // facts on one line, a pair each, wrapping between pairs: [term, value (html)]
  function faitsLigne(boite, lignes) {
    boite.innerHTML = lignes.filter((l) => l && l[1] != null && l[1] !== "")
      .map(([k, v]) => `<span class="fait"><span class="doux">${esc(k)}</span> <b>${v}</b></span>`).join("");
    boite.hidden = !boite.innerHTML;
  }
  // a definition list of facts: [term, value (text), value class]
  function faits(dl, lignes) {
    dl.innerHTML = lignes.filter((l) => l && l[1] != null && l[1] !== "")
      .map(([k, v, cls]) => `<dt>${esc(k)}</dt><dd${cls ? ` class="${cls}"` : ""}>${v}</dd>`).join("");
    dl.hidden = !dl.innerHTML;
  }

  // ------------------------------------------------------------------ the tooltip (one for every chart and the map)
  let bulle = null;
  function montrerBulle(x, y, titre, lignes) {
    if (!bulle) { bulle = el("div", {class: "infobulle", role: "tooltip"}, document.body); }
    bulle.textContent = "";
    if (titre) el("div", {class: "infobulle__titre"}, bulle).textContent = titre;
    for (const l of lignes) {
      const r = el("div", {class: "infobulle__ligne"}, bulle);
      if (l.c) el("span", {class: "infobulle__trait", style: `--c:${l.c}`}, r);
      el("b", null, r).textContent = l.valeur;
      if (l.nom) el("span", null, r).textContent = l.nom;
    }
    bulle.hidden = false;
    const w = bulle.offsetWidth, h = bulle.offsetHeight;
    let gx = x + 14, gy = y + 16;
    if (gx + w > innerWidth - 8) gx = Math.max(8, x - w - 14);
    if (gy + h > innerHeight - 8) gy = Math.max(8, y - h - 12);
    bulle.style.left = `${gx}px`;
    bulle.style.top = `${gy}px`;
  }
  const cacherBulle = () => { if (bulle) bulle.hidden = true; };

  // ------------------------------------------------------------------ charts: 60 s lines, stacked areas, columns (SVG)
  function sommet(max) {                                // a clean top for the axis
    if (!(max > 0)) return 1;
    const p = Math.pow(10, Math.floor(Math.log10(max)));
    for (const k of [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (k * p >= max * 1.0001) return k * p;
    return 10 * p;
  }
  const tic = (v) => (v >= 10000 ? `${nf(v / 1000)}k` : v >= 1000 ? `${nf(v / 1000, v % 1000 ? 1 : 0)}k` : nf(v, v > 0 && v < 10 && v % 1 ? 1 : 0));
  // el: a .graphe box. o: {series: [{nom, c (css color), v: [...]}], empile, max, plancher, unite, fmt, hauteur, aire}
  function graphe(boite, o) {
    boite._g = o;
    const W = boite.clientWidth;
    if (!W) return;
    const H = o.hauteur || 112;
    boite.style.height = `${H}px`;
    boite.textContent = "";
    const s = svg("svg", {width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.titre || ""}, boite);
    const n = Math.max(0, ...o.series.map((x) => (x.v || []).length));
    if (n < 2 && o.vide !== false) {
      boite.textContent = "";
      el("div", {class: "colonnes__vide doux"}, boite).textContent = o.vide || "The curve starts within a few seconds";
      return;
    }
    const val = (x, i) => { const v = (x.v || [])[i]; return v == null || !Number.isFinite(+v) ? null : Math.max(0, +v); };
    let haut = 0;
    for (let i = 0; i < n; i++) {
      if (o.empile) haut = Math.max(haut, o.series.reduce((a, x) => a + (val(x, i) || 0), 0));
      else for (const x of o.series) haut = Math.max(haut, val(x, i) || 0);
    }
    const max = o.max != null ? o.max : sommet(Math.max(haut, o.plancher || 1));
    const ticks = [0, max / 2, max];
    const etiquettes = ticks.map((t) => (o.fmtTic || tic)(t));
    const g = Math.max(...etiquettes.map((e) => e.length)) * 6.6 + 10;
    const x0 = Math.round(g), x1 = W - 4, y0 = 6, y1 = H - 20;
    const X = (i) => (n <= 1 ? x1 : x0 + ((x1 - x0) * i) / (n - 1));
    const Y = (v) => y1 - ((y1 - y0) * Math.min(v, max)) / max;
    ticks.forEach((t, k) => {
      const y = Math.round(Y(t)) + 0.5;
      svg("line", {x1: x0, x2: x1, y1: y, y2: y, class: k ? "grille" : "base"}, s);
      svg("text", {x: x0 - 8, y: y + 4, "text-anchor": "end", class: "tic"}, s).textContent = etiquettes[k];
    });
    svg("text", {x: x0, y: H - 4, class: "tic"}, s).textContent = `${nf(n)} s ago`;
    svg("text", {x: x1, y: H - 4, "text-anchor": "end", class: "tic"}, s).textContent = "now";
    if (n >= 2) {
      const bas = new Array(n).fill(0);
      for (const x of o.series) {
        const pts = [];
        let d = "", aire = "", ouvert = false, debut = 0;
        for (let i = 0; i < n; i++) {
          const v = val(x, i);
          if (v == null && !o.empile) { if (ouvert) { aire += `L${X(i - 1).toFixed(1)},${y1}L${X(debut).toFixed(1)},${y1}Z`; } ouvert = false; continue; }
          const haut2 = (o.empile ? bas[i] : 0) + (v || 0);
          const px = X(i).toFixed(1), py = Y(haut2).toFixed(1);
          d += `${ouvert ? "L" : "M"}${px},${py}`;
          if (!ouvert) { debut = i; aire += `M${px},${o.empile ? Y(bas[i]).toFixed(1) : y1}`; }
          aire += `L${px},${py}`;
          pts.push([i, haut2]);
          ouvert = true;
        }
        if (o.empile) {
          for (let i = n - 1; i >= 0; i--) aire += `L${X(i).toFixed(1)},${Y(bas[i]).toFixed(1)}`;
          aire += "Z";
          for (let i = 0; i < n; i++) bas[i] += val(x, i) || 0;
          svg("path", {d: aire, class: "bande", style: `fill:${x.c}`}, s);
        } else {
          if (ouvert) aire += `L${X(n - 1).toFixed(1)},${y1}L${X(debut).toFixed(1)},${y1}Z`;
          if (o.aire !== false) svg("path", {d: aire, class: "lavis", style: `fill:${x.c}`}, s);
        }
        svg("path", {d, class: "serie", style: `stroke:${x.c}`}, s);
        const fin = pts[pts.length - 1];
        if (fin && fin[0] === n - 1 && !o.empile) svg("circle", {cx: X(fin[0]), cy: Y(fin[1]), r: 3.5, class: "point", style: `fill:${x.c}`}, s);
      }
    }
    // the hover layer: a crosshair that snaps to the nearest second, one readout for every series
    const viseur = svg("line", {y1: y0, y2: y1, class: "viseur", visibility: "hidden"}, s);
    const zone = svg("rect", {x: x0, y: 0, width: Math.max(1, x1 - x0), height: H, fill: "transparent"}, s);
    const suivre = (ev) => {
      if (n < 1) return;
      const r = s.getBoundingClientRect();
      const i = Math.max(0, Math.min(n - 1, Math.round(((ev.clientX - r.left - x0) / Math.max(1, x1 - x0)) * (n - 1))));
      viseur.setAttribute("x1", X(i)); viseur.setAttribute("x2", X(i)); viseur.setAttribute("visibility", "visible");
      const fmt = o.fmt || ((v) => `${nf(v, v < 10 && v % 1 ? 1 : 0)}${o.unite ? ` ${o.unite}` : ""}`);
      const lignes = o.series.map((x) => ({c: x.c, valeur: val(x, i) == null ? "–" : fmt(val(x, i)), nom: x.nom}));
      if (o.empile && o.series.length > 1) lignes.push({valeur: fmt(o.series.reduce((a, x) => a + (val(x, i) || 0), 0)), nom: "in all"});
      const ago = n - 1 - i;
      montrerBulle(ev.clientX, ev.clientY, ago === 0 ? "Now" : `${nf(ago)} s ago`, lignes);
    };
    zone.addEventListener("pointermove", suivre);
    zone.addEventListener("pointerdown", suivre);
    zone.addEventListener("pointerleave", () => { viseur.setAttribute("visibility", "hidden"); cacherBulle(); });
  }
  // columns: one per item, stacked parts. o: {items: [{titre, parts: [v...]}], parts: [{nom, c}], fmt, hauteur}
  function colonnes(boite, o) {
    boite._c = o;
    const W = boite.clientWidth;
    if (!W) return;
    const H = o.hauteur || 168;
    boite.style.height = `${H}px`;
    boite.textContent = "";
    const s = svg("svg", {width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.titre || ""}, boite);
    const n = o.items.length;
    const haut = Math.max(0, ...o.items.map((it) => it.parts.reduce((a, v) => a + (v || 0), 0)));
    const max = o.max || sommet(Math.max(haut, 1));
    const ticks = [0, max / 2, max];
    const etiquettes = ticks.map((t) => (o.fmtTic || tic)(t));
    const x0 = Math.round(Math.max(...etiquettes.map((e) => e.length)) * 6.6 + 10), x1 = W - 2, y0 = 6, y1 = H - (o.basEtiquette ? 22 : 6);
    const Y = (v) => y1 - ((y1 - y0) * Math.min(v, max)) / max;
    ticks.forEach((t, k) => {
      const y = Math.round(Y(t)) + 0.5;
      svg("line", {x1: x0, x2: x1, y1: y, y2: y, class: k ? "grille" : "base"}, s);
      svg("text", {x: x0 - 8, y: y + 4, "text-anchor": "end", class: "tic"}, s).textContent = etiquettes[k];
    });
    if (!n) {
      boite.textContent = "";
      el("div", {class: "colonnes__vide doux"}, boite).textContent = o.vide || "After the first response";
      return;
    }
    const pas = (x1 - x0) / n;
    const large = Math.max(2, Math.min(24, pas - Math.max(2, pas * 0.28)));
    // the labels under the columns: every one when they fit, else one in 2, 5, 10... (no two labels touch)
    const plusLong = o.basEtiquette ? Math.max(1, ...o.items.map((it, i) => String(o.basEtiquette(it, i) || "").length)) : 1;
    const chaque = [1, 2, 5, 10, 20, 50].find((k) => k * pas >= plusLong * 6.6 + 6) || 100;
    o.items.forEach((it, i) => {
      const cx = x0 + pas * (i + 0.5);
      let bas = 0;
      const vis = it.parts.map((v, k) => [v || 0, k]).filter(([v]) => v > 0);
      vis.forEach(([v, k], j) => {
        const yb = Y(bas), yh = Y(bas + v);
        const top = j === vis.length - 1;
        const h = Math.max(top ? 1.5 : 0, yb - yh - (j ? 2 : 0));   // a 2 px surface gap between the parts
        const yy = yb - (j ? 2 : 0) - h;
        if (h > 0) {
          const r = top ? Math.min(4, large / 2, h) : 0;
          const xg = cx - large / 2;
          const dd = r ? `M${xg},${yy + h}V${yy + r}Q${xg},${yy} ${xg + r},${yy}H${xg + large - r}Q${xg + large},${yy} ${xg + large},${yy + r}V${yy + h}Z`
            : `M${xg},${yy + h}V${yy}H${xg + large}V${yy + h}Z`;
          svg("path", {d: dd, style: `fill:${o.parts[k].c}`, class: "colonne"}, s);
        }
        bas += v;
      });
      if (o.basEtiquette && i % chaque === 0) svg("text", {x: cx, y: H - 5, "text-anchor": "middle", class: "tic"}, s).textContent = o.basEtiquette(it, i);
      const zone = svg("rect", {x: x0 + pas * i, y: 0, width: pas, height: H, fill: "transparent", tabindex: "-1"}, s);
      const montrer = (ev) => montrerBulle(ev.clientX, ev.clientY, it.titre,
        [...o.parts.map((p, k) => ({c: p.c, valeur: (o.fmt || nf)(it.parts[k] || 0), nom: p.nom})), ...(it.extra || [])]);
      zone.addEventListener("pointermove", montrer);
      zone.addEventListener("pointerdown", montrer);
      zone.addEventListener("pointerleave", cacherBulle);
    });
  }
  // a thin bar split in parts: [[name, css color, share]]
  const partsBarre = (parts) => parts.filter((p) => p[2] > 0).map((p) => `<i style="width:${(p[2] * 100).toFixed(2)}%;background:${p[1]}"></i>`).join("");
  const legende = (parts, avecPart = true) => parts.map((p) => `<span class="cle" style="--c:${p[1]}">${esc(p[0])}${avecPart && p[2] != null ? ` <b>${pctFin(p[2])}</b>` : ""}</span>`).join("");
  const C = {gpu0: "var(--s-gpu0)", gpu1: "var(--s-gpu1)", ram: "var(--s-ram)", ssd: "var(--s-ssd)",
             lu: "var(--s-lu)", ecrit: "var(--s-ecrit)", reutilise: "var(--s-reutilise)", ecriture: "var(--accent-vif)", lecture: "var(--lecture)"};
  const MAISONS = [["GPU 0", C.gpu0], ["GPU 1", C.gpu1], ["RAM", C.ram], ["SSD", C.ssd]];

  // the experts' split of a finished response, from its counts: GPU 0, the other GPU (or a copy over PCIe), the CPU
  function partsReponse(r) {
    if (!r || !Number.isFinite(+r.hit_rate)) return null;
    const pair = Number.isFinite(+r.pcie_share) ? +r.pcie_share : 0, t = +r.hit_rate;
    return [["GPU 0", C.gpu0, t * (1 - pair)], ["GPU 1 or PCIe", C.gpu1, pair], ["CPU", C.ram, Math.max(0, (1 - t) * (1 - pair))]];
  }

  // ------------------------------------------------------------------ sections
  function init() {
    // the Overview's tiles
    const TUILES = [{id: "carte0", titre: "GPU 0", icone: "carte"}, {id: "carte1", titre: "GPU 1", icone: "carte"},
                    {id: "processeur", titre: "CPU", icone: "processeur"}, {id: "ram", titre: "RAM", icone: "memoire"},
                    {id: "disque", titre: "Disk", icone: "disque"}, {id: "experts", titre: "Experts", icone: "experts"}];
    $("tuiles").innerHTML = TUILES.map((t) => `
      <div class="carte tuile" id="tuile-${t.id}">
        <div class="tuile__tete">${ic(t.icone, "ic ic--p")}<span id="tt-${t.id}">${esc(t.titre)}</span>${t.id === "experts"
          ? `<a class="tuile__lien" href="#experts">Open the map</a>` : ""}</div>
        <div class="tuile__valeurs"><span class="tuile__valeur" id="tv-${t.id}">–</span></div>
        <div class="tuile__sous" id="ts-${t.id}"></div>
        ${t.id === "ram" ? `<div class="barre" id="tb-ram"><div class="barre__plein"></div></div>`
          : t.id === "experts" ? `<div class="repartition repartition--fine" id="tb-experts" aria-hidden="true"></div>`
          : `<div class="graphe graphe--mini" id="tg-${t.id}"></div>`}
      </div>`).join("");
    for (const b of document.querySelectorAll(".sous-onglet")) {
      b.onclick = () => montrerSousVue(b.dataset.sous, true);
      b.onkeydown = (e) => {                            // the arrow keys move between the sections (tabs pattern)
        const k = SOUS.indexOf(b.dataset.sous);
        const j = e.key === "ArrowRight" ? k + 1 : e.key === "ArrowLeft" ? k - 1 : e.key === "Home" ? 0 : e.key === "End" ? SOUS.length - 1 : null;
        if (j == null) return;
        e.preventDefault();
        const n = SOUS[(j + SOUS.length) % SOUS.length];
        montrerSousVue(n, true);
        $(`so-${n}`).focus();
      };
    }
    // the map's colors
    carte.mode = ["vivant", "place", "activite"].includes(memoire.lire("carte", "vivant")) ? memoire.lire("carte", "vivant") : "vivant";
    for (const b of document.querySelectorAll("#ex-mode button")) {
      b.setAttribute("aria-checked", String(b.dataset.v === carte.mode));
      b.onclick = () => {
        carte.mode = b.dataset.v;
        memoire.ecrire("carte", carte.mode);
        for (const x of document.querySelectorAll("#ex-mode button")) x.setAttribute("aria-checked", String(x === b));
        if (dernier) dessinerExperts(dernier);
      };
    }
    const toile = $("ex-canevas");
    toile.addEventListener("pointermove", (e) => survolCarte(e));
    toile.addEventListener("pointerdown", (e) => survolCarte(e));
    toile.addEventListener("pointerleave", () => { carte.survol = -1; peindreCarte(); cacherBulle(); });
    // the log
    for (const b of document.querySelectorAll("#journal-filtre button")) {
      b.onclick = () => {
        journal.filtre = b.dataset.v;
        for (const x of document.querySelectorAll("#journal-filtre button")) x.setAttribute("aria-checked", String(x === b));
        $("journal").dataset.filtre = journal.filtre;
        compterJournal();
        if (journal.suivre) allerEnBas();
      };
    }
    $("journal-suivre").onclick = () => poserSuivre($("journal-suivre").getAttribute("aria-checked") !== "true");
    $("journal-dernier").onclick = () => poserSuivre(true);
    $("journal").addEventListener("scroll", () => {
      const j = $("journal");
      const enBas = j.scrollHeight - j.scrollTop - j.clientHeight < 24;
      if (!enBas && journal.suivre && !journal.auto) poserSuivre(false);
      $("journal-dernier").hidden = enBas;
    });
    // a size change redraws the charts and the map at their new width
    if (window.ResizeObserver) {
      let attente = 0;
      new ResizeObserver(() => {
        cancelAnimationFrame(attente);
        attente = requestAnimationFrame(() => {
          for (const b of document.querySelectorAll(".graphe")) if (b._g && b.offsetParent) graphe(b, b._g);
          for (const b of document.querySelectorAll(".colonnes")) if (b._c && b.offsetParent) colonnes(b, b._c);
          if (carte.m && $("vue-experts").hidden === false) { carte.geo = null; peindreCarte(); }
        });
      }).observe(document.querySelector(".corps"));
    }
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { carte.couleurs = null; });
  }

  const sousVue = () => sous;
  function montrerSousVue(nom, parClic) {
    sous = SOUS.includes(nom) ? nom : "apercu";
    for (const b of document.querySelectorAll(".sous-onglet")) {
      const oui = b.dataset.sous === sous;
      b.setAttribute("aria-selected", String(oui));
      b.tabIndex = oui ? 0 : -1;
    }
    for (const n of SOUS) $(`sv-${n}`).hidden = n !== sous;
    cacherBulle();
    if (parClic) {
      const h = sous === "apercu" ? "#moniteur" : `#moniteur/${sous}`;
      if (location.hash !== h) history.replaceState(null, "", location.pathname + location.search + h);
      $("vue-moniteur").scrollTop = 0;
    }
    if (dernier) dessiner(dernier);
  }

  function dessiner(m) {
    dernier = m;
    if (sous === "apercu") apercu(m);
    else if (sous === "requetes") requetes(m);
    else if (sous === "creneaux") { lireCreneaux(); dessinerCreneaux(m); }
    else if (sous === "materiel") materiel(m);
    else if (sous === "journal") lireJournal();
  }

  // ---- Overview
  function apercu(m) {
    const live = m.live || {}, hw = m.hardware || {}, st = m.hardware_static || {}, eng = m.engine || {}, h = m.history || {};
    const req = m.requests || [], dern = req[0];
    const e = etatDe(live);
    // the model's status
    const libelles = {connexion: "Connecting…", chargement: "Loading the model", lecture: "Reading the message", reflexion: "Thinking",
                      ecriture: "Writing", repos: "Idle", attente: live.state === "unloaded" ? "Model unloaded" : live.state === "asleep" ? "Asleep" : "Waiting"};
    $("etat-libelle").textContent = libelles[e] || "Idle";
    for (const b of document.querySelectorAll("#etat-badges .badge")) b.classList.toggle("allume", b.dataset.b === e || (b.dataset.b === "attente" && live.queued > 0));
    let detail = [], part = 0, ton = null;
    if (e === "chargement") {
      part = ((live.load || {}).percent || 0) / 100; ton = "lecture";
      detail = [(live.load || {}).percent != null ? `${nf(live.load.percent)}%` : null];
    } else if (e === "lecture") {
      const lu = partLue(live); part = lu || 0; ton = "lecture";
      detail = [`${nf(live.prompt_tokens)} tokens`, lu != null ? `about ${nf(lu * 100)}%` : null,
                live.prefill_tok_s_mean ? `${nf(live.prefill_tok_s_mean)} tokens/s` : null];
    } else if (e === "reflexion" || e === "ecriture") {
      ton = e === "reflexion" ? "reflexion" : null;
      const max = enCours && reglages.longueur > 0 ? reglages.longueur : 0;
      part = max ? (live.generated || 0) / max : 0;
      detail = [`${nf(live.generated)} tokens written`, live.tok_s ? `${nf(live.tok_s, 1)} tokens/s` : null, live.elapsed_s != null ? `${nf(live.elapsed_s, 1)} s` : null];
    } else if (dern) {
      detail = [`Last response: ${nf(dern.output_tokens)} tokens${dern.decode_tok_s ? ` at ${nf(dern.decode_tok_s, 1)} tokens/s` : ""}`, ilYa(dern.time * 1000)];
    } else {
      detail = ["Waiting for a question"];
    }
    $("etat-detail").innerHTML = morceaux(detail) || "&nbsp;";
    const b = $("etat-barre");
    b.querySelector(".barre__plein").style.width = `${Math.max(0, Math.min(100, part * 100)).toFixed(1)}%`;
    if (ton) b.dataset.ton = ton; else delete b.dataset.ton;

    // writing, all slots
    const slots = live.slots || {total: 1, active: 0};
    const ecrit = live.state === "generating";
    const ecr = ecrit ? live.tok_s : dern ? dern.decode_tok_s : null;
    chiffre("vv-ecr", "vu-ecr", ecr == null ? null : nf(ecr, 1));
    $("vc-ecr-note").textContent = ecrit ? "now" : dern ? "last response" : "";
    $("vs-ecr").innerHTML = morceaux([
      slots.total > 1 ? `${nf(slots.total)} slots · ${nf(slots.active)} busy` : null,
      ecrit ? `${nf(live.generated)} tokens so far` : dern ? `${nf(dern.output_tokens)} tokens` : "no response yet",
      ecrit ? null : dern && dern.drafts_offered ? `drafts ${pct(dern.drafts_accepted / dern.drafts_offered)} accepted` : null]);
    graphe($("g-ecr"), {titre: "Writing speed over the last minute", series: [{nom: "Writing", c: C.ecriture, v: h.tok_s}], unite: "tokens/s", plancher: 10});
    // reading: this message, otherwise the last long one (a short one is read in an instant)
    // the same rule as a finished response's: a speed only for a message of 256 tokens or more (a short one is read at once)
    const longue = req.find((r) => lectureDe(r) != null);
    const lit = (live.state === "reading" || live.state === "generating") && live.prefill_tok_s_mean > 0 && (live.prompt_tokens || 0) >= 256;
    const lec = lit ? live.prefill_tok_s_mean : longue ? lectureDe(longue) : null;
    chiffre("vv-lec", "vu-lec", lec == null || lec <= 0 ? null : nf(lec));
    $("vc-lec-note").textContent = lit ? "this message" : longue ? "last long message" : "";
    $("vs-lec").innerHTML = lit ? morceaux([`${nf(live.prompt_tokens)} tokens`, live.state === "reading"
      ? (partLue(live) != null ? `about ${nf(partLue(live) * 100)}% read` : null) : "read"])
      : longue ? morceaux([`${nf(longue.prompt_tokens - (longue.reused || 0))} tokens in ${secondes(longue.prompt_ms / 1000)}`,
                          longue.prompt_tokens ? `${pctFin((longue.reused || 0) / longue.prompt_tokens)} cached` : null])
      : morceaux(["after a message of 256 tokens or more"]);
    graphe($("g-lec"), {titre: "Reading speed over the last minute", series: [{nom: "Reading", c: C.lecture, v: h.prefill_tok_s_mean}], unite: "tokens/s", plancher: 100});

    // tiles: the two GPUs, the CPU, RAM, disk, the experts
    const cartesHist = h.gpus || [];
    for (const k of [0, 1]) {
      const g = (hw.gpus || [])[k];
      $(`tuile-carte${k}`).hidden = !g && k === 1;
      if (!g) { tuile(`carte${k}`, null); $(`ts-carte${k}`).innerHTML = morceaux(["no reading"]); continue; }
      tuile(`carte${k}`, g.util == null ? null : nf(g.util), "%");
      $(`ts-carte${k}`).innerHTML = morceaux([g.mem_used != null ? `${go(g.mem_used)} of ${go(g.mem_total, 0)} GB` : null,
                                               g.temp != null ? `${nf(g.temp)} °C` : null, g.power != null ? `${nf(g.power)} W` : null]);
      const hg = cartesHist.find((x) => x.index === g.index) || cartesHist[k] || {};
      graphe($(`tg-carte${k}`), {titre: `GPU ${k} load`, series: [{nom: "Load", c: C.ecriture, v: hg.util}], max: 100, unite: "%", hauteur: 46, vide: false, mini: true, fmtTic: () => ""});
    }
    tuile("processeur", hw.cpu == null ? null : nf(hw.cpu), "%");
    $("ts-processeur").innerHTML = morceaux([nomProcesseur(st), st.cores ? `${nf(st.cores)} cores` : null, st.threads ? `${nf(st.threads)} threads` : null]);
    graphe($("tg-processeur"), {titre: "CPU load", series: [{nom: "Load", c: C.ecriture, v: h.cpu}], max: 100, unite: "%", hauteur: 46, vide: false, fmtTic: () => ""});
    const ramPart = hw.ram_total ? hw.ram_used / hw.ram_total : 0;
    tuile("ram", hw.ram_used == null ? null : go(hw.ram_used), hw.ram_total ? `of ${go(hw.ram_total)} GB` : "GB");
    const exRam = ((m.experts || {}).tiers || {}).ram;
    $("ts-ram").innerHTML = morceaux([hw.ram_total ? `${go(hw.ram_total - hw.ram_used)} GB free` : null,
                                      exRam && exRam.bytes ? `experts ${go(exRam.bytes)} GB` : null]);
    const rb = $("tb-ram");
    rb.querySelector(".barre__plein").style.width = `${(Math.min(1, ramPart) * 100).toFixed(1)}%`;
    if (ramPart > 0.92) rb.dataset.ton = "danger"; else delete rb.dataset.ton;
    const dl = hw.disk_read_mb, dw = hw.disk_write_mb;
    tuile("disque", dl == null ? null : dl >= 1000 ? nf(dl / 1024, 2) : nf(dl, dl < 10 ? 1 : 0), dl == null ? "" : dl >= 1000 ? "GB/s" : "MB/s");
    $("ts-disque").innerHTML = morceaux([dl == null ? "no reading" : "read", dw != null ? `write ${debit(dw)}` : null]);
    graphe($("tg-disque"), {titre: "Disk reads", series: [{nom: "Read", c: C.lecture, v: h.disk_read_mb}], unite: "MB/s", hauteur: 46, vide: false, plancher: 1, fmtTic: () => ""});
    // the experts: lookups a second now, and where they were served
    const ex = resumeExperts(m);
    $("tuile-experts").hidden = !ex;
    if (ex) {
      tuile("experts", ex.total > 0 ? parS(ex.total) : ex.carte ? "0" : null, ex.carte ? "lookups/s" : "");
      const parts = ex.parts || partsReponse(dern);
      $("ts-experts").innerHTML = parts ? morceaux(parts.filter((p) => p[2] > 0).map((p) => `${p[0]} ${pctFin(p[2])}`)) :
        morceaux([ex.carte ? "after the first response" : "no expert map from this engine"]);
      $("tb-experts").innerHTML = parts ? partsBarre(parts) : "";
    }

    // context
    const ctx = eng.max_context || 0;
    let utilise = 0;
    if (live.state === "reading" || live.state === "generating") utilise = (live.prompt_tokens || 0) + (live.generated || 0);
    else if (dern) utilise = (dern.prompt_tokens || 0) + (dern.output_tokens || 0);
    const frac = ctx ? Math.min(1, utilise / ctx) : 0;
    $("ctx-plein").setAttribute("stroke-dasharray", `${(235.6 * frac).toFixed(1)} 314.2`);
    $("ctx-plein").style.opacity = 235.6 * frac >= 3 ? "1" : "0";
    $("ctx-pct").textContent = `${frac > 0 && frac < 0.01 ? "< 1" : nf(frac * 100)}%`;
    $("ctx-sous").textContent = ctx ? `${kf(utilise)} of ${kf(ctx)} tokens` : "–";
    const kv = {int8: "8-bit", q4_0: "4-bit", fp16: "16-bit", f16: "16-bit", bf16: "16-bit", "fp8+fp4": "8- and 4-bit"}[eng.kv] || eng.kv;
    const conv = m.conversation || {};
    faits($("cache-faits"), [
      ["Context size", ctx ? `${nf(ctx)} tokens` : null],
      ["In use", ctx ? `${nf(utilise)} tokens${live.state === "reading" || live.state === "generating" ? "" : dern ? " (last response)" : ""}` : null],
      ["Context memory", kv || null],
      ["Kept conversations", conv.parked != null ? `${nf(conv.parked)}${conv.bytes ? ` · ${go(conv.bytes)} GB` : ""}` : null],
      ["Restored", conv.restored != null ? `${nf(conv.restored)}${conv.restored_ms ? ` · ${nf(conv.restored_ms, 1)} ms` : ""}` : null],
      ["Evicted", conv.evictions != null ? nf(conv.evictions) : null],
    ]);

    // last response
    if (dern) {
      const [cls, texte] = FINS[dern.finish] || ["", dern.finish || "–"];
      $("derniere-quand").innerHTML = morceaux([`#${numero(dern.id)}`, quand(dern.time * 1000, true)]);
      const lect = lectureDe(dern), lus = Math.max(0, (dern.prompt_tokens || 0) - (dern.reused || 0));
      faits($("derniere-faits"), [
        ["Message", `${nf(dern.prompt_tokens)} tokens`],
        ["Reused from the cache", dern.prompt_tokens ? `${nf(dern.reused || 0)} · ${pctFin((dern.reused || 0) / dern.prompt_tokens)}` : null],
        ["Read", lect != null ? `${nf(lus)} at ${nf(lect)} tokens/s` : `${nf(lus)} tokens`],
        ["First token", dern.ttft_s != null ? `after ${secondes(dern.ttft_s)}` : null],
        ["Written", `${nf(dern.output_tokens)}${dern.decode_tok_s ? ` at ${nf(dern.decode_tok_s, 1)} tokens/s` : ""}`],
        ["Drafts accepted", dern.drafts_offered ? `${nf(dern.drafts_accepted)} of ${nf(dern.drafts_offered)} · ${pct(dern.drafts_accepted / dern.drafts_offered)}` : null],
        ["Finish", `<span class="badge ${cls}">${esc(texte)}</span>`],
        ["Duration", secondes(dern.duration_s)],
      ]);
      const parts = partsReponse(dern);
      $("repartition").innerHTML = parts ? partsBarre(parts) : "";
      $("repartition-legende").innerHTML = parts ? legende(parts) : `<span class="doux">No expert counts for this response</span>`;
      $("derniere-part-note").textContent = dern.ssd_share ? `${pctFin(dern.ssd_share)} read from the SSD` : "";
    } else {
      $("derniere-quand").textContent = "";
      faits($("derniere-faits"), [["Status", "No response yet"]]);
      $("repartition").innerHTML = "";
      $("repartition-legende").innerHTML = `<span class="doux">After the first response</span>`;
      $("derniere-part-note").textContent = "";
    }

    // records and totals
    const best = m.best || {}, t = m.totals || {};
    const rec = [];
    if (best.decode) rec.push(`<div class="record">${ic("trophee", "ic ic--p")}<span>Fastest writing <b>${nf(best.decode.tok_s, 1)} tokens/s</b> <span class="doux">${morceauxEnLigne([`${nf(best.decode.output_tokens)} tokens`, `#${numero(best.decode.id)}`, quand(best.decode.time * 1000)])}</span></span></div>`);
    if (best.prefill) rec.push(`<div class="record">${ic("trophee", "ic ic--p")}<span>Fastest reading <b>${nf(best.prefill.tok_s)} tokens/s</b> <span class="doux">${morceauxEnLigne([`${nf(best.prefill.read_tokens)} tokens`, `#${numero(best.prefill.id)}`, quand(best.prefill.time * 1000)])}</span></span></div>`);
    $("records").innerHTML = rec.join("") || `<div class="record doux">${ic("trophee", "ic ic--p")}<span>Records appear after a few responses.</span></div>`;
    const lus = (t.prompt || 0) - (t.reused || 0);
    // since when: in the card's head (a server up since another day writes a long date, which would wrap a value)
    $("totaux-depuis").textContent = t.since ? `since ${quand(t.since * 1000)}` : "";
    faits($("totaux-faits"), t.requests ? [
      ["Responses", nf(t.requests)],
      ["Tokens read", `${nf(lus)}${t.prompt_ms > 0 && lus > 0 ? ` at ${nf(lus / (t.prompt_ms / 1000))} tokens/s` : ""}`],
      ["Reused from the cache", t.reused ? nf(t.reused) : "none"],
      ["Tokens written", nf(t.written)],
    ] : [["Responses", "none yet"]]);
  }
  // pieces inside a sentence (a record's details): the same unbreakable pieces, inline
  const morceauxEnLigne = (liste) => `<span class="morceaux-ligne">${morceaux(liste)}</span>`;
  function chiffre(idV, idU, texte) {
    $(idV).textContent = texte == null ? "–" : texte;
    $(idV).classList.toggle("est-vide", texte == null);
    $(idU).hidden = texte == null;
  }
  function tuile(id, texte, unite) {
    const vide = texte == null || texte === "–";
    $(`tv-${id}`).innerHTML = vide ? "–" : `${esc(texte)}${unite ? `<small>${esc(unite)}</small>` : ""}`;
    $(`tv-${id}`).classList.toggle("est-vide", vide);
  }

  // ---- Requests
  function requetes(m) {
    const live = m.live || {}, req = m.requests || [], t = m.totals || {};
    const q = live.queue || [];
    $("attente-nombre").textContent = q.length ? `${nf(q.length)} waiting` : "";
    $("file-attente").innerHTML = q.length ? `<ul class="file">${q.map((w) => `<li>${morceaux([w.api === "anthropic" ? "Anthropic API" : "OpenAI API",
      w.model || null, w.prompt_tokens != null ? `${nf(w.prompt_tokens)} tokens` : null, w.waited_s != null ? `waiting for ${secondes(w.waited_s)}` : null])}</li>`).join("")}</ul>`
      : `<p class="doux vide-ligne">No request is waiting.${(live.slots || {}).total > 1 ? ` ${nf(live.slots.total)} slots serve requests side by side.` : " One slot: a request waits while another runs."}</p>`;
    $("requetes-nombre").textContent = t.requests ? `${nf(Math.min(req.length, 12))} of ${nf(t.requests)} since ${quand(t.since * 1000)}` : "";
    const corps = $("requetes-corps");
    $("table-requetes").dataset.unCreneau = String(((live.slots || {}).total || 1) <= 1 ? 1 : 0);
    corps.innerHTML = !req.length ? `<tr><td colspan="12" class="doux">No responses yet</td></tr>` : req.slice(0, 12).map((r) => {
      const [cls, texte] = FINS[r.finish] || ["", r.finish || "–"];
      const lect = lectureDe(r);
      return `<tr><td>${esc(quand(r.time * 1000, true))}</td><td class="doux c-num">${esc(numero(r.id))}</td><td class="c-creneau">${r.slot == null ? "–" : nf(r.slot + 1)}</td>` +
        `<td class="c-point">${esc(pointDe(r.path))}</td><td><span class="badge ${cls}">${esc(texte)}</span></td>` +
        `<td class="num c-msg">${nf(r.prompt_tokens)}</td><td class="num c-reut">${nf(r.reused)}</td><td class="num c-ecrit">${nf(r.output_tokens)}</td>` +
        `<td class="num c-lec">${lect == null ? "–" : nf(lect)}</td><td class="num">${nf(r.decode_tok_s, 1)}</td>` +
        `<td class="num c-pj">${r.ttft_s == null ? "–" : secondes(r.ttft_s)}</td><td class="num c-duree">${secondes(r.duration_s)}</td></tr>`;
    }).join("");
    // tokens per response: reused, read, written, oldest on the left
    const parts = [{nom: "Reused", c: C.reutilise}, {nom: "Read", c: C.lu}, {nom: "Written", c: C.ecrit}];
    $("jetons-legende").innerHTML = parts.map((p) => `<span class="cle" style="--c:${p.c}">${esc(p.nom)}</span>`).join("");
    const items = req.slice(0, 12).reverse().map((r) => ({
      titre: `#${numero(r.id)} · ${quand(r.time * 1000, true)}`,
      parts: [r.reused || 0, Math.max(0, (r.prompt_tokens || 0) - (r.reused || 0)), r.output_tokens || 0],
      extra: [{valeur: secondes(r.duration_s), nom: "duration"}]}));
    colonnes($("g-jetons"), {titre: "Tokens per response", items, parts, fmt: (v) => `${nf(v)} tokens`, hauteur: 180, vide: "No responses yet",
                              basEtiquette: (it) => it.titre.split(" · ")[0]});
  }

  // ---- Slots (/slots, read only while this section is open)
  async function lireCreneaux() {
    if (creneaux.enVol || Date.now() - creneaux.quand < 400) return;
    creneaux.enVol = true;
    try {
      const r = await fetch("slots", {cache: "no-store", headers: entetes()});
      if (!r.ok) throw await refus(r);
      creneaux.rangs = await r.json();
      creneaux.erreur = null;
    } catch (e) {
      creneaux.erreur = String((e && e.message) || e);
    }
    creneaux.quand = Date.now();
    creneaux.enVol = false;
    if (sous === "creneaux" && dernier) dessinerCreneaux(dernier);
  }
  function dessinerCreneaux(m) {
    const boite = $("creneaux");
    if (!creneaux.rangs) {
      boite.innerHTML = `<div class="carte vide doux">${creneaux.erreur ? `The slots could not be read: ${esc(creneaux.erreur)}` : "Reading the slots…"}</div>`;
      return;
    }
    if (!Array.isArray(creneaux.rangs) || !creneaux.rangs.length) {
      boite.innerHTML = `<div class="carte vide doux">The model is not loaded: no slot to show.</div>`;
      return;
    }
    const n = creneaux.rangs.length;
    boite.dataset.n = String(Math.min(n, 3));
    // one card per slot, built once, then filled at each reading
    if (boite.children.length !== n || !boite.querySelector(".creneau")) {
      boite.innerHTML = creneaux.rangs.map((_, i) => `
        <div class="carte creneau" id="cr-${i}">
          <div class="carte__tete"><span class="carte__titre">Slot ${i + 1}</span><span class="badge" id="cr-etat-${i}"></span></div>
          <div class="vitesse__chiffre"><span class="vitesse__valeur vitesse__valeur--petit" id="cr-v-${i}">–</span><span class="vitesse__unite" id="cr-u-${i}">tokens/s</span></div>
          <div class="vitesse__sous" id="cr-s-${i}"></div>
          <div class="ligne-barre"><span>Context used</span><span class="doux" id="cr-ctx-t-${i}"></span></div>
          <div class="barre" id="cr-ctx-${i}"><div class="barre__plein"></div></div>
          <div class="faits-ligne" id="cr-f-${i}"></div>
          <div class="creneau__bloc"><div class="ligne-barre"><span>Drafts accepted, by position</span><span class="doux" id="cr-acc-t-${i}"></span></div>
            <div class="colonnes colonnes--petites" id="cr-acc-${i}"></div></div>
          <div class="creneau__bloc"><div class="ligne-barre"><span>Tokens per window</span><span class="doux" id="cr-dl-t-${i}"></span></div>
            <div class="graphe graphe--petit" id="cr-dl-${i}"></div></div>
          <div class="creneau__bloc"><div class="ligne-barre"><span>Its experts</span><span class="doux" id="cr-ex-t-${i}"></span></div>
            <div class="repartition" id="cr-ex-${i}" aria-hidden="true"></div>
            <div class="repartition-legende" id="cr-ex-l-${i}"></div></div>
        </div>`).join("");
    }
    // /slots and /metrics are read one after the other: at the end of a response the slots may still say "decoding" while
    // /metrics, which the header's pill shows, says nothing runs; then the slots show their last figures, idle - never a
    // "Writing" slot beside an "Idle" pill (whichever of the two readings is newer)
    const live = (m && m.live) || {};
    const fini = live.state === "idle" && !((live.slots || {}).active > 0);
    creneaux.rangs.forEach((s0, i) => {
      const s = fini && (s0.is_processing || s0.state === "decoding" || s0.state === "prefill") ? {...s0, state: "idle", is_processing: false} : s0;
      const etat = s.state === "decoding" ? ["badge--ecriture allume", "Writing"] : s.state === "prefill" ? ["badge--lecture allume", "Reading"] : ["", "Idle"];
      const be = $(`cr-etat-${i}`);
      be.className = `badge ${etat[0]}`;
      be.textContent = etat[1];
      if (s.state === "decoding") { chiffre(`cr-v-${i}`, `cr-u-${i}`, nf(s.tok_s, 1)); }
      else if (s.state === "prefill") {
        const p = s.prompt_tokens ? (s.prefill_done || 0) / s.prompt_tokens : null;
        $(`cr-v-${i}`).textContent = p == null ? "Reading" : `${nf(p * 100)}%`;
        $(`cr-u-${i}`).hidden = false;
        $(`cr-u-${i}`).textContent = "read";
      } else { $(`cr-v-${i}`).textContent = "Idle"; $(`cr-u-${i}`).hidden = true; }
      if (s.state === "decoding") $(`cr-u-${i}`).textContent = "tokens/s";
      $(`cr-s-${i}`).innerHTML = morceaux(s.is_processing ? [pointDe(s.path), s.api === "anthropic" ? "Anthropic API" : s.api ? "OpenAI API" : null,
        s.started ? `started ${quand(s.started * 1000, true)}` : null] : ["figures of its last request"]);
      const ctxPart = s.n_ctx ? Math.min(1, (s.ctx_used || 0) / s.n_ctx) : 0;
      $(`cr-ctx-${i}`).querySelector(".barre__plein").style.width = `${(ctxPart * 100).toFixed(1)}%`;
      $(`cr-ctx-t-${i}`).textContent = s.n_ctx ? `${kf(s.ctx_used || 0)} of ${kf(s.n_ctx)} tokens` : "–";
      faitsLigne($(`cr-f-${i}`), s.is_processing ? [
        ["Message", `${nf(s.prompt_tokens)} tokens${s.reused ? ` · ${nf(s.reused)} reused` : ""}`],
        ["Written", `${nf(s.generated)}${s.max_tokens ? ` of ${nf(s.max_tokens)} at most` : ""}`],
        ["First token", s.ttft_s != null ? `after ${secondes(s.ttft_s)}` : s.state === "prefill" ? "after the reading" : null],
      ] : [
        ["Drafts accepted", s.drafts_offered ? `${nf(s.drafts_accepted)} of ${nf(s.drafts_offered)} · ${pct(s.drafts_accepted / s.drafts_offered)}` : "none"],
      ]);
      // acceptance by draft position: the share of drafts kept at position 1, 2, ...
      const acc = s.accept_pos || [];
      $(`cr-acc-t-${i}`).textContent = s.drafts_offered ? `${pct(s.drafts_accepted / s.drafts_offered)} in all` : "no drafts";
      colonnes($(`cr-acc-${i}`), {titre: "Drafts accepted by position", parts: [{nom: "accepted", c: C.ecriture}], max: 1, hauteur: 92,
        vide: "No drafts in its last request",
        fmt: (v) => pct(v), fmtTic: (v) => `${nf(v * 100)}%`, basEtiquette: (it, k) => String(k + 1),
        items: acc.map((v, k) => ({titre: `Draft position ${k + 1}`, parts: [v]}))});
      const dl = (s.accept_hist || []).filter((v) => v != null);
      $(`cr-dl-t-${i}`).textContent = s.draft_len != null ? `${nf(s.draft_len, 2)} on average` : dl.length ? `${nf(dl.reduce((a, v) => a + v, 0) / dl.length, 2)} on average` : "–";
      graphe($(`cr-dl-${i}`), {titre: "Tokens written per verify window, by second", series: [{nom: "Tokens per window", c: C.ecriture, v: s.accept_hist || []}],
        hauteur: 72, plancher: 2, unite: "tokens"});
      const hh = s.hit;
      const parts = hh ? [["GPU 0", C.gpu0, hh.vram], ["GPU 1 or PCIe", C.gpu1, hh.peer], ["CPU", C.ram, hh.cpu]] : null;
      $(`cr-ex-${i}`).innerHTML = parts ? partsBarre(parts) : "";
      $(`cr-ex-l-${i}`).innerHTML = parts ? legende(parts) : `<span class="doux">No expert counts yet</span>`;
      // a rate only while the slot works (the server keeps its last one 2.5 s after the end: never "Idle" beside 18k/s)
      $(`cr-ex-t-${i}`).textContent = hh && s.is_processing && hh.lookups_s ? `${parS(hh.lookups_s)} lookups/s` : hh ? "idle" : "";
    });
  }

  // ---- Hardware
  function materiel(m) {
    const hw = m.hardware || {}, st = m.hardware_static || {}, h = m.history || {}, ex = (m.experts || {}).tiers || {};
    const gpus = hw.gpus || [];
    const boite = $("materiel-cartes");
    if (boite.children.length !== gpus.length || !boite.querySelector(".gpu")) {
      boite.innerHTML = gpus.map((g, k) => `
        <div class="carte gpu" id="gp-${k}">
          <div class="carte__tete"><span class="carte__titre">GPU ${esc(g.index)}</span><span class="doux petit" id="gp-nom-${k}"></span></div>
          <div class="gpu__chiffres">
            <div class="gpu__chiffre"><span class="tuile__mini">Load</span><span class="gpu__valeur" id="gp-util-${k}">–</span></div>
            <div class="gpu__chiffre"><span class="tuile__mini">Power</span><span class="gpu__valeur" id="gp-w-${k}">–</span></div>
            <div class="gpu__chiffre"><span class="tuile__mini">Temperature</span><span class="gpu__valeur" id="gp-t-${k}">–</span></div>
            <div class="gpu__chiffre"><span class="tuile__mini">PCIe link</span><span class="gpu__valeur" id="gp-l-${k}">–</span></div>
          </div>
          <div class="ligne-barre"><span>Memory</span><span class="doux" id="gp-mem-t-${k}"></span></div>
          <div class="repartition repartition--memoire" id="gp-mem-${k}" aria-hidden="true"></div>
          <div class="repartition-legende" id="gp-mem-l-${k}"></div>
          <div class="gpu__graphes">
            <div><div class="graphe__titre">Load <span class="doux">%</span></div><div class="graphe graphe--petit" id="gp-g-util-${k}"></div></div>
            <div><div class="graphe__titre">Power <span class="doux">W</span></div><div class="graphe graphe--petit" id="gp-g-w-${k}"></div></div>
            <div><div class="graphe__titre">Temperature <span class="doux">°C</span></div><div class="graphe graphe--petit" id="gp-g-t-${k}"></div></div>
            <div><div class="graphe__titre">PCIe traffic <span class="doux">MB/s</span><span class="legende legende--titre" id="gp-pcie-l-${k}"></span></div><div class="graphe graphe--petit" id="gp-g-p-${k}"></div></div>
          </div>
        </div>`).join("") || `<div class="carte vide doux">No GPU reading.</div>`;
    }
    gpus.forEach((g, k) => {
      const nom = ((st.gpus || []).find((x) => x.index === g.index) || {}).name;
      $(`gp-nom-${k}`).textContent = nom || "";
      $(`gp-util-${k}`).innerHTML = g.util == null ? "–" : `${nf(g.util)}<small>%</small>`;
      $(`gp-w-${k}`).innerHTML = g.power == null ? "–" : `${nf(g.power)}<small>${g.power_limit ? `of ${nf(g.power_limit)} W` : "W"}</small>`;
      $(`gp-t-${k}`).innerHTML = g.temp == null ? "–" : `${nf(g.temp)}<small>°C</small>`;
      $(`gp-l-${k}`).innerHTML = g.pcie_gen ? `Gen ${nf(g.pcie_gen)}<small>×${nf(g.pcie_width)}${g.pcie_gen_max && g.pcie_gen_max !== g.pcie_gen ? ` · Gen ${nf(g.pcie_gen_max)} max` : ""}</small>` : "–";
      // its memory: the experts it keeps (the engine's own figures), the rest (weights, context, work buffers)
      const tier = k === 0 ? ex.main : k === 1 ? ex.peer : null;
      const used = g.mem_used, total = g.mem_total;
      $(`gp-mem-t-${k}`).textContent = used != null && total ? `${go(used)} of ${go(total, 0)} GB` : "–";
      const parts = used != null && total ? (tier && tier.bytes ? [["Experts", k === 0 ? C.gpu0 : C.gpu1, Math.min(used, tier.bytes) / total],
        ["Weights, context and work buffers", C.reutilise, Math.max(0, used - tier.bytes) / total]] : [["In use", k === 0 ? C.gpu0 : C.gpu1, used / total]]) : [];
      $(`gp-mem-${k}`).innerHTML = partsBarre(parts);
      $(`gp-mem-l-${k}`).innerHTML = parts.map((p) => `<span class="cle" style="--c:${p[1]}">${esc(p[0])} <b>${go(p[2] * total)} GB</b></span>`).join("");
      const hg = (h.gpus || []).find((x) => x.index === g.index) || {};
      graphe($(`gp-g-util-${k}`), {titre: `GPU ${g.index} load`, series: [{nom: "Load", c: C.ecriture, v: hg.util}], max: 100, unite: "%", hauteur: 84});
      graphe($(`gp-g-w-${k}`), {titre: `GPU ${g.index} power`, series: [{nom: "Power", c: C.ecriture, v: hg.power}], max: g.power_limit || null, plancher: 50, unite: "W", hauteur: 84});
      graphe($(`gp-g-t-${k}`), {titre: `GPU ${g.index} temperature`, series: [{nom: "Temperature", c: C.ecriture, v: hg.temp}], max: 100, unite: "°C", hauteur: 84});
      const pcie = [{nom: "To the GPU", c: C.lu, v: hg.pcie_rx_mb}, {nom: "From the GPU", c: C.ecrit, v: hg.pcie_tx_mb}];
      $(`gp-pcie-l-${k}`).innerHTML = pcie.map((p) => `<span class="cle cle--trait" style="--c:${p.c}">${esc(p.nom)}</span>`).join("");
      graphe($(`gp-g-p-${k}`), {titre: `GPU ${g.index} PCIe traffic`, series: pcie, plancher: 10, unite: "MB/s", hauteur: 84, aire: false});
    });
    // the machine: CPU, RAM, disk
    const mach = $("materiel-machine");
    if (!mach.querySelector(".machine")) {
      mach.innerHTML = `
        <div class="carte machine"><div class="carte__tete"><span class="carte__titre">CPU</span><span class="doux petit" id="mc-nom"></span></div>
          <div class="vitesse__chiffre"><span class="vitesse__valeur vitesse__valeur--petit" id="mc-v">–</span><span class="vitesse__unite" id="mc-u">% load</span></div>
          <div class="vitesse__sous" id="mc-s"></div><div class="graphe graphe--petit" id="mc-g"></div></div>
        <div class="carte machine"><div class="carte__tete"><span class="carte__titre">RAM</span><span class="doux petit" id="mr-note"></span></div>
          <div class="vitesse__chiffre"><span class="vitesse__valeur vitesse__valeur--petit" id="mr-v">–</span><span class="vitesse__unite" id="mr-u"></span></div>
          <div class="repartition repartition--memoire" id="mr-b" aria-hidden="true"></div><div class="repartition-legende" id="mr-l"></div></div>
        <div class="carte machine"><div class="carte__tete"><span class="carte__titre">Disk</span><span class="legende" id="md-l"></span></div>
          <div class="vitesse__chiffre"><span class="vitesse__valeur vitesse__valeur--petit" id="md-v">–</span><span class="vitesse__unite" id="md-u">MB/s read</span></div>
          <div class="vitesse__sous" id="md-s"></div><div class="graphe graphe--petit" id="md-g"></div></div>`;
    }
    $("mc-nom").textContent = nomProcesseur(st) || "";
    chiffre("mc-v", "mc-u", hw.cpu == null ? null : nf(hw.cpu));
    $("mc-s").innerHTML = morceaux([st.cores ? `${nf(st.cores)} cores` : null, st.threads ? `${nf(st.threads)} threads` : null]);
    graphe($("mc-g"), {titre: "CPU load", series: [{nom: "Load", c: C.ecriture, v: h.cpu}], max: 100, unite: "%", hauteur: 84});
    const exRam = ex.ram && ex.ram.bytes ? Math.min(ex.ram.bytes, hw.ram_used || 0) : 0;
    if (hw.ram_total) {
      $("mr-v").textContent = go(hw.ram_used);
      $("mr-u").hidden = false;
      $("mr-u").textContent = `of ${go(hw.ram_total)} GB in use`;
      const parts = [["Experts", C.ram, exRam / hw.ram_total], ["Everything else", C.reutilise, Math.max(0, hw.ram_used - exRam) / hw.ram_total]].filter((p) => p[2] > 0);
      $("mr-b").innerHTML = partsBarre(parts);
      $("mr-l").innerHTML = parts.map((p) => `<span class="cle" style="--c:${p[1]}">${esc(p[0])} <b>${go(p[2] * hw.ram_total)} GB</b></span>`).join("") +
        `<span class="cle" style="--c:var(--surface-3)">Free <b>${go(hw.ram_total - hw.ram_used)} GB</b></span>`;
      $("mr-note").textContent = exRam ? "experts kept in RAM are pinned" : "";
    } else { chiffre("mr-v", "mr-u", null); $("mr-b").innerHTML = ""; $("mr-l").innerHTML = ""; }
    const dl = hw.disk_read_mb, dw = hw.disk_write_mb;
    chiffre("md-v", "md-u", dl == null ? null : dl >= 1000 ? nf(dl / 1024, 2) : nf(dl, dl < 10 ? 1 : 0));
    $("md-u").textContent = dl != null && dl >= 1000 ? "GB/s read" : "MB/s read";
    $("md-s").innerHTML = morceaux([dw != null ? `write ${debit(dw)}` : null]);
    const disque = [{nom: "Read", c: C.lu, v: h.disk_read_mb}, {nom: "Write", c: C.ecrit, v: h.disk_write_mb}];
    $("md-l").innerHTML = disque.map((p) => `<span class="cle cle--trait" style="--c:${p.c}">${esc(p.nom)}</span>`).join("");
    graphe($("md-g"), {titre: "Disk traffic", series: disque, plancher: 1, unite: "MB/s", hauteur: 84, aire: false});
    $("materiel-pied").innerHTML = morceaux([st.driver ? `NVIDIA driver ${st.driver}` : null, st.cuda ? `CUDA ${st.cuda}` : null, "60 seconds, one reading a second"]);
  }

  // ---- Logs (/api/logs, read only while this section is open)
  async function lireJournal() {
    if (journal.enVol || Date.now() - journal.quand < 900) return;
    journal.enVol = true;
    try {
      const r = await fetch(journal.dernier ? `api/logs?after=${journal.dernier}` : "api/logs", {cache: "no-store", headers: entetes()});
      if (!r.ok) throw await refus(r);
      const j = await r.json();
      const neuves = (j.lines || []).filter((l) => l.id > journal.dernier);
      if (j.last != null && j.last < journal.dernier) { journal.lignes = []; journal.rendus = 0; journal.jour = null; $("journal").textContent = ""; }   // a restarted server
      journal.lignes.push(...neuves);
      if (neuves.length) journal.dernier = neuves[neuves.length - 1].id;
      else if (j.last != null && j.last > journal.dernier && !(j.lines || []).length) journal.dernier = j.last;
      journal.erreur = null;
    } catch (e) {
      journal.erreur = String((e && e.message) || e);
    }
    journal.quand = Date.now();
    journal.enVol = false;
    rendreJournal();
  }
  const NIVEAUX = {warn: "warning", err: "error"};
  function rendreJournal() {
    const boite = $("journal");
    if (journal.lignes.length > 3000) {                // the newest 3,000 lines stay
      const coupe = journal.lignes.length - 2500;
      journal.lignes = journal.lignes.slice(coupe);
      boite.textContent = "";
      journal.rendus = 0;
      journal.jour = null;
    }
    const frag = document.createDocumentFragment();
    for (let i = journal.rendus; i < journal.lignes.length; i++) {
      const l = journal.lignes[i];
      // a time alone does not say which day: each day's lines open with its name (a server up for days keeps yesterday's)
      const jour = debutJour(l.t * 1000);
      if (jour !== journal.jour) {
        journal.jour = jour;
        el("div", {class: "journal__jour"}, frag).textContent = jourDe(l.t * 1000);
      }
      const d = el("div", {class: `journal__ligne${NIVEAUX[l.lvl] ? ` ${l.lvl}` : ""}`});
      el("span", {class: "journal__heure"}, d).textContent = heure(l.t * 1000, true);
      el("span", {class: "journal__src"}, d).textContent = l.src === "eng" ? "engine" : "server";
      const msg = el("span", {class: "journal__msg"}, d);
      let texte = l.msg;
      if (NIVEAUX[l.lvl]) {
        el("b", {class: "journal__niveau"}, msg).textContent = `${NIVEAUX[l.lvl]} `;
        texte = texte.replace(new RegExp(`^${NIVEAUX[l.lvl]}:\\s*`, "i"), "");
      }
      msg.appendChild(document.createTextNode(texte));
      frag.appendChild(d);
    }
    const ajout = journal.lignes.length - journal.rendus;
    journal.rendus = journal.lignes.length;
    if (ajout > 0) boite.appendChild(frag);
    let vide = boite.querySelector(".journal__vide");
    const montres = compterJournal();
    if (!montres) {
      if (!vide) vide = el("div", {class: "journal__vide doux"}, boite);
      vide.textContent = journal.erreur ? `The log could not be read: ${journal.erreur}` : journal.filtre === "alertes" ? "No warnings or errors." : "No lines yet.";
    } else if (vide) vide.remove();
    if (journal.suivre && ajout > 0) allerEnBas();
  }
  function compterJournal() {
    const alertes = journal.lignes.filter((l) => NIVEAUX[l.lvl]).length;
    const montres = journal.filtre === "alertes" ? alertes : journal.lignes.length;
    $("journal-nombre").textContent = journal.lignes.length ? `${nf(montres)} line${montres === 1 ? "" : "s"}${journal.filtre === "tout" && alertes ? ` · ${nf(alertes)} warning${alertes === 1 ? "" : "s"} or error${alertes === 1 ? "" : "s"}` : ""}` : "";
    return montres;
  }
  function allerEnBas() {
    const j = $("journal");
    journal.auto = true;
    j.scrollTop = j.scrollHeight;
    requestAnimationFrame(() => { journal.auto = false; });
    $("journal-dernier").hidden = true;
  }
  function poserSuivre(oui) {
    journal.suivre = oui;
    $("journal-suivre").setAttribute("aria-checked", String(oui));
    if (oui) allerEnBas();
  }

  // ------------------------------------------------------------------ the Experts tab
  // what the server says of the model's experts: the map (all the experts, or a sample), the tiers, their lookups now
  function resumeExperts(m) {
    const em = m.experts_map, ti = m.experts, eng = m.engine || {}, vit = m.experts_live;
    if (!em && !vit && !ti && !eng.expert_slots) return null;
    const out = {carte: !!(em || vit), total: 0, parts: null, homes: [0, 0, 0, 0], maintenant: [0, 0, 0, 0]};
    if (!(em && em.tier) && vit) {                    // the server's summary (/metrics without ?experts=full)
      out.homes = (vit.homes || [0, 0, 0, 0]).slice(0, 4);
      out.maintenant = (vit.lookups_s || [0, 0, 0, 0]).slice(0, 4);
    } else if (em && em.tier) {
      const echelle = em.total && em.tier.length ? em.total / em.tier.length : 1;
      em.tier.forEach((h, i) => {
        const k = Math.max(0, Math.min(3, h | 0));
        out.homes[k] += echelle;
        out.maintenant[k] += (em.routed[i] || 0) * echelle;
      });
    }
    out.total = out.maintenant.reduce((a, v) => a + v, 0);
    if (out.total > 0) {
      const t = out.total;
      // the map's RAM home: the CPU's, or also GPU 0's copies over PCIe (the Flash-Next): then it is "RAM", not "CPU"
      out.parts = [["GPU 0", C.gpu0, out.maintenant[0] / t], ["GPU 1", C.gpu1, out.maintenant[1] / t],
                   [copiesPcie(eng) ? "RAM" : "CPU", C.ram, (out.maintenant[2] + out.maintenant[3]) / t]];
    }
    return out;
  }
  const aDesExperts = (m) => !!(m && (m.experts_map || m.experts_live || m.experts || (m.engine || {}).expert_slots));

  function dessinerExperts(m) {
    dernier = m;
    const em = m.experts_map, ti = m.experts, eng = m.engine || {}, dern = (m.requests || [])[0];
    const ex = resumeExperts(m) || {homes: [0, 0, 0, 0], maintenant: [0, 0, 0, 0], total: 0};
    const couches = ti ? ti.layers : null, parCouche = ti ? ti.per_layer : null;
    const tous = em ? em.total : ti ? ti.layers * ti.per_layer : eng.expert_total;
    $("ex-resume").innerHTML = morceaux([
      couches && parCouche ? `${nf(couches)} layers × ${nf(parCouche)} experts` : null,
      tous ? `${nf(tous)} experts in all` : null,
      eng.expert_slots ? `${nf(eng.expert_slots)} places on GPU 0` : null]);
    const actif = ex.total > 0;
    const be = $("ex-etat");
    be.className = `badge ${actif ? "badge--ecriture allume" : "allume"}`;
    // the header's word: "Thinking" while the model thinks (its tokens are written too), "Writing" otherwise
    const verbe = ((m.live || {}).state === "generating" && (m.live || {}).phase === "thinking") ? "Thinking" : "Writing";
    be.textContent = actif ? `${verbe}: live lookups` : "No lookups right now";
    // the figures: lookups now, and who served them (now, otherwise the last response)
    const parts = ex.parts || partsReponse(dern);
    const source = ex.parts ? "now" : parts ? "last response" : null;
    const kpi = (nom, valeur, unite, sous2, c) => `<div class="kpi">${c ? `<span class="kpi__cle" style="--c:${c}"></span>` : ""}<span class="kpi__nom">${esc(nom)}</span>` +
      `<span class="kpi__valeur${valeur === "–" ? " est-vide" : ""}">${esc(valeur)}${unite && valeur !== "–" ? `<small>${esc(unite)}</small>` : ""}</span><span class="kpi__sous">${sous2 || "&nbsp;"}</span></div>`;
    const homes = ex.homes;
    // now (the map): GPU 1 = the experts it keeps, RAM = the CPU or a PCIe copy; the last response (DONE): GPU 1 and PCIe
    // together, the CPU alone
    const pcie = copiesPcie(eng);
    const nom1 = pcie && source === "last response" ? "Served by GPU 1 or PCIe" : "Served by GPU 1";
    const nom2 = pcie && source === "now" ? "Served from RAM" : "Computed by the CPU";
    $("ex-kpis").innerHTML =
      kpi("Lookups", ex.carte ? (actif ? parS(ex.total) : "0") : "–", ex.carte ? "/s" : "", morceaux([ex.carte ? (actif ? `while ${verbe.toLowerCase()}` : "they are counted while writing") : "no map from this engine"])) +
      kpi("Served by GPU 0", parts ? pctFin(parts[0][2]) : "–", "", morceaux([source, ex.carte ? `${nf(Math.round(homes[0]))} experts there` : null]), C.gpu0) +
      kpi(nom1, parts ? pctFin(parts[1][2]) : "–", "", morceaux([source, ex.carte ? `${nf(Math.round(homes[1]))} experts there` : null]), C.gpu1) +
      kpi(nom2, parts ? pctFin(parts[2][2]) : "–", "", morceaux([source, ex.carte ? `${nf(Math.round(homes[2] + homes[3]))} only in RAM${homes[3] ? " or on the SSD" : ""}` : null]), C.ram);

    // the map
    carte.m = em ? {em, couches: couches || 1, parCouche: parCouche || em.total} : null;
    const boite = $("ex-carte");
    boite.classList.toggle("ex-carte--vide", !em);
    $("ex-mode").hidden = !em && !m.experts_live;
    if (!em) {
      $("ex-canevas").hidden = true;
      let v = boite.querySelector(".ex-carte__vide");
      if (!v) v = el("div", {class: "ex-carte__vide doux"}, boite);
      // a summary without the arrays: the tab has just opened, its reading with the map is on its way
      v.textContent = m.experts_live ? "Reading the map…"
        : ti || eng.expert_slots ? "This engine does not send its expert map: the map appears with an engine that does (Lapilli with DeepSeek)."
        : "No experts: this model is not a mixture of experts.";
      $("ex-legende").innerHTML = "";
      $("ex-note").textContent = "";
    } else {
      $("ex-canevas").hidden = false;
      const v = boite.querySelector(".ex-carte__vide");
      if (v) v.remove();
      legendeCarte(ex);
      peindreCarte();
    }

    // lookups a second over the last minute, by where the expert is kept
    const h = m.history || {};
    const flux = [{nom: "GPU 0", c: C.gpu0, v: h.experts_vram_s}, {nom: "GPU 1", c: C.gpu1, v: h.experts_peer_s},
                  {nom: copiesPcie(eng) ? "RAM: CPU or PCIe" : "CPU, from RAM", c: C.ram, v: h.experts_ram_s}];
    if ((h.experts_file_s || []).some((v) => v > 0)) flux.push({nom: "CPU, from the SSD", c: C.ssd, v: h.experts_file_s});
    $("ex-flux-legende").innerHTML = em ? flux.map((f) => `<span class="cle" style="--c:${f.c}">${esc(f.nom)}</span>`).join("") : "";
    $("ex-flux-note").textContent = em ? "while writing, one reading a second" : "";
    graphe($("g-experts"), {titre: "Expert lookups a second over the last minute", series: flux, empile: true, unite: "/s", plancher: 100,
                            vide: em ? null : "No lookups from this engine: it does not send its expert map",
                            fmt: (v) => `${parS(Math.round(v))}/s`, hauteur: 168});

    // where the experts are kept: the engine's tiers
    occupation(ti, ex, tous, eng);

    // by layer: who served each layer's recent lookups (or, with none lately, where its experts are kept)
    parCouches(em, couches, parCouche);
    // the busiest experts
    plusUtilises(em, parCouche);
    aide(ti, em, eng);
  }

  function legendeCarte(ex) {
    const total = ex.homes.reduce((a, v) => a + v, 0) || 1;
    if (carte.mode === "activite") {
      $("ex-legende").innerHTML = `<span class="ex-echelle"><span class="doux">quiet</span><i class="ex-echelle__barre"></i><span class="doux">busy</span></span>`;
      $("ex-note").textContent = "Activity: how many lookups each expert had in the last seconds, wherever it is kept. Hover an expert for its figures.";
    } else {
      $("ex-legende").innerHTML = MAISONS.map((mm, k) => (ex.homes[k] > 0 || k < 3 ? `<span class="cle" style="--c:${mm[1]}">${k === 2 ? (copiesPcie() ? "RAM (the CPU, or GPU 0 over PCIe)" : "RAM (the CPU computes it)") : esc(mm[0])} <b>${nf(Math.round(ex.homes[k]))}</b> <span class="doux">${pctFin(ex.homes[k] / total)}</span></span>` : "")).join("");
      $("ex-note").textContent = carte.mode === "place"
        ? "Placement: where each expert is kept now. Hover an expert for its figures."
        : "One hexagon per expert, one block per layer. The color says where it is kept, the brightness how much it was used in the last seconds. Hover an expert for its figures.";
    }
  }

  // the map's colors, read from the page's tokens (both themes), and the ramp of the activity view
  function couleursCarte() {
    const theme = document.documentElement.dataset.theme;
    if (carte.couleurs && carte.couleurs.theme === theme) return carte.couleurs;
    const rgb = (hex) => { const h = hex.replace("#", ""); const n = parseInt(h.length === 3 ? h.split("").map((x) => x + x).join("") : h, 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; };
    const melange = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
    const fond = rgb(css("--surface") || "#ffffff");
    const maisons = ["--s-gpu0", "--s-gpu1", "--s-ram", "--s-ssd"].map((v) => rgb(css(v) || "#888888"));
    const NIV = 24;
    // live: the color of its home, faded toward the surface when quiet (22% of the color) up to full when busy
    const vivant = maisons.map((c) => Array.from({length: NIV}, (_, k) => `rgb(${melange(fond, c, 0.22 + (0.78 * k) / (NIV - 1)).join(",")})`));
    const pleine = maisons.map((c) => `rgb(${c.join(",")})`);
    const rampe = (css("--s-rampe") || "").split(",").map((x) => rgb(x.trim()));
    const activite = Array.from({length: NIV}, (_, k) => {
      const t = (k / (NIV - 1)) * (rampe.length - 1), i = Math.min(rampe.length - 2, Math.floor(t));
      return `rgb(${melange(rampe[i], rampe[i + 1], t - i).join(",")})`;
    });
    carte.couleurs = {theme, vivant, pleine, activite, NIV, encre: css("--encre"), surface: css("--surface"), doux: css("--encre-douce")};
    return carte.couleurs;
  }

  // where each cell goes: one block of hexagons per layer (all the experts), or one honeycomb for a sample
  function geometrie(W) {
    const {em, couches, parCouche} = carte.m;
    const n = em.tier.length, total = em.total || n;
    const complet = n === total && couches > 1 && parCouche * couches === total;
    const cx = new Float32Array(n), cy = new Float32Array(n);
    let H, rayon, blocs = [];
    if (complet) {
      const cols = Math.max(4, Math.round(Math.sqrt(parCouche * 4 / 3))), rangs = Math.ceil(parCouche / cols);
      const ecart = Math.max(10, Math.min(18, W / 64)), titre = 16;
      let par = 1, w = 0;
      for (let b = Math.min(couches, 12); b >= 1; b--) {
        w = (W - (b - 1) * ecart) / (b * (cols + 0.5));
        // big layers (the Flash-Next's 512 experts) take smaller hexagons, or the map would be taller than a few screens
        if (w >= (W < 560 ? (parCouche > 256 ? 4.6 : 5.6) : (parCouche > 256 ? 6.2 : 7.4))) { par = b; break; }
        par = 1;
      }
      w = Math.min(14, (W - (par - 1) * ecart) / (par * (cols + 0.5)));
      rayon = w / Math.sqrt(3);
      const largeurBloc = (cols + 0.5) * w, hauteurBloc = (rangs - 1) * 1.5 * rayon + 2 * rayon;
      const pasX = largeurBloc + ecart + ((W - par * largeurBloc - (par - 1) * ecart) / Math.max(1, par - 1 || 1)) * (par > 1 ? 1 : 0);
      const pasY = hauteurBloc + titre + ecart;
      for (let l = 0; l < couches; l++) {
        const bx = (l % par) * (par > 1 ? pasX : 0), by = Math.floor(l / par) * pasY;
        blocs.push({x: bx, y: by, l});
        for (let e = 0; e < parCouche; e++) {
          const i = l * parCouche + e, r = Math.floor(e / cols), c = e % cols;
          cx[i] = bx + w / 2 + c * w + (r % 2 ? w / 2 : 0);
          cy[i] = by + titre + rayon + r * 1.5 * rayon;
        }
      }
      H = Math.ceil(Math.floor((couches - 1) / par) * pasY + titre + hauteurBloc + 2);
      return {W, H, cx, cy, rayon, w, blocs, complet, cols};
    }
    // a sample: one honeycomb, as wide as the box
    const cols = Math.max(8, Math.round(Math.sqrt(n * (W / 360)))), rangs = Math.ceil(n / cols);
    const w = Math.min(18, W / (cols + 0.5));
    rayon = w / Math.sqrt(3);
    for (let i = 0; i < n; i++) {
      const r = Math.floor(i / cols), c = i % cols;
      cx[i] = w / 2 + c * w + (r % 2 ? w / 2 : 0);
      cy[i] = rayon + r * 1.5 * rayon;
    }
    H = Math.ceil((rangs - 1) * 1.5 * rayon + 2 * rayon + 2);
    return {W, H, cx, cy, rayon, w, blocs, complet};
  }
  // the (layer, expert) of the sample's cell i: the same pairs as the engine's (core/expert_map.hpp)
  function paireDe(i) {
    const {em, parCouche} = carte.m;
    const n = em.tier.length, total = em.total || n;
    const p = n === total ? i : Math.floor(((2 * i + 1) * total) / (2 * n));
    return [Math.floor(p / parCouche), p % parCouche];
  }
  function peindreCarte() {
    if (!carte.m) return;
    const toile = $("ex-canevas"), boite = $("ex-carte");
    const W = Math.floor(boite.clientWidth);
    if (!W) return;
    if (!carte.geo || carte.geo.W !== W || carte.geo.n !== carte.m.em.tier.length) { carte.geo = geometrie(W); carte.geo.n = carte.m.em.tier.length; }
    const g = carte.geo, k = couleursCarte(), em = carte.m.em;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    if (toile.width !== Math.round(W * dpr) || toile.height !== Math.round(g.H * dpr)) {
      toile.width = Math.round(W * dpr); toile.height = Math.round(g.H * dpr);
      toile.style.height = `${g.H}px`;
    }
    const cx = toile.getContext("2d");
    cx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cx.clearRect(0, 0, W, g.H);
    // the layers' names
    if (g.complet) {
      cx.fillStyle = k.doux;
      cx.font = `500 11px ${css("--police") || "sans-serif"}`;
      cx.textBaseline = "alphabetic";
      for (const b of g.blocs) cx.fillText(`Layer ${b.l}`, b.x + 1, b.y + 11);
    }
    // the cells, grouped by color: one path per color
    const n = em.tier.length;
    let maxR = 1;
    for (let i = 0; i < n; i++) if (em.recent[i] > maxR) maxR = em.recent[i];
    const lmax = Math.log1p(maxR);
    const groupes = new Map();
    for (let i = 0; i < n; i++) {
      const h = Math.max(0, Math.min(3, em.tier[i] | 0));
      const v = em.recent[i] > 0 ? Math.log1p(em.recent[i]) / lmax : 0;
      const niv = Math.max(0, Math.min(k.NIV - 1, Math.round(v * (k.NIV - 1))));
      const coul = carte.mode === "place" ? k.pleine[h] : carte.mode === "activite" ? k.activite[niv] : k.vivant[h][niv];
      let liste = groupes.get(coul);
      if (!liste) { liste = []; groupes.set(coul, liste); }
      liste.push(i);
    }
    const r = g.rayon * (g.w > 9 ? 0.9 : 0.86), dx = r * Math.sqrt(3) / 2;
    for (const [coul, liste] of groupes) {
      cx.beginPath();
      for (const i of liste) hexagone(cx, g.cx[i], g.cy[i], r, dx);
      cx.fillStyle = coul;
      cx.fill();
    }
    if (carte.survol >= 0 && carte.survol < n) {
      cx.beginPath();
      hexagone(cx, g.cx[carte.survol], g.cy[carte.survol], g.rayon * 1.08, g.rayon * 1.08 * Math.sqrt(3) / 2, true);
      cx.lineWidth = 2;
      cx.strokeStyle = k.encre;
      cx.stroke();
    }
    const ex = resumeExperts(dernier) || {homes: [0, 0, 0, 0]};
    toile.setAttribute("role", "img");
    toile.setAttribute("aria-label", `Map of the ${nf(em.total)} experts: ${nf(Math.round(ex.homes[0]))} kept on GPU 0, ${nf(Math.round(ex.homes[1]))} on GPU 1, ` +
      `${nf(Math.round(ex.homes[2]))} only in RAM${ex.homes[3] ? `, ${nf(Math.round(ex.homes[3]))} on the SSD` : ""}. The busiest experts are listed below the map.`);
  }
  // one hexagon in the current path; `fermer` for a stroke only: fill() closes each shape itself, and closePath in a path
  // of 24,576 hexagons cost Chrome 128 ms a draw (profiled on the Flash-Next's size)
  function hexagone(cx, x, y, r, dx, fermer = false) {
    cx.moveTo(x, y - r);
    cx.lineTo(x + dx, y - r / 2);
    cx.lineTo(x + dx, y + r / 2);
    cx.lineTo(x, y + r);
    cx.lineTo(x - dx, y + r / 2);
    cx.lineTo(x - dx, y - r / 2);
    if (fermer) cx.closePath();
  }
  function survolCarte(e) {
    if (!carte.m || !carte.geo) return;
    const b = $("ex-canevas").getBoundingClientRect();
    const x = e.clientX - b.left, y = e.clientY - b.top, g = carte.geo;
    let meilleur = -1, d2 = g.rayon * g.rayon * 1.3;
    for (let i = 0; i < g.cx.length; i++) {
      const ddx = g.cx[i] - x, ddy = g.cy[i] - y, d = ddx * ddx + ddy * ddy;
      if (d < d2) { d2 = d; meilleur = i; }
    }
    if (meilleur !== carte.survol) { carte.survol = meilleur; peindreCarte(); }
    if (meilleur < 0) { cacherBulle(); return; }
    const em = carte.m.em, [l, ex] = paireDe(meilleur), h = Math.max(0, Math.min(3, em.tier[meilleur] | 0));
    const ou = ["GPU 0", "GPU 1", copiesPcie() ? "RAM: the CPU computes it, or GPU 0 copies it over PCIe" : "RAM: the CPU computes it",
                "the SSD: the CPU reads and computes it"][h];
    montrerBulle(e.clientX, e.clientY, `Layer ${l} · expert ${ex}`, [
      {c: MAISONS[h][1], valeur: `Kept on ${ou}`},
      {valeur: `${nf(em.routed[meilleur] || 0, (em.routed[meilleur] || 0) % 1 ? 1 : 0)}`, nom: "lookups a second now"},
      {valeur: nf(em.recent[meilleur] || 0, 1), nom: "recent lookups (fading)"}]);
  }

  function occupation(ti, ex, tous, eng = {}) {
    const boite = $("ex-occupation");
    if (!ti) {
      boite.innerHTML = `<p class="doux">This engine does not say where it keeps its experts.</p>`;
      $("ex-occ-note").textContent = "";
      return;
    }
    const t = ti.tiers, total = tous || ti.layers * ti.per_layer || 1;
    // the Flash-Next's engine copies a share of the RAM experts it needs to GPU 0 over PCIe (pcie_frac of the misses)
    const ramDit = +eng.pcie_frac > 0 ? "the CPU computes them, GPU 0 copies some over PCIe" : "the CPU computes them";
    const parCouche = (tier) => (tier.per_layer == null ? null : `${nf(tier.per_layer)} per layer`);
    const ligne = (nom, c, tier, detail, mot = "experts") => `<div class="occ"><div class="occ__tete"><span class="cle" style="--c:${c}">${esc(nom)}</span>` +
      `<span class="occ__chiffre"><b>${nf(tier.experts)}</b> ${mot}<span class="doux"> · ${go(tier.bytes)} GB</span></span></div>` +
      `<div class="barre barre--fine"><div class="barre__plein" style="width:${Math.min(100, (100 * tier.experts) / total).toFixed(1)}%;background:${c}"></div></div>` +
      `<div class="occ__detail doux">${morceaux(detail)}</div></div>`;
    boite.innerHTML =
      // with copies, a place swapped for another expert is free at once and counts again once the copy has arrived
      (ti.pool
        ? ligne("GPU 0", C.gpu0, t.main, ["one pool for every layer", "the experts used most first", ex.homes[0] ? `${nf(Math.round(ex.homes[0]))} there now` : null], "places")
        : ligne("GPU 0", C.gpu0, t.main, ti.main_copies
          ? [parCouche(t.main), "copies of the experts used most, swapped while it writes", `${nf(Math.round(ex.homes[0]))} ready now`]
          : [parCouche(t.main), "kept only there"], ti.main_copies ? "places" : "experts")) +
      (t.peer.experts || !ti.pool ? ligne("GPU 1", C.gpu1, t.peer, [parCouche(t.peer), "it computes them and sends the result over PCIe"]) : "") +
      ligne("RAM", C.ram, t.ram, [parCouche(t.ram), ti.main_copies && t.main.experts ? `${nf(Math.round(ex.homes[2]))} only there, ${ramDit}` : ramDit]) +
      (t.file.experts ? ligne("SSD", C.ssd, t.file, [parCouche(t.file), "read from the disk when needed"]) : "");
    $("ex-occ-note").textContent = t.file.experts ? "" : "none on the SSD: every expert fits in memory";
  }

  function parCouches(em, couches, parCouche) {
    const boite = $("ex-couches");
    if (!em || !couches || !parCouche || em.tier.length !== couches * parCouche) {
      boite.innerHTML = `<p class="doux">${em ? "The map holds a sample of the experts: no figures by layer." : "No expert map from this engine."}</p>`;
      boite.classList.remove("colonnes");
      boite.style.height = "";
      boite._c = null;
      $("ex-couches-legende").innerHTML = "";
      $("ex-couches-note").textContent = "";
      return;
    }
    const recentes = new Array(couches).fill(0).map(() => [0, 0, 0]);
    const gardes = new Array(couches).fill(0).map(() => [0, 0, 0]);
    for (let i = 0; i < em.tier.length; i++) {
      const l = Math.floor(i / parCouche), h = Math.min(2, Math.max(0, em.tier[i] | 0));
      recentes[l][h] += em.recent[i] || 0;
      gardes[l][h] += 1;
    }
    const somme = recentes.reduce((a, r) => a + r[0] + r[1] + r[2], 0);
    const vivant = somme > couches;                   // lookups lately: who served them; otherwise where the experts are kept
    // the map's RAM home (the SSD's with it): "RAM" when GPU 0 also copies from it over PCIe (the Flash-Next), else "CPU"
    const parts = [{nom: "GPU 0", c: C.gpu0}, {nom: "GPU 1", c: C.gpu1}, {nom: copiesPcie() ? "RAM" : "CPU", c: C.ram}];
    $("ex-couches-legende").innerHTML = parts.map((p) => `<span class="cle" style="--c:${p.c}">${esc(p.nom)}</span>`).join("");
    $("ex-couches-note").textContent = vivant ? "who served its recent lookups" : "where its experts are kept (no lookups lately)";
    if (!boite.classList.contains("colonnes")) { boite.textContent = ""; boite.classList.add("colonnes"); }
    colonnes(boite, {titre: "By layer", parts, max: 1, hauteur: 170, fmt: (v) => pct(v), fmtTic: (v) => `${nf(v * 100)}%`,
      basEtiquette: (it, k) => String(k),
      items: (vivant ? recentes : gardes).map((r, l) => {
        const s = r[0] + r[1] + r[2] || 1;
        return {titre: `Layer ${l}`, parts: r.map((v) => v / s)};
      })});
  }

  function plusUtilises(em, parCouche) {
    const corps = $("ex-top");
    if (!em) { corps.innerHTML = `<tr><td colspan="5" class="doux">No expert map from this engine</td></tr>`; $("ex-top-note").textContent = ""; return; }
    // the ten busiest in one pass (a sort of every expert used lately cost ~45 ms on the Flash-Next's 24,576)
    const plus = (a, b) => (em.recent[a] || 0) - (em.recent[b] || 0) || (em.routed[a] || 0) - (em.routed[b] || 0);
    const top = [];
    let utilises = 0;
    for (let i = 0; i < em.tier.length; i++) {
      if (!((em.recent[i] || 0) > 0.05 || (em.routed[i] || 0) > 0)) continue;
      utilises++;
      if (top.length < 10) { top.push(i); top.sort((a, b) => plus(b, a)); }
      else if (plus(i, top[9]) > 0) { top[9] = i; top.sort((a, b) => plus(b, a)); }
    }
    $("ex-top-note").textContent = utilises ? `${nf(Math.min(10, utilises))} of ${nf(utilises)} used lately` : "";
    corps.innerHTML = top.length ? top.map((i) => {
      const [l, e] = paireDe(i), h = Math.max(0, Math.min(3, em.tier[i] | 0));
      return `<tr><td>${nf(l)}</td><td>${nf(e)}</td><td><span class="cle" style="--c:${MAISONS[h][1]}">${esc(MAISONS[h][0])}</span></td>` +
        `<td class="num">${nf(em.routed[i] || 0, (em.routed[i] || 0) % 1 ? 1 : 0)}</td><td class="num">${nf(em.recent[i] || 0, 1)}</td></tr>`;
    }).join("") : `<tr><td colspan="5" class="doux">No lookups lately: they are counted while the model writes.</td></tr>`;
    void parCouche;
  }

  function aide(ti, em, eng) {
    const boite = $("ex-aide");
    const pcie = !!eng && +eng.pcie_frac > 0;
    const cle = [!!em, ti ? ti.layers : 0, ti ? ti.per_layer : 0, ti ? !!ti.pool : false, ti ? ti.tiers.file.experts > 0 : false, pcie].join("/");
    if (boite.dataset.fait === cle) return;     // rewritten when the model (or its tiers) changes, not at each reading
    boite.dataset.fait = cle;
    const lesCouches = ti ? `each of its ${nf(ti.layers)} layers holds ${nf(ti.per_layer)}` : "each of its layers holds many";
    boite.innerHTML = `
      <p><b>What an expert is.</b> This model is a mixture of experts: ${lesCouches} small networks, the experts.
        For every token, a router picks a few experts in each layer and skips the others, so a token only goes through a small part of the model.</p>
      <p><b>Where they are kept.</b> Lapilli keeps as many experts as it can on the GPUs. GPU 0, the main one, keeps the experts used most
        and changes them as the text changes. GPU 1 computes the experts it keeps and sends the result back over PCIe. The other experts stay in RAM
        and the CPU computes them, which is much slower${pcie ? ", or GPU 0 copies them over PCIe and computes them itself" : ""}${ti && ti.tiers.file.experts ? "; those on the SSD are read from the disk first" : ""}.</p>
      <p><b>Reading the map.</b> One hexagon per expert, one block per layer, expert 0 at the top left of its block.
        <i>Live</i>: the color says where the expert is kept, its brightness how much it was used in the last seconds.
        <i>Placement</i>: where each expert is kept. <i>Activity</i>: how busy each expert is, wherever it is kept.</p>
      <p><b>What is counted.</b> Lookups are counted while the model writes: in each verify window, every expert it needs counts once.
        Reading a message is not counted. The figures come from the engine itself, twice a second.</p>`;
    void em;
  }

  return {init, sousVue, montrerSousVue, dessiner, dessinerExperts, aDesExperts, cacherBulle};
})();
