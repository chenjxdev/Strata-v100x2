// Lapilli: the chat page of the Basalt port to NVIDIA V100 GPUs (Chat, Monitor, About).
// No framework; nothing is sent anywhere but to this server. The layout follows that of Strata's page
// (Niko1221, MIT license); text is formatted by marked (MIT license, web/marked.umd.js).
"use strict";

const $ = (id) => document.getElementById(id);
const ICONES = "web/lapilli-icons.svg";
const ic = (nom, cls = "ic") => `<svg class="${cls}" aria-hidden="true"><use href="${ICONES}#i-${nom}"/></svg>`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const nf = (n, d = 0) => (n == null || !Number.isFinite(+n) ? "–"
  : Number(n).toLocaleString("en-US", {maximumFractionDigits: d, minimumFractionDigits: d}));
const go = (octets, d = 1) => (octets == null ? "–" : nf(octets / 1073741824, d));      // binary GB (GiB), as Windows shows them
const kf = (n) => (n == null ? "–" : n >= 10000 ? `${nf(n / 1000)}k` : n >= 1000 ? `${nf(n / 1000, 1)}k` : nf(n));
// a line of info: unbreakable pieces separated by a middle dot (the dot comes from the CSS, between two pieces)
const morceaux = (liste) => liste.filter((x) => x != null && x !== "").map((x) => `<span class="morceau">${esc(x)}</span>`).join("");
const estTactile = () => matchMedia("(hover: none)").matches;

const memoire = {
  lire(k, d) { try { const v = localStorage.getItem("lapilli." + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  ecrire(k, v) { try { localStorage.setItem("lapilli." + k, JSON.stringify(v)); return true; } catch (e) { return false; } },
};
const nouvelId = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8);

// ------------------------------------------------------------------ toasts (closable, replaced when the state changes)
function annonce(genre, titre, texte = "", ms = 4000, action = null) {
  const el = document.createElement("div");
  el.className = `annonce annonce--${genre}`;
  el.setAttribute("role", genre === "erreur" ? "alert" : "status");
  el.innerHTML = `${ic({info: "info", succes: "succes", alerte: "alerte", erreur: "erreur"}[genre] || "info")}` +
    `<div class="annonce__corps"><div class="annonce__titre"></div><div class="annonce__texte"></div></div><div class="annonce__actions"></div>`;
  el.querySelector(".annonce__titre").textContent = titre;
  if (texte) el.querySelector(".annonce__texte").textContent = texte; else el.querySelector(".annonce__texte").remove();
  const actions = el.querySelector(".annonce__actions");
  if (action) {
    const b = document.createElement("button");
    b.type = "button"; b.className = "btn btn--doux btn--petit"; b.textContent = action.libelle;
    b.onclick = () => { action.faire(); el.remove(); };
    actions.appendChild(b);
  }
  const x = document.createElement("button");
  x.type = "button"; x.className = "btn btn--icone"; x.title = "Close"; x.setAttribute("aria-label", "Dismiss notification");
  x.innerHTML = ic("croix");
  x.onclick = () => el.remove();
  actions.appendChild(x);
  const boite = $("annonces");
  boite.appendChild(el);
  while (boite.children.length > 3) boite.firstElementChild.remove();
  if (ms) setTimeout(() => el.remove(), ms);
  return el;
}

async function copier(texte, bouton) {
  let ok = true;
  try {
    await navigator.clipboard.writeText(texte);
  } catch (e) {                                   // over plain http to another machine there is no async clipboard
    const ta = document.createElement("textarea");
    ta.value = texte; ta.setAttribute("readonly", ""); ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select();
    try { ok = document.execCommand("copy"); } catch (e2) { ok = false; }
    ta.remove();
  }
  if (!ok) { annonce("erreur", "Could not copy", "The browser denied access to the clipboard.", 6000); return; }
  const u = bouton && bouton.querySelector("use");
  if (u) { u.setAttribute("href", `${ICONES}#i-coche`); setTimeout(() => u.setAttribute("href", `${ICONES}#i-copier`), 1500); }
  annonce("succes", "Copied to the clipboard", "", 1800);
}

// ------------------------------------------------------------------ times and days (a time alone does not say which day)
const debutJour = (t) => { const d = new Date(t); d.setHours(0, 0, 0, 0); return d.getTime(); };
const heure = (t, secondes = false) => new Date(t).toLocaleTimeString("en-US", {hour: "numeric", minute: "2-digit", ...(secondes ? {second: "2-digit"} : {})});
function jourDe(t) {
  const j = Math.round((debutJour(Date.now()) - debutJour(t)) / 86400000);
  if (j <= 0) return "Today";
  if (j === 1) return "Yesterday";
  const d = new Date(t);
  return d.toLocaleDateString("en-US", {day: "numeric", month: "short", ...(d.getFullYear() !== new Date().getFullYear() ? {year: "numeric"} : {})});
}
function quand(t, secondes = false) {              // a past moment, with its day when it is not today
  const j = Math.round((debutJour(Date.now()) - debutJour(t)) / 86400000);
  const h = heure(t, secondes);
  return j <= 0 ? h : j === 1 ? `yesterday ${h}` : `${new Date(t).toLocaleDateString("en-US", {day: "numeric", month: "short"})}, ${h}`;
}
function ilYa(t) {
  const s = Math.max(0, (Date.now() - t) / 1000);
  return s < 45 ? "just now" : s < 3600 ? `${nf(Math.max(1, Math.round(s / 60)))} min ago` : s < 86400 ? `${nf(s / 3600, 0)} h ago` : quand(t);
}

// ------------------------------------------------------------------ theme and views
function poserTheme(t, garder) {
  document.documentElement.dataset.theme = t;
  if (garder) { try { localStorage.setItem("lapilli.theme", t); } catch (e) { /* private browsing: for this visit only */ } }
  $("ic-theme").setAttribute("href", `${ICONES}#i-${t === "dark" ? "soleil" : "lune"}`);
  $("btn-theme").setAttribute("aria-label", t === "dark" ? "Switch to light theme" : "Switch to dark theme");
  $("bascule-sombre").setAttribute("aria-checked", String(t === "dark"));
}
const basculerTheme = () => poserTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark", true);
$("btn-theme").onclick = basculerTheme;
$("bascule-sombre").onclick = basculerTheme;
poserTheme(document.documentElement.dataset.theme === "dark" ? "dark" : "light", false);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
  let garde = null;
  try { garde = localStorage.getItem("lapilli.theme"); } catch (err) { /* nothing to do */ }
  if (garde !== "dark" && garde !== "light") poserTheme(e.matches ? "dark" : "light", false);
});

const VUES = ["discussion", "moniteur", "experts", "apropos"];
let vue = "discussion";
function montrerVue(nom) {
  const [v, sous] = String(nom || "").split("/");     // #moniteur/requetes: the Monitor, on its Requests section
  vue = VUES.includes(v) ? v : "discussion";
  for (const b of document.querySelectorAll(".onglet")) b.setAttribute("aria-selected", String(b.dataset.vue === vue));
  for (const x of VUES) $(`vue-${x}`).hidden = x !== vue;
  $("btn-panneau").hidden = vue !== "discussion";
  Moniteur.cacherBulle();
  if (vue === "moniteur") Moniteur.montrerSousVue(sous || Moniteur.sousVue(), false);   // without one, the last section seen
  const so = Moniteur.sousVue();
  const h = vue === "discussion" ? "" : vue === "moniteur" && so !== "apercu" ? `#moniteur/${so}` : `#${vue}`;
  if (location.hash !== h) history.replaceState(null, "", location.pathname + location.search + h);
  if (vue === "discussion" && !estTactile()) $("saisie").focus();
  if (mesures) dessinerMesures(mesures);
  if (vue === "experts" && mesures && !mesures.experts_map) relever();   // its map now, not at the next reading
}
for (const b of document.querySelectorAll(".onglet")) b.onclick = () => montrerVue(b.dataset.vue);
window.addEventListener("hashchange", () => montrerVue(location.hash.slice(1)));
$("marque").onclick = (e) => { e.preventDefault(); montrerVue("discussion"); };

// ------------------------------------------------------------------ the server
function entetes(json = false) {
  const h = {};
  const cle = memoire.lire("cle", "");
  if (cle) h.Authorization = "Bearer " + cle;
  if (json) h["Content-Type"] = "application/json";
  return h;
}
// the server's refusal, in plain words when it is a known one; otherwise its own message, as is
async function refus(r) {
  let code = "", message = "";
  try {
    const j = await r.json();
    const e = (j && j.error) || j || {};
    code = e.code || e.type || ""; message = e.message || e.detail || j.detail || "";
  } catch (e) { /* not JSON */ }
  const connus = {
    401: "This server requires an API key: add it under About.",
    403: "The server refuses requests coming from this page's address.",
    503: "The engine is starting or restarting: try again in a moment.",
  };
  let texte = connus[r.status] || "";
  if (code === "context_length_exceeded" || /context/i.test(message) && r.status === 400) texte = "The message is too long for the model's context.";
  if (code === "model_busy" || r.status === 409 && /busy/i.test(message)) texte = "The model is busy with another request.";
  if (code === "not_thinking") texte = "Thinking has already finished.";
  if (code === "unknown_request") texte = "The server no longer knows this response.";
  const err = new Error(texte || `The server returned ${r.status}${message ? `: ${message}` : ""}.`);
  err.code = code; err.statut = r.status; err.detail = message;
  return err;
}

let sante = {model: "flash-next", max_context: 0, api_key: false, loaded: true};
let defautsModele = {temperature: 1, top_p: 0.95, top_k: 20};
async function lireSante() {
  try {
    const r = await fetch("health", {cache: "no-store"});
    if (!r.ok) throw await refus(r);
    sante = await r.json();
    chargerTiroir();   // the length steps follow this server's context
    $("champ-cle").hidden = !sante.api_key && !memoire.lire("cle", "");
    // the button says what the server can read
    const joindre = sante.images ? "Attach a text file or an image (or drop them here, or paste a screenshot)" : "Attach a text file (or drop it here)";
    $("btn-joindre").title = joindre;
    $("btn-joindre").setAttribute("aria-label", sante.images ? "Attach a text file or an image" : "Attach a text file");
  } catch (e) {
    setTimeout(lireSante, 3000);
  }
}
async function lireProps() {
  try {
    const r = await fetch("props", {cache: "no-store", headers: entetes()});
    if (!r.ok) return;
    const p = await r.json();
    const g = (p.default_generation_settings || {}).params || {};
    for (const k of ["temperature", "top_p", "top_k"]) if (Number.isFinite(+g[k])) defautsModele[k] = +g[k];
    chargerTiroir();
  } catch (e) { /* the known values stay */ }
}
$("cle-api").value = memoire.lire("cle", "");
$("cle-api").onchange = () => {
  memoire.ecrire("cle", $("cle-api").value.trim());
  annonce("succes", "API key saved", "It stays in this browser.", 3000);
  relever();
};

// ------------------------------------------------------------------ metrics (/metrics), every second
let mesures = null, echecs = 0, cleAvertie = false, minuterieReleve = null, releveEnVol = false, releverEncore = false;
async function relever() {
  if (releveEnVol) { releverEncore = true; return; }         // one poll at a time: this one runs again right after
  releveEnVol = true;
  clearTimeout(minuterieReleve);
  try {
    // the expert map's arrays only for the Experts tab (24,576 experts make hundreds of KB): a summary otherwise
    const r = await fetch(vue === "experts" ? "metrics?experts=full" : "metrics", {cache: "no-store", headers: entetes()});
    if (r.status === 401) {
      poserPastille("horsligne", "API key required");
      $("champ-cle").hidden = false;
      if (!cleAvertie) { cleAvertie = true; annonce("alerte", "API key required", "This server requires a key: add it under About.", 8000); }
    } else if (!r.ok) {
      throw await refus(r);
    } else {
      mesures = await r.json();
      if (echecs >= 3) annonce("succes", "Server reachable again", "", 3000);
      echecs = 0;
      dessinerMesures(mesures);
    }
  } catch (e) {
    if (++echecs === 3) { poserPastille("horsligne", "Server unreachable"); if (enCours) activite(); }
  }
  releveEnVol = false;
  if (releverEncore) { releverEncore = false; relever(); return; }
  minuterieReleve = setTimeout(relever, enCours ? 500 : document.hidden ? 5000 : 1000);
}
document.addEventListener("visibilitychange", () => { if (!document.hidden) relever(); });

function poserPastille(etat, texte) {
  $("pastille").dataset.etat = etat;
  $("pastille-texte").textContent = texte;
}
// the share of the message already read, estimated: elapsed time times the mean speed of this read
function partLue(live) {
  if (!live.prompt_tokens || !live.prefill_tok_s_mean || !live.elapsed_s) return null;
  return Math.max(0, Math.min(0.99, (live.elapsed_s * live.prefill_tok_s_mean) / live.prompt_tokens));
}
function etatDe(live) {
  if (!live) return "connexion";
  if (live.state === "loading") return "chargement";
  if (live.queued > 0 && live.state === "idle") return "attente";
  if (live.state === "reading") return "lecture";
  if (live.state === "generating") return live.phase === "thinking" ? "reflexion" : "ecriture";
  if (live.state === "unloaded" || live.state === "asleep") return "attente";
  return "repos";
}
function dessinerPastille(m) {
  const live = m.live || {};
  const e = etatDe(live);
  const pct = (live.load || {}).percent;
  const lu = partLue(live);
  const textes = {
    chargement: `Loading${pct != null ? ` · ${nf(pct)}%` : ""}`,
    lecture: `Reading${lu != null ? ` · ${nf(lu * 100)}%` : ""}`,
    reflexion: `Thinking${live.tok_s ? ` · ${nf(live.tok_s, 1)} tokens/s` : ""}`,
    ecriture: `Writing${live.tok_s ? ` · ${nf(live.tok_s, 1)} tokens/s` : ""}`,
    attente: live.state === "unloaded" ? "Model unloaded" : live.state === "asleep" ? "Asleep" : `Waiting · ${nf(live.queued)}`,
    repos: "Idle",
  };
  let t = textes[e] || "Idle";
  if (live.queued > 0 && e !== "attente") t += ` · ${nf(live.queued)} waiting`;
  poserPastille(e, t);
}

// ------------------------------------------------------------------ Monitor and Experts (web/lapilli-monitor.js)
const nomProcesseur = (st) => {
  const n = String(st.cpu_name || "");
  if (/^Intel64 Family/i.test(n)) return "Intel";                       // the raw name Windows gives to Python
  if (/^AMD64 Family/i.test(n)) return "AMD";
  return n.replace(/\((R|TM)\)/gi, "").replace(/\s+\d+-Core Processor$/i, "").replace(/\s+CPU\s+@.*$/i, "").replace(/\s+/g, " ").trim();
};
Moniteur.init();

function faits(el, lignes) {
  el.innerHTML = lignes.filter((l) => l[1] != null && l[1] !== "").map(([k, v, copie]) => `<dt>${esc(k)}</dt><dd>${copie
    ? `<code>${esc(v)}</code><button type="button" class="btn btn--icone" data-copier="${esc(v)}" aria-label="Copy ${esc(k)}" title="Copy">${ic("copier")}</button>`
    : esc(v)}</dd>`).join("");
}
function dessinerApropos(m) {
  const eng = m.engine || {}, hw = m.hardware || {}, st = m.hardware_static || {};
  // the experts both cards hold, from the tiers the engine publishes (GPU 0's places, GPU 1's experts: 7,311 + 10,501
  // for the Flash-Next); GPU 0's places alone only from an engine that does not publish them
  const ti = m.experts && m.experts.tiers;
  const surCartes = ti ? (ti.main.experts || 0) + (ti.peer.experts || 0) : eng.expert_slots;
  const octetsCartes = ti ? (ti.main.bytes || 0) + (ti.peer.bytes || 0) : (eng.expert_cache_mib || 0) * 1048576;
  const tousExperts = m.experts && m.experts.layers ? m.experts.layers * m.experts.per_layer : eng.expert_total;
  const kv = {int8: "8-bit", q4_0: "4-bit", fp16: "16-bit", f16: "16-bit", bf16: "16-bit"}[eng.kv] || eng.kv;
  faits($("faits-moteur"), [
    ["Model", `${eng.model || sante.model}${eng.quant ? ` · ${eng.quant}` : ""}`],
    ["Engine", `Lapilli, based on Basalt ${eng.version || eng.engine || ""}`.trim()],
    ["Context", eng.max_context ? `${nf(eng.max_context)} tokens` : null],
    ["Images", sante.images ? "supported (vision on)" : "no (vision off)"],
    ["Context memory", kv ? `${kv}, ${eng.kv_resident ? `${nf(eng.kv_resident)} positions per layer on the GPUs, the rest in RAM` : "all on the GPUs"}` : null],
    ["Experts on the GPUs", surCartes ? `${nf(surCartes)}${tousExperts ? ` of ${nf(tousExperts)}` : ""} (${go(octetsCartes)} GB)` : null],
    ["MTP drafts", eng.spec ? "enabled" : null],
  ]);
  const cartes = (st.gpus || []).map((g) => g.name).filter(Boolean);
  const memes = cartes.length > 1 && cartes.every((n) => n === cartes[0]);
  faits($("faits-machine"), [
    ["GPUs", cartes.length ? (memes ? `${cartes.length} × ${cartes[0]}` : cartes.join(", ")) : null],
    ["PCIe links", (hw.gpus || []).length ? (hw.gpus || []).map((g) => `GPU ${g.index}: PCIe ${g.pcie_gen_max || g.pcie_gen} x${g.pcie_width}`).join(" · ") : null],
    ["Driver", st.driver ? `${st.driver}${st.cuda ? `, CUDA ${st.cuda}` : ""}` : null],
    ["CPU", st.cpu_name ? `${nomProcesseur(st)}${st.cores ? `, ${nf(st.cores)} cores` : ""}${st.threads ? `, ${nf(st.threads)} threads` : ""}` : null],
    ["RAM", hw.ram_total ? `${go(hw.ram_total)} GB` : null],
  ]);
  const base = location.origin;
  faits($("faits-api"), [
    ["OpenAI URL", `${base}/v1`, true],
    ["Anthropic URL", base, true],
    ["Model name", eng.model || sante.model, true],
  ]);
}
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-copier]");
  if (b) copier(b.dataset.copier, b);
});

function dessinerMesures(m) {
  dessinerPastille(m);
  // the Experts tab: only for a model whose server says where its experts are (a mixture of experts)
  const experts = Moniteur.aDesExperts(m);
  $("onglet-experts").hidden = !experts;
  if (!experts && vue === "experts") { montrerVue("moniteur"); return; }
  if (vue === "moniteur") Moniteur.dessiner(m);
  if (vue === "experts") Moniteur.dessinerExperts(m);
  if (vue === "apropos") dessinerApropos(m);
  if (enCours) activite();
}

// ------------------------------------------------------------------ formatting (marked; the model's HTML is escaped)
// a lone number (thousands separated by a space, a dot or a comma, decimals, a short unit); "2, 3, 5, 7" is not one
const estNombre = (s) => /^[-+−~≈<>]?\s?(\d{1,3}([\s\u00a0\u202f.,]\d{3})+|\d+)([.,]\d+)?\s?(%|€|\$|°C|°|GB|MB|KB|GiB|MiB|ms|s|min|h|W|V|A|Hz|MHz|GHz|x|×|tokens(\/s)?|t\/s)?$/i.test(s);
let rendu = null;
function preparerMarked() {
  if (rendu || !window.marked) return rendu;
  const base = new marked.Renderer();
  marked.use({
    gfm: true, breaks: false,
    renderer: {
      html(t) { const x = esc(t.text || t.raw || ""); return t.block ? `<p>${x}</p>\n` : x; },   // no HTML from the model in the page
      code(t) {
        const langue = String(t.lang || "").trim().split(/\s+/)[0] || "text";
        return `<div class="code-bloc"><div class="code-bloc__tete"><span>${esc(langue)}</span>` +
          `<button type="button" class="btn btn--icone" data-copier-code aria-label="Copy code" title="Copy code">${ic("copier")}</button></div>` +
          `<pre><code>${t.escaped ? t.text || "" : esc(t.text || "")}</code></pre></div>\n`;
      },
      link(t) {
        const texte = this.parser.parseInline(t.tokens);
        if (!/^(https?:|mailto:)/i.test(t.href || "")) return texte;          // neither javascript: nor a link to this server
        return `<a href="${esc(t.href)}" target="_blank" rel="noopener noreferrer"${t.title ? ` title="${esc(t.title)}"` : ""}>${texte}</a>`;
      },
      image(t) {                                                             // no image loaded from elsewhere: a link instead
        return /^https?:/i.test(t.href || "") ? `<a href="${esc(t.href)}" target="_blank" rel="noopener noreferrer">${esc(t.text || t.href)}</a>` : esc(t.text || "");
      },
      table(t) {
        const p = this.parser, n = t.header.length, num = [];
        for (let c = 0; c < n; c++) {
          const vals = t.rows.map((r) => String((r[c] && r[c].text) || "").replace(/\*\*/g, "").trim()).filter(Boolean);
          num[c] = vals.length > 0 && vals.every(estNombre);
        }
        const cel = (x, c, tag) => `<${tag}${num[c] ? ' class="md-num"' : ""}>${p.parseInline(x.tokens)}</${tag}>`;
        let h = `<div class="md-table-defile"><table><thead><tr>${t.header.map((x, c) => cel(x, c, "th")).join("")}</tr></thead>`;
        if (t.rows.length) h += `<tbody>${t.rows.map((r) => `<tr>${r.map((x, c) => cel(x, c, "td")).join("")}</tr>`).join("")}</tbody>`;
        return `${h}</table></div>\n`;
      },
      list(t) {
        let h = base.list.call(this, t);
        const debut = +t.start;
        if (t.ordered && Number.isFinite(debut) && debut !== 1) h = h.replace(/^<ol start="(\d+)">/, (x, s) => `<ol start="${s}" style="counter-reset: lp-ol ${+s - 1}">`);
        return h;
      },
      listitem(t) {
        const h = base.listitem.call(this, t);
        return t.task ? h.replace(/^<li>/, '<li class="tache">') : h;
      },
    },
  });
  rendu = (texte) => marked.parse(texte, {async: false});
  return rendu;
}
function enHtml(texte) {
  const r = preparerMarked();
  if (!r) return `<p>${esc(texte).replace(/\n/g, "<br>")}</p>`;          // marked not loaded yet: the raw text
  try { return r(texte); } catch (e) { return `<p>${esc(texte).replace(/\n/g, "<br>")}</p>`; }
}

// ------------------------------------------------------------------ conversations (kept in this browser)
const MAX_CONVS = 60;
let convs = memoire.lire("conversations", []);
if (!Array.isArray(convs)) convs = [];
let courante = memoire.lire("courante", null);
if (!convs.some((c) => c.id === courante)) courante = null;
const convCourante = () => convs.find((c) => c.id === courante) || null;
const titreDe = (texte, fichiers) => {
  const t = String(texte || "").replace(/\s+/g, " ").trim();
  return (t || (fichiers && fichiers[0] ? fichiers[0].nom : "Conversation")).slice(0, 80);
};
let averti = false, avertiImages = false;
// browser storage full: the images of the oldest conversations go first (their conversations stay, the model
// reads "no longer kept"); false when none is left
function allegerImages() {
  for (let k = convs.length - 1; k >= 0; k--) {
    for (const m of convs[k].messages) for (const i of m.images || []) if (i.image) { i.image = null; return true; }
  }
  return false;
}
function garderConvs() {
  convs.sort((a, b) => b.modifie - a.modifie);
  if (convs.length > MAX_CONVS) convs.length = MAX_CONVS;
  let retirees = 0, imagesRetirees = 0;
  while (!memoire.ecrire("conversations", convs)) {
    if (allegerImages()) { imagesRetirees++; continue; }
    if (convs.length <= 1) { annonce("alerte", "Browser storage full", "This conversation will not be kept after the page is closed.", 8000); break; }
    convs.pop(); retirees++;
  }
  if (retirees && !averti) { averti = true; annonce("alerte", "Made room in the browser", `Oldest conversations removed: ${nf(retirees)}.`, 8000); }
  if (imagesRetirees && !avertiImages) {
    avertiImages = true;
    annonce("alerte", "Images removed from the browser", `${imagesRetirees > 1 ? `The ${nf(imagesRetirees)} oldest images are no longer kept` : "The oldest image is no longer kept"}; their conversations remain.`, 8000);
  }
  memoire.ecrire("courante", courante);
}
function dessinerConvs() {
  const liste = $("convs-liste");
  if (!convs.length) { liste.innerHTML = `<div class="convs__vide">No conversations yet.</div>`; return; }
  let h = "", avant = null;
  for (const c of convs) {
    const j = jourDe(c.modifie);
    if (j !== avant) { h += `<div class="convs__jour">${esc(j)}</div>`; avant = j; }
    h += `<div class="conv${c.id === courante ? " conv--active" : ""}" data-id="${esc(c.id)}">` +
      `<button type="button" class="conv__ouvrir" data-ouvrir="${esc(c.id)}" title="${esc(c.titre)}"${c.id === courante ? ' aria-current="true"' : ""}>` +
      `<span class="conv__titre">${esc(c.titre)}</span><span class="conv__heure">${esc(heure(c.modifie))}</span></button>` +
      `<button type="button" class="conv__suppr" data-suppr="${esc(c.id)}" title="Delete conversation" aria-label="Delete conversation: ${esc(c.titre)}">${ic("corbeille", "ic ic--p")}</button></div>`;
  }
  liste.innerHTML = h;
}
$("convs-liste").addEventListener("click", (e) => {
  const s = e.target.closest("[data-suppr]");
  if (s) { supprimerConv(s.dataset.suppr); return; }
  const o = e.target.closest("[data-ouvrir]");
  if (o) choisirConv(o.dataset.ouvrir);
});
function choisirConv(id) {
  if (enCours) { annonce("alerte", "Lapilli is still writing", "Stop the response before switching conversations.", 4000); return; }
  courante = id;
  memoire.ecrire("courante", courante);
  dessinerConvs(); dessinerFil();
  if (etroit()) ouvrirConvs(false);
  if (!estTactile()) $("saisie").focus();
}
function nouvelleConv() {
  if (enCours) { annonce("alerte", "Lapilli is still writing", "Stop the response before starting a new conversation.", 4000); return; }
  courante = null;
  memoire.ecrire("courante", null);
  pieces = []; dessinerPieces();
  dessinerConvs(); dessinerFil();
  if (etroit()) ouvrirConvs(false);
  $("saisie").focus();
}
function supprimerConv(id) {
  if (enCours && enCours.c.id === id) { annonce("alerte", "Lapilli is still writing", "Stop the response before deleting this conversation.", 4000); return; }
  const i = convs.findIndex((c) => c.id === id);
  if (i < 0) return;
  const [retiree] = convs.splice(i, 1);
  const etaitCourante = courante === id;
  if (etaitCourante) courante = null;
  garderConvs(); dessinerConvs();
  if (etaitCourante) dessinerFil();
  annonce("info", "Conversation deleted", retiree.titre, 7000, {libelle: "Undo", faire: () => {
    convs.push(retiree);
    if (etaitCourante) courante = retiree.id;
    garderConvs(); dessinerConvs(); if (etaitCourante) dessinerFil();
  }});
}
$("btn-nouvelle").onclick = nouvelleConv;
$("btn-nouvelle-2").onclick = nouvelleConv;

// the conversation panel: beside the thread on a wide screen, over it on a narrow one
const etroit = () => matchMedia("(max-width: 960px)").matches;
function ouvrirConvs(ouvert) {
  $("convs").dataset.ouvert = String(ouvert);
  $("btn-panneau").setAttribute("aria-expanded", String(ouvert));
  $("voile-convs").hidden = !(ouvert && etroit());
  if (!etroit()) memoire.ecrire("panneau", ouvert);
}
$("btn-panneau").onclick = () => ouvrirConvs($("convs").dataset.ouvert !== "true");
$("voile-convs").onclick = () => ouvrirConvs(false);
matchMedia("(max-width: 960px)").addEventListener("change", () => ouvrirConvs(etroit() ? false : memoire.lire("panneau", true)));

// ------------------------------------------------------------------ the thread
const elements = new WeakMap();            // a Lapilli message -> its element in the thread
function elementMessage(m, i) {
  const el = document.createElement("div");
  el.dataset.i = i;
  if (m.role === "user") {
    el.className = "msg msg--moi";
    if (m.fichiers && m.fichiers.length) {
      const w = document.createElement("div");
      w.className = "pieces-msg";
      for (const f of m.fichiers) {
        const p = document.createElement("span");
        p.className = "piece piece--fixe";
        p.innerHTML = ic("fichier", "ic ic--p");
        const n = document.createElement("span"); n.className = "piece__nom"; n.textContent = f.nom; p.appendChild(n);
        w.appendChild(p);
      }
      el.appendChild(w);
    }
    if (m.images && m.images.length) {               // one click shows it full size
      const w = document.createElement("div");
      w.className = "images-msg";
      for (const i of m.images) {
        if (i.image) {
          const b = document.createElement("button");
          b.type = "button"; b.className = "image-msg"; b.title = `${i.nom}: view full size`;
          b.setAttribute("aria-label", `View image ${i.nom} full size`);
          const v = document.createElement("img");
          v.src = i.image; v.alt = i.nom;
          // its on-screen size, computed here (280 x 200 at most, proportions kept) so the frame fits tightly around it
          // (CSS alone left empty space on its right: the width was capped AFTER the height)
          if (i.l && i.h) { const k = Math.min(1, 280 / i.l, 200 / i.h); v.width = Math.round(i.l * k); v.height = Math.round(i.h * k); }
          b.appendChild(v);
          b.onclick = () => voirImage(i);
          w.appendChild(b);
        } else {
          const p = document.createElement("span");
          p.className = "piece piece--fixe";
          p.innerHTML = ic("fichier", "ic ic--p");
          const n = document.createElement("span"); n.className = "piece__nom"; n.textContent = `${i.nom} (no longer kept)`; p.appendChild(n);
          w.appendChild(p);
        }
      }
      el.appendChild(w);
    }
    if (m.texte) {
      const b = document.createElement("div");
      b.className = "bulle";
      b.textContent = m.texte;
      el.appendChild(b);
    }
    const meta = document.createElement("div");
    meta.className = "msg__meta";
    meta.innerHTML = `<span class="meta-texte">${morceaux(["You", quand(m.temps)])}</span><span class="meta-actions">` +
      `<button type="button" class="btn btn--icone" data-copier-msg title="Copy message" aria-label="Copy message">${ic("copier")}</button></span>`;
    el.appendChild(meta);
    return el;
  }
  el.className = "msg msg--lapilli";
  el.innerHTML = `<div class="msg__avatar"><img src="web/lapilli-logo.svg" alt=""></div><div class="msg__contenu">` +
    `<details class="reflexion" hidden><summary>${ic("reflexion", "ic ic--p")}<span class="reflexion__titre"></span>${ic("chevron", "ic ic--p chevron")}</summary>` +
    `<div class="reflexion__corps"></div></details>` +
    `<div class="reponse md"></div><div class="outils" hidden></div><div class="activite" hidden></div><div class="fin" hidden></div>` +
    `<div class="msg__meta"><span class="meta-texte"></span><span class="meta-actions">` +
    `<button type="button" class="btn btn--icone" data-copier-msg title="Copy response" aria-label="Copy response">${ic("copier")}</button>` +
    `<button type="button" class="btn btn--icone" data-regenerer title="Regenerate response" aria-label="Regenerate response">${ic("regenerer")}</button>` +
    `</span></div></div>`;
  elements.set(m, el);
  majLapilli(el, m, false);
  return el;
}

// a long run without spaces (a URL, a path) may wrap after "/ . - _ @ ? & = : , ;", never in the middle
// of a word (otherwise a narrow screen splits a word inside a URL)
function coupable(texte) {
  return String(texte).split(/(\S{24,})/).map((x, i) => (i % 2 ? x.split(/(?<=[\/.\-_@?&=:,;])(?=\S)/).map(esc).join("<wbr>") : esc(x))).join("");
}
function majReflexion(el, m, vivant) {
  const det = el.querySelector("details.reflexion");
  if (!m.reflexion) { det.hidden = true; return; }
  det.hidden = false;
  const enTrain = vivant && !m.texte && !aDesOutils(m) && !m.erreur;   // a tool call ends the thinking
  det.dataset.vivant = enTrain ? "1" : "0";
  const s = enTrain && enCours && enCours.debutRefl ? (performance.now() - enCours.debutRefl) / 1000 : m.reflexionS;
  el.querySelector(".reflexion__titre").textContent = enTrain ? `Thinking${s != null ? ` · ${nf(s, 0)} s` : ""}`
    : m.reflexionS != null ? `Thought for ${nf(m.reflexionS, 1)} s` : "Thinking";
  const corps = el.querySelector(".reflexion__corps");
  if (det.open) {
    const enBas = corps.scrollHeight - corps.scrollTop - corps.clientHeight < 40;
    if (corps.textContent.length !== m.reflexion.length) corps.innerHTML = coupable(m.reflexion);
    if (enBas) corps.scrollTop = corps.scrollHeight;
  } else {
    corps.dataset.attente = "1";
  }
  // open while it is being written (if wanted), closed when the answer starts; unless the user toggled it
  if (enTrain && reglages.montrer && !det.dataset.touchee && !det.open) { det._auto = true; det.open = true; corps.innerHTML = coupable(m.reflexion); }
  if (!enTrain && enCours && enCours.m === m && enCours.finRefl && !enCours.refermee) {
    enCours.refermee = true;
    if (det.open && !det.dataset.touchee) { det._auto = true; det.open = false; }
  }
}
function majLapilli(el, m, vivant) {
  majReflexion(el, m, vivant);
  majOutils(el, m, vivant);
  const rep = el.querySelector(".reponse");
  const outils = aDesOutils(m);
  rep.classList.remove("curseur-ecriture");
  if (m.erreur && !m.texte) {
    rep.hidden = false;
    rep.innerHTML = `<div class="erreur-msg">${ic("erreur")}<span></span></div>`;
    rep.querySelector("span").textContent = m.erreur;
  } else if (!m.texte && (vivant || outils)) {
    rep.innerHTML = "";                              // nothing written (yet): the activity line, or the tool call, says what is going on
    rep.hidden = true;
  } else {
    rep.hidden = false;
    rep.innerHTML = enHtml(m.texte || "") + (m.erreur ? `<div class="erreur-msg">${ic("erreur")}<span>${esc(m.erreur)}</span></div>` : "");
    if (vivant && !outils) {                         // the cursor at the end of the last text block (not during a tool call)
      const dernier = [...rep.querySelectorAll("p, li, h1, h2, h3, h4, td")].pop();
      if (dernier) dernier.classList.add("curseur-ecriture"); else rep.classList.add("curseur-ecriture");
    }
    if (!m.texte && !m.erreur) {
      rep.innerHTML = `<p class="doux">${m.arretee ? "Stopped before writing a response." : !m.meta ? "Response interrupted: the page was reloaded while it was being written." : "No written response."}</p>`;
    }
  }
  const fin = el.querySelector(".fin");
  const notes = [];
  if (!vivant && outils) {                           // the page has no tools
    notes.push(`<div class="fin-note"><span class="fin-note__texte">${ic("info", "ic ic--p")}<span>${m.outils.length > 1 ? "The model wanted to use tools." : "The model wanted to use a tool."} This page has none: nothing was run.</span></span></div>`);
  }
  if (!vivant && m.fin === "length") {
    // the length the response asked for, or the end of the context: with no length, or fewer tokens than the
    // length asked (the server cut it to the room the conversation left), it is the context that stopped it, and the
    // settings cannot give it more room
    const contexte = !m.limite || (m.ecrits != null && m.ecrits < m.limite);
    const ctx = +sante.max_context || 0;
    notes.push(contexte
      ? `<div class="fin-note"><span class="fin-note__texte">${ic("alerte", "ic ic--p")}<span>The response filled the model's context${ctx ? ` (${nf(ctx)}\u00a0tokens)` : ""}.</span></span></div>`
      : `<div class="fin-note"><span class="fin-note__texte">${ic("alerte", "ic ic--p")}<span>The response reached the maximum length (${nf(m.limite)}\u00a0tokens).</span></span>` +
        `<button type="button" class="btn btn--doux" data-ouvrir-reglages>Settings</button></div>`);
  }
  const notesHtml = notes.join("");
  if (fin._notes !== notesHtml) { fin._notes = notesHtml; fin.innerHTML = notesHtml; }
  fin.hidden = !notes.length;
  el.querySelector(".activite").hidden = !vivant;
  el.querySelector(".msg__meta .meta-texte").innerHTML = vivant ? "" : morceaux(m.meta || ["Interrupted"]);
  if (m.metaTitre) el.querySelector(".msg__meta .meta-texte").title = m.metaTitre;
  el.querySelector("[data-copier-msg]").hidden = vivant || !m.texte;
  const c = convCourante();
  const dernier = c && c.messages[c.messages.length - 1] === m;
  el.querySelector("[data-regenerer]").hidden = vivant || !dernier || !!enCours;
}
function dessinerFil() {
  const fil = $("fil");
  fil.querySelectorAll(".msg").forEach((e) => e.remove());
  const c = convCourante();
  const vide = !c || !c.messages.length;
  $("accueil").hidden = !vide;
  if (c) c.messages.forEach((m, i) => fil.appendChild(elementMessage(m, i)));
  descendre(true);
}
const filDefile = $("fil-defile");
const presDuBas = () => filDefile.scrollHeight - filDefile.scrollTop - filDefile.clientHeight < 140;
let suivre = true;                              // follow the end of the text, unless the user scrolled up
function descendre(force) {
  if (force || suivre) filDefile.scrollTop = filDefile.scrollHeight;
  majVersBas();
}
function majVersBas() { $("vers-bas").hidden = presDuBas(); }
filDefile.addEventListener("scroll", () => { suivre = presDuBas(); majVersBas(); }, {passive: true});
$("vers-bas").onclick = () => { suivre = true; filDefile.scrollTo({top: filDefile.scrollHeight, behavior: "smooth"}); };

$("fil").addEventListener("click", (e) => {
  const cc = e.target.closest("[data-copier-code]");
  if (cc) { copier(cc.closest(".code-bloc").querySelector("pre").textContent, cc); return; }
  const cm = e.target.closest("[data-copier-msg]");
  if (cm) {
    const c = convCourante(), m = c && c.messages[+cm.closest(".msg").dataset.i];
    if (m) copier(m.texte || "", cm);
    return;
  }
  if (e.target.closest("[data-regenerer]")) { regenerer(); return; }
  if (e.target.closest("[data-repondre-maintenant]")) { repondreMaintenant(); return; }
  if (e.target.closest("[data-ouvrir-reglages]")) { ouvrirTiroir(true); return; }
  const s = e.target.closest(".suggestion");
  if (s) { $("saisie").value = s.dataset.q; ajusterSaisie(); envoyer(); }
});
$("fil").addEventListener("toggle", (e) => {
  const d = e.target;
  if (d.tagName !== "DETAILS" || !d.classList.contains("reflexion")) return;
  if (d._auto) { d._auto = false; return; }          // our own openings, not the user's
  d.dataset.touchee = "1";
  const corps = d.querySelector(".reflexion__corps");
  if (d.open && corps.dataset.attente) {
    const c = convCourante(), m = c && c.messages[+d.closest(".msg").dataset.i];
    if (m) corps.innerHTML = coupable(m.reflexion);
    delete corps.dataset.attente;
  }
}, true);

// ------------------------------------------------------------------ tool calls
// The server sends a tool call the model writes on its own ("tool_calls"), even when the page declares no tools: the
// name, then the arguments (JSON) in chunks. They are shown collapsed (a whole file can weigh 40 KB) and are sent
// back with the history.
function aDesOutils(m) { return Array.isArray(m.outils) && m.outils.length > 0; }
function lireArgs(texte) {
  // a call's arguments: a JSON object, read even when cut off or still being written (we try to close it)
  const brut = String(texte || "");
  if (!brut.trim()) return {objet: {}, complet: true};
  try { return {objet: JSON.parse(brut), complet: true}; } catch (e) { /* cut off, or not finished yet */ }
  let b = brut.replace(/\\u[0-9a-fA-F]{0,3}$/, "");
  if (/(?:^|[^\\])(?:\\\\)*\\$/.test(b)) b = b.slice(0, -1);   // a lone escape backslash at the end
  const essais = [b, b.replace(/,\s*"(?:[^"\\]|\\.)*"?\s*:?\s*$/, "")];   // without a half-written key at the end
  for (const debut of essais) {
    for (const fin of ['"}', "}", '"]}', "]}", '"}]}', "}]}"]) {
      try { return {objet: JSON.parse(debut + fin), complet: false}; } catch (e) { /* try the next closing */ }
    }
  }
  return {objet: null, complet: false};
}
function argsValides(texte) {
  if (!String(texte || "").trim()) return true;
  try { const v = JSON.parse(texte); return v !== null && typeof v === "object" && !Array.isArray(v); } catch (e) { return false; }
}
function blocCode(texte, langue) {   // a markdown code block whose fence cannot occur in the text
  const n = Math.max(3, ...(String(texte).match(/`{3,}/g) || []).map((x) => x.length + 1));
  const f = "`".repeat(n);
  return `${f}${langue || ""}\n${texte}\n${f}\n`;
}
function langueOutil(objet) {        // the target file's language, from its name (index.html: html)
  for (const k of ["path", "file_path", "filepath", "filename", "file", "fichier", "chemin", "target"]) {
    const v = objet && objet[k];
    const x = typeof v === "string" ? /\.([a-z0-9]{1,8})$/i.exec(v.trim()) : null;
    if (x) return x[1].toLowerCase();
  }
  return "";
}
function estLong(v) { return typeof v === "string" && (v.includes("\n") || v.length > 100); }
function resumeOutil(o, enTrain) {
  // the header: "Tool call: name", then the first short argument (a file name) and the size of the longest one
  const {objet} = lireArgs(o.args);
  const infos = [];
  if (objet && typeof objet === "object" && !Array.isArray(objet)) {
    const valeurs = Object.values(objet);
    const court = valeurs.find((v) => typeof v === "string" && v.trim() && !estLong(v));
    if (court) infos.push(court.length > 48 ? court.slice(0, 47) + "…" : court);
    const longs = valeurs.filter(estLong);
    if (longs.length) {
      const plus = longs.reduce((a, b) => (b.length > a.length ? b : a));
      const lignes = plus.split("\n").length;
      infos.push(lignes > 1 ? `${nf(lignes)} lines` : `${nf(plus.length)} characters`);
    }
  } else if (o.args) {
    infos.push(`${nf(String(o.args).length)} characters`);
  }
  return `<span class="outil__nom"><span class="insecable">${enTrain ? "Tool call in progress:" : "Tool call:"}</span> ` +
    `<code>${esc(o.nom || "unnamed")}</code></span>` + (infos.length ? `<span class="outil__infos">${morceaux(infos)}</span>` : "");
}
function corpsOutil(o, vivant) {
  const {objet, complet} = lireArgs(o.args);
  let h = "";
  if (!objet || typeof objet !== "object" || Array.isArray(objet)) {
    h = enHtml(blocCode(String(o.args || ""), "json"));     // not a readable object: the text as is
  } else {
    const langue = langueOutil(objet);
    for (const [cle, v] of Object.entries(objet)) {
      if (estLong(v)) h += `<div class="outil__cle">${esc(cle)}</div>` + enHtml(blocCode(v, langue));
      else if (v === null || typeof v !== "object") h += `<div class="outil__ligne"><span class="outil__cle">${esc(cle)}</span><code>${esc(String(v))}</code></div>`;
      else h += `<div class="outil__cle">${esc(cle)}</div>` + enHtml(blocCode(JSON.stringify(v, null, 2), "json"));
    }
    if (!h) h = `<p class="doux">No arguments.</p>`;
  }
  if (!complet && !vivant) h += `<p class="doux">The call was cut off before it ended.</p>`;
  return h;
}
function outilEnMarkdown(o) {        // for the Markdown export: readable arguments, long ones as code blocks
  const {objet} = lireArgs(o.args);
  const md = [`_Tool call: ${o.nom || "unnamed"} (not run)_`, ""];
  if (!objet || typeof objet !== "object" || Array.isArray(objet)) return md.concat([blocCode(String(o.args || ""), "json")]);
  const langue = langueOutil(objet);
  for (const [cle, v] of Object.entries(objet)) {
    if (estLong(v)) md.push(`${cle}:`, "", blocCode(v, langue));
    else if (v === null || typeof v !== "object") md.push(`- ${cle}: ${String(v)}`, "");
    else md.push(`${cle}:`, "", blocCode(JSON.stringify(v, null, 2), "json"));
  }
  return md;
}
function peindreCorpsOutil(det, o, vivant) {
  const corps = det.querySelector(".outil__corps");
  const cle = `${String(o.args || "").length}|${vivant ? 1 : 0}`;
  if (corps._cle === cle) return;
  corps._cle = cle;
  corps.innerHTML = corpsOutil(o, vivant);
}
function majOutils(el, m, vivant) {
  const box = el.querySelector(".outils");
  if (!box) return;
  const liste = aDesOutils(m) ? m.outils : [];
  box.hidden = !liste.length;
  while (box.children.length > liste.length) box.lastElementChild.remove();
  liste.forEach((o, k) => {
    let det = box.children[k];
    if (!det) {
      det = document.createElement("details");
      det.className = "outil";
      det.dataset.k = String(k);
      det.innerHTML = `<summary>${ic("outil", "ic ic--p")}<span class="outil__titre"></span>${ic("chevron", "ic ic--p chevron")}</summary>` +
        `<div class="outil__corps md"></div>`;
      box.appendChild(det);
    }
    const enTrain = vivant && k === liste.length - 1;
    det.dataset.vivant = enTrain ? "1" : "0";
    const t = det.querySelector(".outil__titre"), titre = resumeOutil(o, enTrain);
    if (t._titre !== titre) { t._titre = titre; t.innerHTML = titre; }
    if (det.open) peindreCorpsOutil(det, o, vivant);   // collapsed: its body is written when it opens
  });
}
$("fil").addEventListener("toggle", (e) => {
  const d = e.target;
  if (d.tagName !== "DETAILS" || !d.classList.contains("outil") || !d.open) return;
  const c = convCourante(), m = c && c.messages[+d.closest(".msg").dataset.i];
  const o = m && aDesOutils(m) ? m.outils[+d.dataset.k] : null;
  if (o) peindreCorpsOutil(d, o, !!(enCours && enCours.m === m));
}, true);

// ------------------------------------------------------------------ response settings
const DEFAUTS = {reflexion: "medium", longueur: 8192, montrer: true, temperature: null, top_p: null, top_k: null, graine: ""};
let reglages = {...DEFAUTS, ...memoire.lire("reglages", {})};
const NOMS_REFLEXION = {none: "No thinking", low: "Short thinking", medium: "Medium thinking", high: "Long thinking"};
const AIDE_REFLEXION = {none: "answers right away", low: "thinks a little", medium: "thinks, then answers", high: "thinks it through"};
function resumeReglages() {
  const l = reglages.longueur > 0 ? `${nf(reglages.longueur)} tokens max` : "no limit";
  $("reglage-resume").textContent = `${NOMS_REFLEXION[reglages.reflexion] || "Medium thinking"} · ${l}`;
}
// the lengths a response may ask for: from 1,024 up to the server's context (/health max_context: the steps below
// it, then "No limit", the rest of the context), not a list that stopped at 32,768 whatever the model; an unknown context
// keeps that list.  A chosen length goes with fit_max_tokens (the server cuts it to the room the conversation leaves).
const PALIERS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];
let paliersPour = -1;
function construirePaliers() {
  const ctx = +sante.max_context || 0;
  if (ctx === paliersPour) return;
  paliersPour = ctx;
  const option = (v, texte) => { const o = document.createElement("option"); o.value = String(v); o.textContent = texte; return o; };
  const valeurs = PALIERS.filter((v) => (ctx > 0 ? v < ctx : v <= 32768));
  $("r-longueur").replaceChildren(...valeurs.map((v) => option(v, `${nf(v)} tokens`)), option(0, "No limit"));
  const c = $("r-contexte");
  c.hidden = !(ctx > 0);
  c.textContent = ctx > 0 ? `This model's context: ${nf(ctx)}\u00a0tokens, the conversation and the response together.` : "";
}
function chargerTiroir() {
  for (const b of $("r-reflexion").children) b.setAttribute("aria-checked", String(b.dataset.v === reglages.reflexion));
  $("o-reflexion").textContent = AIDE_REFLEXION[reglages.reflexion] || "";
  construirePaliers();
  const sel = $("r-longueur");
  if (![...sel.options].some((o) => +o.value === +reglages.longueur)) {   // a length kept from another model: in its place
    const o = document.createElement("option"); o.value = String(reglages.longueur); o.textContent = `${nf(reglages.longueur)} tokens`;
    sel.insertBefore(o, [...sel.options].find((x) => +x.value === 0 || +x.value > reglages.longueur) || null);
  }
  sel.value = String(reglages.longueur);
  $("r-montrer").setAttribute("aria-checked", String(!!reglages.montrer));
  const t = reglages.temperature ?? defautsModele.temperature, p = reglages.top_p ?? defautsModele.top_p, k = reglages.top_k ?? defautsModele.top_k;
  $("r-temperature").value = t; $("r-topp").value = p; $("r-topk").value = k;
  $("o-temperature").textContent = `${+t === 0 ? "0 · always the same" : nf(+t, 2)}${reglages.temperature == null ? " (default)" : ""}`;
  $("o-topp").textContent = `${nf(+p, 2)}${reglages.top_p == null ? " (default)" : ""}`;
  $("o-topk").textContent = `${nf(+k)}${reglages.top_k == null ? " (default)" : ""}`;
  $("r-topp").disabled = $("r-topk").disabled = +t === 0;
  $("r-graine").value = reglages.graine || "";
  $("r-defauts").textContent = `Model defaults: temperature ${nf(defautsModele.temperature, 2)}, top-p ${nf(defautsModele.top_p, 2)}, top-k ${nf(defautsModele.top_k)}.`;
  resumeReglages();
}
function garderReglages(note = "Saved in this browser.") {
  memoire.ecrire("reglages", reglages);
  chargerTiroir();
  $("r-note").textContent = note;
}
for (const b of $("r-reflexion").children) b.onclick = () => { reglages.reflexion = b.dataset.v; garderReglages(); };
$("r-longueur").onchange = () => { reglages.longueur = +$("r-longueur").value; garderReglages(); };
$("r-montrer").onclick = () => { reglages.montrer = !reglages.montrer; garderReglages(); };
$("r-temperature").oninput = () => { reglages.temperature = +$("r-temperature").value; garderReglages(); };
$("r-topp").oninput = () => { reglages.top_p = +$("r-topp").value; garderReglages(); };
$("r-topk").oninput = () => { reglages.top_k = +$("r-topk").value; garderReglages(); };
$("r-graine").onchange = () => { reglages.graine = $("r-graine").value.trim(); garderReglages(); };
$("r-retablir").onclick = () => { reglages = {...DEFAUTS}; garderReglages("Default settings restored."); };
function ouvrirTiroir(ouvert) {
  $("tiroir").dataset.ouvert = String(ouvert);
  $("tiroir").setAttribute("aria-hidden", String(!ouvert));
  $("voile").hidden = !ouvert;
  if (ouvert) { chargerTiroir(); $("r-note").textContent = "Saved in this browser."; setTimeout(() => $("tiroir-fermer").focus(), 60); }
  else if (vue === "discussion" && !estTactile()) $("saisie").focus();
}
$("btn-reglages").onclick = () => ouvrirTiroir(true);
$("tiroir-fermer").onclick = () => ouvrirTiroir(false);
$("voile").onclick = () => ouvrirTiroir(false);
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  const vue = document.querySelector(".visionneuse");
  if (vue) { vue.remove(); return; }
  if ($("tiroir").dataset.ouvert === "true") { ouvrirTiroir(false); return; }
  if (etroit() && $("convs").dataset.ouvert === "true") ouvrirConvs(false);
});

// ------------------------------------------------------------------ attached files (text; images when the server reads them)
const EXT_TEXTE = /\.(txt|md|markdown|rst|tex|py|pyi|ipynb|js|mjs|cjs|ts|tsx|jsx|vue|svelte|json|jsonl|csv|tsv|log|ya?ml|toml|ini|cfg|conf|env|xml|html?|css|scss|less|c|cc|cpp|cxx|h|hh|hpp|cu|cuh|rs|go|java|kt|kts|swift|rb|php|pl|lua|r|jl|scala|sql|sh|bash|zsh|fish|ps1|psm1|bat|cmd|diff|patch|gradle|cmake|mk|dockerfile|gitignore|proto|graphql|ino|srt)$/i;
const MAX_FICHIER = 512 * 1024;
let pieces = [];                                // {nom, texte}; an image: {nom, image, l, h}
// an image is accepted only if the server reads images (/health "images"); scaled down to 1600 px on its longest
// side, as JPEG (transparency becomes white): fewer tokens to read, less room in this browser. It is sent as bytes
// ("data:"): the server never reads a file by its path.
const IMAGE_COTE = 1600;
const MAX_IMAGE = 32 * 1024 * 1024;             // the server's limit for one image
function lireImage(f) {
  return new Promise((ok, ko) => {
    const url = URL.createObjectURL(f);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      const k = Math.min(1, IMAGE_COTE / Math.max(img.naturalWidth, img.naturalHeight, 1));
      const l = Math.max(1, Math.round(img.naturalWidth * k)), h = Math.max(1, Math.round(img.naturalHeight * k));
      const cv = document.createElement("canvas");
      cv.width = l; cv.height = h;
      const cx = cv.getContext("2d");
      cx.fillStyle = "#ffffff"; cx.fillRect(0, 0, l, h);
      cx.drawImage(img, 0, 0, l, h);
      ok({nom: f.name || "screenshot.png", image: cv.toDataURL("image/jpeg", 0.9), l, h});
    };
    img.onerror = () => { URL.revokeObjectURL(url); ko(new Error("unreadable")); };
    img.src = url;
  });
}
const estTexte = (f) => f.type.startsWith("text/") || /json|xml|javascript|yaml|toml|x-sh|x-python/.test(f.type) || EXT_TEXTE.test(f.name) ||
  /(^|[\\/])(makefile|dockerfile|readme|license)$/i.test(f.name);
function ajouterFichiers(liste) {
  for (const f of liste) {
    if (f.type.startsWith("image/")) {
      if (!sante.images) { annonce("alerte", "No images", "This server reads text only (vision is off).", 6000); continue; }
      if (f.size > MAX_IMAGE) { annonce("alerte", "Image too large", `${f.name} is over 32 MB.`, 6000); continue; }
      lireImage(f).then((p) => { pieces.push(p); dessinerPieces(); })
        .catch(() => annonce("erreur", "Unreadable image", `${f.name}: this browser cannot read it.`, 6000));
      continue;
    }
    if (!estTexte(f)) { annonce("alerte", "Not a text file", `${f.name}: attach text (code, notes, logs, data).`, 6000); continue; }
    if (f.size > MAX_FICHIER) { annonce("alerte", "File too large", `${f.name} is over 512 KB.`, 6000); continue; }
    const r = new FileReader();
    r.onload = () => {
      const texte = String(r.result);
      if (texte.includes("\u0000")) { annonce("alerte", "Not a text file", `${f.name} looks like a binary file.`, 6000); return; }
      pieces.push({nom: f.name, texte});
      dessinerPieces();
    };
    r.onerror = () => annonce("erreur", "Could not read the file", f.name, 6000);
    r.readAsText(f);
  }
}
function dessinerPieces() {
  const box = $("pieces");
  box.hidden = !pieces.length;
  box.innerHTML = "";
  pieces.forEach((p, i) => {
    const c = document.createElement("span");
    c.className = "piece";
    if (p.image) {                                 // its thumbnail
      c.classList.add("piece--image");
      const v = document.createElement("img");
      v.className = "piece__vignette"; v.src = p.image; v.alt = "";
      c.appendChild(v);
    } else c.innerHTML = ic("fichier", "ic ic--p");
    const n = document.createElement("span"); n.className = "piece__nom"; n.textContent = p.nom; n.title = p.nom; c.appendChild(n);
    const x = document.createElement("button");
    x.type = "button"; x.className = "btn btn--icone"; x.title = "Remove"; x.setAttribute("aria-label", `Remove ${p.nom}`);
    x.innerHTML = ic("croix");
    x.onclick = () => { pieces.splice(i, 1); dessinerPieces(); };
    c.appendChild(x);
    box.appendChild(c);
  });
}
// an image at full size, over the page; a click or Escape closes it
function voirImage(i) {
  const v = document.createElement("div");
  v.className = "visionneuse";
  v.setAttribute("role", "dialog"); v.setAttribute("aria-modal", "true"); v.setAttribute("aria-label", i.nom);
  v.innerHTML = `<img alt=""><button type="button" class="btn btn--icone visionneuse__fermer" aria-label="Close" title="Close">${ic("croix")}</button>`;
  v.querySelector("img").src = i.image;
  v.querySelector("img").alt = i.nom;
  v.onclick = () => v.remove();
  document.body.appendChild(v);
  v.querySelector("button").focus();
}
// a pasted screenshot (Ctrl+V) is attached like a dropped file
$("saisie").addEventListener("paste", (e) => {
  const fs = [...((e.clipboardData || {}).files || [])].filter((f) => f.type.startsWith("image/"));
  if (!fs.length) return;
  e.preventDefault();
  ajouterFichiers(fs);
});
function blocFichier(f) {                       // the file's text, fenced with more backticks than it contains
  const plusLong = Math.max(2, ...(f.texte.match(/`+/g) || []).map((s) => s.length));
  const clotures = "`".repeat(plusLong + 1);
  return `File: ${f.nom}\n${clotures}\n${f.texte}\n${clotures}`;
}
$("btn-joindre").onclick = () => $("fichiers").click();
$("fichiers").onchange = () => { ajouterFichiers($("fichiers").files); $("fichiers").value = ""; };
for (const cible of [$("fil-defile"), $("saisie-form")]) {
  cible.addEventListener("dragover", (e) => {
    if (![...((e.dataTransfer || {}).types || [])].includes("Files")) return;
    e.preventDefault();
    $("saisie-form").classList.add("depot");
  });
  cible.addEventListener("dragleave", () => $("saisie-form").classList.remove("depot"));
  cible.addEventListener("drop", (e) => {
    $("saisie-form").classList.remove("depot");
    if (!e.dataTransfer || !e.dataTransfer.files.length) return;
    e.preventDefault();
    ajouterFichiers(e.dataTransfer.files);
    $("saisie").focus();
  });
}

// ------------------------------------------------------------------ send, follow the response, stop it
let enCours = null;          // {controleur, c, m, el, idClient, debut, premier, debutRefl, finRefl, morceaux, phase}
function occupe(oui) {
  $("btn-arreter").hidden = !oui;
  $("btn-envoyer").disabled = oui;
  $("saisie-aide").hidden = oui;
}
function ajusterSaisie() {
  const t = $("saisie");
  t.style.height = "auto";
  t.style.height = `${Math.min(t.scrollHeight, Math.round(innerHeight * 0.4))}px`;
}
$("saisie").addEventListener("input", ajusterSaisie);
$("saisie").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); envoyer(); }
});
$("saisie-form").onsubmit = (e) => { e.preventDefault(); envoyer(); };
$("btn-arreter").onclick = () => { if (enCours) enCours.controleur.abort(); };

function messagesApi(c, sauf) {
  const out = [];
  for (const m of c.messages) {
    if (m === sauf) continue;
    if (m.role === "user") {
      const texteMsg = [m.texte, ...(m.fichiers || []).map(blocFichier)].filter((s) => s).join("\n\n");
      // its images as bytes, after its text; an image this browser no longer keeps (storage full) is mentioned
      // to the model instead of vanishing
      const imgs = m.images || [];
      out.push({role: "user", content: !imgs.length ? texteMsg : [...(texteMsg ? [{type: "text", text: texteMsg}] : []),
        ...imgs.map((i) => (i.image ? {type: "image_url", image_url: {url: i.image}}
                                    : {type: "text", text: `[image "${i.nom}": no longer kept by this browser]`}))]});
    } else {
      // a turn without text (only thinking, a stop, an error) stays a Lapilli turn: the history alternates.
      // The thinking goes back with its answer: Flash-Next's chat template keeps it by default, and the server then
      // recognizes its own memory (a follow-up reuses about 95% of its message and starts in 0.2 s, instead of
      // reading the whole previous answer again)
      const msg = {role: "assistant", content: m.texte || ""};
      if (m.reflexion) msg.reasoning_content = m.reflexion;
      // its tool calls go back too (the server recognizes its own memory); a cut-off call (incomplete JSON) would be
      // refused by the server: it stays on screen only
      const appels = aDesOutils(m) ? m.outils.filter((o) => o.nom && argsValides(o.args)) : [];
      if (appels.length) msg.tool_calls = appels.map((o, k) => ({id: o.id || `call-${k + 1}`, type: "function",
                                                                 function: {name: o.nom, arguments: o.args || "{}"}}));
      out.push(msg);
    }
  }
  return out;
}
function envoyer() {
  const texte = $("saisie").value.trim();
  if ((!texte && !pieces.length) || enCours) return;
  let c = convCourante();
  if (!c) {
    c = {id: nouvelId(), titre: titreDe(texte, pieces), cree: Date.now(), modifie: Date.now(), messages: []};
    convs.unshift(c);
    courante = c.id;
  }
  c.messages.push({role: "user", texte, fichiers: pieces.filter((p) => !p.image).map((p) => ({nom: p.nom, texte: p.texte})),
                   images: pieces.filter((p) => p.image).map((p) => ({nom: p.nom, image: p.image, l: p.l, h: p.h})),
                   temps: Date.now()});
  pieces = []; dessinerPieces();
  $("saisie").value = ""; ajusterSaisie();
  demander(c);
}
function regenerer() {
  const c = convCourante();
  if (!c || enCours) return;
  const dernier = c.messages[c.messages.length - 1];
  if (!dernier || dernier.role !== "assistant") return;
  c.messages.pop();
  demander(c);
}

let peintureDemandee = 0, dernierePeinture = 0;
function peindreBientot() {
  if (peintureDemandee) return;
  const taille = enCours ? enCours.m.texte.length : 0;
  const pause = taille > 80000 ? 300 : taille > 20000 ? 150 : 40;    // a long text is re-rendered less often
  const attente = Math.max(0, pause - (performance.now() - dernierePeinture));
  peintureDemandee = setTimeout(() => requestAnimationFrame(() => {
    peintureDemandee = 0; dernierePeinture = performance.now();
    if (enCours) { majLapilli(enCours.el, enCours.m, true); activite(); descendre(); }
  }), attente);
}

function activite() {
  if (!enCours) return;
  const {el, m} = enCours;
  const box = el.querySelector(".activite");
  const live = (mesures && mesures.live) || {};
  const maintenant = performance.now();
  const dur = (maintenant - enCours.debut) / 1000;
  const serveur = live.state === "generating" && live.generated ? live.generated : null;
  const jetons = Math.max(serveur || 0, enCours.morceaux);
  const vitesse = live.state === "generating" && live.tok_s ? live.tok_s
    : enCours.premier && maintenant - enCours.premier > 800 ? enCours.morceaux / ((maintenant - enCours.premier) / 1000) : null;
  let phase = enCours.phase, txt = [], barreLue = null, bouton = "";
  if (echecs >= 3 && phase !== "ecriture" && phase !== "reflexion" && phase !== "outil") {
    txt = ["Server unreachable", `${nf(dur, 0)} s`];
  } else if (phase === "envoi" || phase === "lecture") {
    if (live.state === "loading") {
      phase = "lecture";
      txt = ["Loading the model", (live.load || {}).percent != null ? `${nf(live.load.percent)}%` : null];
      barreLue = ((live.load || {}).percent || 0) / 100;
    } else if (live.queued > 0 && live.state !== "reading") {
      txt = ["Waiting in line", `${nf(live.queued)} ahead`];
    } else {
      phase = "lecture";
      const lu = live.state === "reading" ? partLue(live) : null;
      txt = ["Reading your message", live.state === "reading" && live.prompt_tokens ? `${nf(live.prompt_tokens)} tokens` : null,
             lu != null ? `${nf(lu * 100)}%` : null, dur >= 1 ? `${nf(dur, 0)} s` : null];
      barreLue = live.state === "reading" ? lu || 0 : null;
    }
  } else if (phase === "reflexion") {
    const s = enCours.debutRefl ? (maintenant - enCours.debutRefl) / 1000 : 0;
    txt = [enCours.maintenantDemande ? "End of thinking requested" : "Thinking", `${nf(s, 0)} s`, `${nf(jetons)} tokens`, vitesse ? `${nf(vitesse, 1)} tokens/s` : null];
    if (!enCours.maintenantDemande) bouton = `<button type="button" class="btn btn--doux" data-repondre-maintenant title="Stop thinking and answer right away">${ic("maintenant", "ic ic--p")}Answer now</button>`;
  } else if (phase === "outil") {
    const o = aDesOutils(m) ? m.outils[m.outils.length - 1] : null;
    txt = ["Tool call", o && o.nom ? o.nom : null, `${nf(jetons)} tokens`, vitesse ? `${nf(vitesse, 1)} tokens/s` : null];
  } else {
    txt = ["Writing", `${nf(jetons)} tokens`, vitesse ? `${nf(vitesse, 1)} tokens/s` : null];
  }
  box.dataset.phase = phase;
  const cle = JSON.stringify([phase, txt, barreLue == null ? null : Math.round(barreLue * 100), bouton ? 1 : 0]);
  if (box._cle === cle) return;
  box._cle = cle;
  box.innerHTML = `<span class="activite__texte"><span class="activite__point"></span>${morceaux(txt)}</span>` +
    (barreLue != null ? `<span class="activite__barre"><i style="width:${(barreLue * 100).toFixed(1)}%"></i></span>` : "") + bouton;
}

async function repondreMaintenant() {
  if (!enCours || enCours.maintenantDemande) return;
  const en = enCours;
  en.maintenantDemande = true;
  activite();
  try {
    const r = await fetch("v1/answer_now", {method: "POST", headers: entetes(true), body: JSON.stringify({request_id: en.idClient})});
    if (!r.ok) throw await refus(r);
    annonce("info", "Answer requested", "Lapilli is wrapping up its thinking to answer.", 3000);
  } catch (e) {
    if (e.code !== "not_thinking") en.maintenantDemande = false;
    annonce(e.code === "not_thinking" ? "info" : "erreur", "Answer now", String(e.message || e), 6000);
    activite();
  }
}

async function demander(c) {
  const m = {role: "assistant", texte: "", reflexion: "", temps: Date.now()};
  c.messages.push(m);
  c.modifie = Date.now();
  garderConvs(); dessinerConvs(); dessinerFil();
  const el = elements.get(m);
  const controleur = new AbortController();
  enCours = {controleur, c, m, el, idClient: `lapilli-${nouvelId()}`, debut: performance.now(), premier: null, debutRefl: null, finRefl: null,
             morceaux: 0, phase: "envoi", maintenantDemande: false};
  suivre = true;
  occupe(true);
  majLapilli(el, m, true);
  activite();
  const minuterie = setInterval(activite, 250);
  const garde = setInterval(() => { c.modifie = Date.now(); garderConvs(); }, 5000);
  relever();

  const corps = {model: (mesures && mesures.engine && mesures.engine.model) || sante.model || "flash-next", messages: messagesApi(c, m), stream: true,
                 reasoning_effort: reglages.reflexion};
  if (reglages.longueur > 0) {   // a cap the server fits to the room the conversation leaves (fit_max_tokens), never a refusal
    corps.max_tokens = reglages.longueur;
    corps.fit_max_tokens = true;
  }
  if (reglages.temperature != null) corps.temperature = reglages.temperature;
  if (reglages.temperature !== 0) {
    if (reglages.top_p != null) corps.top_p = reglages.top_p;
    if (reglages.top_k != null) corps.top_k = reglages.top_k;
  }
  if (reglages.graine !== "" && Number.isFinite(+reglages.graine)) corps.seed = +reglages.graine;
  m.limite = corps.max_tokens || null;

  let usage = null, timings = null, fin = null;
  try {
    const r = await fetch("v1/chat/completions", {method: "POST", headers: {...entetes(true), "x-client-request-id": enCours.idClient},
                                                    body: JSON.stringify(corps), signal: controleur.signal});
    if (!r.ok) throw await refus(r);
    enCours.phase = "lecture";
    activite();
    const lecteur = r.body.getReader(), dec = new TextDecoder();
    let tampon = "";
    const traiter = (ligne) => {
      if (!ligne.startsWith("data:")) return;                   // ": keep-alive" while a long message is being read
      const donnee = ligne.slice(5).trim();
      if (!donnee || donnee === "[DONE]") return;
      let j;
      try { j = JSON.parse(donnee); } catch (e) { return; }
      if (j.error) throw new Error(j.error.message || "The engine reported an error.");
      if (j.usage) usage = j.usage;
      if (j.timings) timings = j.timings;
      const ch = (j.choices && j.choices[0]) || {};
      if (ch.finish_reason) fin = ch.finish_reason;
      const d = ch.delta || {};
      const t = performance.now();
      if (d.reasoning_content) {
        if (!enCours.premier) enCours.premier = t;
        if (!enCours.debutRefl) enCours.debutRefl = t;
        m.reflexion += d.reasoning_content;
        enCours.morceaux++;
        enCours.phase = "reflexion";
      }
      if (d.content) {
        if (!enCours.premier) enCours.premier = t;
        if (enCours.debutRefl && !enCours.finRefl) { enCours.finRefl = t; m.reflexionS = (t - enCours.debutRefl) / 1000; }
        m.texte += d.content;
        enCours.morceaux++;
        enCours.phase = "ecriture";
      }
      if (Array.isArray(d.tool_calls) && d.tool_calls.length) {   // sent separately
        if (!enCours.premier) enCours.premier = t;
        if (enCours.debutRefl && !enCours.finRefl) { enCours.finRefl = t; m.reflexionS = (t - enCours.debutRefl) / 1000; }
        if (!Array.isArray(m.outils)) m.outils = [];
        for (const tc of d.tool_calls) {
          const k = Number.isInteger(tc.index) && tc.index >= 0 && tc.index < 64 ? tc.index : m.outils.length;
          while (m.outils.length <= k) m.outils.push({id: "", nom: "", args: ""});
          const o = m.outils[k], fn = tc.function || {};
          if (tc.id) o.id = String(tc.id);
          if (fn.name) o.nom += String(fn.name);
          if (fn.arguments != null) o.args += typeof fn.arguments === "string" ? fn.arguments : JSON.stringify(fn.arguments);
        }
        enCours.morceaux++;
        enCours.phase = "outil";
      }
      if (d.reasoning_content || d.content || d.tool_calls) peindreBientot();
    };
    for (;;) {
      const {value, done} = await lecteur.read();
      if (done) break;
      tampon += dec.decode(value, {stream: true});
      let k;
      while ((k = tampon.indexOf("\n")) >= 0) {
        const ligne = tampon.slice(0, k).trim();
        tampon = tampon.slice(k + 1);
        traiter(ligne);
      }
    }
    tampon += dec.decode();
    if (tampon.trim()) traiter(tampon.trim());                  // a last line without a line ending
  } catch (e) {
    if (e.name === "AbortError") m.arretee = true;
    else { m.erreur = String(e.message || e); annonce("erreur", "The response failed", m.erreur, 9000); }
  }
  clearInterval(minuterie);
  clearInterval(garde);
  clearTimeout(peintureDemandee); peintureDemandee = 0;
  const maintenant = performance.now();
  if (enCours.debutRefl && m.reflexionS == null) m.reflexionS = ((enCours.finRefl || maintenant) - enCours.debutRefl) / 1000;
  m.fin = fin;
  const duree = (maintenant - enCours.debut) / 1000;
  const n = usage ? usage.completion_tokens : null;
  m.ecrits = n;   // the end notice tells a length reached from a context filled
  const meta = [];
  if (m.arretee) meta.push("Stopped");
  if (m.erreur) meta.push("Error");
  if (n != null) meta.push(`${nf(n)} tokens`);
  else if (enCours.morceaux) meta.push(`about ${nf(enCours.morceaux)} tokens`);
  const vit = timings && timings.predicted_per_second ? timings.predicted_per_second
    : n && enCours.premier && maintenant - enCours.premier > 250 ? n / ((maintenant - enCours.premier) / 1000) : null;
  if (vit) meta.push(`${nf(vit, 1)} tokens/s`);
  // the reading speed only means something for a long message (a short one is read in an instant, fixed costs included)
  if (timings && timings.prompt_n >= 1024 && timings.prompt_ms >= 100 && timings.prompt_per_second) meta.push(`reading ${nf(timings.prompt_per_second)} tokens/s`);
  if (m.reflexionS != null) meta.push(`thinking ${nf(m.reflexionS, 1)} s`);
  meta.push(`${nf(duree, 1)} s`);
  m.meta = meta;
  const titre = [];
  if (usage) titre.push(`Message read: ${nf(usage.prompt_tokens)} tokens${usage.prompt_tokens_details && usage.prompt_tokens_details.cached_tokens ? ` (${nf(usage.prompt_tokens_details.cached_tokens)} reused from the cache)` : ""}`);
  if (timings && timings.draft_n) titre.push(`Drafts accepted: ${nf(timings.draft_n_accepted)} of ${nf(timings.draft_n)}`);
  m.metaTitre = titre.join("\n");
  const fini = enCours;
  enCours = null;
  occupe(false);
  c.modifie = Date.now();
  garderConvs(); dessinerConvs();
  majLapilli(fini.el, m, false);
  // the "Regenerate" button lives on the last message only: redraw the meta lines of the others
  for (const x of c.messages) { const e2 = elements.get(x); if (e2 && x !== m) majLapilli(e2, x, false); }
  descendre();
  relever();
  if (m.arretee) { setTimeout(relever, 250); setTimeout(relever, 800); }   // the server sees the closed connection at the next token
}

// ------------------------------------------------------------------ save the conversation (Markdown)
function exporter() {
  const c = convCourante();
  if (!c || !c.messages.length) { annonce("info", "Nothing to save", "The conversation is empty.", 3000); return; }
  const md = [`# ${c.titre}`, "", `_${quand(c.cree)}, with Lapilli (${(mesures && mesures.engine && mesures.engine.model) || sante.model})_`, ""];
  for (const m of c.messages) {
    if (m.role === "user") {
      md.push("## You", "", m.texte || "");
      for (const f of m.fichiers || []) md.push("", `_Attached file: ${f.nom}_`);
      for (const i of m.images || []) md.push("", `_Attached image: ${i.nom}_`);
      md.push("");
    } else {
      md.push("## Lapilli", "");
      if (m.reflexion) md.push("<details><summary>Thinking</summary>", "", m.reflexion, "", "</details>", "");
      md.push(m.texte || m.erreur || "", "");
      for (const o of aDesOutils(m) ? m.outils : []) md.push(...outilEnMarkdown(o), "");
    }
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([md.join("\n")], {type: "text/markdown;charset=utf-8"}));
  const d = new Date(), z = (x) => String(x).padStart(2, "0");
  a.download = `lapilli-${d.getFullYear()}-${z(d.getMonth() + 1)}-${z(d.getDate())}-${z(d.getHours())}${z(d.getMinutes())}.md`;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  annonce("succes", "Conversation saved", a.download, 3000);
}

// ------------------------------------------------------------------ startup
(function demarrer() {
  ouvrirConvs(etroit() ? false : memoire.lire("panneau", true));
  chargerTiroir();
  occupe(false);
  dessinerConvs();
  dessinerFil();
  montrerVue(location.hash.slice(1) || "discussion");
  lireSante();
  lireProps();
  relever();
  window.addEventListener("resize", () => { ajusterSaisie(); majVersBas(); });
  // marked arrives with "defer" just before this script; if it is missing, the text stays readable as is
  if (!window.marked) annonce("alerte", "Formatting unavailable", "Text is shown raw (marked could not be loaded).", 8000);
})();
$("btn-exporter").onclick = exporter;
