"use strict";
const GLYPH = { k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟︎" };
const NAMES = { k: "king", q: "queen", r: "rook", b: "bishop", n: "knight", p: "pawn" };
const FILES = "abcdefgh";
const $ = (id) => document.getElementById(id);
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

let S = null;                     // latest server state
let selected = null;              // square name
let flipped = false;              // manual flip, on top of the human's color
let busy = false;
let want = { bot: "echo", dial: "balanced", color: "white" };
let typed = "";
let unlockTimer = null;
let lastTypedShown = -1;          // chat index already typed out
let viewPly = null;               // null = the live position; otherwise the number of plies shown while stepping back
let showHint = false;             // the best-move arrow from the Hint button

// Practice-mode move signs. "sym" is what the badge shows; the colors live in styles.css (.L-<label>).
const SIGNS = {
  brilliant: { sym: "!!", name: "Brilliant" }, great: { sym: "!", name: "Great move" },
  best: { sym: "\u2605", name: "Best move" }, excellent: { sym: "\u2713", name: "Excellent" },
  good: { sym: "\u2713", name: "Good" }, book: { sym: "B", name: "Book move" },
  forced: { sym: "\u2192", name: "Forced" }, inaccuracy: { sym: "?!", name: "Inaccuracy" },
  mistake: { sym: "?", name: "Mistake" }, blunder: { sym: "??", name: "Blunder" },
};
const BAD = new Set(["inaccuracy", "mistake", "blunder"]);
const SHEET_SIGNS = new Set(["brilliant", "great", "best", "book", "inaccuracy", "mistake", "blunder"]);

const plyShown = () => (viewPly === null ? S.san.length : viewPly);
const viewing = () => viewPly !== null && viewPly < S.san.length;
function setView(p) {
  showHint = false;
  viewPly = p >= S.san.length ? null : Math.max(0, p);
  selected = null;
  render();
}

function toast(msg, err) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast show" + (err ? " err" : "");
  clearTimeout(toast.t);
  toast.t = setTimeout(() => (t.className = "toast"), err ? 4200 : 2800);
}

async function api(path, body) {
  const opts = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({ error: "unreadable reply" }));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function parseFen(fen) {
  const map = {};
  fen.split(" ")[0].split("/").forEach((row, i) => {
    let f = 0;
    for (const ch of row) {
      if (/\d/.test(ch)) f += +ch;
      else { map[FILES[f] + (8 - i)] = ch; f++; }
    }
  });
  return map;
}

function orientation() { return (S.human === "w") !== flipped ? "w" : "b"; }

// Straight arrow between two squares as SVG (board units: one square = 1), oriented like the board.
function arrowSvg(uci, cls, o) {
  const pt = (sq) => {
    const f = FILES.indexOf(sq[0]), r = +sq[1];
    return [(o === "w" ? f : 7 - f) + 0.5, (o === "w" ? 8 - r : r - 1) + 0.5];
  };
  const [x1, y1] = pt(uci.slice(0, 2)), [x2, y2] = pt(uci.slice(2, 4));
  const len = Math.hypot(x2 - x1, y2 - y1);
  if (!len) return "";
  const ux = (x2 - x1) / len, uy = (y2 - y1) / len, head = 0.46, half = 0.3;
  const bx = x2 - ux * head, by = y2 - uy * head;
  const f = (n) => n.toFixed(3);
  return `<g class="arrow ${cls}"><line x1="${f(x1 + ux * 0.1)}" y1="${f(y1 + uy * 0.1)}" x2="${f(bx)}" y2="${f(by)}"/>` +
    `<polygon points="${f(x2)},${f(y2)} ${f(bx - uy * half)},${f(by + ux * half)} ${f(bx + uy * half)},${f(by - ux * half)}"/></g>`;
}

// Arrows you draw yourself: right-drag on a computer, press-and-hold then drag on a phone. Drawing the same arrow again
// removes it, and a click on the board, a move, or stepping through the game clears them all.
let drawn = [], drawnSig = "", draw = null, noClickUntil = 0;
function sqAt(x, y) {
  const el = document.elementFromPoint(x, y), b = el && el.closest ? el.closest(".sq") : null;
  return b ? b.dataset.sq : null;
}
// Engine arrows (Hint, and the better move while stepping back) plus the ones you drew.
function paintArrows() {
  const board = $("board"), old = board.querySelector(".arrows");
  if (old) old.remove();
  if (!S) return;
  const past = viewing(), ply = plyShown(), o = orientation();
  const sig = S.fen + "|" + ply + "|" + past;
  if (sig !== drawnSig) { drawn = []; drawnSig = sig; }
  let arrows = "";
  if (S.practice && !past && showHint && S.hint) arrows = arrowSvg(S.hint.uci, "hint", o);
  const next = past && S.practice ? S.reviews[ply] : null;
  if (next && BAD.has(next.label) && next.best) arrows = arrowSvg(next.best, "better", o);
  for (const u of drawn) arrows += arrowSvg(u, "mine", o);
  if (draw && draw.active && draw.to !== draw.from) arrows += arrowSvg(draw.from + draw.to, "mine preview", o);
  if (!arrows) return;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "arrows");
  svg.setAttribute("viewBox", "0 0 8 8");
  svg.setAttribute("aria-hidden", "true");
  svg.innerHTML = arrows;
  board.append(svg);
}
function endDraw() {
  if (draw) clearTimeout(draw.timer);
  const d = draw;
  draw = null;
  if (d && d.active) paintArrows();
  return d;
}
{
  const boardEl = $("board");
  boardEl.addEventListener("contextmenu", (e) => e.preventDefault());
  boardEl.addEventListener("pointerdown", (e) => {
    if (!S || (e.pointerType === "mouse" && e.button !== 2)) return;
    const from = sqAt(e.clientX, e.clientY);
    if (!from) return;
    draw = { from, to: from, active: e.pointerType === "mouse", x: e.clientX, y: e.clientY };
    if (draw.active) e.preventDefault();
    else draw.timer = setTimeout(() => {
      if (!draw) return;
      draw.active = true;
      if (navigator.vibrate) navigator.vibrate(12);
    }, 380);
  });
  addEventListener("pointermove", (e) => {
    if (!draw) return;
    if (!draw.active) { if (Math.hypot(e.clientX - draw.x, e.clientY - draw.y) > 10) endDraw(); return; }
    const to = sqAt(e.clientX, e.clientY);
    if (to && to !== draw.to) { draw.to = to; paintArrows(); }
  });
  addEventListener("pointerup", (e) => {
    const d = endDraw();
    if (!d || !d.active) return;
    if (e.pointerType !== "mouse") noClickUntil = performance.now() + 400;           // the touch that ends a drag must not also select a square
    if (d.to === d.from) return;
    const u = d.from + d.to, i = drawn.indexOf(u);
    if (i >= 0) drawn.splice(i, 1); else drawn.push(u);
    paintArrows();
  });
  addEventListener("pointercancel", () => { endDraw(); });
  addEventListener("touchmove", (e) => { if (draw && draw.active && e.cancelable) e.preventDefault(); }, { passive: false });
}

function renderBoard() {
  const past = viewing(), ply = plyShown();
  const pieces = parseFen(past ? S.fens[ply] : S.fen);
  const last = past ? (ply > 0 ? S.ucis[ply - 1] : null) : S.last;
  const sign = S.practice && ply > 0 ? S.reviews[ply - 1] : null;
  const o = orientation();
  const targets = new Set(selected && !past ? S.legal.filter((u) => u.startsWith(selected)).map((u) => u.slice(2, 4)) : []);
  const board = $("board");
  const focusSq = document.activeElement && board.contains(document.activeElement) ? document.activeElement.dataset.sq : null;
  board.className = "board" + (busy || S.pending || S.result || past ? " locked" : "");
  board.textContent = "";
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const file = o === "w" ? c : 7 - c;
      const rank = o === "w" ? 8 - r : r + 1;
      const name = FILES[file] + rank;
      const ch = pieces[name];
      const btn = document.createElement("button");
      btn.type = "button";
      btn.dataset.sq = name;
      btn.className = "sq" + ((file + rank) % 2 === 1 ? "" : " dark");
      if (last && (last.slice(0, 2) === name || last.slice(2, 4) === name)) btn.classList.add("last");
      if (selected === name && !past) btn.classList.add("sel");
      if (S.check === name && !past) btn.classList.add("check");
      if (targets.has(name)) btn.classList.add("target", ch ? "has-piece" : "empty");
      if (c === 0) { const k = document.createElement("span"); k.className = "coord rank"; k.textContent = rank; btn.append(k); }
      if (r === 7) { const k = document.createElement("span"); k.className = "coord file"; k.textContent = FILES[file]; btn.append(k); }
      let label = name;
      if (ch) {
        const p = document.createElement("span");
        const white = ch === ch.toUpperCase();
        p.className = "piece " + (white ? "w" : "b");
        p.textContent = GLYPH[ch.toLowerCase()];
        btn.append(p);
        label += ", " + (white ? "white " : "black ") + NAMES[ch.toLowerCase()];
      }
      if (sign && SIGNS[sign.label] && last && last.slice(2, 4) === name) {
        const badge = document.createElement("span");
        badge.className = "badge L-" + sign.label;
        badge.textContent = SIGNS[sign.label].sym;
        badge.setAttribute("aria-hidden", "true");
        btn.append(badge);
        label += ", last move: " + SIGNS[sign.label].name;
      }
      btn.setAttribute("aria-label", label);
      btn.addEventListener("click", () => { if (performance.now() < noClickUntil) return; if (drawn.length) { drawn = []; paintArrows(); }
      onSquare(name); });
      board.append(btn);
    }
  }
  paintArrows();
  if (focusSq) { const again = board.querySelector(`[data-sq="${focusSq}"]`); if (again) again.focus(); }
}

function renderCaptured() {
  const mk = (side) => {
    const wrap = document.createElement("div");
    wrap.className = "captured";
    const row = document.createElement("span");
    row.className = "row";
    // pieces `side` has captured are of the other color
    const cls = side === "w" ? "b" : "w";
    row.innerHTML = S.captured[side].map((p) => `<span class="${cls}">${GLYPH[p]}</span>`).join("");
    wrap.append(row);
    const diff = (side === "w" ? 1 : -1) * S.material;
    if (diff > 0) { const d = document.createElement("span"); d.textContent = "+" + diff; wrap.append(d); }
    return wrap;
  };
  const o = orientation();
  const top = o === "w" ? "b" : "w", bottom = o;
  $("capTop").replaceChildren(...mk(top).childNodes);
  $("capBottom").replaceChildren(...mk(bottom).childNodes);
}

function renderSheet() {
  const box = $("sheet");
  if (!S.san.length) { box.innerHTML = '<p class="empty">No moves yet.</p>'; return; }
  const list = document.createElement("ol");
  list.className = "moves";
  const shown = plyShown() - 1;                          // the move whose resulting position is on the board
  for (let i = 0; i < S.san.length; i += 2) {
    const li = document.createElement("li");
    const n = document.createElement("span");
    n.className = "n"; n.textContent = i / 2 + 1 + ".";
    li.append(n);
    for (const ply of [i, i + 1]) {
      if (S.san[ply] === undefined) { li.append(document.createElement("span")); continue; }
      const m = document.createElement("button");
      m.type = "button";
      m.className = "m" + (ply === shown ? " cur" : "");
      m.append(S.san[ply]);
      const rv = S.practice ? S.reviews[ply] : null;
      if (rv && SHEET_SIGNS.has(rv.label)) {
        const mk = document.createElement("span");
        mk.className = "mk L-" + rv.label;
        mk.textContent = SIGNS[rv.label].sym;
        mk.title = SIGNS[rv.label].name;
        m.append(mk);
        m.setAttribute("aria-label", S.san[ply] + ", " + SIGNS[rv.label].name);
      }
      m.addEventListener("click", () => setView(ply + 1));
      li.append(m);
    }
    list.append(li);
  }
  box.replaceChildren(list);
  const wrap = box.parentElement;                      // scrolls down on wide screens, along on phones
  const cur = box.querySelector(".cur");
  if (cur && viewing()) {                              // centre the shown move inside the sheet only, never the whole page
    wrap.scrollTo({ left: cur.offsetLeft - wrap.clientWidth / 2 + cur.offsetWidth / 2,
                    top: cur.offsetTop - wrap.clientHeight / 2 + cur.offsetHeight / 2 });
  } else { wrap.scrollTop = wrap.scrollHeight; wrap.scrollLeft = wrap.scrollWidth; }
}

function typeOut(el, text) {
  if (reduceMotion) { el.textContent = text; return; }
  let i = 0;
  el.textContent = "";
  const tick = () => { el.textContent = text.slice(0, ++i); if (i < text.length) setTimeout(tick, 16); };
  tick();
}

function renderChat() {
  const chat = $("chat");
  chat.textContent = "";
  const n = S.chat.length;
  const hall = S.bot.persona === "fischer";
  const row = (b, old) => {
    const m = document.createElement("div");
    m.className = "msg" + (old ? " old" : "");
    const a = document.createElement("span");
    a.className = "mini" + (hall ? " hall" : "");
    a.setAttribute("aria-hidden", "true");
    a.innerHTML = hall ? "&#9812;" : "&#9822;";
    m.append(a, b);
    return m;
  };
  S.chat.forEach((c, i) => {
    const b = document.createElement("div");
    b.className = "bubble" + (i < n - 1 ? " old" : "");
    if (i === n - 1 && i > lastTypedShown) { typeOut(b, c.text); lastTypedShown = i; } else b.textContent = c.text;
    chat.append(row(b, i < n - 1));
  });
  const coach = coachInfo();
  if (coach) {
    const c = document.createElement("div");
    c.className = "bubble coach";
    const chip = document.createElement("span");
    chip.className = "chip L-" + coach.label;
    chip.textContent = SIGNS[coach.label].sym;
    const text = document.createElement("span");
    const lost = Math.round(coach.loss);
    text.textContent = `${SIGNS[coach.label].name}. ${coach.best_san ? "Better was " + coach.best_san + ". " : ""}` +
      (lost >= 1 ? `That cost about ${lost}% of your winning chances.` : "");
    const back = document.createElement("button");
    back.type = "button"; back.className = "btn"; back.textContent = "Take it back";
    back.addEventListener("click", takeBack);
    c.append(chip, text, back);
    chat.append(row(c, false));
  }
  if (S.result) {
    const v = document.createElement("div");
    v.className = "bubble verdict";
    const humanWhite = S.human === "w";
    const won = (S.result === "1-0") === humanWhite && S.result !== "1/2-1/2";
    v.textContent = S.result === "1/2-1/2" ? "Draw." : won ? "You won." : S.bot.name + " won.";
    if (S.saved) v.textContent += " Game saved to " + S.saved + ".";
    const again = document.createElement("button");
    again.type = "button"; again.className = "btn primary"; again.textContent = "New game";
    again.addEventListener("click", newGame);
    v.append(again);
    chat.append(row(v, false));
  } else if (busy || S.pending) {
    const t = document.createElement("div");
    t.className = "bubble thinking";
    t.innerHTML = '<span class="dots"><span></span><span></span><span></span></span>';
    chat.append(row(t, false));
  }
  chat.scrollTop = chat.scrollHeight;
}

// Practice mode: after your own move was an inaccuracy, mistake or blunder, offer to take it back and try again.
function coachInfo() {
  if (!S.practice || viewing() || S.result || !S.san.length) return null;
  let idx = S.san.length - 1;
  if ((idx % 2 === 0) !== (S.human === "w")) idx -= 1;          // the latest ply that was yours
  if (idx < 0 || idx < S.san.length - 2) return null;
  const rv = S.reviews[idx];
  return rv && BAD.has(rv.label) ? rv : null;
}

function segSet(id, attr, value) {
  document.querySelectorAll(`#${id} button`).forEach((b) => b.setAttribute("aria-checked", String(b.dataset[attr] === value)));
}

function renderBots(reveal) {
  const seg = $("botSeg");
  seg.textContent = "";
  S.bots.forEach((b) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.setAttribute("role", "radio");
    btn.dataset.bot = b.id;
    btn.textContent = b.name;
    btn.disabled = !b.available;
    btn.title = b.available ? b.blurb : b.note;
    if (reveal && b.id === reveal) btn.classList.add("reveal");
    btn.addEventListener("click", () => { want.bot = b.id; want.dial = b.id === "fischer" ? "sharp" : want.dial; segSet("botSeg", "bot", want.bot); segSet("dialSeg", "dial", want.dial); });
    seg.append(btn);
  });
  segSet("botSeg", "bot", want.bot);
}

function renderOpponent() {
  const b = S.bot;
  $("botName").textContent = b.name;
  $("botBlurb").textContent = b.blurb;
  const hall = b.persona === "fischer";
  document.body.classList.toggle("theme-hall", hall);
  const av = $("avatar");
  av.classList.toggle("hall", hall);
  av.innerHTML = hall ? "&#9812;" : "&#9822;";
}

const DIAL_HINT = {
  faithful: "Plays most like the profile and allows small inaccuracies.",
  balanced: "The profile's style with fewer slips.",
  sharp: "Near-perfect play; style only breaks ties.",
};

function renderStatus() {
  const el = $("status");
  const me = S.human === "w" ? "White" : "Black";
  let text;
  if (viewing()) text = `Viewing move ${plyShown()} of ${S.san.length}.`;
  else if (S.result) text = "Game over, " + S.result + ".";
  else if (busy || S.pending) text = S.bot.name + " is thinking.";
  else text = "You play " + me + ". Your move" + (S.check ? ", and you are in check." : ".");
  if (showHint && S.hint && !viewing()) text += " Hint: " + S.hint.san + ".";
  el.textContent = text;
  const op = $("opening");
  op.hidden = !S.opening;
  if (S.opening) op.textContent = S.opening.name + " \u00b7 " + S.opening.eco;
}

function renderControls() {
  $("dialHint").textContent = DIAL_HINT[want.dial] || "";
  const cap = (t) => t.charAt(0).toUpperCase() + t.slice(1);
  const botName = (S.bots.find((b) => b.id === want.bot) || S.bot).name;
  $("settingsNow").textContent = [botName, cap(want.dial), cap(want.color)].join(" \u00b7 ");
  $("undoBtn").disabled = busy || (!S.result && S.san.length < (S.human === S.turn ? 2 : 1)) || (S.result && !S.resigned);
  $("resignBtn").disabled = !!S.result || busy;
  const idx = plyShown();
  $("navFirst").disabled = $("navPrev").disabled = idx === 0;
  $("navNext").disabled = $("navLast").disabled = !viewing();
  const hintBtn = $("navHint");
  hintBtn.hidden = !S.practice;
  hintBtn.disabled = busy || S.pending || !!S.result || viewing() || !S.hint;
  hintBtn.setAttribute("aria-pressed", String(showHint && !hintBtn.disabled));
  const pb = $("practiceBtn");
  pb.setAttribute("aria-checked", String(S.practice));
  pb.textContent = S.practice ? "Practice: on" : "Practice";
  segSet("dialSeg", "dial", want.dial);
  segSet("colorSeg", "color", want.color);
}

function render(reveal) {
  renderOpponent(); renderStatus(); renderBoard(); renderCaptured(); renderSheet(); renderChat(); renderBots(reveal);
  renderControls();
}

async function afterState(state) {
  S = state;
  selected = null;
  viewPly = null;
  showHint = false;
  render();
  if (S.pending) {
    busy = true; render();
    try { S = await api("/api/reply", {}); } catch (e) { toast(e.message, true); }
    busy = false; selected = null; render();
  }
}

async function onSquare(name) {
  if (!S || busy || S.pending || S.result || viewing()) return;
  const pieces = parseFen(S.fen);
  if (selected) {
    const moves = S.legal.filter((u) => u.slice(0, 2) === selected && u.slice(2, 4) === name);
    if (moves.length === 1) return sendMove(moves[0]);
    if (moves.length > 1) return askPromotion(moves);
  }
  const ch = pieces[name];
  const mine = ch && ((ch === ch.toUpperCase()) === (S.human === "w"));
  selected = mine && S.legal.some((u) => u.startsWith(name)) && selected !== name ? name : null;
  renderBoard();
}

function askPromotion(moves) {
  const dlg = $("promo"), pick = $("promoPick");
  pick.textContent = "";
  ["q", "r", "b", "n"].forEach((p) => {
    const uci = moves.find((m) => m[4] === p);
    if (!uci) return;
    const b = document.createElement("button");
    b.type = "button"; b.textContent = GLYPH[p]; b.setAttribute("aria-label", NAMES[p]);
    b.style.color = S.human === "w" ? "#fbf8ef" : "#1c242b";
    b.style.webkitTextStroke = S.human === "w" ? "1.3px #1f262c" : "1px #0b0f13";
    b.addEventListener("click", () => { dlg.close(); sendMove(uci); });
    pick.append(b);
  });
  dlg.showModal();
}

async function sendMove(uci) {
  busy = true;
  let next = null;
  try { next = await api("/api/move", { uci }); } catch (e) { toast(e.message, true); selected = null; }
  busy = false;
  if (next) await afterState(next); else render();
}

// On a phone the settings fold away once a game starts, so the board stays in view.
const narrow = matchMedia("(max-width: 900px)");
const syncSettings = () => { $("settings").open = !narrow.matches; };
narrow.addEventListener("change", syncSettings);
syncSettings();

async function newGame() {
  if (narrow.matches) $("settings").open = false;
  busy = true;
  let next = null;
  try { lastTypedShown = -1; next = await api("/api/new", want); } catch (e) { toast(e.message, true); }
  busy = false;
  if (next) await afterState(next); else render();
}

async function takeBack() {
  try { await afterState(await api("/api/undo", {})); } catch (e) { toast(e.message, true); }
}

$("newBtn").onclick = newGame;
$("navFirst").onclick = () => setView(0);
$("navPrev").onclick = () => setView(plyShown() - 1);
$("navNext").onclick = () => setView(plyShown() + 1);
$("navLast").onclick = () => setView(S.san.length);
$("navHint").onclick = () => { showHint = !showHint; render(); };
$("practiceBtn").onclick = async () => {
  try { S = await api("/api/practice", { on: !S.practice }); showHint = false; selected = null; render(); }
  catch (e) { toast(e.message, true); }
};
$("flipBtn").onclick = () => { flipped = !flipped; render(); };
$("undoBtn").onclick = takeBack;
$("resignBtn").onclick = () => $("confirm").showModal();
$("confirm").addEventListener("close", async () => {
  if ($("confirm").returnValue !== "resign") return;
  try { await afterState(await api("/api/resign", {})); } catch (e) { toast(e.message, true); }
});
$("dialSeg").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { want.dial = b.dataset.dial; renderControls(); } });
$("colorSeg").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { want.color = b.dataset.color; renderControls(); } });

// ---- the hidden extra: type the right word anywhere on the page ----
async function unlockWith(text) {
  const was = S && S.unlocked;
  const r = await api("/api/unlock", { typed: text });
  if (r.unlocked && !was) {
    try { localStorage.setItem("echo.unlock", text); } catch (_) { /* private mode */ }
    S.bots = r.bots; S.unlocked = true;
    renderBots("fischer");
    const f = r.bots.find((b) => b.id === "fischer");
    toast(f && f.available ? "A new opponent is waiting: Fischer." : "Unlocked: Fischer. " + (f ? f.note : ""));
  }
  return r.unlocked;
}
function tryUnlock() {
  clearTimeout(unlockTimer);
  unlockTimer = setTimeout(async () => {
    if (typed.length < 4) return;
    try { await unlockWith(typed); } catch (_) { /* ignore: typing must never show errors */ }
  }, 250);
}

// Phones have no keyboard events: tap the opponent's portrait five times to get a small box for the code.
let taps = 0, tapTimer = null;
$("avatar").addEventListener("click", () => {
  taps += 1;
  clearTimeout(tapTimer);
  tapTimer = setTimeout(() => { taps = 0; }, 1500);
  if (taps >= 5) {
    taps = 0;
    const dlg = $("code");
    $("codeInput").value = "";
    if (!dlg.open) dlg.showModal();
    $("codeInput").focus();
  }
});
$("codeForm").addEventListener("submit", async (e) => {
  if (!e.submitter || e.submitter.value !== "go") return;
  const text = $("codeInput").value.replace(/[^a-z0-9]/gi, "").toLowerCase();
  if (text.length < 4) return;
  try {
    if (!(await unlockWith(text))) toast("Not that one.");
  } catch (_) { toast("Not that one."); }
});

addEventListener("keydown", (e) => {
  if (e.key === "Escape" && selected && !document.querySelector("dialog[open]")) { selected = null; if (S) renderBoard(); return; }
  if (document.querySelector("dialog[open]")) return;
  if (S && !e.ctrlKey && !e.metaKey && !e.altKey && ["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key) &&
      !/^(input|textarea|select)$/i.test(document.activeElement && document.activeElement.tagName)) {
    e.preventDefault();
    if (e.key === "ArrowLeft") setView(plyShown() - 1);
    else if (e.key === "ArrowRight") setView(plyShown() + 1);
    else if (e.key === "Home") setView(0);
    else setView(S.san.length);
    return;
  }
  if (e.ctrlKey || e.metaKey || e.altKey || e.key.length !== 1) return;
  if (/^(input|textarea|select)$/i.test(document.activeElement && document.activeElement.tagName)) return;
  typed = (typed + e.key.toLowerCase()).replace(/[^a-z0-9]/g, "").slice(-24);
  tryUnlock();
});

(async function init() {
  try {
    let stored = null;
    try { stored = localStorage.getItem("echo.unlock"); } catch (_) { /* ignore */ }
    if (stored) await api("/api/unlock", { typed: stored }).catch(() => {});
    S = await api("/api/state");
    want.color = S.human === "w" ? "white" : "black";
    want.dial = S.dial;
    want.bot = S.bot.id;
    lastTypedShown = S.chat.length - 1;
    render();
    if (S.pending) await afterState(S);
  } catch (e) { toast("Could not reach the Echo server: " + e.message, true); }
})();
