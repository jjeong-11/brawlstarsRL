/* webui/static/app.js
 *
 * Pick a brawler, start or stop a session, watch the trace, read the numbers.
 *
 * THREE INDEPENDENT CHANNELS, deliberately
 *   roster   Inlined in the page as JSON. Selection is a pure local lookup with
 *            no request behind it, so there is nothing that can fail per click.
 *   frames   The <img> takes the MJPEG stream directly — no JS in the path, so
 *            frames render at tick rate however busy the main thread is.
 *   status   Polled twice a second, and it only writes to the DOM when a value
 *            actually changed. A 500ms full repaint of the controls is what
 *            makes a page feel broken under the cursor.
 *
 * MOTION (see the apple-design notes in style.css)
 *   Anything the user can touch mid-flight is driven by the spring below rather
 *   than a CSS transition, because a CSS transition cannot be grabbed and
 *   reversed — it always animates to its target from wherever it started, and
 *   re-targeting restarts it with a visible jump. The spring re-targets from
 *   the CURRENT value and CURRENT velocity, which is the whole point.
 */
'use strict';

const $ = (id) => document.getElementById(id);

const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)');

/* ── a critically damped spring ────────────────────────────────────────
 * Apple's two parameters rather than mass/stiffness/damping: `response` is how
 * quickly it reaches the target in seconds, `bounce` is overshoot. Default is
 * critically damped (bounce 0) — overshoot on something that merely appeared
 * feels wrong; it is only right when a gesture carried momentum into it.
 */
function spring({ from = 0, response = 0.36, bounce = 0, onUpdate }) {
  const zeta = 1 - bounce;                       // 1.0 = critically damped
  const omega = (2 * Math.PI) / response;
  let x = from, v = 0, target = from, raf = null, last = 0;

  function frame(now) {
    const dt = Math.min(0.064, (now - last) / 1000 || 0.016);
    last = now;
    // Semi-implicit Euler: stable at the frame rates a browser actually hits.
    const a = -2 * zeta * omega * v - omega * omega * (x - target);
    v += a * dt;
    x += v * dt;
    if (Math.abs(x - target) < 0.001 && Math.abs(v) < 0.01) {
      x = target; v = 0; raf = null; onUpdate(x); return;
    }
    onUpdate(x);
    raf = requestAnimationFrame(frame);
  }

  return {
    set(next, { velocity } = {}) {
      target = next;
      if (velocity != null) v = velocity;
      if (REDUCED.matches) { x = target; v = 0; onUpdate(x); return; }
      if (raf == null) { last = performance.now(); raf = requestAnimationFrame(frame); }
    },
    jump(next) { x = target = next; v = 0; if (raf) cancelAnimationFrame(raf); raf = null; onUpdate(x); },
    get value() { return x; },
  };
}

/* ── elements ─────────────────────────────────────────────────────────── */
const el = {
  search: $('search'), roster: $('roster'),
  hero: $('hero'), heroIcon: $('heroIcon'), heroName: $('heroName'), heroMeta: $('heroMeta'),
  facts: $('facts'), aimNote: $('aimNote'), multiWarn: $('multiWarn'), modelLine: $('modelLine'),
  mode: $('mode'), segThumb: $('segThumb'), modeHint: $('modeHint'),
  serial: $('serial'), fps: $('fps'), fpsVal: $('fpsVal'),
  backend: $('backend'), deterministic: $('deterministic'),
  saveTraces: $('saveTraces'), videoPath: $('videoPath'),
  start: $('startBtn'), stop: $('stopBtn'),
  pill: $('statusPill'), stream: $('stream'), placeholder: $('placeholder'),
  frameInfo: $('frameInfo'), stats: $('stats'), breakdown: $('breakdown'),
  decision: $('decision'), log: $('log'),
};

const ROSTER = JSON.parse($('rosterData').textContent);
const BY_ID = new Map(ROSTER.brawlers.map((b) => [b.id, b]));

let selected = BY_ID.has('shelly') ? 'shelly' : ROSTER.brawlers[0]?.id;
let mode = 'observe';
let running = false;

const MODE_HINT = {
  observe: 'The policy decides and the planner plans, but no executor is ' +
           'built — nothing is sent to the phone.',
  control: 'Taps, drags and aimed shots are sent to the phone. Calibrate ' +
           'controls.json first or they will miss.',
};

/* The segmented thumb rides a spring so switching mode twice quickly stays
   continuous instead of restarting from the wrong place. */
const segSpring = spring({
  from: 0,
  onUpdate: (v) => { el.segThumb.style.transform = `translateX(${v * 100}%)`; },
});

/* ── the picker ───────────────────────────────────────────────────────── */
function renderRoster(query = '') {
  const q = query.trim().toLowerCase();
  const hits = ROSTER.brawlers.filter(
    (b) => !q || b.name.toLowerCase().includes(q) || b.id.includes(q));

  el.roster.replaceChildren();
  if (!hits.length) {
    const p = document.createElement('p');
    p.className = 'empty';
    p.textContent = 'No brawler matches that.';
    el.roster.append(p);
    return;
  }
  // Group headers only when browsing; while searching they are noise.
  const groups = q ? [[null, hits]]
    : ROSTER.rarity_order.map((r) => [r, hits.filter((b) => b.rarity === r)]);
  const frag = document.createDocumentFragment();
  for (const [rarity, group] of groups) {
    if (!group.length) continue;
    if (rarity) {
      const h = document.createElement('div');
      h.className = 'rarity';
      h.textContent = `${rarity} · ${group.length}`;
      frag.append(h);
    }
    for (const b of group) frag.append(cell(b));
  }
  el.roster.append(frag);
}

function cell(b) {
  const c = document.createElement('div');
  c.className = 'cell'
    + (b.model.is_brawler_specific ? ' trained' : '')
    + (b.aim === 'manual' ? ' manual' : '');
  c.dataset.id = b.id;
  c.setAttribute('role', 'option');
  c.setAttribute('tabindex', '0');
  c.setAttribute('aria-selected', String(b.id === selected));
  c.title = [b.name, b.rarity, b.brawler_class,
             b.range_tiles != null ? `${b.range_tiles} tiles` : null,
             `${b.aim} aim`,
             b.multi_body ? 'multi-body' : null].filter(Boolean).join(' · ');

  const img = document.createElement('img');
  img.loading = 'lazy'; img.alt = ''; img.src = b.icon;
  // A missing portrait must not leave a broken-image glyph in a 106-tile grid.
  img.addEventListener('error', () => img.replaceWith(fallback(b)), { once: true });

  const nm = document.createElement('span');
  nm.className = 'nm';
  nm.textContent = b.name;

  c.append(img, nm);

  // Feedback on pointer DOWN, not on click. Waiting for the release feels dead.
  c.addEventListener('pointerdown', () => { c.style.transform = 'scale(.93)'; });
  for (const ev of ['pointerup', 'pointerleave', 'pointercancel']) {
    c.addEventListener(ev, () => { c.style.transform = ''; });
  }
  return c;
}

function fallback(b) {
  const s = document.createElement('span');
  s.className = 'fallback';
  s.textContent = b.name.replace(/[^A-Z0-9]/g, '').slice(0, 2) || '??';
  return s;
}

function fact(label, value, attrs = {}) {
  const d = document.createElement('div');
  d.className = 'fact';
  for (const [k, v] of Object.entries(attrs)) if (v != null) d.dataset[k] = v;
  const b = document.createElement('b'); b.textContent = label;
  const s = document.createElement('span'); s.textContent = value;
  d.append(b, s);
  return d;
}

function select(id) {
  if (!BY_ID.has(id) || running) return;
  selected = id;
  for (const c of el.roster.querySelectorAll('.cell')) {
    c.setAttribute('aria-selected', String(c.dataset.id === id));
  }
  const b = BY_ID.get(id);

  el.heroName.textContent = b.name;
  el.heroMeta.textContent = [b.rarity, b.brawler_class].filter(Boolean).join(' · ');
  el.heroIcon.src = b.icon;
  el.heroIcon.onerror = () => el.heroIcon.removeAttribute('src');

  // The two facts the combat script actually reads. Showing them here is not
  // decoration: an unexplained "manual" is exactly the kind of flag nobody can
  // tell is wrong, which is why the reason sits underneath it.
  const range = b.range_tiles != null ? `${b.range_tiles} tiles` : 'unknown';
  el.facts.replaceChildren(
    fact('Attack range', range, { conf: b.range_confidence }),
    fact('Aim', b.aim === 'manual' ? 'Manual' : 'Auto', { aim: b.aim }),
  );
  el.aimNote.textContent = b.aim_reason
    ? `Aimed by hand — ${b.aim_reason}.`
    : 'Auto-aimed: a plain tap, which the game points at the nearest enemy in range.';

  el.multiWarn.classList.toggle('hidden', !b.multi_body);
  el.modelLine.textContent = b.model.label;
  el.modelLine.style.color = b.model.is_brawler_specific ? 'var(--good)' : '';
}

el.roster.addEventListener('click', (e) => {
  const c = e.target.closest('.cell');
  if (c) select(c.dataset.id);
});
el.roster.addEventListener('keydown', (e) => {
  const c = e.target.closest('.cell');
  if (c && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); select(c.dataset.id); }
});
el.search.addEventListener('input', () => renderRoster(el.search.value));
el.search.addEventListener('keydown', (e) => {
  if (e.key !== 'Enter') return;
  const cells = el.roster.querySelectorAll('.cell');
  if (cells.length === 1) select(cells[0].dataset.id);
});

/* ── mode ─────────────────────────────────────────────────────────────── */
function setMode(next) {
  mode = next;
  const control = mode === 'control';
  el.mode.classList.toggle('control', control);
  for (const b of el.mode.querySelectorAll('button')) {
    const on = b.dataset.mode === mode;
    b.classList.toggle('on', on);
    b.setAttribute('aria-checked', String(on));
  }
  segSpring.set(control ? 1 : 0);
  el.modeHint.textContent = MODE_HINT[mode];
  el.modeHint.classList.toggle('warn', control);
}
el.mode.addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (b && !b.disabled) setMode(b.dataset.mode);
});

/* ── devices ──────────────────────────────────────────────────────────── */
async function refreshDevices() {
  try {
    const { devices } = await (await fetch('/api/devices')).json();
    const want = ['<option value="">Auto — first device</option>'].concat(
      devices.map((d) => `<option value="${d.serial}">${d.serial} — ${d.state}</option>`)).join('');
    if (el.serial.innerHTML !== want) {          // never rebuild while it is open
      const cur = el.serial.value;
      el.serial.innerHTML = want;
      el.serial.value = cur;
    }
  } catch { /* adb missing is fine; auto still works */ }
}

/* ── start / stop ─────────────────────────────────────────────────────── */
async function start() {
  el.start.disabled = true;
  const video = el.videoPath.value.trim();
  try {
    const r = await fetch('/api/start', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        brawler: selected, mode,
        serial: el.serial.value || null,
        fps: Number(el.fps.value),
        source: video ? 'video' : 'phone',
        video_path: video || null,
        backend: el.backend.value,
        deterministic: el.deterministic.checked,
        save_traces: el.saveTraces.checked,
      }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
    attachStream();
  } catch (e) {
    el.log.textContent += `\ncould not start: ${e.message}`;
    el.start.disabled = false;
  }
}

async function stop() {
  el.stop.disabled = true;
  try { await fetch('/api/stop', { method: 'POST' }); } catch { /* poll recovers */ }
  detachStream();
}

function attachStream() {
  // Cache-bust so a restarted session opens a fresh multipart connection rather
  // than the browser reusing the finished one.
  el.stream.src = `/api/stream?t=${Date.now()}`;
  el.stream.classList.add('live');
  el.placeholder.classList.add('hidden');
}
function detachStream() {
  el.stream.classList.remove('live');
  el.stream.removeAttribute('src');
  el.placeholder.classList.remove('hidden');
}

/* ── readout ──────────────────────────────────────────────────────────── */
function kv(target, pairs) {
  // Rebuild only on change. Blowing away innerHTML twice a second costs nothing
  // visually but steals focus and fights the cursor.
  const sig = JSON.stringify(pairs);
  if (target.dataset.sig === sig) return;
  target.dataset.sig = sig;
  const frag = document.createDocumentFragment();
  for (const [k, v, cls] of pairs) {
    const a = document.createElement('div'); a.className = 'k'; a.textContent = k;
    const b = document.createElement('div');
    b.className = 'v' + (cls ? ' ' + cls : ''); b.textContent = v;
    frag.append(a, b);
  }
  target.replaceChildren(frag);
}

function renderBreakdown(bd) {
  const sig = JSON.stringify(bd || {});
  if (el.breakdown.dataset.sig === sig) return;
  el.breakdown.dataset.sig = sig;
  const keys = Object.keys(bd || {});
  if (!keys.length) {
    el.breakdown.replaceChildren(
      Object.assign(document.createElement('p'), { className: 'note', textContent: '—' }));
    return;
  }
  const max = Math.max(...keys.map((k) => Math.abs(bd[k])), 0.001);
  const frag = document.createDocumentFragment();
  for (const k of keys.sort((a, b) => Math.abs(bd[b]) - Math.abs(bd[a]))) {
    const v = bd[k];
    const row = document.createElement('div');
    row.className = 'row';
    // Bars grow from the centre so a penalty reads as a penalty rather than a
    // shorter reward.
    row.innerHTML =
      `<span class="name" title="${k}">${k}</span>` +
      `<span class="track"><span class="fill ${v >= 0 ? 'pos' : 'neg'}" ` +
      `style="width:${(Math.abs(v) / max) * 50}%"></span></span>` +
      `<span class="val">${v >= 0 ? '+' : ''}${v.toFixed(2)}</span>`;
    frag.append(row);
  }
  el.breakdown.replaceChildren(frag);
}

const setDisabled = (n, want) => { if (n.disabled !== want) n.disabled = want; };

function render(s) {
  running = !!s.running;

  if (el.pill.dataset.status !== s.status) el.pill.dataset.status = s.status;
  el.pill.classList.toggle('control', running && s.mode === 'control');
  const label = s.status === 'running' && s.mode === 'control' ? 'Control'
    : s.status === 'between-matches' ? 'Between matches'
    : s.status.charAt(0).toUpperCase() + s.status.slice(1);
  const span = el.pill.querySelector('span');
  if (span.textContent !== label) span.textContent = label;

  setDisabled(el.start, running);
  setDisabled(el.stop, !running);
  setDisabled(el.search, running);
  el.mode.querySelectorAll('button').forEach((b) => setDisabled(b, running));
  el.roster.classList.toggle('locked', running);

  const info = s.tick
    ? `tick ${s.tick} · ${s.fps} fps · ${s.elapsed}s${s.episode > 1 ? ` · match ${s.episode}` : ''}`
    : '';
  if (el.frameInfo.textContent !== info) el.frameInfo.textContent = info;

  const g = s.game || {};
  kv(el.stats, [
    ['reward', (s.reward >= 0 ? '+' : '') + Number(s.reward).toFixed(3), s.reward >= 0 ? 'on' : 'alert'],
    ['return', (s.episode_return >= 0 ? '+' : '') + Number(s.episode_return).toFixed(2)],
    ['match', g.state ?? '—'],
    ['hp', g.hp ?? '—'],
    ['cubes', g.cubes ?? '—'],
    ['ammo', g.ammo ?? 'unread'],
    ['super', g.super != null ? `${g.super}%` : '—'],
    ['left', g.players_left ?? '—'],
    ['enemies', g.enemies ?? 0],
    ['boxes', g.boxes ?? 0],
    ['in gas', g.in_gas ? 'YES' : 'no', g.in_gas ? 'alert' : 'off'],
    // "digits" is the reliable anchor path; a bare "ring" read is the weak
    // fallback. Worth seeing, because a WRONG anchor and a MISSING one look
    // identical on the picture but are different bugs.
    ['anchor', g.anchor_source ?? '—', g.anchor_source === 'digits' ? 'on' : 'off'],
  ]);

  const d = s.decision || {};
  const p = d.planner || {};
  kv(el.decision, [
    ['intent', d.intent || '—'],
    ['policy', d.policy === 'brawler' ? 'brawler weights'
      : d.policy === 'base' ? 'shared base' : 'random',
      d.policy === 'brawler' ? 'on' : 'off'],
    ['aim', d.aim_mode === 'manual'
      ? (d.aim ? `manual ${d.aim[0].toFixed(2)}, ${d.aim[1].toFixed(2)}` : 'manual')
      : 'auto', d.aim_mode === 'manual' ? 'info' : 'off'],
    ['attack', d.attack ? 'FIRE' : '—', d.attack ? 'on' : 'off'],
    ['super', d.super ? 'FIRE' : '—', d.super ? 'on' : 'off'],
    ['why', d.combat_reason || '—'],
    ['waypoint', d.waypoint ? d.waypoint.join(', ') : '—'],
    ['blocked', p.blocked ? 'yes' : 'no', p.blocked ? 'alert' : 'off'],
    ['gas escape', p.escaping_gas ? 'YES' : 'no', p.escaping_gas ? 'alert' : 'off'],
  ]);

  renderBreakdown(s.breakdown);

  if (s.log) {
    const text = s.log.join('\n');
    if (el.log.textContent !== text) {
      const atBottom = el.log.scrollHeight - el.log.scrollTop <= el.log.clientHeight + 24;
      el.log.textContent = text;
      if (atBottom) el.log.scrollTop = el.log.scrollHeight;
    }
  }

  if (!running && el.stream.classList.contains('live') &&
      (s.status === 'stopped' || s.status === 'error')) detachStream();
}

async function poll() {
  try { render(await (await fetch('/api/status')).json()); } catch { /* retry */ }
}

/* ── wire ─────────────────────────────────────────────────────────────── */
el.fps.addEventListener('input', () => { el.fpsVal.textContent = el.fps.value; });
el.start.addEventListener('click', start);
el.stop.addEventListener('click', stop);

renderRoster();
select(selected);
setMode('observe');
segSpring.jump(0);
refreshDevices();
poll();
setInterval(poll, 500);
setInterval(() => { if (!running) refreshDevices(); }, 10000);
