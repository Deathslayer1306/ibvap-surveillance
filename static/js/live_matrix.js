/**
 * IBVAP — Live Matrix Page (live_matrix.js)
 * 4-camera MJPEG grid + Threat Escalation Feed + Threat Intelligence Panel
 */

function _safeArr(v) {
  if (Array.isArray(v)) return v;
  if (typeof v === 'string') { try { return JSON.parse(v); } catch { return []; } }
  return [];
}
function _safeBool(v) { return v === true || v === 1; }

/* ─── Camera Grid ─────────────────────────────────────────── */
let _cameras = [];
let _threatAlerts = [];

async function loadCameras() {
  try {
    const res = await fetch('/api/cameras');
    const { cameras } = await res.json();
    _cameras = cameras;
    renderCameraGrid(cameras);
  } catch(e) {
    console.error('Failed to load cameras', e);
  }
}

function renderCameraGrid(cameras) {
  const grid = document.getElementById('cam-grid');
  if (!grid) return;
  grid.innerHTML = '';

  // Always render 4 slots
  for (let i = 0; i < 4; i++) {
    const cam = cameras[i];
    const tile = createCamTile(cam, i);
    grid.appendChild(tile);
  }
}

function createCamTile(cam, idx) {
  const tile = document.createElement('div');
  tile.className = 'cam-tile';
  tile.id = cam ? `cam-tile-${cam.id}` : `cam-tile-empty-${idx}`;

  if (!cam) {
    tile.innerHTML = `
      <div class="cam-tile-hdr">
        <span class="cam-tile-label">CAM-0${idx+1} [NO SIGNAL]</span>
      </div>
      <div class="cam-tile-body offline">
        <div class="cam-no-signal">
          <div class="no-sig-icon">📡</div>
          <div class="no-sig-text">SIGNAL LOST</div>
        </div>
      </div>`;
    return tile;
  }

  const isLive = cam.status === 'live';
  tile.innerHTML = `
    <div class="cam-tile-hdr">
      <div style="display:flex;align-items:center;gap:6px;">
        <span class="cam-tile-label">CAM-0${idx+1} [${cam.name.toUpperCase()}]</span>
        ${isLive ? '<span class="live-badge">● LIVE</span>' : '<span class="offline-badge">○ OFFLINE</span>'}
      </div>
      <div style="display:flex;align-items:center;gap:6px;">
        <span class="cam-fps-badge" id="fps-${cam.id}">${cam.inf_fps?.toFixed(1)||0} FPS</span>
        <span class="cam-tile-id">#0${idx+1}</span>
      </div>
    </div>
    <div class="cam-tile-body">
      ${isLive
        ? `<img class="cam-img" src="/video_feed?camera=${cam.id}" alt="${cam.name}" loading="lazy"/>`
        : `<div class="cam-no-signal">
             <div class="no-sig-icon">📡</div>
             <div class="no-sig-text">SIGNAL LOST</div>
           </div>`
      }
      <div class="cam-corners">
        <div class="cc tl"></div><div class="cc tr"></div>
        <div class="cc bl"></div><div class="cc br"></div>
      </div>
      <div class="cam-overlay-info">
        <span class="cam-overlay-id" id="cam-ol-id-${cam.id}">${cam.id}</span>
        <span class="cam-overlay-score" id="cam-ol-score-${cam.id}">0.0</span>
      </div>
    </div>`;

  return tile;
}

/* ─── Threat Escalation Feed ─────────────────────────────── */
function initThreatFeed() {
  fetch('/api/events?limit=20')
    .then(r => r.json())
    .then(({ events }) => {
      events.reverse().forEach(ev => appendThreatCard(ev, false));
    })
    .catch(() => {});
}

function appendThreatCard(ev, animate = true) {
  const feed = document.getElementById('threat-escalation-feed');
  if (!feed) return;

  _threatAlerts.unshift(ev);
  if (_threatAlerts.length > 30) _threatAlerts.pop();

  const card = buildThreatCard(ev, animate);
  feed.insertBefore(card, feed.firstChild);

  while (feed.children.length > 20) {
    feed.removeChild(feed.lastChild);
  }
}

function buildThreatCard(ev, animate) {
  const div = document.createElement('div');
  div.className = `threat-card threat-card-${(ev.alert_level||'INFO').toLowerCase()}`;
  if (animate) div.classList.add('new');

  const icons = { CRITICAL: '🔴', WARNING: '🟡', INFO: '🟢' };
  const icon = icons[ev.alert_level] || '🟢';
  const ts = fmtTime(ev.timestamp);

  let summary = '';
  const items = [];
  if (_safeBool(ev.weapon_detected)) items.push('⚠️ ARMED PERSON DETECTED');
  if (_safeArr(ev.fence_breaches).length > 0) items.push('🚧 UNAUTHORIZED ACCESS — GeoFence breach');
  if (_safeArr(ev.anpr_hits).length > 0) items.push(`🚗 VEHICLE IDENTIFIED — Plate: ${_safeArr(ev.anpr_hits)[0]}`);
  if (items.length === 0) items.push(`📊 THREAT SCORE ${ev.threat_score}/10`);
  summary = items[0];

  const confidence = Math.round((ev.threat_score / 10) * 100);
  const cam = ev.camera_id || 'UNKNOWN';

  div.innerHTML = `
    <div class="tc-hdr">
      <div class="tc-icon">${icon}</div>
      <div class="tc-body">
        <div class="tc-title">${summary}</div>
        <div class="tc-meta">
          ${ev.weapon_detected ? `<span class="tc-conf">Weapon Confidence: ${confidence}%</span>` : ''}
        </div>
      </div>
      <div class="tc-ts">${ts}</div>
    </div>
    <div class="tc-footer">
      <span class="tc-loc">LOC: ${cam}</span>
      ${ev.alert_level === 'CRITICAL'
        ? `<button class="btn btn-danger btn-sm tc-action" onclick="handleDispatch('${ev.id}', this)">DISPATCH</button>`
        : `<button class="btn btn-ghost btn-sm tc-action" onclick="handleAcknowledge('${ev.id}', this)">ACKNOWLEDGE</button>`
      }
    </div>`;

  return div;
}

function handleDispatch(id, btn) {
  btn.textContent = 'DISPATCHED';
  btn.disabled = true; btn.style.opacity = '.5';
}
function handleAcknowledge(id, btn) {
  btn.textContent = 'ACKNOWLEDGED';
  btn.disabled = true; btn.style.opacity = '.5';
}

/* ─── System Metrics Bar ─────────────────────────────────── */
function updateSysMetrics(data) {
  if (data.camera_id) {
    const scoreEl = document.getElementById(`cam-ol-score-${data.camera_id}`);
    if (scoreEl) scoreEl.textContent = `${data.threat_score||0}/10`;

    const fpsEl = document.getElementById(`fps-${data.camera_id}`);
    if (fpsEl && data.inf_fps) fpsEl.textContent = `${(data.inf_fps||0).toFixed(1)} FPS`;
  }

  const latEl  = document.getElementById('sys-lat');
  const fpsEl  = document.getElementById('sys-fps');
  if (latEl && data.inference_ms) latEl.textContent = `${Math.round(data.inference_ms)}ms`;
  if (fpsEl && data.inf_fps)      fpsEl.textContent = `${(data.inf_fps||0).toFixed(1)}`;

  const latBar = document.getElementById('sys-lat-bar');
  if (latBar && data.inference_ms) {
    const pct = Math.min((data.inference_ms / 100) * 100, 100);
    latBar.style.width = pct + '%';
    latBar.className = `metric-bar-fill ${pct > 70 ? 'red' : pct > 40 ? 'amber' : 'green'}`;
  }
}

/* ═══════════════════════════════════════════════════════════
   THREAT INTELLIGENCE PANEL
   ═══════════════════════════════════════════════════════════ */

let _tiChart = null;   // Chart.js instance

const _TI_PILL_MAP = {
  weapon:          { id: 'tip-weapon',  countId: 'tipc-weapon'  },
  anpr_blacklist:  { id: 'tip-anpr',    countId: 'tipc-anpr'    },
  aggressive_pose: { id: 'tip-pose',    countId: 'tipc-pose'    },
  raised_hands:    { id: 'tip-hands',   countId: 'tipc-hands'   },
  fence_breach:    { id: 'tip-fence',   countId: 'tipc-fence'   },
  hostile_emotion: { id: 'tip-emotion', countId: 'tipc-emotion' },
  vehicle:         { id: 'tip-vehicle', countId: 'tipc-vehicle' },
};

function _initTiChart() {
  const canvas = document.getElementById('ti-timeline-chart');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const gridColor = 'rgba(255,255,255,0.04)';
  const tickColor = 'rgba(255,255,255,0.25)';

  _tiChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: [],
      datasets: [
        { label: 'Weapon',  backgroundColor: 'rgba(255,69,69,0.75)',   data: [] },
        { label: 'ANPR',    backgroundColor: 'rgba(255,136,0,0.75)',   data: [] },
        { label: 'Pose',    backgroundColor: 'rgba(255,204,0,0.7)',    data: [] },
        { label: 'Fence',   backgroundColor: 'rgba(0,200,255,0.65)',   data: [] },
        { label: 'Emotion', backgroundColor: 'rgba(192,96,255,0.65)',  data: [] },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(10,20,40,0.92)',
          titleColor: '#4dc8ff',
          bodyColor: '#ccd6e0',
          borderColor: 'rgba(0,200,255,0.2)',
          borderWidth: 1,
          callbacks: {
            title: (items) => `⏱ ${items[0].label}`,
            label: (item) => ` ${item.dataset.label}: ${item.raw}`,
          },
        },
      },
      scales: {
        x: {
          stacked: true,
          grid: { color: gridColor },
          ticks: { color: tickColor, font: { family: 'JetBrains Mono', size: 8 }, maxRotation: 0 },
        },
        y: {
          stacked: true,
          grid: { color: gridColor },
          ticks: { color: tickColor, font: { family: 'JetBrains Mono', size: 8 }, stepSize: 1 },
          beginAtZero: true,
        },
      },
    },
  });
}

function _updateTiChart(timeline) {
  if (!_tiChart) return;
  const labels  = timeline.map(b => b.time);
  const weapon  = timeline.map(b => b.weapon);
  const anpr    = timeline.map(b => b.anpr);
  const pose    = timeline.map(b => b.pose);
  const fence   = timeline.map(b => b.fence);
  const emotion = timeline.map(b => b.emotion);

  _tiChart.data.labels          = labels;
  _tiChart.data.datasets[0].data = weapon;
  _tiChart.data.datasets[1].data = anpr;
  _tiChart.data.datasets[2].data = pose;
  _tiChart.data.datasets[3].data = fence;
  _tiChart.data.datasets[4].data = emotion;
  _tiChart.update('none');
}

function _updateTiPills(live_threats) {
  for (const [key, cfg] of Object.entries(_TI_PILL_MAP)) {
    const count = live_threats[key] || 0;
    const pill  = document.getElementById(cfg.id);
    const badge = document.getElementById(cfg.countId);
    if (pill) {
      pill.classList.toggle('active',   count > 0);
      pill.classList.toggle('inactive', count === 0);
    }
    if (badge) badge.textContent = count;
  }
}

function _updateTiCameraRows(per_camera) {
  const container = document.getElementById('ti-camera-rows');
  if (!container) return;

  // Keep the header label
  const hdr = container.querySelector('div');
  container.innerHTML = '';
  if (hdr) container.appendChild(hdr);

  for (const [camId, info] of Object.entries(per_camera)) {
    const row = document.createElement('div');
    row.className = `ti-cam-row ${info.alert_level || 'INFO'}`;

    const threatsHtml = info.active_threats && info.active_threats.length > 0
      ? info.active_threats.map(t => `<span class="ti-cam-tag">${t}</span>`).join('')
      : `<span class="ti-cam-ok">ALL CLEAR</span>`;

    row.innerHTML = `
      <span class="ti-cam-id">${camId.toUpperCase()}</span>
      <span class="ti-cam-score">${(info.threat_score||0).toFixed(1)}</span>
      <div class="ti-cam-threats">${threatsHtml}</div>`;
    container.appendChild(row);
  }
}

async function refreshThreatIntel() {
  try {
    const res  = await fetch('/api/threat-breakdown?minutes=30');
    const data = await res.json();

    _updateTiPills(data.live_threats || {});
    _updateTiCameraRows(data.per_camera || {});
    _updateTiChart(data.timeline || []);

    const updEl = document.getElementById('ti-updated');
    if (updEl) {
      const t = new Date();
      updEl.textContent = `Updated ${t.getHours().toString().padStart(2,'0')}:${t.getMinutes().toString().padStart(2,'0')}:${t.getSeconds().toString().padStart(2,'0')}`;
    }
  } catch(e) {
    console.warn('[ThreatIntel] Refresh failed', e);
  }
}

/* ─── WS Integration ─────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  loadCameras();
  initThreatFeed();

  WS.on('state', (data) => {
    updateSysMetrics(data);

    // Push to threat feed on non-INFO alerts
    if (data.alert_level && data.alert_level !== 'INFO' || data.weapon_detected) {
      appendThreatCard(data, true);
    }
    if (data.alert_level === 'CRITICAL') {
      appendThreatCard(data, true);
    }
  });

  // Lockdown button
  const lockBtn = document.getElementById('lockdown-btn');
  if (lockBtn) {
    lockBtn.addEventListener('click', () => {
      lockBtn.textContent = lockBtn.textContent === '🔒 LOCKDOWN'
        ? '🔓 UNLOCK' : '🔒 LOCKDOWN';
      lockBtn.classList.toggle('active');
    });
  }
});

/* helper: format timestamp → HH:MM:SS */
function fmtTime(ts) {
  if (!ts) return '--:--';
  try {
    const d = new Date(ts);
    return d.toLocaleTimeString('en-GB', { hour12: false });
  } catch { return ts.slice(11, 19) || '--:--'; }
}

