/**
 * IBVAP — Threat Intelligence Page (threat_intel.js)
 * Drives the full dedicated threat breakdown dashboard.
 */

let _tiChart     = null;
let _tiWindowMin = 30;

// ── Card config ───────────────────────────────────────────────────────────────
const CARDS = [
  { key:'weapon',          id:'tic-weapon',  label:'ARMED PERSON',     color:'#ff4545' },
  { key:'anpr_blacklist',  id:'tic-anpr',    label:'BLACKLISTED PLATE', color:'#ff8800' },
  { key:'aggressive_pose', id:'tic-pose',    label:'AGGR. STANCE',     color:'#ffcc00' },
  { key:'raised_hands',    id:'tic-hands',   label:'RAISED HANDS',     color:'#f0a500' },
  { key:'fence_breach',    id:'tic-fence',   label:'FENCE BREACH',     color:'#00c8ff' },
  { key:'hostile_emotion', id:'tic-emotion', label:'HOSTILE EMOTION',  color:'#c060ff' },
  { key:'vehicle',         id:'tic-vehicle', label:'VEHICLE',          color:'#60d090' },
];

const SUMMARY_TYPES = [
  { key:'weapon',  label:'Armed Person',      color:'#ff4545', liveKey:'weapon'          },
  { key:'anpr',    label:'Blacklisted Plate',  color:'#ff8800', liveKey:'anpr_blacklist'  },
  { key:'pose',    label:'Aggressive Stance',  color:'#ffcc00', liveKey:'aggressive_pose' },
  { key:'fence',   label:'Fence Breach',       color:'#00c8ff', liveKey:'fence_breach'    },
  { key:'emotion', label:'Hostile Emotion',    color:'#c060ff', liveKey:'hostile_emotion' },
];

// ── Chart init ────────────────────────────────────────────────────────────────
function initChart() {
  const canvas = document.getElementById('ti-timeline-chart');
  if (!canvas || _tiChart) return;

  const gridColor = 'rgba(255,255,255,0.04)';
  const tickColor = 'rgba(255,255,255,0.28)';

  _tiChart = new Chart(canvas.getContext('2d'), {
    type: 'bar',
    data: {
      labels: [],
      datasets: [
        { label: 'Weapon',  backgroundColor: 'rgba(255,69,69,0.8)',   data: [], borderRadius:3 },
        { label: 'ANPR',    backgroundColor: 'rgba(255,136,0,0.8)',   data: [], borderRadius:3 },
        { label: 'Pose',    backgroundColor: 'rgba(255,204,0,0.75)',  data: [], borderRadius:3 },
        { label: 'Fence',   backgroundColor: 'rgba(0,200,255,0.7)',   data: [], borderRadius:3 },
        { label: 'Emotion', backgroundColor: 'rgba(192,96,255,0.7)',  data: [], borderRadius:3 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 500 },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(8,16,36,0.95)',
          titleColor: '#4dc8ff',
          bodyColor: '#ccd6e0',
          borderColor: 'rgba(0,200,255,0.2)',
          borderWidth: 1,
          padding: 10,
          callbacks: {
            title: (items) => '\u23f1 ' + items[0].label,
            label: (item)  => '  ' + item.dataset.label + ': ' + item.raw,
          },
        },
      },
      scales: {
        x: {
          stacked: true,
          grid: { color: gridColor },
          ticks: { color: tickColor, font:{ family:'JetBrains Mono', size:8 }, maxRotation:0 },
        },
        y: {
          stacked: true,
          grid: { color: gridColor },
          ticks: { color: tickColor, font:{ family:'JetBrains Mono', size:8 }, stepSize:1 },
          beginAtZero: true,
        },
      },
    },
  });
}

function updateChart(timeline) {
  if (!_tiChart) return;
  _tiChart.data.labels           = timeline.map(b => b.time);
  _tiChart.data.datasets[0].data = timeline.map(b => b.weapon);
  _tiChart.data.datasets[1].data = timeline.map(b => b.anpr);
  _tiChart.data.datasets[2].data = timeline.map(b => b.pose);
  _tiChart.data.datasets[3].data = timeline.map(b => b.fence);
  _tiChart.data.datasets[4].data = timeline.map(b => b.emotion);
  _tiChart.update('none');
}

// ── Threat cards ─────────────────────────────────────────────────────────────
function updateCards(live_threats) {
  for (const card of CARDS) {
    const count  = live_threats[card.key] || 0;
    const el     = document.getElementById(card.id);
    const cntEl  = document.getElementById(card.id + '-count');
    const stEl   = document.getElementById(card.id + '-status');
    if (!el) continue;

    el.classList.toggle('active',   count > 0);
    el.classList.toggle('inactive', count === 0);
    if (cntEl) cntEl.textContent = count;
    if (stEl)  stEl.textContent  = count > 0 ? '\u25cf ACTIVE' : '\u25cb CLEAR';
  }
}

// ── Per-camera section ────────────────────────────────────────────────────────
function updateCameras(per_camera) {
  const list = document.getElementById('ti-camera-list');
  const badge = document.getElementById('ti-cam-count');
  if (!list) return;

  const entries = Object.entries(per_camera);
  if (badge) badge.textContent = entries.length + ' CAM' + (entries.length !== 1 ? 'S' : '');

  list.innerHTML = '';
  if (entries.length === 0) {
    list.innerHTML = '<div class="empty-state"><div class="empty-state-icon">📡</div><div class="empty-state-text">No cameras online</div></div>';
    return;
  }

  for (const [camId, info] of entries) {
    const lvl  = info.alert_level || 'INFO';
    const tags  = info.active_threats || [];
    const score = (info.threat_score || 0).toFixed(1);

    const tagsHtml = tags.length > 0
      ? tags.map(t => '<span class="ti-cam-tag">' + t + '</span>').join('')
      : '<span class="ti-cam-ok">\u2713 ALL CLEAR</span>';

    const card = document.createElement('div');
    card.className = 'ti-cam-card ' + lvl;
    card.innerHTML =
      '<div class="ti-cam-card-hdr">' +
        '<span class="ti-cam-name">' + camId.toUpperCase() + '</span>' +
        '<span class="ti-cam-score-chip ' + lvl + '">' + score + '/10</span>' +
      '</div>' +
      '<div class="ti-cam-tags">' + tagsHtml + '</div>' +
      '<div class="ti-cam-persons">\u{1f465} ' + (info.num_persons || 0) + ' person(s) detected</div>';
    list.appendChild(card);
  }
}

// ── Detection summary table ───────────────────────────────────────────────────
function updateSummary(timeline, live_threats) {
  const tbody   = document.getElementById('ti-summary-tbody');
  const totEvEl = document.getElementById('ti-total-events');
  if (!tbody) return;

  // Compute totals & max across window
  const totals = {};
  const peaks  = {};
  for (const t of SUMMARY_TYPES) {
    totals[t.key] = timeline.reduce((s, b) => s + (b[t.key] || 0), 0);
    peaks[t.key]  = Math.max(...timeline.map(b => b[t.key] || 0), 0);
  }
  const maxTotal = Math.max(...Object.values(totals), 1);
  const grandTotal = Object.values(totals).reduce((s, v) => s + v, 0);

  if (totEvEl) totEvEl.textContent = grandTotal + ' event' + (grandTotal !== 1 ? 's' : '');

  tbody.innerHTML = '';
  for (const t of SUMMARY_TYPES) {
    const liveCount = live_threats[t.liveKey] || 0;
    const total     = totals[t.key];
    const peak      = peaks[t.key];
    const pct       = Math.round((total / maxTotal) * 100);
    const status    = liveCount > 0 ? '\u26a0 ACTIVE' : total > 0 ? '\u23f3 DETECTED' : '\u2713 CLEAR';
    const statusCol = liveCount > 0 ? '#ff4545' : total > 0 ? '#ffaa00' : '#60d090';

    const tr = document.createElement('tr');
    tr.innerHTML =
      '<td><span class="tst-type" style="color:' + t.color + ';">\u25cf ' + t.label + '</span></td>' +
      '<td class="tst-count">' + (liveCount > 0 ? liveCount : '—') + '</td>' +
      '<td class="tst-count">' + total + '</td>' +
      '<td class="tst-peak">' + (peak > 0 ? peak + ' / bucket' : '—') + '</td>' +
      '<td class="tst-bar-cell"><div class="tst-bar-bg"><div class="tst-bar-fill" style="width:' + pct + '%;background:' + t.color + ';opacity:.75;"></div></div></td>' +
      '<td style="color:' + statusCol + ';font-weight:600;">' + status + '</td>';
    tbody.appendChild(tr);
  }
}

// ── Main refresh ──────────────────────────────────────────────────────────────
async function refreshAll() {
  try {
    const res  = await fetch('/api/threat-breakdown?minutes=' + _tiWindowMin);
    const data = await res.json();

    updateCards(data.live_threats || {});
    updateChart(data.timeline || []);
    updateCameras(data.per_camera || {});
    updateSummary(data.timeline || [], data.live_threats || {});

    const updEl = document.getElementById('ti-upd-text');
    if (updEl) {
      const t = new Date();
      const hh = String(t.getHours()).padStart(2,'0');
      const mm = String(t.getMinutes()).padStart(2,'0');
      const ss = String(t.getSeconds()).padStart(2,'0');
      updEl.textContent = 'Updated ' + hh + ':' + mm + ':' + ss;
    }
  } catch(e) {
    console.warn('[ThreatIntel] refresh failed', e);
  }
}

function onWindowChange(val) {
  _tiWindowMin = parseInt(val, 10);
  refreshAll();
}

// ── Boot ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initChart();
  refreshAll();

  // Auto-refresh every 30 s
  setInterval(refreshAll, 30_000);

  // Also refresh on any WS CRITICAL/WARNING event
  if (window.WS) {
    WS.on('state', (data) => {
      if (data.alert_level && data.alert_level !== 'INFO') {
        if (!refreshAll._pending) {
          refreshAll._pending = true;
          setTimeout(() => { refreshAll(); refreshAll._pending = false; }, 2500);
        }
      }
    });
  }
});
