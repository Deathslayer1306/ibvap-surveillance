/**
 * IBVAP — Incident Logs / Forensic Evidence Vault (incident_logs.js)
 */

// Safe parse — handles both stringified and already-parsed values
function _safeArr(v) {
  if (Array.isArray(v)) return v;
  if (typeof v === 'string') { try { return JSON.parse(v); } catch { return []; } }
  return [];
}
function _safeBool(v) { return v === true || v === 1; }

let _events = [];
let _currentFilter = { severity: 'ALL', date: '' };

/* ─── Load Events ────────────────────────────────────────── */
async function loadEvents() {
  showLoading(true);
  try {
    const params = new URLSearchParams({ limit: 100 });
    if (_currentFilter.severity !== 'ALL') params.set('level', _currentFilter.severity);

    const [evRes, snapRes] = await Promise.all([
      fetch(`/api/events?${params}`),
      fetch('/api/snapshots')
    ]);

    const { events } = await evRes.json();
    const { snapshots } = await snapRes.json();

    // Build snapshot map
    const snapMap = {};
    (snapshots || []).forEach(s => {
      const key = (s.event_id || s.filename || '').replace('.jpg','');
      snapMap[key] = s.filename || s;
    });

    _events = events.map(ev => ({ ...ev, _snapshot: snapMap[ev.id] || null }));
    applyFilters();
  } catch(e) {
    console.error('Failed to load events', e);
    showError();
  } finally {
    showLoading(false);
  }
}

function applyFilters() {
  let data = [..._events];

  if (_currentFilter.severity !== 'ALL') {
    data = data.filter(e => e.alert_level === _currentFilter.severity);
  }
  if (_currentFilter.date) {
    const d = _currentFilter.date; // YYYY-MM-DD
    data = data.filter(e => (e.timestamp || '').startsWith(d));
  }

  renderCards(data);
  updateCount(data.length);
}

function updateCount(n) {
  const el = document.getElementById('incident-count');
  if (el) el.textContent = n;
}

/* ─── Render Evidence Cards ──────────────────────────────── */
function renderCards(events) {
  const grid = document.getElementById('evidence-grid');
  if (!grid) return;

  if (!events.length) {
    grid.innerHTML = `
      <div class="empty-state" style="grid-column:1/-1;padding:60px">
        <div class="empty-state-icon">📂</div>
        <div class="empty-state-text">No incidents found matching filters</div>
      </div>`;
    return;
  }

  grid.innerHTML = events.map(ev => buildEvidenceCard(ev)).join('');
}

function buildEvidenceCard(ev) {
  const level = ev.alert_level || 'INFO';
  const score = ev.threat_score || 0;
  const scoreStr = score.toFixed(1);
  const ts = fmtTime(ev.timestamp);
  const dt = fmtDate(ev.timestamp);
  const cam = ev.camera_id || 'UNKNOWN';

  // Incident ID
  const incId = `INCIDNT_TD: XF-${(ev.id || '000').toString().padStart(3,'0')}`;

  // Tags
  const tags = [];
  if (_safeBool(ev.weapon_detected)) tags.push('WPN-DETECTED');
  if (_safeArr(ev.fence_breaches).length) tags.push('BREACH');
  if (_safeArr(ev.anpr_hits).length) tags.push('PLATE-HIT');
  if (!tags.length) tags.push(ev.pose_anomaly || 'NORMAL');

  // Snapshot
  const hasSnap = !!ev._snapshot;
  const snapUrl = hasSnap ? `/snapshots/${ev._snapshot}` : null;

  const levelColors = {
    CRITICAL: '#ff2828',
    WARNING:  '#ffaa00',
    INFO:     '#00dc78',
  };
  const borderColor = levelColors[level] || levelColors.INFO;

  return `
    <div class="evidence-card" style="border-color:${borderColor}20;"
         onclick="openEvidence(${ev.id})">
      <!-- Thumbnail -->
      <div class="ev-thumb">
        ${hasSnap
          ? `<img src="${snapUrl}" alt="snapshot" class="ev-img" loading="lazy"/>`
          : `<div class="ev-no-thumb">
               <div style="font-size:28px;opacity:.3;">📷</div>
               <div style="font-size:9px;color:var(--dim);margin-top:4px;">NO SNAPSHOT</div>
             </div>`
        }
        <!-- Score badge overlay -->
        <div class="ev-score-badge" style="background:rgba(0,0,0,.75);border:1px solid ${borderColor};">
          <span style="color:${borderColor};font-weight:700;font-size:12px;">${scoreStr}</span>
          <span style="color:var(--muted);font-size:9px;">/10</span>
        </div>
        ${level === 'CRITICAL' ? `<div class="ev-breach-badge">BREACH DETECTED</div>` : ''}
        <!-- Level chip -->
        <div class="ev-level-chip badge badge-${levelClass(level)}">${level}</div>
      </div>

      <!-- Card body -->
      <div class="ev-body">
        <div class="ev-inc-id">${incId}</div>
        <div class="ev-level-row">
          <span class="badge badge-${levelClass(level)}" style="font-size:8px;">${level}</span>
          ${tags.map(t => `<span class="badge badge-dark" style="font-size:8px;">${t}</span>`).join('')}
        </div>
        <div class="ev-meta-grid">
          <div class="ev-meta-item">
            <span class="ev-meta-label">TME:</span>
            <span class="ev-meta-val" style="font-family:var(--mono)">${ts}</span>
          </div>
          <div class="ev-meta-item">
            <span class="ev-meta-label">DTE:</span>
            <span class="ev-meta-val" style="font-family:var(--mono)">${dt}</span>
          </div>
          <div class="ev-meta-item">
            <span class="ev-meta-label">CAM:</span>
            <span class="ev-meta-val">${cam}</span>
          </div>
          <div class="ev-meta-item">
            <span class="ev-meta-label">SCR:</span>
            <span class="ev-meta-val" style="color:${borderColor}">${scoreStr}/10</span>
          </div>
        </div>
      </div>
    </div>`;
}

function openEvidence(id) {
  const ev = _events.find(e => e.id == id);
  if (!ev) return;
  // Simple detail modal — could be expanded
  const details = [
    `ID: ${ev.id}`,
    `Camera: ${ev.camera_id}`,
    `Time: ${ev.timestamp}`,
    `Level: ${ev.alert_level}`,
    `Score: ${ev.threat_score}/10`,
    `Weapon: ${ev.weapon_detected ? 'YES — ' + ev.weapon_labels : 'No'}`,
    `Pose: ${ev.pose_anomaly}`,
    `Emotion: ${ev.emotion}`,
    `Fence Breaches: ${ev.fence_breaches}`,
    `ANPR Plates: ${ev.anpr_plates}`,
    `ANPR Hits: ${ev.anpr_hits}`,
  ].join('\n');
  alert(details);
}

function showLoading(v) {
  const grid = document.getElementById('evidence-grid');
  if (!grid) return;
  if (v) {
    grid.innerHTML = Array(8).fill(0).map(() => `
      <div class="evidence-card">
        <div class="ev-thumb skeleton" style="height:160px;"></div>
        <div class="ev-body">
          <div class="skeleton" style="height:12px;width:80%;margin-bottom:6px;"></div>
          <div class="skeleton" style="height:10px;width:60%;"></div>
        </div>
      </div>`).join('');
  }
}
function showError() {
  const grid = document.getElementById('evidence-grid');
  if (grid) grid.innerHTML = `
    <div class="empty-state" style="grid-column:1/-1">
      <div class="empty-state-icon">⚠️</div>
      <div class="empty-state-text">Failed to load incidents. Server may be offline.</div>
    </div>`;
}

/* ─── Init ───────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  loadEvents();

  // Severity filter chips
  document.querySelectorAll('.sev-filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.sev-filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      _currentFilter.severity = btn.dataset.level;
      applyFilters();
    });
  });

  // Date filter
  const dateInput = document.getElementById('date-filter');
  if (dateInput) {
    dateInput.addEventListener('change', e => {
      _currentFilter.date = e.target.value;
      applyFilters();
    });
  }

  // Apply button
  const applyBtn = document.getElementById('apply-filter-btn');
  if (applyBtn) applyBtn.addEventListener('click', loadEvents);

  // WS: new critical events push to grid
  WS.on('state', data => {
    if (data.alert_level === 'CRITICAL' || _safeBool(data.weapon_detected)) {
      // Re-fetch to get the saved DB event
      setTimeout(loadEvents, 2000);
    }
  });
});
