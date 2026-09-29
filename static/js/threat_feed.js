/**
 * IBVAP — Threat Feed Page (threat_feed.js)
 * Real-time threat event stream with severity filters
 */

// Safe JSON parse — handles both stringified and already-parsed values
function _safeArr(v) {
  if (Array.isArray(v)) return v;
  if (typeof v === 'string') { try { return JSON.parse(v); } catch { return []; } }
  return [];
}
function _safeBool(v) { return v === true || v === 1; }

let _allEvents = [];
let _activeLevels = new Set(['CRITICAL', 'WARNING', 'INFO']);
let _searchQuery = '';

/* ─── Load History ───────────────────────────────────────── */
async function loadHistory() {
  setLoadingState(true);
  try {
    const res = await fetch('/api/events?limit=100');
    const { events } = await res.json();
    _allEvents = events || [];
    renderFeed();
  } catch(e) {
    console.error('Failed to load threat feed', e);
  } finally {
    setLoadingState(false);
  }
}

/* ─── Render ─────────────────────────────────────────────── */
function renderFeed() {
  const list = document.getElementById('threat-feed-list');
  if (!list) return;

  const filtered = _allEvents.filter(ev => {
    const levelOk = _activeLevels.has(ev.alert_level);
    const searchOk = !_searchQuery || [
      ev.camera_id, ev.alert_level, ev.emotion, ev.pose_anomaly,
      ev.weapon_labels, ev.anpr_plates
    ].some(v => (v||'').toLowerCase().includes(_searchQuery));
    return levelOk && searchOk;
  });

  updateCounts(filtered);

  if (!filtered.length) {
    list.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">📡</div>
        <div class="empty-state-text">No threats detected matching filter</div>
      </div>`;
    return;
  }

  list.innerHTML = '';
  filtered.forEach(ev => {
    const row = buildEventRow(ev, false);
    list.appendChild(row);
  });
}

function buildEventRow(ev, isNew) {
  const div = document.createElement('div');
  div.className = `feed-event-row ${isNew ? 'new' : ''}`;
  div.dataset.id = ev.id;
  div.dataset.level = ev.alert_level;

  const level = ev.alert_level || 'INFO';
  const ts = fmtTime(ev.timestamp);
  const dt = fmtDate(ev.timestamp);
  const score = (ev.threat_score || 0).toFixed(1);

  // Build tag list
  const tags = [];
  if (ev.weapon_detected) tags.push({ label: 'WEAPON', cls: 'badge-critical' });
  if (ev.fence_breaches && JSON.parse(ev.fence_breaches||'[]').length)
    tags.push({ label: 'GEOFENCE', cls: 'badge-warning' });
  if (ev.anpr_hits && JSON.parse(ev.anpr_hits||'[]').length)
    tags.push({ label: 'ANPR HIT', cls: 'badge-warning' });
  if (ev.emotion && ev.emotion !== 'neutral')
    tags.push({ label: ev.emotion.toUpperCase(), cls: 'badge-cyan' });
  if (ev.pose_anomaly && ev.pose_anomaly !== 'normal')
    tags.push({ label: ev.pose_anomaly.toUpperCase(), cls: 'badge-cyan' });

  const levelIcons = { CRITICAL: '🔴', WARNING: '🟡', INFO: '🟢' };

  div.innerHTML = `
    <div class="fe-indicator feed-level-${level.toLowerCase()}"></div>
    <div class="fe-icon">${levelIcons[level]||'⚪'}</div>
    <div class="fe-content">
      <div class="fe-top-row">
        <span class="badge badge-${levelClass(level)} fe-level">${level}</span>
        <span class="fe-cam">${ev.camera_id||'—'}</span>
        <div class="fe-tags">
          ${tags.map(t => `<span class="badge ${t.cls}">${t.label}</span>`).join('')}
        </div>
      </div>
      <div class="fe-summary">${buildSummary(ev)}</div>
    </div>
    <div class="fe-meta">
      <div class="fe-score ${level === 'CRITICAL' ? 'score-crit' : level === 'WARNING' ? 'score-warn' : 'score-ok'}">${score}</div>
      <div class="fe-ts">${ts}</div>
      <div class="fe-dt">${dt}</div>
    </div>`;

  return div;
}

function buildSummary(ev) {
  const parts = [];
  if (_safeBool(ev.weapon_detected)) {
    const labels = _safeArr(ev.weapon_labels);
    parts.push(`⚠ Weapon detected: ${labels.join(', ')||'unknown'}`);
  }
  const breaches = _safeArr(ev.fence_breaches);
  if (breaches.length) parts.push(`🚧 Geofence breach in zone: ${breaches.join(', ')}`);
  const hits = _safeArr(ev.anpr_hits);
  if (hits.length) parts.push(`🚗 Plate match: ${hits.join(', ')}`);
  if (ev.emotion && ev.emotion !== 'neutral') parts.push(`😠 Emotion: ${ev.emotion}`);
  if (ev.pose_anomaly && ev.pose_anomaly !== 'normal') parts.push(`🧍 Pose: ${ev.pose_anomaly}`);
  if (!parts.length) parts.push(`Routine monitoring — score ${ev.threat_score||0}/10`);
  return parts.join(' · ');
}

function updateCounts(filtered) {
  const counts = { CRITICAL: 0, WARNING: 0, INFO: 0 };
  _allEvents.forEach(ev => { if (counts[ev.alert_level] !== undefined) counts[ev.alert_level]++; });
  ['CRITICAL', 'WARNING', 'INFO'].forEach(l => {
    const el = document.getElementById(`count-${l.toLowerCase()}`);
    if (el) el.textContent = counts[l];
  });
  const totalEl = document.getElementById('feed-total-count');
  if (totalEl) totalEl.textContent = filtered.length;
}

function setLoadingState(v) {
  const list = document.getElementById('threat-feed-list');
  if (!list) return;
  if (v) {
    list.innerHTML = Array(6).fill(0).map(() => `
      <div class="feed-event-row">
        <div class="fe-indicator"></div>
        <div class="fe-icon skeleton" style="width:20px;height:20px;border-radius:50%;"></div>
        <div class="fe-content">
          <div class="skeleton" style="height:12px;width:70%;margin-bottom:6px;"></div>
          <div class="skeleton" style="height:10px;width:50%;"></div>
        </div>
        <div class="fe-meta">
          <div class="skeleton" style="height:24px;width:30px;"></div>
        </div>
      </div>`).join('');
  }
}

/* ─── WS: Push new events ────────────────────────────────── */
function pushLiveEvent(data) {
  if (!data.alert_level) return;

  // Prepend to allEvents
  const ev = { ...data, id: data.id || Date.now() };
  _allEvents.unshift(ev);
  if (_allEvents.length > 200) _allEvents.pop();

  if (!_activeLevels.has(ev.alert_level)) return;
  if (_searchQuery) return; // Don't push during search

  const list = document.getElementById('threat-feed-list');
  if (!list) return;
  const row = buildEventRow(ev, true);
  list.insertBefore(row, list.firstChild);

  // Flash counter
  updateCounts(_allEvents.filter(e => _activeLevels.has(e.alert_level)));

  // Prune
  while (list.children.length > 100) list.removeChild(list.lastChild);
}

/* ─── Init ───────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  loadHistory();

  // Level filter toggle buttons
  document.querySelectorAll('.level-filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const level = btn.dataset.level;
      btn.classList.toggle('active');
      if (_activeLevels.has(level)) _activeLevels.delete(level);
      else _activeLevels.add(level);
      renderFeed();
    });
  });

  // Search
  const searchEl = document.getElementById('feed-search');
  if (searchEl) {
    searchEl.addEventListener('input', e => {
      _searchQuery = e.target.value.toLowerCase().trim();
      renderFeed();
    });
  }

  // Refresh
  const refreshBtn = document.getElementById('feed-refresh-btn');
  if (refreshBtn) refreshBtn.addEventListener('click', loadHistory);

  // WS live push
  WS.on('state', pushLiveEvent);
});
