/**
 * IBVAP — Shared WebSocket Manager (ws.js)
 * Singleton WebSocket connection with pub/sub event bus.
 * All pages import this and subscribe to events they care about.
 */

const WS = (() => {
  const _subs = {};          // { eventName: [callback, ...] }
  const _latestState = {};   // camera_id → last known state dict
  let _ws = null;
  let _retryMs = 1000;
  let _maxRetry = 16000;
  let _retryTimer = null;

  function on(event, cb) {
    if (!_subs[event]) _subs[event] = [];
    _subs[event].push(cb);
  }

  function off(event, cb) {
    if (!_subs[event]) return;
    _subs[event] = _subs[event].filter(f => f !== cb);
  }

  function _emit(event, data) {
    (_subs[event] || []).forEach(cb => { try { cb(data); } catch(e) {} });
  }

  function _setStatus(online) {
    const dot = document.getElementById('ws-dot');
    const pill = document.getElementById('ws-indicator');
    const lbl = document.getElementById('ws-label');
    if (dot) dot.className = online ? 'on' : '';
    if (pill) pill.className = online ? 'on' : '';
    if (lbl) lbl.textContent = online ? 'LIVE' : 'OFFLINE';
    _emit('connection', { online });
  }

  function connect() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${proto}://${location.host}/ws/alerts`;

    try { _ws = new WebSocket(url); } catch(e) { _scheduleRetry(); return; }

    _ws.onopen = () => {
      _retryMs = 1000;
      _setStatus(true);
      _emit('open', {});
    };

    _ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data.ping) return;

        // data may be per-camera state or array of states
        const states = Array.isArray(data) ? data : [data];
        states.forEach(state => {
          if (state.camera_id) {
            _latestState[state.camera_id] = state;
          }
        });

        _emit('state', data);
        _emit('alert', data);

        // Update topbar threat level
        _updateTopbarThreat(data);
      } catch(e) {}
    };

    _ws.onerror = () => {};
    _ws.onclose = () => {
      _setStatus(false);
      _emit('close', {});
      _scheduleRetry();
    };
  }

  function _scheduleRetry() {
    if (_retryTimer) clearTimeout(_retryTimer);
    _retryTimer = setTimeout(() => {
      _retryMs = Math.min(_retryMs * 1.5, _maxRetry);
      connect();
    }, _retryMs);
  }

  function _updateTopbarThreat(data) {
    const chip = document.getElementById('threat-level-chip');
    if (!chip) return;
    const level = data.alert_level || data.level || 'INFO';
    chip.textContent = `THREAT LEVEL: ${level}`;
    chip.className = level;
  }

  function getLatestState(camId) {
    return camId ? _latestState[camId] : _latestState;
  }

  // Clock
  function _startClock() {
    const el = document.getElementById('topbar-clock');
    if (!el) return;
    const tick = () => {
      const now = new Date();
      el.textContent = now.toLocaleTimeString('en-GB', {
        hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'
      });
    };
    tick();
    setInterval(tick, 1000);
  }

  // Topbar system metrics polling
  function _startMetricsPoll() {
    const poll = async () => {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        const states = Object.values(data);
        if (states.length === 0) return;
        const avg_fps = (states.reduce((s, c) => s + (c.inf_fps || 0), 0) / states.length).toFixed(0);
        const avg_lat = Math.round(states.reduce((s, c) => s + (c.inference_ms || 0), 0) / states.length);
        const el_fps = document.getElementById('topbar-fps');
        const el_lat = document.getElementById('topbar-lat');
        if (el_fps) el_fps.textContent = avg_fps;
        if (el_lat) el_lat.textContent = avg_lat + 'ms';
      } catch(e) {}
    };
    poll();
    setInterval(poll, 3000);
  }

  // Auto-initialise
  document.addEventListener('DOMContentLoaded', () => {
    connect();
    _startClock();
    _startMetricsPoll();
  });

  return { on, off, connect, getLatestState };
})();

// Global helpers
function fmtTime(iso) {
  if (!iso) return '--:--:--';
  try {
    return new Date(iso).toLocaleTimeString('en-GB', { hour12: false });
  } catch { return iso; }
}
function fmtDate(iso) {
  if (!iso) return '--/--/--';
  try {
    return new Date(iso).toLocaleDateString('en-GB');
  } catch { return iso; }
}
function levelClass(level) {
  const map = { CRITICAL: 'critical', WARNING: 'warning', INFO: 'info' };
  return map[level] || 'info';
}
function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}
