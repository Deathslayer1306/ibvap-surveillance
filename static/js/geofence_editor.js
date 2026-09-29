/**
 * IBVAP — Geofence Editor Page (geofence_editor.js)
 * Visual zone viewer with enable/disable toggles
 */

let _cameras = [];
let _selectedCam = null;
let _zones = [];

async function loadCamerasAndZones() {
  try {
    const res = await fetch('/api/cameras');
    const { cameras } = await res.json();
    _cameras = cameras;

    // Populate camera selector
    const sel = document.getElementById('cam-selector');
    if (sel) {
      sel.innerHTML = cameras.map(c =>
        `<option value="${c.id}">${c.id} — ${c.name}</option>`
      ).join('');
      sel.addEventListener('change', () => selectCamera(sel.value));
    }

    if (cameras.length > 0) selectCamera(cameras[0].id);
  } catch(e) {
    console.error('Failed to load cameras', e);
  }
}

function selectCamera(camId) {
  _selectedCam = _cameras.find(c => c.id === camId);
  if (!_selectedCam) return;

  _zones = (_selectedCam.zones || []).map(z => ({ ...z, _enabled: z.enabled !== false }));

  // Update video preview
  const vid = document.getElementById('geo-video');
  if (vid) vid.src = `/video_feed?camera=${camId}`;

  renderZoneList();
  drawZones();
}

/* ─── Zone List ──────────────────────────────────────────── */
function renderZoneList() {
  const list = document.getElementById('zone-list');
  if (!list) return;

  if (!_zones.length) {
    list.innerHTML = `
      <div class="empty-state" style="padding:24px;">
        <div class="empty-state-icon" style="font-size:24px;">🔷</div>
        <div class="empty-state-text">No zones configured for this camera.<br>Edit cameras.yaml to add zones.</div>
      </div>`;
    return;
  }

  list.innerHTML = _zones.map((z, i) => `
    <div class="zone-item ${z._enabled ? 'enabled' : 'disabled'}" id="zone-item-${i}">
      <div class="zone-item-left">
        <div class="zone-color-dot" style="background:${zoneColor(i)};"></div>
        <div>
          <div class="zone-name">${z.name || z.id}</div>
          <div class="zone-sub">${z.points?.length || 0} vertices · dwell: ${z.dwell_threshold_sec||0}s</div>
        </div>
      </div>
      <div class="zone-item-right">
        <label class="toggle">
          <input type="checkbox" ${z._enabled ? 'checked' : ''} onchange="toggleZone(${i}, this.checked)"/>
          <span class="toggle-slider"></span>
        </label>
      </div>
    </div>`).join('');
}

function toggleZone(idx, enabled) {
  _zones[idx]._enabled = enabled;
  const item = document.getElementById(`zone-item-${idx}`);
  if (item) {
    item.className = `zone-item ${enabled ? 'enabled' : 'disabled'}`;
  }
  drawZones();
}

/* ─── Canvas Overlay ─────────────────────────────────────── */
function drawZones() {
  const canvas = document.getElementById('zone-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const W = canvas.width;
  const H = canvas.height;

  _zones.forEach((z, i) => {
    if (!z.points || !z.points.length) return;
    const color = zoneColor(i);
    const alpha = z._enabled ? 0.3 : 0.1;

    ctx.beginPath();
    z.points.forEach(([nx, ny], pi) => {
      const x = nx * W, y = ny * H;
      if (pi === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();

    ctx.fillStyle = hexToRgba(color, alpha);
    ctx.fill();

    ctx.strokeStyle = color;
    ctx.lineWidth = z._enabled ? 2 : 1;
    ctx.setLineDash(z._enabled ? [] : [4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Label
    if (z._enabled) {
      const [lx, ly] = centroid(z.points, W, H);
      ctx.fillStyle = 'rgba(0,0,0,.7)';
      ctx.fillRect(lx - 40, ly - 10, 80, 20);
      ctx.fillStyle = color;
      ctx.font = 'bold 10px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(z.name || z.id, lx, ly + 4);
    }
  });
}

function centroid(points, W, H) {
  const x = points.reduce((s, p) => s + p[0], 0) / points.length * W;
  const y = points.reduce((s, p) => s + p[1], 0) / points.length * H;
  return [x, y];
}

function zoneColor(i) {
  const colors = ['#00c8ff', '#00dc78', '#ffaa00', '#ff2828', '#9b59ff', '#ff6b35'];
  return colors[i % colors.length];
}

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1,3), 16);
  const g = parseInt(hex.slice(3,5), 16);
  const b = parseInt(hex.slice(5,7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

/* ─── Resize canvas to match video ──────────────────────── */
function fitCanvas() {
  const vid = document.getElementById('geo-video');
  const canvas = document.getElementById('zone-canvas');
  if (!vid || !canvas) return;
  canvas.width  = vid.offsetWidth  || 640;
  canvas.height = vid.offsetHeight || 360;
  drawZones();
}

/* ─── Init ───────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  loadCamerasAndZones();
  window.addEventListener('resize', fitCanvas);
  setTimeout(fitCanvas, 500);

  // Redraw on video load
  const vid = document.getElementById('geo-video');
  if (vid) vid.addEventListener('load', fitCanvas);
});
