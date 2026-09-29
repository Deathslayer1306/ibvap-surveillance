/**
 * IBVAP — ANPR Tracker Page (anpr_tracker.js)
 * Watchlist registry table with add/delete + live WS plate hits
 */

let _plates = [];

/* ─── Load & Render ──────────────────────────────────────── */
async function loadWatchlist() {
  setLoading(true);
  try {
    const res = await fetch('/api/watchlist/plates/detail');
    const data = await res.json();
    _plates = Array.isArray(data.plates) ? data.plates : [];
    updateRegistryCount(_plates.length);
    renderTable(_plates);
  } catch(e) {
    console.error('Failed to load watchlist', e);
    renderTable([]);
  } finally {
    setLoading(false);
  }
}

function setLoading(v) {
  const tbl = document.getElementById('anpr-tbl-body');
  if (!tbl) return;
  if (v) {
    tbl.innerHTML = `
      <tr><td colspan="6" style="padding:30px;text-align:center;">
        <div class="spinner" style="margin:0 auto;"></div>
        <div style="margin-top:8px;font-size:11px;color:var(--muted)">Loading registry...</div>
      </td></tr>`;
  }
}

function updateRegistryCount(n) {
  const el = document.getElementById('registry-count');
  if (el) el.textContent = `V${n.toString().padStart(2,'0')}`;
}

function renderTable(plates) {
  const tbl = document.getElementById('anpr-tbl-body');
  if (!tbl) return;

  if (!plates.length) {
    tbl.innerHTML = `
      <tr><td colspan="6">
        <div class="empty-state">
          <div class="empty-state-icon">🚗</div>
          <div class="empty-state-text">No plates in watchlist registry</div>
        </div>
      </td></tr>`;
    return;
  }

  tbl.innerHTML = plates.map(p => buildRow(p)).join('');
}

function buildRow(p) {
  const statusMap = {
    blacklisted:    { cls: 'badge-critical', label: 'BLACKLISTED' },
    poi:            { cls: 'badge-warning',  label: 'POI (PERSON OF INTEREST)' },
    cleared:        { cls: 'badge-info',     label: 'CLEARED (VIP)' },
    watch:          { cls: 'badge-purple',   label: 'WATCH' },
    unknown:        { cls: 'badge-dark',     label: 'NO MATCH' },
  };
  const st = statusMap[p.status] || statusMap.unknown;
  const ts = p.timestamp ? fmtDate(p.timestamp) + ' ' + fmtTime(p.timestamp) : '--';

  return `
    <tr id="row-${p.plate}" data-plate="${p.plate}">
      <td style="color:var(--dim);font-family:var(--mono);font-size:10px;">${ts}</td>
      <td><span class="tbl-mono plate-num" style="color:var(--cyan);letter-spacing:2px;">${escHtml(p.plate)}</span></td>
      <td><span style="font-size:11px;color:var(--text);">${escHtml(p.vehicle_type||'Unknown')}</span></td>
      <td><span class="badge ${st.cls}" style="font-size:9px;">${st.label}</span></td>
      <td>
        <div class="row-actions">
          <button class="btn btn-ghost btn-icon btn-sm" data-tooltip="View details"
            onclick="viewPlate('${p.plate}')">👁</button>
          <button class="btn btn-danger btn-icon btn-sm" data-tooltip="Remove"
            onclick="deletePlate('${p.plate}')">🗑</button>
        </div>
      </td>
    </tr>`;
}

/* ─── Search / Filter ────────────────────────────────────── */
function filterTable(query) {
  const q = query.toLowerCase().trim();
  const filtered = q
    ? _plates.filter(p =>
        p.plate.toLowerCase().includes(q) ||
        (p.vehicle_type||'').toLowerCase().includes(q) ||
        (p.status||'').toLowerCase().includes(q)
      )
    : _plates;
  renderTable(filtered);
}

/* ─── Add Plate Modal ────────────────────────────────────── */
function openAddModal() {
  document.getElementById('modal-plate-num').value = '';
  document.getElementById('modal-vehicle-type').value = '';
  document.getElementById('modal-status').value = 'watch';
  document.getElementById('modal-notes').value = '';
  document.getElementById('add-plate-modal').classList.add('open');
  document.getElementById('modal-plate-num').focus();
}
function closeAddModal() {
  document.getElementById('add-plate-modal').classList.remove('open');
}

async function submitAddPlate() {
  const plate       = document.getElementById('modal-plate-num').value.trim().toUpperCase();
  const vehicleType = document.getElementById('modal-vehicle-type').value.trim();
  const status      = document.getElementById('modal-status').value;
  const notes       = document.getElementById('modal-notes').value.trim();

  if (!plate) {
    alert('Please enter a plate number.');
    return;
  }

  const submitBtn = document.getElementById('modal-submit-btn');
  submitBtn.textContent = 'ADDING...';
  submitBtn.disabled = true;

  try {
    const res = await fetch('/api/watchlist/plates/detail', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plate, vehicle_type: vehicleType, status, notes })
    });
    if (!res.ok) throw new Error('Failed');
    closeAddModal();
    await loadWatchlist();
  } catch(e) {
    alert('Failed to add plate. Check the server.');
  } finally {
    submitBtn.textContent = 'ADD ENTRY';
    submitBtn.disabled = false;
  }
}

/* ─── Delete Plate ───────────────────────────────────────── */
async function deletePlate(plate) {
  if (!confirm(`Remove plate ${plate} from watchlist?`)) return;
  try {
    const res = await fetch(`/api/watchlist/plates/${encodeURIComponent(plate)}`, {
      method: 'DELETE'
    });
    if (!res.ok) throw new Error('Failed');
    await loadWatchlist();
  } catch(e) {
    alert('Failed to delete plate.');
  }
}

function viewPlate(plate) {
  const p = _plates.find(x => x.plate === plate);
  if (!p) return;
  alert(`Plate: ${p.plate}\nType: ${p.vehicle_type||'-'}\nStatus: ${p.status}\nNotes: ${p.notes||'-'}`);
}

/* ─── WS Live Hit Flash ──────────────────────────────────── */
function flashPlateHit(plateStr) {
  const rows = document.querySelectorAll(`[data-plate="${plateStr}"]`);
  rows.forEach(row => {
    row.classList.remove('flash');
    void row.offsetWidth; // reflow
    row.classList.add('flash');
  });
}

/* ─── Init ───────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  loadWatchlist();

  // Search
  const searchInput = document.getElementById('anpr-search');
  if (searchInput) {
    searchInput.addEventListener('input', e => filterTable(e.target.value));
  }

  // Add modal
  const addBtn = document.getElementById('add-plate-btn');
  if (addBtn) addBtn.addEventListener('click', openAddModal);

  const cancelBtn = document.getElementById('modal-cancel-btn');
  if (cancelBtn) cancelBtn.addEventListener('click', closeAddModal);

  const submitBtn = document.getElementById('modal-submit-btn');
  if (submitBtn) submitBtn.addEventListener('click', submitAddPlate);

  // Close modal on overlay click
  const overlay = document.getElementById('add-plate-modal');
  if (overlay) {
    overlay.addEventListener('click', e => {
      if (e.target === overlay) closeAddModal();
    });
  }

  // WS: live plate hits
  WS.on('state', (data) => {
    const hits = Array.isArray(data.anpr_hits)
      ? data.anpr_hits
      : (typeof data.anpr_hits === 'string' ? (() => { try { return JSON.parse(data.anpr_hits); } catch { return []; } })() : []);
    hits.forEach(plate => flashPlateHit(plate));
  });

  // Keyboard shortcut
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeAddModal();
  });
});
