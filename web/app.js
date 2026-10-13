/**
 * LifePulse — Emergency Blood Donor Matching (browser client)
 *
 * Maps: Leaflet + OpenStreetMap tiles. Address search: Nominatim (OpenStreetMap).
 * Road routes come from the backend (OSRM). No API keys are needed.
 */

const STATE = {
  apiBase: window.location.origin + '/api/v1',
  token: sessionStorage.getItem('token') || null,
  user: JSON.parse(sessionStorage.getItem('user') || 'null'),
  health: null,
  timers: {},
  maps: {},
  layers: {},
  activeRequestId: null,
  detailView: 'accepted',
  detailData: null,
  detailFitted: false,
  seenInvitations: null,
  donorProfile: null,
  liveWatchId: null,
  lastLiveSentAt: 0,
  alertsEnabled: false,
  newRequest: { lat: null, lon: null, facilityId: null },
  regLocation: null,
  pickLocation: null,
  directory: { page: 1, pageSize: 25, hasNext: false, statesLoaded: false },
};

const NOMINATIM = 'https://nominatim.openstreetmap.org/search';
const OSM_TILES = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
const OSM_ATTRIBUTION = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';
const DEFAULT_CENTER = [20.5937, 78.9629]; // India
const RBC_COMPATIBLE = {
  'O-': ['O-'], 'O+': ['O-', 'O+'], 'A-': ['O-', 'A-'], 'A+': ['O-', 'O+', 'A-', 'A+'],
  'B-': ['O-', 'B-'], 'B+': ['O-', 'O+', 'B-', 'B+'], 'AB-': ['O-', 'A-', 'B-', 'AB-'],
  'AB+': ['O-', 'O+', 'A-', 'A+', 'B-', 'B+', 'AB-', 'AB+'],
};
const ROUTE_SOURCES = {
  osrm: 'OpenStreetMap road route (OSRM, no live traffic)',
  synthetic_fallback: 'Approximate — road routing unavailable',
  synthetic: 'Approximate (offline estimate)',
  google: 'Google Routes',
};

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- helpers

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

async function api(path, options = {}) {
  const headers = { 'Content-Type': 'application/json' };
  if (STATE.token) headers.Authorization = `Bearer ${STATE.token}`;
  const res = await fetch(`${STATE.apiBase}${path}`, {
    method: options.method || 'GET',
    headers,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (res.status === 401 && STATE.token && !path.startsWith('/auth/')) {
    // Sessions end when the server restarts.
    handleLogout('Your session ended. Please sign in again.');
  }
  if (!res.ok) throw new Error(data.error || `HTTP error ${res.status}`);
  return data;
}

function showToast(message, type = 'info') {
  const container = $('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  const text = document.createElement('span');
  text.textContent = message;
  const close = document.createElement('button');
  close.className = 'toast-close';
  close.textContent = '×';
  close.onclick = () => toast.remove();
  toast.append(text, close);
  container.appendChild(toast);
  setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 300); }, 5000);
}

const fmtTime = (iso) => (iso ? new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '—');
const fmtDate = (iso) => (iso ? new Date(iso).toLocaleDateString([], { year: 'numeric', month: 'short', day: 'numeric' }) : '—');
const fmtDateTime = (iso) => (iso ? `${fmtDate(iso)} ${fmtTime(iso)}` : '—');
const fmtKm = (km) => (typeof km === 'number' ? `${km < 10 ? km.toFixed(1) : Math.round(km)} km` : '—');
const fmtPct = (p) => (typeof p === 'number' ? `${Math.round(p * 100)}%` : '—');

function fmtMinutes(min) {
  if (typeof min !== 'number') return '—';
  if (min < 60) return `${Math.round(min)} min`;
  const h = Math.floor(min / 60), m = Math.round(min % 60);
  return m ? `${h} h ${m} min` : `${h} h`;
}

function timeAgo(iso) {
  if (!iso) return '—';
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  const days = Math.round(s / 86400);
  if (days < 60) return `${days} days ago`;
  return `${Math.round(days / 30.4)} months ago`;
}

function fmtCountdown(seconds) {
  if (seconds <= 0) return 'Time limit reached';
  const h = Math.floor(seconds / 3600), m = Math.floor((seconds % 3600) / 60), s = Math.floor(seconds % 60);
  return h ? `${h}h ${String(m).padStart(2, '0')}m` : `${m}m ${String(s).padStart(2, '0')}s`;
}

// Elements with data-deadline show a live countdown.
function tickCountdowns() {
  document.querySelectorAll('[data-deadline]').forEach((el) => {
    const seconds = (new Date(el.dataset.deadline).getTime() - Date.now()) / 1000;
    el.textContent = fmtCountdown(seconds);
    el.classList.toggle('text-critical', seconds > 0 && seconds < 600);
  });
}

function decodePolyline(encoded) {
  if (!encoded) return [];
  const points = [];
  let index = 0, lat = 0, lng = 0;
  while (index < encoded.length) {
    for (const axis of [0, 1]) {
      let b, shift = 0, result = 0;
      do {
        b = encoded.charCodeAt(index++) - 63;
        result |= (b & 0x1f) << shift;
        shift += 5;
      } while (b >= 0x20);
      const delta = (result & 1) ? ~(result >> 1) : (result >> 1);
      if (axis === 0) lat += delta; else lng += delta;
    }
    points.push([lat / 1e5, lng / 1e5]);
  }
  return points;
}

function getPosition(timeout = 10000) {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) { reject(new Error('This browser cannot share location')); return; }
    navigator.geolocation.getCurrentPosition(resolve, (err) => reject(new Error(
      err.code === 1 ? 'Location permission was denied' : 'Could not get a GPS fix')), { enableHighAccuracy: true, timeout, maximumAge: 0 });
  });
}

function positionPayload(pos) {
  return {
    consent: true, source: 'device',
    latitude: pos.coords.latitude, longitude: pos.coords.longitude,
    accuracy_meters: pos.coords.accuracy,
    captured_at: new Date(pos.timestamp || Date.now()).toISOString(),
  };
}

// Nominatim: free OpenStreetMap geocoder. Its usage policy allows searches on
// explicit user action only (no search-as-you-type), at most ~1 per second.
let lastGeocodeAt = 0;
async function geocode(query) {
  const wait = 1100 - (Date.now() - lastGeocodeAt);
  if (wait > 0) await new Promise((r) => setTimeout(r, wait));
  lastGeocodeAt = Date.now();
  const res = await fetch(`${NOMINATIM}?format=jsonv2&limit=6&q=${encodeURIComponent(query)}`, { headers: { 'Accept-Language': 'en' } });
  if (!res.ok) throw new Error('Address search is unavailable right now');
  return (await res.json()).map((r) => ({ name: r.name || r.display_name.split(',')[0], address: r.display_name, lat: +r.lat, lon: +r.lon }));
}

// ---------------------------------------------------------------- maps

function ensureMap(id, center = DEFAULT_CENTER, zoom = 5) {
  if (!window.L) return null;
  if (!STATE.maps[id]) {
    const map = L.map(id).setView(center, zoom);
    L.tileLayer(OSM_TILES, { maxZoom: 19, attribution: OSM_ATTRIBUTION }).addTo(map);
    STATE.maps[id] = map;
    STATE.layers[id] = L.layerGroup().addTo(map);
  }
  // Maps created inside hidden modals need a size refresh once visible.
  setTimeout(() => STATE.maps[id] && STATE.maps[id].invalidateSize(), 120);
  return STATE.maps[id];
}

const hospitalIcon = () => window.L && L.divIcon({ className: 'map-pin pin-hospital', html: '🏥', iconSize: [30, 30], iconAnchor: [15, 15] });
const donorIcon = (cls = 'pin-donor', label = '🩸') => window.L && L.divIcon({ className: `map-pin ${cls}`, html: label, iconSize: [26, 26], iconAnchor: [13, 13] });

function setSinglePin(mapId, lat, lon, icon, zoom = 14) {
  const map = ensureMap(mapId);
  if (!map) return;
  STATE.layers[mapId].clearLayers();
  L.marker([lat, lon], icon ? { icon } : {}).addTo(STATE.layers[mapId]);
  map.setView([lat, lon], zoom);
}

function renderSearchResults(containerId, results, onPick) {
  const box = $(containerId);
  if (!results.length) { box.innerHTML = '<div class="search-empty">No results</div>'; return; }
  box.innerHTML = results.map((r, i) => `
    <button type="button" class="search-result" data-index="${i}">
      <strong>${esc(r.name)}</strong><span>${esc(r.address || '')}${r.distance_km != null ? ` · ${fmtKm(r.distance_km)}` : ''}</span>
    </button>`).join('');
  box.querySelectorAll('.search-result').forEach((btn) => {
    btn.onclick = () => { onPick(results[+btn.dataset.index]); box.innerHTML = ''; };
  });
}

// ---------------------------------------------------------------- navigation & session

function stopTimers(...names) {
  (names.length ? names : Object.keys(STATE.timers)).forEach((n) => { clearInterval(STATE.timers[n]); delete STATE.timers[n]; });
}

function switchTab(tab) {
  const isStaff = STATE.user?.role === 'hospital_staff';
  if (!isStaff && tab !== 'dashboard') tab = 'dashboard';
  document.querySelectorAll('.nav-btn').forEach((b) => b.classList.toggle('active', b.dataset.tab === tab));
  document.querySelectorAll('.portal-view').forEach((v) => { v.style.display = 'none'; });
  stopTimers('staffRequests', 'donorInvites', 'donorProfile');

  if (tab === 'dashboard') {
    if (isStaff) {
      $('view-staff').style.display = 'block';
      fetchStaffRequests();
      STATE.timers.staffRequests = setInterval(fetchStaffRequests, 5000);
    } else {
      $('view-donor').style.display = 'block';
      fetchDonorProfile();
      fetchDonorInvitations();
      STATE.timers.donorInvites = setInterval(fetchDonorInvitations, 4000);
      STATE.timers.donorProfile = setInterval(fetchDonorProfile, 30000);
    }
  } else if (tab === 'directory') {
    $('view-directory').style.display = 'block';
    ensureMap('directory-map');
    loadDirectory();
  } else if (tab === 'system') {
    $('view-system').style.display = 'block';
    checkHealth();
  }
}

function setAuthMode(mode) {
  $('auth-tab-login').classList.toggle('active', mode === 'login');
  $('auth-tab-register').classList.toggle('active', mode === 'register');
  $('form-login').style.display = mode === 'login' ? 'block' : 'none';
  $('form-register').style.display = mode === 'register' ? 'block' : 'none';
}

function saveSession(data) {
  STATE.token = data.token;
  STATE.user = { user_id: data.user_id, name: data.name, role: data.role };
  sessionStorage.setItem('token', STATE.token);
  sessionStorage.setItem('user', JSON.stringify(STATE.user));
}

async function handleLoginSubmit(event) {
  event.preventDefault();
  const payload = { username: $('login-username').value.trim() };
  if ($('login-password').value) payload.password = $('login-password').value;
  try {
    const data = await api('/auth/login', { method: 'POST', body: payload });
    saveSession(data);
    showToast(`Welcome, ${data.name}`, 'success');
    renderSessionState();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function handleRegisterSubmit(event) {
  event.preventDefault();
  const prior = parseInt($('reg-prior-count').value || '0', 10);
  const lastDate = $('reg-last-date').value;
  const firstDate = $('reg-first-date').value;
  const payload = {
    username: $('reg-username').value.trim(),
    name: $('reg-fullname').value.trim(),
    blood_type: $('reg-bloodtype').value,
    phone: $('reg-phone').value.trim(),
    email: $('reg-email').value.trim(),
    password: $('reg-password').value,
  };
  if (prior > 0 || lastDate) {
    if (!lastDate) { showToast('Enter the date of your last donation', 'error'); return; }
    payload.prior_donation_count = Math.max(prior, 1);
    payload.last_donation_date = lastDate;
    if (firstDate) payload.first_donation_date = firstDate;
  }
  if ($('reg-location-consent').checked) {
    if (!STATE.regLocation) { showToast('Set your location on the map, or untick location sharing', 'error'); return; }
    payload.location = { ...STATE.regLocation, captured_at: new Date().toISOString() };
  }
  try {
    const data = await api('/auth/register', { method: 'POST', body: payload });
    saveSession(data);
    showToast(`Registered. Welcome, ${data.name}!`, 'success');
    renderSessionState();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

function handleLogout(message = 'Signed out.') {
  stopTimers();
  stopLiveLocation();
  STATE.token = null;
  STATE.user = null;
  STATE.seenInvitations = null;
  sessionStorage.removeItem('token');
  sessionStorage.removeItem('user');
  document.querySelectorAll('.modal-backdrop').forEach((m) => { m.style.display = 'none'; });
  renderSessionState();
  showToast(message, 'info');
}

function renderSessionState() {
  const signedIn = Boolean(STATE.token && STATE.user);
  $('main-nav').style.display = signedIn ? 'flex' : 'none';
  $('user-badge').style.display = signedIn ? 'flex' : 'none';
  $('view-auth').style.display = signedIn ? 'none' : 'block';
  if (!signedIn) {
    document.querySelectorAll('.portal-view').forEach((v) => { v.style.display = 'none'; });
    return;
  }
  const isStaff = STATE.user.role === 'hospital_staff';
  $('current-user-role').textContent = isStaff ? 'DOCTOR' : 'DONOR';
  $('current-user-name').textContent = STATE.user.name;
  $('nav-dashboard-label').textContent = isStaff ? 'Requests' : 'My Alerts';
  // Donors do not need the blood-bank directory or system pages.
  document.querySelectorAll('.staff-only').forEach((el) => { el.style.display = isStaff ? '' : 'none'; });
  switchTab('dashboard');
}

// ---------------------------------------------------------------- doctor: requests list

async function fetchStaffRequests() {
  try {
    const { requests = [] } = await api('/emergency/requests');
    renderStaffRequests(requests);
    const active = requests.filter((r) => r.status === 'ACTIVE');
    $('kpi-active-requests').textContent = active.length;
    $('kpi-accepted').textContent = active.reduce((s, r) => s + (r.accepted_count || 0), 0);
    $('kpi-fulfilled-units').textContent = requests.reduce((s, r) => s + (r.units_fulfilled || 0), 0);
  } catch (err) {
    console.error('Failed to fetch requests', err);
  }
}

function statusBadge(status) {
  const cls = { ACTIVE: 'badge-critical', FULFILLED: 'badge-success', EXPIRED: 'badge-info', CANCELLED: 'badge-info' }[status] || 'badge-info';
  const label = { ACTIVE: 'OPEN' }[status] || status;
  return `<span class="badge-pill ${cls}">${esc(label)}</span>`;
}

function renderStaffRequests(requests) {
  const tbody = $('emergency-requests-body');
  if (!requests.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="text-center text-muted py-4">No requests yet. Click “Raise Blood Request”.</td></tr>';
    return;
  }
  tbody.innerHTML = requests.map((r) => {
    const open = r.status === 'ACTIVE';
    const blood = `<span class="blood-tag">${esc(r.blood_type_required)}</span>${r.include_compatible ? '<span class="tiny-note">+ compatible</span>' : ''}`;
    return `
      <tr class="${open ? '' : 'row-closed'}">
        <td><strong>${esc(r.request_id)}</strong><div class="tiny-note">${fmtDateTime(r.created_at)}</div></td>
        <td>${esc(r.hospital_name)}<div class="tiny-note">${esc(r.hospital_address || '')}</div></td>
        <td>${blood}</td>
        <td>${r.units_fulfilled} / ${r.units_required}</td>
        <td>${open ? `<strong data-deadline="${esc(r.deadline_at)}">${fmtCountdown(r.seconds_remaining)}</strong>` : '—'}
            <div class="tiny-note">limit ${fmtMinutes(r.time_limit_minutes)}</div></td>
        <td><span class="count-chip ok">${r.accepted_count} accepted</span>
            <span class="count-chip">${r.pending_count} waiting</span>
            <div class="tiny-note">${r.invited_count} alerted · ${r.declined_count} declined</div></td>
        <td>${statusBadge(r.status)}</td>
        <td><button class="btn btn-sm btn-primary" data-action="open-request" data-id="${esc(r.request_id)}">View donors</button></td>
      </tr>`;
  }).join('');
}

// ---------------------------------------------------------------- doctor: new request

function updateCompatibleLabel() {
  $('req-compatible-groups').textContent = RBC_COMPATIBLE[$('req-bloodtype').value].join(', ');
}

function setRequestLocation(lat, lon, { name, address, facilityId = null } = {}) {
  STATE.newRequest = { lat, lon, facilityId };
  if (name !== undefined) $('req-hospital-name').value = name;
  if (address !== undefined) $('req-hospital-address').value = address;
  setSinglePin('req-map', lat, lon, hospitalIcon(), 15);
  $('req-location-label').textContent = `Hospital location: ${lat.toFixed(5)}, ${lon.toFixed(5)}${facilityId ? ' (from blood bank directory)' : ''}`;
  $('req-facility-confirm-row').style.display = facilityId ? 'flex' : 'none';
  $('req-facility-confirm').checked = false;
}

function openNewRequestModal(prefill) {
  $('modal-request').style.display = 'flex';
  updateCompatibleLabel();
  $('req-contact-phone').value = localStorage.getItem('lp_contact_phone') || $('req-contact-phone').value;
  const map = ensureMap('req-map');
  if (map && !map._clickBound) {
    map.on('click', (e) => setRequestLocation(e.latlng.lat, e.latlng.lng));
    map._clickBound = true;
  }
  if (prefill) {
    setRequestLocation(prefill.latitude, prefill.longitude, { name: prefill.name, address: prefill.address, facilityId: prefill.facility_id });
    return;
  }
  const last = JSON.parse(localStorage.getItem('lp_last_hospital') || 'null');
  if (last && STATE.newRequest.lat == null) {
    setRequestLocation(last.lat, last.lon, { name: last.name, address: last.address });
  }
}

async function searchFacilitiesForRequest() {
  const q = $('req-facility-search').value.trim();
  if (!q) { showToast('Type a blood bank name, city or district', 'info'); return; }
  try {
    const data = await api(`/facilities?search=${encodeURIComponent(q)}&has_coordinates=true&page_size=8`);
    const results = data.facilities.map((f) => ({ ...f, address: facilityAddress(f) }));
    renderSearchResults('req-search-results', results, (f) => setRequestLocation(f.latitude, f.longitude,
      { name: f.name, address: f.address, facilityId: f.facility_id }));
  } catch (err) { showToast(err.message, 'error'); }
}

async function searchAddressForRequest() {
  const q = $('req-facility-search').value.trim() || $('req-hospital-address').value.trim() || $('req-hospital-name').value.trim();
  if (!q) { showToast('Type a hospital name or address to search', 'info'); return; }
  try {
    const results = await geocode(q);
    renderSearchResults('req-search-results', results, (r) => setRequestLocation(r.lat, r.lon,
      { name: $('req-hospital-name').value || r.name, address: r.address }));
  } catch (err) { showToast(err.message, 'error'); }
}

async function handleCreateRequestSubmit(event) {
  event.preventDefault();
  const loc = STATE.newRequest;
  if (loc.lat == null) { showToast('Set the hospital location (search or click the map)', 'error'); return; }
  const limitChoice = $('req-time-limit').value;
  const timeLimit = parseInt(limitChoice === 'custom' ? $('req-time-custom').value : limitChoice, 10);
  if (!(timeLimit >= 10 && timeLimit <= 2880)) { showToast('Time limit must be 10–2880 minutes', 'error'); return; }
  const radius = $('req-radius').value;
  const payload = {
    blood_type_required: $('req-bloodtype').value,
    units_required: parseInt($('req-units').value, 10),
    time_limit_minutes: timeLimit,
    search_radius_km: radius ? Number(radius) : null,
    include_compatible: $('req-compatible').checked,
    contact_phone: $('req-contact-phone').value.trim(),
    hospital_name: $('req-hospital-name').value.trim(),
    hospital_address: $('req-hospital-address').value.trim(),
    notes: $('req-notes').value.trim(),
    latitude: loc.lat,
    longitude: loc.lon,
  };
  if (loc.facilityId) {
    if (!$('req-facility-confirm').checked) { showToast('Confirm the directory entry, or click the map to set the location yourself', 'error'); return; }
    payload.facility_id = loc.facilityId;
    payload.facility_confirmed = true;
  }
  try {
    const data = await api('/emergency/requests', { method: 'POST', body: payload });
    localStorage.setItem('lp_contact_phone', payload.contact_phone);
    localStorage.setItem('lp_last_hospital', JSON.stringify({ lat: loc.lat, lon: loc.lon, name: payload.hospital_name, address: payload.hospital_address }));
    showToast(data.invited_count
      ? `Request sent to ${data.invited_count} matching donor${data.invited_count === 1 ? '' : 's'}.`
      : 'Request is open. No matching donor can reach you in time yet — new donors are added automatically.', data.invited_count ? 'success' : 'info');
    closeModal('modal-request');
    fetchStaffRequests();
    openRequestDetail(data.request_id);
  } catch (err) {
    showToast(err.message, 'error');
  }
}

// ---------------------------------------------------------------- doctor: request detail

function openRequestDetail(requestId) {
  STATE.activeRequestId = requestId;
  STATE.detailFitted = false;
  $('modal-recruitment').style.display = 'flex';
  ensureMap('detail-map');
  refreshRequestDetail();
  stopTimers('detail');
  STATE.timers.detail = setInterval(refreshRequestDetail, 3000);
}

async function refreshRequestDetail() {
  if (!STATE.activeRequestId) return;
  try {
    const data = await api(`/emergency/recommendations?request_id=${encodeURIComponent(STATE.activeRequestId)}&page_size=500`);
    STATE.detailData = data;
    renderRequestDetail(data);
  } catch (err) {
    console.error('Failed to refresh request', err);
  }
}

function switchDetailView(view) {
  STATE.detailView = view;
  $('tab-accepted').classList.toggle('active', view === 'accepted');
  $('tab-candidates').classList.toggle('active', view === 'candidates');
  if (STATE.detailData) renderRequestDetail(STATE.detailData);
}

function renderRequestDetail(data) {
  const req = data.emergency_request;
  const sum = data.recruitment;
  const open = req.status === 'ACTIVE';
  $('recruitment-modal-title').textContent = `${req.blood_type_required} blood · ${req.hospital_name}`;
  $('recruitment-modal-subtitle').textContent =
    `${req.request_id} · raised ${fmtDateTime(req.created_at)} by ${req.doctor_name || 'staff'} · alerting ${sum.matching_blood_types.join(', ')} donors ` +
    `${sum.search_radius_km ? `within ${sum.search_radius_km} km` : 'at any distance'} who can arrive in ${fmtMinutes(sum.time_limit_minutes)}`;
  $('detail-status').innerHTML = statusBadge(req.status);
  const countdown = $('detail-countdown');
  if (open) { countdown.dataset.deadline = sum.deadline_at; countdown.textContent = fmtCountdown(sum.seconds_remaining); }
  else { delete countdown.dataset.deadline; countdown.textContent = req.status === 'FULFILLED' ? 'Fulfilled' : 'Closed'; }
  $('detail-units').textContent = `${sum.units_fulfilled} donated / ${sum.units_required} needed`;
  $('detail-alerted').textContent = `${sum.invited_count} (${sum.pending_count} waiting, ${sum.declined_count} declined)`;
  $('detail-accepted').textContent = `${sum.accepted_count} on the way · ${sum.remaining_needed} still needed`;
  $('btn-cancel-request').style.display = open ? '' : 'none';
  $('tab-accepted').textContent = `Accepted donors (${data.accepted_donors.length})`;
  $('tab-candidates').textContent = `All alerted donors (${sum.invited_count})`;

  renderDetailMap(req, data.accepted_donors);
  const container = $('candidates-container');
  if (STATE.detailView === 'accepted') {
    container.innerHTML = data.accepted_donors.length
      ? data.accepted_donors.map((d) => acceptedDonorCard(d, req)).join('')
      : `<div class="empty-state"><span class="empty-icon">⏳</span><p>${open
          ? (sum.invited_count ? `Waiting for donors to accept. ${sum.pending_count} donor(s) have the alert.` : 'No matching donor can reach the hospital in time yet. Donors who register or share their location are alerted automatically.')
          : 'No donor accepted this request.'}</p></div>`;
  } else {
    container.innerHTML = alertedDonorsTable(data.recommendations);
  }
}

function renderDetailMap(req, accepted) {
  const map = ensureMap('detail-map');
  if (!map) return;
  const layer = STATE.layers['detail-map'];
  layer.clearLayers();
  const points = [[req.latitude, req.longitude]];
  L.marker([req.latitude, req.longitude], { icon: hospitalIcon() }).bindPopup(`<b>${esc(req.hospital_name)}</b>`).addTo(layer);
  if (req.search_radius_km) {
    L.circle([req.latitude, req.longitude], { radius: req.search_radius_km * 1000, color: '#E11D48', weight: 1, fillOpacity: 0.04 }).addTo(layer);
  }
  accepted.forEach((d) => {
    if (d.latitude == null) return;
    points.push([d.latitude, d.longitude]);
    L.marker([d.latitude, d.longitude], { icon: donorIcon(d.invitation_status === 'COMPLETED' ? 'pin-done' : 'pin-donor') })
      .bindPopup(`<b>${esc(d.donor_name)}</b><br>${esc(d.blood_type)} · ETA ${fmtMinutes(d.travel_time_minutes)}`).addTo(layer);
  });
  $('detail-map-label').textContent = accepted.length
    ? 'Hospital (🏥) and donors who accepted (🩸). Donor locations are shown only after they accept.'
    : 'Hospital location. Donor locations appear here once they accept.';
  if (!STATE.detailFitted) {
    if (points.length > 1) map.fitBounds(points, { padding: [40, 40], maxZoom: 15 });
    else map.setView(points[0], req.search_radius_km ? Math.max(8, 13 - Math.log2(req.search_radius_km / 2)) : 12);
    STATE.detailFitted = true;
  }
}

function locationNote(d) {
  if (d.location_live) return '<span class="text-success">live GPS</span>';
  if (d.location_source === 'manual') return `pinned on map (${timeAgo(d.location_updated_at)})`;
  if (d.location_source === 'device') return `GPS ${timeAgo(d.location_updated_at)}`;
  return 'demo location';
}

function acceptedDonorCard(d, req) {
  const coming = d.invitation_status === 'ACCEPTED';
  const late = coming && !d.is_feasible;
  const phone = d.phone && /\d/.test(d.phone) ? `<a href="tel:${esc(d.phone.replace(/\s/g, ''))}">${esc(d.phone)}</a>` : esc(d.phone || 'not provided');
  return `
    <div class="donor-card ${coming ? '' : 'donor-card-done'}">
      <div class="donor-card-head">
        <span class="blood-tag">${esc(d.blood_type)}</span>
        <div>
          <h4>${esc(d.donor_name)} <span class="tiny-note">@${esc(d.donor_id)}</span></h4>
          <div class="tiny-note">${coming ? `Accepted ${fmtTime(d.accepted_at)} (${timeAgo(d.accepted_at)})` : `Donated ${fmtDateTime(d.completed_at)}`}</div>
        </div>
        <span class="badge-pill ${coming ? 'badge-success' : 'badge-info'}">${coming ? 'ON THE WAY' : 'DONATED'}</span>
      </div>
      <div class="donor-facts">
        <div><span>📞 Contact</span><strong>${phone}</strong></div>
        <div><span>✉️ Email</span><strong>${d.email ? `<a href="mailto:${esc(d.email)}">${esc(d.email)}</a>` : '—'}</strong></div>
        <div><span>🩸 Last donation</span><strong>${d.last_donation_at ? `${fmtDate(d.last_donation_at)} (${timeAgo(d.last_donation_at)})` : 'No previous donation'}</strong></div>
        <div><span>Total donations</span><strong>${d.total_donations}</strong></div>
        ${coming ? `
        <div><span>🚗 Road ETA</span><strong class="${late ? 'text-critical' : ''}">${fmtMinutes(d.travel_time_minutes)}${late ? ' — may miss the time limit' : ''}</strong></div>
        <div><span>Expected arrival</span><strong>${fmtTime(d.expected_arrival_at)}</strong></div>
        <div><span>Distance</span><strong>${fmtKm(d.road_distance_km ?? d.distance_km)}</strong></div>
        <div><span>Location</span><strong>${locationNote(d)}</strong></div>` : ''}
      </div>
      <div class="donor-card-actions">
        ${coming ? `<button class="btn btn-sm btn-ghost" data-action="route" data-donor="${esc(d.donor_id)}" data-request="${esc(req.request_id)}">🗺️ Route</button>
        <button class="btn btn-sm btn-primary" data-action="confirm-donation" data-donor="${esc(d.donor_id)}" data-request="${esc(req.request_id)}">✅ Donor arrived — confirm donation</button>` : ''}
      </div>
    </div>`;
}

function alertedDonorsTable(rows) {
  if (!rows.length) return '<div class="empty-state"><span class="empty-icon">👥</span><p>No donors alerted yet.</p></div>';
  const statusLabel = {
    PENDING: ['badge-warning', 'WAITING'], ACCEPTED: ['badge-success', 'ACCEPTED'], COMPLETED: ['badge-info', 'DONATED'],
    DECLINED: ['badge-info', 'DECLINED'], WITHDRAWN: ['badge-info', 'WITHDREW'], CANCELLED: ['badge-info', 'CLOSED'], EXPIRED: ['badge-info', 'EXPIRED'],
  };
  return `
    <div class="table-responsive"><table class="data-table">
      <thead><tr><th>#</th><th>Donor</th><th>Blood</th><th>Distance</th><th>Road ETA</th><th title="Predicted from the donor's donation history">Likely to donate</th><th>Match</th><th>Status</th><th>Alerted</th></tr></thead>
      <tbody>${rows.map((r) => {
        const [cls, label] = statusLabel[r.invitation_status] || ['badge-info', r.invitation_status];
        return `<tr>
          <td>${r.rank}</td>
          <td><strong>${esc(r.donor_name)}</strong>${r.total_donations != null ? `<div class="tiny-note">${r.total_donations} donations${r.last_donation_at ? `, last ${timeAgo(r.last_donation_at)}` : ''}</div>` : ''}</td>
          <td><span class="blood-tag">${esc(r.blood_type)}</span></td>
          <td>${fmtKm(r.road_distance_km ?? r.distance_km)}</td>
          <td>${fmtMinutes(r.travel_time_minutes)}${r.can_arrive_in_time === false ? '<div class="tiny-note text-critical">too far now</div>' : ''}</td>
          <td>${r.donation_likelihood != null ? fmtPct(r.donation_likelihood) : '<span class="tiny-note">no history</span>'}</td>
          <td><div class="score-bar"><div style="width:${Math.round((r.match_score || 0) * 100)}%"></div></div></td>
          <td><span class="badge-pill ${cls}">${label}</span></td>
          <td class="tiny-note">${fmtTime(r.sent_at)}${r.responded_at ? `<br>answered ${fmtTime(r.responded_at)}` : ''}</td>
        </tr>`;
      }).join('')}</tbody>
    </table></div>
    <p class="field-help">Contact details appear in “Accepted donors” once a donor accepts. Ranking combines predicted likelihood to donate (from donation history) with road travel time.</p>`;
}

async function confirmDonation(donorId, requestId) {
  if (!confirm('Confirm that this donor arrived and the donation was completed?')) return;
  try {
    const data = await api('/emergency/donation/complete', { method: 'POST', body: { donor_id: donorId, request_id: requestId } });
    showToast(`Donation recorded. ${data.request_status === 'FULFILLED' ? 'Request fulfilled.' : ''}`, 'success');
    refreshRequestDetail();
    fetchStaffRequests();
  } catch (err) { showToast(err.message, 'error'); }
}

async function cancelActiveRequest() {
  if (!STATE.activeRequestId || !confirm('Cancel this request? All donors will be told it is no longer needed.')) return;
  try {
    await api('/emergency/requests/cancel', { method: 'POST', body: { request_id: STATE.activeRequestId } });
    showToast('Request cancelled.', 'info');
    refreshRequestDetail();
    fetchStaffRequests();
  } catch (err) { showToast(err.message, 'error'); }
}

function closeModal(id) {
  $(id).style.display = 'none';
  if (id === 'modal-recruitment') { stopTimers('detail'); STATE.activeRequestId = null; }
}

// ---------------------------------------------------------------- route modal

async function openRouteModal(donorId, requestId) {
  $('modal-route').style.display = 'flex';
  $('route-distance-val').textContent = '…';
  $('route-duration-val').textContent = '…';
  $('route-provider-val').textContent = 'Loading route…';
  const map = ensureMap('route-map');
  STATE.layers['route-map']?.clearLayers();
  try {
    const data = await api(`/emergency/routes?donor_id=${encodeURIComponent(donorId)}&request_id=${encodeURIComponent(requestId)}`);
    $('route-modal-title').textContent = `Route to ${data.hospital_name}`;
    $('route-modal-subtitle').textContent = `${data.donor_name} → ${data.hospital_address || data.hospital_name}`;
    $('route-distance-val').textContent = fmtKm(data.distance_km);
    $('route-duration-val').textContent = fmtMinutes(data.duration_minutes);
    $('route-provider-val').textContent = ROUTE_SOURCES[data.provider] || data.provider || '—';
    const o = data.origin, d = data.destination;
    $('route-osm-link').href = `https://www.openstreetmap.org/directions?engine=fossgis_osrm_car&route=${o.latitude},${o.longitude};${d.latitude},${d.longitude}`;
    if (!map) return;
    const layer = STATE.layers['route-map'];
    L.marker([o.latitude, o.longitude], { icon: donorIcon() }).bindPopup(`<b>${esc(data.donor_name)}</b><br>Donor location`).addTo(layer);
    L.marker([d.latitude, d.longitude], { icon: hospitalIcon() }).bindPopup(`<b>${esc(data.hospital_name)}</b>`).addTo(layer);
    let points = decodePolyline(data.polyline_encoded);
    const roadGeometry = points.length >= 2;
    if (!roadGeometry) points = [[o.latitude, o.longitude], [d.latitude, d.longitude]];
    const line = L.polyline(points, { color: '#E11D48', weight: 5, opacity: 0.85, dashArray: roadGeometry ? null : '8, 8' }).addTo(layer);
    setTimeout(() => { map.invalidateSize(); map.fitBounds(line.getBounds(), { padding: [40, 40] }); }, 150);
    if (!roadGeometry) $('route-provider-val').textContent += ' · straight line shown';
  } catch (err) {
    $('route-provider-val').textContent = err.message;
    showToast(err.message, 'error');
  }
}

// ---------------------------------------------------------------- donor portal

async function fetchDonorProfile() {
  try {
    const p = await api('/donor/profile');
    STATE.donorProfile = p;
    renderDonorProfile(p);
  } catch (err) { console.error('Failed to fetch profile', err); }
}

function renderDonorProfile(p) {
  $('donor-profile-name').textContent = p.name;
  $('donor-profile-id').textContent = p.user_id;
  $('donor-profile-bloodtype').textContent = p.blood_type;
  $('donor-profile-phone').textContent = p.phone || '—';
  $('donor-profile-email').textContent = p.email || '—';
  $('donor-profile-donations').textContent = p.total_donations;
  $('donor-profile-last').textContent = p.last_donation_at ? `${fmtDate(p.last_donation_at)} (${timeAgo(p.last_donation_at)})` : 'None recorded';

  const badge = $('donor-eligibility-badge');
  if (!p.is_temporally_eligible) {
    badge.className = 'badge-pill badge-critical';
    badge.textContent = 'Resting after donation';
    $('donor-next-eligible-row').style.display = 'flex';
    $('donor-next-eligible-date').textContent = fmtDate(p.next_eligible_at);
  } else if (!p.availability) {
    badge.className = 'badge-pill badge-warning';
    badge.textContent = 'Not receiving alerts';
    $('donor-next-eligible-row').style.display = 'none';
  } else {
    badge.className = 'badge-pill badge-success';
    badge.textContent = 'Receiving alerts';
    $('donor-next-eligible-row').style.display = 'none';
  }
  $('donor-available-toggle').checked = Boolean(p.availability);

  if (!document.activeElement || !document.activeElement.closest('#form-donor-edit')) {
    $('edit-phone').value = p.phone || '';
    $('edit-email').value = p.email || '';
  }

  const registered = p.account_source === 'registered';
  $('donor-current-lat').textContent = p.latitude != null ? p.latitude.toFixed(5) : '—';
  $('donor-current-lon').textContent = p.longitude != null ? p.longitude.toFixed(5) : '—';
  $('donor-current-updated').textContent = p.location_captured_at ? timeAgo(p.location_captured_at) : (registered ? '—' : 'demo');
  const loc = $('donor-location-status-badge');
  if (!p.location_ready) { loc.className = 'badge-pill badge-warning'; loc.textContent = 'Not shared — no alerts'; }
  else if (p.location_live) { loc.className = 'badge-pill badge-success'; loc.textContent = 'Live GPS'; }
  else if (p.location_source === 'manual') { loc.className = 'badge-pill badge-info'; loc.textContent = 'Pinned on map'; }
  else if (registered) { loc.className = 'badge-pill badge-info'; loc.textContent = 'Last known GPS'; }
  else { loc.className = 'badge-pill badge-info'; loc.textContent = 'Demo location'; }
  ['btn-share-location', 'btn-pick-location', 'btn-revoke-location', 'donor-live-toggle'].forEach((id) => { $(id).disabled = !registered; });
}

function beep() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    [0, 0.35].forEach((t) => {
      const osc = ctx.createOscillator(); const gain = ctx.createGain();
      osc.frequency.value = 880; osc.connect(gain); gain.connect(ctx.destination);
      gain.gain.setValueAtTime(0.2, ctx.currentTime + t);
      osc.start(ctx.currentTime + t); osc.stop(ctx.currentTime + t + 0.2);
    });
  } catch (e) { /* audio unavailable */ }
}

function notifyNewAlerts(newOnes) {
  if (!newOnes.length) return;
  const first = newOnes[0];
  showToast(`🚨 ${first.blood_type_required} blood needed at ${first.hospital_name}`, 'error');
  if (STATE.alertsEnabled) beep();
  if (STATE.alertsEnabled && 'Notification' in window && Notification.permission === 'granted') {
    new Notification('Emergency blood request', { body: `${first.blood_type_required} needed at ${first.hospital_name}. Open LifePulse to respond.` });
  }
}

async function fetchDonorInvitations() {
  try {
    const { invitations = [] } = await api('/donor/invitations');
    const pending = invitations.filter((i) => i.status === 'PENDING');
    if (STATE.seenInvitations) notifyNewAlerts(pending.filter((i) => !STATE.seenInvitations.has(i.invitation_id)));
    STATE.seenInvitations = new Set(invitations.map((i) => i.invitation_id));
    renderDonorInvitations(invitations);
  } catch (err) { console.error('Failed to poll invitations', err); }
}

const CLOSED_TEXT = {
  TARGET_REACHED: 'Enough donors have already confirmed. Thank you!',
  REQUEST_CANCELLED: 'The hospital cancelled this request.',
  REQUEST_FULFILLED: 'The request has been fulfilled. Thank you!',
  REQUEST_EXPIRED: 'The time limit passed.',
  DONOR_DONATED: 'You donated recently — thank you!',
  DONOR_UNAVAILABLE: 'Closed because you switched alerts off.',
  LOCATION_REVOKED: 'Closed because you stopped sharing your location.',
};

function renderDonorInvitations(invitations) {
  const container = $('donor-invitations-container');
  const pending = invitations.filter((i) => i.status === 'PENDING');
  const badge = $('invitation-count-badge');
  badge.textContent = `${pending.length} new`;
  badge.className = `badge-pill ${pending.length ? 'badge-critical pulse-badge' : ''}`;
  if (!invitations.length) {
    const p = STATE.donorProfile;
    const hint = p && !p.location_ready ? 'Share your location below so nearby hospitals can reach you.' : 'You will be alerted here when a nearby hospital needs your blood group.';
    container.innerHTML = `<div class="empty-state"><span class="empty-icon">🔔</span><p>No emergency alerts right now. ${hint}</p></div>`;
    return;
  }
  const active = invitations.filter((i) => i.status === 'PENDING' || i.status === 'ACCEPTED');
  const past = invitations.filter((i) => i.status !== 'PENDING' && i.status !== 'ACCEPTED');
  container.innerHTML = active.map(invitationCard).join('') + (past.length ? `
    <details class="past-alerts"><summary>Past alerts (${past.length})</summary>
      ${past.map((i) => `<div class="past-row"><span class="blood-tag">${esc(i.blood_type_required)}</span>
        <span>${esc(i.hospital_name)} · ${fmtDateTime(i.sent_at)}</span>
        <span class="badge-pill badge-info">${esc(i.status === 'COMPLETED' ? 'DONATED' : i.status)}</span>
        <span class="tiny-note">${esc(CLOSED_TEXT[i.closed_reason] || '')}</span></div>`).join('')}
    </details>` : '');
}

function invitationCard(inv) {
  const accepted = inv.status === 'ACCEPTED';
  const phone = inv.contact_phone ? `<a href="tel:${esc(inv.contact_phone.replace(/\s/g, ''))}">${esc(inv.contact_phone)}</a>` : 'not provided';
  return `
    <div class="invitation-card ${accepted ? 'invitation-accepted' : 'invitation-pending'}">
      <div class="invitation-body">
        <div class="invitation-title">
          <span class="blood-tag">${esc(inv.blood_type_required)}</span>
          <span>${accepted ? 'You accepted — please go to' : 'Blood needed at'} <strong>${esc(inv.hospital_name)}</strong></span>
          <span class="badge-pill ${accepted ? 'badge-success' : 'badge-critical pulse-badge'}">${accepted ? 'CONFIRMED' : 'URGENT'}</span>
        </div>
        <div class="invitation-details">
          <span>⏱️ Arrive within: <strong data-deadline="${esc(inv.deadline_at)}">${fmtCountdown(inv.seconds_remaining)}</strong></span>
          <span>📍 ${esc(inv.hospital_address || `${inv.hospital_lat.toFixed(4)}, ${inv.hospital_lon.toFixed(4)}`)}</span>
          <span>🚗 Your travel time: <strong>${fmtMinutes(inv.travel_time_minutes)}</strong> (${fmtKm(inv.road_distance_km ?? inv.distance_km)})</span>
          <span>👨‍⚕️ ${esc(inv.doctor_name || 'Hospital staff')} · 📞 ${phone}</span>
          <span>🩸 Donors still needed: <strong>${inv.remaining_needed ?? '—'}</strong> of ${inv.units_required ?? '—'}</span>
          ${inv.notes ? `<span>📝 ${esc(inv.notes)}</span>` : ''}
        </div>
      </div>
      <div class="invitation-actions">
        ${accepted ? '' : `<button class="btn btn-primary" data-action="respond" data-id="${esc(inv.invitation_id)}" data-response="ACCEPT">✅ I can donate</button>
        <button class="btn btn-ghost" data-action="respond" data-id="${esc(inv.invitation_id)}" data-response="DECLINE">Can't come</button>`}
        <button class="btn btn-ghost" data-action="route" data-donor="${esc(inv.donor_id)}" data-request="${esc(inv.request_id)}">🗺️ Route</button>
        ${accepted ? `<button class="btn btn-ghost" data-action="respond" data-id="${esc(inv.invitation_id)}" data-response="WITHDRAW">I can no longer come</button>` : ''}
      </div>
    </div>`;
}

async function respondInvitation(invitationId, action) {
  if (action === 'WITHDRAW' && !confirm('Tell the hospital you can no longer come?')) return;
  const body = { invitation_id: invitationId, action };
  if (action === 'ACCEPT' && STATE.donorProfile?.account_source === 'registered') {
    // Check the travel time from where the donor is now; fall back to the saved location.
    try { body.location = positionPayload(await getPosition(8000)); } catch (e) { /* use saved location */ }
  }
  try {
    await api('/donor/invitations/respond', { method: 'POST', body });
    showToast({ ACCEPT: 'Accepted — the doctor can now see your details. Please head to the hospital.', DECLINE: 'Declined. Thank you for responding.', WITHDRAW: 'The hospital has been told you cannot come.' }[action], 'success');
    fetchDonorInvitations();
    fetchDonorProfile();
  } catch (err) { showToast(err.message, 'error'); }
}

async function sendLocation(payload, quiet = false) {
  try {
    await api('/donor/location', { method: 'POST', body: payload });
    if (!quiet) showToast('Location shared.', 'success');
    fetchDonorProfile();
    fetchDonorInvitations();
  } catch (err) { if (!quiet) showToast(err.message, 'error'); }
}

async function handleShareLocation() {
  try {
    showToast('Getting your GPS location…', 'info');
    const pos = await getPosition();
    if (pos.coords.accuracy > 250) {
      showToast(`GPS accuracy is too low (${Math.round(pos.coords.accuracy)} m). Try again outdoors or pick your location on the map.`, 'error');
      return;
    }
    sendLocation(positionPayload(pos));
  } catch (err) { showToast(`${err.message}. You can pick your location on the map instead.`, 'error'); }
}

function startLiveLocation() {
  if (!navigator.geolocation) { showToast('This browser cannot share location', 'error'); $('donor-live-toggle').checked = false; return; }
  stopLiveLocation();
  STATE.liveWatchId = navigator.geolocation.watchPosition((pos) => {
    if (pos.coords.accuracy > 250 || Date.now() - STATE.lastLiveSentAt < 60000) return;
    STATE.lastLiveSentAt = Date.now();
    sendLocation(positionPayload(pos), true);
  }, (err) => { showToast(err.code === 1 ? 'Location permission was denied' : 'Live GPS unavailable', 'error'); $('donor-live-toggle').checked = false; stopLiveLocation(); },
  { enableHighAccuracy: true, maximumAge: 30000 });
  showToast('Live location on while this page is open.', 'success');
}

function stopLiveLocation() {
  if (STATE.liveWatchId != null) navigator.geolocation.clearWatch(STATE.liveWatchId);
  STATE.liveWatchId = null;
  STATE.lastLiveSentAt = 0;
}

async function handleRevokeLocation() {
  if (!confirm('Stop sharing your location? You will not receive emergency alerts until you share it again.')) return;
  stopLiveLocation();
  $('donor-live-toggle').checked = false;
  try {
    await api('/donor/location', { method: 'POST', body: { consent: false } });
    showToast('Location removed. You will not receive alerts.', 'info');
    fetchDonorProfile();
    fetchDonorInvitations();
  } catch (err) { showToast(err.message, 'error'); }
}

async function handleAvailabilityToggle(event) {
  try {
    await api('/donor/profile', { method: 'POST', body: { available: event.target.checked } });
    showToast(event.target.checked ? 'You will receive emergency alerts.' : 'Alerts paused.', 'info');
  } catch (err) {
    event.target.checked = !event.target.checked;
    showToast(err.message, 'error');
  }
  fetchDonorProfile();
  fetchDonorInvitations();
}

async function handleDonorEdit(event) {
  event.preventDefault();
  try {
    await api('/donor/profile', { method: 'POST', body: { phone: $('edit-phone').value.trim(), email: $('edit-email').value.trim() } });
    const count = $('edit-prior-count').value, last = $('edit-last-date').value, first = $('edit-first-date').value;
    if (count !== '' || last) {
      const history = { prior_donation_count: parseInt(count || '0', 10) };
      if (last) history.last_donation_date = last;
      if (first) history.first_donation_date = first;
      await api('/donor/history', { method: 'POST', body: history });
    }
    showToast('Profile saved.', 'success');
    fetchDonorProfile();
  } catch (err) { showToast(err.message, 'error'); }
}

function openPickModal() {
  $('modal-pick').style.display = 'flex';
  const p = STATE.donorProfile;
  const map = ensureMap('pick-map', p?.latitude != null ? [p.latitude, p.longitude] : DEFAULT_CENTER, p?.latitude != null ? 14 : 5);
  if (map && !map._clickBound) {
    map.on('click', (e) => setPickLocation(e.latlng.lat, e.latlng.lng));
    map._clickBound = true;
  }
}

function setPickLocation(lat, lon, label) {
  STATE.pickLocation = { lat, lon };
  setSinglePin('pick-map', lat, lon, donorIcon(), 15);
  $('pick-label').textContent = `${label ? `${label} · ` : ''}${lat.toFixed(5)}, ${lon.toFixed(5)}`;
}

async function savePickedLocation() {
  if (!STATE.pickLocation) { showToast('Click the map to place your location', 'error'); return; }
  await sendLocation({ consent: true, source: 'manual', latitude: STATE.pickLocation.lat, longitude: STATE.pickLocation.lon, captured_at: new Date().toISOString() });
  closeModal('modal-pick');
}

async function enableAlerts() {
  STATE.alertsEnabled = true;
  beep();
  if ('Notification' in window && Notification.permission === 'default') await Notification.requestPermission();
  showToast('Alert sound and notifications enabled for this tab.', 'success');
}

// ---------------------------------------------------------------- registration location

function setRegLocation(lat, lon, source, accuracy, label) {
  STATE.regLocation = { latitude: lat, longitude: lon, source, ...(accuracy != null ? { accuracy_meters: accuracy } : {}) };
  setSinglePin('reg-map', lat, lon, donorIcon(), 15);
  $('reg-location-label').textContent = `${label ? `${label} · ` : ''}${lat.toFixed(5)}, ${lon.toFixed(5)} (${source === 'device' ? 'GPS' : 'picked on map'})`;
}

function toggleRegLocation() {
  const on = $('reg-location-consent').checked;
  $('reg-location-tools').style.display = on ? 'block' : 'none';
  if (!on) { STATE.regLocation = null; return; }
  const map = ensureMap('reg-map');
  if (map && !map._clickBound) {
    map.on('click', (e) => setRegLocation(e.latlng.lat, e.latlng.lng, 'manual'));
    map._clickBound = true;
  }
}

// ---------------------------------------------------------------- blood bank directory

function facilityAddress(f) {
  const clean = (v) => (v && v !== 'NA' ? v : '');
  return [clean(f.address), clean(f.city) || clean(f.district), clean(f.state), clean(f.pincode)].filter(Boolean).join(', ');
}

async function loadDirectory() {
  const d = STATE.directory;
  const params = new URLSearchParams({ page: d.page, page_size: d.pageSize });
  const search = $('directory-search-input').value.trim();
  if (search) params.set('search', search);
  if ($('directory-state').value) params.set('state', $('directory-state').value);
  if ($('directory-has-coords').checked) params.set('has_coordinates', 'true');
  try {
    const data = await api(`/facilities?${params}`);
    if (!d.statesLoaded) {
      $('directory-state').innerHTML = '<option value="">All states</option>' + data.states.map((s) => `<option>${esc(s)}</option>`).join('');
      d.statesLoaded = true;
    }
    d.hasNext = data.has_next;
    d.rows = data.facilities;
    const start = data.total ? (d.page - 1) * d.pageSize + 1 : 0;
    $('directory-summary').textContent = `Showing ${start}–${start + data.facilities.length - 1 < start ? 0 : start + data.facilities.length - 1} of ${data.total.toLocaleString()} blood banks (${data.with_coordinates.toLocaleString()} with a map location). ${data.notice}.`;
    $('directory-page-label').textContent = `Page ${d.page} of ${Math.max(1, Math.ceil(data.total / d.pageSize))}`;
    $('directory-prev').disabled = d.page <= 1;
    $('directory-next').disabled = !d.hasNext;
    renderDirectory(data.facilities);
  } catch (err) {
    $('directory-tbody').innerHTML = `<tr><td colspan="7" class="text-center text-muted py-4">${esc(err.message)}</td></tr>`;
  }
}

function renderDirectory(rows) {
  const clean = (v) => (v && v !== 'NA' ? esc(v) : '<span class="text-muted">—</span>');
  $('directory-tbody').innerHTML = rows.length ? rows.map((f, i) => {
    const located = f.latitude != null && f.longitude != null;
    const place = [f.district, f.city].filter((v) => v && v !== 'NA');
    return `<tr>
      <td><strong>${esc(f.name)}</strong><div class="tiny-note">${esc(f.facility_id)}</div></td>
      <td>${clean(f.address)}</td>
      <td>${place.length ? esc([...new Set(place)].join(' / ')) : '<span class="text-muted">—</span>'}</td>
      <td>${esc(f.state)}</td>
      <td>${clean(f.pincode)}</td>
      <td>${located ? `<button class="btn btn-sm btn-ghost" data-action="facility-map" data-index="${i}">📍 ${f.latitude.toFixed(3)}, ${f.longitude.toFixed(3)}</button>` : '<span class="tiny-note">no map location</span>'}</td>
      <td>${located ? `<button class="btn btn-sm btn-primary" data-action="facility-use" data-index="${i}">Use for request</button>` : ''}</td>
    </tr>`;
  }).join('') : '<tr><td colspan="7" class="text-center text-muted py-4">No blood banks match these filters.</td></tr>';

  const map = ensureMap('directory-map');
  if (!map) return;
  const layer = STATE.layers['directory-map'];
  layer.clearLayers();
  const points = [];
  rows.forEach((f) => {
    if (f.latitude == null) return;
    points.push([f.latitude, f.longitude]);
    L.marker([f.latitude, f.longitude], { icon: hospitalIcon() })
      .bindPopup(`<b>${esc(f.name)}</b><br>${esc(facilityAddress(f))}<br><span style="font-size:11px">Position from 2015 directory, unverified</span>`).addTo(layer);
  });
  if (points.length) map.fitBounds(points, { padding: [30, 30], maxZoom: 13 });
}

function useFacility(f) {
  switchTab('dashboard');
  openNewRequestModal({ latitude: f.latitude, longitude: f.longitude, name: f.name, address: facilityAddress(f), facility_id: f.facility_id });
  showToast('Directory entry selected. Please confirm its name and map position.', 'info');
}

// ---------------------------------------------------------------- health

async function checkHealth() {
  try {
    const h = await api('/health');
    STATE.health = h;
    const routing = h.routing_provider === 'osrm'
      ? (h.routing_available ? 'OSRM road routing' : 'OSRM unreachable — approximate ETAs')
      : h.routing_provider === 'google' ? 'Google routing' : 'Offline ETA estimates';
    $('system-status-text').textContent = `Online · ${routing}`;
    $('system-status-pill').className = `status-pill ${h.routing_provider === 'osrm' && !h.routing_available ? 'status-warning' : 'status-healthy'}`;
    $('telemetry-health').textContent = `Healthy (checked ${fmtTime(h.timestamp)})`;
    $('telemetry-provider').textContent = h.routing_provider === 'osrm'
      ? `OSRM at ${h.routing_base_url} — ${h.routing_available ? 'reachable' : 'unreachable, using approximate estimates'}`
      : h.routing_provider === 'google' ? 'Google Maps Routes API' : 'Offline synthetic estimates (set ROUTING_PROVIDER=osrm for real roads)';
    $('telemetry-webhook').textContent = h.external_notifications_configured ? 'Configured' : 'Not configured (in-app alerts only)';
    $('telemetry-facilities').textContent = `${h.facility_records.toLocaleString()} (${h.facilities_with_coordinates.toLocaleString()} with map location)`;
    $('telemetry-donors').textContent = h.registered_donors;
    $('kpi-registered-donors').textContent = h.registered_donors;
    $('hero-facilities').textContent = h.facility_records.toLocaleString();
    $('hero-donors').textContent = h.registered_donors;
    renderQuickAccounts(h);
  } catch (err) {
    $('system-status-text').textContent = 'Offline';
    $('system-status-pill').className = 'status-pill status-error';
    $('telemetry-health').textContent = 'Cannot reach the backend';
  }
}

function renderQuickAccounts(h) {
  const chips = [];
  if (h.staff_demo_login) chips.push(['staff_001', '👨‍⚕️ Doctor (staff_001)']);
  if (h.demo_accounts_enabled) chips.push(['donor_001', '🩸 Demo donor O+'], ['donor_002', '🩸 Demo donor A+']);
  $('quick-accounts').style.display = chips.length ? 'block' : 'none';
  $('quick-buttons').innerHTML = chips.map(([u, label]) => `<button type="button" class="btn-chip" data-action="quick-login" data-user="${u}">${label}</button>`).join('');
}

// ---------------------------------------------------------------- wiring

function wire() {
  $('form-login').addEventListener('submit', handleLoginSubmit);
  $('form-register').addEventListener('submit', handleRegisterSubmit);
  $('btn-logout').addEventListener('click', () => handleLogout());
  document.querySelectorAll('[data-auth]').forEach((b) => b.addEventListener('click', () => setAuthMode(b.dataset.auth)));
  document.querySelectorAll('.nav-btn').forEach((b) => b.addEventListener('click', () => switchTab(b.dataset.tab)));
  document.querySelectorAll('[data-close]').forEach((b) => b.addEventListener('click', () => closeModal(b.dataset.close)));
  document.querySelectorAll('[data-view]').forEach((b) => b.addEventListener('click', () => switchDetailView(b.dataset.view)));

  $('reg-prior-count').addEventListener('input', (e) => { $('reg-first-date-group').style.display = +e.target.value > 1 ? 'block' : 'none'; });
  $('reg-location-consent').addEventListener('change', toggleRegLocation);
  $('reg-use-gps').addEventListener('click', async () => {
    try {
      const pos = await getPosition();
      if (pos.coords.accuracy > 250) { showToast(`GPS accuracy is ${Math.round(pos.coords.accuracy)} m — please pick on the map instead`, 'error'); return; }
      setRegLocation(pos.coords.latitude, pos.coords.longitude, 'device', pos.coords.accuracy);
    } catch (err) { showToast(`${err.message}. Pick your location on the map instead.`, 'error'); }
  });
  $('reg-address-search').addEventListener('click', async () => {
    const q = $('reg-address').value.trim();
    if (!q) return;
    try { renderSearchResults('reg-address-results', await geocode(q), (r) => setRegLocation(r.lat, r.lon, 'manual', null, r.name)); }
    catch (err) { showToast(err.message, 'error'); }
  });

  $('btn-new-request').addEventListener('click', () => openNewRequestModal());
  $('form-new-request').addEventListener('submit', handleCreateRequestSubmit);
  $('req-bloodtype').addEventListener('change', updateCompatibleLabel);
  $('req-time-limit').addEventListener('change', (e) => { $('req-time-custom').style.display = e.target.value === 'custom' ? 'block' : 'none'; });
  $('req-facility-search-btn').addEventListener('click', searchFacilitiesForRequest);
  $('req-address-search-btn').addEventListener('click', searchAddressForRequest);
  $('req-facility-search').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); searchFacilitiesForRequest(); } });
  $('req-use-gps').addEventListener('click', async () => {
    try { const pos = await getPosition(); setRequestLocation(pos.coords.latitude, pos.coords.longitude); }
    catch (err) { showToast(err.message, 'error'); }
  });
  $('btn-cancel-request').addEventListener('click', cancelActiveRequest);

  $('btn-share-location').addEventListener('click', handleShareLocation);
  $('btn-revoke-location').addEventListener('click', handleRevokeLocation);
  $('btn-pick-location').addEventListener('click', openPickModal);
  $('pick-save').addEventListener('click', savePickedLocation);
  $('pick-address-search').addEventListener('click', async () => {
    const q = $('pick-address').value.trim();
    if (!q) return;
    try { renderSearchResults('pick-results', await geocode(q), (r) => setPickLocation(r.lat, r.lon, r.name)); }
    catch (err) { showToast(err.message, 'error'); }
  });
  $('donor-live-toggle').addEventListener('change', (e) => (e.target.checked ? startLiveLocation() : stopLiveLocation()));
  $('donor-available-toggle').addEventListener('change', handleAvailabilityToggle);
  $('form-donor-edit').addEventListener('submit', handleDonorEdit);
  $('btn-enable-alerts').addEventListener('click', enableAlerts);

  $('directory-filters').addEventListener('submit', (e) => { e.preventDefault(); STATE.directory.page = 1; loadDirectory(); });
  $('directory-state').addEventListener('change', () => { STATE.directory.page = 1; loadDirectory(); });
  $('directory-has-coords').addEventListener('change', () => { STATE.directory.page = 1; loadDirectory(); });
  $('directory-prev').addEventListener('click', () => { if (STATE.directory.page > 1) { STATE.directory.page--; loadDirectory(); } });
  $('directory-next').addEventListener('click', () => { if (STATE.directory.hasNext) { STATE.directory.page++; loadDirectory(); } });
  $('btn-refresh-health').addEventListener('click', checkHealth);

  // Buttons inside lists rendered from data use data-action (no inline handlers).
  document.addEventListener('click', (e) => {
    const el = e.target.closest('[data-action]');
    if (!el) return;
    const a = el.dataset.action;
    if (a === 'open-request') openRequestDetail(el.dataset.id);
    else if (a === 'route') openRouteModal(el.dataset.donor, el.dataset.request);
    else if (a === 'confirm-donation') confirmDonation(el.dataset.donor, el.dataset.request);
    else if (a === 'respond') respondInvitation(el.dataset.id, el.dataset.response);
    else if (a === 'quick-login') { setAuthMode('login'); $('login-username').value = el.dataset.user; $('login-password').value = ''; }
    else if (a === 'facility-use') useFacility(STATE.directory.rows[+el.dataset.index]);
    else if (a === 'facility-map') {
      const f = STATE.directory.rows[+el.dataset.index];
      STATE.maps['directory-map']?.setView([f.latitude, f.longitude], 15);
      $('directory-map').scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  });
}

document.addEventListener('DOMContentLoaded', () => {
  wire();
  renderSessionState();
  checkHealth();
  setInterval(tickCountdowns, 1000);
  setInterval(checkHealth, 30000);
});
