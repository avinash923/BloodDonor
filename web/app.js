/**
 * LifePulse AI — Emergency Blood Recommendation & Dynamic Recruitment
 * Full Live Production Client
 */

// Global State
const STATE = {
  apiBase: window.location.origin + '/api/v1',
  token: sessionStorage.getItem('token') || null,
  user: JSON.parse(sessionStorage.getItem('user') || 'null'),
  activeRequestId: null,
  recruitmentView: 'candidates', // 'candidates' or 'accepted'
  recruitmentTimer: null,
  donorPollingTimer: null,
  countdownInterval: null,
  mapInstance: null,
  facilitiesCache: [],
};

// Polyline Decoder for Google Routes / Encoded polyline format
function decodePolyline(encoded) {
  if (!encoded) return [];
  const points = [];
  let index = 0, len = encoded.length;
  let lat = 0, lng = 0;

  while (index < len) {
    let b, shift = 0, result = 0;
    do {
      b = encoded.charCodeAt(index++) - 63;
      result |= (b & 0x1f) << shift;
      shift += 5;
    } while (b >= 0x20);
    const dlat = ((result & 1) ? ~(result >> 1) : (result >> 1));
    lat += dlat;

    shift = 0;
    result = 0;
    do {
      b = encoded.charCodeAt(index++) - 63;
      result |= (b & 0x1f) << shift;
      shift += 5;
    } while (b >= 0x20);
    const dlng = ((result & 1) ? ~(result >> 1) : (result >> 1));
    lng += dlng;

    points.push([lat / 1e5, lng / 1e5]);
  }
  return points;
}

// REST Client Helper
async function api(path, options = {}) {
  const headers = { 'Content-Type': 'application/json' };
  if (STATE.token) {
    headers['Authorization'] = `Bearer ${STATE.token}`;
  }
  const config = {
    method: options.method || 'GET',
    headers: { ...headers, ...(options.headers || {}) },
  };
  if (options.body) {
    config.body = JSON.stringify(options.body);
  }

  try {
    const res = await fetch(`${STATE.apiBase}${path}`, config);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const errorMsg = data.error || `HTTP error ${res.status}`;
      throw new Error(errorMsg);
    }
    return data;
  } catch (err) {
    console.error(`API Error on ${path}:`, err);
    throw err;
  }
}

// UI Toasts
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <span>${message}</span>
    <button style="background:none; border:none; color:inherit; font-size:1.1rem; cursor:pointer;" onclick="this.parentElement.remove()">×</button>
  `;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
  }, 4500);
}

// Tab Switching
function switchTab(tab) {
  document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.portal-view').forEach(view => view.style.display = 'none');

  if (tab === 'dashboard') {
    document.getElementById('nav-btn-dashboard')?.classList.add('active');
    if (STATE.user && STATE.user.role === 'hospital_staff') {
      document.getElementById('view-staff').style.display = 'block';
      fetchStaffRequests();
    } else if (STATE.user && STATE.user.role === 'donor') {
      document.getElementById('view-donor').style.display = 'block';
      fetchDonorProfile();
      startDonorPolling();
    }
  } else if (tab === 'directory') {
    document.getElementById('nav-btn-directory')?.classList.add('active');
    document.getElementById('view-directory').style.display = 'block';
    loadDirectory('');
  } else if (tab === 'system') {
    document.getElementById('nav-btn-system')?.classList.add('active');
    document.getElementById('view-system').style.display = 'block';
    checkHealth();
  }
}

// Auth Tab Switching (Sign in vs Register)
function setAuthMode(mode) {
  const loginTab = document.getElementById('auth-tab-login');
  const regTab = document.getElementById('auth-tab-register');
  const loginForm = document.getElementById('form-login');
  const regForm = document.getElementById('form-register');

  if (mode === 'login') {
    loginTab.classList.add('active');
    regTab.classList.remove('active');
    loginForm.style.display = 'block';
    regForm.style.display = 'none';
  } else {
    loginTab.classList.remove('active');
    regTab.classList.add('active');
    loginForm.style.display = 'none';
    regForm.style.display = 'block';
  }
}

function quickFillLogin(username, password) {
  setAuthMode('login');
  document.getElementById('login-username').value = username;
  document.getElementById('login-password').value = password;
}

// Handle Login
async function handleLoginSubmit(event) {
  event.preventDefault();
  const username = document.getElementById('login-username').value.trim();
  const password = document.getElementById('login-password').value;

  try {
    const payload = { username };
    if (password) payload.password = password;

    const data = await api('/auth/login', { method: 'POST', body: payload });
    STATE.token = data.token;
    STATE.user = { user_id: data.user_id, name: data.name, role: data.role };
    sessionStorage.setItem('token', STATE.token);
    sessionStorage.setItem('user', JSON.stringify(STATE.user));

    showToast(`Welcome back, ${data.name}!`, 'success');
    renderSessionState();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

// Handle Registration
async function handleRegisterSubmit(event) {
  event.preventDefault();
  const username = document.getElementById('reg-username').value.trim();
  const blood_type = document.getElementById('reg-bloodtype').value;
  const name = document.getElementById('reg-fullname').value.trim();
  const password = document.getElementById('reg-password').value;
  const prior = parseInt(document.getElementById('reg-prior-count').value || '0', 10);
  const monthsLast = document.getElementById('reg-months-last').value;
  const monthsFirst = document.getElementById('reg-months-first').value;

  const payload = {
    username,
    name,
    blood_type,
    password,
    prior_donation_count: prior,
  };
  if (monthsLast) payload.months_since_last_donation = parseFloat(monthsLast);
  if (monthsFirst) payload.months_since_first_donation = parseFloat(monthsFirst);

  try {
    const data = await api('/auth/register', { method: 'POST', body: payload });
    STATE.token = data.token;
    STATE.user = { user_id: data.user_id, name: data.name, role: data.role };
    sessionStorage.setItem('token', STATE.token);
    sessionStorage.setItem('user', JSON.stringify(STATE.user));

    showToast(`Registered successfully as ${data.name}!`, 'success');
    renderSessionState();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

// Logout
function handleLogout() {
  STATE.token = null;
  STATE.user = null;
  sessionStorage.removeItem('token');
  sessionStorage.removeItem('user');
  clearInterval(STATE.donorPollingTimer);
  clearInterval(STATE.recruitmentTimer);
  clearInterval(STATE.countdownInterval);

  document.getElementById('main-nav').style.display = 'none';
  document.getElementById('user-badge').style.display = 'none';
  document.querySelectorAll('.portal-view').forEach(view => view.style.display = 'none');
  document.getElementById('view-auth').style.display = 'block';
  showToast('Logged out of system session.', 'info');
}

// Render Session State
function renderSessionState() {
  if (!STATE.token || !STATE.user) {
    document.getElementById('main-nav').style.display = 'none';
    document.getElementById('user-badge').style.display = 'none';
    document.getElementById('view-auth').style.display = 'block';
    document.querySelectorAll('.portal-view').forEach(view => view.style.display = 'none');
    return;
  }

  document.getElementById('view-auth').style.display = 'none';
  document.getElementById('main-nav').style.display = 'flex';
  document.getElementById('user-badge').style.display = 'flex';
  document.getElementById('current-user-role').textContent = STATE.user.role === 'hospital_staff' ? 'STAFF' : 'DONOR';
  document.getElementById('current-user-name').textContent = STATE.user.name;

  switchTab('dashboard');
}

// ==========================================================================
// HOSPITAL STAFF DISPATCH DESK LOGIC
// ==========================================================================

async function fetchStaffRequests() {
  try {
    const data = await api('/emergency/requests');
    const requests = data.requests || [];
    renderStaffRequests(requests);

    // Update KPI counters
    const active = requests.filter(r => r.status === 'ACTIVE').length;
    const fulfilled = requests.reduce((sum, r) => sum + (r.units_fulfilled || 0), 0);
    document.getElementById('kpi-active-requests').textContent = active;
    document.getElementById('kpi-fulfilled-units').textContent = fulfilled;
  } catch (err) {
    console.error('Failed to fetch requests:', err);
  }
}

function renderStaffRequests(requests) {
  const tbody = document.getElementById('emergency-requests-body');
  if (!tbody) return;

  if (requests.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="8" class="text-center text-muted py-4">No active emergency requests. Click "Create Emergency Request" to begin.</td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = requests.map(req => {
    const isFulfilled = req.status === 'FULFILLED';
    const statusClass = req.status === 'ACTIVE' ? 'badge-critical' : isFulfilled ? 'badge-success' : 'badge-info';
    const urgencyClass = req.urgency === 'CRITICAL' ? 'badge-critical' : 'badge-warning';
    const pct = Math.min(100, Math.round(((req.units_fulfilled || 0) / req.units_required) * 100));

    return `
      <tr>
        <td><strong>${req.request_id}</strong></td>
        <td>${req.hospital_name || 'Hospital Facility'}</td>
        <td><span class="blood-tag">${req.blood_type_required}</span></td>
        <td>${req.units_fulfilled || 0} / ${req.units_required} Units</td>
        <td><span class="badge-pill ${urgencyClass}">${req.urgency}</span></td>
        <td><span class="badge-pill ${statusClass}">${req.status}</span></td>
        <td>
          <div style="width: 100px; display:inline-block; vertical-align:middle; margin-right:8px;">
            <div class="progress-bar-bg" style="margin-bottom:0;">
              <div class="progress-bar-fill" style="width: ${pct}%;"></div>
            </div>
          </div>
          <span style="font-size:0.75rem; color:var(--text-dim);">${pct}%</span>
        </td>
        <td>
          <button class="btn btn-sm btn-primary" onclick="openRecruitmentModal('${req.request_id}')">
            ⚡ Wave Recruitment
          </button>
        </td>
      </tr>
    `;
  }).join('');
}

// Modal: Create Request
function openNewRequestModal() {
  document.getElementById('modal-request').style.display = 'flex';
}

function closeModal(id) {
  document.getElementById(id).style.display = 'none';
  if (id === 'modal-recruitment') {
    clearInterval(STATE.recruitmentTimer);
    clearInterval(STATE.countdownInterval);
  }
}

async function handleCreateRequestSubmit(event) {
  event.preventDefault();
  const blood_type_required = document.getElementById('req-bloodtype').value;
  const units_required = parseInt(document.getElementById('req-units').value, 10);
  const urgency = document.getElementById('req-urgency').value;
  const hospital_name = document.getElementById('req-hospital-name').value;
  const hospital_latitude = parseFloat(document.getElementById('req-hospital-lat').value);
  const hospital_longitude = parseFloat(document.getElementById('req-hospital-lon').value);

  const payload = {
    blood_type_required,
    units_required,
    urgency,
    hospital_name,
    hospital_latitude,
    hospital_longitude,
  };

  try {
    const data = await api('/emergency/requests', { method: 'POST', body: payload });
    showToast(`Emergency request ${data.request_id} created! Wave 1 dispatched.`, 'success');
    closeModal('modal-request');
    fetchStaffRequests();
    openRecruitmentModal(data.request_id);
  } catch (err) {
    showToast(err.message, 'error');
  }
}

// Recruitment Wave Monitor
async function openRecruitmentModal(requestId) {
  STATE.activeRequestId = requestId;
  document.getElementById('modal-recruitment').style.display = 'flex';
  await refreshRecruitmentModal();

  // Poll recruitment wave every 4 seconds while modal is open
  clearInterval(STATE.recruitmentTimer);
  STATE.recruitmentTimer = setInterval(refreshRecruitmentModal, 4000);
}

function switchRecruitmentView(view) {
  STATE.recruitmentView = view;
  document.getElementById('tab-candidates').classList.toggle('active', view === 'candidates');
  document.getElementById('tab-accepted').classList.toggle('active', view === 'accepted');
  refreshRecruitmentModal();
}

async function refreshRecruitmentModal() {
  if (!STATE.activeRequestId) return;
  try {
    const data = await api(`/emergency/recommendations?request_id=${STATE.activeRequestId}`);
    renderRecruitmentModal(data);
  } catch (err) {
    console.error('Failed to refresh recruitment wave:', err);
  }
}

function renderRecruitmentModal(data) {
  const req = data.emergency_request;
  const summary = data.recruitment_summary || {};
  const candidates = data.candidates || [];
  const accepted = data.accepted_donors || [];

  document.getElementById('recruitment-modal-title').textContent = 
    `Dynamic Recruitment: ${req.blood_type_required} (${req.units_fulfilled || 0}/${req.units_required} Units)`;
  document.getElementById('recruitment-modal-subtitle').textContent = 
    `Facility: ${req.hospital_name} | Urgency: ${req.urgency}`;

  // Update Wave Banner
  document.getElementById('wave-current-batch').textContent = `Wave #${summary.current_batch || 1}`;
  document.getElementById('wave-status-text').textContent = summary.status || 'ACTIVE';
  document.getElementById('wave-reservation-count').textContent = 
    `${req.units_fulfilled || 0} Fulfilled (${summary.accepted_count || 0} Accepted, ${summary.pending_count || 0} Pending)`;

  // Wave Deadline Timer
  const deadlineStr = summary.response_deadline;
  if (deadlineStr) {
    const deadline = new Date(deadlineStr).getTime();
    updateCountdown(deadline);
  } else {
    document.getElementById('wave-deadline-timer').textContent = 'Fulfilled / Inactive';
  }

  // Render list based on current subtab
  const list = STATE.recruitmentView === 'candidates' ? candidates : accepted;
  const container = document.getElementById('candidates-container');

  if (list.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <span class="empty-icon">👥</span>
        <p>No ${STATE.recruitmentView === 'candidates' ? 'candidates in current wave' : 'accepted donors yet'}.</p>
      </div>
    `;
    return;
  }

  container.innerHTML = list.map(c => {
    const status = c.invitation_status || (c.is_accepted ? 'ACCEPTED' : 'CANDIDATE');
    const isAccepted = status === 'ACCEPTED';
    const statusBadge = isAccepted ? 'badge-success' : status === 'INVITED' ? 'badge-warning pulse-badge' : 'badge-info';
    const udi = typeof c.udi_score === 'number' ? c.udi_score.toFixed(3) : '0.000';
    const eta = typeof c.travel_time_minutes === 'number' ? `${c.travel_time_minutes.toFixed(1)}m` : 'Unreachable';
    const propensity = typeof c.donation_propensity === 'number' ? `${Math.round(c.donation_propensity * 100)}%` : 'N/A';

    return `
      <div class="candidate-card">
        <div class="candidate-primary">
          <span class="blood-tag">${c.blood_type}</span>
          <div class="candidate-info">
            <h4>${c.name} <span style="font-size:0.75rem; color:var(--text-dim);">(${c.donor_id})</span></h4>
            <div class="candidate-metrics mt-1">
              <span>UDI Score: <strong>${udi}</strong></span>
              <span>Road ETA: <strong>${eta}</strong></span>
              <span>UCI Return Propensity: <strong>${propensity}</strong></span>
            </div>
          </div>
        </div>

        <div style="display:flex; align-items:center; gap:12px;">
          <span class="badge-pill ${statusBadge}">${status}</span>
          ${isAccepted ? `
            <button class="btn btn-sm btn-primary" onclick="confirmDonation('${c.donor_id}', '${req.request_id}')">
              🩸 Confirm Donation
            </button>
          ` : ''}
          <button class="btn btn-sm btn-ghost" onclick="openRouteModal('${c.donor_id}', '${req.request_id}')">
            🗺️ View Route
          </button>
        </div>
      </div>
    `;
  }).join('');
}

function updateCountdown(targetTimeMs) {
  clearInterval(STATE.countdownInterval);
  const timerElem = document.getElementById('wave-deadline-timer');

  const tick = () => {
    const remaining = Math.max(0, Math.round((targetTimeMs - Date.now()) / 1000));
    if (remaining <= 0) {
      timerElem.textContent = 'Advancing wave...';
      clearInterval(STATE.countdownInterval);
    } else {
      timerElem.textContent = `${remaining}s`;
    }
  };
  tick();
  STATE.countdownInterval = setInterval(tick, 1000);
}

// Staff: Confirm Donation Completed
async function confirmDonation(donorId, requestId) {
  try {
    const data = await api('/emergency/donation/complete', {
      method: 'POST',
      body: { donor_id: donorId, request_id: requestId }
    });
    showToast(`Donation confirmed! 90-day clinical deferral applied to ${donorId}.`, 'success');
    refreshRecruitmentModal();
    fetchStaffRequests();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

// ==========================================================================
// DONOR RESCUE PORTAL LOGIC
// ==========================================================================

async function fetchDonorProfile() {
  try {
    const data = await api('/donor/profile');
    renderDonorProfile(data);
  } catch (err) {
    console.error('Failed to fetch donor profile:', err);
  }
}

function renderDonorProfile(profile) {
  document.getElementById('donor-profile-name').textContent = profile.name;
  document.getElementById('donor-profile-id').textContent = profile.user_id;
  document.getElementById('donor-profile-bloodtype').textContent = profile.blood_type;
  document.getElementById('donor-profile-donations').textContent = `${profile.completed_donations_count || 0} units`;

  // Eligibility Status
  const eligBadge = document.getElementById('donor-eligibility-badge');
  const deferralReason = document.getElementById('donor-deferral-reason');
  const nextEligRow = document.getElementById('donor-next-eligible-row');
  const nextEligDate = document.getElementById('donor-next-eligible-date');

  if (profile.is_temporally_eligible) {
    eligBadge.className = 'badge-pill badge-success';
    eligBadge.textContent = 'Temporally Eligible';
    deferralReason.textContent = 'Eligible for emergency call';
    nextEligRow.style.display = 'none';
  } else {
    eligBadge.className = 'badge-pill badge-critical';
    eligBadge.textContent = 'Clinical Deferral';
    deferralReason.textContent = profile.eligibility_reason || 'Mandatory safety interval active';
    if (profile.next_eligible_at) {
      nextEligRow.style.display = 'flex';
      nextEligDate.textContent = new Date(profile.next_eligible_at).toLocaleDateString();
    }
  }

  // UCI XGBoost Propensity
  const score = profile.donation_propensity;
  const scoreElem = document.getElementById('donor-propensity-score');
  const barElem = document.getElementById('donor-propensity-bar');
  if (typeof score === 'number') {
    const pct = Math.round(score * 100);
    scoreElem.textContent = `${pct}%`;
    barElem.style.width = `${pct}%`;
  } else {
    scoreElem.textContent = 'No Prior History';
    barElem.style.width = '0%';
  }

  // Location Snapshot
  const locBadge = document.getElementById('donor-location-status-badge');
  document.getElementById('donor-current-lat').textContent = profile.latitude ? profile.latitude.toFixed(4) : '—';
  document.getElementById('donor-current-lon').textContent = profile.longitude ? profile.longitude.toFixed(4) : '—';
  document.getElementById('donor-current-accuracy').textContent = profile.location_accuracy_meters ? `${Math.round(profile.location_accuracy_meters)}m` : '—';

  if (profile.location_ready) {
    locBadge.className = 'badge-pill badge-success';
    locBadge.textContent = 'Live & Ready (<15m)';
  } else {
    locBadge.className = 'badge-pill badge-warning';
    locBadge.textContent = profile.location_consent ? 'Stale Location (>15m)' : 'Not Shared';
  }
}

function startDonorPolling() {
  fetchDonorInvitations();
  clearInterval(STATE.donorPollingTimer);
  STATE.donorPollingTimer = setInterval(fetchDonorInvitations, 4000);
}

async function fetchDonorInvitations() {
  try {
    const data = await api('/donor/invitations');
    renderDonorInvitations(data.invitations || []);
  } catch (err) {
    console.error('Failed to poll invitations:', err);
  }
}

function renderDonorInvitations(invitations) {
  const container = document.getElementById('donor-invitations-container');
  const badge = document.getElementById('invitation-count-badge');
  if (!container) return;

  const pending = invitations.filter(i => i.status === 'PENDING');
  badge.textContent = `${pending.length} Pending`;

  if (invitations.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <span class="empty-icon">🔔</span>
        <p>No active emergency invitations right now. Keep your location updated to stay available for local emergency waves.</p>
      </div>
    `;
    return;
  }

  container.innerHTML = invitations.map(inv => {
    const isPending = inv.status === 'PENDING';
    const isAccepted = inv.status === 'ACCEPTED';
    const statusClass = isPending ? 'badge-warning pulse-badge' : isAccepted ? 'badge-success' : 'badge-info';

    return `
      <div class="invitation-card">
        <div class="invitation-body">
          <div class="invitation-title">
            <span class="blood-tag">${inv.blood_type_required}</span>
            <span>Emergency Alert at ${inv.hospital_name}</span>
            <span class="badge-pill badge-critical">${inv.urgency}</span>
            <span class="badge-pill ${statusClass}">${inv.status}</span>
          </div>
          <div class="invitation-details">
            <span>Invitation ID: <strong>${inv.invitation_id}</strong></span>
            <span>Hospital Coordinates: <strong>${inv.hospital_lat.toFixed(4)}, ${inv.hospital_lon.toFixed(4)}</strong></span>
            <span>Time Sent: <strong>${new Date(inv.sent_at).toLocaleTimeString()}</strong></span>
          </div>
        </div>

        <div class="invitation-actions">
          ${isPending ? `
            <button class="btn btn-primary" onclick="respondInvitation('${inv.invitation_id}', 'ACCEPT')">
              ✅ Accept Emergency Call
            </button>
            <button class="btn btn-ghost" onclick="respondInvitation('${inv.invitation_id}', 'DECLINE')">
              ❌ Decline
            </button>
          ` : ''}
          <button class="btn btn-ghost" onclick="openRouteModal('${inv.donor_id}', '${inv.request_id}')">
            🗺️ View Route
          </button>
        </div>
      </div>
    `;
  }).join('');
}

async function respondInvitation(invitationId, action) {
  try {
    await api('/donor/invitations/respond', {
      method: 'POST',
      body: { invitation_id: invitationId, action }
    });
    showToast(`Emergency invitation ${action.toLowerCase()}ed!`, 'success');
    fetchDonorInvitations();
    fetchDonorProfile();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

// Live GPS Sharing
function handleShareLocation() {
  if (!navigator.geolocation) {
    showToast('Geolocation is not supported by your browser.', 'error');
    return;
  }

  showToast('Acquiring high-accuracy foreground GPS fix...', 'info');
  navigator.geolocation.getCurrentPosition(
    async (pos) => {
      const payload = {
        consent: true,
        latitude: pos.coords.latitude,
        longitude: pos.coords.longitude,
        accuracy_meters: Math.min(250, pos.coords.accuracy || 25),
        captured_at: new Date(pos.timestamp).toISOString(),
      };
      try {
        await api('/donor/location', { method: 'POST', body: payload });
        showToast('Live GPS coordinates verified & shared with hospital desk!', 'success');
        fetchDonorProfile();
      } catch (err) {
        showToast(err.message, 'error');
      }
    },
    (err) => {
      // Fallback for development if device blocks GPS
      console.warn('Geolocation blocked, using city center fallback:', err);
      const fallbackPayload = {
        consent: true,
        latitude: 12.9716 + (Math.random() - 0.5) * 0.04,
        longitude: 77.5946 + (Math.random() - 0.5) * 0.04,
        accuracy_meters: 35.0,
        captured_at: new Date().toISOString(),
      };
      api('/donor/location', { method: 'POST', body: fallbackPayload })
        .then(() => {
          showToast('Foreground location simulation updated (accuracy: 35m).', 'success');
          fetchDonorProfile();
        })
        .catch(e => showToast(e.message, 'error'));
    },
    { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
  );
}

async function handleRevokeLocation() {
  try {
    await api('/donor/location', { method: 'POST', body: { consent: false } });
    showToast('Saved coordinates revoked. Donor marked unavailable.', 'info');
    fetchDonorProfile();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

// ==========================================================================
// INTERACTIVE ROUTE MAP MODAL
// ==========================================================================

async function openRouteModal(donorId, requestId) {
  document.getElementById('modal-route').style.display = 'flex';

  try {
    const data = await api(`/emergency/routes?donor_id=${donorId}&request_id=${requestId}`);
    document.getElementById('route-distance-val').textContent = `${data.distance_km.toFixed(1)} km`;
    document.getElementById('route-duration-val').textContent = `${data.duration_minutes.toFixed(1)} min`;
    document.getElementById('route-provider-val').textContent = data.provider || 'Synthetic Engine';

    setTimeout(() => {
      renderLeafletMap(data);
    }, 150);
  } catch (err) {
    showToast(err.message, 'error');
  }
}

function renderLeafletMap(routeData) {
  const container = document.getElementById('route-map');
  if (!container || !window.L) return;

  if (STATE.mapInstance) {
    STATE.mapInstance.remove();
    STATE.mapInstance = null;
  }

  const origLat = routeData.origin_latitude;
  const origLon = routeData.origin_longitude;
  const destLat = routeData.destination_latitude;
  const destLon = routeData.destination_longitude;

  const map = L.map('route-map').setView([destLat, destLon], 13);
  STATE.mapInstance = map;

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '© OpenStreetMap contributors'
  }).addTo(map);

  // Donor Marker
  L.marker([origLat, origLon], { title: 'Donor Location' })
    .addTo(map)
    .bindPopup('<b>Donor Location</b><br>Responding Candidate')
    .openPopup();

  // Hospital Marker
  const hospitalIcon = L.divIcon({
    className: 'custom-hosp-icon',
    html: `<div style="background:#E11D48; color:white; border-radius:50%; width:28px; height:28px; display:flex; align-items:center; justify-content:center; font-weight:bold; border:2px solid white; box-shadow:0 0 10px rgba(225,29,72,0.6)">🏥</div>`,
    iconSize: [28, 28],
    iconAnchor: [14, 14]
  });

  L.marker([destLat, destLon], { icon: hospitalIcon })
    .addTo(map)
    .bindPopup('<b>Emergency Destination</b><br>Hospital Blood Bank');

  // Route Polyline
  let points = [];
  if (routeData.encoded_polyline) {
    points = decodePolyline(routeData.encoded_polyline);
  }
  if (points.length < 2) {
    // Synthetic straight-line / grid fallback
    points = [[origLat, origLon], [destLat, destLon]];
  }

  const polyline = L.polyline(points, {
    color: '#E11D48',
    weight: 5,
    opacity: 0.85,
    dashArray: routeData.provider === 'synthetic' ? '8, 8' : null
  }).addTo(map);

  map.fitBounds(polyline.getBounds(), { padding: [50, 50] });
}

// ==========================================================================
// BLOOD BANK DIRECTORY EXPLORER
// ==========================================================================

async function loadDirectory(searchQuery = '') {
  const tbody = document.getElementById('directory-tbody');
  if (!tbody) return;

  try {
    const data = await api(`/facilities?search=${encodeURIComponent(searchQuery)}`);
    const facilities = data.facilities || [];
    STATE.facilitiesCache = facilities;

    if (facilities.length === 0) {
      tbody.innerHTML = `<tr><td colspan="5" class="text-center text-muted py-4">No matching blood bank facilities found.</td></tr>`;
      return;
    }

    tbody.innerHTML = facilities.slice(0, 50).map(f => {
      const coords = (f.latitude && f.longitude) ? `${f.latitude.toFixed(4)}, ${f.longitude.toFixed(4)}` : 'Coordinates missing';
      return `
        <tr>
          <td><strong>${f.name || 'Blood Bank'}</strong></td>
          <td>${f.city || f.district || '—'}</td>
          <td>${f.state || '—'}</td>
          <td><span style="font-size:0.8rem; color:var(--text-muted);">${coords}</span></td>
          <td>
            <button class="btn btn-sm btn-ghost" onclick="useFacilityInRequest('${f.name}', ${f.latitude}, ${f.longitude})">
              🏥 Select for Request
            </button>
          </td>
        </tr>
      `;
    }).join('');
  } catch (err) {
    console.error('Failed to load blood banks:', err);
  }
}

function handleDirectorySearch(query) {
  loadDirectory(query);
}

function useFacilityInRequest(name, lat, lon) {
  switchTab('dashboard');
  openNewRequestModal();
  document.getElementById('req-hospital-name').value = name;
  if (lat && lon) {
    document.getElementById('req-hospital-lat').value = lat;
    document.getElementById('req-hospital-lon').value = lon;
  }
  showToast(`Selected "${name}" as hospital destination.`, 'info');
}

// ==========================================================================
// SYSTEM TELEMETRY & HEALTH
// ==========================================================================

async function checkHealth() {
  try {
    const data = await api('/health');
    document.getElementById('telemetry-health').textContent = 'Healthy (Active REST Gateway)';
    document.getElementById('telemetry-provider').textContent = 
      data.routing_provider === 'google' ? 'Google Maps Platform Routes API v2' : 'Synthetic (Traffic-Aware Local Grid Engine)';
    document.getElementById('telemetry-webhook').textContent = 
      data.external_notifications_configured ? 'Active (Connected Webhook Outbox)' : 'Not Configured (In-App Only)';
    document.getElementById('telemetry-facilities').textContent = `${data.facility_records || 2947} Facilities Available`;
    document.getElementById('kpi-routing-engine').textContent = data.routing_provider === 'google' ? 'Google Routes' : 'Synthetic';
  } catch (err) {
    document.getElementById('telemetry-health').textContent = 'Error connecting to backend API';
  }
}

// Initial Boot
document.addEventListener('DOMContentLoaded', () => {
  renderSessionState();
  checkHealth();
});
