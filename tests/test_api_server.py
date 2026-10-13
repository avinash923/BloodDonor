"""HTTP regression tests for the local end-to-end demonstration."""
import json
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import pytest
import src.api.server as api


@pytest.fixture
def api_server(monkeypatch):
    monkeypatch.setattr(api, 'STATE', api.SystemState())
    server = api.run_api_server(port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_port}'
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def request(base, path, method='GET', token=None, body=None, raw=None):
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(base + '/api/v1/' + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def login(base, user):
    status, data = request(base, 'auth/login', 'POST', body={'username': user})
    assert status == 200
    return data['token']


def create(base, staff, **overrides):
    return request(base, 'emergency/requests', 'POST', staff, {
        'blood_type_required': 'O+', 'units_required': 1, 'urgency': 'CRITICAL', **overrides})


def accept(base, donor, invitation='INV_1001'):
    return request(base, 'donor/invitations/respond', 'POST', donor,
                   {'invitation_id': invitation, 'action': 'ACCEPT'})


def complete(base, staff, donor='donor_001', req='REQ_TRAUMA_101'):
    return request(base, 'emergency/donation/complete', 'POST', staff,
                   {'donor_id': donor, 'request_id': req})


def test_health_and_login(api_server):
    assert request(api_server, 'health')[1]['status'] == 'healthy'
    token = login(api_server, 'donor_001')
    assert token != 'donor_001'
    assert request(api_server, 'donor/profile', token=token)[1]['blood_type'] == 'O+'
    assert request(api_server, 'auth/login', 'POST', body={'username': 'unknown'})[0] == 401
    assert request(api_server, 'auth/login', 'POST', body={'username': []})[0] == 400


@pytest.mark.parametrize('path', ['emergency/requests', 'emergency/recommendations?request_id=REQ_TRAUMA_101',
                                  'emergency/routes?donor_id=donor_001&request_id=REQ_TRAUMA_101'])
def test_protected_resources(api_server, path):
    assert request(api_server, path)[0] == 401
    assert request(api_server, path, token='staff_001')[0] == 401


def test_role_and_route_ownership(api_server):
    donor = login(api_server, 'donor_001')
    assert request(api_server, 'emergency/requests', token=donor)[0] == 403
    assert request(api_server, 'emergency/routes?donor_id=donor_002&request_id=REQ_TRAUMA_101', token=donor)[0] == 403
    status, route = request(api_server, 'emergency/routes?donor_id=donor_001&request_id=REQ_TRAUMA_101', token=donor)
    assert status == 200
    assert route['duration_minutes'] > 0
    assert 'API_KEY' not in json.dumps(route)


def test_recommendations_only_matching_available_eligible_donors(api_server):
    staff = login(api_server, 'staff_001')
    path = 'emergency/recommendations?request_id=REQ_TRAUMA_101'
    recs = request(api_server, path, token=staff)[1]['recommendations']
    assert [r['donor_id'] for r in recs] == ['donor_001']
    assert recs[0]['rank'] == 1 and recs[0]['travel_time_minutes'] > 0
    api.STATE.users['donor_001']['availability'] = False
    assert create(api_server, staff)[1]['invited_count'] == 0
    # Already-invited donors stay visible to the doctor, without a live ETA.
    assert request(api_server, path, token=staff)[1]['recommendations'][0]['can_arrive_in_time'] is None


@pytest.mark.parametrize('overrides', [
    {'units_required': 0}, {'units_required': -1}, {'units_required': 101},
    {'units_required': True}, {'units_required': 'bad'}, {'units_required': 1.5},
    {'blood_type_required': 'UNKNOWN'}, {'urgency': 'bad'},
    {'time_limit_minutes': 5}, {'time_limit_minutes': 3000}, {'time_limit_minutes': '60'},
    {'time_limit_minutes': True}, {'search_radius_km': 0}, {'search_radius_km': 'far'},
    {'include_compatible': 'yes'}, {'contact_phone': 'call me'},
    {'latitude': 91}, {'longitude': -181}, {'latitude': 'nan'}, {'longitude': None},
])
def test_invalid_request_input(api_server, overrides):
    assert create(api_server, login(api_server, 'staff_001'), **overrides)[0] == 400
    assert len(api.STATE.emergencies) == 1


@pytest.mark.parametrize('raw', [b'[]', b'null', b'"text"', b'{', b'123'])
def test_malformed_json(api_server, raw):
    assert request(api_server, 'auth/login', 'POST', raw=raw)[0] == 400
    assert request(api_server, 'health')[0] == 200


def test_unique_requests_and_invitation_alert_counts(api_server):
    staff = login(api_server, 'staff_001')
    first = create(api_server, staff)[1]
    second = create(api_server, staff)[1]
    assert first['request_id'] != second['request_id']
    assert api.STATE.users['donor_001']['recent_alerts'] == 3
    assert len(api.STATE.invitations) == 3


def test_donation_requires_acceptance_and_is_not_repeatable(api_server):
    staff = login(api_server, 'staff_001')
    donor = login(api_server, 'donor_001')
    assert complete(api_server, staff)[0] == 409
    assert accept(api_server, donor)[0] == 200
    assert accept(api_server, donor)[0] == 409
    status, data = complete(api_server, staff)
    assert status == 200 and data['outcome_state'] == 'DONATION_COMPLETED'
    assert data['next_eligible_at']
    assert complete(api_server, staff)[0] == 409
    assert api.STATE.emergencies['REQ_TRAUMA_101']['units_fulfilled'] == 2
    profile = request(api_server, 'donor/profile', token=donor)[1]
    assert not profile['is_temporally_eligible'] and profile['completed_donations_count'] == 1
    rows = request(api_server, 'emergency/recommendations?request_id=REQ_TRAUMA_101', token=staff)[1]
    assert [r['invitation_status'] for r in rows['recommendations']] == ['COMPLETED']
    assert rows['accepted_donors'][0]['completed_at'] and rows['accepted_donors'][0]['total_donations'] == 7
    before = len(api.STATE.invitations)
    create(api_server, staff)
    assert len(api.STATE.invitations) == before


def test_request_fulfillment_and_restart(api_server, tmp_path, monkeypatch):
    filename = tmp_path / 'state.json'
    api.STATE.state_path = filename
    staff, donor = login(api_server, 'staff_001'), login(api_server, 'donor_001')
    created = create(api_server, staff)[1]
    invitation = api.STATE.invitations[-1]['invitation_id']
    assert accept(api_server, donor, invitation)[0] == 200
    assert complete(api_server, staff, req=created['request_id'])[0] == 200
    restored = api.SystemState(str(filename))
    assert restored.emergencies[created['request_id']]['status'] == 'FULFILLED'
    assert restored.emergencies[created['request_id']]['units_fulfilled'] == 1
    assert restored.invitations[0]['status'] == 'CANCELLED'
    assert not restored.donation_tracker.check_temporal_eligibility('donor_001', datetime.now(timezone.utc))[0]
    assert len(restored.donation_tracker.donor_states['donor_001'].completed_donations) == 1
    assert restored.sessions == {}


def test_invalid_donation_does_not_change_units(api_server):
    staff = login(api_server, 'staff_001')
    assert complete(api_server, staff, donor='donor_002')[0] == 409
    assert complete(api_server, staff, donor='staff_001')[0] == 404
    assert complete(api_server, staff, req='missing')[0] == 404
    assert api.STATE.emergencies['REQ_TRAUMA_101']['units_fulfilled'] == 1


@pytest.mark.parametrize('field', ['donor_id', 'request_id', 'invitation_id'])
def test_invalid_identifier_types(api_server, field):
    staff = login(api_server, 'staff_001')
    assert request(api_server, 'emergency/donation/complete', 'POST', staff, {field: []})[0] == 400


def test_web_assets_and_path_containment(tmp_path):
    root = tmp_path / 'web'
    root.mkdir()
    (root / 'index.html').write_text('<title>Demo</title>')
    (root / 'main.dart.js').write_text('console.log("demo")')
    (tmp_path / 'private.txt').write_text('private')
    server = api.run_api_server(port=0, web_root=str(root))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        for path, content in [('/', b'<title>Demo</title>'), ('/main.dart.js', b'console.log("demo")')]:
            with urllib.request.urlopen(base + path) as response:
                assert response.status == 200
                assert response.read() == content
        with pytest.raises(urllib.error.HTTPError) as failure:
            urllib.request.urlopen(base + '/%2e%2e/private.txt')
        assert failure.value.code == 403
        # An occupied port must fail rather than silently share another listener.
        with pytest.raises(OSError):
            api.run_api_server(port=server.server_port)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_broadcast_invites_every_matching_donor_at_once(api_server):
    api.STATE = api.SystemState(demo_donors=24)
    staff = login(api_server, 'staff_001')
    created = create(api_server, staff, units_required=3, time_limit_minutes=90)[1]
    assert created['time_limit_minutes'] == 90 and created['invited_count'] == 25
    path = 'emergency/recommendations?request_id=' + created['request_id']
    first = request(api_server, path, token=staff)[1]
    second = request(api_server, path + '&page=2', token=staff)[1]
    assert len(first['recommendations']) == len(second['recommendations']) == 10
    assert first['has_next'] and first['total'] == 25
    assert {r['donor_id'] for r in first['recommendations']}.isdisjoint(r['donor_id'] for r in second['recommendations'])
    assert first['recruitment']['contacted_count'] == 25 and first['recruitment']['pending_count'] == 25
    scores = [r['match_score'] for r in first['recommendations']]
    assert scores == sorted(scores, reverse=True)
    assert all(r['invitation_status'] == 'PENDING' and r['travel_time_minutes'] > 0 for r in first['recommendations'])
    assert request(api_server, path + '&page=0', token=staff)[0] == 400
    assert request(api_server, path + '&view=accepted', token=staff)[1]['recommendations'] == []
    # A different blood group receives nothing unless compatible groups are allowed.
    a_neg = create(api_server, staff, blood_type_required='AB+')[1]
    assert a_neg['invited_count'] == 0
    compatible = create(api_server, staff, blood_type_required='AB+', include_compatible=True)[1]
    assert compatible['invited_count'] == 26  # every O+ donor and the A+ donor


def test_accepted_donor_contact_and_history_visible_to_doctor(api_server):
    staff, donor = login(api_server, 'staff_001'), login(api_server, 'donor_001')
    path = 'emergency/recommendations?request_id=REQ_TRAUMA_101'
    before = request(api_server, path, token=staff)[1]['recommendations'][0]
    assert 'phone' not in before  # contact details only after acceptance
    assert accept(api_server, donor)[0] == 200
    data = request(api_server, path, token=staff)[1]
    accepted = data['accepted_donors']
    assert len(accepted) == 1
    row = accepted[0]
    assert row['donor_name'] == 'Sarah Chen' and row['phone'] == 'Demo account'
    assert row['last_donation_at'] and row['total_donations'] == 6
    assert row['accepted_at'] and row['expected_arrival_at'] and row['travel_time_minutes'] > 0
    assert data['recruitment']['accepted_count'] == 1 and data['recruitment']['remaining_needed'] == 1
    invitations = request(api_server, 'donor/invitations', token=donor)[1]['invitations']
    assert invitations[0]['status'] == 'ACCEPTED' and invitations[0]['doctor_name'] == 'Dr. Ramesh Patel'
    assert invitations[0]['seconds_remaining'] > 0
    # Withdrawing releases the reservation.
    assert request(api_server, 'donor/invitations/respond', 'POST', donor,
                   {'invitation_id': 'INV_1001', 'action': 'WITHDRAW'})[0] == 200
    assert request(api_server, path, token=staff)[1]['recruitment']['accepted_count'] == 0


def test_doctor_can_cancel_request(api_server):
    staff, donor = login(api_server, 'staff_001'), login(api_server, 'donor_001')
    created = create(api_server, staff)[1]
    status, cancelled = request(api_server, 'emergency/requests/cancel', 'POST', staff, {'request_id': created['request_id']})
    assert status == 200 and cancelled['status'] == 'CANCELLED'
    invitation = next(i for i in api.STATE.invitations if i['request_id'] == created['request_id'])
    assert invitation['status'] == 'CANCELLED'
    assert accept(api_server, donor, invitation['invitation_id'])[0] == 409
    assert request(api_server, 'emergency/requests/cancel', 'POST', staff, {'request_id': created['request_id']})[0] == 409
    assert request(api_server, 'emergency/requests/cancel', 'POST', donor, {'request_id': created['request_id']})[0] == 401


def test_time_limit_closes_request(api_server):
    staff = login(api_server, 'staff_001')
    created = create(api_server, staff, time_limit_minutes=10)[1]
    emergency = api.STATE.emergencies[created['request_id']]
    emergency['created_at'] = (datetime.now(timezone.utc) - timedelta(minutes=11)).isoformat()
    listed = request(api_server, 'emergency/requests', token=staff)[1]['requests']
    assert next(r for r in listed if r['request_id'] == created['request_id'])['status'] == 'EXPIRED'
    assert all(i['status'] == 'EXPIRED' for i in api.STATE.invitations if i['request_id'] == created['request_id'])


def test_route_payload_for_map(api_server):
    staff = login(api_server, 'staff_001')
    status, route = request(api_server, 'emergency/routes?donor_id=donor_001&request_id=REQ_TRAUMA_101', token=staff)
    assert status == 200
    assert route['origin'] == {'latitude': 12.9352, 'longitude': 77.6245}
    assert route['destination'] == {'latitude': 12.9915, 'longitude': 77.5852}
    assert route['distance_km'] > route['straight_line_km'] > 0
    assert route['provider'] == 'synthetic'


def test_facility_directory_filters_and_pages(api_server):
    staff = login(api_server, 'staff_001')
    data = request(api_server, 'facilities?state=Karnataka&page_size=25', token=staff)[1]
    assert data['total'] == 192 and len(data['facilities']) == 25 and data['has_next']
    assert 'Karnataka' in data['states']
    located = request(api_server, 'facilities?state=Karnataka&has_coordinates=true', token=staff)[1]
    assert 0 < located['total'] == located['with_coordinates'] < 192
    near = request(api_server, 'facilities?near=12.97,77.59&page_size=5', token=staff)[1]['facilities']
    assert [f['distance_km'] for f in near] == sorted(f['distance_km'] for f in near)
    assert request(api_server, 'facilities?page_size=500', token=staff)[0] == 400


def test_sweep_invites_newly_available_donor_without_polling(api_server):
    api.STATE = api.SystemState(sweep_interval_seconds=0.3)
    staff = login(api_server, 'staff_001')
    api.STATE.users['donor_001']['availability'] = False
    created = create(api_server, staff)[1]
    assert created['invited_count'] == 0
    api.STATE.users['donor_001']['availability'] = True
    time.sleep(1.0)
    # Inspect state directly: no HTTP request triggers this sweep.
    assert [i['donor_id'] for i in api.STATE.recruitment.invitations(created['request_id'])] == ['donor_001']
