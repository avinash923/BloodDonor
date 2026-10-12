"""Donor account isolation, freshness, consent and facility import regression checks."""
from datetime import datetime, timedelta, timezone
import json

import pytest

import src.api.server as api
from src.api.donor_accounts import location_ready, update_location
from tests.test_api_server import api_server, request, login, create


def register(base, **overrides):
    return request(base, 'auth/register', 'POST', body={
        'username': 'new_donor', 'name': 'New Donor', 'password': 'a-long-test-password',
        'blood_type': 'O+', **overrides})


def fix(**overrides):
    return {'consent': True, 'latitude': 12.9915, 'longitude': 77.5852,
            'accuracy_meters': 15, 'captured_at': datetime.now(timezone.utc).isoformat(), **overrides}


def test_registration_password_privacy_and_persistence(api_server, tmp_path):
    api.STATE.state_path = tmp_path / 'state.json'
    status, account = register(api_server)
    assert status == 201
    token = account['token']
    assert 'password' not in account and 'password_record' not in account
    profile = request(api_server, 'donor/profile', token=token)[1]
    assert profile['latitude'] is None and profile['availability'] is False
    assert not profile['location_ready']
    assert 'password_record' not in profile
    assert request(api_server, 'auth/login', 'POST', body={'username': 'new_donor'})[0] == 401
    assert request(api_server, 'auth/login', 'POST', body={'username': 'new_donor', 'password': 'wrong'})[0] == 401
    assert request(api_server, 'auth/login', 'POST', body={'username': 'new_donor', 'password': 'a-long-test-password'})[0] == 200
    assert register(api_server)[0] == 409
    assert 'a-long-test-password' not in api.STATE.state_path.read_text()
    restored = api.SystemState(str(api.STATE.state_path))
    assert 'new_donor' in restored.users
    assert restored.users['new_donor']['account_source'] == 'registered'
    assert 'new_donor' in restored.donation_tracker.donor_states


@pytest.mark.parametrize('fields', [
    {'username': 'staff_hacker'}, {'username': []}, {'username': 'bad'},
    {'name': ''}, {'password': 'short'}, {'blood_type': 'unknown'},
])
def test_invalid_registration(api_server, fields):
    assert register(api_server, **fields)[0] == 400
    assert len(api.STATE.users) == 3


def test_location_is_owned_consented_and_revocable(api_server):
    token = register(api_server)[1]['token']
    path = 'donor/location'
    assert request(api_server, path, 'POST', body=fix())[0] == 401
    assert request(api_server, path, 'POST', login(api_server, 'staff_001'), fix())[0] == 401
    assert request(api_server, path, 'POST', token, fix(consent=None))[0] == 400
    assert request(api_server, path, 'POST', token, fix())[1]['location_ready']
    assert api.STATE.users['new_donor']['latitude'] == 12.9915
    assert request(api_server, path, 'POST', token, {'consent': False})[0] == 200
    assert api.STATE.users['new_donor']['latitude'] is None
    assert not api.STATE.users['new_donor']['availability']


@pytest.mark.parametrize('fields', [
    {'latitude': True}, {'longitude': 181}, {'accuracy_meters': 251},
    {'accuracy_meters': -1}, {'latitude': '12'}, {'latitude': float('nan')},
    {'captured_at': 'bad'}, {'captured_at': '2026-10-12T10:00:00'},
    {'captured_at': (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()},
    {'captured_at': (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()},
])
def test_bad_location_is_rejected_without_overwriting(fields):
    donor = {'account_source': 'registered', 'latitude': None}
    with pytest.raises(ValueError):
        update_location(donor, fix(**fields))
    assert donor['latitude'] is None


def test_freshness_prefilter_and_acceptance_recheck(api_server, monkeypatch):
    token = register(api_server)[1]['token']
    staff = login(api_server, 'staff_001')
    api.STATE.users['donor_001']['availability'] = False
    _, emergency = create(api_server, staff)
    assert emergency['recruitment']['status'] == 'EXHAUSTED'
    assert request(api_server, 'donor/location', 'POST', token, fix())[0] == 200
    invitations = request(api_server, 'donor/invitations', token=token)[1]['invitations']
    assert len(invitations) == 1  # Newly available donor wakes the exhausted request.
    donor = api.STATE.users['new_donor']
    captured = datetime.fromisoformat(donor['location_captured_at'])
    assert location_ready(donor, captured + timedelta(minutes=14))
    assert not location_ready(donor, captured + timedelta(minutes=16))
    donor['location_captured_at'] = (captured - timedelta(minutes=16)).isoformat()
    assert request(api_server, 'donor/invitations/respond', 'POST', token,
                   {'invitation_id': invitations[0]['invitation_id'], 'action': 'ACCEPT'})[0] == 409
    assert request(api_server, 'donor/location', 'POST', token, fix())[0] == 200
    assert request(api_server, 'donor/invitations/respond', 'POST', token,
                   {'invitation_id': invitations[0]['invitation_id'], 'action': 'ACCEPT'})[0] == 200
    # A faraway location must be filtered before routing is invoked for that donor.
    donor.update(latitude=28.6, longitude=77.2)
    seen = []
    def routes(candidates, coords, **kwargs):
        seen.extend(coords)
        return []
    monkeypatch.setattr(api.STATE.routing_service, 'populate_candidates_with_etas', routes)
    api.STATE.recruitment.rank(emergency, datetime.now(timezone.utc))
    assert 'new_donor' not in seen


def test_directory_access_and_request_coordinates(api_server):
    staff = login(api_server, 'staff_001')
    assert request(api_server, 'facilities')[0] == 401
    assert request(api_server, 'facilities', token=login(api_server, 'donor_001'))[0] == 403
    directory = request(api_server, 'facilities?search=Karnataka', token=staff)[1]
    assert directory['total'] == 192
    facility = next(f for f in directory['facilities'] if f['latitude'] is not None)
    assert create(api_server, staff, facility_id=facility['facility_id'])[0] == 400
    status, emergency = create(api_server, staff, facility_id=facility['facility_id'], facility_confirmed=True)
    assert status == 201
    assert emergency['latitude'] == facility['latitude']
    assert emergency['hospital_name'] == facility['name']
    assert emergency['facility_source'] == 'historical_directory_2015'


def test_demo_login_can_be_disabled(api_server, monkeypatch):
    monkeypatch.setenv('ALLOW_DEMO_LOGIN', 'false')
    assert request(api_server, 'auth/login', 'POST', body={'username': 'staff_001'})[0] == 401
    register(api_server)
    assert request(api_server, 'auth/login', 'POST', body={'username': 'new_donor', 'password': 'a-long-test-password'})[0] == 200
