"""Deterministic broadcast-recruitment lifecycle and reservation tests."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from src.api.server import SystemState
from src.spatial.routing_provider import RouteEstimate


NOW = datetime(2026, 10, 12, 9, 0, tzinfo=timezone.utc)


@pytest.fixture
def state(monkeypatch):
    state = SystemState(demo_donors=24)
    state.invitations = []
    state.emergencies = {}
    # Demo donors sit north-east of the hospital in a line; ETA grows with latitude.
    state.users['donor_001'].update(latitude=12.9915 + 26 * 0.0007, longitude=77.5852)

    def etas(donors, hospital_id='H', **kwargs):
        return {did: RouteEstimate(did, hospital_id, distance_meters=(lat - 12.99) * 1e5,
                                   duration_minutes=(lat - 12.99) * 1000, provider='test')
                for did, lat, lon in donors}

    monkeypatch.setattr(state.routing_service, 'compute_donor_etas', etas)
    return state


def create(state, req_id='BROADCAST_TEST', units=3, minutes=60, **extra):
    request = {'request_id': req_id, 'hospital_id': 'TEST', 'hospital_name': 'Demo Hospital',
               'latitude': 12.9915, 'longitude': 77.5852, 'blood_type_required': 'O+',
               'units_required': units, 'units_fulfilled': 0, 'created_at': NOW.isoformat(),
               'deadline_minutes': minutes, 'status': 'ACTIVE', **extra}
    state.emergencies[req_id] = request
    state.recruitment.start(request, NOW)
    return request


def test_all_feasible_donors_invited_at_once_ranked_by_match(state):
    request = create(state, minutes=12)
    invited = {i['donor_id'] for i in state.invitations}
    rows = state.recruitment.evaluate(request, NOW)
    feasible = [r for r in rows if r['can_arrive_in_time']]
    assert invited == {r['donor_id'] for r in feasible}
    assert len(feasible) < len(rows)  # donors who cannot arrive within the time limit are not invited
    assert all(r["travel_time_minutes"] <= 12 for r in feasible)
    assert [r['match_score'] for r in feasible] == sorted((r['match_score'] for r in feasible), reverse=True)
    assert all(i['expires_at'] == (NOW + timedelta(minutes=12)).isoformat() for i in state.invitations)
    assert request['recruitment']['status'] == 'BROADCASTING'
    # The score combines the history-based likelihood with timeliness.
    top = feasible[0]
    assert top['match_score'] == pytest.approx(
        top['donation_likelihood'] * (1 - top["travel_time_minutes"] / 12), abs=2e-3)


def test_search_radius_limits_broadcast(state):
    create(state, radius_req := 'RADIUS', search_radius_km=1)
    invited = [i for i in state.invitations if i['request_id'] == radius_req]
    assert invited and all(i['distance_km'] <= 1 for i in invited)
    assert len(invited) < 24


def test_acceptance_stops_at_required_count_and_withdrawal_reopens(state):
    request = create(state)
    pending = list(state.invitations)
    for invitation in pending[:3]:
        state.recruitment.respond(invitation, 'ACCEPT', NOW)
    assert request['recruitment']['status'] == 'TARGET_REACHED'
    assert not any(i['status'] == 'PENDING' for i in state.invitations)
    with pytest.raises(ValueError):
        state.recruitment.respond(pending[3], 'ACCEPT', NOW)
    state.recruitment.respond(pending[0], 'WITHDRAW', NOW + timedelta(minutes=1))
    assert request['recruitment']['status'] == 'BROADCASTING'
    assert pending[3]['status'] == 'PENDING'
    state.recruitment.respond(pending[3], 'ACCEPT', NOW + timedelta(minutes=2))
    assert sum(i['status'] == 'ACCEPTED' for i in state.invitations) == 3


def test_simultaneous_acceptances_cannot_overbook(state):
    request = create(state, units=1)
    invitations = list(state.invitations)

    def accept(invitation):
        try:
            state.recruitment.respond(invitation, 'ACCEPT', NOW)
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=10) as executor:
        assert sum(executor.map(accept, invitations)) == 1
    assert request['recruitment']['status'] == 'TARGET_REACHED'


def test_request_deadline_expires_invitations(state):
    request = create(state)
    first = state.invitations[0]
    state.recruitment.tick(NOW + timedelta(minutes=60))
    assert request['status'] == 'EXPIRED'
    assert not any(i['status'] == 'PENDING' for i in state.invitations)
    with pytest.raises(ValueError):
        state.recruitment.respond(first, 'ACCEPT', NOW + timedelta(minutes=61))


def test_accepted_list_refreshes_road_order_and_shows_contact(state):
    request = create(state)
    first, second = state.invitations[:2]
    state.recruitment.respond(first, 'ACCEPT', NOW)
    state.recruitment.respond(second, 'ACCEPT', NOW)
    state.users[first['donor_id']]['latitude'] = 13.02
    _, accepted, summary = state.recruitment.snapshot(request, NOW)
    assert [a['donor_id'] for a in accepted] == [second['donor_id'], first['donor_id']]
    assert [a['rank'] for a in accepted] == [1, 2]
    assert all(a['phone'] and a['total_donations'] > 0 and a['last_donation_at'] for a in accepted)
    assert summary['remaining_needed'] == 1


def test_failed_routes_are_not_invited_and_accepted_failures_stay_visible(state, monkeypatch):
    request = create(state)
    first = state.invitations[0]
    state.recruitment.respond(first, 'ACCEPT', NOW)
    monkeypatch.setattr(state.routing_service, 'compute_donor_etas', lambda donors, **kw: {})
    _, accepted, summary = state.recruitment.snapshot(request, NOW)
    assert len(accepted) == 1 and not accepted[0]['is_feasible']
    assert summary['accepted_count'] == 1
    state.users['donor_new'] = dict(state.users['donor_003'], user_id='donor_new')
    assert state.recruitment.sweep(NOW + timedelta(minutes=1)) == 0


def test_donor_cannot_commit_to_two_requests(state):
    create(state)
    first = state.invitations[0]
    create(state, 'SECOND')
    competing = next(i for i in state.invitations if i['request_id'] == 'SECOND' and i['donor_id'] == first['donor_id'])
    state.recruitment.respond(first, 'ACCEPT', NOW)
    with pytest.raises(ValueError):
        state.recruitment.respond(competing, 'ACCEPT', NOW)


def test_sweep_invites_new_donors_once_and_resumes_after_restart(state, tmp_path):
    request = create(state)
    before = len(state.invitations)
    state.users['donor_new'] = dict(state.users['donor_003'], user_id='donor_new', name='New')
    state.recruitment.tick(NOW + timedelta(seconds=10))  # before the sweep interval
    assert len(state.invitations) == before
    state.recruitment.tick(NOW + timedelta(seconds=31))
    assert len(state.invitations) == before + 1
    state.recruitment.tick(NOW + timedelta(seconds=70))
    assert len(state.invitations) == before + 1
    state.state_path = tmp_path / 'broadcast_state.json'
    state.save()
    restored = SystemState(str(state.state_path))
    assert len(restored.users) == len(state.users)
    assert len(restored.invitations) == before + 1
    restored.recruitment.tick(NOW + timedelta(minutes=61))
    assert restored.emergencies['BROADCAST_TEST']['status'] == 'EXPIRED'
