"""Deterministic recruitment lifecycle and reservation tests."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from src.api.server import SystemState
from src.spatial.routing_provider import RouteEstimate


NOW = datetime(2026, 10, 12, 9, 0, tzinfo=timezone.utc)


@pytest.fixture
def state(monkeypatch):
    state = SystemState(response_window_seconds=10, demo_donors=24)
    state.invitations = []
    state.emergencies = {}

    def etas(candidates, coordinates, **kwargs):
        for candidate in candidates:
            candidate.travel_time_minutes = coordinates[candidate.donor_id][0] - 12 + 1
        return candidates

    monkeypatch.setattr(state.routing_service, 'populate_candidates_with_etas', etas)
    return state


def create(state, req_id='WAVE_TEST', units=3):
    request = {'request_id': req_id, 'hospital_id': 'TEST', 'hospital_name': 'Demo Hospital',
               'latitude': 12.9915, 'longitude': 77.5852, 'blood_type_required': 'O+',
               'urgency': 'CRITICAL', 'units_required': units, 'units_fulfilled': 0,
               'created_at': NOW.isoformat(), 'deadline_minutes': 60, 'status': 'ACTIVE'}
    state.emergencies[req_id] = request
    state.recruitment.start(request, NOW)
    return request


def test_top_ten_then_next_ten_then_remaining_without_duplicates(state):
    request = create(state)
    first = list(state.invitations)
    assert len(first) == 10
    assert all(i['batch_number'] == 1 for i in first)
    pool = request['recruitment']['pool']
    assert [i['donor_id'] for i in first] == [r['donor_id'] for r in pool[:10]]
    state.recruitment.tick(NOW + timedelta(seconds=9))
    assert len(state.invitations) == 10
    state.recruitment.tick(NOW + timedelta(seconds=10))
    assert len(state.invitations) == 20
    assert all(i['status'] == 'EXPIRED' for i in first)
    state.recruitment.tick(NOW + timedelta(seconds=20))
    assert len(state.invitations) == 25
    assert len({i['donor_id'] for i in state.invitations}) == 25
    state.recruitment.tick(NOW + timedelta(seconds=30))
    assert request['recruitment']['status'] == 'EXHAUSTED'


def test_partial_acceptance_escalates_and_stops_at_required_count(state):
    request = create(state)
    state.recruitment.respond(state.invitations[0], 'ACCEPT', NOW)
    state.recruitment.respond(state.invitations[1], 'ACCEPT', NOW)
    state.recruitment.tick(NOW + timedelta(seconds=10))
    assert len(state.invitations) == 20
    state.recruitment.respond(state.invitations[10], 'ACCEPT', NOW + timedelta(seconds=11))
    assert request['recruitment']['status'] == 'TARGET_RESERVED'
    assert sum(i['status'] == 'ACCEPTED' for i in state.invitations) == 3
    assert not any(i['status'] == 'PENDING' for i in state.invitations)
    state.recruitment.tick(NOW + timedelta(seconds=40))
    assert len(state.invitations) == 20
    with pytest.raises(ValueError):
        state.recruitment.respond(state.invitations[11], 'ACCEPT', NOW + timedelta(seconds=41))


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
    assert request['recruitment']['status'] == 'TARGET_RESERVED'


def test_expired_invitation_and_request_deadline(state):
    request = create(state)
    first = state.invitations[0]
    with pytest.raises(ValueError):
        state.recruitment.respond(first, 'ACCEPT', NOW + timedelta(seconds=11))
    state.recruitment.tick(NOW + timedelta(minutes=60))
    assert request['status'] == 'EXPIRED'
    assert not any(i['status'] == 'PENDING' for i in state.invitations)


def test_declining_entire_wave_advances_early(state):
    create(state)
    for invitation in list(state.invitations):
        state.recruitment.respond(invitation, 'DECLINE', NOW)
    assert len(state.invitations) == 20
    assert state.invitations[-1]['batch_number'] == 2


def test_accepted_list_refreshes_road_order(state):
    request = create(state)
    first, second = state.invitations[:2]
    state.recruitment.respond(first, 'ACCEPT', NOW)
    state.recruitment.respond(second, 'ACCEPT', NOW)
    state.users[first['donor_id']]['latitude'] = 14
    _, accepted, summary = state.recruitment.snapshot(request, NOW)
    assert [a['donor_id'] for a in accepted] == [second['donor_id'], first['donor_id']]
    assert [a['rank'] for a in accepted] == [1, 2]
    assert summary['remaining_needed'] == 1


def test_failed_routes_are_not_invited_and_accepted_failures_stay_visible(state, monkeypatch):
    request = create(state)
    first = state.invitations[0]
    state.recruitment.respond(first, 'ACCEPT', NOW)

    def failed(candidates, *args, **kwargs):
        for candidate in candidates:
            candidate.travel_time_minutes = None
        return candidates

    monkeypatch.setattr(state.routing_service, 'populate_candidates_with_etas', failed)
    results, accepted, summary = state.recruitment.snapshot(request, NOW)
    assert results == []
    assert len(accepted) == 1 and accepted[0]['travel_time_minutes'] is None
    assert not accepted[0]['is_feasible']
    assert summary['accepted_count'] == 1
    state.recruitment.tick(NOW + timedelta(seconds=11))
    assert len(state.invitations) == 10
    assert request['recruitment']['status'] == 'EXHAUSTED'


def test_donor_cannot_commit_to_two_requests(state):
    create(state)
    first = state.invitations[0]
    other = create(state, 'SECOND')
    competing = next(i for i in state.invitations if i['request_id'] == 'SECOND' and i['donor_id'] == first['donor_id'])
    state.recruitment.respond(first, 'ACCEPT', NOW)
    with pytest.raises(ValueError):
        state.recruitment.respond(competing, 'ACCEPT', NOW)


def test_pending_waves_resume_after_restart(state, tmp_path):
    request = create(state)
    state.state_path = tmp_path / 'wave_state.json'
    state.save()
    restored = SystemState(str(state.state_path), response_window_seconds=10)
    assert len(restored.users) == len(state.users)
    restored.recruitment.tick(NOW + timedelta(seconds=11))
    assert restored.emergencies['WAVE_TEST']['recruitment']['batch_number'] == 2
    assert len(restored.invitations) == 20
