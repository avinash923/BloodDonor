"""OSRM provider against a local fake OSRM server (no internet needed)."""
import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.spatial.osrm_provider import OSRMRoutingProvider, FALLBACK_PROVIDER
from src.spatial.routing_provider import LocationCoordinate
from src.spatial.routing_service import RoutingService

HOSPITAL = LocationCoordinate(12.9915, 77.5852)
NEAR = LocationCoordinate(12.9352, 77.6245)
FAR = LocationCoordinate(12.9716, 77.5946)


@pytest.fixture
def fake_osrm():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            parsed = urllib.parse.urlsplit(self.path)  # keep ";" inside the path
            calls.append(parsed)
            parts = parsed.path.split('/')  # ['', service, 'v1', profile, coords]
            service, coords = parts[1], parts[4].split(';')
            if service == 'table':
                sources = urllib.parse.parse_qs(parsed.query)['sources'][0].split(';')
                # Second source is unreachable (OSRM returns null).
                durations = [[600.0 * (n + 1)] if n != 1 else [None] for n in range(len(sources))]
                distances = [[5000.0 * (n + 1)] if n != 1 else [None] for n in range(len(sources))]
                body = {'code': 'Ok', 'durations': durations, 'distances': distances}
            else:
                body = {'code': 'Ok', 'routes': [{'geometry': '_p~iF~ps|U_ulLnnqC', 'distance': 8123.4,
                                                  'duration': 905.0}]}
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_port}', calls
    server.shutdown()
    server.server_close()


def test_table_batch_uses_lon_lat_and_handles_unreachable(fake_osrm):
    base, calls = fake_osrm
    provider = OSRMRoutingProvider(base_url=base)
    result = provider.compute_donor_eta_batch({'a': NEAR, 'b': FAR, 'c': NEAR}, HOSPITAL, 'H1')
    assert result['a'].is_success and result['a'].duration_minutes == 10.0 and result['a'].provider == 'osrm'
    assert result['a'].distance_meters == 5000.0
    assert not result['b'].is_success and result['b'].duration_minutes is None
    assert result['c'].duration_minutes == 30.0
    first = calls[0]
    assert first.path.startswith('/table/v1/driving/77.585200,12.991500;77.624500,12.935200')
    assert urllib.parse.parse_qs(first.query)['destinations'] == ['0']
    # Cached: the same donors do not trigger another request within the TTL.
    provider.compute_donor_eta_batch({'a': NEAR}, HOSPITAL, 'H1')
    assert len(calls) == 1


def test_detailed_route_returns_polyline(fake_osrm):
    base, calls = fake_osrm
    route = OSRMRoutingProvider(base_url=base).compute_detailed_route(NEAR, HOSPITAL, 'donor', 'H1')
    assert route.is_success and route.provider == 'osrm'
    assert route.polyline_encoded == '_p~iF~ps|U_ulLnnqC'
    assert route.distance_meters == 8123.4 and round(route.duration_minutes, 2) == 15.08
    assert 'overview=full' in calls[0].query and 'geometries=polyline' in calls[0].query


def test_unreachable_server_falls_back_and_trips_breaker():
    provider = OSRMRoutingProvider(base_url='http://127.0.0.1:9', timeout_seconds=1)
    route = provider.compute_detailed_route(NEAR, HOSPITAL)
    assert route.is_success and route.provider == FALLBACK_PROVIDER and route.polyline_encoded is None
    assert not provider.available
    batch = provider.compute_donor_eta_batch({'a': NEAR}, HOSPITAL)
    assert batch['a'].provider == FALLBACK_PROVIDER and batch['a'].duration_minutes > 0


def test_routing_service_selects_osrm(monkeypatch):
    monkeypatch.setenv('ROUTING_PROVIDER', 'osrm')
    monkeypatch.setenv('OSRM_BASE_URL', 'http://osrm.internal:5000/')
    service = RoutingService()
    assert isinstance(service.provider, OSRMRoutingProvider)
    assert service.provider.base_url == 'http://osrm.internal:5000'
