// lib/services/api_service.dart
// HTTP REST Client connecting to the Python backend recommendation service.

import 'dart:convert';
import 'package:http/http.dart' as http;
import '../config/app_config.dart';
import '../models/models.dart';
import 'demo_store.dart';

class ApiService {
  final http.Client _client;
  final DemoStore _demo = DemoStore();
  String? _authToken;
  String? _currentUserId;

  ApiService({http.Client? client}) : _client = client ?? http.Client();

  void setAuth(String token, String userId) {
    _authToken = token;
    _currentUserId = userId;
  }

  void clearAuth() {
    _authToken = null;
    _currentUserId = null;
  }

  String? get currentUserId => _currentUserId;
  bool get isAuthenticated => _authToken != null;

  Map<String, String> _headers() {
    final headers = {'Content-Type': 'application/json'};
    if (_authToken != null) {
      headers['Authorization'] = 'Bearer $_authToken';
    }
    return headers;
  }

  // 1. Health check
  Future<bool> checkHealth() async {
    try {
      final uri = Uri.parse('${AppConfig.baseUrl}/api/v1/health');
      final resp = await _client.get(uri).timeout(AppConfig.connectTimeout);
      return resp.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  // 2. Login
  Future<Map<String, dynamic>> login(
    String username, {
    String password = '',
  }) async {
    if (AppConfig.mockMode) {
      final data = _demo.login(username);
      setAuth(data['token'], data['user_id']);
      return data;
    }

    final uri = Uri.parse('${AppConfig.baseUrl}/api/v1/auth/login');
    final resp = await _client
        .post(
          uri,
          headers: _headers(),
          body: jsonEncode({'username': username, 'password': password}),
        )
        .timeout(AppConfig.connectTimeout);

    if (resp.statusCode == 200) {
      final data = jsonDecode(resp.body) as Map<String, dynamic>;
      setAuth(data['token'], data['user_id']);
      return data;
    } else {
      final err = jsonDecode(resp.body);
      throw Exception(err['error'] ?? 'Login failed (${resp.statusCode})');
    }
  }

  // 3. Donor Profile
  Future<Map<String, dynamic>> registerDonor(
    Map<String, dynamic> fields,
  ) async {
    if (AppConfig.mockMode) {
      throw Exception(
        'Turn off Offline Mock Mode to register a saved donor account',
      );
    }
    final response = await _client
        .post(
          Uri.parse('${AppConfig.baseUrl}/api/v1/auth/register'),
          headers: _headers(),
          body: jsonEncode(fields),
        )
        .timeout(AppConfig.connectTimeout);
    final data = jsonDecode(response.body) as Map<String, dynamic>;
    if (response.statusCode != 201) {
      throw Exception(data['error'] ?? 'Registration failed');
    }
    setAuth(data['token'], data['user_id']);
    return data;
  }

  Future<void> updateDonorLocation(Map<String, dynamic> fields) async {
    if (AppConfig.mockMode) {
      throw Exception('Location sharing requires the backend');
    }
    final response = await _client
        .post(
          Uri.parse('${AppConfig.baseUrl}/api/v1/donor/location'),
          headers: _headers(),
          body: jsonEncode(fields),
        )
        .timeout(AppConfig.connectTimeout);
    if (response.statusCode != 200) {
      throw Exception(
        jsonDecode(response.body)['error'] ?? 'Location update failed',
      );
    }
  }

  Future<List<Map<String, dynamic>>> searchFacilities(String search) async {
    if (AppConfig.mockMode) return [];
    final uri = Uri.parse(
      '${AppConfig.baseUrl}/api/v1/facilities',
    ).replace(queryParameters: {'search': search});
    final response = await _client
        .get(uri, headers: _headers())
        .timeout(AppConfig.connectTimeout);
    if (response.statusCode != 200) {
      throw Exception('Could not load facility directory');
    }
    return (jsonDecode(response.body)['facilities'] as List)
        .cast<Map<String, dynamic>>();
  }

  Future<UserProfile> getDonorProfile() async {
    if (AppConfig.mockMode) {
      return _demo.profile(_currentUserId ?? 'donor_001');
    }

    final uri = Uri.parse('${AppConfig.baseUrl}/api/v1/donor/profile');
    final resp = await _client
        .get(uri, headers: _headers())
        .timeout(AppConfig.receiveTimeout);

    if (resp.statusCode == 200) {
      return UserProfile.fromJson(jsonDecode(resp.body));
    } else {
      throw Exception('Failed to fetch donor profile (${resp.statusCode})');
    }
  }

  // 4. Donor Invitations
  Future<List<InvitationModel>> getDonorInvitations() async {
    if (AppConfig.mockMode) {
      _demo.tick();
      return _demo.invitations
          .where((inv) => inv.donorId == _currentUserId)
          .toList();
    }

    final uri = Uri.parse('${AppConfig.baseUrl}/api/v1/donor/invitations');
    final resp = await _client
        .get(uri, headers: _headers())
        .timeout(AppConfig.receiveTimeout);

    if (resp.statusCode == 200) {
      final data = jsonDecode(resp.body);
      final list = (data['invitations'] as List? ?? []);
      return list.map((item) => InvitationModel.fromJson(item)).toList();
    } else {
      throw Exception('Failed to fetch invitations (${resp.statusCode})');
    }
  }

  // 5. Respond to Invitation
  Future<bool> respondToInvitation(String invitationId, String action) async {
    if (AppConfig.mockMode) {
      return _demo.respond(_currentUserId ?? '', invitationId, action);
    }

    final uri = Uri.parse(
      '${AppConfig.baseUrl}/api/v1/donor/invitations/respond',
    );
    final resp = await _client
        .post(
          uri,
          headers: _headers(),
          body: jsonEncode({'invitation_id': invitationId, 'action': action}),
        )
        .timeout(AppConfig.connectTimeout);

    return resp.statusCode == 200;
  }

  // 6. Emergency Requests
  Future<List<EmergencyRequestModel>> getEmergencyRequests() async {
    if (AppConfig.mockMode) {
      _demo.tick();
      return _demo.requests.values.map(EmergencyRequestModel.fromJson).toList();
    }

    final uri = Uri.parse('${AppConfig.baseUrl}/api/v1/emergency/requests');
    final resp = await _client
        .get(uri, headers: _headers())
        .timeout(AppConfig.receiveTimeout);

    if (resp.statusCode == 200) {
      final data = jsonDecode(resp.body);
      final list = (data['requests'] as List? ?? []);
      return list.map((item) => EmergencyRequestModel.fromJson(item)).toList();
    } else {
      throw Exception(
        'Failed to fetch emergency requests (${resp.statusCode})',
      );
    }
  }

  // 7. Create Emergency Request
  Future<EmergencyRequestModel> createEmergencyRequest({
    required String bloodTypeRequired,
    required int unitsRequired,
    required String urgency,
    String? facilityId,
    bool facilityConfirmed = false,
  }) async {
    if (AppConfig.mockMode) {
      return _demo.create(bloodTypeRequired, unitsRequired, urgency);
    }

    final uri = Uri.parse('${AppConfig.baseUrl}/api/v1/emergency/requests');
    final resp = await _client
        .post(
          uri,
          headers: _headers(),
          body: jsonEncode({
            'blood_type_required': bloodTypeRequired,
            'units_required': unitsRequired,
            'urgency': urgency,
            if (facilityId != null) 'facility_id': facilityId,
            if (facilityId != null) 'facility_confirmed': facilityConfirmed,
          }),
        )
        .timeout(AppConfig.connectTimeout);

    if (resp.statusCode == 201) {
      return EmergencyRequestModel.fromJson(jsonDecode(resp.body));
    } else {
      throw Exception(
        'Failed to create emergency request (${resp.statusCode})',
      );
    }
  }

  // Candidate pages and accepted donors are separate from invitation dispatch.
  Future<List<CandidateRecommendationModel>> getRecommendations(
    String requestId,
  ) async {
    return (await getRecruitmentPage(requestId)).candidates;
  }

  Future<RecruitmentPage> getRecruitmentPage(
    String requestId, {
    int page = 1,
    String view = 'all',
  }) async {
    if (AppConfig.mockMode) {
      return _demo.recruitmentPage(requestId, page: page, view: view);
    }
    final uri =
        Uri.parse(
          '${AppConfig.baseUrl}/api/v1/emergency/recommendations',
        ).replace(
          queryParameters: {
            'request_id': requestId,
            'page': '$page',
            'view': view,
          },
        );
    final response = await _client
        .get(uri, headers: _headers())
        .timeout(AppConfig.receiveTimeout);
    if (response.statusCode != 200) {
      throw Exception('Failed to load recruitment (${response.statusCode})');
    }
    return RecruitmentPage.fromJson(jsonDecode(response.body));
  }

  // 9. Get Route Details & Encoded Polyline
  Future<RouteDetailModel> getRouteDetail(
    String donorId,
    String requestId,
  ) async {
    if (AppConfig.mockMode) {
      final request = _demo.requests[requestId];
      if (request == null) throw Exception('Request not found');
      final profile = _demo.profile(donorId);
      return RouteDetailModel(
        donorId: donorId,
        requestId: requestId,
        hospitalName: request['hospital_name'],
        originLat: profile.latitude!,
        originLon: profile.longitude!,
        destLat: request['latitude'],
        destLon: request['longitude'],
        isSuccess: true,
        durationMinutes: 28.5,
        distanceKm: 9.8,
        provider: 'synthetic_mock',
      );
    }

    final uri = Uri.parse(
      '${AppConfig.baseUrl}/api/v1/emergency/routes?donor_id=$donorId&request_id=$requestId',
    );
    final resp = await _client
        .get(uri, headers: _headers())
        .timeout(AppConfig.receiveTimeout);

    if (resp.statusCode == 200) {
      return RouteDetailModel.fromJson(jsonDecode(resp.body));
    } else {
      throw Exception('Failed to fetch route detail (${resp.statusCode})');
    }
  }

  // 10. Complete Donation (Updates backend temporal deferral)
  Future<Map<String, dynamic>> completeDonation(
    String donorId,
    String requestId,
  ) async {
    if (AppConfig.mockMode) {
      return _demo.complete(donorId, requestId);
    }

    final uri = Uri.parse(
      '${AppConfig.baseUrl}/api/v1/emergency/donation/complete',
    );
    final resp = await _client
        .post(
          uri,
          headers: _headers(),
          body: jsonEncode({'donor_id': donorId, 'request_id': requestId}),
        )
        .timeout(AppConfig.connectTimeout);

    if (resp.statusCode == 200) {
      return jsonDecode(resp.body);
    } else {
      throw Exception(
        'Failed to record completed donation (${resp.statusCode})',
      );
    }
  }
}
