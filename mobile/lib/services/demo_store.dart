import '../models/models.dart';

/// Local demo data survives refresh and role changes for one app session.
class DemoStore {
  final Map<String, Map<String, dynamic>> requests = {};
  final List<InvitationModel> invitations = [];
  final Map<String, DateTime> deferredUntil = {};
  final Map<String, int> alerts = {'donor_001': 1, 'donor_002': 0};
  int _sequence = 1001;
  final Map<String, int> invitationWaves = {};
  final Map<String, DateTime> invitationExpiries = {};
  final DateTime Function() now;
  final int responseWindowSeconds;

  DemoStore({
    DateTime Function()? clock,
    this.responseWindowSeconds = 60,
    int extraDonors = 30,
  }) : now = clock ?? DateTime.now {
    for (var index = 3; index < extraDonors + 3; index++) {
      final id = 'donor_${index.toString().padLeft(3, '0')}';
      bloodTypes[id] = 'O+';
      names[id] = 'Demo Donor ${index.toString().padLeft(3, '0')}';
      alerts[id] = 0;
    }
    final request = _request('REQ_TRAUMA_101', 'O+', 3, 'CRITICAL');
    request['units_fulfilled'] = 1;
    requests['REQ_TRAUMA_101'] = request;
    _invite(request, 'donor_001');
  }

  final bloodTypes = {'donor_001': 'O+', 'donor_002': 'A+'};
  final names = {'donor_001': 'Sarah Chen', 'donor_002': 'Rahul Sharma'};

  Map<String, dynamic> login(String username) {
    if (!bloodTypes.containsKey(username) && username != 'staff_001') {
      throw Exception('Use donor_001, donor_002 or staff_001');
    }
    return {
      'token': username,
      'user_id': username,
      'name': names[username] ?? 'Dr. Ramesh Patel',
      'role': username == 'staff_001' ? 'hospital_staff' : 'donor',
    };
  }

  bool eligible(String donorId) =>
      deferredUntil[donorId] == null ||
      !now().isBefore(deferredUntil[donorId]!);

  UserProfile profile(String donorId) => UserProfile(
    userId: donorId,
    name: '${names[donorId]} (Mock)',
    role: 'donor',
    bloodType: bloodTypes[donorId],
    phone: '+91 98765 43210',
    latitude: donorId == 'donor_001' ? 12.9352 : 12.9716,
    longitude: donorId == 'donor_001' ? 77.6245 : 77.5946,
    recentAlerts: alerts[donorId] ?? 0,
    isTemporallyEligible: eligible(donorId),
    nextEligibleAt: deferredUntil[donorId]?.toIso8601String(),
    completedDonationsCount: deferredUntil.containsKey(donorId) ? 1 : 0,
  );

  Map<String, dynamic> _request(
    String id,
    String blood,
    int units,
    String urgency,
  ) => {
    'request_id': id,
    'hospital_id': 'HOSP_CITY_01',
    'hospital_name': 'City Central Trauma & Blood Bank',
    'latitude': 12.9915,
    'longitude': 77.5852,
    'blood_type_required': blood,
    'units_required': units,
    'units_fulfilled': 0,
    'urgency': urgency,
    'deadline_minutes': urgency == 'CRITICAL'
        ? 60
        : urgency == 'URGENT'
        ? 120
        : 240,
    'created_at': now().toUtc().toIso8601String(),
    'status': 'ACTIVE',
  };

  void _invite(Map<String, dynamic> request, String donorId) {
    invitations.add(
      InvitationModel(
        invitationId: 'INV_${_sequence++}',
        requestId: request['request_id'],
        donorId: donorId,
        bloodTypeRequired: request['blood_type_required'],
        urgency: request['urgency'],
        hospitalName: request['hospital_name'],
        hospitalLat: request['latitude'],
        hospitalLon: request['longitude'],
        status: 'PENDING',
        sentAt: request['created_at'],
      ),
    );
  }

  EmergencyRequestModel create(String blood, int units, String urgency) {
    if (!['O+', 'O-', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-'].contains(blood) ||
        units < 1 ||
        units > 100 ||
        !['CRITICAL', 'URGENT', 'SCHEDULED'].contains(urgency)) {
      throw Exception('Enter a valid blood group, urgency and 1–100 units');
    }
    final request = _request('REQ_MOCK_${_sequence++}', blood, units, urgency);
    requests[request['request_id']] = request;
    request['recruitment'] = {
      'batch_number': 0,
      'status': 'READY',
      'next_batch_at': null,
    };
    _advance(request);
    return EmergencyRequestModel.fromJson(request);
  }

  bool respond(String donorId, String invitationId, String action) {
    tick();
    final invitation = invitations.firstWhere(
      (inv) => inv.invitationId == invitationId && inv.donorId == donorId,
    );
    if (!['ACCEPT', 'DECLINE'].contains(action) ||
        invitation.status != 'PENDING' ||
        requests[invitation.requestId]?['status'] != 'ACTIVE' ||
        (action == 'ACCEPT' &&
            (!eligible(donorId) ||
                _reservedElsewhere(donorId, invitation.requestId)))) {
      throw Exception('Invitation is no longer actionable');
    }
    final request = requests[invitation.requestId]!;
    if (action == 'ACCEPT' && _remaining(request) == 0) {
      throw Exception('Enough donors have already accepted');
    }
    invitation.status = action == 'ACCEPT' ? 'ACCEPTED' : 'DECLINED';
    if (request['recruitment'] != null) _advance(request);
    return true;
  }

  List<CandidateRecommendationModel> recommendations(String requestId) {
    final request = requests[requestId];
    if (request == null) throw Exception('Request not found');
    if (request['status'] != 'ACTIVE') return [];
    return bloodTypes.keys
        .where(
          (id) =>
              bloodTypes[id] == request['blood_type_required'] &&
              eligible(id) &&
              !_reservedElsewhere(id, requestId),
        )
        .map(
          (id) => CandidateRecommendationModel(
            donorId: id,
            donorName: names[id]!,
            bloodType: bloodTypes[id]!,
            udiScore: 0.2845,
            rank:
                bloodTypes.keys
                    .where(
                      (key) =>
                          bloodTypes[key] == request['blood_type_required'],
                    )
                    .toList()
                    .indexOf(id) +
                1,
            travelTimeMinutes: id == 'donor_001'
                ? 28.5
                : 29 + int.parse(id.split('_').last) / 10,
            isFeasible: true,
            isEligible: true,
            recentAlerts: alerts[id] ?? 0,
            batchNumber:
                invitationWaves[invitations
                    .where((i) => i.donorId == id && i.requestId == requestId)
                    .firstOrNull
                    ?.invitationId] ??
                0,
            invitationStatus:
                invitations
                    .where(
                      (inv) => inv.donorId == id && inv.requestId == requestId,
                    )
                    .map((inv) => inv.status)
                    .firstOrNull ??
                'QUEUED',
          ),
        )
        .toList();
  }

  Map<String, dynamic> complete(String donorId, String requestId) {
    tick();
    final request = requests[requestId];
    final accepted = invitations.where(
      (inv) =>
          inv.donorId == donorId &&
          inv.requestId == requestId &&
          inv.status == 'ACCEPTED',
    );
    if (request == null ||
        request['status'] != 'ACTIVE' ||
        !eligible(donorId) ||
        bloodTypes[donorId] != request['blood_type_required'] ||
        accepted.isEmpty) {
      throw Exception(
        'An eligible donor must accept this active request before confirmation',
      );
    }
    accepted.first.status = 'COMPLETED';
    deferredUntil[donorId] = now().add(const Duration(days: 90));
    request['units_fulfilled'] = (request['units_fulfilled'] as int) + 1;
    if (request['units_fulfilled'] >= request['units_required']) {
      request['status'] = 'FULFILLED';
    }
    for (final inv in invitations) {
      if ((inv.donorId == donorId ||
              (inv.requestId == requestId &&
                  request['status'] == 'FULFILLED')) &&
          ['PENDING', 'ACCEPTED'].contains(inv.status)) {
        inv.status = 'CANCELLED';
      }
    }
    if (request['recruitment'] != null) _advance(request);
    return {
      'message': 'Donation recorded (Mock)',
      'deferral_days': 90.0,
      'next_eligible_at': deferredUntil[donorId]!.toIso8601String(),
    };
  }

  bool _reservedElsewhere(String donorId, String requestId) => invitations.any(
    (i) =>
        i.donorId == donorId &&
        i.requestId != requestId &&
        i.status == 'ACCEPTED',
  );

  int _remaining(Map<String, dynamic> request) {
    final accepted = invitations
        .where(
          (i) => i.requestId == request['request_id'] && i.status == 'ACCEPTED',
        )
        .length;
    final needed =
        (request['units_required'] as int) -
        (request['units_fulfilled'] as int) -
        accepted;
    return needed < 0 ? 0 : needed;
  }

  void tick() {
    for (final request in requests.values) {
      if (request['recruitment'] != null && request['status'] == 'ACTIVE') {
        _advance(request);
      }
    }
  }

  void _advance(Map<String, dynamic> request) {
    final recruitment = request['recruitment'] as Map<String, dynamic>;
    final related = invitations
        .where((i) => i.requestId == request['request_id'])
        .toList();
    final deadline = DateTime.parse(
      request['created_at'],
    ).add(Duration(minutes: request['deadline_minutes']));
    for (final invite in related) {
      if (invite.status == 'PENDING' &&
          invitationExpiries[invite.invitationId] != null &&
          !now().isBefore(invitationExpiries[invite.invitationId]!)) {
        invite.status = 'EXPIRED';
      }
    }
    if (request['status'] != 'ACTIVE' || !now().isBefore(deadline)) {
      if (request['status'] == 'ACTIVE') request['status'] = 'EXPIRED';
      for (final invite in related) {
        if (['PENDING', 'ACCEPTED'].contains(invite.status)) {
          invite.status = 'CANCELLED';
        }
      }
      recruitment['status'] = 'CLOSED';
      recruitment['next_batch_at'] = null;
      return;
    }
    if (_remaining(request) == 0) {
      for (final invite in related) {
        if (invite.status == 'PENDING') invite.status = 'CANCELLED';
      }
      recruitment['status'] = 'TARGET_RESERVED';
      recruitment['next_batch_at'] = null;
      return;
    }
    if (related.any((i) => i.status == 'PENDING')) return;
    final contacted = related.map((i) => i.donorId).toSet();
    final remaining = recommendations(
      request['request_id'],
    ).where((c) => !contacted.contains(c.donorId)).take(10).toList();
    if (remaining.isEmpty) {
      recruitment['status'] = 'EXHAUSTED';
      recruitment['next_batch_at'] = null;
      return;
    }
    recruitment['batch_number'] = (recruitment['batch_number'] as int) + 1;
    var expiry = now().add(Duration(seconds: responseWindowSeconds));
    if (expiry.isAfter(deadline)) expiry = deadline;
    recruitment['next_batch_at'] = expiry.toUtc().toIso8601String();
    recruitment['status'] = 'AWAITING_RESPONSES';
    for (final candidate in remaining) {
      _invite(request, candidate.donorId);
      final invite = invitations.last;
      invitationWaves[invite.invitationId] = recruitment['batch_number'];
      invitationExpiries[invite.invitationId] = expiry;
      alerts[candidate.donorId] = (alerts[candidate.donorId] ?? 0) + 1;
    }
  }

  RecruitmentPage recruitmentPage(
    String requestId, {
    int page = 1,
    String view = 'all',
  }) {
    tick();
    final request = requests[requestId];
    if (request == null) throw Exception('Request not found');
    final related = invitations.where((i) => i.requestId == requestId).toList();
    final all = recommendations(requestId);
    final selected = view == 'accepted'
        ? all.where((c) => c.invitationStatus == 'ACCEPTED').toList()
        : all;
    final start = (page - 1) * 10;
    return RecruitmentPage(
      candidates: selected.skip(start).take(10).toList(),
      summary: {
        ...?request['recruitment'] as Map<String, dynamic>?,
        'accepted_count': related.where((i) => i.status == 'ACCEPTED').length,
        'pending_count': related.where((i) => i.status == 'PENDING').length,
        'contacted_count': related.length,
        'remaining_needed': _remaining(request),
        'units_fulfilled': request['units_fulfilled'],
        'units_required': request['units_required'],
      },
      page: page,
      total: selected.length,
      hasNext: start + 10 < selected.length,
    );
  }
}
