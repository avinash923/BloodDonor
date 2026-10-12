// lib/models/models.dart
// Typed data models for the Phase 9 mobile application matching backend schemas.

class UserProfile {
  final String userId;
  final String name;
  final String role; // 'donor' or 'hospital_staff'
  final String? bloodType;
  final String? phone;
  final double? latitude;
  final double? longitude;
  final bool availability;
  final int recentAlerts;
  final bool isTemporallyEligible;
  final String? eligibilityReason;
  final String? nextEligibleAt;
  final int completedDonationsCount;
  final String accountSource;
  final bool locationReady;
  final bool locationConsent;
  final String? locationCapturedAt;
  final double? locationAccuracyMeters;

  UserProfile({
    required this.userId,
    required this.name,
    required this.role,
    this.bloodType,
    this.phone,
    this.latitude,
    this.longitude,
    this.availability = true,
    this.recentAlerts = 0,
    this.isTemporallyEligible = true,
    this.eligibilityReason,
    this.nextEligibleAt,
    this.completedDonationsCount = 0,
    this.accountSource = 'synthetic_demo',
    this.locationReady = false,
    this.locationConsent = false,
    this.locationCapturedAt,
    this.locationAccuracyMeters,
  });

  factory UserProfile.fromJson(Map<String, dynamic> json) {
    return UserProfile(
      userId: json['user_id'] ?? '',
      name: json['name'] ?? '',
      role: json['role'] ?? 'donor',
      bloodType: json['blood_type'],
      phone: json['phone'],
      latitude: json['latitude'] != null
          ? (json['latitude'] as num).toDouble()
          : null,
      longitude: json['longitude'] != null
          ? (json['longitude'] as num).toDouble()
          : null,
      availability: json['availability'] ?? true,
      recentAlerts: json['recent_alerts'] ?? 0,
      isTemporallyEligible: json['is_temporally_eligible'] ?? true,
      eligibilityReason: json['eligibility_reason'],
      nextEligibleAt: json['next_eligible_at'],
      completedDonationsCount: json['completed_donations_count'] ?? 0,
      accountSource: json['account_source'] ?? 'synthetic_demo',
      locationReady: json['location_ready'] ?? false,
      locationConsent: json['location_consent'] ?? false,
      locationCapturedAt: json['location_captured_at'],
      locationAccuracyMeters: (json['location_accuracy_meters'] as num?)
          ?.toDouble(),
    );
  }
}

class EmergencyRequestModel {
  final String requestId;
  final String hospitalId;
  final String hospitalName;
  final double latitude;
  final double longitude;
  final String bloodTypeRequired;
  final int unitsRequired;
  final int unitsFulfilled;
  final String urgency; // CRITICAL, URGENT, SCHEDULED
  final int deadlineMinutes;
  final String createdAt;
  final String status;

  EmergencyRequestModel({
    required this.requestId,
    required this.hospitalId,
    required this.hospitalName,
    required this.latitude,
    required this.longitude,
    required this.bloodTypeRequired,
    required this.unitsRequired,
    required this.unitsFulfilled,
    required this.urgency,
    required this.deadlineMinutes,
    required this.createdAt,
    required this.status,
  });

  factory EmergencyRequestModel.fromJson(Map<String, dynamic> json) {
    return EmergencyRequestModel(
      requestId: json['request_id'] ?? '',
      hospitalId: json['hospital_id'] ?? '',
      hospitalName: json['hospital_name'] ?? 'Hospital',
      latitude: (json['latitude'] as num?)?.toDouble() ?? 0.0,
      longitude: (json['longitude'] as num?)?.toDouble() ?? 0.0,
      bloodTypeRequired: json['blood_type_required'] ?? 'O+',
      unitsRequired: json['units_required'] ?? 1,
      unitsFulfilled: json['units_fulfilled'] ?? 0,
      urgency: json['urgency'] ?? 'URGENT',
      deadlineMinutes: json['deadline_minutes'] ?? 120,
      createdAt: json['created_at'] ?? '',
      status: json['status'] ?? 'ACTIVE',
    );
  }
}

class InvitationModel {
  final String invitationId;
  final String requestId;
  final String donorId;
  final String bloodTypeRequired;
  final String urgency;
  final String hospitalName;
  final double hospitalLat;
  final double hospitalLon;
  String status; // PENDING, ACCEPTED, DECLINED
  final String sentAt;

  InvitationModel({
    required this.invitationId,
    required this.requestId,
    required this.donorId,
    required this.bloodTypeRequired,
    required this.urgency,
    required this.hospitalName,
    required this.hospitalLat,
    required this.hospitalLon,
    required this.status,
    required this.sentAt,
  });

  factory InvitationModel.fromJson(Map<String, dynamic> json) {
    return InvitationModel(
      invitationId: json['invitation_id'] ?? '',
      requestId: json['request_id'] ?? '',
      donorId: json['donor_id'] ?? '',
      bloodTypeRequired: json['blood_type_required'] ?? '',
      urgency: json['urgency'] ?? '',
      hospitalName: json['hospital_name'] ?? 'Emergency Site',
      hospitalLat: (json['hospital_lat'] as num?)?.toDouble() ?? 0.0,
      hospitalLon: (json['hospital_lon'] as num?)?.toDouble() ?? 0.0,
      status: json['status'] ?? 'PENDING',
      sentAt: json['sent_at'] ?? '',
    );
  }
}

class CandidateRecommendationModel {
  final int batchNumber;
  final String invitationStatus;
  final String donorId;
  final String donorName;
  final String bloodType;
  final double udiScore;
  final int rank;
  final double? travelTimeMinutes;
  final bool isFeasible;
  final bool isEligible;
  final int recentAlerts;

  CandidateRecommendationModel({
    this.batchNumber = 0,
    this.invitationStatus = 'NONE',
    required this.donorId,
    required this.donorName,
    required this.bloodType,
    required this.udiScore,
    required this.rank,
    this.travelTimeMinutes,
    required this.isFeasible,
    required this.isEligible,
    required this.recentAlerts,
  });

  factory CandidateRecommendationModel.fromJson(Map<String, dynamic> json) {
    return CandidateRecommendationModel(
      batchNumber: json['batch_number'] ?? 0,
      invitationStatus: json['invitation_status'] ?? 'NONE',
      donorId: json['donor_id'] ?? '',
      donorName: json['donor_name'] ?? '',
      bloodType: json['blood_type'] ?? '',
      udiScore: (json['udi_score'] as num?)?.toDouble() ?? 0.0,
      rank: json['rank'] ?? 0,
      travelTimeMinutes: (json['travel_time_minutes'] as num?)?.toDouble(),
      isFeasible: json['is_feasible'] ?? false,
      isEligible: json['is_eligible'] ?? false,
      recentAlerts: json['recent_alerts'] ?? 0,
    );
  }
}

class RecruitmentPage {
  final List<CandidateRecommendationModel> candidates;
  final Map<String, dynamic> summary;
  final int page;
  final int total;
  final bool hasNext;

  RecruitmentPage({
    required this.candidates,
    required this.summary,
    required this.page,
    required this.total,
    required this.hasNext,
  });

  factory RecruitmentPage.fromJson(Map<String, dynamic> json) =>
      RecruitmentPage(
        candidates: (json['recommendations'] as List? ?? [])
            .map((item) => CandidateRecommendationModel.fromJson(item))
            .toList(),
        summary: Map<String, dynamic>.from(json['recruitment'] ?? {}),
        page: json['page'] ?? 1,
        total: json['total'] ?? 0,
        hasNext: json['has_next'] ?? false,
      );
}

class RouteDetailModel {
  final String donorId;
  final String requestId;
  final String hospitalName;
  final double originLat;
  final double originLon;
  final double destLat;
  final double destLon;
  final bool isSuccess;
  final double? durationMinutes;
  final double? distanceKm;
  final String? polylineEncoded;
  final String provider;
  final String? errorCode;

  RouteDetailModel({
    required this.donorId,
    required this.requestId,
    required this.hospitalName,
    required this.originLat,
    required this.originLon,
    required this.destLat,
    required this.destLon,
    required this.isSuccess,
    this.durationMinutes,
    this.distanceKm,
    this.polylineEncoded,
    required this.provider,
    this.errorCode,
  });

  factory RouteDetailModel.fromJson(Map<String, dynamic> json) {
    final orig = json['origin'] as Map<String, dynamic>? ?? {};
    final dest = json['destination'] as Map<String, dynamic>? ?? {};

    return RouteDetailModel(
      donorId: json['donor_id'] ?? '',
      requestId: json['request_id'] ?? '',
      hospitalName: json['hospital_name'] ?? 'Hospital',
      originLat: (orig['latitude'] as num?)?.toDouble() ?? 0.0,
      originLon: (orig['longitude'] as num?)?.toDouble() ?? 0.0,
      destLat: (dest['latitude'] as num?)?.toDouble() ?? 0.0,
      destLon: (dest['longitude'] as num?)?.toDouble() ?? 0.0,
      isSuccess: json['is_success'] ?? false,
      durationMinutes: (json['duration_minutes'] as num?)?.toDouble(),
      distanceKm: (json['distance_km'] as num?)?.toDouble(),
      polylineEncoded: json['polyline_encoded'],
      provider: json['provider'] ?? 'synthetic',
      errorCode: json['error_code'],
    );
  }
}
