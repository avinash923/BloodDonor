import 'package:flutter_test/flutter_test.dart';
import 'package:blood_donor_app/models/models.dart';

void main() {
  group('Models Serialization and Zero-ETA Safety Tests', () {
    test('UserProfile JSON parsing and temporal eligibility flag', () {
      final json = {
        'user_id': 'donor_42',
        'name': 'Sarah Chen',
        'role': 'donor',
        'blood_type': 'O_NEG',
        'phone': '+91-9876543210',
        'latitude': 12.9716,
        'longitude': 77.5946,
        'availability': true,
        'recent_alerts': 1,
        'is_temporally_eligible': true,
        'eligibility_reason': 'Passed 90 days inter-donation interval',
        'next_eligible_at': '2026-10-01T00:00:00Z',
        'completed_donations_count': 3,
      };

      final profile = UserProfile.fromJson(json);
      expect(profile.userId, equals('donor_42'));
      expect(profile.name, equals('Sarah Chen'));
      expect(profile.role, equals('donor'));
      expect(profile.bloodType, equals('O_NEG'));
      expect(profile.latitude, equals(12.9716));
      expect(profile.longitude, equals(77.5946));
      expect(profile.isTemporallyEligible, isTrue);
      expect(profile.recentAlerts, equals(1));
      expect(profile.completedDonationsCount, equals(3));
    });

    test('EmergencyRequestModel JSON parsing and status check', () {
      final json = {
        'request_id': 'req_101',
        'hospital_id': 'HOSP_CENTRAL',
        'hospital_name': 'Central City Hospital',
        'latitude': 12.9352,
        'longitude': 77.6245,
        'blood_type_required': 'B_POS',
        'units_required': 3,
        'units_fulfilled': 1,
        'urgency': 'CRITICAL',
        'deadline_minutes': 45,
        'created_at': '2026-10-11T20:00:00Z',
        'status': 'ACTIVE',
      };

      final req = EmergencyRequestModel.fromJson(json);
      expect(req.requestId, equals('req_101'));
      expect(req.hospitalName, equals('Central City Hospital'));
      expect(req.bloodTypeRequired, equals('B_POS'));
      expect(req.unitsRequired, equals(3));
      expect(req.unitsFulfilled, equals(1));
      expect(req.urgency, equals('CRITICAL'));
      expect(req.deadlineMinutes, equals(45));
      expect(req.status, equals('ACTIVE'));
    });

    test('InvitationModel JSON parsing and response tracking', () {
      final json = {
        'invitation_id': 'inv_001',
        'request_id': 'req_101',
        'donor_id': 'donor_42',
        'blood_type_required': 'B_POS',
        'urgency': 'HIGH',
        'hospital_name': 'Metro Care Emergency Site',
        'hospital_lat': 12.935,
        'hospital_lon': 77.624,
        'status': 'PENDING',
        'sent_at': '2026-10-11T20:05:00Z',
      };

      final inv = InvitationModel.fromJson(json);
      expect(inv.invitationId, equals('inv_001'));
      expect(inv.status, equals('PENDING'));
      expect(inv.hospitalName, equals('Metro Care Emergency Site'));
      expect(inv.hospitalLat, equals(12.935));
    });

    test(
      'CandidateRecommendationModel parses UDI scores and prioritization',
      () {
        final json = {
          'donor_id': 'donor_99',
          'donor_name': 'Alex Kumar',
          'blood_type': 'O_POS',
          'udi_score': 0.8421,
          'rank': 1,
          'travel_time_minutes': 18.5,
          'is_feasible': true,
          'is_eligible': true,
          'recent_alerts': 0,
        };

        final rec = CandidateRecommendationModel.fromJson(json);
        expect(rec.donorId, equals('donor_99'));
        expect(rec.udiScore, equals(0.8421));
        expect(rec.rank, equals(1));
        expect(rec.travelTimeMinutes, equals(18.5));
        expect(rec.isFeasible, isTrue);
        expect(rec.isEligible, isTrue);
      },
    );

    test(
      'RouteDetailModel enforces Zero-ETA Safety Policy (Never converts failed route to 0)',
      () {
        // Scenario A: Failed route has is_success=false and durationMinutes=null
        final failedJson = {
          'donor_id': 'donor_10',
          'request_id': 'req_20',
          'hospital_name': 'District Hospital',
          'origin': {'latitude': 12.97, 'longitude': 77.59},
          'destination': {'latitude': 12.93, 'longitude': 77.62},
          'is_success': false,
          'duration_minutes': null,
          'distance_km': null,
          'polyline_encoded': null,
          'provider': 'google_routes',
          'error_code': 'ROUTING_UNAVAILABLE',
        };

        final failedRoute = RouteDetailModel.fromJson(failedJson);
        expect(failedRoute.isSuccess, isFalse);
        expect(failedRoute.durationMinutes, isNull);
        expect(failedRoute.distanceKm, isNull);
        expect(failedRoute.errorCode, equals('ROUTING_UNAVAILABLE'));
        // Verifying Zero-ETA Safety: A failed route must never evaluate to 0 minutes
        expect(failedRoute.durationMinutes != 0.0, isTrue);

        // Scenario B: Successful route
        final successJson = {
          'donor_id': 'donor_10',
          'request_id': 'req_20',
          'hospital_name': 'District Hospital',
          'origin': {'latitude': 12.97, 'longitude': 77.59},
          'destination': {'latitude': 12.93, 'longitude': 77.62},
          'is_success': true,
          'duration_minutes': 14.2,
          'distance_km': 6.8,
          'polyline_encoded': 'czenA_c|xMsk@_k@',
          'provider': 'synthetic',
        };

        final successRoute = RouteDetailModel.fromJson(successJson);
        expect(successRoute.isSuccess, isTrue);
        expect(successRoute.durationMinutes, equals(14.2));
        expect(successRoute.distanceKm, equals(6.8));
        expect(successRoute.polylineEncoded, equals('czenA_c|xMsk@_k@'));
      },
    );
  });
}
