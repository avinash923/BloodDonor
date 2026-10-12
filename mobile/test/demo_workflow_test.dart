import 'package:flutter_test/flutter_test.dart';
import 'package:blood_donor_app/config/app_config.dart';
import 'package:blood_donor_app/services/api_service.dart';

void main() {
  setUp(() => AppConfig.mockMode = true);
  tearDown(() => AppConfig.mockMode = false);

  test(
    'Offline creation, acceptance, confirmation and refresh share state',
    () async {
      final api = ApiService();
      await api.login('staff_001');
      final request = await api.createEmergencyRequest(
        bloodTypeRequired: 'O+',
        unitsRequired: 1,
        urgency: 'CRITICAL',
      );
      expect(
        (await api.getEmergencyRequests()).any(
          (r) => r.requestId == request.requestId,
        ),
        isTrue,
      );
      final candidates = await api.getRecommendations(request.requestId);
      expect(candidates.map((c) => c.bloodType), everyElement('O+'));
      await expectLater(
        api.completeDonation('donor_001', request.requestId),
        throwsException,
      );
      api.clearAuth();
      await api.login('donor_001');
      final invite = (await api.getDonorInvitations()).firstWhere(
        (i) => i.requestId == request.requestId,
      );
      await api.respondToInvitation(invite.invitationId, 'ACCEPT');
      expect(
        (await api.getDonorInvitations())
            .firstWhere((i) => i.invitationId == invite.invitationId)
            .status,
        'ACCEPTED',
      );
      api.clearAuth();
      await api.login('staff_001');
      expect(
        (await api.getRecommendations(
          request.requestId,
        )).firstWhere((c) => c.donorId == 'donor_001').invitationStatus,
        'ACCEPTED',
      );
      await api.completeDonation('donor_001', request.requestId);
      await expectLater(
        api.completeDonation('donor_001', request.requestId),
        throwsException,
      );
      final refreshed = (await api.getEmergencyRequests()).firstWhere(
        (r) => r.requestId == request.requestId,
      );
      expect(refreshed.unitsFulfilled, 1);
      expect(refreshed.status, 'FULFILLED');
      expect(await api.getRecommendations(request.requestId), isEmpty);
      await api.login('donor_001');
      expect((await api.getDonorProfile()).isTemporallyEligible, isFalse);
    },
  );

  test(
    'Offline demo rejects invalid users and retains declined invitations',
    () async {
      final api = ApiService();
      await expectLater(api.login('invalid'), throwsException);
      await api.login('donor_001');
      final invitation = (await api.getDonorInvitations()).first;
      await api.respondToInvitation(invitation.invitationId, 'DECLINE');
      expect((await api.getDonorInvitations()).first.status, 'DECLINED');
      await expectLater(
        api.respondToInvitation(invitation.invitationId, 'ACCEPT'),
        throwsException,
      );
    },
  );
}
