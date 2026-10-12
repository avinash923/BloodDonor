import 'package:flutter_test/flutter_test.dart';
import 'package:blood_donor_app/services/demo_store.dart';

void main() {
  test('Offline waves invite 10 at a time and preserve reservations', () {
    var now = DateTime.utc(2026, 10, 12, 9);
    final store = DemoStore(clock: () => now, responseWindowSeconds: 10);
    final request = store.create('O+', 3, 'CRITICAL');
    var page = store.recruitmentPage(request.requestId);
    expect(page.candidates.length, 10);
    expect(page.summary['pending_count'], 10);
    final first = store.invitations
        .where((i) => i.requestId == request.requestId)
        .toList();
    for (final invitation in first.take(2)) {
      store.respond(invitation.donorId, invitation.invitationId, 'ACCEPT');
    }
    now = now.add(const Duration(seconds: 10));
    store.tick();
    page = store.recruitmentPage(request.requestId);
    expect(page.summary['batch_number'], 2);
    expect(page.summary['contacted_count'], 20);
    expect(page.summary['accepted_count'], 2);
    expect(page.summary['pending_count'], 10);
    final third = store.invitations.firstWhere(
      (i) => i.requestId == request.requestId && i.status == 'PENDING',
    );
    store.respond(third.donorId, third.invitationId, 'ACCEPT');
    final accepted = store.recruitmentPage(request.requestId, view: 'accepted');
    expect(accepted.candidates.length, 3);
    expect(accepted.summary['pending_count'], 0);
    expect(accepted.summary['status'], 'TARGET_RESERVED');
    now = now.add(const Duration(seconds: 20));
    store.tick();
    expect(
      store.recruitmentPage(request.requestId).summary['contacted_count'],
      20,
    );
  });

  test('Browsing page two does not dispatch a second wave', () {
    final store = DemoStore();
    final request = store.create('O+', 3, 'URGENT');
    final page = store.recruitmentPage(request.requestId, page: 2);
    expect(page.candidates.length, 10);
    expect(
      page.candidates.every((c) => c.invitationStatus == 'QUEUED'),
      isTrue,
    );
    expect(page.summary['contacted_count'], 10);
  });
}
