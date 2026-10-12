import 'package:flutter_test/flutter_test.dart';
import 'package:blood_donor_app/main.dart';
import 'package:blood_donor_app/services/api_service.dart';

void main() {
  testWidgets('App root smoke test mounts LoginScreen', (
    WidgetTester tester,
  ) async {
    final apiService = ApiService();
    await tester.pumpWidget(BloodDonorApp(apiService: apiService));
    expect(find.text('Emergency Blood Donor'), findsOneWidget);
    expect(find.text('AI Time-Aware Recommendation System'), findsOneWidget);
    expect(find.text('Sign In'), findsOneWidget);
  });
}
