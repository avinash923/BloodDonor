import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:blood_donor_app/screens/login_screen.dart';
import 'package:blood_donor_app/screens/donor_dashboard_screen.dart';
import 'package:blood_donor_app/screens/staff_dashboard_screen.dart';
import 'package:blood_donor_app/screens/recommendations_screen.dart';
import 'package:blood_donor_app/services/api_service.dart';
import 'package:blood_donor_app/models/models.dart';
import 'package:blood_donor_app/config/app_config.dart';

void main() {
  setUp(() {
    AppConfig.mockMode = true;
  });

  group('Mobile Screens Widget Tests', () {
    testWidgets('LoginScreen renders branding and role selection chips', (
      WidgetTester tester,
    ) async {
      final apiService = ApiService();

      await tester.pumpWidget(
        MaterialApp(home: LoginScreen(apiService: apiService)),
      );

      // Verify header and branding
      expect(find.text('Emergency Blood Donor'), findsOneWidget);
      expect(find.text('AI Time-Aware Recommendation System'), findsOneWidget);
      expect(find.text('Sign In'), findsOneWidget);
      expect(find.text('Username or ID'), findsOneWidget);
      expect(find.text('Donor (Sarah)'), findsOneWidget);
      expect(find.text('Hospital Staff (Dr. Patel)'), findsOneWidget);
      expect(find.text('Continue'), findsOneWidget);
      expect(find.text('Offline Mock Mode'), findsOneWidget);
    });

    testWidgets(
      'DonorDashboardScreen displays donor profile and emergency alerts in mock mode',
      (WidgetTester tester) async {
        final apiService = ApiService();
        apiService.setAuth('donor_001', 'donor_001');

        await tester.pumpWidget(
          MaterialApp(home: DonorDashboardScreen(apiService: apiService)),
        );

        // Allow async invitation and profile fetching to finish
        await tester.pumpAndSettle();

        // Check donor status card
        expect(find.text('Donor Portal'), findsOneWidget);
        expect(find.text('Sarah Chen (Mock)'), findsOneWidget);
        expect(find.text('O+'), findsWidgets);
        expect(find.text('Clinically Eligible to Donate'), findsOneWidget);

        // Check emergency invitations
        expect(find.text('Emergency Invitations'), findsOneWidget);
        expect(find.text('City Central Trauma & Blood Bank'), findsOneWidget);
        expect(find.text('CRITICAL'), findsOneWidget);
        expect(find.text('Accept Alert'), findsOneWidget);
        expect(find.text('Decline'), findsOneWidget);
      },
    );

    testWidgets(
      'StaffDashboardScreen renders request list and dialog trigger',
      (WidgetTester tester) async {
        final apiService = ApiService();
        apiService.setAuth('staff_001', 'staff_001');

        await tester.pumpWidget(
          MaterialApp(home: StaffDashboardScreen(apiService: apiService)),
        );

        await tester.pumpAndSettle();

        expect(find.text('Hospital Emergency Dispatch'), findsOneWidget);
        expect(find.text('New Emergency'), findsOneWidget);
        expect(find.text('City Central Trauma & Blood Bank'), findsOneWidget);
        expect(find.text('View Recommendations & Recruits'), findsOneWidget);

        // Tap New Emergency button to open dialog
        await tester.tap(find.text('New Emergency'));
        await tester.pumpAndSettle();

        expect(find.text('Create Emergency Request'), findsOneWidget);
        expect(find.text('Blood Group Required'), findsOneWidget);
        expect(find.text('Submit Request'), findsOneWidget);
      },
    );

    testWidgets(
      'RecommendationsScreen displays ranked candidates and UDI metrics',
      (WidgetTester tester) async {
        final apiService = ApiService();
        apiService.setAuth('staff_001', 'staff_001');

        final mockRequest = EmergencyRequestModel(
          requestId: 'REQ_TRAUMA_101',
          hospitalId: 'HOSP_CENTRAL',
          hospitalName: 'City Central Trauma & Blood Bank',
          latitude: 12.9915,
          longitude: 77.5852,
          bloodTypeRequired: 'O+',
          unitsRequired: 2,
          unitsFulfilled: 0,
          urgency: 'CRITICAL',
          deadlineMinutes: 60,
          createdAt: '2026-10-11T20:00:00Z',
          status: 'ACTIVE',
        );

        await tester.pumpWidget(
          MaterialApp(
            home: RecommendationsScreen(
              apiService: apiService,
              emergencyRequest: mockRequest,
            ),
          ),
        );

        await tester.pumpAndSettle();

        // Verify request header
        expect(find.text('Donor Recruitment'), findsOneWidget);
        expect(find.textContaining('Multiplicative UDI'), findsOneWidget);

        // Verify ranked candidates list
        expect(find.text('Sarah Chen'), findsOneWidget);
        expect(find.text('Rahul Sharma'), findsNothing);
        expect(find.text('UDI Score'), findsWidgets);
        expect(find.text('View Map'), findsWidgets);
        expect(find.text('Confirm Donation'), findsWidgets);
      },
    );
  });
}
