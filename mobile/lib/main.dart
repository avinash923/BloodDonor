// lib/main.dart
// Application Entry Point for AI Blood Donor Recommendation Mobile App (Phase 9).

import 'package:flutter/material.dart';
import 'services/api_service.dart';
import 'screens/login_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  final apiService = ApiService();
  runApp(BloodDonorApp(apiService: apiService));
}

class BloodDonorApp extends StatelessWidget {
  final ApiService apiService;

  const BloodDonorApp({super.key, required this.apiService});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Emergency Blood Donor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFFD32F2F),
          primary: const Color(0xFFD32F2F),
          secondary: const Color(0xFF1E293B),
        ),
        useMaterial3: true,
        scaffoldBackgroundColor: const Color(0xFFF8FAFC),
        appBarTheme: const AppBarTheme(centerTitle: false, elevation: 0),
      ),
      home: LoginScreen(apiService: apiService),
    );
  }
}
