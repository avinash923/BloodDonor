// lib/config/app_config.dart
// Application environment configuration for Phase 9 Flutter client.

import 'package:flutter/foundation.dart';

class AppConfig {
  // Backend REST API Base URL
  // For local Android emulator, use 10.0.2.2 instead of 127.0.0.1
  static const String defaultBaseUrl = 'http://127.0.0.1:8000';
  static const String emulatorBaseUrl = 'http://10.0.2.2:8000';

  // Toggle between live backend and client-side mock development mode
  static bool mockMode = const bool.fromEnvironment('MOCK_MODE');
  static const bool mapsEnabled = bool.fromEnvironment('MAPS_ENABLED');

  // Active base URL
  static String baseUrl =
      const String.fromEnvironment('API_BASE_URL', defaultValue: '').isNotEmpty
      ? const String.fromEnvironment('API_BASE_URL')
      : (kIsWeb
            ? Uri.base.origin
            : defaultTargetPlatform == TargetPlatform.android
            ? emulatorBaseUrl
            : defaultBaseUrl);

  // Timeouts
  static const Duration connectTimeout = Duration(seconds: 10);
  static const Duration receiveTimeout = Duration(seconds: 15);
}
