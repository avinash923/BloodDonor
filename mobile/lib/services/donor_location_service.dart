import 'package:geolocator/geolocator.dart';
import 'api_service.dart';

/// One foreground fix per explicit action. Never starts background tracking.
class DonorLocationService {
  static Future<void> shareCurrent(ApiService api) async {
    var permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
    }
    if (permission == LocationPermission.denied ||
        permission == LocationPermission.deniedForever) {
      throw Exception(
        'Location permission denied. Enable it in your device or browser settings to join nearby matching.',
      );
    }
    final position = await Geolocator.getCurrentPosition(
      locationSettings: const LocationSettings(
        accuracy: LocationAccuracy.high,
        timeLimit: Duration(seconds: 30),
      ),
    );
    await api.updateDonorLocation({
      'consent': true,
      'latitude': position.latitude,
      'longitude': position.longitude,
      'accuracy_meters': position.accuracy,
      'captured_at': position.timestamp.toUtc().toIso8601String(),
    });
  }
}
