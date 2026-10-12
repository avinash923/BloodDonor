import 'package:flutter_test/flutter_test.dart';
import 'package:blood_donor_app/utils/polyline_decoder.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';

void main() {
  group('PolylineDecoder Tests', () {
    test('Decodes standard encoded polyline accurately', () {
      // Standard Google polyline encoding for [(38.5, -120.2), (40.7, -120.95), (43.252, -126.453)]
      // is '_p~iF~ps|U_ulLnnqC_mqNvxq`@'
      const encoded = '_p~iF~ps|U_ulLnnqC_mqNvxq`@';
      final points = PolylineDecoder.decodePolyline(encoded);

      expect(points.length, equals(3));
      expect(points[0].latitude, closeTo(38.5, 0.001));
      expect(points[0].longitude, closeTo(-120.2, 0.001));
      expect(points[1].latitude, closeTo(40.7, 0.001));
      expect(points[1].longitude, closeTo(-120.95, 0.001));
      expect(points[2].latitude, closeTo(43.252, 0.001));
      expect(points[2].longitude, closeTo(-126.453, 0.001));
    });

    test('Decodes two close points correctly', () {
      // (12.9716, 77.5946) to (12.9800, 77.6000)
      const encoded = 'czenA_c|xMsk@_k@';
      final points = PolylineDecoder.decodePolyline(encoded);

      expect(points.isNotEmpty, isTrue);
      expect(points.first.latitude, closeTo(12.9716, 0.01));
      expect(points.first.longitude, closeTo(77.6454, 0.01));
    });

    test('Handles empty and whitespace strings gracefully', () {
      expect(PolylineDecoder.decodePolyline('').isEmpty, isTrue);
      expect(PolylineDecoder.decodePolyline('   ').isEmpty, isTrue);
    });

    test('Handles truncated or malformed polyline without crashing', () {
      const malformed = '???invalid_data_string_not_encoded!!!';
      final points = PolylineDecoder.decodePolyline(malformed);
      expect(points, isA<List<LatLng>>());
    });

    test(
      'Calculates LatLngBounds correctly for single and multiple points',
      () {
        final points = [
          const LatLng(12.9716, 77.5946),
          const LatLng(13.0000, 77.6500),
          const LatLng(12.9500, 77.5500),
        ];

        final bounds = PolylineDecoder.boundsFromLatLngList(points);
        expect(bounds.southwest.latitude, closeTo(12.9500, 0.0001));
        expect(bounds.southwest.longitude, closeTo(77.5500, 0.0001));
        expect(bounds.northeast.latitude, closeTo(13.0000, 0.0001));
        expect(bounds.northeast.longitude, closeTo(77.6500, 0.0001));
      },
    );

    test('Throws ArgumentError when bounds requested for empty list', () {
      expect(
        () => PolylineDecoder.boundsFromLatLngList([]),
        throwsArgumentError,
      );
    });
  });
}
