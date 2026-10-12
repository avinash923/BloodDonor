// lib/utils/polyline_decoder.dart
// Decodes Google encoded polyline strings to List<LatLng> for google_maps_flutter.

import 'package:google_maps_flutter/google_maps_flutter.dart';

class PolylineDecoder {
  /// Decodes a Google encoded polyline string into a list of [LatLng] points.
  /// Handles empty, malformed, or invalid strings gracefully without crashing.
  static List<LatLng> decodePolyline(String? encoded) {
    if (encoded == null || encoded.trim().isEmpty) {
      return [];
    }

    final List<LatLng> points = [];
    int index = 0;
    final int len = encoded.length;
    int lat = 0;
    int lng = 0;

    try {
      while (index < len) {
        int b;
        int shift = 0;
        int result = 0;
        do {
          if (index >= len) break;
          b = encoded.codeUnitAt(index++) - 63;
          result |= (b & 0x1f) << shift;
          shift += 5;
        } while (b >= 0x20);

        final int dlat = ((result & 1) != 0 ? ~(result >> 1) : (result >> 1));
        lat += dlat;

        shift = 0;
        result = 0;
        do {
          if (index >= len) break;
          b = encoded.codeUnitAt(index++) - 63;
          result |= (b & 0x1f) << shift;
          shift += 5;
        } while (b >= 0x20);

        final int dlng = ((result & 1) != 0 ? ~(result >> 1) : (result >> 1));
        lng += dlng;

        final double finalLat = lat / 1E5;
        final double finalLng = lng / 1E5;

        // Coordinate sanity check
        if (finalLat >= -90.0 &&
            finalLat <= 90.0 &&
            finalLng >= -180.0 &&
            finalLng <= 180.0) {
          points.add(LatLng(finalLat, finalLng));
        }
      }
    } catch (_) {
      // In case of truncated or corrupted polyline string, return parsed points so far
      return points;
    }

    return points;
  }

  /// Computes the bounding box covering a list of [LatLng] coordinates.
  /// Throws [ArgumentError] if the list is empty.
  static LatLngBounds boundsFromLatLngList(List<LatLng> list) {
    if (list.isEmpty) {
      throw ArgumentError('Cannot calculate bounds of an empty list');
    }
    double? x0, x1, y0, y1;
    for (final latLng in list) {
      if (x0 == null) {
        x0 = x1 = latLng.latitude;
        y0 = y1 = latLng.longitude;
      } else {
        if (latLng.latitude > x1!) x1 = latLng.latitude;
        if (latLng.latitude < x0) x0 = latLng.latitude;
        if (latLng.longitude > y1!) y1 = latLng.longitude;
        if (latLng.longitude < y0!) y0 = latLng.longitude;
      }
    }
    return LatLngBounds(
      southwest: LatLng(x0!, y0!),
      northeast: LatLng(x1!, y1!),
    );
  }
}
