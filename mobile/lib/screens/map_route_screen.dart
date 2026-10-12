// lib/screens/map_route_screen.dart
// Interactive Google Map Screen rendering hospital marker, donor marker & route polyline.

import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';
import '../services/api_service.dart';
import '../models/models.dart';
import '../utils/polyline_decoder.dart';
import '../config/app_config.dart';

class MapRouteScreen extends StatefulWidget {
  final ApiService apiService;
  final String donorId;
  final String requestId;
  final String hospitalName;

  const MapRouteScreen({
    super.key,
    required this.apiService,
    required this.donorId,
    required this.requestId,
    required this.hospitalName,
  });

  @override
  State<MapRouteScreen> createState() => _MapRouteScreenState();
}

class _MapRouteScreenState extends State<MapRouteScreen> {
  GoogleMapController? _mapController;
  @override
  void dispose() {
    _mapController?.dispose();
    super.dispose();
  }

  RouteDetailModel? _routeDetail;
  bool _isLoading = true;
  String? _errorMessage;

  final Set<Marker> _markers = {};
  final Set<Polyline> _polylines = {};

  @override
  void initState() {
    super.initState();
    _loadRoute();
  }

  Future<void> _loadRoute() async {
    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    try {
      final detail = await widget.apiService.getRouteDetail(
        widget.donorId,
        widget.requestId,
      );
      if (!mounted) return;

      setState(() {
        _routeDetail = detail;
        _setupMapElements(detail);
      });
    } catch (e) {
      if (!mounted) return;
      setState(
        () => _errorMessage = e.toString().replaceAll('Exception:', '').trim(),
      );
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  void _setupMapElements(RouteDetailModel detail) {
    _markers.clear();
    _polylines.clear();

    final origin = LatLng(detail.originLat, detail.originLon);
    final destination = LatLng(detail.destLat, detail.destLon);

    // 1. Donor Marker
    _markers.add(
      Marker(
        markerId: const MarkerId('donor_origin'),
        position: origin,
        infoWindow: InfoWindow(
          title: 'Donor Location (${widget.donorId})',
          snippet: 'ETA: ${detail.durationMinutes ?? "--"} min',
        ),
        icon: BitmapDescriptor.defaultMarkerWithHue(BitmapDescriptor.hueAzure),
      ),
    );

    // 2. Hospital Marker
    _markers.add(
      Marker(
        markerId: const MarkerId('hospital_destination'),
        position: destination,
        infoWindow: InfoWindow(
          title: widget.hospitalName,
          snippet: 'Emergency Destination Site',
        ),
        icon: BitmapDescriptor.defaultMarkerWithHue(BitmapDescriptor.hueRed),
      ),
    );

    // 3. Polyline Decoding & Construction
    if (detail.polylineEncoded != null && detail.polylineEncoded!.isNotEmpty) {
      final decodedPoints = PolylineDecoder.decodePolyline(
        detail.polylineEncoded!,
      );
      if (decodedPoints.isNotEmpty) {
        _polylines.add(
          Polyline(
            polylineId: const PolylineId('emergency_route_path'),
            points: decodedPoints,
            color: const Color(0xFFD32F2F),
            width: 5,
            jointType: JointType.round,
            endCap: Cap.roundCap,
            startCap: Cap.roundCap,
          ),
        );
      }
    }

    // Adjust camera if controller is ready
    if (_mapController != null) {
      _fitBounds(origin, destination);
    }
  }

  void _fitBounds(LatLng p1, LatLng p2) {
    final southWest = LatLng(
      p1.latitude < p2.latitude ? p1.latitude : p2.latitude,
      p1.longitude < p2.longitude ? p1.longitude : p2.longitude,
    );
    final northEast = LatLng(
      p1.latitude > p2.latitude ? p1.latitude : p2.latitude,
      p1.longitude > p2.longitude ? p1.longitude : p2.longitude,
    );
    final bounds = LatLngBounds(southwest: southWest, northeast: northEast);
    _mapController?.animateCamera(CameraUpdate.newLatLngBounds(bounds, 60.0));
  }

  @override
  Widget build(BuildContext context) {
    final initialPosition = _routeDetail != null
        ? LatLng(_routeDetail!.destLat, _routeDetail!.destLon)
        : const LatLng(12.9716, 77.5946); // Default Bangalore Center

    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Emergency Route Navigation',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        backgroundColor: const Color(0xFF1E293B),
        foregroundColor: Colors.white,
        elevation: 1,
        actions: [
          IconButton(icon: const Icon(Icons.refresh), onPressed: _loadRoute),
        ],
      ),
      body: Stack(
        children: [
          // Google Map Widget
          if (!AppConfig.mapsEnabled || AppConfig.mockMode)
            Container(
              color: const Color(0xFFEFF6FF),
              alignment: Alignment.center,
              padding: const EdgeInsets.all(24),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.route, size: 72, color: Color(0xFFD32F2F)),
                  const SizedBox(height: 16),
                  const Text(
                    'Demo route preview',
                    style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 12),
                  if (_routeDetail != null) ...[
                    Text(
                      'Donor: ${_routeDetail!.originLat}, ${_routeDetail!.originLon}',
                    ),
                    const Icon(Icons.arrow_downward),
                    Text(
                      'Hospital: ${_routeDetail!.destLat}, ${_routeDetail!.destLon}',
                    ),
                  ],
                ],
              ),
            )
          else
            GoogleMap(
              initialCameraPosition: CameraPosition(
                target: initialPosition,
                zoom: 12.0,
              ),
              markers: _markers,
              polylines: _polylines,
              myLocationEnabled: false,
              zoomControlsEnabled: false,
              mapToolbarEnabled: false,
              onMapCreated: (controller) {
                _mapController = controller;
                if (_routeDetail != null) {
                  _fitBounds(
                    LatLng(_routeDetail!.originLat, _routeDetail!.originLon),
                    LatLng(_routeDetail!.destLat, _routeDetail!.destLon),
                  );
                }
              },
            ),

          // Loading overlay
          if (_isLoading)
            Container(
              color: Colors.white.withValues(alpha: 0.7),
              child: const Center(
                child: CircularProgressIndicator(color: Color(0xFFD32F2F)),
              ),
            ),

          // Error overlay
          if (_errorMessage != null)
            Positioned(
              top: 16,
              left: 16,
              right: 16,
              child: Card(
                color: Colors.red.shade50,
                child: Padding(
                  padding: const EdgeInsets.all(12.0),
                  child: Text(
                    'Route calculation failed: $_errorMessage\n(Check Google Maps Routes API quota/key)',
                    style: TextStyle(color: Colors.red.shade900, fontSize: 13),
                  ),
                ),
              ),
            ),

          // Bottom Route Info Sheet
          if (_routeDetail != null)
            Positioned(
              bottom: 0,
              left: 0,
              right: 0,
              child: Container(
                padding: const EdgeInsets.all(20.0),
                decoration: const BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black12,
                      blurRadius: 10,
                      offset: Offset(0, -2),
                    ),
                  ],
                ),
                child: SafeArea(
                  top: false,
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                _routeDetail!.isSuccess
                                    ? '${_routeDetail!.durationMinutes ?? "--"} min'
                                    : 'Route Infeasible',
                                style: TextStyle(
                                  fontSize: 26,
                                  fontWeight: FontWeight.bold,
                                  color: _routeDetail!.isSuccess
                                      ? const Color(0xFFD32F2F)
                                      : Colors.grey,
                                ),
                              ),
                              Text(
                                _routeDetail!.distanceKm != null
                                    ? '${_routeDetail!.distanceKm} km via road network'
                                    : 'Distance unavailable',
                                style: const TextStyle(
                                  color: Color(0xFF64748B),
                                  fontSize: 13,
                                ),
                              ),
                            ],
                          ),
                          Chip(
                            label: Text(
                              _routeDetail!.provider.toUpperCase(),
                              style: const TextStyle(fontSize: 11),
                            ),
                            backgroundColor: Colors.blue.shade50,
                            labelStyle: TextStyle(
                              color: Colors.blue.shade900,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 12),
                      const Divider(),
                      const SizedBox(height: 6),
                      Text(
                        'Destination: ${widget.hospitalName}',
                        style: const TextStyle(
                          fontWeight: FontWeight.w600,
                          fontSize: 14,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'Emergency Request ID: ${widget.requestId}',
                        style: const TextStyle(
                          fontSize: 12,
                          color: Color(0xFF94A3B8),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}
