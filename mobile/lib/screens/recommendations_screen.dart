// lib/screens/recommendations_screen.dart
// Displays ranked donor candidates computed by the Python backend's UDI engine & routing.

import 'dart:async';
import 'package:flutter/material.dart';
import '../services/api_service.dart';
import '../models/models.dart';
import 'map_route_screen.dart';

class RecommendationsScreen extends StatefulWidget {
  final ApiService apiService;
  final EmergencyRequestModel emergencyRequest;

  const RecommendationsScreen({
    super.key,
    required this.apiService,
    required this.emergencyRequest,
  });

  @override
  State<RecommendationsScreen> createState() => _RecommendationsScreenState();
}

class _RecommendationsScreenState extends State<RecommendationsScreen> {
  List<CandidateRecommendationModel> _candidates = [];
  bool _isLoading = true;
  Map<String, dynamic> _summary = {};
  int _page = 1;
  int _total = 0;
  bool _hasNext = false;
  String _view = 'all';
  Timer? _poller;
  bool _fetching = false;
  String? _errorMessage;
  final Set<String> _confirming = {};

  @override
  void initState() {
    super.initState();
    _loadRecommendations();
    _poller = Timer.periodic(
      const Duration(seconds: 5),
      (_) => _loadRecommendations(silent: true),
    );
  }

  @override
  void dispose() {
    _poller?.cancel();
    super.dispose();
  }

  Future<void> _loadRecommendations({bool silent = false}) async {
    if (_fetching || !mounted) return;
    _fetching = true;
    final requestedPage = _page;
    final requestedView = _view;
    if (!silent) {
      setState(() {
        _isLoading = true;
        _errorMessage = null;
      });
    }
    try {
      final result = await widget.apiService.getRecruitmentPage(
        widget.emergencyRequest.requestId,
        page: requestedPage,
        view: requestedView,
      );
      if (!mounted || _page != requestedPage || _view != requestedView) return;
      setState(() {
        _candidates = result.candidates;
        _summary = result.summary;
        _total = result.total;
        _hasNext = result.hasNext;
        _errorMessage = null;
      });
    } catch (error) {
      if (mounted) setState(() => _errorMessage = '$error');
    } finally {
      _fetching = false;
      if (mounted) setState(() => _isLoading = false);
    }
  }

  Future<void> _handleConfirmDonation(String donorId) async {
    if (_confirming.contains(donorId)) return;
    setState(() => _confirming.add(donorId));
    try {
      final res = await widget.apiService.completeDonation(
        donorId,
        widget.emergencyRequest.requestId,
      );
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'Donation recorded! Donor deferred for ${res['deferral_days'] ?? 90} days.',
          ),
          backgroundColor: Colors.green,
        ),
      );
      _loadRecommendations();
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: $e'), backgroundColor: Colors.red),
      );
    } finally {
      if (mounted) setState(() => _confirming.remove(donorId));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8FAFC),
      appBar: AppBar(
        title: const Text('Donor Recruitment'),
        backgroundColor: const Color(0xFF1E293B),
        foregroundColor: Colors.white,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loadRecommendations,
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : Column(
              children: [
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(16),
                  color: Colors.blue.shade50,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Batch ${_summary['batch_number'] ?? 0} • Up to 10 invitations',
                        style: const TextStyle(fontWeight: FontWeight.bold),
                      ),
                      Text(
                        'Accepted: ${_summary['accepted_count'] ?? 0}  |  Awaiting response: ${_summary['pending_count'] ?? 0}',
                      ),
                      Text(
                        'Still needed: ${_summary['remaining_needed'] ?? 0}  |  Completed: ${_summary['units_fulfilled'] ?? 0}/${_summary['units_required'] ?? widget.emergencyRequest.unitsRequired}',
                      ),
                      Text('Status: ${_summary['status'] ?? 'READY'}'),
                      Text(
                        'In-app invitations • Routing: ${_summary['routing_provider'] ?? 'synthetic_mock'}',
                      ),
                      if (_summary['next_batch_at'] != null)
                        Text(
                          'Next response deadline: ${DateTime.parse(_summary['next_batch_at']).toLocal()}',
                        ),
                      const SizedBox(height: 6),
                      const Text(
                        'Sorted by road travel time; Multiplicative UDI breaks ties. Accepted places are reserved; donations count after confirmation.',
                        style: TextStyle(fontSize: 12),
                      ),
                    ],
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.all(8),
                  child: Wrap(
                    spacing: 12,
                    children: [
                      ChoiceChip(
                        label: const Text('All candidates'),
                        selected: _view == 'all',
                        onSelected: _fetching
                            ? null
                            : (_) {
                                setState(() {
                                  _view = 'all';
                                  _page = 1;
                                });
                                _loadRecommendations();
                              },
                      ),
                      ChoiceChip(
                        label: const Text('Accepted donors'),
                        selected: _view == 'accepted',
                        onSelected: _fetching
                            ? null
                            : (_) {
                                setState(() {
                                  _view = 'accepted';
                                  _page = 1;
                                });
                                _loadRecommendations();
                              },
                      ),
                    ],
                  ),
                ),
                if (_errorMessage != null)
                  Padding(
                    padding: const EdgeInsets.all(8),
                    child: Text(
                      _errorMessage!,
                      style: const TextStyle(color: Colors.red),
                    ),
                  ),
                Expanded(
                  child: _candidates.isEmpty
                      ? Center(
                          child: Text(
                            _view == 'accepted'
                                ? 'No donors have accepted yet.'
                                : 'No reachable matching candidates available.',
                          ),
                        )
                      : ListView.builder(
                          padding: const EdgeInsets.all(16),
                          itemCount: _candidates.length,
                          itemBuilder: (_, index) =>
                              _buildCandidateCard(_candidates[index]),
                        ),
                ),
                SafeArea(
                  top: false,
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      TextButton(
                        onPressed: _page > 1 && !_fetching
                            ? () {
                                setState(() => _page--);
                                _loadRecommendations();
                              }
                            : null,
                        child: const Text('Previous 10'),
                      ),
                      Text('Page $_page • $_total donors'),
                      TextButton(
                        onPressed: _hasNext && !_fetching
                            ? () {
                                setState(() => _page++);
                                _loadRecommendations();
                              }
                            : null,
                        child: const Text('Next 10'),
                      ),
                    ],
                  ),
                ),
              ],
            ),
    );
  }

  Widget _buildCandidateCard(CandidateRecommendationModel c) {
    final etaStr = c.travelTimeMinutes != null
        ? '${c.travelTimeMinutes} min'
        : 'Route Infeasible';

    return Card(
      elevation: 2,
      margin: const EdgeInsets.only(bottom: 12),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Row(
              children: [
                // Rank Circle
                Container(
                  width: 36,
                  height: 36,
                  decoration: BoxDecoration(
                    color: c.rank == 1
                        ? const Color(0xFFD32F2F)
                        : const Color(0xFF1E293B),
                    shape: BoxShape.circle,
                  ),
                  child: Center(
                    child: Text(
                      '#${c.rank}',
                      style: const TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                        fontSize: 14,
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        c.donorName,
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        '${c.donorId} | Blood: ${c.bloodType} | Alerts: ${c.recentAlerts}',
                        style: const TextStyle(
                          fontSize: 13,
                          color: Color(0xFF64748B),
                        ),
                      ),
                    ],
                  ),
                ),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    Text(
                      'UDI Score',
                      style: const TextStyle(
                        fontSize: 11,
                        color: Color(0xFF94A3B8),
                      ),
                    ),
                    Text(
                      c.udiScore.toStringAsFixed(4),
                      style: const TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 16,
                        color: Color(0xFF1E293B),
                      ),
                    ),
                  ],
                ),
              ],
            ),
            const SizedBox(height: 12),
            Text(
              'Batch ${c.batchNumber} • ${c.invitationStatus}. Confirm after donation is completed.',
            ),
            const Divider(),
            const SizedBox(height: 6),

            // Travel Time & Feasibility Badges
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                Row(
                  children: [
                    Icon(
                      Icons.directions_car,
                      size: 16,
                      color: c.isFeasible ? Colors.green.shade700 : Colors.red,
                    ),
                    const SizedBox(width: 6),
                    Text(
                      'ETA: $etaStr',
                      style: TextStyle(
                        fontWeight: FontWeight.w600,
                        color: c.isFeasible
                            ? Colors.green.shade800
                            : Colors.red.shade800,
                      ),
                    ),
                  ],
                ),
                Row(
                  children: [
                    TextButton(
                      child: const Text(
                        'View Map',
                        style: TextStyle(fontSize: 13),
                      ),
                      onPressed: () {
                        Navigator.of(context).push(
                          MaterialPageRoute(
                            builder: (_) => MapRouteScreen(
                              apiService: widget.apiService,
                              donorId: c.donorId,
                              requestId: widget.emergencyRequest.requestId,
                              hospitalName:
                                  widget.emergencyRequest.hospitalName,
                            ),
                          ),
                        );
                      },
                    ),
                    ElevatedButton(
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.green.shade700,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(
                          horizontal: 10,
                          vertical: 6,
                        ),
                      ),
                      onPressed:
                          c.isEligible &&
                              c.isFeasible &&
                              c.invitationStatus == 'ACCEPTED' &&
                              !_confirming.contains(c.donorId)
                          ? () => _handleConfirmDonation(c.donorId)
                          : null,
                      child: const Text(
                        'Confirm Donation',
                        style: TextStyle(fontSize: 12),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
