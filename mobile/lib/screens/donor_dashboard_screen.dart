// lib/screens/donor_dashboard_screen.dart
// Donor-facing screen displaying profile, temporal eligibility, and emergency alerts.

import 'dart:async';
import 'package:flutter/material.dart';
import '../services/api_service.dart';
import '../models/models.dart';
import 'map_route_screen.dart';
import 'login_screen.dart';
import '../services/donor_location_service.dart';

class DonorDashboardScreen extends StatefulWidget {
  final ApiService apiService;

  const DonorDashboardScreen({super.key, required this.apiService});

  @override
  State<DonorDashboardScreen> createState() => _DonorDashboardScreenState();
}

class _DonorDashboardScreenState extends State<DonorDashboardScreen> {
  UserProfile? _profile;
  List<InvitationModel> _invitations = [];
  bool _isLoading = true;
  String? _errorMessage;
  Timer? _poller;
  bool _fetching = false;
  bool _sharing = false;
  bool _consent = false;

  @override
  void initState() {
    super.initState();
    _loadData();
    _poller = Timer.periodic(
      const Duration(seconds: 5),
      (_) => _loadData(silent: true),
    );
  }

  @override
  void dispose() {
    _poller?.cancel();
    super.dispose();
  }

  Future<void> _loadData({bool silent = false}) async {
    if (_fetching || !mounted) return;
    _fetching = true;
    if (!silent) {
      setState(() {
        _isLoading = true;
        _errorMessage = null;
      });
    }

    try {
      final prof = await widget.apiService.getDonorProfile();
      final invites = await widget.apiService.getDonorInvitations();
      if (!mounted) return;
      setState(() {
        _profile = prof;
        _invitations = invites;
      });
    } catch (e) {
      if (!mounted) return;
      setState(
        () => _errorMessage = e.toString().replaceAll('Exception:', '').trim(),
      );
    } finally {
      _fetching = false;
      if (mounted) setState(() => _isLoading = false);
    }
  }

  Future<void> _handleResponse(InvitationModel inv, String action) async {
    try {
      if (action == 'ACCEPT' && _profile?.accountSource == 'registered') {
        if (_profile?.locationConsent != true && !_consent) {
          throw Exception('Share your current location before accepting.');
        }
        await DonorLocationService.shareCurrent(widget.apiService);
      }
      final ok = await widget.apiService.respondToInvitation(
        inv.invitationId,
        action,
      );
      if (!ok) {
        throw Exception(
          'Invitation could not be updated. Refresh and try again.',
        );
      }
      if (ok && mounted) {
        setState(() {
          inv.status = action == 'ACCEPT' ? 'ACCEPTED' : 'DECLINED';
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Invitation ${action.toLowerCase()}ed!'),
            backgroundColor: action == 'ACCEPT'
                ? Colors.green
                : Colors.grey.shade700,
          ),
        );
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Failed to update: $e'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  void _handleLogout() {
    widget.apiService.clearAuth();
    Navigator.of(context).pushReplacement(
      MaterialPageRoute(
        builder: (_) => LoginScreen(apiService: widget.apiService),
      ),
    );
  }

  Future<void> _shareLocation({bool revoke = false}) async {
    setState(() => _sharing = true);
    try {
      if (revoke) {
        await widget.apiService.updateDonorLocation({'consent': false});
        if (mounted) setState(() => _consent = false);
      } else {
        await DonorLocationService.shareCurrent(widget.apiService);
      }
      // A poll may already be active; the following poll also refreshes the card.
      await _loadData(silent: true);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              revoke
                  ? 'Location removed. You are unavailable for matching.'
                  : 'Location saved. Nearby matching is enabled for up to 15 minutes.',
            ),
          ),
        );
      }
    } catch (error) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('$error')));
      }
    } finally {
      if (mounted) setState(() => _sharing = false);
    }
  }

  Widget _buildLocationCard() {
    final profile = _profile!;
    if (profile.accountSource != 'registered') {
      return const Card(
        child: Padding(
          padding: EdgeInsets.all(16),
          child: Text(
            'Demo account: saved coordinates and travel estimates are simulated. Register a donor account to share your device location.',
          ),
        ),
      );
    }
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              'Location for nearby requests',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            Text(
              profile.locationReady
                  ? 'Location ready for matching'
                  : 'Share a fresh location to appear in nearby matching',
            ),
            if (profile.locationCapturedAt != null)
              Text(
                'Last captured: ${profile.locationCapturedAt}\nAccuracy: ${profile.locationAccuracyMeters?.toStringAsFixed(0)} metres',
              ),
            const Text(
              'We save your latest position, accuracy and capture time. It expires from matching after 15 minutes. This does not continuously track you.',
            ),
            CheckboxListTile(
              contentPadding: EdgeInsets.zero,
              value: _consent || profile.locationConsent,
              title: const Text(
                'I agree to share my location for donor matching',
              ),
              onChanged: _sharing
                  ? null
                  : (v) {
                      if (v == false && profile.locationConsent) {
                        _shareLocation(revoke: true);
                      } else {
                        setState(() => _consent = v ?? false);
                      }
                    },
            ),
            FilledButton.icon(
              onPressed: _sharing || !(_consent || profile.locationConsent)
                  ? null
                  : () => _shareLocation(),
              icon: const Icon(Icons.my_location),
              label: Text(
                _sharing ? 'Updating…' : 'Share / refresh current location',
              ),
            ),
            if (profile.locationConsent)
              TextButton(
                onPressed: _sharing ? null : () => _shareLocation(revoke: true),
                child: const Text('Stop sharing and remove saved location'),
              ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8FAFC),
      appBar: AppBar(
        title: const Text(
          'Donor Portal',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        backgroundColor: const Color(0xFFD32F2F),
        foregroundColor: Colors.white,
        elevation: 1,
        actions: [
          IconButton(icon: const Icon(Icons.refresh), onPressed: _loadData),
          IconButton(icon: const Icon(Icons.logout), onPressed: _handleLogout),
        ],
      ),
      body: _isLoading
          ? const Center(
              child: CircularProgressIndicator(color: Color(0xFFD32F2F)),
            )
          : _errorMessage != null
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(Icons.error_outline, size: 48, color: Colors.red),
                  const SizedBox(height: 12),
                  Text(
                    _errorMessage!,
                    style: const TextStyle(color: Colors.red),
                  ),
                  const SizedBox(height: 16),
                  ElevatedButton(
                    onPressed: _loadData,
                    child: const Text('Retry'),
                  ),
                ],
              ),
            )
          : RefreshIndicator(
              onRefresh: _loadData,
              child: SingleChildScrollView(
                physics: const AlwaysScrollableScrollPhysics(),
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    // Profile & Eligibility Card
                    _buildProfileCard(),
                    const SizedBox(height: 16),
                    _buildLocationCard(),
                    const SizedBox(height: 16),

                    // Active Emergency Alerts Header
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        const Text(
                          'Emergency Invitations',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                            color: Color(0xFF1E293B),
                          ),
                        ),
                        Badge(
                          label: Text('${_invitations.length}'),
                          backgroundColor: const Color(0xFFD32F2F),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),

                    // Invitation List
                    if (_invitations.isEmpty)
                      Card(
                        elevation: 0,
                        color: Colors.white,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: const Padding(
                          padding: EdgeInsets.all(28.0),
                          child: Center(
                            child: Text(
                              'No active emergency recruitment alerts.',
                              style: TextStyle(color: Color(0xFF64748B)),
                            ),
                          ),
                        ),
                      )
                    else
                      ..._invitations.map((inv) => _buildInvitationCard(inv)),
                  ],
                ),
              ),
            ),
    );
  }

  Widget _buildProfileCard() {
    if (_profile == null) return const SizedBox.shrink();

    final isEligible = _profile!.isTemporallyEligible;
    final bloodType = _profile!.bloodType ?? 'Unknown';

    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(18.0),
        child: Column(
          children: [
            Row(
              children: [
                // Blood Badge
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 14,
                    vertical: 10,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.red.shade600,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    bloodType,
                    style: const TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        _profile!.name,
                        style: const TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'Donor ID: ${_profile!.userId}',
                        style: const TextStyle(
                          fontSize: 12,
                          color: Color(0xFF64748B),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            const Divider(),
            const SizedBox(height: 8),

            // Clinical Eligibility Status
            Row(
              children: [
                Icon(
                  isEligible ? Icons.check_circle : Icons.timer,
                  color: isEligible ? Colors.green : Colors.amber.shade800,
                  size: 20,
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    isEligible
                        ? 'Clinically Eligible to Donate'
                        : 'In Temporary Deferral Interval',
                    style: TextStyle(
                      fontWeight: FontWeight.w600,
                      color: isEligible
                          ? Colors.green.shade800
                          : Colors.amber.shade900,
                    ),
                  ),
                ),
              ],
            ),
            if (!isEligible && _profile!.nextEligibleAt != null)
              Padding(
                padding: const EdgeInsets.only(left: 28.0, top: 4.0),
                child: Text(
                  'Next Eligible: ${_profile!.nextEligibleAt}',
                  style: const TextStyle(
                    fontSize: 12,
                    color: Color(0xFF64748B),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildInvitationCard(InvitationModel inv) {
    final isCritical = inv.urgency == 'CRITICAL';
    final isAccepted = inv.status == 'ACCEPTED';

    return Card(
      elevation: 2,
      margin: const EdgeInsets.only(bottom: 14),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 10,
                    vertical: 4,
                  ),
                  decoration: BoxDecoration(
                    color: isCritical
                        ? Colors.red.shade100
                        : Colors.amber.shade100,
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: Text(
                    inv.urgency,
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.bold,
                      color: isCritical
                          ? Colors.red.shade800
                          : Colors.amber.shade900,
                    ),
                  ),
                ),
                Text(
                  'Blood: ${inv.bloodTypeRequired}',
                  style: const TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 14,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Text(
              inv.hospitalName,
              style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 6),
            Text(
              'Request ID: ${inv.requestId}',
              style: const TextStyle(fontSize: 12, color: Color(0xFF64748B)),
            ),
            const SizedBox(height: 16),

            // Action / Status Buttons
            if (inv.status == 'PENDING')
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton(
                      onPressed: () => _handleResponse(inv, 'DECLINE'),
                      style: OutlinedButton.styleFrom(
                        foregroundColor: Colors.grey.shade700,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(8),
                        ),
                      ),
                      child: const Text('Decline'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: ElevatedButton(
                      onPressed: () => _handleResponse(inv, 'ACCEPT'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFFD32F2F),
                        foregroundColor: Colors.white,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(8),
                        ),
                      ),
                      child: const Text('Accept Alert'),
                    ),
                  ),
                ],
              )
            else
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Chip(
                    label: Text(inv.status),
                    backgroundColor: isAccepted
                        ? Colors.green.shade100
                        : Colors.grey.shade200,
                    labelStyle: TextStyle(
                      color: isAccepted
                          ? Colors.green.shade900
                          : Colors.grey.shade800,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  if (isAccepted)
                    ElevatedButton.icon(
                      icon: const Icon(Icons.directions_car, size: 16),
                      label: const Text('View Route'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFF1E293B),
                        foregroundColor: Colors.white,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(8),
                        ),
                      ),
                      onPressed: () {
                        Navigator.of(context).push(
                          MaterialPageRoute(
                            builder: (_) => MapRouteScreen(
                              apiService: widget.apiService,
                              donorId: inv.donorId,
                              requestId: inv.requestId,
                              hospitalName: inv.hospitalName,
                            ),
                          ),
                        );
                      },
                    ),
                ],
              ),
          ],
        ),
      ),
    );
  }
}
