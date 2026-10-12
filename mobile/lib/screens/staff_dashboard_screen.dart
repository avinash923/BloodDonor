// lib/screens/staff_dashboard_screen.dart
// Hospital staff screen: emergency request creation, tracking & UDI recommendations.

import 'package:flutter/material.dart';
import '../services/api_service.dart';
import '../models/models.dart';
import 'recommendations_screen.dart';
import 'login_screen.dart';
import 'facility_picker.dart';

class StaffDashboardScreen extends StatefulWidget {
  final ApiService apiService;

  const StaffDashboardScreen({super.key, required this.apiService});

  @override
  State<StaffDashboardScreen> createState() => _StaffDashboardScreenState();
}

class _StaffDashboardScreenState extends State<StaffDashboardScreen> {
  List<EmergencyRequestModel> _requests = [];
  bool _isLoading = true;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _loadRequests();
  }

  Future<void> _loadRequests() async {
    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    try {
      final reqs = await widget.apiService.getEmergencyRequests();
      if (!mounted) return;
      setState(() => _requests = reqs);
    } catch (e) {
      if (!mounted) return;
      setState(
        () => _errorMessage = e.toString().replaceAll('Exception:', '').trim(),
      );
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  void _showCreateDialog() {
    final formKey = GlobalKey<FormState>();
    String selectedBloodType = 'O+';
    int units = 2;
    String selectedUrgency = 'CRITICAL';
    Map<String, dynamic>? facility;
    bool facilityConfirmed = false;

    showDialog(
      context: context,
      builder: (ctx) {
        return StatefulBuilder(
          builder: (context, setDialogState) {
            return AlertDialog(
              title: const Text('Create Emergency Request'),
              content: SingleChildScrollView(
                child: Form(
                  key: formKey,
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      OutlinedButton.icon(
                        icon: const Icon(Icons.local_hospital),
                        label: Text(
                          facility == null
                              ? 'Choose from blood-bank directory (optional)'
                              : facility!['name'],
                        ),
                        onPressed: () async {
                          final chosen = await showDialog<Map<String, dynamic>>(
                            context: context,
                            builder: (_) =>
                                FacilityPicker(apiService: widget.apiService),
                          );
                          if (!context.mounted || chosen == null) return;
                          setDialogState(() {
                            facility = chosen;
                            facilityConfirmed = false;
                          });
                        },
                      ),
                      if (facility != null) ...[
                        Text(
                          '${facility!['latitude']}, ${facility!['longitude']}',
                        ),
                        CheckboxListTile(
                          contentPadding: EdgeInsets.zero,
                          value: facilityConfirmed,
                          title: const Text(
                            'I checked this historical facility name and coordinates',
                          ),
                          onChanged: (v) => setDialogState(
                            () => facilityConfirmed = v ?? false,
                          ),
                        ),
                      ],
                      DropdownButtonFormField<String>(
                        initialValue: selectedBloodType,
                        decoration: const InputDecoration(
                          labelText: 'Blood Group Required',
                        ),
                        items:
                            ['O+', 'O-', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-']
                                .map(
                                  (b) => DropdownMenuItem(
                                    value: b,
                                    child: Text(b),
                                  ),
                                )
                                .toList(),
                        onChanged: (val) =>
                            setDialogState(() => selectedBloodType = val!),
                      ),
                      const SizedBox(height: 12),
                      DropdownButtonFormField<String>(
                        initialValue: selectedUrgency,
                        decoration: const InputDecoration(
                          labelText: 'Urgency Tier',
                        ),
                        items: [
                          DropdownMenuItem(
                            value: 'CRITICAL',
                            child: Text('CRITICAL (60 min)'),
                          ),
                          DropdownMenuItem(
                            value: 'URGENT',
                            child: Text('URGENT (120 min)'),
                          ),
                          DropdownMenuItem(
                            value: 'SCHEDULED',
                            child: Text('SCHEDULED (240 min)'),
                          ),
                        ],
                        onChanged: (val) =>
                            setDialogState(() => selectedUrgency = val!),
                      ),
                      const SizedBox(height: 12),
                      TextFormField(
                        initialValue: '2',
                        decoration: const InputDecoration(
                          labelText: 'Units Required (Pints)',
                        ),
                        keyboardType: TextInputType.number,
                        validator: (value) {
                          final count = int.tryParse(value ?? '');
                          return count == null || count < 1 || count > 100
                              ? 'Enter 1–100 whole units'
                              : null;
                        },
                        onChanged: (val) => units = int.tryParse(val) ?? 0,
                      ),
                    ],
                  ),
                ),
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.of(ctx).pop(),
                  child: const Text('Cancel'),
                ),
                ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFFD32F2F),
                    foregroundColor: Colors.white,
                  ),
                  onPressed: () async {
                    if (!formKey.currentState!.validate()) return;
                    if (facility != null && !facilityConfirmed) return;
                    Navigator.of(ctx).pop();
                    try {
                      await widget.apiService.createEmergencyRequest(
                        bloodTypeRequired: selectedBloodType,
                        unitsRequired: units,
                        urgency: selectedUrgency,
                        facilityId: facility?['facility_id'],
                        facilityConfirmed: facilityConfirmed,
                      );
                      if (!mounted) return;
                      _loadRequests();
                      if (mounted) {
                        ScaffoldMessenger.of(this.context).showSnackBar(
                          const SnackBar(
                            content: Text(
                              'Emergency request created & dispatch triggered!',
                            ),
                          ),
                        );
                      }
                    } catch (e) {
                      if (mounted) {
                        ScaffoldMessenger.of(this.context).showSnackBar(
                          SnackBar(
                            content: Text('Failed: $e'),
                            backgroundColor: Colors.red,
                          ),
                        );
                      }
                    }
                  },
                  child: const Text('Submit Request'),
                ),
              ],
            );
          },
        );
      },
    );
  }

  void _handleLogout() {
    widget.apiService.clearAuth();
    Navigator.of(context).pushReplacement(
      MaterialPageRoute(
        builder: (_) => LoginScreen(apiService: widget.apiService),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8FAFC),
      appBar: AppBar(
        title: const Text(
          'Hospital Emergency Dispatch',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        backgroundColor: const Color(0xFF1E293B),
        foregroundColor: Colors.white,
        elevation: 1,
        actions: [
          IconButton(icon: const Icon(Icons.refresh), onPressed: _loadRequests),
          IconButton(icon: const Icon(Icons.logout), onPressed: _handleLogout),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        backgroundColor: const Color(0xFFD32F2F),
        foregroundColor: Colors.white,
        icon: const Icon(Icons.add_alert),
        label: const Text('New Emergency'),
        onPressed: _showCreateDialog,
      ),
      body: _isLoading
          ? const Center(
              child: CircularProgressIndicator(color: Color(0xFFD32F2F)),
            )
          : _errorMessage != null
          ? Center(
              child: Text(
                _errorMessage!,
                style: const TextStyle(color: Colors.red),
              ),
            )
          : RefreshIndicator(
              onRefresh: _loadRequests,
              child: ListView.builder(
                padding: const EdgeInsets.all(16.0),
                itemCount: _requests.length,
                itemBuilder: (ctx, i) {
                  final req = _requests[i];
                  return _buildRequestCard(req);
                },
              ),
            ),
    );
  }

  Widget _buildRequestCard(EmergencyRequestModel req) {
    final isCritical = req.urgency == 'CRITICAL';

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
                    req.urgency,
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
                  'Demand: ${req.unitsFulfilled} / ${req.unitsRequired} Units',
                  style: const TextStyle(fontWeight: FontWeight.bold),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Text(
              '${req.bloodTypeRequired} Whole Blood Required',
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 4),
            Text(
              req.hospitalName,
              style: const TextStyle(color: Color(0xFF64748B)),
            ),
            const SizedBox(height: 2),
            Text(
              'ID: ${req.requestId}',
              style: const TextStyle(fontSize: 12, color: Color(0xFF94A3B8)),
            ),
            Text(
              'Status: ${req.status}',
              style: const TextStyle(fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 16),
            ElevatedButton.icon(
              icon: const Icon(Icons.people_outline, size: 18),
              label: const Text('View Recommendations & Recruits'),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF1E293B),
                foregroundColor: Colors.white,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
              ),
              onPressed: () async {
                await Navigator.of(context).push(
                  MaterialPageRoute(
                    builder: (_) => RecommendationsScreen(
                      apiService: widget.apiService,
                      emergencyRequest: req,
                    ),
                  ),
                );
                if (mounted) _loadRequests();
              },
            ),
          ],
        ),
      ),
    );
  }
}
