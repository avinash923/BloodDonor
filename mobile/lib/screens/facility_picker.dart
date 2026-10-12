import 'package:flutter/material.dart';
import '../services/api_service.dart';

class FacilityPicker extends StatefulWidget {
  final ApiService apiService;
  const FacilityPicker({super.key, required this.apiService});
  @override
  State<FacilityPicker> createState() => _FacilityPickerState();
}

class _FacilityPickerState extends State<FacilityPicker> {
  final _query = TextEditingController(text: 'Karnataka');
  List<Map<String, dynamic>> _facilities = [];
  bool _busy = false;
  String? _error;
  @override
  void initState() {
    super.initState();
    _search();
  }

  @override
  void dispose() {
    _query.dispose();
    super.dispose();
  }

  Future<void> _search() async {
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final results = await widget.apiService.searchFacilities(
        _query.text.trim(),
      );
      if (mounted) setState(() => _facilities = results);
    } catch (e) {
      if (mounted) setState(() => _error = '$e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) => AlertDialog(
    title: const Text('Blood-bank directory'),
    content: SizedBox(
      width: 550,
      height: 450,
      child: Column(
        children: [
          const Text(
            'Historical September 2015 directory. Names, operations and coordinates need verification. No live stock data.',
          ),
          TextField(
            controller: _query,
            decoration: InputDecoration(
              labelText: 'Search facility, city, district or state',
              suffixIcon: IconButton(
                onPressed: _busy ? null : _search,
                icon: const Icon(Icons.search),
              ),
            ),
            onSubmitted: (_) => _search(),
          ),
          const SizedBox(height: 12),
          if (_error != null)
            Text(_error!, style: const TextStyle(color: Colors.red)),
          Expanded(
            child: _busy
                ? const Center(child: CircularProgressIndicator())
                : _facilities.isEmpty
                ? const Center(
                    child: Text('No results. Search another name or state.'),
                  )
                : ListView.builder(
                    itemCount: _facilities.length,
                    itemBuilder: (context, i) {
                      final facility = _facilities[i];
                      final hasCoordinates =
                          facility['latitude'] != null &&
                          facility['longitude'] != null;
                      return ListTile(
                        title: Text(facility['name']),
                        subtitle: Text(
                          '${facility['district']}, ${facility['state']}\n${hasCoordinates ? '${facility['latitude']}, ${facility['longitude']}' : 'Coordinates missing — unavailable for routing'}',
                        ),
                        enabled: hasCoordinates,
                        onTap: hasCoordinates
                            ? () => Navigator.pop(context, facility)
                            : null,
                      );
                    },
                  ),
          ),
          const Text(
            'Showing at most 100 matches. Narrow your search if needed.',
          ),
        ],
      ),
    ),
    actions: [
      TextButton(
        onPressed: () => Navigator.pop(context),
        child: const Text('Cancel'),
      ),
    ],
  );
}
