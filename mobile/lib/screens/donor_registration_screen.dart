import 'package:flutter/material.dart';
import '../services/api_service.dart';
import 'donor_dashboard_screen.dart';

class DonorRegistrationScreen extends StatefulWidget {
  final ApiService apiService;
  const DonorRegistrationScreen({super.key, required this.apiService});
  @override
  State<DonorRegistrationScreen> createState() =>
      _DonorRegistrationScreenState();
}

class _DonorRegistrationScreenState extends State<DonorRegistrationScreen> {
  final _form = GlobalKey<FormState>();
  final _name = TextEditingController();
  final _username = TextEditingController();
  final _password = TextEditingController();
  String _blood = 'O+';
  String? _error;
  bool _busy = false;

  @override
  void dispose() {
    _name.dispose();
    _username.dispose();
    _password.dispose();
    super.dispose();
  }

  Future<void> _register() async {
    if (!_form.currentState!.validate()) return;
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      await widget.apiService.registerDonor({
        'name': _name.text.trim(),
        'username': _username.text.trim(),
        'password': _password.text,
        'blood_type': _blood,
      });
      if (!mounted) return;
      Navigator.of(context).pushAndRemoveUntil(
        MaterialPageRoute(
          builder: (_) => DonorDashboardScreen(apiService: widget.apiService),
        ),
        (_) => false,
      );
    } catch (error) {
      if (mounted) {
        setState(
          () => _error = error.toString().replaceFirst('Exception: ', ''),
        );
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('Register as a donor')),
    body: Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 500),
        child: ListView(
          padding: const EdgeInsets.all(24),
          children: [
            Form(
              key: _form,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  const Text(
                    'Create your donor account. You can choose to share your current location from your donor portal.',
                  ),
                  const SizedBox(height: 20),
                  TextFormField(
                    controller: _name,
                    decoration: const InputDecoration(labelText: 'Name'),
                    validator: (v) =>
                        (v?.trim().length ?? 0) < 2 ? 'Enter your name' : null,
                  ),
                  TextFormField(
                    controller: _username,
                    decoration: const InputDecoration(
                      labelText: 'Username',
                      helperText:
                          '4–40 lowercase letters, numbers or underscores',
                    ),
                    validator: (v) =>
                        RegExp(r'^[a-z][a-z0-9_]{3,39}$').hasMatch(v ?? '')
                        ? null
                        : 'Enter a valid username',
                  ),
                  TextFormField(
                    controller: _password,
                    obscureText: true,
                    decoration: const InputDecoration(
                      labelText: 'Password',
                      helperText: '10–128 characters',
                    ),
                    validator: (v) =>
                        (v?.length ?? 0) >= 10 && (v?.length ?? 0) <= 128
                        ? null
                        : 'Use 10–128 characters',
                  ),
                  const SizedBox(height: 16),
                  DropdownButtonFormField<String>(
                    initialValue: _blood,
                    decoration: const InputDecoration(
                      labelText: 'Blood group (self-reported)',
                    ),
                    items: ['O+', 'O-', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-']
                        .map((b) => DropdownMenuItem(value: b, child: Text(b)))
                        .toList(),
                    onChanged: (v) => setState(() => _blood = v!),
                  ),
                  const SizedBox(height: 16),
                  const Text(
                    'Donation eligibility and blood group must be confirmed by the collecting facility.',
                  ),
                  if (_error != null)
                    Padding(
                      padding: const EdgeInsets.only(top: 16),
                      child: Text(
                        _error!,
                        style: const TextStyle(color: Colors.red),
                      ),
                    ),
                  const SizedBox(height: 24),
                  FilledButton(
                    onPressed: _busy ? null : _register,
                    child: Text(
                      _busy ? 'Creating account…' : 'Create donor account',
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    ),
  );
}
