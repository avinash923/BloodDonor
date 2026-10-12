# Blood Donor Flutter App

See the project README at ../README.md for the demo workflow, accounts, configuration and startup commands.

```powershell
flutter pub get
flutter run --dart-define=MOCK_MODE=true
```

Offline mode retains requests, invitation responses and donations for the current app session. Leave it disabled to use the Python backend. Android emulator connections default to http://10.0.2.2:8000; browser builds use the page origin. Override with --dart-define=API_BASE_URL=... .

Maps are disabled by default so route previews work without keys. Configure the platform SDK before enabling --dart-define=MAPS_ENABLED=true. Backend Routes credentials must stay on the backend.

```powershell
flutter analyze
flutter test
flutter build web --no-web-resources-cdn
```
