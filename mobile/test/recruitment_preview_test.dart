// Optional visual verification using snapshots captured from the live local API.
import 'dart:convert';
import 'dart:io';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:blood_donor_app/models/models.dart';
import 'package:blood_donor_app/screens/recommendations_screen.dart';
import 'package:blood_donor_app/services/api_service.dart';

class CapturedApi extends ApiService {
  final String directory;
  CapturedApi(this.directory);

  @override
  Future<RecruitmentPage> getRecruitmentPage(
    String requestId, {
    int page = 1,
    String view = 'all',
  }) async {
    final filename = view == 'accepted'
        ? 'accepted_donors.json'
        : 'first_batch.json';
    return RecruitmentPage.fromJson(
      jsonDecode(File('$directory/$filename').readAsStringSync()),
    );
  }
}

void main() {
  const directory = String.fromEnvironment('RECRUITMENT_PREVIEW_DIR');
  const fontFile = String.fromEnvironment('RECRUITMENT_PREVIEW_FONT');
  const iconsFile = String.fromEnvironment('RECRUITMENT_PREVIEW_ICONS');
  testWidgets(
    'Render actual live API batch and accepted donor results',
    (tester) async {
      tester.view.physicalSize = const Size(1100, 1000);
      tester.view.devicePixelRatio = 1;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);
      final loader = FontLoader('PreviewFont')
        ..addFont(
          Future.value(ByteData.sublistView(File(fontFile).readAsBytesSync())),
        );
      await loader.load();
      if (iconsFile.isNotEmpty) {
        await (FontLoader('MaterialIcons')..addFont(
              Future.value(
                ByteData.sublistView(File(iconsFile).readAsBytesSync()),
              ),
            ))
            .load();
      }
      debugDisableShadows = false;
      addTearDown(() => debugDisableShadows = true);
      final request = EmergencyRequestModel.fromJson(
        jsonDecode(File('$directory/request.json').readAsStringSync()),
      );
      final boundaryKey = GlobalKey();
      await tester.pumpWidget(
        RepaintBoundary(
          key: boundaryKey,
          child: MaterialApp(
            debugShowCheckedModeBanner: false,
            theme: ThemeData(
              fontFamily: 'PreviewFont',
              useMaterial3: true,
              colorScheme: ColorScheme.fromSeed(
                seedColor: const Color(0xFFD32F2F),
              ),
            ),
            home: RecommendationsScreen(
              apiService: CapturedApi(directory),
              emergencyRequest: request,
            ),
          ),
        ),
      );
      await tester.pumpAndSettle();
      expect(find.text('Batch 1 • Up to 10 invitations'), findsOneWidget);

      Future<void> capture(String filename) async {
        final boundary =
            boundaryKey.currentContext!.findRenderObject()
                as RenderRepaintBoundary;
        await tester.runAsync(() async {
          final image = await boundary.toImage(pixelRatio: 1);
          final data = await image.toByteData(format: ui.ImageByteFormat.png);
          File(
            '$directory/$filename',
          ).writeAsBytesSync(data!.buffer.asUint8List());
          image.dispose();
        });
      }

      await capture('batch_1_screen.png');
      await tester.tap(find.text('Accepted donors'));
      await tester.pumpAndSettle();
      expect(find.text('Accepted: 3  |  Awaiting response: 0'), findsOneWidget);
      expect(find.text('Status: TARGET_RESERVED'), findsOneWidget);
      await capture('accepted_donors_screen.png');
      expect(tester.takeException(), isNull);
      await tester.pumpWidget(const SizedBox());
      debugDisableShadows = true;
    },
    skip: directory.isEmpty || fontFile.isEmpty,
  );
}
