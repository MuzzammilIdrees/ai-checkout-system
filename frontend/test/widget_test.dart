import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:ai_checkout_frontend/app.dart';

void main() {
  testWidgets('App loads login screen', (WidgetTester tester) async {
    await tester.pumpWidget(
      const ProviderScope(
        child: CheckoutApp(),
      ),
    );

    // The app should show the login screen first (unauthenticated state)
    expect(find.text('Admin Login'), findsOneWidget);
  });
}
