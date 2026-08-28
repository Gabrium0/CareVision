import UIKit

/// Classic AppDelegate window setup — deliberately no scene manifest, so the app
/// runs unchanged on iPadOS 13.3 (pre-scene-lifecycle) through current iPadOS.
@UIApplicationMain
final class AppDelegate: UIResponder, UIApplicationDelegate {
    var window: UIWindow?

    func application(_ application: UIApplication,
                     didFinishLaunchingWithOptions launchOptions:
                     [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        let window = UIWindow(frame: UIScreen.main.bounds)
        let nav = UINavigationController(rootViewController: PairingViewController())
        window.rootViewController = nav
        window.makeKeyAndVisible()
        self.window = window
        return true
    }
}
