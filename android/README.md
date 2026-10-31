# NpuStudio Companion (Android)

A thin WebView shell. The whole UI comes from the desktop (`webapp/`, served at `http://<desktop>:47860/app/`), so UI changes only need a desktop update, not a new APK.

- **Pairing:** type the address and key, or scan the QR from `Show-PairingQR.bat`. The format is `npustudio://pair?host=&port=&key=`.
- **Network rule:** the app only connects to private LAN addresses, Tailscale `100.64.0.0/10` and `*.ts.net`. Any other host is refused in `HostRules.kt` and by request interception.
- **JS bridge:** `window.NpuShell` with `shellVersion()`, `pickImage(cbId)` (the result comes back through `window.npuCallback(cbId, base64|null)`), `saveImage(b64, name)` (saves to Pictures/NpuStudio), `share(b64, name)`, `notify(title, text)` and `unpair()`.
- **Build:**
  ```
  ./gradlew assembleRelease
  ```
  The output is `app/build/outputs/apk/release/app-release.apk`, renamed to NpuStudio-Companion.apk.
- **Signing:** by default the release is signed with `~/.android/debug.keystore`, which is enough for sideloading. Set `NPU_KEYSTORE`, `NPU_KEYSTORE_PASS`, `NPU_KEY_ALIAS` and `NPU_KEY_PASS` to use your own key. An update must be signed with the same key, or Android makes you uninstall first.
- **Bump `HostRules.SHELL_VERSION`** when the bridge changes, and raise `min_shell` in `webapp/version.json` when the UI needs the new bridge.
