# NpuStudio Companion (Android)

A thin WebView shell. The whole UI comes from the desktop (`webapp/`, served at `http://<desktop>:47860/app/`), so UI changes only need a desktop update, not a new APK.

- **Pairing:** type the address and key, or scan the QR from `Show-PairingQR.bat`. The format is `npustudio://pair?host=&remote=&port=&key=` (`remote` = optional Tailscale address; the app tries home first and falls back to it).
- **Network rule:** the app only connects to private LAN addresses, Tailscale `100.64.0.0/10` and `*.ts.net`. Any other host is refused in `HostRules.kt` and by request interception.
- **JS bridge:** `window.NpuShell` with `shellVersion()`, `pickImage(cbId)` (the result comes back through `window.npuCallback(cbId, base64|null)`), `saveImage(b64, name)` (saves to Pictures/NpuStudio), `share(b64, name)`, `notify(title, text)` and `unpair()`. Shell v2 adds `getConnection()`, `setConnection(home, remote, mode)`, `probe(cbId)`, `fallback()`, `rePair()` and `phoneAction(json, cbId)` (allowlisted actions in `PhoneActions.kt`). Shell v3 adds `getBackground()` / `setBackground(on)`. Shell v4 (Heat tab, `Heat.kt`) adds `heatList()`, `heatStop(package)`, `heatAppInfo(package)` and `openUsageAccess()`. Shell v5 adds memory, CPU, and network fields on `heatList()` where Android allows them (`resources: true`; null fields mean N/A).
- **Heat:** needs Usage access granted once in system settings (the Heat tab opens that page). The list is recent UsageStats, not a CPU or heat meter. Memory is PSS for processes this app can still see (other apps are N/A on Android 10+). CPU percent needs a second refresh and is limited the same way. Network is Wi-Fi plus mobile bytes for the last 6 hours via NetworkStatsManager. Stop uses `killBackgroundProcesses` when Android allows it; otherwise it opens that app's App info screen so the user can tap Force stop. No root and no device admin. Maps / navigation, Tailscale, NpuStudio, the dialer and SMS app, the home screen, the keyboard, and system core are refused.
- **Build:**
  ```
  ./gradlew assembleRelease
  ```
  The output is `app/build/outputs/apk/release/app-release.apk`, renamed to NpuStudio-Companion.apk.
- **Signing:** by default the release is signed with `~/.android/debug.keystore`, which is enough for sideloading. Set `NPU_KEYSTORE`, `NPU_KEYSTORE_PASS`, `NPU_KEY_ALIAS` and `NPU_KEY_PASS` to use your own key. An update must be signed with the same key, or Android makes you uninstall first.
- **Bump `HostRules.SHELL_VERSION`** when the bridge changes, and set `latest_shell` (the UI shows an update banner below it) or `min_shell` in `webapp/version.json`. Copy the APK to `webapp/NpuStudio-Companion.apk` on the desktop so the banner's link can download it (git-ignored).
