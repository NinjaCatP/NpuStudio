package com.ninjacatp.npustudio

import android.app.Activity
import android.app.NotificationManager
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.media.AudioManager
import android.net.Uri
import android.os.BatteryManager
import android.content.res.Resources
import android.os.Build
import android.provider.AlarmClock
import android.provider.Settings
import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.RingtoneManager
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.os.StatFs
import android.os.SystemClock
import android.provider.CalendarContract
import android.provider.ContactsContract
import android.provider.MediaStore
import android.speech.tts.TextToSpeech
import android.view.KeyEvent
import android.view.WindowManager
import java.io.File
import java.text.SimpleDateFormat
import java.util.Locale
import org.json.JSONObject

/** Allowlisted phone actions requested by the chat (shell v2). No arbitrary intents, no shell commands. */
object PhoneActions {
    val NAMES = setOf("open_app", "flashlight", "volume", "volume_change", "mute", "unmute", "alarm", "timer",
        "open_url", "web_search", "battery", "clipboard_set", "notify",
        // shell v3
        "brightness", "brightness_change", "brightness_auto", "dnd", "ring_mode", "rotation_lock", "settings_panel",
        "keep_screen_on", "camera", "navigate", "maps_search", "call", "sms", "calendar_event", "media", "storage", "network",
        "note", "notes", "speak", "app_settings", "find_phone", "stop_sound")

    fun run(c: Context, json: String, notify: (String, String) -> Unit): String = try {
        val o = JSONObject(json); val n = o.optString("name"); val g = o.optJSONObject("args") ?: JSONObject()
        if (n !in NAMES) "I can't do '$n' on the phone." else when (n) {
            "open_app" -> openApp(c, g.optString("app"))
            "flashlight" -> torch(c, g.opt("on").let { it == true || it.toString().lowercase() in setOf("true", "on", "1", "yes") })
            "volume" -> setVol(c, g.optInt("level", 50), g.optString("stream"))
            "volume_change" -> { val am = audio(c); val st = stream(g.optString("stream")).first; val mx = am.getStreamMaxVolume(st)
                setVol(c, am.getStreamVolume(st) * 100 / mx + g.optInt("delta", 10), g.optString("stream")) }
            "mute" -> { audio(c).adjustStreamVolume(AudioManager.STREAM_MUSIC, AudioManager.ADJUST_MUTE, AudioManager.FLAG_SHOW_UI); "Media muted." }
            "unmute" -> { audio(c).adjustStreamVolume(AudioManager.STREAM_MUSIC, AudioManager.ADJUST_UNMUTE, AudioManager.FLAG_SHOW_UI); "Media unmuted." }
            "alarm" -> { val h = g.optInt("hour", 7).coerceIn(0, 23); val m = g.optInt("minute", 0).coerceIn(0, 59)
                start(c, Intent(AlarmClock.ACTION_SET_ALARM).putExtra(AlarmClock.EXTRA_HOUR, h).putExtra(AlarmClock.EXTRA_MINUTES, m)
                    .putExtra(AlarmClock.EXTRA_MESSAGE, g.optString("message", "NpuStudio alarm").take(60)).putExtra(AlarmClock.EXTRA_SKIP_UI, true))
                "Alarm set for %d:%02d.".format(h, m) }
            "timer" -> { val s = (g.optDouble("minutes", 5.0) * 60).toInt().coerceIn(1, 86400)
                start(c, Intent(AlarmClock.ACTION_SET_TIMER).putExtra(AlarmClock.EXTRA_LENGTH, s)
                    .putExtra(AlarmClock.EXTRA_MESSAGE, g.optString("message", "NpuStudio timer").take(60)).putExtra(AlarmClock.EXTRA_SKIP_UI, true))
                "Timer set for " + (if (s < 60) "$s seconds." else "${s / 60} min" + (if (s % 60 > 0) " ${s % 60} s." else ".")) }
            "open_url" -> { var u = g.optString("url").trim(); if (!u.startsWith("http://") && !u.startsWith("https://")) u = "https://$u"
                val uri = Uri.parse(u); if (uri.scheme !in setOf("http", "https")) "Only web links." else { start(c, Intent(Intent.ACTION_VIEW, uri)); "Opened $u." } }
            "web_search" -> { val q = g.optString("query"); start(c, Intent(Intent.ACTION_VIEW, Uri.parse("https://www.bing.com/search?q=" + Uri.encode(q)))); "Searched for '$q'." }
            "battery" -> { val bm = c.getSystemService(BatteryManager::class.java)
                val pct = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
                "Battery is at $pct%" + (if (bm.isCharging) " and charging." else ".") }
            "clipboard_set" -> { val t = g.optString("text").take(20000)
                c.getSystemService(ClipboardManager::class.java).setPrimaryClip(ClipData.newPlainText("NpuStudio", t)); "Copied to the clipboard." }
            "brightness" -> setBright(c, g.optInt("level", 50))
            "brightness_change" -> setBright(c, curBright(c) + g.optInt("delta", 15))
            "brightness_auto" -> { val on = truthy(g.opt("on"))
                if (!canWrite(c)) askWrite(c, "turn auto-brightness ${if (on) "on" else "off"}") else {
                    Settings.System.putInt(c.contentResolver, Settings.System.SCREEN_BRIGHTNESS_MODE,
                        if (on) Settings.System.SCREEN_BRIGHTNESS_MODE_AUTOMATIC else Settings.System.SCREEN_BRIGHTNESS_MODE_MANUAL)
                    "Auto-brightness ${if (on) "on" else "off"}." } }
            "rotation_lock" -> { val lock = truthy(g.opt("on"))
                if (!canWrite(c)) askWrite(c, "change rotation lock") else {
                    Settings.System.putInt(c.contentResolver, Settings.System.ACCELEROMETER_ROTATION, if (lock) 0 else 1)
                    if (lock) "Rotation locked." else "Auto-rotate on." } }
            "dnd" -> { val nm = c.getSystemService(NotificationManager::class.java); val on = truthy(g.opt("on"))
                if (!nm.isNotificationPolicyAccessGranted) {
                    start(c, Intent(Settings.ACTION_NOTIFICATION_POLICY_ACCESS_SETTINGS))
                    "To switch Do Not Disturb I need 'Do Not Disturb access'. I opened that settings page: turn it on for NpuStudio Companion, then ask again."
                } else { nm.setInterruptionFilter(if (on) NotificationManager.INTERRUPTION_FILTER_PRIORITY else NotificationManager.INTERRUPTION_FILTER_ALL)
                    "Do Not Disturb ${if (on) "on" else "off"}." } }
            "ring_mode" -> { val m = g.optString("mode", "normal").lowercase(); val am = audio(c)
                val nm = c.getSystemService(NotificationManager::class.java)
                when (m) {
                    "vibrate" -> { am.ringerMode = AudioManager.RINGER_MODE_VIBRATE; "Ringer set to vibrate." }
                    "silent" -> if (!nm.isNotificationPolicyAccessGranted) { start(c, Intent(Settings.ACTION_NOTIFICATION_POLICY_ACCESS_SETTINGS))
                        "Silent mode needs 'Do Not Disturb access'. I opened that page: allow NpuStudio Companion, then ask again." }
                        else { am.ringerMode = AudioManager.RINGER_MODE_SILENT; "Ringer set to silent." }
                    else -> { am.ringerMode = AudioManager.RINGER_MODE_NORMAL; "Ringer set to normal (sound on)." }
                } }
            "settings_panel" -> { val k = g.optString("panel").lowercase()
                val (act, label) = when {
                    k.contains("wi") && Build.VERSION.SDK_INT >= 29 -> Settings.Panel.ACTION_WIFI to "Wi-Fi panel"
                    k.contains("wi") -> Settings.ACTION_WIFI_SETTINGS to "Wi-Fi settings"
                    k.contains("internet") || k.contains("data") -> (if (Build.VERSION.SDK_INT >= 29) Settings.Panel.ACTION_INTERNET_CONNECTIVITY else Settings.ACTION_WIRELESS_SETTINGS) to "Internet panel"
                    k.contains("nfc") -> (if (Build.VERSION.SDK_INT >= 29) Settings.Panel.ACTION_NFC else Settings.ACTION_NFC_SETTINGS) to "NFC panel"
                    k.contains("blue") -> Settings.ACTION_BLUETOOTH_SETTINGS to "Bluetooth settings"
                    k.contains("vol") || k.contains("sound") -> (if (Build.VERSION.SDK_INT >= 29) Settings.Panel.ACTION_VOLUME else Settings.ACTION_SOUND_SETTINGS) to "Volume panel"
                    k.contains("display") || k.contains("bright") -> Settings.ACTION_DISPLAY_SETTINGS to "Display settings"
                    k.contains("airplane") -> Settings.ACTION_AIRPLANE_MODE_SETTINGS to "Airplane mode settings"
                    k.contains("location") -> Settings.ACTION_LOCATION_SOURCE_SETTINGS to "Location settings"
                    k.contains("battery") -> Settings.ACTION_BATTERY_SAVER_SETTINGS to "Battery saver settings"
                    else -> Settings.ACTION_SETTINGS to "Settings"
                }
                start(c, Intent(act)); "Opened $label." + (if (label.startsWith("Wi-Fi") || label.startsWith("Bluetooth")) " (Android doesn't let apps switch it directly - tap the toggle.)" else "") }
            "keep_screen_on" -> { val on = truthy(g.opt("on")); val a = c as? Activity
                if (a == null) "Not available." else {
                if (on) a.window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) else a.window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                if (on) "Screen will stay on while NpuStudio is open." else "Screen can turn off normally again." } }
            "camera" -> { val video = g.optString("mode").lowercase().startsWith("vid")
                start(c, Intent(if (video) MediaStore.INTENT_ACTION_VIDEO_CAMERA else MediaStore.INTENT_ACTION_STILL_IMAGE_CAMERA))
                if (video) "Opened the camera (video)." else "Opened the camera." }
            "navigate" -> { val q = g.optString("place").trim(); if (q.isEmpty()) "Where to?" else {
                start(c, Intent(Intent.ACTION_VIEW, Uri.parse("google.navigation:q=" + Uri.encode(q))).let { i ->
                    if (i.resolveActivity(c.packageManager) != null) i else Intent(Intent.ACTION_VIEW, Uri.parse("geo:0,0?q=" + Uri.encode(q))) })
                "Directions to $q." } }
            "maps_search" -> { val q = g.optString("query"); start(c, Intent(Intent.ACTION_VIEW, Uri.parse("geo:0,0?q=" + Uri.encode(q)))); "Searching maps for '$q'." }
            "call" -> who(c, g.optString("who")) { num, name -> start(c, Intent(Intent.ACTION_DIAL, Uri.parse("tel:" + Uri.encode(num))))
                "Opened the dialer with $name - tap call to ring." }
            "sms" -> who(c, g.optString("who")) { num, name -> start(c, Intent(Intent.ACTION_SENDTO, Uri.parse("smsto:" + Uri.encode(num))).putExtra("sms_body", g.optString("text").take(1000)))
                "Opened a text to $name - check it and tap send." }
            "calendar_event" -> { val i = Intent(Intent.ACTION_INSERT, CalendarContract.Events.CONTENT_URI).putExtra(CalendarContract.Events.TITLE, g.optString("title", "Event").take(200))
                val t = parseWhen(g.optString("start")); val mins = g.optInt("minutes", 60)
                if (t != null) i.putExtra(CalendarContract.EXTRA_EVENT_BEGIN_TIME, t).putExtra(CalendarContract.EXTRA_EVENT_END_TIME, t + mins * 60000L)
                start(c, i); "Opened a new calendar event - check it and tap save." }
            "media" -> { val k = when (g.optString("cmd").lowercase()) { "play" -> KeyEvent.KEYCODE_MEDIA_PLAY; "pause" -> KeyEvent.KEYCODE_MEDIA_PAUSE
                    "next" -> KeyEvent.KEYCODE_MEDIA_NEXT; "previous", "prev", "back" -> KeyEvent.KEYCODE_MEDIA_PREVIOUS; else -> KeyEvent.KEYCODE_MEDIA_PLAY_PAUSE }
                val am = audio(c); val t = SystemClock.uptimeMillis()
                am.dispatchMediaKeyEvent(KeyEvent(t, t, KeyEvent.ACTION_DOWN, k, 0)); am.dispatchMediaKeyEvent(KeyEvent(t, t, KeyEvent.ACTION_UP, k, 0))
                "Media: " + g.optString("cmd", "play/pause") + "." }
            "storage" -> { val f = StatFs(Environment.getDataDirectory().path); val gb = 1e9
                "Storage: %.1f GB free of %.1f GB.".format(f.availableBytes / gb, f.totalBytes / gb) }
            "network" -> { val cm = c.getSystemService(ConnectivityManager::class.java); val nc = cm.getNetworkCapabilities(cm.activeNetwork)
                when { nc == null -> "No network connection."; nc.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) -> "Connected over Wi-Fi."
                    nc.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) -> "Connected over mobile data."; nc.hasTransport(NetworkCapabilities.TRANSPORT_VPN) -> "Connected through a VPN."
                    nc.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET) -> "Connected over Ethernet."; else -> "Connected." } }
            "note" -> { val t = g.optString("text").trim().take(2000); if (t.isEmpty()) "What should the note say?" else {
                File(c.filesDir, "notes.txt").appendText(SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).format(System.currentTimeMillis()) + "  " + t.replace("\n", " ") + "\n")
                "Noted: $t" } }
            "notes" -> { val f = File(c.filesDir, "notes.txt"); if (!f.exists()) "No notes yet." else "Your last notes:\n" + f.readLines().takeLast(g.optInt("count", 5).coerceIn(1, 20)).joinToString("\n") }
            "speak" -> { val t = g.optString("text").take(3000); speak(c, t); "Reading it aloud." }
            "app_settings" -> { val pkg = g.optString("app").trim().let { if (it.isEmpty()) c.packageName else findPkg(c, it) }
                if (pkg == null) "I couldn't find an app called '${g.optString("app")}'." else {
                start(c, Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:$pkg"))); "Opened app settings." } }
            "find_phone" -> { LinkService.start(c, LinkService.ACT_FIND); "Ringing loudly for up to 60 s - tap 'Found it - Stop' or say 'stop the sound'." }
            "stop_sound" -> { ring?.stop(); ring = null; LinkService.start(c, LinkService.ACT_STOP); "Stopped." }
            "notify" -> { notify(g.optString("title", "NpuStudio").take(80), g.optString("text").take(300)); "Notification sent." }
            else -> "Unknown action."
        }
    } catch (e: Exception) { "That didn't work on the phone: ${e.message}" }

    private fun truthy(v: Any?) = v == true || v.toString().lowercase() in setOf("true", "on", "1", "yes")
    private fun canWrite(c: Context) = Settings.System.canWrite(c)
    /** Opens "Modify system settings" for this app once; later calls just explain. */
    private fun askWrite(c: Context, what: String): String {
        val pf = c.getSharedPreferences("npu", Context.MODE_PRIVATE)
        val why = "To $what I need Android's 'Modify system settings' permission for NpuStudio Companion (it only changes brightness/rotation when you ask)."
        if (pf.getBoolean("asked_write", false)) return "$why Turn it on in Settings > Apps > NpuStudio Companion > Modify system settings."
        pf.edit().putBoolean("asked_write", true).apply()
        start(c, Intent(Settings.ACTION_MANAGE_WRITE_SETTINGS, Uri.parse("package:" + c.packageName)))
        return "$why I opened that page: allow it, then ask again."
    }
    private fun maxBright(): Int {
        val id = Resources.getSystem().getIdentifier("config_screenBrightnessSettingMaximum", "integer", "android")
        val v = if (id != 0) try { Resources.getSystem().getInteger(id) } catch (e: Exception) { 255 } else 255
        return if (v in 1..10000) v else 255
    }
    private fun curBright(c: Context): Int = try {
        Math.round(Settings.System.getInt(c.contentResolver, Settings.System.SCREEN_BRIGHTNESS) * 100f / maxBright()).coerceIn(0, 100)
    } catch (e: Exception) { 50 }
    private fun setBright(c: Context, pct: Int): String {
        val p = pct.coerceIn(0, 100)
        if (canWrite(c)) {
            val cr = c.contentResolver; val v = Math.round(p * maxBright() / 100f).coerceAtLeast(1)
            Settings.System.putInt(cr, Settings.System.SCREEN_BRIGHTNESS_MODE, Settings.System.SCREEN_BRIGHTNESS_MODE_MANUAL)
            Settings.System.putInt(cr, Settings.System.SCREEN_BRIGHTNESS, v)
            (c as? Activity)?.window?.let { w -> w.attributes = w.attributes.apply { screenBrightness = -1f } }  // follow the system value
            return "Screen brightness set to $p%."
        }
        (c as? Activity)?.window?.let { w -> w.attributes = w.attributes.apply { screenBrightness = (p / 100f).coerceAtLeast(0.01f) } }
        return "Set this app's window to $p% brightness only (the rest of the phone is unchanged). " + askWrite(c, "change the whole phone's brightness")
    }
    private var tts: TextToSpeech? = null
    private fun speak(c: Context, t: String) {
        tts?.let { it.speak(t, TextToSpeech.QUEUE_FLUSH, null, "npu"); return }
        var e: TextToSpeech? = null
        e = TextToSpeech(c.applicationContext) { st -> if (st == TextToSpeech.SUCCESS) { tts = e; e?.speak(t, TextToSpeech.QUEUE_FLUSH, null, "npu") } }
    }
    private var ring: android.media.Ringtone? = null
    private fun findPhone(c: Context) {
        val am = audio(c); am.setStreamVolume(AudioManager.STREAM_ALARM, am.getStreamMaxVolume(AudioManager.STREAM_ALARM), 0)
        val u = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM) ?: RingtoneManager.getDefaultUri(RingtoneManager.TYPE_RINGTONE)
        ring?.stop(); ring = RingtoneManager.getRingtone(c, u)?.apply {
            audioAttributes = AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_ALARM).build(); play() }
        val r = ring; Handler(Looper.getMainLooper()).postDelayed({ if (ring === r) { r?.stop(); ring = null } }, 20000)
    }
    private fun parseWhen(s: String): Long? {
        for (f in listOf("yyyy-MM-dd'T'HH:mm", "yyyy-MM-dd HH:mm", "yyyy-MM-dd")) try { return SimpleDateFormat(f, Locale.US).parse(s.trim())?.time } catch (e: Exception) {}
        return null
    }
    /** A number dials directly; a name is looked up in contacts (asks for the Contacts permission once). */
    private fun who(c: Context, w: String, go: (String, String) -> String): String {
        val q = w.trim(); if (q.isEmpty()) return "Who?"
        if (q.matches(Regex("[+0-9 ()\\-.]{3,}"))) return go(q.filter { it.isDigit() || it == '+' }, q)
        if (c.checkSelfPermission(Manifest.permission.READ_CONTACTS) != PackageManager.PERMISSION_GRANTED) {
            (c as? Activity)?.requestPermissions(arrayOf(Manifest.permission.READ_CONTACTS), 77)
            return "To find '$q' I need access to your contacts (only to look up the number). Allow it, then ask again - or give me the number."
        }
        c.contentResolver.query(Uri.withAppendedPath(ContactsContract.CommonDataKinds.Phone.CONTENT_FILTER_URI, Uri.encode(q)),
            arrayOf(ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME, ContactsContract.CommonDataKinds.Phone.NUMBER), null, null, null)?.use {
            if (it.moveToFirst()) return go(it.getString(1), it.getString(0))
        }
        return "I couldn't find '$q' in your contacts."
    }
    private fun findPkg(c: Context, name: String): String? {
        val q = name.lowercase(); val pm = c.packageManager
        val apps = pm.queryIntentActivities(Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER), 0).map { it.loadLabel(pm).toString().lowercase() to it.activityInfo.packageName }
        return (apps.firstOrNull { it.first == q } ?: apps.firstOrNull { it.first.startsWith(q) } ?: apps.firstOrNull { it.first.contains(q) })?.second
    }
    private fun stream(s: String): Pair<Int, String> = when (s.lowercase()) {
        "ring", "ringer", "ringtone" -> AudioManager.STREAM_RING to "Ring"; "alarm" -> AudioManager.STREAM_ALARM to "Alarm"
        "notification", "notifications" -> AudioManager.STREAM_NOTIFICATION to "Notification"; "call", "voice", "in-call" -> AudioManager.STREAM_VOICE_CALL to "Call"
        else -> AudioManager.STREAM_MUSIC to "Media" }
    private fun audio(c: Context) = c.getSystemService(AudioManager::class.java)
    private fun setVol(c: Context, pct: Int, which: String = ""): String {
        val (st, lab) = stream(which); val am = audio(c); val p = pct.coerceIn(0, 100); val mx = am.getStreamMaxVolume(st)
        am.setStreamVolume(st, Math.round(p * mx / 100f), AudioManager.FLAG_SHOW_UI); return "$lab volume set to $p%."
    }
    private fun start(c: Context, i: Intent) = c.startActivity(i.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
    private fun torch(c: Context, on: Boolean): String {
        val cm = c.getSystemService(CameraManager::class.java)
        val id = cm.cameraIdList.firstOrNull { cm.getCameraCharacteristics(it).get(CameraCharacteristics.FLASH_INFO_AVAILABLE) == true }
            ?: return "This phone has no flashlight."
        cm.setTorchMode(id, on); return if (on) "Flashlight on." else "Flashlight off."
    }
    private fun openApp(c: Context, name: String): String {
        val q = name.trim().lowercase(); if (q.isEmpty()) return "Which app?"
        val pm = c.packageManager
        val apps = pm.queryIntentActivities(Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER), 0)
            .map { it.loadLabel(pm).toString() to it.activityInfo.packageName }
        val hit = apps.firstOrNull { it.first.lowercase() == q } ?: apps.firstOrNull { it.first.lowercase().startsWith(q) }
            ?: apps.firstOrNull { it.first.lowercase().contains(q) } ?: apps.firstOrNull { it.second.lowercase().contains(q.replace(" ", "")) }
            ?: return "I couldn't find an app called '$name'."
        val i = pm.getLaunchIntentForPackage(hit.second) ?: return "Can't open ${hit.first}."
        start(c, i); return "Opened ${hit.first}."
    }
}
