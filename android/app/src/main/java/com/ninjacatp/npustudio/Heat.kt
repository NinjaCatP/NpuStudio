package com.ninjacatp.npustudio

import android.app.Activity
import android.app.ActivityManager
import android.app.AppOpsManager
import android.app.usage.UsageEvents
import android.app.usage.UsageStatsManager
import android.content.Context
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.os.Build
import android.os.Looper
import android.os.Process
import android.provider.Settings
import android.telecom.TelecomManager
import android.net.Uri
import android.Manifest
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import org.json.JSONArray
import org.json.JSONObject

/**
 * Heat tab (shell v4). Ranks recent UsageStats / usage events. Android does not
 * expose per-app CPU or battery heat to third-party apps, so this is recent
 * activity only. Stopping another app without root is killBackgroundProcesses
 * when the system allows it, otherwise the system App info page (Force stop).
 */
object Heat {
    private const val WINDOW_MS = 6L * 60L * 60L * 1000L
    private const val LIMIT = 40
    private val PKG = Regex("^[A-Za-z][A-Za-z0-9_]*(\\.[A-Za-z][A-Za-z0-9_]*)+$")

    /** Packages that must keep running for the phone, Maps GPS, or this app's link to stay usable. */
    private val SYSTEM_CORE = setOf(
        "android",
        "com.android.systemui",
        "com.android.settings",
        "com.android.phone",
        "com.android.server.telecom",
        "com.android.bluetooth",
        "com.android.nfc",
        "com.android.shell",
        "com.android.emergency",
        "com.android.cellbroadcastreceiver",
        "com.android.providers.telephony",
        "com.android.providers.contacts",
        "com.android.providers.settings",
        "com.android.providers.media",
        "com.android.providers.media.module",
        "com.android.providers.downloads",
        "com.android.permissioncontroller",
        "com.google.android.permissioncontroller",
        "com.samsung.android.permissioncontroller",
        "com.android.packageinstaller",
        "com.google.android.packageinstaller",
        "com.samsung.android.packageinstaller",
        "com.google.android.gms",
        "com.google.android.gsf",
        "com.google.android.ext.services",
        "com.android.networkstack",
        "com.google.android.networkstack",
    )
    private val MAPS = setOf(
        "com.google.android.apps.maps",
        "com.google.android.apps.mapslite",
        "com.google.android.projection.gearhead",
        "com.waze",
    )
    private val PHONE = setOf(
        "com.google.android.dialer",
        "com.android.dialer",
        "com.samsung.android.dialer",
        "com.android.incallui",
        "com.samsung.android.incallui",
        "com.google.android.incallui",
        "com.google.android.apps.messaging",
        "com.android.mms",
        "com.samsung.android.messaging",
    )

    fun hasUsage(c: Context): Boolean {
        val appOps = c.getSystemService(AppOpsManager::class.java) ?: return false
        val mode = if (Build.VERSION.SDK_INT >= 29) {
            appOps.unsafeCheckOpNoThrow(AppOpsManager.OPSTR_GET_USAGE_STATS, Process.myUid(), c.packageName)
        } else {
            @Suppress("DEPRECATION")
            appOps.checkOpNoThrow(AppOpsManager.OPSTR_GET_USAGE_STATS, Process.myUid(), c.packageName)
        }
        return mode == AppOpsManager.MODE_ALLOWED
    }

    fun list(c: Context): String {
      return try {
        if (!hasUsage(c)) {
            JSONObject().put("usage_access", false).put("window_hours", 6).put("apps", JSONArray()).toString()
        } else {
            val usm = c.getSystemService(UsageStatsManager::class.java)
                ?: return JSONObject().put("usage_access", true).put("window_hours", 6)
                    .put("error", "Usage stats aren't available on this phone.").put("apps", JSONArray()).toString()
            val end = System.currentTimeMillis()
            val start = end - WINDOW_MS
            val stats = usm.queryAndAggregateUsageStats(start, end) ?: emptyMap()
            val lastMove = HashMap<String, Int>()
            val lastEventAt = HashMap<String, Long>()
            val fgStarts = HashMap<String, Int>()
            val events = usm.queryEvents(start, end)
            val ev = UsageEvents.Event()
            while (events.hasNextEvent()) {
                events.getNextEvent(ev)
                val pkg = ev.packageName ?: continue
                when (ev.eventType) {
                    UsageEvents.Event.ACTIVITY_RESUMED, UsageEvents.Event.ACTIVITY_PAUSED -> {
                        lastMove[pkg] = ev.eventType
                        if (ev.timeStamp > (lastEventAt[pkg] ?: 0L)) lastEventAt[pkg] = ev.timeStamp
                    }
                    UsageEvents.Event.FOREGROUND_SERVICE_START -> {
                        fgStarts[pkg] = (fgStarts[pkg] ?: 0) + 1
                        if (ev.timeStamp > (lastEventAt[pkg] ?: 0L)) lastEventAt[pkg] = ev.timeStamp
                    }
                }
            }
            val guard = Guard(c)
            val rows = ArrayList<Row>()
            for (pkg in (stats.keys + lastMove.keys + fgStarts.keys)) {
                if (pkg.isNullOrEmpty()) continue
                val st = stats[pkg]
                val lastUsed = maxOf(st?.lastTimeUsed ?: 0L, lastEventAt[pkg] ?: 0L)
                val fgMs = st?.totalTimeInForeground ?: 0L
                val starts = fgStarts[pkg] ?: 0
                if (lastUsed < start && fgMs <= 0L && starts == 0) continue
                val why = guard.reason(pkg)
                rows.add(Row(
                    pkg, labelOf(c, pkg), lastUsed, fgMs, starts,
                    lastMove[pkg] == UsageEvents.Event.ACTIVITY_RESUMED,
                    why != null, why ?: "",
                ))
            }
            rows.sortWith(heatOrder<Row>({ it.inForeground }, { it.lastUsed }, { it.fgServiceStarts }, { it.foregroundMs }, { it.label }))
            val arr = JSONArray()
            for (r in rows.take(LIMIT)) {
                arr.put(JSONObject()
                    .put("package", r.pkg)
                    .put("label", r.label)
                    .put("last_used", r.lastUsed)
                    .put("foreground_ms", r.foregroundMs)
                    .put("fg_service_starts", r.fgServiceStarts)
                    .put("in_foreground", r.inForeground)
                    .put("protected", r.protected)
                    .put("reason", r.reason))
            }
            JSONObject().put("usage_access", true).put("window_hours", 6).put("apps", arr).toString()
        }
    } catch (e: Exception) {
        JSONObject().put("usage_access", hasUsage(c)).put("window_hours", 6)
            .put("error", e.message ?: "Couldn't read usage.").put("apps", JSONArray()).toString()
    }
    }

    fun stop(activity: Activity, raw: String): String {
      return try {
        val pkg = cleanPkg(raw) ?: return fail("That package name doesn't look right.")
        val why = Guard(activity).reason(pkg)
        val label = labelOf(activity, pkg)
        if (why != null) return fail("$label is protected ($why).")
        val canKill = activity.checkSelfPermission(Manifest.permission.KILL_BACKGROUND_PROCESSES) == PackageManager.PERMISSION_GRANTED
        if (!canKill || looksOnScreen(activity, pkg)) {
            openDetails(activity, pkg)
            val lead = if (!canKill) "This install can't clear other apps' background processes."
                else "$label looks like it's on screen, so a background kill won't close it."
            return ok("app_info", "$lead Opened App info — tap Force stop. Android does not let this app Force stop another app (no root or device admin).")
        }
        try {
            val am = activity.getSystemService(ActivityManager::class.java) ?: throw SecurityException("no activity manager")
            am.killBackgroundProcesses(pkg)
        } catch (e: SecurityException) {
            openDetails(activity, pkg)
            return ok("app_info", "Android refused a background kill for $label. Opened App info — tap Force stop.")
        }
        ok("background", "Asked Android to stop $label's background processes. That is not a full Force stop: Android may restart them, and a foreground service keeps running. If it keeps running, tap App info and then Force stop.")
    } catch (e: Exception) {
        fail("That didn't work: ${e.message}")
    }
    }

    fun openAppInfo(activity: Activity, raw: String): String {
      return try {
        val pkg = cleanPkg(raw) ?: return fail("That package name doesn't look right.")
        val why = Guard(activity).reason(pkg)
        if (why != null) return fail("${labelOf(activity, pkg)} is protected ($why).")
        openDetails(activity, pkg)
        ok("app_info", "Opened App info for ${labelOf(activity, pkg)}. Tap Force stop there. This app can't Force stop other apps itself.")
    } catch (e: Exception) {
        fail("Couldn't open App info: ${e.message}")
    }
    }

    fun openUsageAccess(activity: Activity): String {
        val plain = Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS)
        val direct = Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS).setData(Uri.parse("package:" + activity.packageName))
        for (i in listOf(plain, direct)) {
            try {
                onMain(activity) { activity.startActivity(Intent(i).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) }
                return "Opened Usage Access settings. Turn on NpuStudio Companion, then come back and tap Refresh."
            } catch (_: Exception) {}
        }
        return "Couldn't open Usage Access settings. Go to Settings, Apps, Special app access, Usage access, and allow NpuStudio Companion."
    }

    /** Name-based protection. Home, keyboard, and the live dialer/SMS packages are passed in. Persistent system apps are checked by the caller. */
    internal fun builtinReason(pkg: String, self: String, home: String?, ime: String?, maps: Set<String>, phones: Set<String>): String? {
        if (pkg == self || pkg.startsWith("com.ninjacatp.npustudio")) return "NpuStudio"
        if (pkg == "com.tailscale.ipn" || pkg.startsWith("com.tailscale.")) return "Tailscale"
        if (pkg in maps) return "Maps / navigation"
        if (pkg in phones) return "Phone or messages"
        if (home != null && pkg == home) return "Home screen"
        if (!ime.isNullOrEmpty() && pkg == ime) return "Keyboard"
        if (pkg in SYSTEM_CORE) return "System"
        return null
    }

    internal data class Rank(val label: String, val inForeground: Boolean, val lastUsed: Long, val fgServiceStarts: Int, val foregroundMs: Long)

    internal fun rank(rows: List<Rank>): List<Rank> =
        rows.sortedWith(heatOrder<Rank>({ it.inForeground }, { it.lastUsed }, { it.fgServiceStarts }, { it.foregroundMs }, { it.label }))

    private fun <T> heatOrder(
        inForeground: (T) -> Boolean, lastUsed: (T) -> Long, fgServiceStarts: (T) -> Int,
        foregroundMs: (T) -> Long, label: (T) -> String,
    ): Comparator<T> = compareBy<T>(inForeground)
        .thenByDescending(lastUsed)
        .thenByDescending(fgServiceStarts)
        .thenByDescending(foregroundMs)
        .thenBy { label(it).lowercase() }

    private data class Row(
        val pkg: String, val label: String, val lastUsed: Long, val foregroundMs: Long,
        val fgServiceStarts: Int, val inForeground: Boolean, val protected: Boolean, val reason: String,
    )

    private class Guard(val c: Context) {
        private val maps = mapsPackages(c)
        private val phones = phonePackages(c)
        private val home = homePackage(c)
        private val ime = imePackage(c)
        fun reason(pkg: String): String? {
            builtinReason(pkg, c.packageName, home, ime, maps, phones)?.let { return it }
            val ai = appInfo(c, pkg) ?: return null
            if (ai.uid == Process.SYSTEM_UID) return "System"
            if (ai.flags and ApplicationInfo.FLAG_PERSISTENT != 0) return "System"
            return null
        }
    }

    private fun mapsPackages(c: Context): Set<String> {
        val s = MAPS.toMutableSet()
        val pm = c.packageManager
        fun add(pkg: String?) {
            if (pkg.isNullOrEmpty()) return
            val p = pkg.lowercase()
            if (p == "com.android.chrome" || p.contains("browser")) return
            s.add(pkg)
        }
        try {
            add(pm.resolveActivity(Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_APP_MAPS), PackageManager.MATCH_DEFAULT_ONLY)
                ?.activityInfo?.packageName)
        } catch (_: Exception) {}
        try {
            add(pm.resolveActivity(Intent(Intent.ACTION_VIEW, Uri.parse("google.navigation:q=home")), PackageManager.MATCH_DEFAULT_ONLY)
                ?.activityInfo?.packageName)
        } catch (_: Exception) {}
        return s
    }

    private fun phonePackages(c: Context): Set<String> {
        val s = PHONE.toMutableSet()
        try { c.getSystemService(TelecomManager::class.java)?.defaultDialerPackage?.let { s.add(it) } } catch (_: Exception) {}
        try { android.provider.Telephony.Sms.getDefaultSmsPackage(c)?.let { s.add(it) } } catch (_: Exception) {}
        return s
    }

    private fun homePackage(c: Context): String? = try {
        c.packageManager.resolveActivity(
            Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_HOME), PackageManager.MATCH_DEFAULT_ONLY
        )?.activityInfo?.packageName
    } catch (_: Exception) { null }

    private fun imePackage(c: Context): String? = try {
        Settings.Secure.getString(c.contentResolver, Settings.Secure.DEFAULT_INPUT_METHOD)?.substringBefore('/')?.ifEmpty { null }
    } catch (_: Exception) { null }

    private fun appInfo(c: Context, pkg: String): ApplicationInfo? = try {
        c.packageManager.getApplicationInfo(pkg, 0)
    } catch (_: Exception) { null }

    private fun labelOf(c: Context, pkg: String): String = try {
        val pm = c.packageManager
        pm.getApplicationLabel(pm.getApplicationInfo(pkg, 0)).toString()
    } catch (_: Exception) { pkg }

    /** Last usage event in the past 15 minutes says the app is on screen. killBackgroundProcesses will not close that. */
    private fun looksOnScreen(c: Context, pkg: String): Boolean {
        if (!hasUsage(c)) return false
        val usm = c.getSystemService(UsageStatsManager::class.java) ?: return false
        val end = System.currentTimeMillis()
        val events = usm.queryEvents(end - 15L * 60L * 1000L, end)
        val ev = UsageEvents.Event()
        var last = -1
        while (events.hasNextEvent()) {
            events.getNextEvent(ev)
            if (ev.packageName == pkg && (ev.eventType == UsageEvents.Event.ACTIVITY_RESUMED || ev.eventType == UsageEvents.Event.ACTIVITY_PAUSED))
                last = ev.eventType
        }
        return last == UsageEvents.Event.ACTIVITY_RESUMED
    }

    private fun openDetails(activity: Activity, pkg: String) {
        onMain(activity) {
            activity.startActivity(Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:$pkg"))
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        }
    }

    private fun onMain(activity: Activity, block: () -> Unit) {
        if (Looper.myLooper() == Looper.getMainLooper()) { block(); return }
        val latch = CountDownLatch(1)
        var err: Exception? = null
        activity.runOnUiThread {
            try { block() } catch (e: Exception) { err = e } finally { latch.countDown() }
        }
        if (!latch.await(3, TimeUnit.SECONDS)) throw IllegalStateException("The screen didn't open in time.")
        err?.let { throw it }
    }

    private fun cleanPkg(raw: String): String? {
        val s = raw.trim()
        return if (s.length in 3..200 && PKG.matches(s)) s else null
    }

    private fun ok(method: String, message: String) =
        JSONObject().put("ok", true).put("method", method).put("message", message).toString()

    private fun fail(message: String) =
        JSONObject().put("ok", false).put("method", "denied").put("message", message).toString()
}
