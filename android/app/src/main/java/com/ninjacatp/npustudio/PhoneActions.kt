package com.ninjacatp.npustudio

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.media.AudioManager
import android.net.Uri
import android.os.BatteryManager
import android.provider.AlarmClock
import org.json.JSONObject

/** Allowlisted phone actions requested by the chat (shell v2). No arbitrary intents, no shell commands. */
object PhoneActions {
    val NAMES = setOf("open_app", "flashlight", "volume", "volume_change", "mute", "unmute", "alarm", "timer",
        "open_url", "web_search", "battery", "clipboard_set", "notify")

    fun run(c: Context, json: String, notify: (String, String) -> Unit): String = try {
        val o = JSONObject(json); val n = o.optString("name"); val g = o.optJSONObject("args") ?: JSONObject()
        if (n !in NAMES) "I can't do '$n' on the phone." else when (n) {
            "open_app" -> openApp(c, g.optString("app"))
            "flashlight" -> torch(c, g.opt("on").let { it == true || it.toString().lowercase() in setOf("true", "on", "1", "yes") })
            "volume" -> setVol(c, g.optInt("level", 50))
            "volume_change" -> { val am = audio(c); val mx = am.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
                setVol(c, am.getStreamVolume(AudioManager.STREAM_MUSIC) * 100 / mx + g.optInt("delta", 10)) }
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
            "notify" -> { notify(g.optString("title", "NpuStudio").take(80), g.optString("text").take(300)); "Notification sent." }
            else -> "Unknown action."
        }
    } catch (e: Exception) { "That didn't work on the phone: ${e.message}" }

    private fun audio(c: Context) = c.getSystemService(AudioManager::class.java)
    private fun setVol(c: Context, pct: Int): String {
        val am = audio(c); val p = pct.coerceIn(0, 100); val mx = am.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
        am.setStreamVolume(AudioManager.STREAM_MUSIC, Math.round(p * mx / 100f), AudioManager.FLAG_SHOW_UI); return "Media volume set to $p%."
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
