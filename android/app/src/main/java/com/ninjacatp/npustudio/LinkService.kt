package com.ninjacatp.npustudio

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioAttributes
import android.media.AudioManager
import android.media.MediaPlayer
import android.media.RingtoneManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import androidx.core.app.NotificationCompat
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import kotlin.concurrent.thread

/** Background link to the desktop (shell v3): long-polls /api/v1/phone/poll (home first, then Tailscale) for queued
 *  commands. Only "find_phone" and "stop_sound" are accepted. Low-priority persistent notification while running. */
class LinkService : Service() {
    companion object {
        const val ACT_FIND = "find"; const val ACT_STOP = "stop"
        fun enabled(c: Context) = c.getSharedPreferences("npu", Context.MODE_PRIVATE).getBoolean("bg_link", true)
        fun setEnabled(c: Context, on: Boolean) {
            c.getSharedPreferences("npu", Context.MODE_PRIVATE).edit().putBoolean("bg_link", on).apply()
            if (on) start(c) else c.stopService(Intent(c, LinkService::class.java))
        }
        fun start(c: Context, action: String? = null) {
            if (action == null && (!enabled(c) || HostRules.load(c) == null)) return
            val i = Intent(c, LinkService::class.java).setAction(action)
            try { if (Build.VERSION.SDK_INT >= 26) c.startForegroundService(i) else c.startService(i) } catch (e: Exception) {}
        }
        @Volatile var lastOk = 0L
        /** Stable id for this phone in the desktop's device list (random, not a hardware id). */
        fun deviceId(c: Context): String { val sp = c.getSharedPreferences("npu", Context.MODE_PRIVATE)
            return sp.getString("dev_id", null) ?: java.util.UUID.randomUUID().toString().take(8).also { sp.edit().putString("dev_id", it).apply() } }
        fun deviceName() = (Build.MANUFACTURER.replaceFirstChar { it.uppercase() } + " " + Build.MODEL).let { if (Build.MODEL.startsWith(Build.MANUFACTURER, true)) Build.MODEL else it }.take(40)
    }
    @Volatile private var running = false
    private var player: MediaPlayer? = null
    private var savedFilter = -1
    private val main = Handler(Looper.getMainLooper())

    override fun onBind(i: Intent?): IBinder? = null

    override fun onStartCommand(i: Intent?, flags: Int, id: Int): Int {
        channels()
        val n = NotificationCompat.Builder(this, "link").setSmallIcon(R.drawable.ic_stat).setContentTitle("NpuStudio connected")
            .setContentText("Listening for Find my phone from your PC").setPriority(NotificationCompat.PRIORITY_MIN).setOngoing(true)
            .setContentIntent(PendingIntent.getActivity(this, 1, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE)).build()
        if (Build.VERSION.SDK_INT >= 34) startForeground(1, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE) else startForeground(1, n)
        when (i?.action) { ACT_FIND -> main.post { ring() }; ACT_STOP -> main.post { stopRing() } }
        if (!enabled(this) && i?.action == null) { stopSelf(); return START_NOT_STICKY }
        if (!running && enabled(this)) { running = true; thread(name = "npu-link") { loop() } }
        return START_STICKY
    }
    override fun onDestroy() { running = false; stopRing(); super.onDestroy() }

    private fun loop() {
        while (running) {
            val p = HostRules.load(this) ?: break
            var ok = false
            for (h in p.order()) {
                try {
                    val c = URL("http://$h:${p.port}/api/v1/phone/poll?wait=25&id=${deviceId(this)}&name=" + android.net.Uri.encode(deviceName())).openConnection() as HttpURLConnection
                    c.connectTimeout = 4000; c.readTimeout = 35000; c.setRequestProperty("X-Key", p.key)
                    if (c.responseCode == 200) {
                        val cmds = JSONObject(c.inputStream.bufferedReader().readText()).optJSONArray("commands")
                        ok = true; lastOk = System.currentTimeMillis()
                        for (k in 0 until (cmds?.length() ?: 0)) {
                            val cmd = cmds!!.getJSONObject(k)
                            val res = when (cmd.optString("name")) {
                                "find_phone" -> { main.post { ring() }; "Ringing loudly for up to 60 s." }
                                "stop_sound" -> { main.post { stopRing() }; "Stopped." }
                                else -> runOnMain { PhoneActions.run(this, cmd.toString()) { t, x -> note(t, x) } }   // same allowlist as the chat
                            }
                            report(h, p, cmd.optString("cid"), res)
                        }
                    }
                    c.disconnect()
                    if (ok) break
                } catch (e: Exception) {}
            }
            if (!ok) try { Thread.sleep(30000) } catch (e: InterruptedException) {}
        }
        running = false
    }

    private fun runOnMain(f: () -> String): String {
        val box = arrayOfNulls<String>(1); val l = java.util.concurrent.CountDownLatch(1)
        main.post { box[0] = try { f() } catch (e: Exception) { "Error: ${e.message}" }; l.countDown() }
        l.await(15, java.util.concurrent.TimeUnit.SECONDS); return box[0] ?: "Timed out on the phone."
    }
    private fun report(h: String, p: HostRules.Pairing, cid: String, text: String) = try {
        val c = URL("http://$h:${p.port}/api/v1/phone/result").openConnection() as HttpURLConnection
        c.connectTimeout = 4000; c.readTimeout = 8000; c.requestMethod = "POST"; c.doOutput = true
        c.setRequestProperty("X-Key", p.key); c.setRequestProperty("Content-Type", "application/json")
        c.outputStream.use { it.write(JSONObject().put("cid", cid).put("text", text).toString().toByteArray()) }; c.responseCode; c.disconnect()
    } catch (e: Exception) {}
    private fun note(t: String, x: String) {
        val nm = getSystemService(NotificationManager::class.java); if (Build.VERSION.SDK_INT >= 26) nm.createNotificationChannel(NotificationChannel("jobs", "Finished jobs", NotificationManager.IMPORTANCE_DEFAULT))
        nm.notify((System.currentTimeMillis() % 100000).toInt() + 10, NotificationCompat.Builder(this, "jobs").setSmallIcon(R.drawable.ic_stat).setContentTitle(t).setContentText(x).setAutoCancel(true).build())
    }
    private fun channels() {
        if (Build.VERSION.SDK_INT < 26) return
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel("link", "Background connection", NotificationManager.IMPORTANCE_MIN))
        nm.createNotificationChannel(NotificationChannel("find", "Find my phone", NotificationManager.IMPORTANCE_HIGH).apply { setSound(null, null) })
    }

    /** Loud looping alarm on the ALARM stream (plays on silent/vibrate; DND overridden when we have policy access). */
    private fun ring() {
        stopRing(false)
        val am = getSystemService(AudioManager::class.java)
        am.setStreamVolume(AudioManager.STREAM_ALARM, am.getStreamMaxVolume(AudioManager.STREAM_ALARM), 0)
        val nm = getSystemService(NotificationManager::class.java)
        if (nm.isNotificationPolicyAccessGranted && nm.currentInterruptionFilter != NotificationManager.INTERRUPTION_FILTER_ALL) {
            savedFilter = nm.currentInterruptionFilter; nm.setInterruptionFilter(NotificationManager.INTERRUPTION_FILTER_ALL)
        }
        val uri = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM) ?: RingtoneManager.getDefaultUri(RingtoneManager.TYPE_RINGTONE)
        player = MediaPlayer().apply {
            setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_ALARM).setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION).build())
            setDataSource(this@LinkService, uri); isLooping = true; prepare(); start()
        }
        val stop = PendingIntent.getService(this, 2, Intent(this, LinkService::class.java).setAction(ACT_STOP), PendingIntent.FLAG_IMMUTABLE)
        val n = NotificationCompat.Builder(this, "find").setSmallIcon(R.drawable.ic_stat).setContentTitle("Find my phone")
            .setContentText("Here I am! Tap to stop.").setPriority(NotificationCompat.PRIORITY_MAX).setCategory(NotificationCompat.CATEGORY_ALARM)
            .setContentIntent(stop).setDeleteIntent(stop).addAction(0, "Found it - Stop", stop).setOngoing(true)
            .setFullScreenIntent(PendingIntent.getActivity(this, 3, Intent(this, MainActivity::class.java).setAction("npu.found"), PendingIntent.FLAG_IMMUTABLE), true).build()
        nm.notify(2, n)
        main.postDelayed({ stopRing() }, 60000)
    }
    private fun stopRing(clearNote: Boolean = true) {
        player?.let { try { it.stop() } catch (e: Exception) {}; it.release() }; player = null
        main.removeCallbacksAndMessages(null)
        val nm = getSystemService(NotificationManager::class.java)
        if (savedFilter > 0 && nm.isNotificationPolicyAccessGranted) nm.setInterruptionFilter(savedFilter); savedFilter = -1
        if (clearNote) nm.cancel(2)
    }
}

class BootReceiver : BroadcastReceiver() {
    override fun onReceive(c: Context, i: Intent) {
        if (i.action == Intent.ACTION_BOOT_COMPLETED || i.action == Intent.ACTION_MY_PACKAGE_REPLACED) LinkService.start(c)
    }
}
