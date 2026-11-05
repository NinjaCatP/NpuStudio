package com.ninjacatp.npustudio

import android.Manifest
import android.annotation.SuppressLint
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.ContentValues
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Base64
import android.view.Gravity
import android.view.View
import android.webkit.*
import android.widget.*
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.FileProvider
import java.io.File
import java.io.FileOutputStream
import kotlin.concurrent.thread

/** Thin shell: loads the UI from the desktop (http://host:port/app/) so UI updates never need a new APK. */
class MainActivity : AppCompatActivity() {
    private lateinit var web: WebView
    private lateinit var offline: LinearLayout
    private lateinit var offMsg: TextView
    private var pairing: HostRules.Pairing? = null
    private var pickCb: String? = null

    private val pairLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) {
        pairing = HostRules.load(this); if (pairing == null) finish() else load()
    }
    private val picker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        val cb = pickCb ?: return@registerForActivityResult; pickCb = null
        if (uri == null) { js("window.npuCallback('$cb', null)"); return@registerForActivityResult }
        thread {
            val b64 = try { contentResolver.openInputStream(uri)!!.use { Base64.encodeToString(it.readBytes(), Base64.NO_WRAP) } } catch (e: Exception) { null }
            runOnUiThread { js(if (b64 == null) "window.npuCallback('$cb', null)" else "window.npuCallback('$cb', '$b64')") }
        }
    }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(b: Bundle?) {
        super.onCreate(b)
        val root = FrameLayout(this)
        web = WebView(this).apply {
            settings.javaScriptEnabled = true; settings.domStorageEnabled = true
            settings.allowFileAccess = false; settings.allowContentAccess = false
            settings.mediaPlaybackRequiresUserGesture = true
            setBackgroundColor(Color.parseColor("#14161C"))
            addJavascriptInterface(Bridge(), "NpuShell")
            webViewClient = Client()
        }
        root.addView(web)
        offline = buildOffline(); root.addView(offline)
        setContentView(root)
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() { if (web.canGoBack()) web.goBack() else finish() }
        })
        if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED)
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.POST_NOTIFICATIONS), 1)
        pairing = HostRules.load(this)
        if (pairing == null) pairLauncher.launch(Intent(this, PairActivity::class.java)) else load()
        LinkService.start(this); if (intent?.action == "npu.found") LinkService.start(this, LinkService.ACT_STOP)
        askBatteryOnce()
    }
    override fun onNewIntent(i: Intent) { super.onNewIntent(i); if (i.action == "npu.found") LinkService.start(this, LinkService.ACT_STOP) }
    /** Explains once, then asks Android to exempt us from battery optimization so Find my phone keeps working. */
    private fun askBatteryOnce() {
        val sp = getSharedPreferences("npu", MODE_PRIVATE)
        val pm = getSystemService(android.os.PowerManager::class.java)
        if (sp.getBoolean("asked_battery", false) || !LinkService.enabled(this) || pm.isIgnoringBatteryOptimizations(packageName)) return
        sp.edit().putBoolean("asked_battery", true).apply()
        android.app.AlertDialog.Builder(this).setTitle("Find my phone")
            .setMessage("NpuStudio keeps a small background connection to your PC so 'Find my phone' can ring this phone. It checks in about twice a minute and uses very little battery. Allow it to run in the background on the next screen? You can turn it off in NpuStudio's Settings tab.")
            .setPositiveButton("Continue") { _, _ -> try { startActivity(Intent(android.provider.Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS, android.net.Uri.parse("package:$packageName"))) } catch (e: Exception) {} }
            .setNegativeButton("Not now", null).show()
    }

    @Volatile private var choosing = false
    /** Picks the first reachable address (auto: home first, then Tailscale) and loads the UI from it. */
    private fun load() {
        val p = pairing ?: return
        if (choosing) return; choosing = true
        offline.visibility = View.GONE; web.visibility = View.VISIBLE
        thread {
            val pick = p.order().firstOrNull { (HostRules.probe(it, p.port, p.key, 1500) ?: -1) >= 0 }
            runOnUiThread {
                choosing = false
                if (pick == null) { pairing = p.copy(host = p.order().firstOrNull() ?: p.host); showOffline("None of the saved addresses answered (${p.order().joinToString()})."); return@runOnUiThread }
                pairing = p.copy(host = pick)
                web.loadUrl("${pairing!!.base}/app/#k=" + Uri.encode(p.key))
            }
        }
    }

    private fun showOffline(why: String) {
        offMsg.text = "Can't reach your desktop (${pairing?.base}).\n\n$why\n\nMake sure NpuStudio-Server is running on the desktop, and that this phone is on the home Wi-Fi or connected to Tailscale."
        offline.visibility = View.VISIBLE; web.visibility = View.INVISIBLE
    }

    private fun buildOffline(): LinearLayout {
        val pad = (24 * resources.displayMetrics.density).toInt()
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL; gravity = Gravity.CENTER; setPadding(pad, pad, pad, pad)
            setBackgroundColor(Color.parseColor("#14161C")); visibility = View.GONE
            addView(TextView(context).apply { text = "Desktop offline"; textSize = 22f; setTextColor(Color.WHITE); gravity = Gravity.CENTER })
            offMsg = TextView(context).apply { textSize = 15f; setTextColor(Color.parseColor("#9AA0AD")); gravity = Gravity.CENTER; setPadding(0, pad / 2, 0, pad / 2) }
            addView(offMsg)
            addView(Button(context).apply { text = "Retry"; setOnClickListener { load() } })
            addView(Button(context).apply { text = "Change pairing"; setOnClickListener { pairLauncher.launch(Intent(context, PairActivity::class.java)) } })
        }
    }

    private fun js(code: String) = web.evaluateJavascript(code, null)

    private inner class Client : WebViewClient() {
        private fun ours(u: Uri) = pairing?.let { u.host == it.host && (u.port == it.port || u.port == -1 && it.port == 80) } ?: false
        override fun shouldOverrideUrlLoading(v: WebView, r: WebResourceRequest): Boolean {
            if (ours(r.url) && r.url.path?.endsWith(".apk") == true) { startActivity(Intent(Intent.ACTION_VIEW, r.url)); return true }   // app update: let the browser download it
            if (ours(r.url)) return false
            if (r.url.scheme == "https") startActivity(Intent(Intent.ACTION_VIEW, r.url))   // open outside links in the browser
            return true
        }
        override fun shouldInterceptRequest(v: WebView, r: WebResourceRequest): WebResourceResponse? {
            val u = r.url
            if (u.scheme == "data" || u.scheme == "blob") return null
            // Only the paired desktop (a LAN / Tailscale address) may be contacted from the WebView.
            if (u.scheme == "http" && (!HostRules.allowed(u.host) || !ours(u))) return WebResourceResponse("text/plain", "utf-8", 403, "Blocked", null, null)
            if (u.scheme != "http" && u.scheme != "https") return WebResourceResponse("text/plain", "utf-8", 403, "Blocked", null, null)
            return null
        }
        override fun onReceivedError(v: WebView, r: WebResourceRequest, e: WebResourceError) {
            if (r.isForMainFrame) { if ((pairing?.order()?.size ?: 0) > 1 && !choosing) load() else showOffline(e.description.toString()) }
        }
        override fun onReceivedHttpError(v: WebView, r: WebResourceRequest, e: WebResourceResponse) {
            if (r.isForMainFrame) showOffline("The desktop answered HTTP ${e.statusCode}.")
        }
    }

    /** window.NpuShell — see API.md "JS bridge". All calls are async; results come back via window.npuCallback(id, data). */
    private inner class Bridge {
        @JavascriptInterface fun shellVersion(): Int = HostRules.SHELL_VERSION
        @JavascriptInterface fun pickImage(cbId: String) = runOnUiThread { pickCb = cbId.filter { it.isLetterOrDigit() }; picker.launch("image/*") }
        @JavascriptInterface fun saveImage(b64: String, name: String) { thread {
            val ok = try { savePicture(Base64.decode(b64, Base64.DEFAULT), safe(name)); true } catch (e: Exception) { false }
            runOnUiThread { Toast.makeText(this@MainActivity, if (ok) "Saved to Pictures/NpuStudio" else "Couldn't save the picture", Toast.LENGTH_SHORT).show() }
        } }
        @JavascriptInterface fun download(b64: String, name: String) = saveImage(b64, name)
        @JavascriptInterface fun share(b64: String, name: String) { thread {
            val dir = File(cacheDir, "share").apply { mkdirs(); listFiles()?.forEach { it.delete() } }
            val f = File(dir, safe(name)); f.writeBytes(Base64.decode(b64, Base64.DEFAULT))
            val uri = FileProvider.getUriForFile(this@MainActivity, "$packageName.files", f)
            val i = Intent(Intent.ACTION_SEND).setType("image/png").putExtra(Intent.EXTRA_STREAM, uri).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            runOnUiThread { startActivity(Intent.createChooser(i, "Share picture")) }
        } }
        @JavascriptInterface fun notify(title: String, text: String) = notifyUser(title, text)
        /** v3: background link (Find my phone) on/off, and when it last reached the desktop (ms epoch, 0 = never). */
        @JavascriptInterface fun getBackground(): String = org.json.JSONObject().put("on", LinkService.enabled(this@MainActivity)).put("last_ok", LinkService.lastOk).put("id", LinkService.deviceId(this@MainActivity)).put("name", LinkService.deviceName()).toString()
        @JavascriptInterface fun setBackground(on: Boolean) = runOnUiThread { LinkService.setEnabled(this@MainActivity, on) }
        /** v2: allowlisted phone action from the chat; result text comes back via npuCallback. */
        @JavascriptInterface fun phoneAction(json: String, cbId: String) {
            val id = cbId.filter { it.isLetterOrDigit() }
            runOnUiThread {
                val r = PhoneActions.run(this@MainActivity, json) { t, x -> notifyUser(t, x) }
                js("window.npuCallback('$id', " + org.json.JSONObject.quote(r) + ")")
            }
        }
        @JavascriptInterface fun getConnection(): String {
            val p = pairing ?: return "{}"
            return org.json.JSONObject().put("home", p.home ?: "").put("remote", p.remote ?: "").put("port", p.port)
                .put("mode", p.mode).put("active", p.host).put("activeKind", if (HostRules.isRemote(p.host)) "remote" else "home").toString()
        }
        /** Returns "" on success (then reloads), or an error message for the page to show. */
        @JavascriptInterface fun setConnection(home: String, remote: String, mode: String): String {
            val p = pairing ?: return "Not paired."
            fun host(s: String) = HostRules.parseAddress(s)?.first
            val h = host(home); val r = host(remote)
            if (h == null && r == null) return "Enter at least one address."
            listOfNotNull(h, r).firstOrNull { !HostRules.allowed(it) }?.let { return "$it isn't a home-network or Tailscale address." }
            val np = p.copy(home = h, remote = r, mode = if (mode in listOf("auto", "home", "remote")) mode else "auto")
            HostRules.save(this@MainActivity, np); pairing = np; runOnUiThread { load() }; return ""
        }
        @JavascriptInterface fun probe(cbId: String) {
            val p = pairing ?: return; val id = cbId.filter { it.isLetterOrDigit() }
            thread {
                val o = org.json.JSONObject()
                p.home?.let { o.put("Home $it", HostRules.probe(it, p.port, p.key) ?: org.json.JSONObject.NULL) }
                p.remote?.let { o.put("Remote $it", HostRules.probe(it, p.port, p.key) ?: org.json.JSONObject.NULL) }
                val s = o.toString().replace("\\", "\\\\").replace("'", "\\'")
                runOnUiThread { js("window.npuCallback('$id', '$s')") }
            }
        }
        /** Page lost the desktop: switch to the other saved address if it answers. */
        @JavascriptInterface fun fallback() = runOnUiThread { if ((pairing?.order()?.size ?: 0) > 1) load() }
        @JavascriptInterface fun rePair() = runOnUiThread { pairLauncher.launch(Intent(this@MainActivity, PairActivity::class.java).putExtra("scan", true)) }
        @JavascriptInterface fun unpair() = runOnUiThread { HostRules.clear(this@MainActivity); pairing = null; pairLauncher.launch(Intent(this@MainActivity, PairActivity::class.java)) }
        /** v4 Heat tab. UsageStats ranking, not a CPU or heat meter. See Heat.kt. */
        @JavascriptInterface fun heatList(): String = Heat.list(this@MainActivity)
        @JavascriptInterface fun heatStop(pkg: String): String = Heat.stop(this@MainActivity, pkg)
        @JavascriptInterface fun heatAppInfo(pkg: String): String = Heat.openAppInfo(this@MainActivity, pkg)
        @JavascriptInterface fun openUsageAccess(): String = Heat.openUsageAccess(this@MainActivity)
    }

    private fun safe(n: String) = n.replace(Regex("[^A-Za-z0-9._-]"), "_").take(80).ifEmpty { "npustudio.png" }

    private fun savePicture(bytes: ByteArray, name: String) {
        if (Build.VERSION.SDK_INT >= 29) {
            val v = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, name); put(MediaStore.Images.Media.MIME_TYPE, when (name.substringAfterLast('.').lowercase()) { "jpg", "jpeg" -> "image/jpeg"; "webp" -> "image/webp"; else -> "image/png" })
                put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/NpuStudio")
            }
            val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, v)!!
            contentResolver.openOutputStream(uri)!!.use { it.write(bytes) }
        } else {
            @Suppress("DEPRECATION")
            val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "NpuStudio").apply { mkdirs() }
            FileOutputStream(File(dir, name)).use { it.write(bytes) }
        }
    }

    private fun notifyUser(title: String, text: String) {
        val nm = getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= 26) nm.createNotificationChannel(NotificationChannel("jobs", "Finished jobs", NotificationManager.IMPORTANCE_DEFAULT))
        if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) return
        val pi = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP), PendingIntent.FLAG_IMMUTABLE)
        NotificationManagerCompat.from(this).notify((System.currentTimeMillis() % 100000).toInt(),
            NotificationCompat.Builder(this, "jobs").setSmallIcon(R.drawable.ic_stat).setContentTitle(title).setContentText(text)
                .setContentIntent(pi).setAutoCancel(true).build())
    }
}
