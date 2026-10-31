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
    }

    private fun load() {
        val p = pairing ?: return
        offline.visibility = View.GONE; web.visibility = View.VISIBLE
        web.loadUrl("${p.base}/app/#k=" + Uri.encode(p.key))
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
            if (r.isForMainFrame) showOffline(e.description.toString())
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
        @JavascriptInterface fun unpair() = runOnUiThread { HostRules.clear(this@MainActivity); pairing = null; pairLauncher.launch(Intent(this@MainActivity, PairActivity::class.java)) }
    }

    private fun safe(n: String) = n.replace(Regex("[^A-Za-z0-9._-]"), "_").take(80).ifEmpty { "npustudio.png" }

    private fun savePicture(bytes: ByteArray, name: String) {
        if (Build.VERSION.SDK_INT >= 29) {
            val v = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, name); put(MediaStore.Images.Media.MIME_TYPE, "image/png")
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
