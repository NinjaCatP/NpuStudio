package com.ninjacatp.npustudio

import android.app.Activity
import android.graphics.Color
import android.os.Bundle
import android.text.InputType
import android.view.Gravity
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import com.journeyapps.barcodescanner.ScanContract
import com.journeyapps.barcodescanner.ScanOptions
import java.net.HttpURLConnection
import java.net.URL
import kotlin.concurrent.thread

/** First-run screen: type the desktop address + pairing key, or scan the QR from the desktop. */
class PairActivity : AppCompatActivity() {
    private lateinit var addr: EditText
    private lateinit var key: EditText
    private lateinit var msg: TextView

    private val scan = registerForActivityResult(ScanContract()) { r ->
        val p = r.contents?.let { HostRules.fromQr(it) }
        if (p == null) { if (r.contents != null) msg.text = "That QR isn't an NpuStudio pairing code." ; return@registerForActivityResult }
        addr.setText("${p.host}:${p.port}"); key.setText(p.key); test(p)
    }

    override fun onCreate(b: Bundle?) {
        super.onCreate(b)
        val pad = (16 * resources.displayMetrics.density).toInt()
        val col = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(pad * 2, pad * 3, pad * 2, pad) }
        fun label(t: String, size: Float = 15f, c: Int = Color.parseColor("#9AA0AD")) =
            TextView(this).apply { text = t; textSize = size; setTextColor(c); setPadding(0, pad / 2, 0, pad / 4) }
        col.addView(label("NpuStudio Companion", 24f, Color.WHITE))
        col.addView(label("Pair with your desktop. On the desktop run Show-PairingQR.bat, then scan the code — or type the address and key shown in the NpuStudio server window."))
        val scanBtn = Button(this).apply { text = "Scan pairing QR"; setOnClickListener {
            scan.launch(ScanOptions().setDesiredBarcodeFormats(ScanOptions.QR_CODE).setPrompt("Point at the NpuStudio pairing QR").setBeepEnabled(false).setOrientationLocked(false))
        } }
        col.addView(scanBtn)
        col.addView(label("Desktop address (home: 192.168.x.x, away: Tailscale 100.x.x.x)"))
        addr = EditText(this).apply { hint = "192.168.0.230:47860"; inputType = InputType.TYPE_TEXT_VARIATION_URI; setTextColor(Color.WHITE); setHintTextColor(Color.GRAY) }
        col.addView(addr)
        col.addView(label("Pairing key"))
        key = EditText(this).apply { inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD; setTextColor(Color.WHITE) }
        col.addView(key)
        col.addView(Button(this).apply { text = "Connect"; setOnClickListener {
            val a = HostRules.parseAddress(addr.text.toString())
            if (a == null || key.text.isBlank()) { msg.text = "Enter the address and the key." ; return@setOnClickListener }
            val old = HostRules.load(this@PairActivity)
            val k = key.text.toString().trim()
            test(if (HostRules.isRemote(a.first)) HostRules.Pairing(old?.home, a.first, a.second, k, old?.mode ?: "auto", a.first)
                 else HostRules.Pairing(a.first, old?.remote, a.second, k, old?.mode ?: "auto", a.first))
        } })
        msg = label("", 14f, Color.parseColor("#FF6B6B")).apply { gravity = Gravity.START }
        col.addView(msg)
        HostRules.load(this)?.let { addr.setText("${it.host}:${it.port}") }
        if (intent.getBooleanExtra("scan", false)) scanBtn.performClick()
        setContentView(ScrollView(this).apply { addView(col) })
    }

    private fun test(p: HostRules.Pairing) {
        val hosts = listOfNotNull(p.host.takeIf { it.isNotEmpty() }, p.home, p.remote).distinct()
        if (hosts.any { !HostRules.allowed(it) }) { msg.text = "For safety the app only connects to home-network or Tailscale addresses (192.168.x.x, 10.x, 172.16-31.x, 100.64-127.x, *.ts.net)."; return }
        msg.setTextColor(Color.parseColor("#9AA0AD")); msg.text = "Connecting…"
        thread {
            val res = hosts.map { it to HostRules.probe(it, p.port, p.key, 4000) }
            runOnUiThread {
                val ok = res.firstOrNull { (it.second ?: -1) >= 0 }
                when {
                    ok != null -> { HostRules.save(this, p); setResult(Activity.RESULT_OK); finish() }
                    res.any { it.second == -403 } -> { msg.setTextColor(Color.parseColor("#FF6B6B")); msg.text = "The desktop refused that key." }
                    p.remote != null && p.home == null || hosts.size > 1 -> {   // saved anyway: remote may only work away from home
                        HostRules.save(this, p); msg.setTextColor(Color.parseColor("#FFB74D"))
                        msg.text = "Saved, but none of ${hosts.joinToString()} answered right now. Check NpuStudio-Server and Tailscale."
                        addr.postDelayed({ setResult(Activity.RESULT_OK); finish() }, 2500)
                    }
                    else -> { msg.setTextColor(Color.parseColor("#FF6B6B")); msg.text = "Can't reach ${p.base}. Is NpuStudio-Server running, and are you on the same Wi-Fi (or Tailscale)?" }
                }
            }
        }
    }
}
