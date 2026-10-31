package com.ninjacatp.npustudio

import android.content.Context
import android.net.Uri

/** Pairing storage and the "home network or Tailscale only" rule for every address the app talks to. */
object HostRules {
    const val SHELL_VERSION = 1

    fun allowed(host: String?): Boolean {
        val h = (host ?: return false).lowercase().trim('[', ']')
        if (h.endsWith(".ts.net") || h.endsWith(".local") || h == "localhost") return true
        val p = h.split(".").mapNotNull { it.toIntOrNull() }
        if (p.size != 4 || p.any { it !in 0..255 }) return false
        return p[0] == 10 || p[0] == 127 ||
            (p[0] == 192 && p[1] == 168) ||
            (p[0] == 172 && p[1] in 16..31) ||
            (p[0] == 100 && p[1] in 64..127)        // Tailscale CGNAT range 100.64.0.0/10
    }

    data class Pairing(val host: String, val port: Int, val key: String) {
        val base get() = "http://$host:$port"
    }

    fun load(c: Context): Pairing? {
        val sp = c.getSharedPreferences("pair", Context.MODE_PRIVATE)
        val h = sp.getString("host", null) ?: return null
        val k = sp.getString("key", null) ?: return null
        return Pairing(h, sp.getInt("port", 47860), k)
    }

    fun save(c: Context, p: Pairing) = c.getSharedPreferences("pair", Context.MODE_PRIVATE).edit()
        .putString("host", p.host).putInt("port", p.port).putString("key", p.key).apply()

    fun clear(c: Context) = c.getSharedPreferences("pair", Context.MODE_PRIVATE).edit().clear().apply()

    /** npustudio://pair?host=..&port=..&key=..  (shown as a QR by the desktop's Show-PairingQR page) */
    fun fromQr(text: String): Pairing? = try {
        val u = Uri.parse(text.trim())
        if (u.scheme != "npustudio") null
        else Pairing(u.getQueryParameter("host")!!, u.getQueryParameter("port")?.toIntOrNull() ?: 47860, u.getQueryParameter("key")!!)
    } catch (e: Exception) { null }

    /** Accepts "192.168.0.230", "192.168.0.230:47860" or "http://host:port/". */
    fun parseAddress(s: String): Pair<String, Int>? {
        val t = s.trim().removePrefix("http://").removePrefix("https://").trimEnd('/')
        if (t.isEmpty()) return null
        val i = t.lastIndexOf(':')
        return if (i > 0) t.substring(0, i) to (t.substring(i + 1).toIntOrNull() ?: return null) else t to 47860
    }
}
