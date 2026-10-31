package com.ninjacatp.npustudio

import android.content.Context
import android.net.Uri

/** Pairing storage and the "home network or Tailscale only" rule for every address the app talks to. */
object HostRules {
    const val SHELL_VERSION = 3

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

    fun isRemote(h: String) = h.lowercase().endsWith(".ts.net") ||
        h.split(".").mapNotNull { it.toIntOrNull() }.let { it.size == 4 && it[0] == 100 && it[1] in 64..127 }

    /** home = Wi-Fi address, remote = Tailscale address; mode = auto | home | remote. [host] is the one in use now. */
    data class Pairing(val home: String?, val remote: String?, val port: Int, val key: String, val mode: String = "auto", val host: String = home ?: remote ?: "") {
        val base get() = "http://$host:$port"
        fun order(): List<String> = when (mode) {
            "home" -> listOfNotNull(home); "remote" -> listOfNotNull(remote)
            else -> listOfNotNull(home, remote)
        }.distinct()
    }

    fun load(c: Context): Pairing? {
        val sp = c.getSharedPreferences("pair", Context.MODE_PRIVATE)
        val k = sp.getString("key", null) ?: return null
        var home = sp.getString("home", null); var remote = sp.getString("remote", null)
        sp.getString("host", null)?.let { old -> if (isRemote(old)) remote = remote ?: old else home = home ?: old }  // v1 storage
        if (home == null && remote == null) return null
        return Pairing(home, remote, sp.getInt("port", 47860), k, sp.getString("mode", "auto")!!)
    }

    fun save(c: Context, p: Pairing) = c.getSharedPreferences("pair", Context.MODE_PRIVATE).edit().remove("host")
        .putString("home", p.home).putString("remote", p.remote).putInt("port", p.port).putString("key", p.key).putString("mode", p.mode).apply()

    fun clear(c: Context) = c.getSharedPreferences("pair", Context.MODE_PRIVATE).edit().clear().apply()

    /** npustudio://pair?host=<lan>&remote=<tailscale>&port=..&key=..  (QR from the desktop or the NpuStudio app) */
    fun fromQr(text: String): Pairing? = try {
        val u = Uri.parse(text.trim())
        if (u.scheme != "npustudio") null else {
            val h = u.getQueryParameter("host"); var r = u.getQueryParameter("remote")?.takeIf { it.isNotBlank() }
            var home = h
            if (h != null && isRemote(h)) { r = r ?: h; home = null }
            Pairing(home, r, u.getQueryParameter("port")?.toIntOrNull() ?: 47860, u.getQueryParameter("key")!!)
        }
    } catch (e: Exception) { null }

    /** Health check; returns round-trip ms or null. 403 (bad key) returns -403. */
    fun probe(host: String, port: Int, key: String, timeout: Int = 2500): Int? = try {
        val t = System.nanoTime()
        val code = (java.net.URL("http://$host:$port/api/v1/health").openConnection() as java.net.HttpURLConnection).run {
            connectTimeout = timeout; readTimeout = timeout + 1500; setRequestProperty("X-Key", key); val c = responseCode; disconnect(); c }
        when (code) { 200 -> ((System.nanoTime() - t) / 1_000_000).toInt(); 403 -> -403; else -> null }
    } catch (e: Exception) { null }

    /** Accepts "192.168.0.230", "192.168.0.230:47860" or "http://host:port/". */
    fun parseAddress(s: String): Pair<String, Int>? {
        val t = s.trim().removePrefix("http://").removePrefix("https://").trimEnd('/')
        if (t.isEmpty()) return null
        val i = t.lastIndexOf(':')
        return if (i > 0) t.substring(0, i) to (t.substring(i + 1).toIntOrNull() ?: return null) else t to 47860
    }
}
