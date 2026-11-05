package com.ninjacatp.npustudio

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class HeatRankTest {
    private val maps = setOf("com.google.android.apps.maps", "com.waze")
    private val phones = setOf("com.google.android.dialer", "com.google.android.apps.messaging")

    private fun why(pkg: String) = Heat.builtinReason(
        pkg, "com.ninjacatp.npustudio.companion", "com.sec.android.app.launcher",
        "com.google.android.inputmethod.latin", maps, phones,
    )

    @Test fun protectsImportantAppsAndLeavesOrdinaryOnes() {
        assertEquals("NpuStudio", why("com.ninjacatp.npustudio.companion"))
        assertEquals("NpuStudio", why("com.ninjacatp.npustudio.something"))
        assertEquals("Tailscale", why("com.tailscale.ipn"))
        assertEquals("Maps / navigation", why("com.google.android.apps.maps"))
        assertEquals("Maps / navigation", why("com.waze"))
        assertEquals("Phone or messages", why("com.google.android.dialer"))
        assertEquals("Phone or messages", why("com.google.android.apps.messaging"))
        assertEquals("Home screen", why("com.sec.android.app.launcher"))
        assertEquals("Keyboard", why("com.google.android.inputmethod.latin"))
        assertEquals("System", why("com.google.android.gms"))
        assertEquals("System", why("com.android.systemui"))
        assertEquals("System", why("android"))
        assertNull(why("com.spotify.music"))
        assertNull(why("com.facebook.katana"))
    }

    @Test fun ranksRecentBackgroundActivityAheadOfWhatIsOnScreen() {
        val rows = listOf(
            Heat.Rank("Maps", true, 100, 2, 50),
            Heat.Rank("Spotify", false, 50, 1, 10),
            Heat.Rank("System UI", false, 90, 0, 0),
            Heat.Rank("Facebook", false, 10, 0, 80),
            Heat.Rank("NpuStudio", true, 200, 1, 5),
        )
        assertEquals(listOf("System UI", "Spotify", "Facebook", "NpuStudio", "Maps"), Heat.rank(rows).map { it.label })
    }
}
