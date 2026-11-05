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

    @Test fun hiddenMemoryIsNotReportedAsZero() {
        assertEquals(12000L, Heat.reportedPssKb(12000L, false))
        assertNull(Heat.reportedPssKb(0L, false))
        assertEquals(0L, Heat.reportedPssKb(0L, true))
        assertNull(Heat.reportedPssKb(-1L, true))
    }

    @Test fun cpuPercentNeedsASecondSampleInRange() {
        assertEquals(25.0, Heat.cpuPercentFromMs(0L, 0L, 250L, 1000L)!!, 0.01)
        assertNull(Heat.cpuPercentFromMs(0L, 0L, 100L, 500L))
        assertNull(Heat.cpuPercentFromMs(0L, 0L, 1000L, 20L * 60L * 1000L))
        assertNull(Heat.cpuPercentFromMs(500L, 0L, 100L, 5000L))
    }

    @Test fun procStatCpuTicksSurviveSpacesInTheProcessName() {
        assertEquals(30L, Heat.procStatCpuTicks("12 (Chrome App) R 1 1 1 0 -1 4194304 0 0 0 0 10 20 0 0"))
        assertNull(Heat.procStatCpuTicks("not a stat file"))
        assertEquals(1000L, Heat.ticksToMs(100L, 100L))
        assertNull(Heat.ticksToMs(10L, 0L))
    }
}
