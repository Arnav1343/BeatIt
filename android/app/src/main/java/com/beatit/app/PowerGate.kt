package com.beatit.app

import android.content.Context
import android.os.PowerManager
import android.util.Log

/**
 * Holds the CPU wake lock only while there is real work to keep awake for.
 *
 * The service used to acquire a PARTIAL_WAKE_LOCK in onCreate and hold it
 * until onDestroy — continuously, from launch, whether or not anything was
 * playing or downloading. That drains the battery for nothing, and it is
 * precisely the signature aggressive OEM power managers (Funtouch/OriginOS,
 * MIUI) look for when deciding which background app to kill.
 *
 * Playback still needs it: the WebView keeps playing while backgrounded, and
 * without a wake lock the CPU can sleep out from under it once the screen is
 * off. So the lock is held while playing OR while any download runs, and
 * dropped the moment neither is true.
 *
 * The lock also carries a timeout as a backstop. If a caller ever leaks a
 * workStarted() without its workFinished(), the system reclaims the lock
 * rather than letting it be held until the process dies.
 */
object PowerGate {

    private const val TAG = "PowerGate"

    /** Backstop only — routine release comes from the counters below. */
    private const val LOCK_TIMEOUT_MS = 60 * 60 * 1000L

    private var wakeLock: PowerManager.WakeLock? = null
    private var playing = false
    private var activeWork = 0

    @Synchronized
    fun init(context: Context) {
        if (wakeLock != null) return
        val pm = context.applicationContext.getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "BeatIt::PlaybackLock").apply {
            setReferenceCounted(false)
        }
    }

    /** Pushed in from the page's playback state via PlaybackSession. */
    @Synchronized
    fun setPlaying(isPlaying: Boolean) {
        playing = isPlaying
        // No early return when the value is unchanged: the page pushes its
        // state about once a second while playing, and each push renews the
        // lock's timeout. That is what keeps a long listening session from
        // running past LOCK_TIMEOUT_MS and losing the lock mid-track.
        apply()
    }

    /** Bracket any download with these two. */
    @Synchronized
    fun workStarted() {
        activeWork++
        apply()
    }

    @Synchronized
    fun workFinished() {
        if (activeWork > 0) activeWork--
        apply()
    }

    /** Service teardown — drop the lock whatever the counters say. */
    @Synchronized
    fun release() {
        playing = false
        activeWork = 0
        apply()
    }

    private fun apply() {
        val lock = wakeLock ?: return
        val wanted = playing || activeWork > 0
        try {
            if (wanted) {
                // Non-reference-counted, so acquiring while already held just
                // pushes the timeout back out — which is the renewal above.
                lock.acquire(LOCK_TIMEOUT_MS)
            } else if (lock.isHeld) {
                lock.release()
            }
        } catch (e: Exception) {
            Log.w(TAG, "Wake lock transition failed: ${e.message}")
        }
    }
}
