package com.beatit.app

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioManager
import android.media.MediaPlayer
import android.os.PowerManager
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * The actual audio player, owned by the foreground service rather than the
 * WebView. Playback used to live in an <audio> element, so swiping the app
 * out of recents destroyed the Activity and with it the music, while the
 * notification stayed up with dead buttons.
 *
 * The page is now only a remote control and a view: it pushes the library
 * order here (setQueue) and polls state(). Track advancing happens in this
 * object, which is what lets the next song start with no UI alive.
 *
 * Everything is serialised on one lock. The page calls in on WebView
 * JavaScript threads, the session and notification buttons on the main
 * thread, and MediaPlayer's own callbacks on the main looper.
 */
object NativePlayer {

    private const val TAG = "NativePlayer"
    private const val RESTART_THRESHOLD_MS = 3_000

    private class Item(val filename: String, val title: String)

    private val lock = Any()
    private var ctx: Context? = null
    private var musicDir: File? = null
    private var audioManager: AudioManager? = null

    private var queue: List<Item> = emptyList()
    private var shuffle = false
    private var repeat = false

    private var current: Item? = null
    private var player: MediaPlayer? = null
    private var prepared = false

    /** What the user asked for; differs from the player's own state during focus loss. */
    private var wantPlaying = false
    private var pausedByFocusLoss = false
    private var focusRequest: AudioFocusRequest? = null
    private var noisyRegistered = false

    fun init(context: Context, musicDirectory: File) {
        synchronized(lock) {
            if (ctx != null) return
            ctx = context.applicationContext
            musicDir = musicDirectory
            audioManager = ctx!!.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        }
    }

    fun release() = synchronized(lock) {
        stopLocked()
        ctx = null
    }

    // ── Page → player ───────────────────────────────────────────────

    /** [json] is `[{"filename":..,"title":..}, ...]` in library order. */
    fun setQueue(json: String) = synchronized(lock) {
        try {
            val arr = JSONArray(json)
            queue = (0 until arr.length()).map {
                val o = arr.getJSONObject(it)
                Item(o.getString("filename"), o.optString("title", o.getString("filename")))
            }
        } catch (e: Exception) {
            Log.w(TAG, "Bad queue payload: ${e.message}")
        }
    }

    fun setModes(shuffleOn: Boolean, repeatOn: Boolean) = synchronized(lock) {
        shuffle = shuffleOn
        repeat = repeatOn
    }

    fun play(filename: String, title: String) = synchronized(lock) {
        startTrack(Item(File(filename).name, title))
    }

    fun resume() = synchronized(lock) { resumeInner() }

    fun pause() = synchronized(lock) { pauseLocked() }

    fun toggle() = synchronized(lock) {
        if (isPlayingLocked()) pauseLocked() else resumeInner()
    }

    fun next() = synchronized(lock) { advance(+1) }

    fun prev() = synchronized(lock) {
        val p = player
        if (p != null && prepared && p.currentPosition > RESTART_THRESHOLD_MS) {
            p.seekTo(0)
            publish()
        } else advance(-1)
    }

    fun seek(ms: Long) = synchronized(lock) {
        val p = player
        if (p != null && prepared) {
            p.seekTo(ms.toInt().coerceIn(0, p.duration))
            publish()
        }
    }

    /** The playing track was deleted from disk. */
    fun stopIfPlaying(filename: String) = synchronized(lock) {
        if (current?.filename == filename) stopLocked()
    }

    // ── Player → page ───────────────────────────────────────────────

    fun state(): String = synchronized(lock) {
        val p = player
        JSONObject()
            .put("hasTrack", current != null)
            .put("filename", current?.filename ?: "")
            .put("title", current?.title ?: "")
            .put("playing", wantPlaying)
            .put("positionMs", if (p != null && prepared) p.currentPosition else 0)
            .put("durationMs", if (p != null && prepared) p.duration else 0)
            .toString()
    }

    // ── Internals ───────────────────────────────────────────────────

    private fun isPlayingLocked() = wantPlaying

    private fun resumeInner() {
        if (current == null) { queue.firstOrNull()?.let { startTrack(it) }; return }
        wantPlaying = true
        if (!requestFocus()) { wantPlaying = false; publish(); return }
        pausedByFocusLoss = false
        if (prepared) player?.start()
        registerNoisy()
        publish()
    }

    private fun pauseLocked() {
        wantPlaying = false
        pausedByFocusLoss = false
        try { if (prepared) player?.pause() } catch (_: Exception) { }
        publish()
    }

    private fun startTrack(item: Item) {
        val dir = musicDir ?: return
        val file = File(dir, item.filename)
        releasePlayer()
        current = item
        if (!file.exists()) {
            Log.w(TAG, "Missing file: ${file.path}")
            wantPlaying = false
            publish()
            return
        }

        if (!requestFocus()) { wantPlaying = false; publish(); return }
        wantPlaying = true
        pausedByFocusLoss = false
        registerNoisy()

        val mp = MediaPlayer()
        player = mp
        prepared = false
        try {
            mp.setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                    .build()
            )
            // Keeps the CPU awake for as long as this player is playing, with
            // no expiry; PowerGate's lock times out after an hour.
            mp.setWakeMode(ctx, PowerManager.PARTIAL_WAKE_LOCK)
            mp.setDataSource(file.absolutePath)
            mp.setOnPreparedListener {
                synchronized(lock) {
                    if (player !== it) return@synchronized
                    prepared = true
                    if (wantPlaying && !pausedByFocusLoss) it.start()
                    publish()
                }
            }
            mp.setOnCompletionListener {
                synchronized(lock) {
                    if (player !== it) return@synchronized
                    if (repeat) { it.seekTo(0); it.start(); publish() } else advance(+1)
                }
            }
            mp.setOnErrorListener { p, what, extra ->
                synchronized(lock) {
                    if (player === p) {
                        Log.w(TAG, "MediaPlayer error $what/$extra on ${current?.filename}")
                        wantPlaying = false
                        prepared = false
                        publish()
                    }
                }
                true
            }
            mp.prepareAsync()
        } catch (e: Exception) {
            Log.w(TAG, "Could not start ${item.filename}: ${e.message}")
            wantPlaying = false
            publish()
        }
        publish()
    }

    private fun advance(dir: Int) {
        if (queue.isEmpty()) return
        val idx = queue.indexOfFirst { it.filename == current?.filename }
        val n = queue.size
        val target = when {
            dir > 0 && shuffle -> kotlin.random.Random.nextInt(n)
            dir > 0 -> (idx + 1) % n
            else -> (idx - 1 + n) % n
        }
        startTrack(queue[target])
    }

    private fun stopLocked() {
        releasePlayer()
        current = null
        wantPlaying = false
        pausedByFocusLoss = false
        abandonFocus()
        unregisterNoisy()
        publish()
    }

    private fun releasePlayer() {
        val p = player ?: return
        player = null
        prepared = false
        try { p.setOnCompletionListener(null); p.setOnErrorListener(null); p.release() } catch (_: Exception) { }
    }

    private fun publish() {
        val p = player
        val item = current
        PlaybackSession.update(
            newTitle = item?.title ?: "",
            newArtist = "BeatIt",
            newDuration = if (p != null && prepared) p.duration.toLong() else 0L,
            newPosition = if (p != null && prepared) p.currentPosition.toLong() else 0L,
            newPlaying = wantPlaying && !pausedByFocusLoss,
            newHasTrack = item != null,
            newFilename = item?.filename ?: ""
        )
        if (item == null) { abandonFocus(); unregisterNoisy() }
    }

    // ── Audio focus & headphone unplug ──────────────────────────────

    private val focusListener = AudioManager.OnAudioFocusChangeListener { change ->
        synchronized(lock) {
            when (change) {
                AudioManager.AUDIOFOCUS_LOSS -> pauseLocked()
                AudioManager.AUDIOFOCUS_LOSS_TRANSIENT -> {
                    if (wantPlaying) {
                        pausedByFocusLoss = true
                        try { player?.pause() } catch (_: Exception) { }
                        publish()
                    }
                }
                AudioManager.AUDIOFOCUS_LOSS_TRANSIENT_CAN_DUCK -> player?.setVolume(0.2f, 0.2f)
                AudioManager.AUDIOFOCUS_GAIN -> {
                    player?.setVolume(1f, 1f)
                    if (pausedByFocusLoss && wantPlaying) {
                        pausedByFocusLoss = false
                        if (prepared) player?.start()
                        publish()
                    }
                }
            }
        }
    }

    private fun requestFocus(): Boolean {
        val am = audioManager ?: return true
        if (focusRequest == null) {
            focusRequest = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN)
                .setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                        .build()
                )
                .setOnAudioFocusChangeListener(focusListener)
                .build()
        }
        return am.requestAudioFocus(focusRequest!!) == AudioManager.AUDIOFOCUS_REQUEST_GRANTED
    }

    private fun abandonFocus() {
        val am = audioManager ?: return
        focusRequest?.let { am.abandonAudioFocusRequest(it) }
    }

    private val noisyReceiver = object : BroadcastReceiver() {
        override fun onReceive(c: Context?, i: Intent?) {
            if (i?.action == AudioManager.ACTION_AUDIO_BECOMING_NOISY) pause()
        }
    }

    private fun registerNoisy() {
        val c = ctx ?: return
        if (noisyRegistered) return
        c.registerReceiver(noisyReceiver, IntentFilter(AudioManager.ACTION_AUDIO_BECOMING_NOISY))
        noisyRegistered = true
    }

    private fun unregisterNoisy() {
        val c = ctx ?: return
        if (!noisyRegistered) return
        try { c.unregisterReceiver(noisyReceiver) } catch (_: Exception) { }
        noisyRegistered = false
    }
}
