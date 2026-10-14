package com.beatit.app

import android.app.Notification
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.graphics.drawable.Icon
import android.media.MediaMetadata
import android.media.session.MediaSession
import android.media.session.PlaybackState
import android.os.Handler
import android.os.Looper
import android.util.Log
import java.io.File

/**
 * Lockscreen and notification transport controls.
 *
 * Purely a view of NativePlayer: the player calls update() when its state
 * changes, and the session / notification buttons call back into it. Nothing
 * here depends on the Activity or the WebView being alive.
 */
object PlaybackSession {

    private const val TAG = "PlaybackSession"
    private const val NOTIF_ID = MusicServerService.NOTIF_ID
    private const val CHANNEL_ID = MusicServerService.CHANNEL_ID

    /** A position this far from where playback should be counts as a seek. */
    private const val SEEK_JUMP_MS = 2_000L

    const val ACTION_TOGGLE = "com.beatit.app.TOGGLE"
    const val ACTION_NEXT = "com.beatit.app.NEXT"
    const val ACTION_PREV = "com.beatit.app.PREV"

    private var appContext: Context? = null
    private var session: MediaSession? = null
    private val main = Handler(Looper.getMainLooper())

    // Last state the page reported.
    private var title = ""
    private var artist = ""
    private var durationMs = 0L
    private var positionMs = 0L
    private var playing = false
    private var hasTrack = false
    private var filename = ""
    private var lastStateAt = 0L

    // Cover art for the lockscreen. Decoded off the main thread and cached,
    // because publish() runs on the main looper and BitmapFactory on a
    // 640px JPEG is not something to do there on every state push.
    private val artExecutor = java.util.concurrent.Executors.newSingleThreadExecutor()
    private var artBitmap: android.graphics.Bitmap? = null
    private var artBitmapKey: String? = null
    private var artDecodePending: String? = null
    private var musicDir: File? = null

    fun init(context: Context, musicDirectory: File? = null) {
        musicDir = musicDirectory ?: musicDir
        if (session != null) return
        val ctx = context.applicationContext
        appContext = ctx

        session = MediaSession(ctx, "BeatIt").apply {
            setCallback(object : MediaSession.Callback() {
                override fun onPlay() = NativePlayer.resume()
                override fun onPause() = NativePlayer.pause()
                override fun onSkipToNext() = NativePlayer.next()
                override fun onSkipToPrevious() = NativePlayer.prev()
                override fun onStop() = NativePlayer.pause()
                override fun onSeekTo(pos: Long) = NativePlayer.seek(pos)
            })
        }
        Log.d(TAG, "MediaSession created")
    }

    /** Called by NativePlayer whenever its state changes. */
    fun update(
        newTitle: String, newArtist: String, newDuration: Long,
        newPosition: Long, newPlaying: Boolean, newHasTrack: Boolean,
        newFilename: String
    ) {
        val changed = synchronized(this) {
            val now = android.os.SystemClock.elapsedRealtime()

            // Where the lockscreen scrubber already thinks we are, since the
            // session advances it on its own while PLAYING.
            val expected = positionMs + if (playing) now - lastStateAt else 0L
            val seeked = kotlin.math.abs(newPosition - expected) > SEEK_JUMP_MS

            val changed = newTitle != title || newArtist != artist || newDuration != durationMs ||
                newPlaying != playing || newHasTrack != hasTrack || newFilename != filename || seeked

            title = newTitle; artist = newArtist; durationMs = newDuration
            positionMs = newPosition; playing = newPlaying; hasTrack = newHasTrack
            filename = newFilename
            lastStateAt = now
            changed
        }

        // The only place that knows whether audio is actually running, so it
        // is also what decides whether the wake lock is warranted. Runs on
        // every push, changed or not: each one renews the lock's timeout.
        PowerGate.setPlaying(playing && hasTrack)

        // The page pushes about once a second while playing, but a re-post of
        // the notification and metadata every second makes OEM lockscreens
        // flicker and can eat button taps. The session extrapolates position
        // by itself, so only a real change (track, play/pause, seek) needs
        // publishing.
        if (!changed) return

        // Arrives on a WebView JavaScript thread; the session and the
        // notification are main-thread business.
        main.post { publish() }
    }

    /** Handles the notification button intents routed through the service. */
    fun handleAction(action: String?) {
        when (action) {
            ACTION_TOGGLE -> NativePlayer.toggle()
            ACTION_NEXT -> NativePlayer.next()
            ACTION_PREV -> NativePlayer.prev()
        }
    }

    fun buildIdleNotification(): Notification = baseNotification()
        .setContentTitle("BeatIt")
        .setContentText("BeatIt is running")
        .build()

    fun release() {
        session?.release()
        session = null
        appContext = null
    }

    // ── Internals ───────────────────────────────────────────────────

    /**
     * Cached cover art for [name], or null if it isn't decoded yet.
     *
     * Never decodes inline: publish() runs on the main looper. On a miss it
     * schedules the decode and re-publishes when the bitmap lands, so the
     * first push of a track shows text and the next one carries the art.
     */
    private fun artworkFor(name: String): android.graphics.Bitmap? {
        if (name.isEmpty()) return null
        if (artBitmapKey == name && artBitmap != null) return artBitmap

        val dir = musicDir ?: return null
        val file = ArtworkStore.artFile(dir, name)
        // Not fetched yet. Return without caching the miss: the lazy lookup
        // often lands a second or two after playback starts, and a cached
        // negative would mean this track never showed art at all.
        if (!file.exists()) return null
        if (artDecodePending == name) return null

        artDecodePending = name
        artExecutor.submit {
            val bmp = run {
                try {
                    // Roughly lockscreen-sized; full-res covers are wasted here.
                    val bounds = android.graphics.BitmapFactory.Options().apply {
                        inJustDecodeBounds = true
                    }
                    android.graphics.BitmapFactory.decodeFile(file.absolutePath, bounds)
                    var sample = 1
                    while (bounds.outWidth / sample > 512) sample *= 2
                    android.graphics.BitmapFactory.decodeFile(
                        file.absolutePath,
                        android.graphics.BitmapFactory.Options().apply { inSampleSize = sample }
                    )
                } catch (e: Exception) {
                    Log.w(TAG, "Art decode failed for $name: ${e.message}")
                    null
                }
            }

            main.post {
                artDecodePending = null
                if (bmp != null) {
                    artBitmap = bmp
                    artBitmapKey = name
                    publish()
                }
            }
        }
        return null
    }

    private fun publish() {
        val s = session ?: return

        if (!hasTrack) {
            s.isActive = false
            notify(buildIdleNotification())
            return
        }

        val art = artworkFor(filename)
        s.setMetadata(
            MediaMetadata.Builder()
                .putString(MediaMetadata.METADATA_KEY_TITLE, title)
                .putString(MediaMetadata.METADATA_KEY_ARTIST, artist)
                .putLong(MediaMetadata.METADATA_KEY_DURATION, durationMs)
                // A Bitmap, not an ART_URI: SystemUI is a different process
                // and shouldn't have to reach our loopback HTTP server.
                .apply { if (art != null) putBitmap(MediaMetadata.METADATA_KEY_ALBUM_ART, art) }
                .build()
        )

        s.setPlaybackState(
            PlaybackState.Builder()
                .setActions(
                    PlaybackState.ACTION_PLAY or
                        PlaybackState.ACTION_PAUSE or
                        PlaybackState.ACTION_PLAY_PAUSE or
                        PlaybackState.ACTION_SKIP_TO_NEXT or
                        PlaybackState.ACTION_SKIP_TO_PREVIOUS or
                        PlaybackState.ACTION_SEEK_TO or
                        PlaybackState.ACTION_STOP
                )
                .setState(
                    if (playing) PlaybackState.STATE_PLAYING else PlaybackState.STATE_PAUSED,
                    positionMs,
                    // 0 while paused, or the scrubber keeps creeping forward.
                    if (playing) 1.0f else 0f
                )
                .build()
        )

        s.isActive = true
        notify(buildMediaNotification(s))
    }

    private fun buildMediaNotification(s: MediaSession): Notification {
        val style = Notification.MediaStyle()
            .setMediaSession(s.sessionToken)
            .setShowActionsInCompactView(0, 1, 2)

        return baseNotification()
            .setContentTitle(if (title.isNotEmpty()) title else "BeatIt")
            .setContentText(artist)
            // MediaStyle feeds the lockscreen from session metadata, but the
            // collapsed notification draws its own large icon.
            // Only this track's cover: artBitmap lingers from the previous
            // track while the new one decodes, or forever if it has no art.
            .apply { artBitmap?.takeIf { artBitmapKey == filename }?.let { setLargeIcon(it) } }
            .addAction(action(android.R.drawable.ic_media_previous, "Previous", ACTION_PREV))
            .addAction(
                if (playing) action(android.R.drawable.ic_media_pause, "Pause", ACTION_TOGGLE)
                else action(android.R.drawable.ic_media_play, "Play", ACTION_TOGGLE)
            )
            .addAction(action(android.R.drawable.ic_media_next, "Next", ACTION_NEXT))
            .setStyle(style)
            .build()
    }

    private fun baseNotification(): Notification.Builder {
        val ctx = appContext!!
        val open = PendingIntent.getActivity(
            ctx, 0, Intent(ctx, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE
        )
        return Notification.Builder(ctx, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_media_play)
            .setContentIntent(open)
            .setOngoing(true)
            .setVisibility(Notification.VISIBILITY_PUBLIC)   // needed on the lockscreen
    }

    /**
     * Buttons route through the service rather than a separate receiver.
     * It is already running in the foreground, so starting it again with an
     * action is safe from the background.
     */
    private fun action(iconRes: Int, label: String, action: String): Notification.Action {
        val ctx = appContext!!
        val intent = Intent(ctx, MusicServerService::class.java).setAction(action)
        val pi = PendingIntent.getService(
            ctx, action.hashCode(), intent, PendingIntent.FLAG_IMMUTABLE
        )
        return Notification.Action.Builder(
            Icon.createWithResource(ctx, iconRes), label, pi
        ).build()
    }

    private fun notify(n: Notification) {
        val ctx = appContext ?: return
        (ctx.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager)
            .notify(NOTIF_ID, n)
    }
}
