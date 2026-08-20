package com.beatit.app

import android.app.*
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import java.io.File

class MusicServerService : Service() {

    private var server: BeatItServer? = null
    private var wakeLock: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        PlaybackSession.init(this)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            // Declare mediaPlayback so this service is exempt from the background
            // execution limits Android imposes on plain dataSync services — that's
            // what keeps WebView audio playback alive once the app is backgrounded.
            ServiceCompat.startForeground(
                this, 1, buildNotification("BeatIt is running"),
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PLAYBACK or
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
            )
        } else {
            startForeground(1, buildNotification("BeatIt is running"))
        }

        // Acquire a partial wake lock so downloads don't stall when screen is off
        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "BeatIt::DownloadLock")
        wakeLock?.acquire()

        val musicDir = getExternalFilesDir(android.os.Environment.DIRECTORY_MUSIC) 
            ?: File(filesDir, "Music")
        BatchManager.init(this, musicDir)

        server = BeatItServer(this, 8080)
        server?.start()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // Notification transport buttons come back in as service actions,
        // which avoids needing a separate exported receiver.
        PlaybackSession.handleAction(intent?.action)
        // If the system kills and restarts, keep the service running
        return START_STICKY
    }

    override fun onDestroy() {
        PlaybackSession.release()
        server?.stop()
        wakeLock?.let { if (it.isHeld) it.release() }
        super.onDestroy()
    }

    fun updateNotification(text: String) {
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        manager.notify(1, buildNotification(text))
    }

    private fun buildNotification(text: String): Notification {
        val intent = Intent(this, MainActivity::class.java)
        val pi = PendingIntent.getActivity(this, 0, intent, PendingIntent.FLAG_IMMUTABLE)
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("BeatIt")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_media_play)
            .setContentIntent(pi)
            .setOngoing(true)
            .build()
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID, "BeatIt Service", NotificationManager.IMPORTANCE_LOW
        ).apply { description = "BeatIt local music server" }
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        manager.createNotificationChannel(channel)
    }

    companion object {
        const val CHANNEL_ID = "beatit_service"
        const val NOTIF_ID = 1
    }
}
