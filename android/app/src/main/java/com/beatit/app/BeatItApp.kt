package com.beatit.app

import android.app.Application

/**
 * Exists so the crash logger is installed before anything else runs, in every
 * process the app starts — the activity and the service both.
 */
class BeatItApp : Application() {
    override fun onCreate() {
        super.onCreate()
        CrashLogger.install(this)
    }
}
