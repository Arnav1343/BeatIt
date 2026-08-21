package com.beatit.app

import android.content.Context
import android.os.Build
import android.util.Log
import java.io.File
import java.io.PrintWriter
import java.io.StringWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Appends every uncaught exception to a file on external app storage, then
 * hands off to whatever handler was installed before.
 *
 * The point is diagnosis on devices we cannot attach to. Crashes here are
 * reported from specific phones — Redmi, Vivo — that behave differently from
 * anything a developer has in front of them, and adb is rarely an option for
 * the person reporting.
 *
 * It also draws a line that nothing else can: an OEM power manager or the
 * low-memory killer terminating the process produces no uncaught exception at
 * all. So "the app closed and crash.log gained an entry" means our code threw,
 * while "the app closed and crash.log is unchanged" means the system killed us
 * — two very different fixes, told apart by one file.
 *
 * The log is capped and rotated so it cannot grow without bound.
 */
object CrashLogger {

    private const val TAG = "CrashLogger"
    private const val FILE_NAME = "crash.log"
    private const val MAX_BYTES = 256 * 1024L

    fun install(context: Context) {
        val appContext = context.applicationContext
        val previous = Thread.getDefaultUncaughtExceptionHandler()

        Thread.setDefaultUncaughtExceptionHandler { thread, error ->
            try {
                write(appContext, thread, error)
            } catch (t: Throwable) {
                Log.e(TAG, "Could not write crash log", t)
            }
            // Never swallow it: the process still has to die, and whatever
            // handler was already there still has to see it.
            previous?.uncaughtException(thread, error)
        }
    }

    /**
     * Read from the package manager rather than BuildConfig: this module does
     * not enable the buildConfig feature, which AGP 8 turns off by default,
     * so the generated class is not there to reference.
     */
    private fun versionName(context: Context): String = try {
        val info = context.packageManager.getPackageInfo(context.packageName, 0)
        "${info.versionName}"
    } catch (e: Exception) {
        "unknown"
    }

    private fun write(context: Context, thread: Thread, error: Throwable) {
        val dir = context.getExternalFilesDir(null) ?: context.filesDir
        val file = File(dir, FILE_NAME)

        // Rotate before appending, so the newest crash always survives.
        if (file.exists() && file.length() > MAX_BYTES) {
            file.renameTo(File(dir, "$FILE_NAME.1"))
        }

        val stack = StringWriter().also { error.printStackTrace(PrintWriter(it)) }.toString()
        val stamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(Date())

        file.appendText(
            buildString {
                append("\n===== $stamp =====\n")
                append("device:  ${Build.MANUFACTURER} ${Build.MODEL}\n")
                append("android: ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})\n")
                append("app:     ${versionName(context)}\n")
                append("thread:  ${thread.name}\n")
                append(stack)
            }
        )
        Log.e(TAG, "Uncaught exception on ${thread.name} — written to ${file.absolutePath}", error)
    }
}
