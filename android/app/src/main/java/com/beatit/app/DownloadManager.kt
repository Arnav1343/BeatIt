package com.beatit.app

import android.content.Context
import android.util.Log
import okhttp3.*
import java.io.BufferedOutputStream
import java.io.File
import java.io.IOException
import java.io.RandomAccessFile
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

data class TaskStatus(
    val status: String,          // extracting | downloading | paused | done | error
    val percent: Int = 0,
    val result: Map<String, Any>? = null,
    val error: String? = null
)

class DownloadManager(
    private val context: Context,
    private val musicDir: File
) {
    companion object {
        private const val TAG = "DownloadManager"
        private const val MAX_RETRIES = 15            // Total retry attempts before giving up
        private const val INITIAL_RETRY_DELAY_MS = 2000L  // 2 seconds
        private const val MAX_RETRY_DELAY_MS = 30000L     // 30 seconds max backoff
        private const val BUFFER_SIZE = 262144             // 256 KB
    }

    private val tasks = ConcurrentHashMap<String, TaskStatus>()
    private val executor = Executors.newCachedThreadPool()
    private val youtubeHelper = YoutubeHelper()

    // Shared OkHttpClient optimised for large downloads
    private val httpClient = OkHttpClient.Builder()
        .connectionPool(ConnectionPool(5, 60, TimeUnit.SECONDS))
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .followRedirects(true)
        .followSslRedirects(true)
        .retryOnConnectionFailure(true)
        .protocols(listOf(Protocol.HTTP_2, Protocol.HTTP_1_1))
        .build()

    fun startDownload(
        videoUrl: String,
        title: String,
        quality: Int,
        codec: String,
        thumbnailUrl: String? = null
    ): String {
        val taskId = UUID.randomUUID().toString()
        tasks[taskId] = TaskStatus("extracting", 0)

        executor.submit {
            runDownload(taskId, videoUrl, title, quality, codec, thumbnailUrl)
        }
        return taskId
    }

    fun getProgress(taskId: String): Map<String, Any?>? {
        val t = tasks[taskId] ?: return null
        return buildMap {
            put("status", t.status)
            put("percent", t.percent)
            t.result?.let { put("result", it) }
            t.error?.let { put("error", it) }
        }
    }

    private fun runDownload(
        taskId: String,
        videoUrl: String,
        title: String,
        quality: Int,
        codec: String,
        thumbnailUrl: String? = null
    ) {
        val extension = if (codec == "opus") "opus" else "mp3"
        val finalFile = File(musicDir, "${sanitize(title)}.$extension")
        val tempFile = File(musicDir, "${sanitize(title)}.$extension.tmp")
        try {
            // Step 1: Get audio stream URL (may be instant if pre-fetched)
            tasks[taskId] = TaskStatus("extracting", 0)
            val (streamUrl, streamError) = youtubeHelper.getAudioStreamUrl(videoUrl)
            if (streamUrl == null) {
                throw IOException(streamError ?: "Could not get audio stream URL")
            }

            // Step 2: Download with automatic pause/resume on network loss
            tasks[taskId] = TaskStatus("downloading", 5)
            downloadWithRetry(streamUrl, tempFile, taskId, videoUrl)

            // Step 3: Move temp file to final location (now it appears in library)
            tempFile.renameTo(finalFile)

            // Step 4: Save the cover art beside it. Strictly best-effort —
            // the download has already succeeded and must not be failed by it.
            try {
                ArtworkStore.saveFrom(thumbnailUrl, musicDir, finalFile.name)
            } catch (e: Exception) {
                android.util.Log.w("DownloadManager", "Artwork save failed: ${e.message}")
            }

            val sizeHuman = humanSize(finalFile.length())
            tasks[taskId] = TaskStatus(
                status = "done",
                percent = 100,
                result = mapOf(
                    "filename" to finalFile.name,
                    "title" to title,
                    "size_human" to sizeHuman
                )
            )
        } catch (e: Exception) {
            tempFile.delete()
            tasks[taskId] = TaskStatus("error", error = e.message ?: "Unknown error")
        }
    }

    /**
     * Downloads a file with automatic pause/resume on network errors.
     * Uses HTTP Range requests to resume from where we left off.
     * Retries with exponential backoff up to MAX_RETRIES times.
     */
    private fun downloadWithRetry(
        streamUrl: String,
        dest: File,
        taskId: String,
        videoUrl: String
    ) {
        var retryCount = 0
        var currentUrl = streamUrl
        var totalBytes = -1L

        while (retryCount <= MAX_RETRIES) {
            val downloadedSoFar = if (dest.exists()) dest.length() else 0L

            // If we already know totalBytes and have it all, we're done
            if (totalBytes > 0 && downloadedSoFar >= totalBytes) {
                Log.d(TAG, "Download complete: $downloadedSoFar bytes")
                return
            }

            try {
                // Build request with Range header if resuming
                val requestBuilder = Request.Builder()
                    .url(currentUrl)
                    .header("User-Agent", "Mozilla/5.0")
                    .header("Accept", "*/*")
                    .header("Accept-Encoding", "identity")
                    .header("Connection", "keep-alive")

                if (downloadedSoFar > 0) {
                    requestBuilder.header("Range", "bytes=$downloadedSoFar-")
                    Log.d(TAG, "Resuming download from byte $downloadedSoFar")
                    tasks[taskId] = TaskStatus("downloading", computePercent(downloadedSoFar, totalBytes))
                }

                val response = httpClient.newCall(requestBuilder.build()).execute()

                if (!response.isSuccessful && response.code != 206) {
                    // If Range not supported and we have partial data, the server might
                    // return 200 with full content — restart from scratch
                    if (response.code == 200 && downloadedSoFar > 0) {
                        Log.d(TAG, "Server doesn't support Range, restarting download")
                        dest.delete()
                    } else if (response.code == 416) {
                        // Range not satisfiable — file already complete, or URL expired
                        // Try refreshing the stream URL
                        Log.d(TAG, "Range not satisfiable, refreshing stream URL")
                        val (newUrl, err) = youtubeHelper.getAudioStreamUrl(videoUrl)
                        if (newUrl != null) {
                            currentUrl = newUrl
                            dest.delete() // Start fresh with new URL
                            retryCount++
                            continue
                        } else {
                            throw IOException("Stream URL expired: ${err ?: "unknown"}")
                        }
                    } else {
                        throw IOException("HTTP ${response.code}")
                    }
                }

                val body = response.body ?: throw IOException("Empty response body")

                // Determine total size from Content-Length or Content-Range
                if (totalBytes <= 0) {
                    val contentRange = response.header("Content-Range")
                    if (contentRange != null) {
                        // Format: bytes 1234-5678/9012
                        val total = contentRange.substringAfter("/", "").toLongOrNull()
                        if (total != null && total > 0) totalBytes = total
                    }
                    if (totalBytes <= 0) {
                        totalBytes = body.contentLength() + downloadedSoFar
                    }
                }

                // Append to file (resume-safe)
                val raf = RandomAccessFile(dest, "rw")
                raf.seek(if (response.code == 206) downloadedSoFar else 0L)

                try {
                    body.byteStream().use { input ->
                        val buffer = ByteArray(BUFFER_SIZE)
                        var written = if (response.code == 206) downloadedSoFar else 0L
                        var lastUpdate = System.currentTimeMillis()
                        var read: Int
                        while (input.read(buffer).also { read = it } != -1) {
                            raf.write(buffer, 0, read)
                            written += read

                            val now = System.currentTimeMillis()
                            if (now - lastUpdate > 200) {
                                tasks[taskId] = TaskStatus("downloading", computePercent(written, totalBytes))
                                lastUpdate = now
                            }
                        }
                    }
                } finally {
                    raf.close()
                }

                // If we get here, download completed successfully
                Log.d(TAG, "Download finished: ${dest.length()} bytes")
                return

            } catch (e: IOException) {
                retryCount++
                if (retryCount > MAX_RETRIES) {
                    throw IOException("Download failed after $MAX_RETRIES retries: ${e.message}")
                }

                // Calculate backoff delay: 2s, 4s, 8s, ... capped at 30s
                val delay = (INITIAL_RETRY_DELAY_MS * (1L shl (retryCount - 1).coerceAtMost(4)))
                    .coerceAtMost(MAX_RETRY_DELAY_MS)

                val currentBytes = if (dest.exists()) dest.length() else 0L
                Log.d(TAG, "Network error (attempt $retryCount/$MAX_RETRIES), " +
                        "pausing ${delay/1000}s, downloaded=${humanSize(currentBytes)}: ${e.message}")

                tasks[taskId] = TaskStatus(
                    "paused",
                    computePercent(currentBytes, totalBytes),
                    error = "Network lost, retrying in ${delay/1000}s... (${retryCount}/$MAX_RETRIES)"
                )

                Thread.sleep(delay)

                // After sleeping, update status back to downloading
                tasks[taskId] = TaskStatus("downloading", computePercent(currentBytes, totalBytes))
            }
        }
    }

    private fun computePercent(downloaded: Long, total: Long): Int {
        if (total <= 0) return 5
        return ((downloaded * 95) / total).toInt().coerceIn(5, 99)
    }

    private fun sanitize(name: String): String =
        name.replace(Regex("[^a-zA-Z0-9 _-]"), "").trim().take(80)

    private fun humanSize(bytes: Long): String = when {
        bytes < 1024 -> "$bytes B"
        bytes < 1024 * 1024 -> "${bytes / 1024} KB"
        else -> "${"%.1f".format(bytes / (1024.0 * 1024.0))} MB"
    }
}
