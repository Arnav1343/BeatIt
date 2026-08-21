package com.beatit.app

import android.content.Context
import android.util.Log
import kotlinx.coroutines.*
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.sync.Semaphore
import okhttp3.OkHttpClient
import okhttp3.Protocol
import okhttp3.Request
import java.io.File
import java.io.IOException
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * Optimized batch download pipeline with:
 * - 8 concurrent downloads (adaptive: drops to 3 on rate-limit, ramps back up)
 * - IDM-style segmented downloads via SegmentedDownloader
 * - Pre-fetch stream URLs for next 5 tracks while downloading
 * - Piped instance fallback via YoutubeHelper
 * - Request spacing to avoid burst patterns
 */
object BatchManager {
    private const val TAG = "BatchManager"
    private const val MAX_CONCURRENT = 8
    private const val MIN_CONCURRENT = 2
    private const val MAX_RETRIES = 3
    private const val HEALTH_CHECK_INTERVAL_MS = 60_000L
    private const val WATCHDOG_TIMEOUT_MS = 90_000L  // Longer for segmented downloads
    private const val REQUEST_SPACING_MS = 250L  // Delay between starting new downloads

    private val executor = Executors.newFixedThreadPool(MAX_CONCURRENT)
    private val activeWorkerCount = AtomicInteger(0)
    private val mutex = Mutex()

    // Adaptive concurrency control
    private var currentMaxConcurrent = MAX_CONCURRENT
    private var isRecovering = false
    private var globalRateLimitUntil = 0L
    private var globalConsecutive429s = 0
    private var lastSuccessTime = 0L

    private val trackWatchdogs = ConcurrentHashMap<String, Long>()

    private lateinit var dao: BatchDao
    private lateinit var youtubeHelper: YoutubeHelper
    private lateinit var musicDir: File

    fun init(context: Context, musicDir: File) {
        val database = AppDatabase.getDatabase(context)
        this.dao = database.batchDao()
        this.youtubeHelper = YoutubeHelper()
        this.musicDir = musicDir

        CoroutineScope(Dispatchers.Default).launch {
            while (true) {
                delay(HEALTH_CHECK_INTERVAL_MS)
                runHealthMonitor()
            }
        }

        // Adaptive ramp-up: gradually increase concurrency after sustained success
        CoroutineScope(Dispatchers.Default).launch {
            while (true) {
                delay(30_000)
                if (currentMaxConcurrent < MAX_CONCURRENT && 
                    globalConsecutive429s == 0 &&
                    System.currentTimeMillis() - lastSuccessTime < 60_000) {
                    currentMaxConcurrent = (currentMaxConcurrent + 1).coerceAtMost(MAX_CONCURRENT)
                    Log.d(TAG, "📈 Ramping up concurrency to $currentMaxConcurrent")
                }
            }
        }

        runRecovery()
        startDispatchLoop()
    }

    private fun startDispatchLoop() {
        CoroutineScope(Dispatchers.IO).launch {
            while (true) {
                if (System.currentTimeMillis() < globalRateLimitUntil) {
                    delay(5000)
                    continue
                }

                if (activeWorkerCount.get() >= currentMaxConcurrent || isRecovering) {
                    delay(500)
                    continue
                }

                dispatchNextIfPossible()
                delay(REQUEST_SPACING_MS)  // Spacing between dispatches
            }
        }
    }

    private suspend fun dispatchNextIfPossible() {
        mutex.withLock {
            CoroutineScope(Dispatchers.IO).launch {
                val nextTrack = dao.getQueuedTracks().firstOrNull() ?: return@launch
                
                if (nextTrack.status != TrackStatus.QUEUED) return@launch
                
                nextTrack.status = TrackStatus.DISPATCHING
                nextTrack.updatedAt = System.currentTimeMillis()
                dao.updateTrack(nextTrack)
                
                activeWorkerCount.incrementAndGet()
                
                // Pre-fetch stream URLs for the NEXT tracks while this one downloads
                prefetchUpcoming()
                
                executor.submit {
                    runWorker(nextTrack.id)
                }
            }
        }
    }

    /**
     * Pre-fetch stream URLs for the next 5 queued tracks.
     * This runs in background so by the time a track starts downloading,
     * its stream URL is already cached.
     */
    private fun prefetchUpcoming() {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val upcoming = dao.getQueuedTracks().take(5)
                for (track in upcoming) {
                    val videoUrl = track.youtubeVideoId ?: continue
                    if (!youtubeHelper.isPrefetched(videoUrl)) {
                        youtubeHelper.prefetchStreamInfo(videoUrl)
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Prefetch failed: ${e.message}")
            }
        }
    }

    private fun runWorker(trackId: String) {
        val track = runBlocking { dao.getTrack(trackId) } ?: run {
            activeWorkerCount.decrementAndGet()
            return
        }

        // Downloading is work worth staying awake for; the lock drops again
        // in the finally below.
        PowerGate.workStarted()
        try {
            track.status = TrackStatus.DOWNLOADING
            track.updatedAt = System.currentTimeMillis()
            runBlocking { dao.updateTrack(track) }
            
            val videoUrl = track.youtubeVideoId ?: throw IOException("No Video ID")
            Log.d(TAG, "⬇ Downloading: ${track.title}")
            val (streamUrl, streamError) = youtubeHelper.getAudioStreamUrl(videoUrl)
            
            if (streamUrl == null) throw IOException(streamError ?: "Stream extraction failed")

            val extension = "opus"  // Native format, no transcoding needed
            val finalFile = File(musicDir, "${sanitize(track.title)}.$extension")
            val tempFile = File(musicDir, "${sanitize(track.title)}.$extension.tmp")
            track.outputFilePath = finalFile.absolutePath
            
            // Use segmented downloader for maximum speed
            runBlocking {
                SegmentedDownloader.download(streamUrl, tempFile) { progress ->
                    track.totalBytes = progress.totalBytes
                    track.bytesDownloaded = progress.downloadedBytes
                    trackWatchdogs[track.id] = System.currentTimeMillis()
                    runBlocking { dao.updateTrack(track) }
                }
            }
            
            if (tempFile.renameTo(finalFile)) {
                // Cover art comes free on this path — thumbnailUrl was captured
                // during extraction. Best-effort; never fails the track.
                try {
                    ArtworkStore.saveFrom(track.thumbnailUrl, musicDir, finalFile.name)
                } catch (e: Exception) {
                    Log.w(TAG, "Artwork save failed for ${track.title}: ${e.message}")
                }

                track.status = TrackStatus.COMPLETED
                track.updatedAt = System.currentTimeMillis()
                runBlocking { dao.updateTrack(track) }
                runBlocking { updateBatchState(track.batchId) }
                globalConsecutive429s = 0
                lastSuccessTime = System.currentTimeMillis()
                Log.d(TAG, "✓ Completed: ${track.title}")
                
                // Ramp up if we've been stable
                if (currentMaxConcurrent < MAX_CONCURRENT) {
                    currentMaxConcurrent = (currentMaxConcurrent + 1).coerceAtMost(MAX_CONCURRENT)
                }
            } else {
                throw IOException("Failed to rename temp file")
            }
        } catch (e: Exception) {
            handleWorkerFailure(track, e)
        } finally {
            PowerGate.workFinished()
            activeWorkerCount.decrementAndGet()
            trackWatchdogs.remove(track.id)
        }
    }

    private fun handleWorkerFailure(track: Track, e: Exception) {
        val msg = e.message ?: "Unknown error"
        Log.e(TAG, "✗ Failed: ${track.title}: $msg")
        
        if (msg.contains("429") || msg.contains("403")) {
            globalConsecutive429s++
            // Adaptive: reduce concurrency on rate limit
            currentMaxConcurrent = (currentMaxConcurrent / 2).coerceAtLeast(MIN_CONCURRENT)
            val cooldown = when {
                globalConsecutive429s <= 1 -> 15_000L  // 15s first time
                globalConsecutive429s <= 3 -> 30_000L  // 30s
                else -> 60_000L                        // 60s
            }
            globalRateLimitUntil = System.currentTimeMillis() + cooldown
            Log.w(TAG, "⚠ Rate limit! Concurrency → $currentMaxConcurrent, cooldown ${cooldown/1000}s")
        }

        track.retryCount++
        track.errorCode = msg
        runBlocking {
            if (track.retryCount < MAX_RETRIES) {
                track.status = TrackStatus.QUEUED
                track.updatedAt = System.currentTimeMillis()
                dao.updateTrack(track)
            } else {
                track.status = TrackStatus.FAILED
                track.updatedAt = System.currentTimeMillis()
                dao.updateTrack(track)
            }
            updateBatchState(track.batchId)
        }
    }

    // ── Internal Helpers ──────────────────────────────────────────

    private fun runRecovery() {
        isRecovering = true
        CoroutineScope(Dispatchers.IO).launch {
            mutex.withLock {
                try {
                    val stalled = dao.getStalledTracks()
                    stalled.forEach { track ->
                        track.outputFilePath?.let { path ->
                            File(path).delete()
                            File("$path.tmp").delete()
                            // Clean up segment files
                            for (i in 0..3) File("$path.tmp.seg$i").delete()
                        }
                        track.status = TrackStatus.QUEUED
                        track.updatedAt = System.currentTimeMillis()
                        dao.updateTrack(track)
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Recovery failed", e)
                } finally {
                    isRecovering = false
                    activeWorkerCount.set(0)
                }
            }
        }
    }

    private fun runHealthMonitor() {
        val now = System.currentTimeMillis()
        trackWatchdogs.forEach { (id, lastTime) ->
            if (now - lastTime > WATCHDOG_TIMEOUT_MS) {
                Log.w(TAG, "Watchdog timeout for track $id. Requeuing.")
                runBlocking {
                    val track = dao.getTrack(id)
                    if (track != null) {
                        track.status = TrackStatus.QUEUED
                        track.updatedAt = now
                        dao.updateTrack(track)
                    }
                }
                trackWatchdogs.remove(id)
            }
        }
        
        if (activeWorkerCount.get() > 0 && trackWatchdogs.isEmpty()) {
            Log.e(TAG, "CRITICAL: activeWorkerCount > 0 but no active watchdogs. Resetting.")
            activeWorkerCount.set(0)
        }
    }

    private suspend fun updateBatchState(batchId: String) {
        val batchWithTracks = dao.getBatchWithTracks(batchId) ?: return
        val batch = batchWithTracks.batch
        val tracks = batchWithTracks.tracks

        val completed = tracks.count { it.status == TrackStatus.COMPLETED }
        val failed = tracks.count { it.status == TrackStatus.FAILED }
        val lowConf = tracks.count { it.status == TrackStatus.MATCHED_LOW_CONFIDENCE }
        val active = tracks.count { it.status in listOf(TrackStatus.MATCHING, TrackStatus.QUEUED, TrackStatus.DISPATCHING, TrackStatus.DOWNLOADING) }

        batch.completedCount = completed
        batch.failedCount = failed
        batch.updatedAt = System.currentTimeMillis()

        when {
            completed + failed == batch.totalTracks && lowConf == 0 -> batch.state = BatchState.COMPLETED
            failed == batch.totalTracks -> batch.state = BatchState.FAILED
            lowConf > 0 && active == 0 -> batch.state = BatchState.AWAITING_USER
            active > 0 -> batch.state = BatchState.DOWNLOADING
            else -> batch.state = BatchState.QUEUED
        }
        dao.updateBatch(batch)
    }

    /**
     * Public transition function for use by BeatItServer (accept/rematch/manual actions).
     */
    suspend fun transition(trackId: String, nextStatus: TrackStatus) {
        mutex.withLock {
            val track = dao.getTrack(trackId) ?: return
            track.status = nextStatus
            track.updatedAt = System.currentTimeMillis()
            dao.updateTrack(track)
            updateBatchState(track.batchId)
        }
    }

    private fun sanitize(name: String): String =
        name.replace(Regex("[^a-zA-Z0-9 _-]"), "").trim().take(80)

    data class ImportResult(val success: Boolean, val trackCount: Int = 0, val error: String? = null)

    suspend fun submitBatch(url: String, platform: SourcePlatform): ImportResult {
        if (isRecovering) return ImportResult(false, error = "System is recovering, please try again in a moment")

        val batch = BatchTask(state = BatchState.EXTRACTING)
        dao.insertBatch(batch)
        
        return try {
            Log.d(TAG, "Starting extraction for URL: $url, platform: $platform")
            val candidates = PlaylistExtractor.extract(url, platform)
            Log.d(TAG, "Extraction returned ${candidates.size} candidates")
            
            if (candidates.isEmpty()) {
                batch.state = BatchState.FAILED
                batch.errorCode = "Could not extract tracks from this URL."
                dao.updateBatch(batch)
                return ImportResult(false, error = batch.errorCode)
            }
            
            if (candidates.size > 500) {
                batch.state = BatchState.FAILED
                batch.errorCode = "Playlist too large (${candidates.size} tracks, max 500)"
                dao.updateBatch(batch)
                return ImportResult(false, error = batch.errorCode)
            }
            
            val tracks = candidates.map { c ->
                Track(
                    batchId = batch.id,
                    fingerprint = PlaylistExtractor.computeFingerprint(c.title, c.artist, c.durationSeconds),
                    title = c.title,
                    artist = c.artist,
                    durationSeconds = c.durationSeconds,
                    thumbnailUrl = c.thumbnailUrl,
                    sourcePlatform = c.sourcePlatform,
                    youtubeVideoId = c.sourceUrl
                )
            }
            dao.insertTracks(tracks)
            batch.totalTracks = tracks.size
            batch.state = BatchState.MATCHING
            dao.updateBatch(batch)
            
            Log.d(TAG, "Batch created with ${tracks.size} tracks, starting matching...")
            
            CoroutineScope(Dispatchers.IO).launch {
                processMatching(batch.id)
            }
            
            ImportResult(true, trackCount = tracks.size)
        } catch (e: Exception) {
            Log.e(TAG, "submitBatch failed: ${e.message}", e)
            batch.state = BatchState.FAILED
            batch.errorCode = e.message
            dao.updateBatch(batch)
            ImportResult(false, error = "Import failed: ${e.message}")
        }
    }

    /**
     * Process matching for tracks in a batch.
     * YouTube tracks skip matching (already have video URL).
     * Spotify/Apple tracks matched in parallel (3 concurrent searches).
     * Pre-fetches stream URLs for matched tracks immediately.
     */
    private suspend fun processMatching(batchId: String) {
        val tracks = dao.getTracksForBatch(batchId)
        val searchSemaphore = Semaphore(3)
        
        coroutineScope {
            tracks.map { track ->
                async(Dispatchers.IO) {
                    if (track.status != TrackStatus.EXTRACTED) return@async
                    
                    // Fast path: YouTube tracks already have the video URL
                    if (track.youtubeVideoId != null) {
                        Log.d(TAG, "⚡ Fast-match: ${track.title}")
                        track.matchConfidence = 1.0
                        track.status = TrackStatus.QUEUED
                        track.updatedAt = System.currentTimeMillis()
                        dao.updateTrack(track)
                        
                        // Pre-fetch the stream URL immediately
                        youtubeHelper.prefetchStreamInfo(track.youtubeVideoId!!)
                        return@async
                    }
                    
                    // Slow path: search YouTube
                    searchSemaphore.acquire()
                    try {
                        Log.d(TAG, "🔍 Searching: ${track.title} ${track.artist}")
                        track.status = TrackStatus.MATCHING
                        track.updatedAt = System.currentTimeMillis()
                        dao.updateTrack(track)
                        
                        val (videoId, confidence) = TrackMapper.mapTrack(track, dao, youtubeHelper)
                        track.youtubeVideoId = videoId
                        track.matchConfidence = confidence
                        
                        if (videoId == null) {
                            track.status = TrackStatus.FAILED
                            track.errorCode = "No match found"
                        } else if (confidence < 0.75) {
                            track.status = TrackStatus.MATCHED_LOW_CONFIDENCE
                        } else {
                            track.status = TrackStatus.QUEUED
                            // Pre-fetch stream URL
                            youtubeHelper.prefetchStreamInfo(videoId)
                        }
                        track.updatedAt = System.currentTimeMillis()
                        dao.updateTrack(track)
                    } finally {
                        searchSemaphore.release()
                    }
                }
            }.awaitAll()
        }
        
        updateBatchState(batchId)
    }
}
