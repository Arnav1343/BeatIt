package com.beatit.app

import org.schabi.newpipe.extractor.NewPipe
import org.schabi.newpipe.extractor.ServiceList
import org.schabi.newpipe.extractor.downloader.Downloader
import org.schabi.newpipe.extractor.downloader.Request
import org.schabi.newpipe.extractor.downloader.Response
import org.schabi.newpipe.extractor.search.SearchInfo
import org.schabi.newpipe.extractor.stream.StreamInfo
import org.schabi.newpipe.extractor.stream.StreamInfoItem
import okhttp3.OkHttpClient
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.MediaType.Companion.toMediaType
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.TimeUnit

class YoutubeHelper {

    init {
        if (!isInitialized) {
            NewPipe.init(OkHttpDownloader.getInstance())
            isInitialized = true
        }
    }

    // ── StreamInfo cache with 1-hour TTL ───────────────────────────
    private data class CachedStream(
        val streamUrl: String,
        val timestamp: Long = System.currentTimeMillis()
    ) {
        fun isExpired() = System.currentTimeMillis() - timestamp > CACHE_TTL_MS
    }

    // ── Extract best stream URL from StreamInfo ────────────────────
    private fun extractBestUrl(info: StreamInfo): String? {
        return info.audioStreams
            ?.filter { it.content.isNotEmpty() }
            ?.maxByOrNull { it.averageBitrate }
            ?.content
            ?: info.videoStreams
                ?.filter { it.content.isNotEmpty() }
                ?.firstOrNull()
                ?.content
    }

    /**
     * Pre-fetch stream info in background. Stores a Future so getAudioStreamUrl()
     * can await it instead of starting a duplicate extraction.
     */
    fun prefetchStreamInfo(videoUrl: String) {
        sweepCache()
        val cached = streamCache[videoUrl]
        if (cached != null && !cached.isExpired()) return
        if (pendingFetches.containsKey(videoUrl)) return

        val future = prefetchExecutor.submit<String?> {
            try {
                val info = StreamInfo.getInfo(ServiceList.YouTube, videoUrl)
                val url = extractBestUrl(info)
                if (url != null) {
                    streamCache[videoUrl] = CachedStream(url)
                }
                url
            } catch (_: Exception) {
                null
            } finally {
                pendingFetches.remove(videoUrl)
            }
        }
        pendingFetches[videoUrl] = future
    }

    fun isPrefetched(videoUrl: String): Boolean {
        val cached = streamCache[videoUrl] ?: return false
        return !cached.isExpired()
    }

    /**
     * Get the best audio stream URL with full fallback chain:
     * 1. Cache hit → instant
     * 2. Pending prefetch → await
     * 3. Fresh YouTube extraction
     * 4. Piped instance fallback (if YouTube fails)
     */
    fun getAudioStreamUrl(videoUrl: String): Pair<String?, String?> {
        sweepCache()
        // 1. Cache hit
        val cached = streamCache[videoUrl]
        if (cached != null && !cached.isExpired()) {
            return Pair(cached.streamUrl, null)
        }

        // 2. Await pending prefetch
        val pending = pendingFetches[videoUrl]
        if (pending != null) {
            try {
                val url = pending.get(30, TimeUnit.SECONDS)
                if (url != null) return Pair(url, null)
            } catch (_: Exception) { }
        }

        // 3. Fresh YouTube extraction
        try {
            val info = StreamInfo.getInfo(ServiceList.YouTube, videoUrl)
            val url = extractBestUrl(info)
            if (url != null) {
                streamCache[videoUrl] = CachedStream(url)
                return Pair(url, null)
            }
        } catch (e: Exception) {
            android.util.Log.w("YoutubeHelper", "YouTube extraction failed: ${e.message}, trying Piped fallback...")
        }

        // 4. Piped fallback — try multiple instances
        val videoId = extractVideoId(videoUrl)
        if (videoId != null) {
            val pipedResult = getStreamFromPiped(videoId)
            if (pipedResult != null) {
                streamCache[videoUrl] = CachedStream(pipedResult)
                return Pair(pipedResult, null)
            }
        }

        return Pair(null, "All extraction methods failed for this video")
    }

    /**
     * Extract video ID from various YouTube URL formats.
     */
    private fun extractVideoId(url: String): String? {
        val patterns = listOf(
            Regex("""(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/)([a-zA-Z0-9_-]{11})"""),
            Regex("""^([a-zA-Z0-9_-]{11})$""")
        )
        for (pattern in patterns) {
            val match = pattern.find(url)
            if (match != null) return match.groupValues[1]
        }
        return null
    }

    /**
     * Try to get audio stream URL from Piped instances.
     * Piped is an open-source YouTube frontend with public API instances.
     */
    private fun getStreamFromPiped(videoId: String): String? {
        for (instance in PIPED_INSTANCES) {
            try {
                android.util.Log.d("YoutubeHelper", "Trying Piped instance: $instance")
                val client = SegmentedDownloader.httpClient
                val request = okhttp3.Request.Builder()
                    .url("$instance/streams/$videoId")
                    .header("User-Agent", "Mozilla/5.0")
                    .build()
                
                val response = client.newCall(request).execute()
                if (!response.isSuccessful) {
                    response.close()
                    continue
                }
                
                val body = response.body?.string() ?: continue
                val json = com.google.gson.JsonParser.parseString(body).asJsonObject
                
                val audioStreams = json.getAsJsonArray("audioStreams") ?: continue
                
                // Find the best audio stream by bitrate
                var bestUrl: String? = null
                var bestBitrate = 0
                
                for (stream in audioStreams) {
                    val streamObj = stream.asJsonObject
                    val streamUrl = streamObj.get("url")?.asString ?: continue
                    val bitrate = streamObj.get("bitrate")?.asInt ?: 0
                    val mimeType = streamObj.get("mimeType")?.asString ?: ""
                    
                    // Prefer audio-only streams (opus/webm or m4a)
                    if (mimeType.startsWith("audio/") && bitrate > bestBitrate) {
                        bestBitrate = bitrate
                        bestUrl = streamUrl
                    }
                }

                if (bestUrl != null) {
                    android.util.Log.d("YoutubeHelper", "✓ Piped stream found: ${bestBitrate}bps from $instance")
                    return bestUrl
                }
            } catch (e: Exception) {
                android.util.Log.w("YoutubeHelper", "Piped instance $instance failed: ${e.message}")
            }
        }
        return null
    }

    // ── Reject pattern for non-music content ─────────────────────
    private val REJECT_PATTERN = Regex(
        """(#shorts|shorts|cricket|wicket|ipl|match|highlights|reaction|gameplay|tutorial|podcast|vlog|unboxing|review|trailer|teaser|behind.the.scenes|interview|news|cooking|recipe|workout|fitness|compilation|prank|challenge|stream|full\s*album|full\s*movie)""",
        RegexOption.IGNORE_CASE
    )

    // ── Search ─────────────────────────────────────────────────────
    fun search(query: String, limit: Int = 5): List<StreamInfoItem> {
        return try {
            val musicQuery = "$query song"
            val searchInfo = SearchInfo.getInfo(
                ServiceList.YouTube,
                ServiceList.YouTube.searchQHFactory.fromQuery(musicQuery)
            )
            searchInfo.relatedItems
                .filterIsInstance<StreamInfoItem>()
                .filter { item ->
                    // Filter shorts by URL (/shorts/)
                    val url = item.url ?: ""
                    if (url.contains("/shorts/")) return@filter false
                    // Filter shorts by isShortFormContent (API detection)
                    if (item.isShortFormContent) return@filter false
                    // Duration: skip very long (>15min) but allow short songs
                    val dur = item.duration
                    if (dur > 900) return@filter false
                    // Keyword filter: skip non-music content
                    val title = item.name ?: ""
                    !REJECT_PATTERN.containsMatchIn(title)
                }
                .take(limit)
        } catch (e: Exception) {
            emptyList()
        }
    }

    /**
     * Drops entries past their TTL.
     *
     * Nothing ever removed them before — an expired entry was only replaced
     * if the same video was requested again — so the map grew for as long as
     * the process lived. Small per entry, but it is one of several things
     * that only misbehave after prolonged use.
     */
    private fun sweepCache() {
        if (streamCache.size < CACHE_SWEEP_THRESHOLD) return
        streamCache.entries.removeAll { it.value.isExpired() }
    }

    fun suggestions(query: String): List<Map<String, Any?>> {
        return search(query, 8).map { item ->
            mapOf(
                "title" to item.name,
                "artist" to (item.uploaderName ?: ""),
                "duration" to item.duration,
                "url" to item.url,
                "thumbnail" to (item.thumbnails.firstOrNull()?.url ?: "")
            )
        }
    }

    companion object {
        private var isInitialized = false
        private const val CACHE_TTL_MS = 3600_000L // 1 hour
        private const val CACHE_SWEEP_THRESHOLD = 64
        private val streamCache = ConcurrentHashMap<String, CachedStream>()
        private val pendingFetches = ConcurrentHashMap<String, Future<String?>>()
        private val prefetchExecutor = Executors.newFixedThreadPool(5)
        
        // Piped instances for fallback when YouTube rate-limits
        private val PIPED_INSTANCES = listOf(
            "https://pipedapi.kavin.rocks",
            "https://pipedapi.adminforge.de",
            "https://api.piped.projectsegfau.lt"
        )
    }
}

/**
 * OkHttp-based downloader for NewPipe Extractor.
 */
class OkHttpDownloader private constructor() : Downloader() {

    private val client = OkHttpClient.Builder()
        .connectionPool(okhttp3.ConnectionPool(5, 60, TimeUnit.SECONDS))
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .followRedirects(true)
        .followSslRedirects(true)
        .protocols(listOf(okhttp3.Protocol.HTTP_2, okhttp3.Protocol.HTTP_1_1))
        .build()

    override fun execute(request: Request): Response {
        val requestBuilder = okhttp3.Request.Builder().url(request.url())
        request.headers().forEach { (key, values) ->
            values.forEach { value -> requestBuilder.addHeader(key, value) }
        }

        request.dataToSend()?.let { body ->
            requestBuilder.post(body.toRequestBody("application/json".toMediaType()))
        } ?: requestBuilder.get()

        val response = client.newCall(requestBuilder.build()).execute()
        val responseBody = response.body?.string() ?: ""
        val headers = mutableMapOf<String, List<String>>()
        response.headers.names().forEach { name ->
            headers[name] = response.headers.values(name)
        }
        return Response(response.code, response.message, headers, responseBody, request.url())
    }

    companion object {
        @Volatile private var instance: OkHttpDownloader? = null
        fun getInstance(): OkHttpDownloader =
            instance ?: synchronized(this) {
                instance ?: OkHttpDownloader().also { instance = it }
            }
    }
}
