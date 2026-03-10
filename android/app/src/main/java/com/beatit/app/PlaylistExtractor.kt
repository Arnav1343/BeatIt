package com.beatit.app

import android.util.Log
import org.schabi.newpipe.extractor.ServiceList
import org.schabi.newpipe.extractor.playlist.PlaylistInfo
import org.schabi.newpipe.extractor.stream.StreamInfoItem
import org.jsoup.Jsoup
import com.google.gson.Gson
import com.google.gson.JsonParser
import java.io.IOException
import java.security.MessageDigest
import java.util.*
import kotlin.math.roundToInt

data class TrackCandidate(
    val title: String,
    val artist: String,
    val durationSeconds: Int?,
    val thumbnailUrl: String?,
    val sourcePlatform: SourcePlatform,
    val sourceUrl: String? = null  // YouTube video URL when extracted from YT playlists
)

object PlaylistExtractor {
    private const val TAG = "PlaylistExtractor"

    suspend fun extract(url: String, platform: SourcePlatform): List<TrackCandidate> {
        return when (platform) {
            SourcePlatform.YOUTUBE -> extractYoutube(url)
            SourcePlatform.SPOTIFY -> extractSpotify(url)
            SourcePlatform.APPLE_MUSIC -> extractAppleMusic(url)
        }
    }

    private fun extractYoutube(url: String): List<TrackCandidate> {
        return try {
            val info = PlaylistInfo.getInfo(ServiceList.YouTube, url)
            val tracks = mutableListOf<TrackCandidate>()
            
            var currentItems = info.relatedItems.filterIsInstance<StreamInfoItem>()
            tracks.addAll(currentItems.map { it.toCandidate() })

            var nextPage = info.nextPage
            var count = tracks.size
            while (nextPage != null && count < 500) {
                val page = PlaylistInfo.getMoreItems(ServiceList.YouTube, url, nextPage)
                val items = page.items.filterIsInstance<StreamInfoItem>()
                tracks.addAll(items.map { it.toCandidate() })
                nextPage = page.nextPage
                count = tracks.size
            }

            tracks.take(500)
        } catch (e: Exception) {
            Log.e(TAG, "YouTube extraction failed: ${e.message}", e)
            emptyList()
        }
    }

    private fun StreamInfoItem.toCandidate() = TrackCandidate(
        title = name ?: "Unknown",
        artist = uploaderName ?: "Unknown",
        durationSeconds = duration.toInt().takeIf { it > 0 },
        thumbnailUrl = thumbnails.firstOrNull()?.url,
        sourcePlatform = SourcePlatform.YOUTUBE,
        sourceUrl = url  // Carry the YouTube video URL so we can skip matching
    )

    private fun extractSpotify(url: String): List<TrackCandidate> {
        // Check if user is connected to Spotify
        if (!SpotifyAuth.isConnected()) {
            Log.e(TAG, "Spotify not connected — user needs to log in first")
            throw IOException("Connect your Spotify account first. Go to Menu → Spotify.")
        }

        // Use the Spotify Web API with the user's OAuth token
        try {
            Log.d(TAG, "Attempting Spotify API extraction for: $url")
            val tracks = SpotifyClient.getTracks(url)
            if (tracks.isNotEmpty()) {
                Log.d(TAG, "Spotify API returned ${tracks.size} tracks")
                return tracks
            }
            Log.w(TAG, "Spotify API returned 0 tracks — playlist may be empty")
        } catch (e: Exception) {
            Log.e(TAG, "Spotify API failed: ${e.message}", e)
            throw IOException("Spotify API error: ${e.message}")
        }

        return emptyList()
    }

    /**
     * Scrapes the Spotify web page to extract track information.
     * Spotify includes track metadata in the page HTML for SEO:
     * - <meta> tags with track listing info
     * - Track names appear in the page's text content
     * - The page description often lists track names
     */
    private fun scrapeSpotifyPage(url: String): List<TrackCandidate> {
        val tracks = mutableListOf<TrackCandidate>()

        val doc = Jsoup.connect(url)
            .userAgent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            .timeout(15000)
            .get()

        val thumb = doc.select("meta[property=og:image]").attr("content")
        
        // Method A: Parse ld+json structured data (MusicPlaylist/MusicAlbum schema)
        val ldJsonScripts = doc.select("script[type=application/ld+json]")
        for (script in ldJsonScripts) {
            try {
                val json = JsonParser.parseString(script.data())
                if (json.isJsonObject) {
                    val obj = json.asJsonObject
                    val type = obj.get("@type")?.asString ?: ""
                    
                    if (type == "MusicPlaylist" || type == "MusicAlbum") {
                        val trackList = obj.getAsJsonArray("track") ?: continue
                        for (trackEl in trackList) {
                            val trackObj = trackEl.asJsonObject
                            val name = trackObj.get("name")?.asString ?: continue
                            val artistObj = trackObj.getAsJsonObject("byArtist")
                            val artist = artistObj?.get("name")?.asString ?: ""
                            val durationStr = trackObj.get("duration")?.asString
                            val durationSec = parseDuration(durationStr)
                            
                            tracks.add(TrackCandidate(
                                title = name,
                                artist = artist,
                                durationSeconds = durationSec,
                                thumbnailUrl = thumb,
                                sourcePlatform = SourcePlatform.SPOTIFY
                            ))
                        }
                    }
                }
            } catch (e: Exception) {
                Log.d(TAG, "ld+json parse attempt failed: ${e.message}")
            }
        }

        if (tracks.isNotEmpty()) {
            Log.d(TAG, "ld+json extracted ${tracks.size} tracks")
            return tracks
        }

        // Method B: Parse description meta tag 
        // Spotify often puts track listing in the description like:
        // "Playlist · User · 50 songs" or lists track names
        val description = doc.select("meta[property=og:description]").attr("content")
        val title = doc.select("meta[property=og:title]").attr("content")
        Log.d(TAG, "Page title='$title', description='${description.take(200)}'")

        // Method C: Look for track rows in the HTML 
        // Spotify renders track list items with specific data attributes
        val trackElements = doc.select("[data-testid=tracklist-row], .tracklist-row, [data-encore-id=listRowContent]")
        Log.d(TAG, "Found ${trackElements.size} track DOM elements")
        
        for (el in trackElements) {
            val trackName = el.select("a[data-testid=internal-track-link], .tracklist-name, [data-encore-id=text]").firstOrNull()?.text()
            val artistName = el.select(".tracklist-row__artist-name-link, span[data-encore-id=text]").getOrNull(1)?.text()
            
            if (!trackName.isNullOrBlank()) {
                tracks.add(TrackCandidate(
                    title = trackName,
                    artist = artistName ?: "",
                    durationSeconds = null,
                    thumbnailUrl = thumb,
                    sourcePlatform = SourcePlatform.SPOTIFY
                ))
            }
        }

        if (tracks.isNotEmpty()) {
            Log.d(TAG, "DOM scraping extracted ${tracks.size} tracks")
            return tracks
        }

        // Method D: If we at least got the title, return it as a single searchable item
        if (title.isNotEmpty()) {
            val artist = description.split("·").getOrNull(0)?.trim() ?: ""
            Log.d(TAG, "Fallback: returning playlist title as single item: $title")
            return listOf(TrackCandidate(title, artist, null, thumb, SourcePlatform.SPOTIFY))
        }

        return emptyList()
    }

    /**
     * Parse ISO 8601 duration like "PT3M45S" to seconds
     */
    private fun parseDuration(iso: String?): Int? {
        if (iso == null) return null
        try {
            var total = 0
            val matcher = Regex("""PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?""").matchEntire(iso) ?: return null
            val (h, m, s) = matcher.destructured
            if (h.isNotEmpty()) total += h.toInt() * 3600
            if (m.isNotEmpty()) total += m.toInt() * 60
            if (s.isNotEmpty()) total += s.toInt()
            return if (total > 0) total else null
        } catch (e: Exception) {
            return null
        }
    }


    private fun extractAppleMusic(url: String): List<TrackCandidate> {
        return try {
            val doc = Jsoup.connect(url)
                .userAgent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
                .timeout(15000)
                .get()
            val title = doc.select("meta[property=og:title]").attr("content")
            val artist = doc.select("meta[property=og:description]").attr("content")
                ?.split("·")?.getOrNull(0)?.trim() ?: ""
            val thumb = doc.select("meta[property=og:image]").attr("content")

            if (title.isNotEmpty()) {
                listOf(TrackCandidate(title, artist, null, thumb, SourcePlatform.APPLE_MUSIC))
            } else emptyList()
        } catch (e: Exception) {
            Log.e(TAG, "Apple Music extraction failed: ${e.message}", e)
            emptyList()
        }
    }

    // ── Normalization logic for Fingerprinting ──────────────────────

    fun computeFingerprint(title: String, artist: String, duration: Int?): String {
        val normTitle = sanitize(title)
        val normArtist = sanitize(artist)
        val bucketedDuration = duration?.let { (it.toDouble() / 5.0).roundToInt() * 5 }
        
        val input = if (bucketedDuration != null) {
            "$normTitle|$normArtist|$bucketedDuration"
        } else {
            "$normTitle|$normArtist"
        }
        
        return sha256(input)
    }

    fun sanitize(text: String): String {
        return text.lowercase()
            .replace(Regex("""\(.*?\)|\[.*?\]"""), "")
            .replace(Regex("""(?i)\b(feat|ft|official|video|audio|remastered|lyrics|hq|hd|high quality)\b"""), "")
            .replace(Regex("""[^\w\s]"""), "")
            .replace(Regex("""\s+"""), " ")
            .trim()
    }

    private fun sha256(input: String): String {
        val bytes = MessageDigest.getInstance("SHA-256").digest(input.toByteArray())
        return bytes.joinToString("") { "%02x".format(it) }
    }
}
