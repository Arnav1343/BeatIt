package com.beatit.app

import android.util.Log
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * Spotify track extraction via the public embed pages.
 *
 * This deliberately does NOT use the Spotify Web API. The app's registration
 * is in Development Mode, so api.spotify.com returns 403 for anyone who is
 * not on the 25-user allowlist — that is, everyone who installs a public
 * build. The embed pages carry the same track data as __NEXT_DATA__ JSON,
 * need no credentials at all, and work for every user.
 *
 * Playlists and albums use the same page structure, so one parser covers
 * both. If you are tempted to reintroduce the Web API, note that it also
 * means shipping a client secret inside a public APK.
 */
object SpotifyClient {
    private const val TAG = "SpotifyClient"

    private val http = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build()

    // ── URL Parsing ────────────────────────────────────────────────

    enum class SpotifyType { PLAYLIST, ALBUM }
    data class SpotifyId(val type: SpotifyType, val id: String)

    /**
     * Extract playlist or album ID from a Spotify URL.
     * Handles:
     *   - https://open.spotify.com/playlist/ID
     *   - https://open.spotify.com/album/ID
     *   - spotify:playlist:ID
     *   - spotify:album:ID
     */
    fun extractSpotifyId(url: String): SpotifyId? {
        // Web URLs: open.spotify.com/playlist/ID or open.spotify.com/album/ID
        val webPlaylist = Regex("""playlist/([a-zA-Z0-9]+)""").find(url)
        if (webPlaylist != null) return SpotifyId(SpotifyType.PLAYLIST, webPlaylist.groupValues[1])

        val webAlbum = Regex("""album/([a-zA-Z0-9]+)""").find(url)
        if (webAlbum != null) return SpotifyId(SpotifyType.ALBUM, webAlbum.groupValues[1])

        // URI format: spotify:playlist:ID or spotify:album:ID
        val uriPlaylist = Regex("""playlist:([a-zA-Z0-9]+)""").find(url)
        if (uriPlaylist != null) return SpotifyId(SpotifyType.PLAYLIST, uriPlaylist.groupValues[1])

        val uriAlbum = Regex("""album:([a-zA-Z0-9]+)""").find(url)
        if (uriAlbum != null) return SpotifyId(SpotifyType.ALBUM, uriAlbum.groupValues[1])

        return null
    }

    /** Backward-compatible: extract playlist ID only */
    fun extractPlaylistId(url: String): String? {
        val id = extractSpotifyId(url) ?: return null
        return if (id.type == SpotifyType.PLAYLIST) id.id else null
    }

    // ── Public API ──────────────────────────────────────────────────

    /**
     * Fetch tracks from any Spotify URL (playlist or album).
     */
    fun getTracks(url: String): List<TrackCandidate> {
        val spotifyId = extractSpotifyId(url)
        if (spotifyId == null) {
            Log.e(TAG, "Could not extract Spotify ID from URL: $url")
            return emptyList()
        }

        Log.d(TAG, "Extracted ${spotifyId.type} ID: ${spotifyId.id}")
        return fetchEmbedTracks(spotifyId.type, spotifyId.id)
    }

    fun getPlaylistTracks(playlistId: String): List<TrackCandidate> =
        fetchEmbedTracks(SpotifyType.PLAYLIST, playlistId)

    fun getAlbumTracks(albumId: String): List<TrackCandidate> =
        fetchEmbedTracks(SpotifyType.ALBUM, albumId)

    // ── Embed scraping ──────────────────────────────────────────────

    /**
     * Fetch every track of a playlist or album by scraping the __NEXT_DATA__
     * JSON out of its embed page.
     */
    private fun fetchEmbedTracks(type: SpotifyType, id: String): List<TrackCandidate> {
        val segment = if (type == SpotifyType.ALBUM) "album" else "playlist"
        val embedUrl = "https://open.spotify.com/embed/$segment/$id"
        Log.d(TAG, "Fetching $segment via embed: $embedUrl")

        val request = Request.Builder()
            .url(embedUrl)
            .header("User-Agent", "Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 Chrome/120.0.0.0 Mobile Safari/537.36")
            .build()
        val response = http.newCall(request).execute()

        if (!response.isSuccessful) {
            throw IOException("Embed page failed: HTTP ${response.code}")
        }

        val html = response.body?.string() ?: throw IOException("Empty embed response")

        // Extract __NEXT_DATA__ JSON from the HTML
        val regex = Regex("""<script id="__NEXT_DATA__"[^>]*>(.*?)</script>""", RegexOption.DOT_MATCHES_ALL)
        val match = regex.find(html) ?: throw IOException("Could not find track data in embed page")

        val root = JsonParser.parseString(match.groupValues[1]).asJsonObject
        val entity = root
            .getAsJsonObject("props")
            ?.getAsJsonObject("pageProps")
            ?.getAsJsonObject("state")
            ?.getAsJsonObject("data")
            ?.getAsJsonObject("entity")
            ?: throw IOException("Could not parse embed data structure")

        val trackList = entity.getAsJsonArray("trackList")
            ?: throw IOException("No trackList in embed data")

        // Playlists and albums both expose artwork here; the API path used to
        // supply this for albums only.
        val coverUrl = largestCoverUrl(entity)

        val tracks = mutableListOf<TrackCandidate>()
        for (item in trackList) {
            val obj = item.asJsonObject
            val title = obj.get("title")?.asString ?: continue
            val artist = obj.get("subtitle")?.asString ?: "Unknown"
            val durationMs = obj.get("duration")?.asLong ?: 0
            val durationSec = if (durationMs > 0) (durationMs / 1000).toInt() else null

            tracks.add(TrackCandidate(
                title = title,
                artist = artist,
                durationSeconds = durationSec,
                thumbnailUrl = coverUrl,
                sourcePlatform = SourcePlatform.SPOTIFY
            ))
        }

        Log.d(TAG, "Fetched ${tracks.size} tracks from $segment embed $id")
        return tracks
    }

    /** Pick the highest-resolution artwork the embed page offers, if any. */
    private fun largestCoverUrl(entity: JsonObject): String? = try {
        entity.getAsJsonObject("visualIdentity")
            ?.getAsJsonArray("image")
            ?.map { it.asJsonObject }
            ?.maxByOrNull { it.get("maxHeight")?.asInt ?: 0 }
            ?.get("url")?.asString
    } catch (e: Exception) {
        Log.w(TAG, "Could not read cover art from embed data: ${e.message}")
        null
    }
}
