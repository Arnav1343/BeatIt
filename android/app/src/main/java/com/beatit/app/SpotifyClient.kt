package com.beatit.app

import android.util.Log
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import okhttp3.*
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * Spotify Web API client.
 * Uses the access token provided by SpotifyAuth (OAuth PKCE flow).
 */
object SpotifyClient {
    private const val TAG = "SpotifyClient"
    private const val API_BASE = "https://api.spotify.com/v1"
    private const val CLIENT_ID = "781875772a3a48aa9bf2f18af745e4c0"
    private const val CLIENT_SECRET = "f22d881d91f7448fa1c4e5c36a336a14"

    private val gson = Gson()
    private val http = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build()

    // Cached Client Credentials token
    private var ccToken: String? = null
    private var ccTokenExpiresAt: Long = 0

    // ── Data classes for API responses ──────────────────────────────

    data class TracksPage(
        val items: List<PlaylistItem>?,
        val next: String?,
        val total: Int?
    )

    data class PlaylistItem(
        val track: SpotifyTrack?
    )

    // Album tracks endpoint returns items directly as SimpleTrack (no wrapper)
    data class AlbumTracksPage(
        val items: List<SpotifySimpleTrack>?,
        val next: String?,
        val total: Int?
    )

    data class SpotifySimpleTrack(
        val name: String?,
        val artists: List<SpotifyArtist>?,
        @SerializedName("duration_ms") val durationMs: Int?,
        @SerializedName("track_number") val trackNumber: Int?
    )

    data class SpotifyTrack(
        val name: String?,
        val artists: List<SpotifyArtist>?,
        @SerializedName("duration_ms") val durationMs: Int?,
        val album: SpotifyAlbum?
    )

    data class SpotifyArtist(val name: String?)
    data class SpotifyAlbum(
        val name: String?,
        val images: List<SpotifyImage>?
    )
    data class SpotifyImage(val url: String?, val height: Int?)

    // Album metadata (for getting album art)
    data class AlbumMetadata(
        val name: String?,
        val images: List<SpotifyImage>?,
        val artists: List<SpotifyArtist>?
    )

    // User playlists response
    data class PlaylistsPage(
        val items: List<UserPlaylist>?,
        val next: String?,
        val total: Int?
    )

    data class UserPlaylist(
        val id: String?,
        val name: String?,
        val description: String?,
        val images: List<SpotifyImage>?,
        val tracks: PlaylistTrackRef?,
        val owner: PlaylistOwner?,
        @SerializedName("external_urls") val externalUrls: Map<String, String>?
    )

    data class PlaylistTrackRef(val total: Int?)
    data class PlaylistOwner(@SerializedName("display_name") val displayName: String?)

    // ── API Helpers ─────────────────────────────────────────────────

    // Cache the last successful OAuth token in memory as a fallback
    // (SharedPreferences reads can intermittently return null on different threads)
    private var cachedOAuthToken: String? = null

    /**
     * Get an access token. Tries OAuth user token first, falls back to cached token,
     * then to Client Credentials. Client Credentials only work for albums — NOT playlists.
     */
    private fun getToken(): String {
        // Prefer fresh OAuth user token
        val userToken = SpotifyAuth.getAccessToken()
        if (userToken != null) {
            cachedOAuthToken = userToken  // cache for future use
            Log.d(TAG, "getToken: using OAuth user token")
            return userToken
        }

        // Use cached OAuth token if available (handles intermittent SharedPreferences failures)
        val cached = cachedOAuthToken
        if (cached != null) {
            Log.d(TAG, "getToken: using cached OAuth token")
            return cached
        }

        Log.d(TAG, "getToken: no OAuth token, falling back to Client Credentials")
        // Fallback: Client Credentials (only works for albums, not playlists)
        return getClientCredentialsToken()
    }

    @Synchronized
    private fun getClientCredentialsToken(): String {
        if (ccToken != null && System.currentTimeMillis() < ccTokenExpiresAt - 60_000) {
            return ccToken!!
        }

        val credentials = okhttp3.Credentials.basic(CLIENT_ID, CLIENT_SECRET)
        val body = FormBody.Builder()
            .add("grant_type", "client_credentials")
            .build()
        val request = Request.Builder()
            .url("https://accounts.spotify.com/api/token")
            .header("Authorization", credentials)
            .post(body)
            .build()

        val response = http.newCall(request).execute()
        val responseBody = response.body?.string() ?: ""
        if (!response.isSuccessful) {
            Log.e(TAG, "Client Credentials token failed: ${response.code}")
            throw IOException("Spotify authentication failed")
        }

        val json = gson.fromJson(responseBody, SpotifyAuth.TokenResponse::class.java)
        ccToken = json.accessToken
        ccTokenExpiresAt = System.currentTimeMillis() + json.expiresIn * 1000L
        Log.d(TAG, "Got Client Credentials token, expires in ${json.expiresIn}s")
        return json.accessToken
    }

    private fun apiGet(url: String): String {
        val token = getToken()

        val request = Request.Builder()
            .url(url)
            .header("Authorization", "Bearer $token")
            .build()
        val response = http.newCall(request).execute()
        if (!response.isSuccessful) {
            val errorBody = response.body?.string() ?: ""
            Log.e(TAG, "Spotify API error: HTTP ${response.code} for $url, body: ${errorBody.take(500)}")
            if (response.code == 401) {
                // Token expired — clear cached token
                cachedOAuthToken = null
                throw IOException("Spotify session expired. Please reconnect.")
            }
            // Show full error details for debugging
            throw IOException("Spotify ${response.code}: ${errorBody.take(200)}")
        }
        return response.body?.string() ?: throw IOException("Empty API response")
    }

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

        return when (spotifyId.type) {
            SpotifyType.PLAYLIST -> getPlaylistTracks(spotifyId.id)
            SpotifyType.ALBUM -> getAlbumTracks(spotifyId.id)
        }
    }

    /**
     * Fetch all tracks from a Spotify playlist.
     * Handles pagination automatically (up to 500 tracks).
     */
    fun getPlaylistTracks(playlistId: String): List<TrackCandidate> {
        val tracks = mutableListOf<TrackCandidate>()

        var url: String? = "$API_BASE/playlists/$playlistId/tracks?limit=100"

        while (url != null && tracks.size < 500) {
            val body = apiGet(url)
            Log.d(TAG, "Playlist API response (first 300 chars): ${body.take(300)}")
            val page = gson.fromJson(body, TracksPage::class.java)
            Log.d(TAG, "Parsed page: ${page.items?.size ?: 0} items, next=${page.next != null}")

            page.items?.forEach { item ->
                val t = item.track ?: return@forEach
                val name = t.name ?: return@forEach
                val artist = t.artists?.mapNotNull { it.name }?.joinToString(", ") ?: "Unknown"
                val durationSec = t.durationMs?.let { it / 1000 }
                val thumb = t.album?.images
                    ?.sortedByDescending { it.height ?: 0 }
                    ?.firstOrNull()?.url

                tracks.add(TrackCandidate(
                    title = name,
                    artist = artist,
                    durationSeconds = durationSec,
                    thumbnailUrl = thumb,
                    sourcePlatform = SourcePlatform.SPOTIFY
                ))
            }

            url = page.next
        }

        Log.d(TAG, "Fetched ${tracks.size} tracks from playlist $playlistId")
        return tracks.take(500)
    }

    /**
     * Fetch all tracks from a Spotify album.
     * Handles pagination automatically (up to 500 tracks).
     */
    fun getAlbumTracks(albumId: String): List<TrackCandidate> {
        // First, get album metadata (for artwork)
        val albumBody = apiGet("$API_BASE/albums/$albumId")
        val album = gson.fromJson(albumBody, AlbumMetadata::class.java)
        val albumThumb = album.images
            ?.sortedByDescending { it.height ?: 0 }
            ?.firstOrNull()?.url
        val albumArtist = album.artists?.mapNotNull { it.name }?.joinToString(", ") ?: ""

        Log.d(TAG, "Album: ${album.name}, artist: $albumArtist, thumb: $albumThumb")

        val tracks = mutableListOf<TrackCandidate>()
        var url: String? = "$API_BASE/albums/$albumId/tracks?limit=50"

        while (url != null && tracks.size < 500) {
            val body = apiGet(url)
            Log.d(TAG, "Album tracks API response (first 300 chars): ${body.take(300)}")
            val page = gson.fromJson(body, AlbumTracksPage::class.java)
            Log.d(TAG, "Parsed album page: ${page.items?.size ?: 0} items, next=${page.next != null}")

            page.items?.forEach { t ->
                val name = t.name ?: return@forEach
                val artist = t.artists?.mapNotNull { it.name }?.joinToString(", ") ?: albumArtist
                val durationSec = t.durationMs?.let { it / 1000 }

                tracks.add(TrackCandidate(
                    title = name,
                    artist = artist,
                    durationSeconds = durationSec,
                    thumbnailUrl = albumThumb,
                    sourcePlatform = SourcePlatform.SPOTIFY
                ))
            }

            url = page.next
        }
        Log.d(TAG, "Fetched ${tracks.size} tracks from album $albumId")
        return tracks.take(500)
    }

    // ── User Playlists ──────────────────────────────────────────────

    /**
     * Fetch the current user's playlists.
     * Requires user to be connected via SpotifyAuth.
     */
    data class PlaylistInfo(
        val id: String?,
        val name: String,
        val trackCount: Int,
        val thumbnail: String?,
        val owner: String,
        val url: String
    )

    fun getUserPlaylists(): List<PlaylistInfo> {
        val playlists = mutableListOf<PlaylistInfo>()
        var url: String? = "$API_BASE/me/playlists?limit=50"

        while (url != null && playlists.size < 200) {
            val body = apiGet(url)
            Log.d(TAG, "Playlists response (first 500 chars): ${body.take(500)}")

            // Parse manually to avoid Gson mapping issues with nested objects
            val root = com.google.gson.JsonParser.parseString(body).asJsonObject
            val items = root.getAsJsonArray("items") ?: continue

            for (item in items) {
                val obj = item.asJsonObject

                val id = obj.get("id")?.asString
                val name = obj.get("name")?.asString ?: "Untitled"

                // tracks field — {"total": N}
                val tracksEl = obj.get("tracks")
                var trackCount = 0
                if (tracksEl != null && tracksEl.isJsonObject) {
                    trackCount = tracksEl.asJsonObject.get("total")?.asInt ?: 0
                }
                Log.d(TAG, "Playlist: $name, tracksEl=${tracksEl}, parsed=$trackCount")

                // images is [{url, height, width}, ...]
                val imagesArr = obj.getAsJsonArray("images")
                val thumb = imagesArr?.firstOrNull()?.asJsonObject?.get("url")?.asString

                // owner is {display_name: "..."}
                val ownerObj = obj.getAsJsonObject("owner")
                val ownerName = ownerObj?.get("display_name")?.asString ?: ""

                // external_urls is {spotify: "..."}
                val extUrls = obj.getAsJsonObject("external_urls")
                val spotifyUrl = extUrls?.get("spotify")?.asString
                    ?: "https://open.spotify.com/playlist/$id"

                Log.d(TAG, "Playlist: $name, trackCount=$trackCount, owner=$ownerName")

                playlists.add(PlaylistInfo(
                    id = id,
                    name = name,
                    trackCount = trackCount,
                    thumbnail = thumb,
                    owner = ownerName,
                    url = spotifyUrl
                ))
            }

            url = root.get("next")?.let { if (it.isJsonNull) null else it.asString }
        }

        Log.d(TAG, "Fetched ${playlists.size} user playlists")
        return playlists
    }
}
