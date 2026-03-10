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

    /**
     * Get an access token. Tries OAuth user token first, falls back to Client Credentials.
     * Client Credentials only work for albums, search, etc. — NOT playlists.
     */
    private fun getToken(): String {
        // Prefer OAuth user token (works for everything including playlists)
        val userToken = SpotifyAuth.getAccessToken()
        if (userToken != null) return userToken

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
            Log.e(TAG, "Spotify API error: HTTP ${response.code} for $url, body: ${errorBody.take(300)}")
            if (response.code == 401) {
                throw IOException("Spotify session expired. Please reconnect.")
            }
            if (response.code == 403) {
                throw IOException("This playlist requires Spotify login. Go to Menu \u2192 Spotify to connect your account.")
            }
            throw IOException("Spotify API error: ${response.code}")
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
    fun getUserPlaylists(): List<Map<String, Any?>> {
        val playlists = mutableListOf<Map<String, Any?>>()
        var url: String? = "$API_BASE/me/playlists?limit=50"

        while (url != null && playlists.size < 200) {
            val body = apiGet(url)
            val page = gson.fromJson(body, PlaylistsPage::class.java)

            page.items?.forEach { p ->
                val thumb = p.images?.firstOrNull()?.url
                val spotifyUrl = p.externalUrls?.get("spotify")
                    ?: "https://open.spotify.com/playlist/${p.id}"

                playlists.add(mapOf(
                    "id" to p.id,
                    "name" to (p.name ?: "Untitled"),
                    "trackCount" to (p.tracks?.total ?: 0),
                    "thumbnail" to thumb,
                    "owner" to (p.owner?.displayName ?: ""),
                    "url" to spotifyUrl
                ))
            }

            url = page.next
        }

        Log.d(TAG, "Fetched ${playlists.size} user playlists")
        return playlists
    }
}
