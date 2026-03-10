package com.beatit.app

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import okhttp3.FormBody
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.IOException
import java.security.MessageDigest
import java.security.SecureRandom
import java.util.concurrent.TimeUnit

/**
 * Spotify OAuth 2.0 with PKCE (Proof Key for Code Exchange).
 * No client_secret required — safe for public/mobile apps.
 *
 * Flow:
 *   1. App generates code_verifier + code_challenge
 *   2. User opens Spotify authorize URL in WebView
 *   3. Spotify redirects to localhost callback with auth code
 *   4. App exchanges code + verifier for access/refresh tokens
 *   5. Tokens stored in SharedPreferences
 */
object SpotifyAuth {
    private const val TAG = "SpotifyAuth"
    private const val CLIENT_ID = "781875772a3a48aa9bf2f18af745e4c0"
    private const val REDIRECT_URI = "beatit://callback"
    private const val AUTH_URL = "https://accounts.spotify.com/authorize"
    private const val TOKEN_URL = "https://accounts.spotify.com/api/token"

    // Scopes: read user's public and private playlists
    private const val SCOPES = "playlist-read-private playlist-read-collaborative"

    private const val PREFS_NAME = "spotify_auth"
    private const val KEY_ACCESS_TOKEN = "access_token"
    private const val KEY_REFRESH_TOKEN = "refresh_token"
    private const val KEY_EXPIRES_AT = "expires_at"
    private const val KEY_USER_NAME = "user_name"

    private val gson = Gson()
    private val http = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(15, TimeUnit.SECONDS)
        .build()

    private lateinit var prefs: SharedPreferences

    // PKCE state — lives only during the login flow
    private var codeVerifier: String? = null

    // ── Data classes ──────────────────────────────────────────────

    data class TokenResponse(
        @SerializedName("access_token") val accessToken: String,
        @SerializedName("refresh_token") val refreshToken: String?,
        @SerializedName("expires_in") val expiresIn: Int
    )

    data class SpotifyUser(
        @SerializedName("display_name") val displayName: String?,
        val id: String?
    )

    // ── Init ──────────────────────────────────────────────────────

    fun init(context: Context) {
        prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    }

    // ── Login Flow ────────────────────────────────────────────────

    /**
     * Generate the Spotify authorization URL with PKCE challenge.
     * Call this when user taps "Connect Spotify".
     */
    fun getLoginUrl(): String {
        // Generate PKCE code verifier (43-128 chars, URL-safe random)
        val verifier = generateCodeVerifier()
        codeVerifier = verifier
        val challenge = generateCodeChallenge(verifier)

        return "$AUTH_URL?" +
                "client_id=$CLIENT_ID" +
                "&response_type=code" +
                "&redirect_uri=${java.net.URLEncoder.encode(REDIRECT_URI, "UTF-8")}" +
                "&scope=${java.net.URLEncoder.encode(SCOPES, "UTF-8")}" +
                "&code_challenge_method=S256" +
                "&code_challenge=$challenge"
    }

    /**
     * Exchange the authorization code for access + refresh tokens.
     * Called when the callback endpoint receives the code from Spotify.
     */
    fun handleCallback(code: String): Boolean {
        val verifier = codeVerifier
        if (verifier == null) {
            Log.e(TAG, "No code verifier found — login flow was not started properly")
            return false
        }

        try {
            val body = FormBody.Builder()
                .add("grant_type", "authorization_code")
                .add("code", code)
                .add("redirect_uri", REDIRECT_URI)
                .add("client_id", CLIENT_ID)
                .add("code_verifier", verifier)
                .build()

            val request = Request.Builder()
                .url(TOKEN_URL)
                .post(body)
                .build()

            val response = http.newCall(request).execute()
            val responseBody = response.body?.string() ?: ""

            if (!response.isSuccessful) {
                Log.e(TAG, "Token exchange failed: HTTP ${response.code}, body: ${responseBody.take(300)}")
                return false
            }

            val token = gson.fromJson(responseBody, TokenResponse::class.java)
            saveTokens(token)
            codeVerifier = null // Clean up

            // Fetch user profile
            fetchAndSaveUserName(token.accessToken)

            Log.d(TAG, "Spotify connected successfully!")
            return true
        } catch (e: Exception) {
            Log.e(TAG, "Token exchange error: ${e.message}", e)
            return false
        }
    }

    // ── Token Management ──────────────────────────────────────────

    /**
     * Get a valid access token, refreshing if expired.
     * Returns null if not connected.
     */
    @Synchronized
    fun getAccessToken(): String? {
        val token = prefs.getString(KEY_ACCESS_TOKEN, null) ?: return null
        val expiresAt = prefs.getLong(KEY_EXPIRES_AT, 0)

        // If token is still valid (with 60s buffer), return it
        if (System.currentTimeMillis() < expiresAt - 60_000) {
            return token
        }

        // Try to refresh
        val refreshToken = prefs.getString(KEY_REFRESH_TOKEN, null)
        if (refreshToken != null) {
            return refreshAccessToken(refreshToken)
        }

        return null
    }

    private fun refreshAccessToken(refreshToken: String): String? {
        try {
            val body = FormBody.Builder()
                .add("grant_type", "refresh_token")
                .add("refresh_token", refreshToken)
                .add("client_id", CLIENT_ID)
                .build()

            val request = Request.Builder()
                .url(TOKEN_URL)
                .post(body)
                .build()

            val response = http.newCall(request).execute()
            val responseBody = response.body?.string() ?: ""

            if (!response.isSuccessful) {
                Log.e(TAG, "Token refresh failed: HTTP ${response.code}")
                // If refresh token is invalid, clear everything
                if (response.code == 400 || response.code == 401) {
                    logout()
                }
                return null
            }

            val token = gson.fromJson(responseBody, TokenResponse::class.java)
            saveTokens(token)
            Log.d(TAG, "Token refreshed, expires in ${token.expiresIn}s")
            return token.accessToken
        } catch (e: Exception) {
            Log.e(TAG, "Token refresh error: ${e.message}", e)
            return null
        }
    }

    private fun saveTokens(token: TokenResponse) {
        prefs.edit().apply {
            putString(KEY_ACCESS_TOKEN, token.accessToken)
            // Spotify may not return a new refresh token on refresh — keep the old one
            if (token.refreshToken != null) {
                putString(KEY_REFRESH_TOKEN, token.refreshToken)
            }
            putLong(KEY_EXPIRES_AT, System.currentTimeMillis() + token.expiresIn * 1000L)
            apply()
        }
    }

    private fun fetchAndSaveUserName(accessToken: String) {
        try {
            val request = Request.Builder()
                .url("https://api.spotify.com/v1/me")
                .header("Authorization", "Bearer $accessToken")
                .build()
            val response = http.newCall(request).execute()
            if (response.isSuccessful) {
                val body = response.body?.string() ?: return
                val user = gson.fromJson(body, SpotifyUser::class.java)
                prefs.edit().putString(KEY_USER_NAME, user.displayName ?: user.id ?: "Spotify User").apply()
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to fetch user profile: ${e.message}")
        }
    }

    // ── Status ────────────────────────────────────────────────────

    fun isConnected(): Boolean = getAccessToken() != null

    fun getUserName(): String = prefs.getString(KEY_USER_NAME, "") ?: ""

    fun logout() {
        prefs.edit().clear().apply()
        codeVerifier = null
        Log.d(TAG, "Spotify disconnected")
    }

    // ── PKCE Helpers ──────────────────────────────────────────────

    private fun generateCodeVerifier(): String {
        val bytes = ByteArray(64)
        SecureRandom().nextBytes(bytes)
        return android.util.Base64.encodeToString(bytes, android.util.Base64.URL_SAFE or android.util.Base64.NO_PADDING or android.util.Base64.NO_WRAP)
            .replace("=", "")
            .take(128)
    }

    private fun generateCodeChallenge(verifier: String): String {
        val digest = MessageDigest.getInstance("SHA-256").digest(verifier.toByteArray(Charsets.US_ASCII))
        return android.util.Base64.encodeToString(digest, android.util.Base64.URL_SAFE or android.util.Base64.NO_PADDING or android.util.Base64.NO_WRAP)
            .replace("=", "")
    }
}
