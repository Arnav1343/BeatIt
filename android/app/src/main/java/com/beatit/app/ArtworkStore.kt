package com.beatit.app

import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * Cover art for the local library, stored as sidecar files.
 *
 * Nothing the app downloads carries embedded artwork: neither download path
 * transcodes or writes metadata, so what lands on disk is the raw stream
 * YouTube served — a WebM container with a single audio track and no
 * attached picture, whatever the file extension claims. Extracting art from
 * the audio is therefore impossible; it has to be fetched and kept beside it.
 *
 * Layout, in the same directory as the audio:
 *   <base>.jpg     the artwork
 *   <base>.noart   empty marker meaning "we looked and found nothing"
 *
 * where <base> is the audio file's name without its extension. Art is keyed
 * on the audio *filename* because that is the key the library listing,
 * /api/music/, /api/delete and the page's currentSong already share — a
 * second naming scheme would be a third place for paths to drift apart.
 *
 * The .noart marker is what stops a library of misses re-hitting the network
 * on every listing.
 */
object ArtworkStore {

    private const val TAG = "ArtworkStore"
    private const val MAX_BYTES = 512L * 1024

    private val http = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(20, TimeUnit.SECONDS)
        .followRedirects(true)
        .build()

    /** Lookups are serialised so a big library cannot fan out onto NewPipe. */
    private val lookupExecutor = Executors.newSingleThreadExecutor()

    /** Filenames with a lookup already queued, so we don't enqueue twice. */
    private val inFlight = java.util.Collections.synchronizedSet(mutableSetOf<String>())

    private fun base(audioName: String) = audioName.substringBeforeLast('.')

    fun artFile(musicDir: File, audioName: String) = File(musicDir, base(audioName) + ".jpg")

    fun noArtMarker(musicDir: File, audioName: String) = File(musicDir, base(audioName) + ".noart")

    fun hasArt(musicDir: File, audioName: String) = artFile(musicDir, audioName).let {
        it.exists() && it.length() > 0
    }

    /** True once we've either found art or established there is none. */
    fun hasTried(musicDir: File, audioName: String) =
        hasArt(musicDir, audioName) || noArtMarker(musicDir, audioName).exists()

    /**
     * Fetch [url] and store it as this track's artwork. Returns false on any
     * failure — callers treat art as strictly optional and must never let a
     * failure here affect the download itself.
     */
    fun saveFrom(url: String?, musicDir: File, audioName: String): Boolean {
        if (url.isNullOrBlank()) return false
        val dest = artFile(musicDir, audioName)
        val tmp = File(dest.absolutePath + ".tmp")
        return try {
            val response = http.newCall(Request.Builder().url(url).build()).execute()
            response.use {
                if (!it.isSuccessful) {
                    Log.w(TAG, "Art fetch failed HTTP ${it.code} for $audioName")
                    return false
                }
                val body = it.body ?: return false
                if (body.contentLength() > MAX_BYTES) {
                    Log.w(TAG, "Art too large (${body.contentLength()}) for $audioName")
                    return false
                }
                tmp.outputStream().use { out -> body.byteStream().copyTo(out) }
            }
            if (tmp.length() == 0L) {
                tmp.delete()
                return false
            }
            tmp.renameTo(dest)
            noArtMarker(musicDir, audioName).delete()
            Log.d(TAG, "Saved art for $audioName (${dest.length()} bytes)")
            true
        } catch (e: Exception) {
            Log.w(TAG, "Art fetch failed for $audioName: ${e.message}")
            tmp.delete()
            false
        }
    }

    /**
     * Queue a title-based lookup for a track that has no art yet. Returns
     * immediately; the caller must not wait on it. Writes a .noart marker on
     * failure so the same miss is not retried on every library refresh.
     */
    fun enqueueLookup(musicDir: File, audioName: String, youtubeHelper: YoutubeHelper) {
        if (hasTried(musicDir, audioName)) return
        if (!inFlight.add(audioName)) return

        lookupExecutor.submit {
            try {
                val query = base(audioName)
                Log.d(TAG, "Looking up art for: $query")
                val url = try {
                    youtubeHelper.search(query, 1)
                        .firstOrNull()
                        ?.let { it.thumbnails.firstOrNull()?.url }
                } catch (e: Exception) {
                    Log.w(TAG, "Lookup failed for $query: ${e.message}")
                    null
                }
                if (!saveFrom(url, musicDir, audioName)) {
                    // Record the miss so we don't search for this one again.
                    runCatching { noArtMarker(musicDir, audioName).createNewFile() }
                }
            } finally {
                inFlight.remove(audioName)
            }
        }
    }

    /** Remove both sidecars — call whenever the audio file is deleted. */
    fun deleteFor(musicDir: File, audioName: String) {
        artFile(musicDir, audioName).delete()
        noArtMarker(musicDir, audioName).delete()
    }
}
