package com.beatit.app

import android.content.Context
import com.google.gson.Gson
import fi.iki.elonen.NanoHTTPD
import java.io.File
import java.io.FileInputStream
import kotlinx.coroutines.*

class BeatItServer(private val context: Context, port: Int) : NanoHTTPD(port) {

    private val gson = Gson()
    private val musicDir: File get() = File(context.getExternalFilesDir(null), "Music").also { it.mkdirs() }
    private val downloadManager = DownloadManager(context, musicDir)
    private val youtubeHelper = YoutubeHelper()

    override fun serve(session: IHTTPSession): Response {
        val uri = session.uri
        val method = session.method

        return try {
            when {
                uri == "/" || uri == "/index.html" -> serveAsset("index.html", "text/html")
                uri.startsWith("/static/") -> serveStaticAsset(uri)
                method == Method.POST && uri == "/api/search" -> handleSearch(session)
                method == Method.POST && uri == "/api/suggestions" -> handleSuggestions(session)
                method == Method.POST && uri == "/api/download" -> handleDownload(session)
                method == Method.POST && uri == "/api/prefetch" -> handlePrefetch(session)
                uri.startsWith("/api/progress/") -> handleProgress(uri)
                method == Method.POST && uri == "/api/import" -> handleImport(session)
                uri == "/api/import/list" -> handleImportList()
                uri.startsWith("/api/import/status/") -> handleImportStatus(uri)
                method == Method.POST && uri == "/api/import/action" -> handleImportAction(session)
                uri == "/api/library" -> handleLibrary()
                uri.startsWith("/api/music/") -> handleMusic(uri)
                method == Method.POST && uri == "/api/delete" -> handleDelete(session)
                else -> newFixedLengthResponse(Response.Status.NOT_FOUND, MIME_PLAINTEXT, "Not found")
            }
        } catch (e: Exception) {
            jsonError(e.message ?: "Internal error")
        }
    }

    // ── Static file serving ─────────────────────────────────────────

    private fun serveAsset(path: String, mime: String): Response {
        val stream = context.assets.open(path)
        return newChunkedResponse(Response.Status.OK, mime, stream)
    }

    private fun serveStaticAsset(uri: String): Response {
        val path = uri.removePrefix("/").replace("..", "") // prevent path traversal
        val mime = when {
            path.endsWith(".js") -> "application/javascript"
            path.endsWith(".css") -> "text/css"
            path.endsWith(".html") -> "text/html"
            else -> "application/octet-stream"
        }
        return try {
            serveAsset(path, mime)
        } catch (e: Exception) {
            newFixedLengthResponse(Response.Status.NOT_FOUND, MIME_PLAINTEXT, "Not found")
        }
    }

    // ── API: Search ─────────────────────────────────────────────────

    private fun handleSearch(session: IHTTPSession): Response {
        val body = readBody(session)
        val query = gson.fromJson(body, Map::class.java)["query"] as? String ?: return jsonError("No query")
        val results = youtubeHelper.search(query)
        if (results.isEmpty()) return jsonError("No results found")
        val best = results.first()
        return jsonOk(mapOf(
            "title" to best.name,
            "url" to best.url,
            "duration" to best.duration,
            "uploader" to (best.uploaderName ?: ""),
            "thumbnail" to (best.thumbnails.firstOrNull()?.url ?: "")
        ))
    }

    private fun handleSuggestions(session: IHTTPSession): Response {
        val body = readBody(session)
        val query = gson.fromJson(body, Map::class.java)["query"] as? String ?: return jsonError("No query")
        val results = youtubeHelper.suggestions(query)
        return jsonOk(results)
    }

    // ── API: Download ───────────────────────────────────────────────

    private fun handleDownload(session: IHTTPSession): Response {
        val body = readBody(session)
        val map = gson.fromJson(body, Map::class.java)
        val url = map["url"] as? String ?: return jsonError("No URL")
        val title = map["title"] as? String ?: "Unknown"
        val quality = (map["quality"] as? Double)?.toInt() ?: 192
        val codec = map["codec"] as? String ?: "mp3"

        val taskId = downloadManager.startDownload(url, title, quality, codec)
        return jsonOk(mapOf("task_id" to taskId))
    }

    private fun handleProgress(uri: String): Response {
        val taskId = uri.removePrefix("/api/progress/")
        val status = downloadManager.getProgress(taskId) ?: return jsonError("Unknown task")
        return jsonOk(status)
    }

    private fun handlePrefetch(session: IHTTPSession): Response {
        val body = readBody(session)
        val url = gson.fromJson(body, Map::class.java)["url"] as? String ?: return jsonError("No URL")
        youtubeHelper.prefetchStreamInfo(url)
        return jsonOk(mapOf("prefetching" to true, "cached" to youtubeHelper.isPrefetched(url)))
    }

    // ── API: Library ────────────────────────────────────────────────

    private fun handleLibrary(): Response {
        val files = musicDir.listFiles()
            ?.filter { it.extension in listOf("mp3", "opus", "ogg") }
            ?.sortedByDescending { it.lastModified() }
            ?.map { f ->
                mapOf(
                    "filename" to f.name,
                    "title" to f.nameWithoutExtension,
                    "size_human" to humanSize(f.length()),
                    "codec" to if (f.extension == "mp3") "mp3" else "opus"
                )
            } ?: emptyList()
        return jsonOk(files)
    }

    // ── API: Stream music ───────────────────────────────────────────

    private fun handleMusic(uri: String): Response {
        val filename = java.net.URLDecoder.decode(uri.removePrefix("/api/music/"), "UTF-8")
        val file = File(musicDir, filename)
        if (!file.exists()) return newFixedLengthResponse(Response.Status.NOT_FOUND, MIME_PLAINTEXT, "Not found")
        val mime = if (filename.endsWith(".mp3")) "audio/mpeg" else "audio/ogg"
        return newChunkedResponse(Response.Status.OK, mime, FileInputStream(file))
    }

    // ── API: Delete ─────────────────────────────────────────────────

    private fun handleDelete(session: IHTTPSession): Response {
        val body = readBody(session)
        val map = gson.fromJson(body, Map::class.java)
        val filename = map["filename"] as? String ?: return jsonError("No filename")
        val file = File(musicDir, filename)
        val deleted = file.delete()
        return jsonOk(mapOf("success" to deleted))
    }

    // ── Helpers ─────────────────────────────────────────────────────

    private fun readBody(session: IHTTPSession): String {
        val map = mutableMapOf<String, String>()
        session.parseBody(map)
        return map["postData"] ?: ""
    }

    private fun jsonOk(data: Any): Response =
        newFixedLengthResponse(Response.Status.OK, "application/json", gson.toJson(data))

    private fun jsonError(msg: String): Response =
        newFixedLengthResponse(Response.Status.OK, "application/json", gson.toJson(mapOf("error" to msg)))

    // ── API: Batch Import ──────────────────────────────────────────

    private fun handleImport(session: IHTTPSession): Response {
        val body = readBody(session)
        val url = gson.fromJson(body, Map::class.java)["url"] as? String ?: return jsonError("No URL")
        val platform = PlatformDetector.detectPlatform(url) ?: return jsonError("Unsupported platform or invalid URL")
        
        // Run extraction synchronously so we can return errors to the UI
        val result = kotlinx.coroutines.runBlocking(kotlinx.coroutines.Dispatchers.IO) {
            BatchManager.submitBatch(url, platform)
        }
        
        return if (result.success) {
            jsonOk(mapOf("success" to true, "trackCount" to result.trackCount))
        } else {
            jsonError(result.error ?: "Unknown import error")
        }
    }

    private fun handleImportList(): Response {
        val dao = AppDatabase.getDatabase(context).batchDao()
        val batches = runBlocking { dao.getAllBatches() }
        return jsonOk(batches)
    }

    private fun handleImportStatus(uri: String): Response {
        val batchId = uri.removePrefix("/api/import/status/")
        val dao = AppDatabase.getDatabase(context).batchDao()
        val data = runBlocking { dao.getBatchWithTracks(batchId) } ?: return jsonError("Batch not found")
        
        return jsonOk(mapOf(
            "batch" to data.batch,
            "tracks" to data.tracks
        ))
    }

    private fun handleImportAction(session: IHTTPSession): Response {
        val body = readBody(session)
        val map = gson.fromJson(body, Map::class.java)
        val trackId = map["track_id"] as? String ?: return jsonError("No track_id")
        val action = map["action"] as? String ?: return jsonError("No action")
        val videoId = map["video_id"] as? String
        
        val dao = AppDatabase.getDatabase(context).batchDao()
        val track = runBlocking { dao.getTrack(trackId) } ?: return jsonError("Track not found")

        val nextStatus = when (action) {
            "accept" -> {
                if (videoId != null) track.youtubeVideoId = videoId
                TrackStatus.MATCHED
            }
            "rematch" -> TrackStatus.MATCHING
            "manual" -> TrackStatus.MATCHING_MANUAL
            else -> return jsonError("Invalid action")
        }
        
        runBlocking { 
            if (videoId != null) {
                dao.updateTrack(track)
            }
            BatchManager.transition(trackId, nextStatus)
            if (nextStatus == TrackStatus.MATCHED) {
                BatchManager.transition(trackId, TrackStatus.QUEUED)
            }
        }
        return jsonOk(mapOf("success" to true))
    }


    private fun humanSize(bytes: Long): String = when {
        bytes < 1024 -> "${bytes} B"
        bytes < 1024 * 1024 -> "${bytes / 1024} KB"
        else -> "${"%.1f".format(bytes / (1024.0 * 1024.0))} MB"
    }
}
