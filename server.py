"""
iPod Music Server
==================
Flask backend that wraps SongScraper to provide REST API endpoints
for searching, downloading, streaming, and managing music.
"""

import os
import sys
import json
import time
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, send_file

from scraper import SongScraper, _human_size

app = Flask(__name__, static_folder="static", static_url_path="/static")

DOWNLOADS_DIR = Path(__file__).parent / "downloads"
DOWNLOADS_DIR.mkdir(exist_ok=True)

# Track download progress per task
download_tasks = {}
progress_lock = threading.Lock()


# ─── Pages ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ─── API ──────────────────────────────────────────────────────────────

_scraper = SongScraper(quality=320)  # Reuse across requests

@app.route("/api/suggestions", methods=["POST"])
def api_suggestions():
    """Return multiple search results. Body: { "query": "song name" }"""
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    if not query or len(query) < 2:
        return jsonify([])

    try:
        results = _scraper.search_multi(query, limit=5)
        return jsonify(results)
    except Exception as e:
        return jsonify([])


@app.route("/api/search", methods=["POST"])
def api_search():
    """Search for a song by name. Body: { "query": "song name" }"""
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        scraper = SongScraper(quality=320)
        result = scraper.search(query)
        return jsonify(result)
    except LookupError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Search failed: {e}"}), 500

@app.route("/api/download", methods=["POST"])
def api_download():
    """
    Start downloading a song. Body: { "url": "...", "title": "..." }
    Returns a task_id for progress polling, then the result when done.
    """
    data = request.get_json(force=True)
    url = data.get("url", "").strip()
    title = data.get("title", "Unknown")
    quality = int(data.get("quality", 192))
    if quality not in (128, 192, 320):
        quality = 192

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    task_id = str(int(time.time() * 1000))

    with progress_lock:
        download_tasks[task_id] = {
            "status": "downloading", "percent": 0,
            "title": title, "result": None, "error": None,
        }

    def _progress_hook(d):
        with progress_lock:
            if d["status"] == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                downloaded = d.get("downloaded_bytes", 0)
                pct = int((downloaded / total * 100)) if total > 0 else 0
                download_tasks[task_id]["status"] = "downloading"
                download_tasks[task_id]["percent"] = pct
            elif d["status"] == "finished":
                download_tasks[task_id]["status"] = "converting"
                download_tasks[task_id]["percent"] = 100

    def _do_download():
        try:
            scraper = SongScraper(quality=quality)
            mp3_path = scraper.download(
                url=url,
                output_dir=DOWNLOADS_DIR,
                progress_hook=_progress_hook,
            )
            file_size = mp3_path.stat().st_size
            with progress_lock:
                download_tasks[task_id]["status"] = "done"
                download_tasks[task_id]["percent"] = 100
                download_tasks[task_id]["result"] = {
                    "success": True,
                    "filename": mp3_path.name,
                    "title": title,
                    "size": file_size,
                    "size_human": _human_size(file_size),
                }
        except Exception as e:
            with progress_lock:
                download_tasks[task_id]["status"] = "error"
                download_tasks[task_id]["error"] = str(e)

    # Run download in background thread
    thread = threading.Thread(target=_do_download, daemon=True)
    thread.start()

    return jsonify({"task_id": task_id, "status": "started"})


@app.route("/api/progress/<task_id>", methods=["GET"])
def api_progress(task_id):
    """Poll download progress. Returns status, percent, and result when done."""
    with progress_lock:
        task = download_tasks.get(task_id)

    if not task:
        return jsonify({"error": "Unknown task"}), 404

    response = {
        "status": task["status"],
        "percent": task["percent"],
    }

    if task["status"] == "done" and task["result"]:
        response["result"] = task["result"]
    elif task["status"] == "error":
        response["error"] = task.get("error", "Download failed")

    return jsonify(response)


@app.route("/api/search-download", methods=["POST"])
def api_search_download():
    """
    Search + download in one call. Body: { "query": "song name" }
    Returns task_id for progress polling.
    """
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400

    task_id = str(int(time.time() * 1000))

    with progress_lock:
        download_tasks[task_id] = {
            "status": "searching", "percent": 0,
            "title": query, "result": None, "error": None,
            "search_result": None,
        }

    def _progress_hook(d):
        with progress_lock:
            if d["status"] == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                downloaded = d.get("downloaded_bytes", 0)
                pct = int((downloaded / total * 100)) if total > 0 else 0
                download_tasks[task_id]["status"] = "downloading"
                download_tasks[task_id]["percent"] = pct
            elif d["status"] == "finished":
                download_tasks[task_id]["status"] = "converting"
                download_tasks[task_id]["percent"] = 100

    def _do_search_download():
        try:
            scraper = SongScraper(quality=320)
            metadata = scraper.search(query)

            with progress_lock:
                download_tasks[task_id]["status"] = "downloading"
                download_tasks[task_id]["search_result"] = metadata
                download_tasks[task_id]["title"] = metadata.get("title", query)

            mp3_path = scraper.download(
                url=metadata["url"],
                output_dir=DOWNLOADS_DIR,
                progress_hook=_progress_hook,
            )
            file_size = mp3_path.stat().st_size
            with progress_lock:
                download_tasks[task_id]["status"] = "done"
                download_tasks[task_id]["percent"] = 100
                download_tasks[task_id]["result"] = {
                    "success": True,
                    "filename": mp3_path.name,
                    "title": metadata.get("title", query),
                    "size": file_size,
                    "size_human": _human_size(file_size),
                }
        except LookupError as e:
            with progress_lock:
                download_tasks[task_id]["status"] = "error"
                download_tasks[task_id]["error"] = f"Not found: {e}"
        except Exception as e:
            with progress_lock:
                download_tasks[task_id]["status"] = "error"
                download_tasks[task_id]["error"] = str(e)

    thread = threading.Thread(target=_do_search_download, daemon=True)
    thread.start()

    return jsonify({"task_id": task_id, "status": "started"})


@app.route("/api/library", methods=["GET"])
def api_library():
    """List all downloaded MP3 files with metadata."""
    songs = []
    for mp3 in sorted(DOWNLOADS_DIR.glob("*.mp3"), key=lambda p: p.stat().st_mtime, reverse=True):
        stat = mp3.stat()

        # Try to get duration via mutagen, fall back to estimate
        duration = 0
        try:
            from mutagen.mp3 import MP3
            audio_info = MP3(str(mp3))
            duration = int(audio_info.info.length)
        except Exception:
            # Rough estimate: filesize / (bitrate/8)
            duration = int(stat.st_size / (320 * 1000 / 8))

        songs.append({
            "filename": mp3.name,
            "title": mp3.stem,
            "size": stat.st_size,
            "size_human": _human_size(stat.st_size),
            "duration": duration,
            "modified": stat.st_mtime,
        })
    return jsonify(songs)


@app.route("/api/music/<path:filename>")
def api_music(filename):
    """Stream an MP3 file for playback."""
    filepath = DOWNLOADS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "File not found"}), 404
    return send_file(filepath, mimetype="audio/mpeg")


@app.route("/api/delete", methods=["POST"])
def api_delete():
    """Delete a song. Body: { "filename": "..." }"""
    data = request.get_json(force=True)
    filename = data.get("filename", "")
    filepath = DOWNLOADS_DIR / filename
    if filepath.exists() and filepath.suffix == ".mp3":
        filepath.unlink()
        return jsonify({"success": True})
    return jsonify({"error": "File not found"}), 404


# ─── Run ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if sys.platform == "win32":
        os.system("")
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n  🎵 iPod Music Server running at http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
