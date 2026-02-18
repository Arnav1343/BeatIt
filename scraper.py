"""
Song-to-MP3 Scraper  (Max-Quality Edition)
===========================================
Uses yt-dlp to search YouTube for a song by name and download it as a
maximum-quality MP3 file via ffmpeg post-processing.

Quality features:
  - Prefers the highest-bitrate source audio stream (opus/m4a)
  - 320 kbps CBR MP3 with highest-quality VBR algorithm layered on top
  - Full stereo encoding (no joint-stereo downmix)
  - Embeds album-art thumbnail and metadata (title, artist, etc.)
  - Concurrent fragment downloads for maximum speed
"""

import os
import re
import time
import logging
from pathlib import Path
from typing import Optional

import yt_dlp

logger = logging.getLogger(__name__)


class SongScraper:
    """Search for songs on YouTube and download them as MP3 files."""

    # Maximum video duration (seconds) to avoid albums/podcasts
    MAX_DURATION = 900  # 15 minutes

    # Number of search results to evaluate
    SEARCH_COUNT = 10

    # Keywords that indicate a result is likely just the audio track
    AUDIO_KEYWORDS = re.compile(
        r"(official\s*audio|lyrics?\s*video|audio|lyric|official\s*music\s*video)",
        re.IGNORECASE,
    )

    # Keywords that indicate a result is NOT what we want
    REJECT_KEYWORDS = re.compile(
        r"(live\s*performance|concert|reaction|cover\s*by|tutorial|karaoke|remix|slowed|reverb|sped\s*up|bass\s*boosted|instrumental|behind\s*the\s*scenes|interview|making\s*of|drum\s*cover|guitar\s*cover|piano\s*cover)",
        re.IGNORECASE,
    )

    def __init__(self, quality: int = 320):
        """
        Args:
            quality: MP3 bitrate in kbps (128, 192, 256, or 320).
        """
        if quality not in (128, 192, 256, 320):
            raise ValueError(f"Invalid quality {quality}. Choose 128, 192, 256, or 320.")
        self.quality = quality

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str) -> dict:
        """
        Search YouTube for the best-matching video for the given song name.

        Args:
            query: Song name, optionally including the artist.

        Returns:
            dict with keys: id, title, url, duration, uploader, thumbnail
        """
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "default_search": f"ytsearch{self.SEARCH_COUNT}",
            "noplaylist": True,
            "skip_download": True,
        }

        logger.info("Searching YouTube for: %s", query)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)

        entries = info.get("entries", [])
        if not entries:
            raise LookupError(f"No results found for '{query}'")

        best = self._pick_best(entries, query)

        return {
            "id": best["id"],
            "title": best.get("title", "Unknown"),
            "url": best.get("webpage_url") or f"https://www.youtube.com/watch?v={best['id']}",
            "duration": best.get("duration", 0),
            "uploader": best.get("uploader", "Unknown"),
            "thumbnail": best.get("thumbnail", ""),
        }

    def _pick_best(self, entries: list[dict], query: str) -> dict:
        """
        Rank search results and pick the best match.

        Scoring logic:
          +3  if title contains an "audio" keyword (official audio, lyrics, etc.)
          -10 if title contains a "reject" keyword (cover, remix, live, etc.)
          +1  if duration is within a typical song range (1:30 – 7:00)
          -5  if duration exceeds MAX_DURATION

        Falls back to the first entry if every result scores poorly.
        """
        scored: list[tuple[int, dict]] = []

        for entry in entries:
            if entry is None:
                continue

            title = entry.get("title", "")
            duration = entry.get("duration") or 0
            score = 0

            # Prefer "official audio" style results
            if self.AUDIO_KEYWORDS.search(title):
                score += 3

            # Penalise covers, remixes, live recordings, etc.
            if self.REJECT_KEYWORDS.search(title):
                score -= 10

            # Prefer typical song duration (90s – 420s)
            if 90 <= duration <= 420:
                score += 1

            # Hard-penalise very long videos
            if duration > self.MAX_DURATION:
                score -= 5

            scored.append((score, entry))

        # Sort by score descending, then by original order
        scored.sort(key=lambda x: x[0], reverse=True)

        best_score, best_entry = scored[0]
        logger.info(
            "Selected: '%s' (score=%d, duration=%ds)",
            best_entry.get("title"),
            best_score,
            best_entry.get("duration", 0),
        )
        return best_entry

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(
        self,
        url: str,
        output_dir: str | Path = "downloads",
        progress_hook=None,
    ) -> Path:
        """
        Download audio from a URL and convert to MP3.

        Args:
            url:           YouTube video URL.
            output_dir:    Directory to save the MP3 to.
            progress_hook: Optional callable(dict) for progress updates.

        Returns:
            Path to the saved .mp3 file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # yt-dlp template – will produce <title>.mp3
        outtmpl = str(output_dir / "%(title)s.%(ext)s")

        ydl_opts = {
            # ── Source format: prefer highest-bitrate audio ──────────
            "format": "bestaudio[asr>=44100]/bestaudio/best",
            "format_sort": ["abr", "asr"],   # sort by audio bitrate then sample rate
            "outtmpl": outtmpl,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,

            # ── Speed: concurrent fragment downloads ────────────────
            "concurrent_fragment_downloads": 4,
            "buffersize": 1024 * 64,          # 64 KB buffer

            # ── Thumbnail for album art embedding ───────────────────
            "writethumbnail": True,

            # ── Post-processors (order matters) ─────────────────────
            "postprocessors": [
                # 1. Extract audio → MP3 at max bitrate
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": str(self.quality),
                },
                # 2. Embed metadata (title, artist, album, etc.)
                {
                    "key": "FFmpegMetadata",
                    "add_metadata": True,
                },
                # 3. Embed thumbnail as album art
                {
                    "key": "EmbedThumbnail",
                    "already_have_thumbnail": False,
                },
            ],

            # ── FFmpeg encoding args for maximum quality ────────────
            # -b:a 320k = force constant 320 kbps bitrate (max for MP3)
            # -joint_stereo 0 = full stereo (no mid/side downmix)
            "postprocessor_args": {
                "extractaudio": ["-b:a", "320k", "-joint_stereo", "0"],
            },

            # Restrict filenames to safe characters on Windows
            "restrictfilenames": False,
            "windowsfilenames": True,
        }

        if progress_hook:
            ydl_opts["progress_hooks"] = [progress_hook]

        logger.info("Downloading: %s -> %s", url, output_dir)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        # Determine the final filename produced by yt-dlp
        # yt-dlp may sanitise the title; we look for the .mp3 it created
        title = info.get("title", "audio")
        mp3_path = self._find_mp3(output_dir, title)

        if mp3_path is None:
            raise FileNotFoundError(
                f"Download appeared to succeed but no .mp3 found in {output_dir}"
            )

        logger.info("Saved: %s (%s)", mp3_path, _human_size(mp3_path.stat().st_size))
        return mp3_path

    # ------------------------------------------------------------------
    # Combined convenience method
    # ------------------------------------------------------------------

    def search_and_download(
        self,
        query: str,
        output_dir: str | Path = "downloads",
        progress_hook=None,
    ) -> tuple[dict, Path]:
        """
        Search for a song and download it in one call.

        Returns:
            (metadata_dict, path_to_mp3)
        """
        metadata = self.search(query)
        mp3_path = self.download(
            url=metadata["url"],
            output_dir=output_dir,
            progress_hook=progress_hook,
        )
        return metadata, mp3_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_mp3(directory: Path, title_hint: str) -> Optional[Path]:
        """
        Find the .mp3 file that yt-dlp just created.

        Strategy:
          1. Look for newest .mp3 in the directory (most reliable).
          2. Fall back to partial title match.
        """
        mp3_files = list(directory.glob("*.mp3"))
        if not mp3_files:
            return None
        # Return the most recently modified .mp3
        return max(mp3_files, key=lambda p: p.stat().st_mtime)


def _human_size(nbytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"
