package com.beatit.app

import org.schabi.newpipe.extractor.stream.StreamInfoItem
import kotlin.math.abs

object TrackMapper {
    private const val MATCHER_VERSION = "v1"

    suspend fun mapTrack(
        track: Track,
        dao: BatchDao,
        youtubeHelper: YoutubeHelper,
        excludeUrl: String? = null
    ): Pair<String?, Double> {
        val fingerprint = "${track.fingerprint}:$MATCHER_VERSION"

        // "Try Again": the cache and the default query would just hand back the
        // candidate the user already rejected, so skip the cache, vary the
        // query, and drop that candidate from the results.
        if (excludeUrl != null) return mapAlternative(track, youtubeHelper, excludeUrl)

        // 1. Check persistent Match Cache
        val cached = dao.getMatchCache(fingerprint)
        if (cached != null) {
            if (cached.youtubeVideoId == "NO_MATCH") return Pair(null, 0.0)
            return Pair(cached.youtubeVideoId, 1.0) // Cached matches treated as high confidence
        }

        // 2. Search YouTube
        val query = "${track.title} ${track.artist}"
        var results = youtubeHelper.search(query, 10)

        // 3. Fallback Search if no results
        if (results.isEmpty()) {
            results = youtubeHelper.search("${track.title} ${track.artist} audio", 10)
        }

        if (results.isEmpty()) {
            dao.insertMatchCache(MatchCacheEntry(fingerprint, "NO_MATCH"))
            return Pair(null, 0.0)
        }

        // 4. Deterministic Scoring & Sorting
        val scoredResults = results.map { item ->
            val score = calculateScore(track, item)
            item to score
        }.sortedWith(compareByDescending<Pair<StreamInfoItem, Double>> { it.second }
            .thenBy { abs((track.durationSeconds ?: 0) - it.first.duration.toInt()) }
            .thenByDescending { it.first.uploaderName?.contains("VEVO") == true || it.first.uploaderName?.lowercase()?.contains("official") == true }
            .thenByDescending { it.first.viewCount }
        )

        val bestMatch = scoredResults.first()
        val videoId = bestMatch.first.url // NewPipe url is often the ID or full link
        val confidence = bestMatch.second

        // 5. Update Cache if high confidence
        if (confidence >= 0.75) {
            dao.insertMatchCache(MatchCacheEntry(fingerprint, videoId))
        }

        return Pair(videoId, confidence)
    }

    private fun mapAlternative(
        track: Track,
        youtubeHelper: YoutubeHelper,
        excludeUrl: String
    ): Pair<String?, Double> {
        val base = "${track.title} ${track.artist}"
        for (query in listOf("$base audio", "$base official audio", "$base lyrics")) {
            val results = youtubeHelper.search(query, 10).filter { it.url != excludeUrl }
            if (results.isEmpty()) continue
            val best = results
                .map { it to calculateScore(track, it) }
                .sortedWith(compareByDescending<Pair<StreamInfoItem, Double>> { it.second }
                    .thenBy { abs((track.durationSeconds ?: 0) - it.first.duration.toInt()) })
                .first()
            // Deliberately not written to the match cache: this is a user-driven
            // second opinion, not a verified match.
            return Pair(best.first.url, best.second)
        }
        return Pair(null, 0.0)
    }

    private fun calculateScore(track: Track, item: StreamInfoItem): Double {
        val normTrackTitle = PlaylistExtractor.sanitize(track.title)
        val normItemTitle = PlaylistExtractor.sanitize(item.name ?: "")
        val normTrackArtist = PlaylistExtractor.sanitize(track.artist)
        val normItemArtist = PlaylistExtractor.sanitize(item.uploaderName ?: "")

        var score = 0.0

        // Title match (30%)
        if (normItemTitle.contains(normTrackTitle) || normTrackTitle.contains(normItemTitle)) score += 0.3
        if (normItemTitle == normTrackTitle) score += 0.2

        // Artist match (30%)
        if (normItemArtist.contains(normTrackArtist) || normItemArtist.contains(normTrackArtist)) score += 0.3
        
        // Duration match (20%)
        track.durationSeconds?.let { target ->
            val itemDur = item.duration.toInt()
            val diff = abs(target - itemDur)
            when {
                diff <= 2 -> score += 0.2
                diff <= 5 -> score += 0.15
                diff <= 10 -> score += 0.1
            }
        } ?: run { score += 0.1 } // Neutral if no duration

        // Channel/Official match (10%)
        if (item.uploaderName?.lowercase()?.contains("official") == true || 
            item.uploaderName?.contains("VEVO") == true) {
            score += 0.1
        }

        return score.coerceIn(0.0, 1.0)
    }
}
