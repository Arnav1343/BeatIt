package com.beatit.app

import android.webkit.JavascriptInterface

/**
 * Exposed to the page as `AndroidMedia`.
 *
 * The page calls updateState() whenever playback changes — track, play/pause,
 * and roughly once a second while playing so the lockscreen scrubber keeps up.
 *
 * Calls arrive on a WebView JavaScript thread, not the main thread, so
 * everything downstream of here has to be safe about that; PlaybackSession
 * posts to the WebView before touching it.
 */
class MediaBridge {

    @JavascriptInterface
    fun updateState(json: String) {
        PlaybackSession.updateFromPage(json)
    }
}
