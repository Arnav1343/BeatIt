package com.beatit.app

import android.content.Context
import android.util.AttributeSet
import android.view.View
import android.webkit.WebView

/**
 * A WebView that never reports itself as hidden.
 *
 * Chromium suspends media playback for WebContents it believes are not
 * visible. Android WebView derives that from the view/window visibility
 * callbacks, so minimising the app or turning the screen off pauses the
 * <audio> element outright — and resumes it when the app comes back, which
 * is exactly the symptom this works around. A MediaSession alone does not
 * prevent it; the suspension happens before the session is consulted.
 *
 * Swallowing both callbacks keeps the contents "visible" as far as Chromium
 * is concerned, so audio keeps running. Playback still only happens while
 * MusicServerService holds its mediaPlayback foreground service and wake
 * lock, so this doesn't keep the app alive on its own.
 */
class BackgroundAudioWebView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : WebView(context, attrs, defStyleAttr) {

    /** Fired when the containing window is hidden (minimise, screen off). */
    override fun onWindowVisibilityChanged(visibility: Int) {
        super.onWindowVisibilityChanged(View.VISIBLE)
    }

    /** Fired when this view or an ancestor changes visibility. */
    override fun onVisibilityChanged(changedView: View, visibility: Int) {
        super.onVisibilityChanged(changedView, View.VISIBLE)
    }
}
