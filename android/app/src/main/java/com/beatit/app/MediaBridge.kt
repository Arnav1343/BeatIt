package com.beatit.app

import android.webkit.JavascriptInterface

/**
 * Exposed to the page as `AndroidPlayer`. The page is a remote control and a
 * view; the audio itself is played by NativePlayer inside the service.
 *
 * Calls arrive on WebView JavaScript threads; NativePlayer serialises them.
 */
class MediaBridge {
    @JavascriptInterface fun play(filename: String, title: String) = NativePlayer.play(filename, title)
    @JavascriptInterface fun resume() = NativePlayer.resume()
    @JavascriptInterface fun pause() = NativePlayer.pause()
    @JavascriptInterface fun toggle() = NativePlayer.toggle()
    @JavascriptInterface fun next() = NativePlayer.next()
    @JavascriptInterface fun prev() = NativePlayer.prev()
    @JavascriptInterface fun seek(ms: Long) = NativePlayer.seek(ms)
    @JavascriptInterface fun setQueue(json: String) = NativePlayer.setQueue(json)
    @JavascriptInterface fun setModes(shuffle: Boolean, repeat: Boolean) = NativePlayer.setModes(shuffle, repeat)
    @JavascriptInterface fun stopIfPlaying(filename: String) = NativePlayer.stopIfPlaying(filename)
    @JavascriptInterface fun state(): String = NativePlayer.state()
}
