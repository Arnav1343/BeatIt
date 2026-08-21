package com.beatit.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.ViewGroup
import android.webkit.*
import android.app.Activity
import androidx.core.view.WindowCompat

class MainActivity : Activity() {

    private lateinit var webView: WebView

    /** Renderer deaths inside RENDERER_DEATH_WINDOW_MS of each other. */
    private var rendererDeaths = 0
    private var lastRendererDeath = 0L

    /** Main-frame load attempts since the WebView was built. */
    private var loadAttempts = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Let the WebView draw under the system bars. Without this the window
        // is inset for them and the wash stops at a black band top and bottom;
        // the page re-insets its own content via env(safe-area-inset-*).
        WindowCompat.setDecorFitsSystemWindows(window, false)

        // Start the foreground service (keeps server + downloads alive in background)
        val serviceIntent = Intent(this, MusicServerService::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(serviceIntent)
        } else {
            startService(serviceIntent)
        }

        installWebView()

        // After installWebView, deliberately: the permission dialog must not
        // sit in front of the page load.
        requestNotificationPermission()
    }

    /**
     * Android 13 made POST_NOTIFICATIONS a runtime permission, denied until
     * asked. It was declared in the manifest but never requested, so on every
     * Android 13+ device NotificationManager.notify() was silently dropped —
     * including the foreground service's own notification.
     *
     * That is what the media controls need to exist at all. The lockscreen
     * transport, and the OEM "dynamic island" surfaces built on top of it
     * (HyperOS Smart Island, vivo Atomic Island, realme Fluid Cloud, One UI
     * Now Bar), all decorate an ongoing MediaStyle notification — with no
     * notification there is nothing for them to show.
     *
     * Denial is not fatal: the server, downloads and playback all carry on,
     * only without a notification to control them from.
     */
    private fun requestNotificationPermission() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) return
        val permission = Manifest.permission.POST_NOTIFICATIONS
        if (checkSelfPermission(permission) == PackageManager.PERMISSION_GRANTED) return
        requestPermissions(arrayOf(permission), REQ_POST_NOTIFICATIONS)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != REQ_POST_NOTIFICATIONS) return
        val granted = grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED
        if (!granted) {
            Log.w(TAG, "Notifications denied — no lockscreen or island controls")
        }
    }

    /**
     * Builds the WebView and makes it the content view.
     *
     * This is the only path that ever creates one, so recovering from a dead
     * renderer goes through exactly the same setup as a cold start — a
     * recovery path that drifts from the real one is a recovery path that
     * stops working.
     */
    private fun installWebView() {
        val view = BackgroundAudioWebView(this)
        view.layoutParams = ViewGroup.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        )
        setContentView(view)

        webView = view
        setupWebView()

        // Load straight away rather than sleeping first. The server usually
        // is up by now, and when it isn't, onReceivedError retries — a fixed
        // delay here bought nothing but a guaranteed stretch of blank window
        // on every cold start.
        loadAttempts = 0
        view.loadUrl(SERVER_URL)
    }

    private fun setupWebView() {
        webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            allowFileAccess = true
            allowContentAccess = true
            mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
            cacheMode = WebSettings.LOAD_NO_CACHE
            mediaPlaybackRequiresUserGesture = false
        }

        // The WebView has to hold Android focus for the IME to attach to it.
        // Without this, tapping a text field focuses the element inside the
        // page but no keyboard appears — the system keeps serving the
        // DecorView instead, so showSoftInput() silently does nothing.
        webView.isFocusable = true
        webView.isFocusableInTouchMode = true
        webView.requestFocus()

        // Lockscreen / notification transport controls. Registered before the
        // first loadUrl so the page always sees it, including on the
        // onReceivedError reload path.
        webView.addJavascriptInterface(MediaBridge(), "AndroidMedia")
        PlaybackSession.attachWebView(webView)

        webView.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(
                view: WebView?, request: WebResourceRequest?
            ): Boolean {
                val url = request?.url ?: return false

                // Let localhost requests load inside the WebView
                if (url.host == "localhost" || url.host == "127.0.0.1") return false

                // Hand any external URL to the system browser
                val intent = Intent(Intent.ACTION_VIEW, url)
                startActivity(intent)
                return true
            }

            override fun onReceivedError(
                view: WebView?, request: WebResourceRequest?, error: WebResourceError?
            ) {
                // Server not ready yet — retry, quickly. This runs during the
                // blank-window part of a cold start, so the retry interval is
                // most of what the user experiences as startup time; a whole
                // second per attempt is why that used to drag.
                if (request?.isForMainFrame != true) return

                // Back off rather than stop. The service starts asynchronously
                // and has a Room database to open before NanoHTTPD binds, so on
                // a loaded device — the very case this is meant to help — the
                // server can take a while. A retry ceiling here would leave a
                // permanently blank app needing a force-close, which is worse
                // than the slow start it was meant to fix.
                loadAttempts++
                val delay = (LOAD_RETRY_MS shl (loadAttempts - 1).coerceAtMost(3))
                    .coerceAtMost(MAX_LOAD_RETRY_MS)
                view?.postDelayed({ view.loadUrl(SERVER_URL) }, delay)
            }

            /**
             * The WebView's renderer runs in its own process, and Android kills
             * it under memory pressure — routinely on low-RAM devices, and
             * without any crash of ours. Returning false (the default when this
             * is not overridden) makes the framework kill the whole app process
             * in response: no exception, no dialog, the app simply vanishes.
             *
             * Returning true keeps our process alive. The dead WebView can
             * never render again and must be thrown away, so build a fresh one
             * and reload. Playback is lost either way — the <audio> element
             * lived in the renderer that just died.
             */
            override fun onRenderProcessGone(
                view: WebView?, detail: RenderProcessGoneDetail?
            ): Boolean {
                Log.w(TAG, "WebView renderer gone (didCrash=${detail?.didCrash()}) — rebuilding")

                val dead = view ?: return true
                if (dead !== webView) return true  // stale client on a discarded view

                PlaybackSession.detachWebView()
                (dead.parent as? ViewGroup)?.removeView(dead)
                dead.destroy()

                if (isFinishing || isDestroyed) return true

                val now = System.currentTimeMillis()
                rendererDeaths =
                    if (now - lastRendererDeath < RENDERER_DEATH_WINDOW_MS) rendererDeaths + 1 else 1
                lastRendererDeath = now

                // Rebuilding into a renderer that dies again immediately would
                // spin forever. Give up cleanly instead of burning the battery.
                if (rendererDeaths > MAX_RENDERER_DEATHS) {
                    Log.e(TAG, "Renderer died $rendererDeaths times in a row — closing")
                    finish()
                    return true
                }

                installWebView()
                return true
            }
        }

        webView.webChromeClient = WebChromeClient()
    }

    @Deprecated("Deprecated in Java")
    override fun onBackPressed() {
        if (!isFinishing && webView.canGoBack()) {
            webView.goBack()
        } else {
            super.onBackPressed()
        }
    }

    // Don't stop the service on destroy — let it keep running for downloads.
    // Playback does go away with the WebView though (the <audio> element lives
    // in the page), so hand back the session rather than leaving a media
    // notification whose buttons would no-op.
    override fun onDestroy() {
        PlaybackSession.detachWebView()
        super.onDestroy()
    }

    companion object {
        private const val TAG = "MainActivity"
        private const val SERVER_URL = "http://localhost:8080"
        private const val REQ_POST_NOTIFICATIONS = 1
        private const val LOAD_RETRY_MS = 150L      // first retry, then doubling
        private const val MAX_LOAD_RETRY_MS = 1_000L
        private const val RENDERER_DEATH_WINDOW_MS = 30_000L
        private const val MAX_RENDERER_DEATHS = 3
    }
}
