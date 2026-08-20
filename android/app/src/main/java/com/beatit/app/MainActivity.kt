package com.beatit.app

import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.webkit.*
import android.app.Activity

class MainActivity : Activity() {

    private lateinit var webView: WebView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Start the foreground service (keeps server + downloads alive in background)
        val serviceIntent = Intent(this, MusicServerService::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(serviceIntent)
        } else {
            startService(serviceIntent)
        }

        webView = findViewById(R.id.webView)
        setupWebView()

        // Wait for server to start, then load
        webView.postDelayed({
            webView.loadUrl("http://localhost:8080")
        }, 1200)
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
                // Server not ready yet — retry after a short delay
                if (request?.isForMainFrame == true) {
                    view?.postDelayed({ view.reload() }, 1000)
                }
            }
        }

        webView.webChromeClient = WebChromeClient()
    }

    @Deprecated("Deprecated in Java")
    override fun onBackPressed() {
        if (webView.canGoBack()) {
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
}
