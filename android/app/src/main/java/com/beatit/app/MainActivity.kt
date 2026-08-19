package com.beatit.app

import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Log
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

        // Handle deep link if the app was launched by it
        handleSpotifyCallback(intent)
    }

    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)
        setIntent(intent)
        intent?.let { handleSpotifyCallback(it) }
    }

    private fun handleSpotifyCallback(intent: Intent) {
        val uri = intent.data ?: return
        if (uri.scheme == "beatit" && uri.host == "callback") {
            val code = uri.getQueryParameter("code")
            val error = uri.getQueryParameter("error")

            if (error != null) {
                Log.e("MainActivity", "Spotify auth denied: $error")
                webView.post {
                    webView.loadUrl("http://localhost:8080")
                    webView.postDelayed({
                        webView.evaluateJavascript(
                            "showToast && showToast('Spotify auth denied', 'error')", null
                        )
                    }, 2000)
                }
                return
            }

            if (code != null) {
                Log.d("MainActivity", "Spotify auth code received, exchanging...")
                Thread {
                    SpotifyAuth.init(this)
                    val success = SpotifyAuth.handleCallback(code)
                    runOnUiThread {
                        webView.loadUrl("http://localhost:8080")
                        if (success) {
                            webView.postDelayed({
                                webView.evaluateJavascript(
                                    "showToast && showToast('Connected to Spotify!', 'success')", null
                                )
                            }, 2000)
                        }
                    }
                }.start()
            }
        }
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

        webView.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(
                view: WebView?, request: WebResourceRequest?
            ): Boolean {
                val url = request?.url ?: return false
                val scheme = url.scheme ?: return false

                // Let localhost requests load inside the WebView
                if (url.host == "localhost" || url.host == "127.0.0.1") return false

                // Handle custom scheme (beatit://callback) or external URLs via system
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

    // Counteract WebView's implicit pause of timers/media when the Activity
    // stops (minimized/switched away) — without this, Chromium suspends the
    // <audio> element's playback along with everything else on the page.
    override fun onStop() {
        super.onStop()
        webView.onResume()
        webView.resumeTimers()
    }

    // Don't stop the service on destroy — let it keep running for downloads
    override fun onDestroy() {
        super.onDestroy()
    }
}
