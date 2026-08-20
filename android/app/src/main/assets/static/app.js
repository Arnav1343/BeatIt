(() => {
    'use strict';

    // Lock viewport height so keyboard doesn't shrink the layout
    const setAppHeight = () => {
        document.documentElement.style.setProperty('--app-height', window.innerHeight + 'px');
    };
    setAppHeight();
    screen.orientation?.addEventListener('change', setAppHeight);

    const $ = s => document.querySelector(s);
    const $$ = s => [...document.querySelectorAll(s)];

    const views = {
        menu: $('#viewMenu'),
        search: $('#viewSearch'),
        library: $('#viewLibrary'),
        nowplaying: $('#viewNowPlaying'),
        import: $('#viewImport'),
        batches: $('#viewBatches'),
        batchDetail: $('#viewBatchDetail'),
        actionMenu: $('#viewActionMenu')
    };
    const audio = $('#audioPlayer');

    // ─── Media Session ────────────────────────────────────────────
    // NOTE: Android WebView does not implement the Media Session API —
    // navigator.mediaSession is undefined there (verified on WebView
    // Chrome 150 / Android 16), so none of this runs inside the app. It is
    // kept for the desktop Flask build, where the same UI runs in a real
    // browser and this provides OS media keys and metadata.
    //
    // This does NOT keep background audio alive. Playback surviving
    // minimise/screen-off is handled natively by BackgroundAudioWebView,
    // which stops Chromium marking the document hidden and suspending
    // <audio>. Do not "fix" background playback by editing this block.
    if ('mediaSession' in navigator) {
        navigator.mediaSession.setActionHandler('play', () => togglePlay());
        navigator.mediaSession.setActionHandler('pause', () => togglePlay());
        navigator.mediaSession.setActionHandler('previoustrack', () => prevTrack());
        navigator.mediaSession.setActionHandler('nexttrack', () => nextTrack());
        navigator.mediaSession.setActionHandler('seekto', (details) => {
            if (details.seekTime != null && audio.duration) audio.currentTime = details.seekTime;
        });
    }

    function updateMediaSession() {
        if (!('mediaSession' in navigator)) return;
        if (currentSong) {
            navigator.mediaSession.metadata = new MediaMetadata({
                title: currentSong.title || 'Unknown',
                artist: 'BeatIt',
            });
            navigator.mediaSession.playbackState = isPlaying ? 'playing' : 'paused';
        } else {
            navigator.mediaSession.playbackState = 'none';
        }
    }

    // ─── Native transport controls (lockscreen / notification) ─────
    // Because WebView has no Media Session API, the native side owns the
    // MediaSession instead. State goes out through the AndroidMedia bridge;
    // commands come back in by calling window.BeatItRemote. Absent on the
    // desktop build, where the real Media Session API above does the job.
    let lastNativePush = 0;

    function pushNativeState() {
        if (!window.AndroidMedia) return;
        try {
            window.AndroidMedia.updateState(JSON.stringify({
                title: currentSong ? (currentSong.title || 'Unknown') : '',
                artist: 'BeatIt',
                durationMs: Math.round((audio.duration || 0) * 1000),
                positionMs: Math.round((audio.currentTime || 0) * 1000),
                playing: isPlaying,
                hasTrack: !!currentSong,
                // Just the name — the native side resolves it to a path.
                filename: currentSong ? currentSong.filename : ''
            }));
        } catch (e) { /* bridge missing or page tearing down */ }
    }

    window.BeatItRemote = {
        play() { if (!isPlaying) togglePlay(); },
        pause() { if (isPlaying) togglePlay(); },
        toggle() { togglePlay(); },
        next() { nextTrack(); },
        prev() { prevTrack(); },
        seek(ms) {
            if (audio.duration) { audio.currentTime = ms / 1000; pushNativeState(); }
        }
    };

    const searchInput = $('#searchInput');
    const suggestionsDropdown = $('#suggestionsDropdown');
    const searchStatus = $('#searchStatus');
    const searchResult = $('#searchResult');
    const resultTitle = $('#resultTitle');
    const resultMeta = $('#resultMeta');
    const downloadBtn = $('#downloadBtn');
    const downloadStatus = $('#downloadStatus');
    const downloadProgressWrap = $('#downloadProgressWrap');
    const downloadProgressFill = $('#downloadProgressFill');
    const downloadProgressText = $('#downloadProgressText');
    const libraryList = $('#libraryList');
    const libraryBadge = $('#libraryBadge');
    const npTitle = $('#npTitle');
    const npArtist = $('#npArtist');
    const npTrackNum = $('#npTrackNum');
    const npProgressFill = $('#npProgressFill');
    const npProgressKnob = $('#npProgressKnob');
    const npTimeElapsed = $('#npTimeElapsed');
    const npTimeTotal = $('#npTimeTotal');
    const loadingOverlay = $('#loadingOverlay');
    const loadingText = $('#loadingText');
    const toastEl = $('#toast');

    let currentView = 'menu';
    let viewHistory = [];
    let menuIndex = 0;
    let libraryIndex = 0;
    let library = [];
    let currentSong = null;
    let currentSongIndex = -1;
    let isPlaying = false;
    let lastSearchResult = null;
    let toastTimeout = null;
    let progressPollInterval = null;
    let isShuffle = false;
    let isRepeat = false;

    // ─── Batch Import state ─────────────────────────────────────────
    let batches = [];
    let batchesIndex = 0;
    let batchTracks = [];
    let batchTracksIndex = 0;
    let activeBatchId = null;
    let activeTrackId = null;
    let batchPollInterval = null;
    let batchDetailPollInterval = null;
    let actionMenuIndex = 0;

    // ─── Theme System ───────────────────────────────────────────────
    const themes = [
        { id: 'default', name: 'Pink / Black', dark: true },
        { id: 'charcoal', name: 'Charcoal / Peach', dark: true },
        { id: 'neon', name: 'Neon Green / Jet Black', dark: true },
        { id: 'purple', name: 'Purple / Black', dark: true },
        { id: 'deeppurple', name: 'Deep Purple / Rose Gold', dark: true },
        { id: 'light', name: 'Light', dark: false },
        { id: 'light-rose', name: 'Rose Light', dark: false },
        { id: 'light-mint', name: 'Mint Light', dark: false },
    ];
    let currentThemeIndex = 0;
    const savedTheme = localStorage.getItem('ipod-theme-id');
    if (savedTheme) {
        const idx = themes.findIndex(t => t.id === savedTheme);
        if (idx >= 0) currentThemeIndex = idx;
    }

    function applyTheme() {
        const t = themes[currentThemeIndex];
        if (t.id === 'default') document.body.removeAttribute('data-theme');
        else document.body.setAttribute('data-theme', t.id);
        localStorage.setItem('ipod-theme-id', t.id);
    }

    function cycleTheme() {
        currentThemeIndex = (currentThemeIndex + 1) % themes.length;
        applyTheme();
        showToast(themes[currentThemeIndex].name);
    }

    function showView(name, pushHistory = true) {
        if (pushHistory && currentView !== name) viewHistory.push(currentView);
        currentView = name;
        Object.entries(views).forEach(([k, el]) => el?.classList.toggle('active', k === name));

        if (batchPollInterval) { clearInterval(batchPollInterval); batchPollInterval = null; }
        if (batchDetailPollInterval) { clearInterval(batchDetailPollInterval); batchDetailPollInterval = null; }

        if (name === 'library') refreshLibrary();
        if (name === 'import') { $('#importInput').focus(); }
        if (name === 'batches') { refreshBatches(); batchPollInterval = setInterval(refreshBatches, 2000); }
        if (name === 'batchDetail') { refreshBatchDetail(); batchDetailPollInterval = setInterval(refreshBatchDetail, 1000); }

        if (name === 'search') {
            if (activeDownloadTaskId) {
                // A download is still running — show its progress rather than
                // wiping the view out from under it.
                downloadProgressWrap.classList.remove('hidden');
                downloadBtn.disabled = true;
                downloadStatus.textContent = 'Downloading...';
                if (activeDownloadTitle) { resultTitle.textContent = activeDownloadTitle; searchResult.classList.remove('hidden'); }
            } else {
                resetSearchView();
            }
            searchInput.focus();
        }
        if (name === 'nowplaying') updateNowPlaying();

        // Reset indices
        if (name === 'menu') menuIndex = 0;
        if (name === 'actionMenu') actionMenuIndex = 0;
        if (name === 'library') libraryIndex = 0;
        if (name === 'batches') batchesIndex = 0;
        if (name === 'batchDetail') batchTracksIndex = 0;

        if (name !== 'search' && name !== 'actionMenu') {
            activeTrackId = null;
            downloadBtn.textContent = 'Download Audio';
        }
        updateAllMenuSels();
    }

    function goBack() { if (viewHistory.length) showView(viewHistory.pop(), false); else if (currentView !== 'menu') showView('menu'); }
    function showLoading(msg) { loadingText.textContent = msg || 'Loading...'; loadingOverlay.classList.remove('hidden'); }
    function hideLoading() { loadingOverlay.classList.add('hidden'); }

    function showToast(msg, type, dur) {
        if (toastTimeout) clearTimeout(toastTimeout);
        toastEl.textContent = msg;
        toastEl.className = 'toast' + (type ? ' toast-' + type : '');
        void toastEl.offsetWidth;
        toastEl.classList.add('visible');
        toastTimeout = setTimeout(() => { toastEl.classList.remove('visible'); setTimeout(() => toastEl.classList.add('hidden'), 250); }, dur || 2000);
    }

    function updateAllMenuSels() {
        $$('#viewMenu .menu-item').forEach((el, i) => el.classList.toggle('selected', i === menuIndex));
        $$('#viewActionMenu .menu-item').forEach((el, i) => el.classList.toggle('selected', i === actionMenuIndex));
    }

    function menuUp() {
        if (currentView === 'menu') menuIndex = Math.max(0, menuIndex - 1);
        else if (currentView === 'actionMenu') actionMenuIndex = Math.max(0, actionMenuIndex - 1);
        updateAllMenuSels();
    }

    function menuDown() {
        if (currentView === 'menu') {
            const items = $$('#viewMenu .menu-item');
            menuIndex = Math.min(items.length - 1, menuIndex + 1);
        } else if (currentView === 'actionMenu') {
            const items = $$('#viewActionMenu .menu-item');
            actionMenuIndex = Math.min(items.length - 1, actionMenuIndex + 1);
        }
        updateAllMenuSels();
    }

    function menuSelect() {
        const items = $$('#viewMenu .menu-item');
        const a = items[menuIndex]?.dataset.action;
        if (!a) return;
        if (a === 'themes') cycleTheme();
        else showView(a);
    }

    async function actionMenuSelect() {
        const items = $$('#viewActionMenu .menu-item');
        const action = items[actionMenuIndex]?.dataset.action;
        if (!action || !activeTrackId) return;

        if (action === 'manual') {
            const track = batchTracks.find(t => t.id === activeTrackId);
            if (track) searchInput.value = track.title;
            downloadBtn.textContent = 'Pick for Import';
            showView('search');
            return;
        }

        showLoading('Applying action...');
        try {
            const r = await fetch('/api/import/action', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ track_id: activeTrackId, action })
            });
            const d = await r.json();
            if (d.success) {
                showToast('Action applied', 'success');
                activeTrackId = null;
                goBack();
            } else {
                showToast(d.error || 'Failed', 'error');
            }
        } catch { showToast('Network error', 'error'); }
        finally { hideLoading(); }
    }

    // ─── Suggestions ───────────────────────────────────────────────
    let suggestDebounce = null;
    let suggestionsData = [];
    let suggestIndex = -1;

    function formatDuration(s) {
        if (!s) return '';
        return Math.floor(s / 60) + ':' + String(Math.floor(s % 60)).padStart(2, '0');
    }

    function renderSuggestions(items) {
        suggestionsData = items;
        suggestIndex = -1;
        if (!items.length) { suggestionsDropdown.classList.add('hidden'); return; }
        suggestionsDropdown.innerHTML = items.map((s, i) => {
            const thumb = s.thumbnail ? `<img class="suggestion-thumb" src="${esc(s.thumbnail)}" alt="">` : `<div class="suggestion-thumb"></div>`;
            return `<div class="suggestion-item" data-idx="${i}">${thumb}<div class="suggestion-info"><div class="suggestion-title">${escH(s.title)}</div><div class="suggestion-artist">${escH(s.artist || '')}</div></div><span class="suggestion-dur">${formatDuration(s.duration)}</span></div>`;
        }).join('');
        suggestionsDropdown.classList.remove('hidden');
    }

    function hideSuggestions() {
        suggestionsDropdown.classList.add('hidden');
        suggestionsData = [];
        suggestIndex = -1;
    }

    function selectSuggestion(idx) {
        const item = suggestionsData[idx];
        if (!item) return;
        hideSuggestions();
        searchInput.value = item.title;
        lastSearchResult = {
            id: '', title: item.title, url: item.url,
            duration: item.duration, uploader: item.artist,
            thumbnail: item.thumbnail, artist: item.artist, album: item.album || '',
        };
        resultTitle.textContent = item.title;
        resultMeta.textContent = (item.artist || '') + '  ·  ' + formatDuration(item.duration);
        searchResult.classList.remove('hidden');
        searchStatus.textContent = '✓ Found';
        searchStatus.className = 'search-status';
        downloadBtn.disabled = false;
        updateQualitySizes(item.duration || 0);
    }

    searchInput.addEventListener('input', () => {
        const q = searchInput.value.trim();
        // Editing the query invalidates whatever the last attempt reported.
        // This is what strands "No results found" on screen after a search
        // made with no connection, once the network comes back.
        clearSearchStatus();
        if (!activeDownloadTaskId) searchResult.classList.add('hidden');
        if (q.length < 2) { hideSuggestions(); return; }
        if (suggestDebounce) clearTimeout(suggestDebounce);
        suggestDebounce = setTimeout(async () => {
            try {
                suggestionsDropdown.innerHTML = '<div class="suggestions-loading">Searching...</div>';
                suggestionsDropdown.classList.remove('hidden');
                const r = await fetch('/api/suggestions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: q }) });
                const items = await r.json();
                if (searchInput.value.trim() === q) renderSuggestions(items);
            } catch { hideSuggestions(); }
        }, 250);
    });

    suggestionsDropdown.addEventListener('click', (e) => {
        const item = e.target.closest('.suggestion-item');
        if (item) selectSuggestion(parseInt(item.dataset.idx));
    });

    searchInput.addEventListener('keydown', (e) => {
        if (!suggestionsData.length) { if (e.key === 'Enter') doSearch(); return; }
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            suggestIndex = Math.min(suggestionsData.length - 1, suggestIndex + 1);
            suggestionsDropdown.querySelectorAll('.suggestion-item').forEach((el, i) => el.classList.toggle('selected', i === suggestIndex));
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            suggestIndex = Math.max(0, suggestIndex - 1);
            suggestionsDropdown.querySelectorAll('.suggestion-item').forEach((el, i) => el.classList.toggle('selected', i === suggestIndex));
        } else if (e.key === 'Enter' && suggestIndex >= 0) {
            e.preventDefault();
            selectSuggestion(suggestIndex);
        } else if (e.key === 'Enter') { doSearch(); }
    });

    /** Clear any status text left over from a previous attempt. */
    function clearSearchStatus() {
        searchStatus.textContent = '';
        searchStatus.className = 'search-status';
    }

    /**
     * Return the search view to a clean slate. Without this, whatever was on
     * screen last time survives — a finished download's "✓ Saved", a stale
     * result panel, or a "No results found" from an attempt made offline.
     */
    function resetSearchView() {
        hideSuggestions();
        searchInput.value = '';
        clearSearchStatus();
        searchResult.classList.add('hidden');
        downloadStatus.textContent = '';
        downloadStatus.className = 'download-status';
        downloadProgressWrap.classList.add('hidden');
        downloadProgressFill.style.width = '0%';
        downloadProgressFill.classList.remove('paused');
        downloadProgressText.textContent = '';
        downloadBtn.disabled = false;
        downloadBtn.textContent = 'Download Audio';
        lastSearchResult = null;
    }

    async function doSearch() {
        const q = searchInput.value.trim(); if (!q) return;
        hideSuggestions();
        searchStatus.textContent = 'Searching...'; searchStatus.className = 'search-status';
        searchResult.classList.add('hidden'); downloadStatus.textContent = '';
        downloadProgressWrap.classList.add('hidden');
        showLoading('Searching...');
        try {
            const r = await fetch('/api/search', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: q }) });
            const d = await r.json();
            if (d.error) { searchStatus.textContent = d.error; searchStatus.className = 'search-status error-text'; showToast('Not found', 'error'); return; }
            lastSearchResult = d;
            resultTitle.textContent = d.title;
            resultMeta.textContent = (d.uploader || d.artist || '') + '  ·  ' + formatDuration(d.duration || 0);
            searchResult.classList.remove('hidden');
            searchStatus.textContent = '✓ Found';
            searchStatus.className = 'search-status';
            downloadBtn.disabled = false;
            updateQualitySizes(d.duration || 0);
        } catch {
            searchStatus.textContent = 'Search failed'; searchStatus.className = 'search-status error-text'; showToast('Failed', 'error');
        } finally { hideLoading(); }
    }

    const CODEC_OPTIONS = {
        mp3: [
            { q: 320, label: 'High', detail: '320 kbps' },
            { q: 192, label: 'Medium', detail: '192 kbps' },
            { q: 128, label: 'Low', detail: '128 kbps' },
        ],
        opus: [
            { q: 160, label: 'High', detail: '160 kbps' },
            { q: 128, label: 'Medium', detail: '128 kbps', default: true },
            { q: 96, label: 'Low', detail: '96 kbps' },
        ],
    };
    let selectedCodec = 'mp3';
    let selectedQuality = 320;

    function estimateSize(durationSec, kbps) {
        if (!durationSec) return '—';
        const bytes = (kbps * 1000 / 8) * durationSec;
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    function renderQualityOptions() {
        const opts = CODEC_OPTIONS[selectedCodec];
        const defOpt = opts.find(o => o.default) || opts[0];
        const validQs = opts.map(o => o.q);
        if (!validQs.includes(selectedQuality)) selectedQuality = defOpt.q;

        const dur = lastSearchResult?.duration || 0;
        const container = $('#qualityOptions');
        container.innerHTML = opts.map(o => {
            const active = o.q === selectedQuality ? 'active' : '';
            return `<button class="quality-opt ${active}" data-quality="${o.q}">
                <span class="q-name">${o.label}</span>
                <span class="q-detail">${o.detail} · <span class="q-size">${estimateSize(dur, o.q)}</span></span>
            </button>`;
        }).join('');

        $$('.quality-opt').forEach(btn => btn.addEventListener('click', () => {
            $$('.quality-opt').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedQuality = parseInt(btn.dataset.quality);
        }));
    }

    function updateQualitySizes(duration) {
        $$('.quality-opt').forEach(btn => {
            const sz = btn.querySelector('.q-size');
            if (sz) sz.textContent = estimateSize(duration, parseInt(btn.dataset.quality));
        });
    }

    $$('.fmt-btn').forEach(btn => btn.addEventListener('click', () => {
        $$('.fmt-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        selectedCodec = btn.dataset.codec;
        renderQualityOptions();
    }));

    renderQualityOptions();

    let activeDownloadTaskId = null;
    let activeDownloadTitle = null;

    async function doDownload() {
        if (!lastSearchResult) return;

        if (activeTrackId) {
            const videoId = lastSearchResult.url.includes('v=') ? lastSearchResult.url.split('v=')[1].split('&')[0] : null;
            if (videoId) {
                showLoading('Attaching...');
                try {
                    await fetch('/api/import/action', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ track_id: activeTrackId, action: 'accept', video_id: videoId })
                    });
                    activeTrackId = null;
                    downloadBtn.textContent = 'Download Audio';
                    showToast('Attached to batch', 'success');
                    showView('batchDetail');
                } catch { showToast('Failed to attach', 'error'); }
                finally { hideLoading(); }
                return;
            }
        }

        downloadBtn.disabled = true; downloadStatus.textContent = 'Starting...'; downloadStatus.className = 'download-status';
        downloadProgressWrap.classList.remove('hidden'); downloadProgressFill.style.width = '0%'; downloadProgressText.textContent = '0%';
        try {
            // Send the thumbnail too — the downloaded audio carries no embedded
            // art, so this is the only chance to capture a cover for it.
            const payload = {
                url: lastSearchResult.url,
                title: lastSearchResult.title,
                quality: selectedQuality,
                codec: selectedCodec,
                thumbnail: lastSearchResult.thumbnail || ''
            };
            const r = await fetch('/api/download', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
            const d = await r.json();
            if (d.error) { downloadStatus.textContent = d.error; downloadStatus.className = 'download-status error'; downloadProgressWrap.classList.add('hidden'); downloadBtn.disabled = false; return; }
            activeDownloadTaskId = d.task_id;
            activeDownloadTitle = lastSearchResult.title;
            pollProgress(d.task_id);
        } catch { downloadStatus.textContent = 'Failed'; downloadStatus.className = 'download-status error'; downloadProgressWrap.classList.add('hidden'); downloadBtn.disabled = false; }
    }

    function pollProgress(id) {
        if (progressPollInterval) clearInterval(progressPollInterval);
        progressPollInterval = setInterval(async () => {
            try {
                const r = await fetch('/api/progress/' + id); const d = await r.json();
                const onSearch = currentView === 'search';
                if (d.status === 'downloading') {
                    if (onSearch) { downloadProgressWrap.classList.remove('hidden'); downloadProgressFill.style.width = d.percent + '%'; downloadProgressText.textContent = d.percent + '%'; downloadStatus.textContent = 'Downloading...'; downloadStatus.className = 'download-status'; downloadProgressFill.classList.remove('paused'); }
                }
                else if (d.status === 'paused') {
                    if (onSearch) { downloadProgressWrap.classList.remove('hidden'); downloadProgressFill.style.width = d.percent + '%'; downloadProgressText.textContent = d.percent + '%'; downloadStatus.textContent = d.error || 'Paused — waiting for network...'; downloadStatus.className = 'download-status warning'; downloadProgressFill.classList.add('paused'); }
                }
                else if (d.status === 'converting') {
                    if (onSearch) { downloadProgressFill.style.width = '100%'; downloadProgressText.textContent = '100%'; downloadStatus.textContent = 'Converting...'; downloadProgressFill.classList.remove('paused'); }
                }
                else if (d.status === 'done' && d.result) {
                    clearInterval(progressPollInterval); progressPollInterval = null;
                    activeDownloadTaskId = null;
                    if (onSearch) { downloadProgressFill.style.width = '100%'; downloadProgressText.textContent = '✓'; downloadStatus.textContent = '✓ Saved (' + d.result.size_human + ')'; downloadStatus.className = 'download-status success'; downloadBtn.disabled = false; downloadProgressFill.classList.remove('paused'); }
                    showToast('Downloaded!', 'success');
                    await refreshLibrary();
                } else if (d.status === 'error') {
                    clearInterval(progressPollInterval); progressPollInterval = null;
                    activeDownloadTaskId = null;
                    if (onSearch) { downloadStatus.textContent = d.error || 'Failed'; downloadStatus.className = 'download-status error'; downloadProgressWrap.classList.add('hidden'); downloadBtn.disabled = false; }
                    showToast('Download failed', 'error');
                }
            } catch { }
        }, 800);
    }


    downloadBtn.addEventListener('click', doDownload);
    $('#btnShuffle')?.addEventListener('click', () => { isShuffle = !isShuffle; $('#btnShuffle').classList.toggle('active', isShuffle); showToast(isShuffle ? 'Shuffle on' : 'Shuffle off'); });
    $('#btnRepeat')?.addEventListener('click', () => { isRepeat = !isRepeat; $('#btnRepeat').classList.toggle('active', isRepeat); showToast(isRepeat ? 'Repeat on' : 'Repeat off'); });

    async function refreshLibrary() {
        try { const r = await fetch('/api/library'); library = await r.json(); } catch { library = []; }
        libraryBadge.textContent = library.length > 0 ? library.length : '';
        if (!library.length) { libraryList.innerHTML = '<div class="library-empty">No songs yet.<br>Search & download!</div>'; return; }
        if (libraryIndex >= library.length) libraryIndex = library.length - 1;
        libraryList.innerHTML = library.map((s, i) => {
            const cur = currentSong?.filename === s.filename;
            const dur = s.duration || 0;
            const ds = dur > 0 ? Math.floor(dur / 60) + ':' + String(dur % 60).padStart(2, '0') : '';
            const icon = cur ? '<div class="lib-eq' + (!isPlaying ? ' paused' : '') + '"><span></span><span></span><span></span></div>' : '<span class="lib-icon">♪</span>';
            return '<div class="library-item' + (i === libraryIndex ? ' selected' : '') + (cur ? ' playing' : '') + '" data-i="' + i + '" data-f="' + esc(s.filename) + '">' +
                icon + '<div class="lib-info"><div class="lib-title">' + escH(s.title) + '</div><div class="lib-meta"><span>' + s.size_human + '</span>' + (ds ? '<span>' + ds + '</span>' : '') + '</div></div>' +
                '<button class="lib-delete" data-f="' + esc(s.filename) + '">✕</button></div>';
        }).join('');
        $$('.library-item').forEach(el => el.addEventListener('click', e => { if (e.target.classList.contains('lib-delete')) return; const i = +el.dataset.i; libraryIndex = i; updateLibSel(); if (library[i]) playSong(library[i].filename, library[i].title); }));
        $$('.lib-delete').forEach(b => b.addEventListener('click', async e => { e.stopPropagation(); await deleteSong(b.dataset.f); }));
    }

    async function deleteSong(fn) {
        try {
            const r = await fetch('/api/delete', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ filename: fn }) });
            const d = await r.json();
            if (d.success) { if (currentSong?.filename === fn) { audio.pause(); audio.src = ''; currentSong = null; currentSongIndex = -1; isPlaying = false; } showToast('Deleted', 'success'); await refreshLibrary(); }
        } catch { showToast('Delete failed', 'error'); }
    }

    function updateLibSel() { $$('.library-item').forEach((el, i) => el.classList.toggle('selected', i === libraryIndex)); }
    function libUp() { libraryIndex = Math.max(0, libraryIndex - 1); updateLibSel(); const items = $$('.library-item'); if (items[libraryIndex]) items[libraryIndex].scrollIntoView({ block: 'nearest', behavior: 'smooth' }); }
    function libDown() { libraryIndex = Math.min(library.length - 1, libraryIndex + 1); updateLibSel(); const items = $$('.library-item'); if (items[libraryIndex]) items[libraryIndex].scrollIntoView({ block: 'nearest', behavior: 'smooth' }); }
    function libSelect() { if (library[libraryIndex]) playSong(library[libraryIndex].filename, library[libraryIndex].title); }

    // ─── Cover art & colour ─────────────────────────────────────────
    // Nothing the app downloads has embedded artwork, so covers live as
    // sidecar files served by the local server at /api/art/<filename>.
    // Same-origin matters: a CDN image would taint the canvas and make
    // getImageData() throw, which is exactly what the tinting needs.

    const npArtworkEl = $('#npArtwork');
    const npPlaceholder = npArtworkEl?.querySelector('.np-artwork-placeholder');
    let npArtImg = null;              // reused, never re-created, to avoid reflow/flicker
    let lastArtKey = null;            // updateNowPlaying() fires on every play/pause too
    const themeAccent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();

    function setTint(a, b, c, accent) {
        const root = document.documentElement.style;
        root.setProperty('--tint-a', a);
        root.setProperty('--tint-b', b);
        root.setProperty('--tint-c', c);
        if (accent) {
            root.setProperty('--accent', accent);
            root.setProperty('--accent-fill', 'linear-gradient(90deg,' + accent + ',' + accent + '99)');
        }
    }

    /** Relative luminance, for the contrast guard below. */
    function luminance(r, g, b) {
        const f = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
        return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
    }

    function parseRgb(css) {
        const m = css.match(/(\d+),\s*(\d+),\s*(\d+)/);
        return m ? [+m[1], +m[2], +m[3]] : [240, 240, 240];
    }

    /**
     * Art can be any colour, including one that vanishes against the
     * current theme's text. Fall back to the theme accent rather than
     * shipping unreadable text.
     */
    function accentIsLegible(r, g, b) {
        const textRgb = parseRgb(getComputedStyle(document.documentElement).getPropertyValue('color') ||
            getComputedStyle(document.body).color);
        const l1 = luminance(r, g, b), l2 = luminance(textRgb[0], textRgb[1], textRgb[2]);
        const ratio = (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
        return ratio >= 1.6;
    }

    /** Stable pseudo-colour for tracks with no art, so each still feels distinct. */
    function hashTint(title) {
        let h = 0;
        for (let i = 0; i < title.length; i++) h = (h * 31 + title.charCodeAt(i)) >>> 0;
        const hue = h % 360;
        return [
            `hsla(${hue},70%,45%,0.38)`,
            `hsla(${(hue + 40) % 360},65%,40%,0.30)`,
            `hsla(${(hue + 310) % 360},60%,35%,0.24)`
        ];
    }

    function applyHashFallback(title) {
        const [a, b, c] = hashTint(title || 'BeatIt');
        setTint(a, b, c, null);
        if (npArtImg) npArtImg.classList.add('hidden');
        if (npPlaceholder) npPlaceholder.classList.remove('hidden');
    }

    /** Pull 3 dominant colours out of the artwork and push them into CSS. */
    function extractTint(img, title) {
        try {
            const N = 8;
            const cv = document.createElement('canvas');
            cv.width = N; cv.height = N;
            const ctx = cv.getContext('2d', { willReadFrequently: true });
            ctx.drawImage(img, 0, 0, N, N);
            const data = ctx.getImageData(0, 0, N, N).data;   // throws if tainted

            const buckets = [];
            for (let i = 0; i < data.length; i += 4) {
                const r = data[i], g = data[i + 1], b = data[i + 2];
                const max = Math.max(r, g, b), min = Math.min(r, g, b);
                if (max < 28 || min > 232) continue;          // skip near-black / near-white
                buckets.push({ r, g, b, sat: max - min, lum: luminance(r, g, b) });
            }
            if (!buckets.length) { applyHashFallback(title); return; }

            buckets.sort((x, y) => y.sat - x.sat);
            const pick = i => buckets[Math.min(i, buckets.length - 1)];
            const p0 = pick(0), p1 = pick(Math.floor(buckets.length / 3)), p2 = pick(Math.floor(buckets.length / 2));
            const rgba = (p, a) => `rgba(${p.r},${p.g},${p.b},${a})`;

            const accent = accentIsLegible(p0.r, p0.g, p0.b)
                ? `rgb(${p0.r},${p0.g},${p0.b})`
                : themeAccent;

            setTint(rgba(p0, 0.42), rgba(p1, 0.32), rgba(p2, 0.26), accent);
        } catch (e) {
            // Tainted canvas or decode failure — degrade, never break playback.
            applyHashFallback(title);
        }
    }

    /** Back to the plain theme when nothing is playing. */
    function clearArtwork() {
        lastArtKey = null;
        setTint('transparent', 'transparent', 'transparent', themeAccent);
        if (npArtImg) npArtImg.classList.add('hidden');
        if (npPlaceholder) npPlaceholder.classList.remove('hidden');
    }

    function applyArtwork(filename, title) {
        if (!npArtworkEl) return;
        if (lastArtKey === filename) return;   // called on every play/pause; only react to track changes
        lastArtKey = filename;

        if (!npArtImg) {
            npArtImg = document.createElement('img');
            npArtImg.alt = '';
            npArtworkEl.appendChild(npArtImg);
        }

        // Assume no art until it loads, so we never show a broken frame.
        applyHashFallback(title);

        const url = '/api/art/' + encodeURIComponent(filename);
        npArtImg.onload = () => {
            npArtImg.classList.remove('hidden');
            if (npPlaceholder) npPlaceholder.classList.add('hidden');
            extractTint(npArtImg, title);
        };
        npArtImg.onerror = () => {
            // 404 also tells the server to go look this one up in the
            // background; a later refresh will find it.
            applyHashFallback(title);
        };
        npArtImg.src = url;
    }

    function playSong(fn, title) {
        currentSong = { filename: fn, title };
        currentSongIndex = library.findIndex(s => s.filename === fn);
        audio.src = '/api/music/' + encodeURIComponent(fn);
        audio.play().catch(() => { });
        isPlaying = true;
        showView('nowplaying');
        updateNowPlaying();
        refreshLibrary();
    }

    function updateNowPlaying() {
        if (currentSong) {
            npTitle.textContent = currentSong.title;
            npArtist.textContent = '';
            npTrackNum.textContent = currentSongIndex >= 0 && library.length > 0 ? (currentSongIndex + 1) + ' of ' + library.length : '';
            $('#viewNowPlaying').classList.toggle('playing', isPlaying);
            applyArtwork(currentSong.filename, currentSong.title);
        } else {
            npTitle.textContent = 'No song playing'; npArtist.textContent = ''; npTrackNum.textContent = '';
            $('#viewNowPlaying').classList.remove('playing');
            clearArtwork();
        }
        updateMediaSession();
        pushNativeState();
    }

    function togglePlay() {
        if (!currentSong) { if (library.length) playSong(library[0].filename, library[0].title); return; }
        if (isPlaying) { audio.pause(); isPlaying = false; } else { audio.play().catch(() => { }); isPlaying = true; }
        updateNowPlaying(); refreshLibrary();
    }

    function nextTrack() {
        if (!library.length) return;
        if (isShuffle) currentSongIndex = Math.floor(Math.random() * library.length);
        else currentSongIndex = (currentSongIndex + 1) % library.length;
        playSong(library[currentSongIndex].filename, library[currentSongIndex].title);
    }

    async function submitImport() {
        const url = $('#importInput').value.trim();
        if (!url) return;
        $('#importStatus').textContent = 'Submitting...';
        showLoading('Processing Link...');
        try {
            const r = await fetch('/api/import', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ url }) });
            const d = await r.json();
            if (d.error) { $('#importStatus').textContent = d.error; showToast(d.error, 'error'); return; }
            const count = d.trackCount || '?';
            showToast(`Found ${count} tracks! Importing...`, 'success');
            $('#importInput').value = '';
            showView('batches');
        } catch { $('#importStatus').textContent = 'Submission failed'; showToast('Failed', 'error'); }
        finally { hideLoading(); }
    }

    async function refreshBatches() {
        try {
            const r = await fetch('/api/import/list');
            batches = await r.json();
            renderBatches();
            const activeBatchCount = batches.filter(b => b.state === 'DOWNLOADING' || b.state === 'AWAITING_USER').length;
            const totalActive = activeBatchCount + (activeDownloadTaskId ? 1 : 0);
            const badge = $('#batchBadge');
            if (badge) badge.textContent = totalActive || '';
        } catch { }
    }


    function renderBatches() {
        let html = '';

        // Show active single-song download if one exists
        if (activeDownloadTaskId) {
            html += `<div class="batch-item">
                <div class="batch-item-info">
                    <div class="batch-item-title">${escH(activeDownloadTitle || 'Downloading...')}</div>
                    <div class="batch-item-meta"><span>DOWNLOADING</span><span>Single song</span></div>
                </div>
            </div>`;
        }

        // Show batch imports
        if (batches.length) {
            html += batches.map((b, i) => {
                const pct = b.totalTracks > 0 ? (b.completedCount / b.totalTracks * 100).toFixed(0) : 0;
                const adjustedIdx = activeDownloadTaskId ? i + 1 : i;
                return `<div class="batch-item ${adjustedIdx === batchesIndex ? 'selected' : ''}" data-id="${b.id}">
                    <div class="batch-item-info">
                        <div class="batch-item-title">Batch ${b.id.substring(0, 8)}</div>
                        <div class="batch-item-meta"><span>${b.state}</span><span>${b.completedCount}/${b.totalTracks} songs</span></div>
                        <div class="batch-item-progress"><div class="batch-item-fill" style="width: ${pct}%"></div></div>
                    </div>
                    <span class="batch-item-arrow">›</span>
                </div>`;
            }).join('');
        }

        if (!html) {
            html = '<div class="library-empty">No active downloads.</div>';
        }

        $('#batchList').innerHTML = html;
        // Scroll selected batch into view
        const selBatch = $('#batchList .batch-item.selected');
        if (selBatch) selBatch.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }


    async function refreshBatchDetail() {
        if (!activeBatchId) return;
        try {
            const r = await fetch('/api/import/status/' + activeBatchId);
            const d = await r.json();
            batchTracks = d.tracks || [];
            $('#batchDetailTitle').textContent = `Import ${activeBatchId.substring(0, 8)}`;
            $('#batchDetailStats').textContent = `${d.batch.completedCount}/${d.batch.totalTracks} songs · ${d.batch.state}`;
            renderBatchTracks();
        } catch { }
    }

    function renderBatchTracks() {
        $('#batchTrackList').innerHTML = batchTracks.map((t, i) => {
            const sc = 'status-' + t.status.toLowerCase().replace(/_/g, '-');
            return `<div class="track-item ${i === batchTracksIndex ? 'selected' : ''}" data-id="${t.id}">
                <div class="track-thumb"></div>
                <div class="track-info"><div class="track-title">${escH(t.title)}</div><div class="track-status-pill ${sc}">${t.status}</div></div>
            </div>`;
        }).join('');
        // Scroll selected track into view
        const selTrack = $('#batchTrackList .track-item.selected');
        if (selTrack) selTrack.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }

    function batchesUp() { batchesIndex = Math.max(0, batchesIndex - 1); renderBatches(); }
    function batchesDown() { batchesIndex = Math.min(batches.length - 1, batchesIndex + 1); renderBatches(); }
    function batchesSelect() {
        const b = batches[batchesIndex];
        if (!b) return;
        activeBatchId = b.id;
        batchTracksIndex = 0;
        showView('batchDetail');
    }

    function tracksUp() { batchTracksIndex = Math.max(0, batchTracksIndex - 1); renderBatchTracks(); }
    function tracksDown() { batchTracksIndex = Math.min(batchTracks.length - 1, batchTracksIndex + 1); renderBatchTracks(); }
    async function tracksSelect() {
        const t = batchTracks[batchTracksIndex];
        if (!t) return;
        if (t.status === 'MATCHED_LOW_CONFIDENCE') {
            activeTrackId = t.id;
            $('#actionMenuHeader').textContent = t.title.substring(0, 15) + (t.title.length > 15 ? '...' : '');
            showView('actionMenu');
        }
    }

    $('#importSubmitBtn').addEventListener('click', submitImport);
    $('#importInput').addEventListener('keydown', e => { if (e.key === 'Enter') submitImport(); });

    function prevTrack() {
        if (audio.currentTime > 3) { audio.currentTime = 0; return; }
        if (!library.length) return;
        currentSongIndex = (currentSongIndex - 1 + library.length) % library.length;
        playSong(library[currentSongIndex].filename, library[currentSongIndex].title);
    }

    audio.addEventListener('timeupdate', () => {
        if (!audio.duration) return;
        const pct = (audio.currentTime / audio.duration) * 100;
        npProgressFill.style.width = pct + '%';
        if (npProgressKnob) npProgressKnob.style.left = pct + '%';
        npTimeElapsed.textContent = fmt(audio.currentTime);
        npTimeTotal.textContent = '-' + fmt(audio.duration - audio.currentTime);

        // Keep the lockscreen scrubber roughly in step without spamming the
        // bridge on every timeupdate.
        const now = Date.now();
        if (now - lastNativePush > 1000) { lastNativePush = now; pushNativeState(); }
    });

    audio.addEventListener('ended', () => { isPlaying = false; if (isRepeat) { audio.currentTime = 0; audio.play().catch(() => { }); isPlaying = true; updateNowPlaying(); } else nextTrack(); });
    audio.addEventListener('play', () => { isPlaying = true; updateNowPlaying(); });
    audio.addEventListener('pause', () => { isPlaying = false; updateNowPlaying(); });

    $('#npProgressBar')?.addEventListener('click', e => {
        if (!audio.duration) return;
        const r = e.currentTarget.getBoundingClientRect();
        audio.currentTime = ((e.clientX - r.left) / r.width) * audio.duration;
    });

    const btnHaptic = () => { if (navigator.vibrate) navigator.vibrate(12); };
    const selectHaptic = () => { if (navigator.vibrate) navigator.vibrate(15); };

    $('#btnMenu')?.addEventListener('click', () => { btnHaptic(); goBack(); });
    $('#btnPlay')?.addEventListener('click', () => { btnHaptic(); togglePlay(); });
    $('#btnNext')?.addEventListener('click', () => { btnHaptic(); nextTrack(); });
    $('#btnPrev')?.addEventListener('click', () => { btnHaptic(); prevTrack(); });
    $('#btnSelect')?.addEventListener('click', () => {
        selectHaptic();
        if (currentView === 'menu') menuSelect();
        else if (currentView === 'library') libSelect();
        else if (currentView === 'nowplaying') togglePlay();
        else if (currentView === 'import') submitImport();
        else if (currentView === 'batches') batchesSelect();
        else if (currentView === 'batchDetail') tracksSelect();
        else if (currentView === 'actionMenu') actionMenuSelect();
    });

    const wheel = $('#clickWheel');
    wheel?.addEventListener('wheel', e => { e.preventDefault(); handleScroll(e.deltaY > 0 ? 1 : -1); }, { passive: false });
    let lastAngle = null, accAngle = 0;
    wheel?.addEventListener('touchstart', e => { lastAngle = getAngle(e.touches[0].clientX, e.touches[0].clientY); accAngle = 0; });
    wheel?.addEventListener('touchmove', e => {
        e.preventDefault(); const a = getAngle(e.touches[0].clientX, e.touches[0].clientY);
        if (lastAngle !== null) { let d = a - lastAngle; if (d > 180) d -= 360; if (d < -180) d += 360; accAngle += d; if (Math.abs(accAngle) >= 20) { handleScroll(accAngle > 0 ? 1 : -1); accAngle = 0; } }
        lastAngle = a;
    }, { passive: false });
    wheel?.addEventListener('touchend', () => { lastAngle = null; });

    let audioCtx = null;
    function dialTick() {
        if (navigator.vibrate) navigator.vibrate(8);
        try {
            if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.type = 'sine'; osc.frequency.value = 3000;
            gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.03);
            osc.connect(gain); gain.connect(audioCtx.destination);
            osc.start(audioCtx.currentTime); osc.stop(audioCtx.currentTime + 0.03);
        } catch (_) { }
    }

    function getAngle(x, y) { const r = wheel.getBoundingClientRect(); return Math.atan2(y - r.top - r.height / 2, x - r.left - r.width / 2) * (180 / Math.PI); }
    function handleScroll(dir) {
        dialTick();
        if (currentView === 'menu') { dir > 0 ? menuDown() : menuUp(); }
        else if (currentView === 'library') { dir > 0 ? libDown() : libUp(); }
        else if (currentView === 'batches') { dir > 0 ? batchesDown() : batchesUp(); }
        else if (currentView === 'batchDetail') { dir > 0 ? tracksDown() : tracksUp(); }
        else if (currentView === 'nowplaying') {
            if (audio.duration) {
                audio.currentTime = Math.max(0, Math.min(audio.duration, audio.currentTime + dir * 5));
                showToast(fmt(audio.currentTime) + ' / ' + fmt(audio.duration));
            }
        }
    }

    document.addEventListener('keydown', e => {
        if (document.activeElement === searchInput && e.key !== 'Escape') return;
        switch (e.key) {
            case 'ArrowUp': e.preventDefault(); handleScroll(-1); break;
            case 'ArrowDown': e.preventDefault(); handleScroll(1); break;
            case 'ArrowLeft': e.preventDefault(); prevTrack(); break;
            case 'ArrowRight': e.preventDefault(); nextTrack(); break;
            case 'Enter': e.preventDefault(); if (currentView === 'menu') menuSelect(); else if (currentView === 'library') libSelect(); else if (currentView === 'batches') batchesSelect(); else if (currentView === 'batchDetail') tracksSelect(); break;
            case 'Escape': case 'Backspace': if (document.activeElement !== searchInput) { e.preventDefault(); goBack(); } break;
            case ' ': if (document.activeElement !== searchInput) { e.preventDefault(); togglePlay(); } break;
            case 'Delete': if (currentView === 'library' && library[libraryIndex]) { e.preventDefault(); deleteSong(library[libraryIndex].filename); } break;
            case 't': case 'T': if (document.activeElement !== searchInput) cycleTheme(); break;
        }
    });

    function fmt(s) { return Math.floor(s / 60) + ':' + String(Math.floor(s % 60)).padStart(2, '0'); }
    function escH(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
    function esc(s) { return s.replace(/"/g, '&quot;').replace(/'/g, '&#39;'); }

    applyTheme();
    showView('menu', false);
    refreshLibrary();
})();
