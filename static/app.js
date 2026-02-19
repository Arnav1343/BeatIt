(() => {
    'use strict';
    const $ = s => document.querySelector(s);
    const $$ = s => [...document.querySelectorAll(s)];

    const views = { menu: $('#viewMenu'), search: $('#viewSearch'), library: $('#viewLibrary'), nowplaying: $('#viewNowPlaying') };
    const audio = $('#audioPlayer');
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

    // ─── Theme System ───────────────────────────────────────────────
    const themes = [
        { id: 'default', name: 'Pink / Black', dark: true },
        { id: 'charcoal', name: 'Charcoal / Peach', dark: true },
        { id: 'neon', name: 'Neon Green / Jet Black', dark: true },
        { id: 'purple', name: 'Purple / Black', dark: true },
        { id: 'deeppurple', name: 'Deep Purple / Rose Gold', dark: true },
    ];
    let currentThemeIndex = 0;
    const savedTheme = localStorage.getItem('ipod-theme-id');
    if (savedTheme) {
        const idx = themes.findIndex(t => t.id === savedTheme);
        if (idx >= 0) currentThemeIndex = idx;
    }

    function applyTheme() {
        const t = themes[currentThemeIndex];
        if (t.id === 'default') {
            document.body.removeAttribute('data-theme');
        } else {
            document.body.setAttribute('data-theme', t.id);
        }
        localStorage.setItem('ipod-theme-id', t.id);
    }

    // Single theme cycle button
    const themeCycleBtn = $('#themeCycle');
    themeCycleBtn.addEventListener('click', () => {
        currentThemeIndex = (currentThemeIndex + 1) % themes.length;
        applyTheme();
        showToast(themes[currentThemeIndex].name);
    });

    function showView(name, pushHistory = true) {
        if (pushHistory && currentView !== name) viewHistory.push(currentView);
        currentView = name;
        Object.entries(views).forEach(([k, el]) => el.classList.toggle('active', k === name));
        if (name === 'library') refreshLibrary();
        if (name === 'search') {
            searchInput.focus();
            // Restore active download progress UI if download is still running
            if (activeDownloadTaskId) {
                downloadProgressWrap.classList.remove('hidden');
                downloadBtn.disabled = true;
                downloadStatus.textContent = 'Downloading...';
                if (activeDownloadTitle) {
                    resultTitle.textContent = activeDownloadTitle;
                    searchResult.classList.remove('hidden');
                }
            }
        }
        if (name === 'nowplaying') updateNowPlaying();
    }

    function goBack() { if (viewHistory.length) showView(viewHistory.pop(), false); }
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

    const menuItems = $$('#viewMenu .menu-item');
    function updateMenuSel() { menuItems.forEach((el, i) => el.classList.toggle('selected', i === menuIndex)); }
    function menuUp() { menuIndex = Math.max(0, menuIndex - 1); updateMenuSel(); }
    function menuDown() { menuIndex = Math.min(menuItems.length - 1, menuIndex + 1); updateMenuSel(); }
    function menuSelect() { const a = menuItems[menuIndex]?.dataset.action; if (a) showView(a); }

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
            const thumb = s.thumbnail ? `<img class="suggestion-thumb" src="${escapeAttr(s.thumbnail)}" alt="">` : `<div class="suggestion-thumb"></div>`;
            return `<div class="suggestion-item" data-idx="${i}">${thumb}<div class="suggestion-info"><div class="suggestion-title">${escapeHtml(s.title)}</div><div class="suggestion-artist">${escapeHtml(s.artist || '')}</div></div><span class="suggestion-dur">${formatDuration(s.duration)}</span></div>`;
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
        // Set as the search result directly
        lastSearchResult = {
            id: '', title: item.title, url: item.url,
            duration: item.duration, uploader: item.artist,
            thumbnail: item.thumbnail, artist: item.artist, album: item.album || '',
        };
        resultTitle.textContent = item.title;
        resultMeta.textContent = (item.artist || '') + '  ·  ' + formatDuration(item.duration);
        searchResult.classList.remove('hidden');
        searchStatus.textContent = '✓ Found';
        downloadBtn.disabled = false;
        updateQualitySizes(item.duration || 0);
    }

    searchInput.addEventListener('input', () => {
        const q = searchInput.value.trim();
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

    // Arrow keys in search input to navigate suggestions
    searchInput.addEventListener('keydown', (e) => {
        if (!suggestionsData.length) return;
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
            return;
        }
    });

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
            const dur = d.duration || 0;
            resultMeta.textContent = (d.uploader || d.artist || '') + '  ·  ' + formatDuration(dur);
            searchResult.classList.remove('hidden');
            searchStatus.textContent = '✓ Found'; downloadBtn.disabled = false;
            updateQualitySizes(d.duration || lastSearchResult.duration || 0);
        } catch {
            searchStatus.textContent = 'Search failed'; searchStatus.className = 'search-status error-text'; showToast('Failed', 'error');
        } finally { hideLoading(); }
    }

    // ─── Format + Quality Selector ──────────────────────────────────
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
        selectedQuality = selectedQuality; // keep previous if valid, else use default
        const validQs = opts.map(o => o.q);
        if (!validQs.includes(selectedQuality)) selectedQuality = defOpt.q;

        const dur = lastSearchResult?.duration || 0;
        const container = $('#qualityOptions');
        container.innerHTML = opts.map(o => {
            const sz = estimateSize(dur, o.q);
            const active = o.q === selectedQuality ? 'active' : '';
            return `<button class="quality-opt ${active}" data-quality="${o.q}">
                <span class="q-name">${o.label}</span>
                <span class="q-detail">${o.detail} · <span class="q-size">${sz}</span></span>
            </button>`;
        }).join('');

        $$('.quality-opt').forEach(btn => {
            btn.addEventListener('click', () => {
                $$('.quality-opt').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                selectedQuality = parseInt(btn.dataset.quality);
            });
        });
    }

    function updateQualitySizes(duration) {
        $$('.quality-opt').forEach(btn => {
            const kbps = parseInt(btn.dataset.quality);
            const sz = btn.querySelector('.q-size');
            if (sz) sz.textContent = estimateSize(duration, kbps);
        });
    }

    // Format toggle (MP3 / Opus)
    $$('.fmt-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            $$('.fmt-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedCodec = btn.dataset.codec;
            renderQualityOptions();
        });
    });

    // Initial render with MP3 defaults
    renderQualityOptions();

    // ─── Download state (persists across view switches) ───
    let activeDownloadTaskId = null;
    let activeDownloadTitle = null;

    async function doDownload() {
        if (!lastSearchResult) return;
        downloadBtn.disabled = true; downloadStatus.textContent = 'Starting...'; downloadStatus.className = 'download-status';
        downloadProgressWrap.classList.remove('hidden'); downloadProgressFill.style.width = '0%'; downloadProgressText.textContent = '0%';
        try {
            const payload = { url: lastSearchResult.url, title: lastSearchResult.title, quality: selectedQuality, codec: selectedCodec };
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
                // Always update internal state even if not on search view
                const onSearch = currentView === 'search';
                if (d.status === 'downloading') {
                    if (onSearch) { downloadProgressWrap.classList.remove('hidden'); downloadProgressFill.style.width = d.percent + '%'; downloadProgressText.textContent = d.percent + '%'; downloadStatus.textContent = 'Downloading...'; }
                }
                else if (d.status === 'converting') {
                    if (onSearch) { downloadProgressFill.style.width = '100%'; downloadProgressText.textContent = '100%'; downloadStatus.textContent = 'Converting...'; }
                }
                else if (d.status === 'done' && d.result) {
                    clearInterval(progressPollInterval); progressPollInterval = null;
                    activeDownloadTaskId = null;
                    if (onSearch) { downloadProgressFill.style.width = '100%'; downloadProgressText.textContent = '✓'; downloadStatus.textContent = '✓ Saved (' + d.result.size_human + ')'; downloadStatus.className = 'download-status success'; downloadBtn.disabled = false; }
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

    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && suggestIndex < 0) { e.preventDefault(); doSearch(); }
    });
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
        } else {
            npTitle.textContent = 'No song playing'; npArtist.textContent = ''; npTrackNum.textContent = '';
            $('#viewNowPlaying').classList.remove('playing');
        }
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
    });

    audio.addEventListener('ended', () => { isPlaying = false; if (isRepeat) { audio.currentTime = 0; audio.play().catch(() => { }); isPlaying = true; updateNowPlaying(); } else nextTrack(); });
    audio.addEventListener('play', () => { isPlaying = true; updateNowPlaying(); });
    audio.addEventListener('pause', () => { isPlaying = false; updateNowPlaying(); });

    $('#npProgressBar')?.addEventListener('click', e => {
        if (!audio.duration) return;
        const r = e.currentTarget.getBoundingClientRect();
        audio.currentTime = ((e.clientX - r.left) / r.width) * audio.duration;
    });

    $('#btnMenu')?.addEventListener('click', goBack);
    $('#btnPlay')?.addEventListener('click', togglePlay);
    $('#btnNext')?.addEventListener('click', nextTrack);
    $('#btnPrev')?.addEventListener('click', prevTrack);
    $('#btnSelect')?.addEventListener('click', () => { if (currentView === 'menu') menuSelect(); else if (currentView === 'library') libSelect(); else if (currentView === 'nowplaying') togglePlay(); });

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

    function getAngle(x, y) { const r = wheel.getBoundingClientRect(); return Math.atan2(y - r.top - r.height / 2, x - r.left - r.width / 2) * (180 / Math.PI); }
    function handleScroll(dir) {
        if (currentView === 'menu') { dir > 0 ? menuDown() : menuUp(); }
        else if (currentView === 'library') { dir > 0 ? libDown() : libUp(); }
        else if (currentView === 'nowplaying') {
            // Dial rotation seeks playback: clockwise (dir>0) = forward, counter-clockwise = backward
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
            case 'Enter': e.preventDefault(); if (currentView === 'menu') menuSelect(); else if (currentView === 'library') libSelect(); break;
            case 'Escape': case 'Backspace': if (document.activeElement !== searchInput) { e.preventDefault(); goBack(); } break;
            case ' ': if (document.activeElement !== searchInput) { e.preventDefault(); togglePlay(); } break;
            case 'Delete': if (currentView === 'library' && library[libraryIndex]) { e.preventDefault(); deleteSong(library[libraryIndex].filename); } break;
            case 't': case 'T': if (document.activeElement !== searchInput) { currentThemeIndex = (currentThemeIndex + 1) % themes.length; applyTheme(); showToast(themes[currentThemeIndex].name); } break;
        }
    });

    function fmt(s) { return Math.floor(s / 60) + ':' + String(Math.floor(s % 60)).padStart(2, '0'); }
    function escH(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
    function esc(s) { return s.replace(/"/g, '&quot;').replace(/'/g, '&#39;'); }

    applyTheme();
    showView('menu', false);
    refreshLibrary();
})();
