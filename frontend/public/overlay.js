// ============================================================
// OBS Ad Overlay — Client
// ============================================================
// This runs inside the OBS browser source. It MUST:
// 1. Never go blank (global error handlers)
// 2. Survive disconnects (keep showing last state)
// 3. Survive OBS refresh (localStorage recovery)
// 4. Never leak listeners (register once, never inside connect)
// 5. Never leak memory (pool DOM nodes, clean up timers)
// ============================================================

(function () {
  'use strict';

  // ── Config ──────────────────────────────────────────────
  const params = new URLSearchParams(window.location.search);
  const TOKEN = params.get('token');
  const DEBUG = params.has('debug');
  const AI_MODE = params.has('ai');
  const MOONDREAM_MODE = params.has('moondream');
  const STATE_KEY = 'overlay_state_v1';
  const STATE_MAX_AGE = 60000; // 60s — don't restore stale state

  // ── State ───────────────────────────────────────────────
  let currentAds = new Map();       // adId -> ad object
  let isConnected = false;
  let lastUpdateTime = 0;
  let durationTimers = new Map();   // adId -> setTimeout id
  let isSourceVisible = true;

  // ── DOM refs ────────────────────────────────────────────
  const adContainer = document.getElementById('ad-container');
  const debugPanel = document.getElementById('debug-panel');
  const debugDot = document.getElementById('debug-dot');
  const debugConn = document.getElementById('debug-conn');
  const debugFps = document.getElementById('debug-fps');
  const debugAds = document.getElementById('debug-ads');
  const debugLast = document.getElementById('debug-last');
  const debugMem = document.getElementById('debug-mem');

  // ── Global Error Handlers (NEVER go blank) ──────────────
  window.onerror = function (msg, src, line, col, err) {
    console.error('[Overlay Error]', msg, 'at', src, ':', line);
    logRemote('error', msg + ' at ' + src + ':' + line);
    return true; // swallow — don't kill the page
  };

  window.addEventListener('unhandledrejection', function (e) {
    console.error('[Unhandled Promise]', e.reason);
    logRemote('error', 'Unhandled rejection: ' + String(e.reason));
    e.preventDefault();
  });

  // ── Bail if no token ────────────────────────────────────
  if (!TOKEN) {
    console.error('[Overlay] No token in URL. Add ?token=YOUR_TOKEN');
    return;
  }

  // ── Debug panel ─────────────────────────────────────────
  if (DEBUG) {
    debugPanel.classList.add('visible');
  }

  // ── Animation maps ──────────────────────────────────────

  function getEnterAnimation(ad) {
    switch (ad.format) {
      case 'corner-logo': return 'anim-fadeScaleIn';
      case 'popup':       return ad.position.x > 960 ? 'anim-slideInRight' : 'anim-slideInLeft';
      case 'lower-third': return 'anim-lowerThirdIn';
      case 'banner':      return ad.position.y < 540 ? 'anim-slideDown' : 'anim-slideUp';
      case 'ticker':      return null; // continuous CSS, no one-shot animation
      case 'takeover':    return 'anim-fadeIn';
      default:            return 'anim-fadeScaleIn';
    }
  }

  function getExitAnimation(ad) {
    switch (ad.format) {
      case 'corner-logo': return 'anim-fadeScaleOut';
      case 'popup':       return ad.position.x > 960 ? 'anim-slideOutRight' : 'anim-slideOutLeft';
      case 'lower-third': return 'anim-lowerThirdOut';
      case 'banner':      return ad.position.y < 540 ? 'anim-slideDownOut' : 'anim-slideUpOut';
      case 'ticker':      return 'anim-fadeOut';
      case 'takeover':    return 'anim-fadeOut';
      default:            return 'anim-fadeScaleOut';
    }
  }

  // ── Render an ad ────────────────────────────────────────

  function renderAd(ad, animate) {
    // Remove existing element for this ad if any
    removeAdElement(ad.id, false);

    const el = document.createElement('div');
    el.className = 'ad-slot ad-' + ad.format;
    el.setAttribute('data-ad-id', ad.id);
    el.style.left = ad.position.x + 'px';
    el.style.top = ad.position.y + 'px';
    el.style.width = ad.position.w + 'px';
    el.style.height = ad.position.h + 'px';

    // Ticker format: wrap image in scrolling track
    if (ad.format === 'ticker') {
      const track = document.createElement('div');
      track.className = 'ticker-track';
      // Duplicate images for seamless scroll
      for (let i = 0; i < 5; i++) {
        const img = createAdImage(ad.imageUrl);
        track.appendChild(img);
      }
      el.appendChild(track);
    } else {
      const img = createAdImage(ad.imageUrl);
      el.appendChild(img);
    }

    adContainer.appendChild(el);

    // Enter animation
    if (animate !== false) {
      const animClass = getEnterAnimation(ad);
      if (animClass) {
        el.classList.add(animClass);
        el.addEventListener('animationend', function onEnd() {
          el.removeEventListener('animationend', onEnd);
          el.classList.remove(animClass);
        }, { once: true });
      }
    }

    // Duration auto-hide
    clearDurationTimer(ad.id);
    let duration = ad.duration || 0;
    // Takeover: force max 5 seconds
    if (ad.format === 'takeover' && (duration === 0 || duration > 5000)) {
      duration = 5000;
    }
    if (duration > 0) {
      const timerId = setTimeout(function () {
        durationTimers.delete(ad.id);
        removeAdWithAnimation(ad.id);
        // Mark as not visible locally
        const localAd = currentAds.get(ad.id);
        if (localAd) localAd.visible = false;
      }, duration);
      durationTimers.set(ad.id, timerId);
    }

    // Store in local state
    currentAds.set(ad.id, { ...ad, visible: true });
    saveState();
    updateDebug();
  }

  function createAdImage(url) {
    const img = document.createElement('img');
    img.src = url;
    img.draggable = false;
    // Handle broken images gracefully
    img.onerror = function () {
      console.warn('[Overlay] Image failed to load:', url);
      img.style.display = 'none';
    };
    return img;
  }

  // ── Remove ad (with animation) ──────────────────────────

  function removeAdWithAnimation(adId) {
    const el = adContainer.querySelector('[data-ad-id="' + adId + '"]');
    if (!el) return;

    const ad = currentAds.get(adId);
    const animClass = ad ? getExitAnimation(ad) : 'anim-fadeOut';

    if (animClass) {
      el.classList.add(animClass);
      el.addEventListener('animationend', function onEnd() {
        el.removeEventListener('animationend', onEnd);
        safeRemove(el);
      }, { once: true });
      // Safety: remove after 1s even if animationend doesn't fire
      setTimeout(function () { safeRemove(el); }, 1000);
    } else {
      safeRemove(el);
    }
  }

  // Remove ad element immediately (no animation)
  function removeAdElement(adId, clearState) {
    const el = adContainer.querySelector('[data-ad-id="' + adId + '"]');
    if (el) safeRemove(el);
    clearDurationTimer(adId);
    if (clearState !== false) {
      const ad = currentAds.get(adId);
      if (ad) ad.visible = false;
    }
  }

  function safeRemove(el) {
    if (el && el.parentNode) {
      el.parentNode.removeChild(el);
    }
  }

  function clearDurationTimer(adId) {
    const timerId = durationTimers.get(adId);
    if (timerId) {
      clearTimeout(timerId);
      durationTimers.delete(adId);
    }
  }

  // ── Clear all ads (panic) ───────────────────────────────

  function clearAllAds() {
    // Clear all duration timers
    for (const [adId, timerId] of durationTimers) {
      clearTimeout(timerId);
    }
    durationTimers.clear();

    // Remove all DOM elements immediately (no animation for panic)
    while (adContainer.firstChild) {
      adContainer.removeChild(adContainer.firstChild);
    }

    // Mark all as not visible
    for (const [id, ad] of currentAds) {
      ad.visible = false;
    }

    saveState();
    updateDebug();
  }

  // ── Move/resize ad ──────────────────────────────────────

  function moveAd(adId, position) {
    const el = adContainer.querySelector('[data-ad-id="' + adId + '"]');
    if (el) {
      el.style.left = position.x + 'px';
      el.style.top = position.y + 'px';
      el.style.width = position.w + 'px';
      el.style.height = position.h + 'px';
    }
    const ad = currentAds.get(adId);
    if (ad) {
      ad.position = position;
      saveState();
    }
  }

  // ── Apply full state ────────────────────────────────────

  function applyFullState(state) {
    // Clear everything first
    while (adContainer.firstChild) {
      adContainer.removeChild(adContainer.firstChild);
    }
    for (const [adId, timerId] of durationTimers) {
      clearTimeout(timerId);
    }
    durationTimers.clear();
    currentAds.clear();

    // Store all ads (visible or not)
    if (state.ads && Array.isArray(state.ads)) {
      for (const ad of state.ads) {
        currentAds.set(ad.id, { ...ad });
        if (ad.visible) {
          renderAd(ad, false); // no animation on state sync
        }
      }
    }

    lastUpdateTime = Date.now();
    saveState();
    updateDebug();
  }

  // ── localStorage persistence ────────────────────────────

  function saveState() {
    try {
      const data = {
        ads: Array.from(currentAds.values()),
        savedAt: Date.now(),
      };
      localStorage.setItem(STATE_KEY, JSON.stringify(data));
    } catch (e) {
      // localStorage might be full or disabled — ignore
    }
  }

  function loadState() {
    try {
      const raw = localStorage.getItem(STATE_KEY);
      if (!raw) return null;
      const data = JSON.parse(raw);
      if (!data || !data.savedAt) return null;
      // Expire old state
      if (Date.now() - data.savedAt > STATE_MAX_AGE) {
        localStorage.removeItem(STATE_KEY);
        return null;
      }
      return data;
    } catch (e) {
      return null;
    }
  }

  // Recover from localStorage on page load (bridge until server state arrives)
  function recoverFromStorage() {
    const saved = loadState();
    if (!saved || !saved.ads) return;
    console.log('[Overlay] Recovering', saved.ads.length, 'ads from localStorage');
    for (const ad of saved.ads) {
      currentAds.set(ad.id, ad);
      if (ad.visible) {
        renderAd(ad, false);
      }
    }
    updateDebug();
  }

  // ── Socket.IO connection ────────────────────────────────

  var socket = io({
    auth: { token: TOKEN, role: 'overlay' },
    reconnection: true,
    reconnectionAttempts: Infinity,
    reconnectionDelay: 1000,
    reconnectionDelayMax: 30000,
    randomizationFactor: 0.5,
    transports: ['websocket'],
    upgrade: false,
  });

  // ── Socket event handlers ───────────────────────────────
  // CRITICAL: All data listeners registered HERE at top level.
  // NEVER inside the 'connect' handler (causes listener leaks).

  // Full state sync
  socket.on('full_state', function (state) {
    applyFullState(state);
    console.log('[Overlay] Full state received:', state.ads.length, 'ads');
  });

  // Ad created (add to local store, don't render until show)
  socket.on('ad_created', function (ad) {
    currentAds.set(ad.id, { ...ad });
    saveState();
    updateDebug();
  });

  // Show ad
  socket.on('show_ad', function (data) {
    if (!data || !data.ad) return;
    renderAd(data.ad, true);
    lastUpdateTime = Date.now();
  });

  // Hide ad
  socket.on('hide_ad', function (data) {
    if (!data || !data.ad_id) return;
    removeAdWithAnimation(data.ad_id);
    lastUpdateTime = Date.now();
    saveState();
    updateDebug();
  });

  // Hide all (panic)
  socket.on('hide_all', function () {
    clearAllAds();
    lastUpdateTime = Date.now();
    console.log('[Overlay] Panic: all ads hidden');
  });

  // Ad position/property updated
  socket.on('ad_updated', function (data) {
    if (!data || !data.ad_id) return;
    if (data.position) {
      moveAd(data.ad_id, data.position);
    }
    const ad = currentAds.get(data.ad_id);
    if (ad) {
      if (data.name !== undefined) ad.name = data.name;
      if (data.duration !== undefined) ad.duration = data.duration;
      saveState();
    }
    lastUpdateTime = Date.now();
  });

  // Format changed (re-render if visible)
  socket.on('format_changed', function (data) {
    if (!data || !data.ad_id) return;
    const ad = currentAds.get(data.ad_id);
    if (!ad) return;
    ad.format = data.format;
    ad.position = data.position;
    // Re-render if currently visible
    if (ad.visible) {
      renderAd(ad, true);
    }
    lastUpdateTime = Date.now();
    saveState();
  });

  // Ad deleted
  socket.on('ad_deleted', function (data) {
    if (!data || !data.ad_id) return;
    removeAdElement(data.ad_id, true);
    currentAds.delete(data.ad_id);
    saveState();
    updateDebug();
  });

  // Rotation events (informational for overlay — rendering handled by show/hide)
  socket.on('rotation_started', function () {
    console.log('[Overlay] Rotation started');
  });

  socket.on('rotation_stopped', function () {
    console.log('[Overlay] Rotation stopped');
  });

  // Detection forwarding (Person B integration)
  socket.on('detection_update', function (data) {
    if (typeof window.renderDetections === 'function') {
      window.renderDetections(data.detections || data);
    }
  });

  // Error messages from server
  socket.on('error_msg', function (data) {
    console.warn('[Server Error]', data.error);
  });

  // ── Connect / disconnect handlers ───────────────────────

  socket.on('connect', function () {
    isConnected = true;
    console.log('[Overlay] Connected to server');
    // Always request fresh state on connect/reconnect
    socket.emit('request_full_state');
    updateDebug();
  });

  socket.on('disconnect', function (reason) {
    isConnected = false;
    console.log('[Overlay] Disconnected:', reason);
    // DO NOT clear ads — keep showing last known state
    updateDebug();
  });

  socket.on('reconnect_attempt', function (attempt) {
    console.log('[Overlay] Reconnect attempt', attempt);
  });

  // ── Remote logging (sends to server for dashboard debug) ─

  function logRemote(level, msg) {
    if (socket && socket.connected) {
      socket.emit('overlay_log', { level: level, msg: msg, ts: Date.now() });
    }
  }

  // ── OBS Visibility API ──────────────────────────────────
  // Pause expensive work when source is hidden in OBS

  window.addEventListener('obsSourceVisibleChanged', function (e) {
    isSourceVisible = e.detail.visible;
    if (!isSourceVisible) {
      pauseAnimations();
      saveState();
    } else {
      resumeAnimations();
    }
  });

  window.addEventListener('obsSourceActiveChanged', function (e) {
    if (!e.detail.active) {
      // Source not in active scene — throttle
      pauseAnimations();
    } else {
      resumeAnimations();
    }
  });

  function pauseAnimations() {
    adContainer.style.animationPlayState = 'paused';
    var slots = adContainer.querySelectorAll('.ad-slot');
    for (var i = 0; i < slots.length; i++) {
      slots[i].style.animationPlayState = 'paused';
    }
  }

  function resumeAnimations() {
    adContainer.style.animationPlayState = 'running';
    var slots = adContainer.querySelectorAll('.ad-slot');
    for (var i = 0; i < slots.length; i++) {
      slots[i].style.animationPlayState = 'running';
    }
  }

  // ── Debug panel update ──────────────────────────────────

  var fpsFrames = [];
  var debugInterval = null;

  function updateDebug() {
    if (!DEBUG) return;

    debugDot.className = 'status-dot ' + (isConnected ? 'connected' : 'disconnected');
    debugConn.textContent = isConnected ? 'Connected' : 'Disconnected';

    var visibleCount = 0;
    for (var [id, ad] of currentAds) {
      if (ad.visible) visibleCount++;
    }
    debugAds.textContent = visibleCount + '/' + currentAds.size;

    if (lastUpdateTime > 0) {
      var secsAgo = Math.round((Date.now() - lastUpdateTime) / 1000);
      debugLast.textContent = secsAgo + 's ago';
    }

    if (performance.memory) {
      var mb = (performance.memory.usedJSHeapSize / (1024 * 1024)).toFixed(1);
      debugMem.textContent = mb + 'MB';
    }
  }

  function trackFps() {
    var now = performance.now();
    fpsFrames.push(now);
    // Keep last second only
    while (fpsFrames.length > 0 && fpsFrames[0] < now - 1000) {
      fpsFrames.shift();
    }
    if (DEBUG) {
      debugFps.textContent = fpsFrames.length;
    }
    requestAnimationFrame(trackFps);
  }

  if (DEBUG) {
    requestAnimationFrame(trackFps);
    debugInterval = setInterval(updateDebug, 1000);
  }

  // ── Periodic cleanup (for 24/7 streams) ─────────────────

  setInterval(function () {
    // Remove orphaned DOM nodes that shouldn't be there
    var els = adContainer.querySelectorAll('.ad-slot');
    for (var i = 0; i < els.length; i++) {
      var adId = els[i].getAttribute('data-ad-id');
      var ad = currentAds.get(adId);
      // Remove if ad was deleted or is marked not visible but element exists
      if (!ad || (!ad.visible && !els[i].classList.contains('anim-fadeOut') &&
          !els[i].classList.contains('anim-fadeScaleOut') &&
          !els[i].classList.contains('anim-slideOutRight') &&
          !els[i].classList.contains('anim-slideOutLeft') &&
          !els[i].classList.contains('anim-slideDownOut') &&
          !els[i].classList.contains('anim-slideUpOut') &&
          !els[i].classList.contains('anim-lowerThirdOut'))) {
        safeRemove(els[i]);
      }
    }
  }, 30000); // Every 30 seconds

  // ── AI Video Background ─────────────────────────────────

  if (AI_MODE || MOONDREAM_MODE) {
    var aiBg = document.getElementById('ai-video-bg');
    if (aiBg) {
      if (MOONDREAM_MODE) {
        aiBg.src = '/api/moondream-stream';
        aiBg.onerror = function () {
          aiBg.src = 'http://localhost:5001/video_feed';
        };
        console.log('[Overlay] Moondream video background enabled');
      } else {
        aiBg.src = '/api/ai-stream';
        aiBg.onerror = function () {
          aiBg.src = 'http://localhost:8765/stream';
        };
        console.log('[Overlay] YOLO video background enabled');
      }
      aiBg.classList.add('active');
    }
  }

  // ── Init ────────────────────────────────────────────────

  // Recover from localStorage first (instant, before server connects)
  recoverFromStorage();

  console.log('[Overlay] Initialized. Token:', TOKEN.slice(0, 8) + '...',
    'Debug:', DEBUG, 'AI:', AI_MODE);

})();
