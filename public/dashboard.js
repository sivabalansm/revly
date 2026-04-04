(function () {
  'use strict';

  // ============================================================
  // State
  // ============================================================

  const TOKEN_KEY = 'dashboard_token_v1';
  let token = localStorage.getItem(TOKEN_KEY) || null;
  let socket = null;
  let ads = new Map();           // adId -> ad object
  let selectedAdId = null;
  let rotation = { active: false, interval: 300000, pool: [], currentIndex: 0 };
  let presets = {};
  let canvasScale = 1;           // preview px → real px multiplier

  // Drag state
  let drag = { active: false, adId: null, offsetX: 0, offsetY: 0 };
  let resize = { active: false, adId: null, handle: '', startX: 0, startY: 0, startPos: null };

  // ============================================================
  // DOM Refs
  // ============================================================

  const $ = (s) => document.querySelector(s);
  const $$ = (s) => document.querySelectorAll(s);

  // Video preview
  let currentStream = null;

  const connBadge      = $('#conn-badge');
  const overlayUrlInput = $('#overlay-url');
  const copyUrlBtn     = $('#copy-url-btn');
  const panicBtn       = $('#panic-btn');
  const fileInput      = $('#file-input');
  const uploadZone     = $('#upload-zone');
  const uploadProgress = $('#upload-progress');
  const adList         = $('#ad-list');
  const adCount        = $('#ad-count');
  const previewCanvas  = $('#preview-canvas');
  const previewAds     = $('#preview-ads');
  const presetsBar     = $('#presets-bar');
  const adControls     = $('#ad-controls');
  const emptyState     = $('#empty-state');
  const toastContainer = $('#toast-container');

  // Controls
  const ctrlName     = $('#ctrl-name');
  const ctrlFormat   = $('#ctrl-format');
  const ctrlDuration = $('#ctrl-duration');
  const ctrlShow     = $('#ctrl-show');
  const ctrlHide     = $('#ctrl-hide');
  const ctrlDelete   = $('#ctrl-delete');
  const ctrlX = $('#ctrl-x'), ctrlY = $('#ctrl-y'), ctrlW = $('#ctrl-w'), ctrlH = $('#ctrl-h');
  const selectedAdName = $('#selected-ad-name');

  // Rotation
  const rotationToggle   = $('#rotation-toggle');
  const rotationInterval = $('#rotation-interval');
  const rotationLabel    = $('#rotation-interval-label');
  const rotationPool     = $('#rotation-pool');

  // ============================================================
  // Init
  // ============================================================

  async function init() {
    // Fetch presets
    try {
      const res = await fetch('/api/presets');
      const data = await res.json();
      presets = data.presets;
    } catch (e) {
      console.error('Failed to load presets:', e);
    }

    // Get or create token — check URL param first, then localStorage
    const urlToken = new URLSearchParams(window.location.search).get('token');
    if (urlToken) {
      token = urlToken;
      localStorage.setItem(TOKEN_KEY, token);
    }

    if (token) {
      // Validate existing token
      try {
        const res = await fetch(`/api/token/${token}/validate`);
        const data = await res.json();
        if (!data.valid) token = null;
      } catch (e) { token = null; }
    }

    if (!token) {
      try {
        const res = await fetch('/api/token');
        const data = await res.json();
        token = data.token;
        localStorage.setItem(TOKEN_KEY, token);
      } catch (e) {
        toast('Failed to get overlay token', 'error');
        return;
      }
    }

    // Set overlay URL
    const base = window.location.origin;
    overlayUrlInput.value = `${base}/overlay.html?token=${token}`;

    // Connect socket
    connectSocket();

    // Bind UI events
    bindEvents();

    // Calculate canvas scale
    recalcCanvasScale();
    window.addEventListener('resize', recalcCanvasScale);
  }

  // ============================================================
  // Socket.IO
  // ============================================================

  function connectSocket() {
    socket = io({
      auth: { token, role: 'dashboard' },
      reconnection: true,
      reconnectionAttempts: Infinity,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 10000,
      transports: ['websocket'],
      upgrade: false,
    });

    setConnStatus('connecting');

    // --- Connection lifecycle ---
    socket.on('connect', () => {
      setConnStatus('connected');
      socket.emit('request_full_state');
    });

    socket.on('disconnect', () => {
      setConnStatus('disconnected');
    });

    // --- State sync ---
    socket.on('full_state', (state) => {
      ads.clear();
      if (state.ads) {
        state.ads.forEach((ad) => ads.set(ad.id, ad));
      }
      if (state.rotation) {
        rotation = state.rotation;
      }
      renderAdList();
      renderPreviewAds();
      renderRotation();
      updateEmptyState();

      // Reselect if previously selected ad still exists
      if (selectedAdId && !ads.has(selectedAdId)) {
        selectedAdId = null;
        hideControls();
      } else if (selectedAdId) {
        showControls(selectedAdId);
      }
    });

    socket.on('ad_created', (ad) => {
      ads.set(ad.id, ad);
      renderAdList();
      renderPreviewAds();
      renderRotation();
      updateEmptyState();
      selectAd(ad.id);
      toast(`"${ad.name}" created`);
    });

    socket.on('show_ad', (data) => {
      if (!data.ad) return;
      ads.set(data.ad.id, data.ad);
      renderAdList();
      renderPreviewAds();
      if (selectedAdId === data.ad.id) showControls(data.ad.id);
    });

    socket.on('hide_ad', (data) => {
      const ad = ads.get(data.ad_id);
      if (ad) {
        ad.visible = false;
        renderAdList();
        renderPreviewAds();
        if (selectedAdId === data.ad_id) showControls(data.ad_id);
      }
    });

    socket.on('hide_all', () => {
      ads.forEach((ad) => { ad.visible = false; });
      renderAdList();
      renderPreviewAds();
      if (selectedAdId) showControls(selectedAdId);
      toast('All ads hidden', 'success');
    });

    socket.on('ad_updated', (data) => {
      const ad = ads.get(data.ad_id);
      if (!ad) return;
      if (data.position) ad.position = data.position;
      if (data.duration !== undefined) ad.duration = data.duration;
      if (data.name !== undefined) ad.name = data.name;
      renderAdList();
      renderPreviewAds();
      if (selectedAdId === data.ad_id) showControls(data.ad_id);
    });

    socket.on('format_changed', (data) => {
      const ad = ads.get(data.ad_id);
      if (!ad) return;
      ad.format = data.format;
      ad.position = data.position;
      renderAdList();
      renderPreviewAds();
      if (selectedAdId === data.ad_id) showControls(data.ad_id);
    });

    socket.on('ad_deleted', (data) => {
      const name = ads.get(data.ad_id)?.name || 'Ad';
      ads.delete(data.ad_id);
      if (selectedAdId === data.ad_id) {
        selectedAdId = null;
        hideControls();
      }
      renderAdList();
      renderPreviewAds();
      renderRotation();
      updateEmptyState();
      toast(`"${name}" deleted`);
    });

    socket.on('rotation_started', (data) => {
      rotation.active = true;
      rotation.interval = data.interval;
      rotation.pool = data.pool;
      rotationToggle.checked = true;
      renderRotation();
    });

    socket.on('rotation_stopped', () => {
      rotation.active = false;
      rotationToggle.checked = false;
      renderRotation();
    });

    socket.on('error_msg', (data) => {
      toast(data.error, 'error');
    });
  }

  function setConnStatus(status) {
    connBadge.className = 'badge ' + status;
    connBadge.textContent = status === 'connected' ? 'Live' :
                            status === 'disconnected' ? 'Offline' : 'Connecting...';
  }

  // ============================================================
  // UI Event Binding
  // ============================================================

  function bindEvents() {
    // Copy URL
    copyUrlBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(overlayUrlInput.value).then(() => {
        toast('Overlay URL copied!', 'success');
      });
    });

    // Panic button
    panicBtn.addEventListener('click', () => {
      socket.emit('panic_hide_all');
    });

    // Upload
    uploadZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileUpload);

    // Drag & drop on upload zone
    uploadZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadZone.classList.add('drag-over');
    });
    uploadZone.addEventListener('dragleave', () => {
      uploadZone.classList.remove('drag-over');
    });
    uploadZone.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadZone.classList.remove('drag-over');
      if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        handleFileUpload();
      }
    });

    // Preset buttons
    presetsBar.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-preset]');
      if (!btn || !selectedAdId) return;
      socket.emit('snap_to_preset', { ad_id: selectedAdId, preset: btn.dataset.preset });
    });

    // Controls — name change
    let nameDebounce = null;
    ctrlName.addEventListener('input', () => {
      clearTimeout(nameDebounce);
      nameDebounce = setTimeout(() => {
        if (!selectedAdId) return;
        socket.emit('update_ad', { ad_id: selectedAdId, name: ctrlName.value });
      }, 400);
    });

    // Controls — format change
    ctrlFormat.addEventListener('change', () => {
      if (!selectedAdId) return;
      socket.emit('set_format', { ad_id: selectedAdId, format: ctrlFormat.value });
    });

    // Controls — duration change
    ctrlDuration.addEventListener('change', () => {
      if (!selectedAdId) return;
      socket.emit('update_ad', {
        ad_id: selectedAdId,
        duration: (parseFloat(ctrlDuration.value) || 0) * 1000,
      });
    });

    // Controls — position inputs
    let posDebounce = null;
    [ctrlX, ctrlY, ctrlW, ctrlH].forEach((input) => {
      input.addEventListener('input', () => {
        clearTimeout(posDebounce);
        posDebounce = setTimeout(() => {
          if (!selectedAdId) return;
          socket.emit('update_ad', {
            ad_id: selectedAdId,
            position: {
              x: parseInt(ctrlX.value) || 0,
              y: parseInt(ctrlY.value) || 0,
              w: parseInt(ctrlW.value) || 100,
              h: parseInt(ctrlH.value) || 100,
            },
          });
        }, 300);
      });
    });

    // Controls — show/hide/delete
    ctrlShow.addEventListener('click', () => {
      if (selectedAdId) socket.emit('show_ad', { ad_id: selectedAdId });
    });

    ctrlHide.addEventListener('click', () => {
      if (selectedAdId) socket.emit('hide_ad', { ad_id: selectedAdId });
    });

    ctrlDelete.addEventListener('click', () => {
      if (selectedAdId) socket.emit('delete_ad', { ad_id: selectedAdId });
    });

    // Rotation toggle
    rotationToggle.addEventListener('change', () => {
      if (rotationToggle.checked) {
        const poolIds = getRotationPoolIds();
        if (poolIds.length === 0) {
          toast('Add ads to the rotation pool first', 'error');
          rotationToggle.checked = false;
          return;
        }
        socket.emit('start_rotation', {
          ad_ids: poolIds,
          interval: parseInt(rotationInterval.value) * 1000,
        });
      } else {
        socket.emit('stop_rotation');
      }
    });

    // Rotation interval slider
    rotationInterval.addEventListener('input', () => {
      rotationLabel.textContent = formatInterval(parseInt(rotationInterval.value));
    });

    // Preview canvas drag/resize
    previewCanvas.addEventListener('mousedown', onCanvasMouseDown);
    window.addEventListener('mousemove', onCanvasMouseMove);
    window.addEventListener('mouseup', onCanvasMouseUp);

    // Video source selector
    $('#video-source').addEventListener('change', onVideoSourceChange);

    // Rotation interval — also update on change (mouseup on slider)
    rotationInterval.addEventListener('change', () => {
      if (rotation.active) {
        const poolIds = getRotationPoolIds();
        if (poolIds.length > 0) {
          socket.emit('start_rotation', {
            ad_ids: poolIds,
            interval: parseInt(rotationInterval.value) * 1000,
          });
        }
      }
    });

    // Keyboard shortcuts
    window.addEventListener('keydown', onKeyDown);
  }

  // ============================================================
  // File Upload
  // ============================================================

  async function handleFileUpload() {
    const file = fileInput.files[0];
    if (!file) return;

    // Validate client-side
    if (file.size > 2 * 1024 * 1024) {
      toast('File too large (max 2MB)', 'error');
      fileInput.value = '';
      return;
    }

    const formData = new FormData();
    formData.append('image', file);

    uploadProgress.style.width = '30%';

    try {
      const res = await fetch('/api/upload', { method: 'POST', body: formData });
      uploadProgress.style.width = '70%';

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || 'Upload failed');
      }

      const data = await res.json();
      uploadProgress.style.width = '100%';

      // Create ad from uploaded image
      const name = data.originalName?.replace(/\.[^.]+$/, '') || 'Untitled';
      socket.emit('create_ad', {
        name,
        imageUrl: data.url,
        format: 'corner-logo',
        duration: 0,
      });
    } catch (e) {
      toast(e.message, 'error');
    } finally {
      setTimeout(() => { uploadProgress.style.width = '0'; }, 600);
      fileInput.value = '';
    }
  }

  // ============================================================
  // Render — Ad List (sidebar)
  // ============================================================

  function renderAdList() {
    adList.innerHTML = '';
    adCount.textContent = ads.size;

    ads.forEach((ad) => {
      const item = document.createElement('div');
      item.className = 'ad-item' + (ad.id === selectedAdId ? ' selected' : '');
      item.dataset.adId = ad.id;

      item.innerHTML = `
        <div class="ad-item-thumb">
          <img src="${escHtml(ad.imageUrl)}" alt="" onerror="this.style.display='none'">
        </div>
        <div class="ad-item-info">
          <div class="ad-item-name">${escHtml(ad.name)}</div>
          <div class="ad-item-meta">${ad.format}${ad.duration ? ' · ' + (ad.duration / 1000) + 's' : ''}</div>
        </div>
        <div class="ad-item-status ${ad.visible ? 'live' : 'off'}"></div>
      `;

      item.addEventListener('click', () => selectAd(ad.id));
      adList.appendChild(item);
    });
  }

  // ============================================================
  // Render — Preview Canvas
  // ============================================================

  function recalcCanvasScale() {
    const rect = previewCanvas.getBoundingClientRect();
    canvasScale = rect.width / 1920;
  }

  function renderPreviewAds() {
    recalcCanvasScale();
    previewAds.innerHTML = '';

    ads.forEach((ad) => {
      const el = document.createElement('div');
      el.className = 'preview-ad' + (ad.id === selectedAdId ? ' selected' : '');
      el.dataset.adId = ad.id;

      const s = canvasScale;
      el.style.left   = (ad.position.x * s) + 'px';
      el.style.top    = (ad.position.y * s) + 'px';
      el.style.width  = (ad.position.w * s) + 'px';
      el.style.height = (ad.position.h * s) + 'px';

      if (!ad.visible) {
        el.style.borderStyle = 'dashed';
        el.style.borderColor = 'var(--text-tertiary)';
        el.style.background = 'rgba(255,255,255,0.02)';
      }

      const label = document.createElement('div');
      label.className = 'preview-ad-label';
      label.textContent = ad.name;
      el.appendChild(label);

      const img = document.createElement('img');
      img.src = ad.imageUrl;
      img.onerror = function () { this.style.display = 'none'; };
      el.appendChild(img);

      // Resize handles (only on selected)
      if (ad.id === selectedAdId) {
        ['nw', 'ne', 'sw', 'se'].forEach((dir) => {
          const h = document.createElement('div');
          h.className = 'resize-handle ' + dir;
          h.dataset.handle = dir;
          el.appendChild(h);
        });
      }

      el.addEventListener('mousedown', (e) => {
        // Check if it's a resize handle
        if (e.target.classList.contains('resize-handle')) return;
        selectAd(ad.id);
      });

      previewAds.appendChild(el);
    });
  }

  // ============================================================
  // Canvas Drag & Resize
  // ============================================================

  function onCanvasMouseDown(e) {
    const resizeHandle = e.target.closest('.resize-handle');
    const previewAd = e.target.closest('.preview-ad');

    if (resizeHandle && previewAd) {
      // Start resize
      const adId = previewAd.dataset.adId;
      const ad = ads.get(adId);
      if (!ad) return;
      e.preventDefault();
      resize.active = true;
      resize.adId = adId;
      resize.handle = resizeHandle.dataset.handle;
      resize.startX = e.clientX;
      resize.startY = e.clientY;
      resize.startPos = { ...ad.position };
      previewAd.classList.add('dragging');
    } else if (previewAd) {
      // Start drag
      const adId = previewAd.dataset.adId;
      selectAd(adId);
      const ad = ads.get(adId);
      if (!ad) return;
      e.preventDefault();
      const rect = previewCanvas.getBoundingClientRect();
      drag.active = true;
      drag.adId = adId;
      drag.offsetX = e.clientX - rect.left - (ad.position.x * canvasScale);
      drag.offsetY = e.clientY - rect.top - (ad.position.y * canvasScale);
      previewAd.classList.add('dragging');
    }
  }

  function onCanvasMouseMove(e) {
    if (drag.active) {
      const rect = previewCanvas.getBoundingClientRect();
      const ad = ads.get(drag.adId);
      if (!ad) return;

      let newX = Math.round((e.clientX - rect.left - drag.offsetX) / canvasScale);
      let newY = Math.round((e.clientY - rect.top - drag.offsetY) / canvasScale);

      // Clamp
      newX = Math.max(0, Math.min(1920 - ad.position.w, newX));
      newY = Math.max(0, Math.min(1080 - ad.position.h, newY));

      ad.position.x = newX;
      ad.position.y = newY;

      // Update preview element directly (no full re-render for performance)
      const el = previewAds.querySelector(`[data-ad-id="${drag.adId}"]`);
      if (el) {
        el.style.left = (newX * canvasScale) + 'px';
        el.style.top = (newY * canvasScale) + 'px';
      }

      // Update position inputs
      if (selectedAdId === drag.adId) {
        ctrlX.value = newX;
        ctrlY.value = newY;
      }
    }

    if (resize.active) {
      const ad = ads.get(resize.adId);
      if (!ad) return;

      const dx = Math.round((e.clientX - resize.startX) / canvasScale);
      const dy = Math.round((e.clientY - resize.startY) / canvasScale);
      const sp = resize.startPos;
      let { x, y, w, h } = sp;

      switch (resize.handle) {
        case 'se': w = sp.w + dx; h = sp.h + dy; break;
        case 'sw': x = sp.x + dx; w = sp.w - dx; h = sp.h + dy; break;
        case 'ne': y = sp.y + dy; w = sp.w + dx; h = sp.h - dy; break;
        case 'nw': x = sp.x + dx; y = sp.y + dy; w = sp.w - dx; h = sp.h - dy; break;
      }

      // Enforce minimums
      w = Math.max(20, w);
      h = Math.max(20, h);
      x = Math.max(0, Math.min(1920 - w, x));
      y = Math.max(0, Math.min(1080 - h, y));

      ad.position = { x, y, w, h };

      const el = previewAds.querySelector(`[data-ad-id="${resize.adId}"]`);
      if (el) {
        el.style.left   = (x * canvasScale) + 'px';
        el.style.top    = (y * canvasScale) + 'px';
        el.style.width  = (w * canvasScale) + 'px';
        el.style.height = (h * canvasScale) + 'px';
      }

      if (selectedAdId === resize.adId) {
        ctrlX.value = x; ctrlY.value = y;
        ctrlW.value = w; ctrlH.value = h;
      }
    }
  }

  function onCanvasMouseUp() {
    if (drag.active) {
      drag.active = false;
      const ad = ads.get(drag.adId);
      if (ad) {
        socket.emit('update_ad', { ad_id: drag.adId, position: ad.position });
      }
      const el = previewAds.querySelector(`[data-ad-id="${drag.adId}"]`);
      if (el) el.classList.remove('dragging');
    }

    if (resize.active) {
      resize.active = false;
      const ad = ads.get(resize.adId);
      if (ad) {
        socket.emit('update_ad', { ad_id: resize.adId, position: ad.position });
      }
      const el = previewAds.querySelector(`[data-ad-id="${resize.adId}"]`);
      if (el) el.classList.remove('dragging');
    }
  }

  // ============================================================
  // Selection & Controls
  // ============================================================

  function selectAd(adId) {
    selectedAdId = adId;
    renderAdList();
    renderPreviewAds();
    showControls(adId);
    updatePresetButtons();
  }

  function showControls(adId) {
    const ad = ads.get(adId);
    if (!ad) { hideControls(); return; }

    adControls.classList.remove('hidden');
    emptyState.classList.add('hidden');

    selectedAdName.textContent = ad.name;
    ctrlName.value = ad.name;
    ctrlFormat.value = ad.format;
    ctrlDuration.value = ad.duration ? ad.duration / 1000 : 0;
    ctrlX.value = ad.position.x;
    ctrlY.value = ad.position.y;
    ctrlW.value = ad.position.w;
    ctrlH.value = ad.position.h;

    // Update show/hide button state
    ctrlShow.textContent = ad.visible ? 'Showing' : 'Show Now';
    ctrlShow.disabled = ad.visible;
    ctrlHide.style.display = ad.visible ? '' : 'none';
  }

  function hideControls() {
    adControls.classList.add('hidden');
    updateEmptyState();
  }

  function updatePresetButtons() {
    presetsBar.querySelectorAll('.btn-preset').forEach((btn) => {
      btn.disabled = !selectedAdId;
    });
  }

  function updateEmptyState() {
    if (ads.size === 0 && !selectedAdId) {
      emptyState.classList.remove('hidden');
    } else {
      emptyState.classList.add('hidden');
    }
  }

  // ============================================================
  // Rotation
  // ============================================================

  function renderRotation() {
    rotationToggle.checked = rotation.active;

    if (rotation.interval) {
      rotationInterval.value = Math.round(rotation.interval / 1000);
      rotationLabel.textContent = formatInterval(Math.round(rotation.interval / 1000));
    }

    // Render pool chips
    rotationPool.innerHTML = '';
    if (ads.size === 0) {
      rotationPool.innerHTML = '<span class="empty-hint">Add ads to create a rotation pool</span>';
      return;
    }

    ads.forEach((ad) => {
      const chip = document.createElement('span');
      chip.className = 'pool-chip' + (rotation.pool.includes(ad.id) ? ' in-pool' : '');
      chip.textContent = ad.name;
      chip.title = 'Click to toggle in rotation pool';
      chip.addEventListener('click', () => togglePoolAd(ad.id));
      rotationPool.appendChild(chip);
    });
  }

  function togglePoolAd(adId) {
    const idx = rotation.pool.indexOf(adId);
    if (idx === -1) {
      rotation.pool.push(adId);
    } else {
      rotation.pool.splice(idx, 1);
    }
    renderRotation();

    // If rotation is active, restart with new pool
    if (rotation.active) {
      const poolIds = getRotationPoolIds();
      if (poolIds.length === 0) {
        socket.emit('stop_rotation');
      } else {
        socket.emit('start_rotation', {
          ad_ids: poolIds,
          interval: parseInt(rotationInterval.value) * 1000,
        });
      }
    }
  }

  function getRotationPoolIds() {
    return rotation.pool.filter((id) => ads.has(id));
  }

  // ============================================================
  // Video Preview Source
  // ============================================================

  async function onVideoSourceChange() {
    const source = $('#video-source').value;
    const video = $('#preview-video');
    const overlayIframe = $('#preview-overlay');

    // Stop existing stream
    if (currentStream) {
      currentStream.getTracks().forEach((t) => t.stop());
      currentStream = null;
    }
    video.srcObject = null;
    video.classList.remove('active');
    overlayIframe.classList.remove('active');

    if (source === 'none') return;

    try {
      let stream;

      if (source === 'obs') {
        // OBS Virtual Camera — shows the full composited OBS output
        const devices = await navigator.mediaDevices.enumerateDevices();
        const obsDevice = devices.find(
          (d) => d.kind === 'videoinput' && d.label.toLowerCase().includes('obs')
        );
        if (obsDevice) {
          stream = await navigator.mediaDevices.getUserMedia({
            video: { deviceId: { exact: obsDevice.deviceId }, width: 1920, height: 1080 },
          });
        } else {
          // Fallback: let user pick — OBS Virtual Camera should appear in the list
          stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1920 }, height: { ideal: 1080 } },
          });
          toast('No OBS Virtual Camera found — using default camera. Enable Virtual Camera in OBS.', 'error');
        }
      } else if (source === 'webcam') {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1920 }, height: { ideal: 1080 } },
        });
      } else if (source === 'screen') {
        stream = await navigator.mediaDevices.getDisplayMedia({
          video: { width: { ideal: 1920 }, height: { ideal: 1080 } },
        });
        // User cancelled the screen picker
        if (!stream) return;
      }

      currentStream = stream;
      video.srcObject = stream;
      video.classList.add('active');

      // Show the overlay iframe on top of the video
      overlayIframe.src = overlayUrlInput.value;
      overlayIframe.classList.add('active');

      // Handle stream ending (user stops screen share)
      stream.getTracks().forEach((track) => {
        track.onended = () => {
          video.classList.remove('active');
          overlayIframe.classList.remove('active');
          $('#video-source').value = 'none';
          currentStream = null;
        };
      });

      toast('Video preview active', 'success');
    } catch (e) {
      console.error('Video source error:', e);
      if (e.name === 'NotAllowedError') {
        toast('Camera/screen permission denied', 'error');
      } else {
        toast('Failed to start video: ' + e.message, 'error');
      }
      $('#video-source').value = 'none';
    }
  }

  // ============================================================
  // Keyboard Shortcuts
  // ============================================================

  function onKeyDown(e) {
    // Don't trigger when typing in inputs
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;

    // Escape — hide all (panic)
    if (e.key === 'Escape') {
      e.preventDefault();
      socket.emit('panic_hide_all');
      toast('Escape: all ads hidden', 'success');
      return;
    }

    // Space — toggle selected ad
    if (e.key === ' ' && selectedAdId) {
      e.preventDefault();
      const ad = ads.get(selectedAdId);
      if (!ad) return;
      if (ad.visible) {
        socket.emit('hide_ad', { ad_id: selectedAdId });
      } else {
        socket.emit('show_ad', { ad_id: selectedAdId });
      }
      return;
    }

    // 1-9 — toggle ad by position in list
    if (e.key >= '1' && e.key <= '9') {
      const idx = parseInt(e.key) - 1;
      const adArray = Array.from(ads.values());
      if (idx >= adArray.length) return;
      const ad = adArray[idx];
      selectAd(ad.id);
      if (ad.visible) {
        socket.emit('hide_ad', { ad_id: ad.id });
      } else {
        socket.emit('show_ad', { ad_id: ad.id });
      }
      toast(`${e.key}: ${ad.name} ${ad.visible ? 'hidden' : 'shown'}`, 'success');
      return;
    }

    // R — toggle rotation
    if (e.key === 'r' || e.key === 'R') {
      rotationToggle.checked = !rotationToggle.checked;
      rotationToggle.dispatchEvent(new Event('change'));
      return;
    }

    // Arrow keys — move selected ad by 10px (or 1px with shift)
    if (selectedAdId && ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
      e.preventDefault();
      const ad = ads.get(selectedAdId);
      if (!ad) return;
      const step = e.shiftKey ? 1 : 10;
      switch (e.key) {
        case 'ArrowUp':    ad.position.y = Math.max(0, ad.position.y - step); break;
        case 'ArrowDown':  ad.position.y = Math.min(1080 - ad.position.h, ad.position.y + step); break;
        case 'ArrowLeft':  ad.position.x = Math.max(0, ad.position.x - step); break;
        case 'ArrowRight': ad.position.x = Math.min(1920 - ad.position.w, ad.position.x + step); break;
      }
      socket.emit('update_ad', { ad_id: selectedAdId, position: ad.position });
      return;
    }

    // Delete/Backspace — delete selected ad
    if ((e.key === 'Delete' || e.key === 'Backspace') && selectedAdId) {
      e.preventDefault();
      socket.emit('delete_ad', { ad_id: selectedAdId });
      return;
    }
  }

  // ============================================================
  // Helpers
  // ============================================================

  function formatInterval(seconds) {
    if (seconds < 60) return seconds + 's';
    if (seconds < 3600) return Math.round(seconds / 60) + 'm';
    return (seconds / 3600).toFixed(1) + 'h';
  }

  function escHtml(str) {
    const d = document.createElement('div');
    d.textContent = str || '';
    return d.innerHTML;
  }

  function toast(msg, type) {
    const el = document.createElement('div');
    el.className = 'toast' + (type ? ' ' + type : '');
    el.textContent = msg;
    toastContainer.appendChild(el);
    setTimeout(() => {
      el.classList.add('out');
      el.addEventListener('animationend', () => el.remove());
    }, 3000);
  }

  // ============================================================
  // Start
  // ============================================================

  init();

})();
