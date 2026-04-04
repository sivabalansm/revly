// ============================================================
// Detection Renderer — Canvas-based sponsor image overlay
// ============================================================
// Draws sponsor images at detected object bounding boxes.
// Called by overlay.js via window.renderDetections(detections).
// Uses canvas 2D for performance, EMA smoothing for stability.
// ============================================================

(function () {
  'use strict';

  // ── Config ──────────────────────────────────────────────
  const CANVAS_W = 1920;
  const CANVAS_H = 1080;
  const EMA_ALPHA = 0.3;           // weight for new detection
  const STALE_TIMEOUT = 2000;      // ms before removing stale tracks
  const GLOW_COLOR = 'rgba(99, 102, 241, 0.45)';  // indigo glow
  const GLOW_LINE_WIDTH = 3;
  const GLOW_BLUR = 8;

  // ── State ───────────────────────────────────────────────
  const tracks = new Map();        // trackId -> { bbox, sponsorImageUrl, lastSeen }
  const imageCache = new Map();    // url -> { img, loaded, failed }

  // ── Create canvas ───────────────────────────────────────
  const container = document.getElementById('detection-container');
  const canvas = document.createElement('canvas');
  canvas.width = CANVAS_W;
  canvas.height = CANVAS_H;
  canvas.style.width = CANVAS_W + 'px';
  canvas.style.height = CANVAS_H + 'px';
  canvas.style.position = 'absolute';
  canvas.style.top = '0';
  canvas.style.left = '0';
  canvas.style.pointerEvents = 'none';
  container.appendChild(canvas);

  const ctx = canvas.getContext('2d');

  // ── Image loader with cache ─────────────────────────────

  function getImage(url) {
    if (!url) return null;

    let entry = imageCache.get(url);
    if (entry) {
      return entry.loaded ? entry.img : null;
    }

    // Start loading
    const img = new Image();
    entry = { img: img, loaded: false, failed: false };
    imageCache.set(url, entry);

    img.onload = function () {
      entry.loaded = true;
    };
    img.onerror = function () {
      entry.failed = true;
      console.warn('[DetectionRenderer] Failed to load image:', url);
    };
    img.src = url;

    return null; // not ready yet
  }

  // ── EMA smoothing ───────────────────────────────────────

  function smoothBbox(existing, detected) {
    return {
      x: existing.x * (1 - EMA_ALPHA) + detected.x * EMA_ALPHA,
      y: existing.y * (1 - EMA_ALPHA) + detected.y * EMA_ALPHA,
      w: existing.w * (1 - EMA_ALPHA) + detected.w * EMA_ALPHA,
      h: existing.h * (1 - EMA_ALPHA) + detected.h * EMA_ALPHA,
    };
  }

  // ── Receive detections from overlay.js ──────────────────

  window.renderDetections = function (detections) {
    if (!Array.isArray(detections)) return;

    const now = Date.now();

    for (var i = 0; i < detections.length; i++) {
      var det = detections[i];
      if (!det || !det.bbox || !det.id) continue;

      var existing = tracks.get(det.id);
      var bbox;

      if (existing) {
        bbox = smoothBbox(existing.bbox, det.bbox);
      } else {
        // First sighting — no smoothing
        bbox = {
          x: det.bbox.x,
          y: det.bbox.y,
          w: det.bbox.w,
          h: det.bbox.h,
        };
      }

      tracks.set(det.id, {
        bbox: bbox,
        sponsorImageUrl: det.sponsorImageUrl || null,
        label: det.label || '',
        lastSeen: now,
      });

      // Pre-load sponsor image
      if (det.sponsorImageUrl) {
        getImage(det.sponsorImageUrl);
      }
    }
  };

  // ── Render loop (60 FPS) ────────────────────────────────

  function draw() {
    requestAnimationFrame(draw);

    var now = Date.now();

    // Clear canvas (transparent)
    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);

    // Remove stale tracks
    for (var [id, track] of tracks) {
      if (now - track.lastSeen > STALE_TIMEOUT) {
        tracks.delete(id);
      }
    }

    // Draw each tracked detection
    for (var [id, track] of tracks) {
      var b = track.bbox;
      var x = Math.round(b.x);
      var y = Math.round(b.y);
      var w = Math.round(b.w);
      var h = Math.round(b.h);

      if (w <= 0 || h <= 0) continue;

      // Draw sponsor image if available and loaded
      var img = track.sponsorImageUrl ? getImage(track.sponsorImageUrl) : null;

      if (img) {
        ctx.drawImage(img, x, y, w, h);
      }

      // Draw subtle glow border around the area
      ctx.save();
      ctx.strokeStyle = GLOW_COLOR;
      ctx.lineWidth = GLOW_LINE_WIDTH;
      ctx.shadowColor = GLOW_COLOR;
      ctx.shadowBlur = GLOW_BLUR;
      ctx.strokeRect(x, y, w, h);
      ctx.restore();
    }
  }

  // Start render loop
  requestAnimationFrame(draw);

  console.log('[DetectionRenderer] Initialized — canvas', CANVAS_W, 'x', CANVAS_H);

})();
