const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const multer = require('multer');
const { v4: uuidv4 } = require('uuid');
const path = require('path');
const fs = require('fs');
const os = require('os');

// ============================================================
// Setup
// ============================================================

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  pingInterval: 10000,
  pingTimeout: 5000,
  maxHttpBufferSize: 5 * 1024 * 1024, // 5MB max for detection frames
});

const PORT = process.env.PORT || 3000;
const uploadsDir = path.join(__dirname, 'public', 'uploads');
if (!fs.existsSync(uploadsDir)) fs.mkdirSync(uploadsDir, { recursive: true });

app.use(express.static(path.join(__dirname, 'public'), { etag: false, maxAge: 0 }));
app.use(express.json());

// Proxy the MJPEG stream from the Python AI camera (localhost:8765)
// so the browser can access it without CORS issues
app.get('/api/ai-stream', (req, res) => {
  const http = require('http');
  const proxyReq = http.get('http://127.0.0.1:8765/stream', (proxyRes) => {
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    proxyRes.pipe(res);
  });
  proxyReq.on('error', () => {
    res.status(502).json({ error: 'AI camera not running' });
  });
  req.on('close', () => proxyReq.destroy());
});

// Proxy the MJPEG stream from Moondream (localhost:5001)
app.get('/api/moondream-stream', (req, res) => {
  const http = require('http');
  const proxyReq = http.get('http://127.0.0.1:5001/video_feed', (proxyRes) => {
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    proxyRes.pipe(res);
  });
  proxyReq.on('error', () => {
    res.status(502).json({ error: 'Moondream not running' });
  });
  req.on('close', () => proxyReq.destroy());
});

app.get('/api/moondream-health', (_req, res) => {
  const http = require('http');
  const check = http.get('http://127.0.0.1:5001/', (r) => {
    res.json({ running: r.statusCode === 200 });
  });
  check.on('error', () => res.json({ running: false }));
});

// ── Moondream detection config ──────────────────────────
const MOONDREAM_CONFIG_PATH = path.join(__dirname, '..', 'Desktop', 'live_stream', 'config.json');
// Try multiple possible locations for the config
function getMoondreamConfigPath() {
  const paths = [
    path.resolve(__dirname, '..', 'config.json'),                          // bagelhacks2/config.json (if live_stream is here)
    path.resolve(os.homedir(), 'Desktop', 'live_stream', 'config.json'),   // ~/Desktop/live_stream/config.json
  ];
  for (const p of paths) {
    if (fs.existsSync(p)) return p;
  }
  return paths[1]; // default to Desktop location
}

app.get('/api/detection-config', (_req, res) => {
  try {
    const configPath = getMoondreamConfigPath();
    const config = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
    res.json(config);
  } catch (e) {
    res.json({ object_name: 'cup' });
  }
});

app.post('/api/detection-config', (req, res) => {
  try {
    const configPath = getMoondreamConfigPath();
    const liveStreamDir = path.dirname(configPath);
    const config = fs.existsSync(configPath) ? JSON.parse(fs.readFileSync(configPath, 'utf-8')) : {};
    if (req.body.object_name) {
      config.object_name = req.body.object_name.trim().slice(0, 100);
    }
    if (req.body.replacement_image) {
      config.replacement_image = req.body.replacement_image;
      // Copy the image to overlay.png in the live_stream folder
      const srcPath = path.join(__dirname, 'public', req.body.replacement_image);
      const dstPath = path.join(liveStreamDir, 'overlay.png');
      if (fs.existsSync(srcPath)) {
        fs.copyFileSync(srcPath, dstPath);
        console.log(`[Detection] Copied ${srcPath} -> ${dstPath}`);
      } else {
        console.warn(`[Detection] Source image not found: ${srcPath}`);
      }
    }
    fs.writeFileSync(configPath, JSON.stringify(config, null, 2));
    console.log(`[Detection] Config updated: ${JSON.stringify(config)}`);
    res.json({ ok: true, config });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get('/api/ai-health', (_req, res) => {
  const http = require('http');
  const check = http.get('http://127.0.0.1:8765/health', (r) => {
    res.json({ running: r.statusCode === 200 });
  });
  check.on('error', () => res.json({ running: false }));
});

// ============================================================
// Multer — image upload handling
// ============================================================

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, uploadsDir),
  filename: (_req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    cb(null, `${uuidv4()}${ext}`);
  },
});

const ALLOWED_MIME = new Set(['image/png', 'image/jpeg', 'image/gif', 'image/webp']);

const upload = multer({
  storage,
  limits: { fileSize: 2 * 1024 * 1024 }, // 2MB
  fileFilter: (_req, file, cb) => {
    if (ALLOWED_MIME.has(file.mimetype)) return cb(null, true);
    cb(new Error('Only PNG, JPG, GIF, WebP images are allowed'));
  },
});

// ============================================================
// In-memory state
// ============================================================

const overlays = new Map(); // token -> OverlayState
const adTimers = new Map(); // adId -> setTimeout handle (duration auto-hide)

function clearAdTimer(adId) {
  const t = adTimers.get(adId);
  if (t) { clearTimeout(t); adTimers.delete(adId); }
}

// Preset zone coordinates (1920x1080)
const PRESETS = {
  'corner-tl':     { x: 20,   y: 20,   w: 150,  h: 80  },
  'corner-tr':     { x: 1750, y: 20,   w: 150,  h: 80  },
  'corner-bl':     { x: 20,   y: 980,  w: 150,  h: 80  },
  'corner-br':     { x: 1750, y: 980,  w: 150,  h: 80  },
  'lower-third':   { x: 200,  y: 750,  w: 1520, h: 150 },
  'popup-right':   { x: 1400, y: 350,  w: 400,  h: 280 },
  'popup-left':    { x: 120,  y: 350,  w: 400,  h: 280 },
  'banner-top':    { x: 0,    y: 0,    w: 1920, h: 80  },
  'banner-bottom': { x: 0,    y: 1000, w: 1920, h: 80  },
  'ticker':        { x: 0,    y: 1040, w: 1920, h: 40  },
  'takeover':      { x: 360,  y: 190,  w: 1200, h: 700 },
};

// When format changes, snap to this preset
const FORMAT_DEFAULT_PRESET = {
  'corner-logo':  'corner-tl',
  'popup':        'popup-right',
  'lower-third':  'lower-third',
  'banner':       'banner-top',
  'ticker':       'ticker',
  'takeover':     'takeover',
};

const VALID_FORMATS = new Set(Object.keys(FORMAT_DEFAULT_PRESET));

function createOverlayState() {
  return {
    ads: [],
    rotation: {
      active: false,
      interval: 300000, // 5 min default
      pool: [],
      currentIndex: 0,
      timerId: null, // never serialized
    },
  };
}

// Strip timerId before sending to clients (it's a node internal handle)
function serializeState(state) {
  return {
    ads: state.ads,
    rotation: {
      active: state.rotation.active,
      interval: state.rotation.interval,
      pool: state.rotation.pool,
      currentIndex: state.rotation.currentIndex,
    },
  };
}

// ============================================================
// REST endpoints
// ============================================================

// Generate overlay token
app.get('/api/token', (_req, res) => {
  const token = uuidv4();
  overlays.set(token, createOverlayState());
  console.log(`[Token] Created: ${token}`);
  res.json({ token });
});

// Check if token exists
app.get('/api/token/:token/validate', (req, res) => {
  res.json({ valid: overlays.has(req.params.token) });
});

// Upload image
app.post('/api/upload', upload.single('image'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No image file provided' });
  const url = `/uploads/${req.file.filename}`;
  console.log(`[Upload] ${req.file.originalname} -> ${url}`);
  res.json({ url, originalName: req.file.originalname });
});

// List presets (useful for dashboard)
app.get('/api/presets', (_req, res) => {
  res.json({ presets: PRESETS, formatDefaults: FORMAT_DEFAULT_PRESET });
});

// ============================================================
// HTTP Trigger Endpoints (for Stream Deck, OBS hotkeys, etc.)
// ============================================================

// Helper: get overlay state by token or return 404
function getOverlayOrFail(req, res) {
  const token = req.params.token || req.query.token;
  if (!token) return res.status(400).json({ error: 'Token required' });
  const state = overlays.get(token);
  if (!state) return res.status(404).json({ error: 'Overlay not found' });
  return { state, token, room: `overlay:${token}` };
}

// GET /api/trigger/:token/show/:adId — show a specific ad
app.get('/api/trigger/:token/show/:adId', (req, res) => {
  const ctx = getOverlayOrFail(req, res);
  if (!ctx) return;
  const ad = ctx.state.ads.find((a) => a.id === req.params.adId);
  if (!ad) return res.status(404).json({ error: 'Ad not found' });
  ad.visible = true;
  io.to(ctx.room).emit('show_ad', { ad: { ...ad } });
  res.json({ ok: true, ad: ad.name, visible: true });
});

// GET /api/trigger/:token/hide/:adId — hide a specific ad
app.get('/api/trigger/:token/hide/:adId', (req, res) => {
  const ctx = getOverlayOrFail(req, res);
  if (!ctx) return;
  const ad = ctx.state.ads.find((a) => a.id === req.params.adId);
  if (!ad) return res.status(404).json({ error: 'Ad not found' });
  ad.visible = false;
  io.to(ctx.room).emit('hide_ad', { ad_id: ad.id });
  res.json({ ok: true, ad: ad.name, visible: false });
});

// GET /api/trigger/:token/toggle/:adId — toggle a specific ad
app.get('/api/trigger/:token/toggle/:adId', (req, res) => {
  const ctx = getOverlayOrFail(req, res);
  if (!ctx) return;
  const ad = ctx.state.ads.find((a) => a.id === req.params.adId);
  if (!ad) return res.status(404).json({ error: 'Ad not found' });
  ad.visible = !ad.visible;
  if (ad.visible) {
    io.to(ctx.room).emit('show_ad', { ad: { ...ad } });
  } else {
    io.to(ctx.room).emit('hide_ad', { ad_id: ad.id });
  }
  res.json({ ok: true, ad: ad.name, visible: ad.visible });
});

// GET /api/trigger/:token/hide-all — panic hide all
app.get('/api/trigger/:token/hide-all', (req, res) => {
  const ctx = getOverlayOrFail(req, res);
  if (!ctx) return;
  ctx.state.ads.forEach((ad) => { ad.visible = false; });
  stopRotation(ctx.state);
  io.to(ctx.room).emit('hide_all');
  console.log(`[Trigger] Hide all for ${ctx.token.slice(0, 8)}...`);
  res.json({ ok: true, action: 'hide_all' });
});

// GET /api/trigger/:token/show-by-index/:index — show ad by list position (1-based, for hotkeys)
app.get('/api/trigger/:token/show-by-index/:index', (req, res) => {
  const ctx = getOverlayOrFail(req, res);
  if (!ctx) return;
  const idx = parseInt(req.params.index) - 1;
  if (idx < 0 || idx >= ctx.state.ads.length) return res.status(404).json({ error: 'No ad at that index' });
  const ad = ctx.state.ads[idx];
  ad.visible = !ad.visible;
  if (ad.visible) {
    io.to(ctx.room).emit('show_ad', { ad: { ...ad } });
  } else {
    io.to(ctx.room).emit('hide_ad', { ad_id: ad.id });
  }
  res.json({ ok: true, ad: ad.name, index: idx + 1, visible: ad.visible });
});

// GET /api/trigger/:token/ads — list all ads (for Stream Deck setup)
app.get('/api/trigger/:token/ads', (req, res) => {
  const ctx = getOverlayOrFail(req, res);
  if (!ctx) return;
  res.json({
    ads: ctx.state.ads.map((a, i) => ({
      index: i + 1,
      id: a.id,
      name: a.name,
      format: a.format,
      visible: a.visible,
      triggerUrl: `/api/trigger/${ctx.token}/toggle/${a.id}`,
    })),
  });
});

// Multer error handler
app.use((err, _req, res, _next) => {
  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({ error: 'File too large (max 2MB)' });
    }
    return res.status(400).json({ error: err.message });
  }
  if (err.message) {
    return res.status(400).json({ error: err.message });
  }
  res.status(500).json({ error: 'Internal server error' });
});

// ============================================================
// Socket.IO — auth middleware
// ============================================================

io.use((socket, next) => {
  const token = socket.handshake.auth.token;
  if (!token || typeof token !== 'string') {
    return next(new Error('Authentication error: no token'));
  }
  // Auto-create state if token doesn't exist yet
  if (!overlays.has(token)) {
    overlays.set(token, createOverlayState());
  }
  socket.overlayToken = token;
  socket.role = socket.handshake.auth.role || 'overlay';
  next();
});

// ============================================================
// Socket.IO — rate limiting per socket
// ============================================================

function createRateLimiter(maxPerSecond) {
  let count = 0;
  let resetAt = Date.now() + 1000;
  return function check() {
    const now = Date.now();
    if (now > resetAt) {
      count = 0;
      resetAt = now + 1000;
    }
    count++;
    return count <= maxPerSecond;
  };
}

// ============================================================
// Socket.IO — connection handler
// ============================================================

io.on('connection', (socket) => {
  const token = socket.overlayToken;
  const room = `overlay:${token}`;
  socket.join(room);

  const limiter = createRateLimiter(30); // 30 commands/sec max

  console.log(`[Socket] ${socket.role} connected (token: ${token.slice(0, 8)}...)`);

  // Helper: get state or disconnect if missing
  function getState() {
    const state = overlays.get(token);
    if (!state) {
      socket.disconnect(true);
      return null;
    }
    return state;
  }

  // Helper: find ad by id
  function findAd(state, adId) {
    if (!adId || typeof adId !== 'string') return null;
    return state.ads.find((a) => a.id === adId) || null;
  }

  // Helper: broadcast to room
  function broadcast(event, data) {
    io.to(room).emit(event, data);
  }

  // Helper: rate-limit check
  function rateLimited() {
    if (!limiter()) {
      socket.emit('error_msg', { error: 'Rate limited — slow down' });
      return true;
    }
    return false;
  }

  // --- Send full state on connect ---
  const state = getState();
  if (state) {
    socket.emit('full_state', serializeState(state));
  }

  // --- Request full state (reconnect) ---
  socket.on('request_full_state', () => {
    const s = getState();
    if (s) socket.emit('full_state', serializeState(s));
  });

  // --- Create ad ---
  socket.on('create_ad', (data) => {
    if (rateLimited()) return;
    const s = getState();
    if (!s) return;
    if (!data || typeof data !== 'object') return;

    const format = VALID_FORMATS.has(data.format) ? data.format : 'corner-logo';
    const presetKey = FORMAT_DEFAULT_PRESET[format];
    const defaultPos = PRESETS[presetKey] || PRESETS['corner-tl'];

    // Validate position if provided
    let position = { ...defaultPos };
    if (data.position && typeof data.position === 'object') {
      position = {
        x: clampNum(data.position.x, 0, 1920),
        y: clampNum(data.position.y, 0, 1080),
        w: clampNum(data.position.w, 10, 1920),
        h: clampNum(data.position.h, 10, 1080),
      };
    }

    const ad = {
      id: uuidv4(),
      name: sanitizeString(data.name, 'Untitled Ad', 100),
      imageUrl: sanitizeString(data.imageUrl, '', 500),
      format,
      position,
      duration: clampNum(data.duration, 0, 300000), // max 5 min
      animation: sanitizeString(data.animation, 'fadeScaleIn', 50),
      visible: false,
    };

    // Cap at 50 ads per overlay to prevent memory bloat
    if (s.ads.length >= 50) {
      socket.emit('error_msg', { error: 'Maximum 50 ads per overlay' });
      return;
    }

    s.ads.push(ad);
    broadcast('ad_created', ad);
    console.log(`[Ad] Created "${ad.name}" (${ad.format}) for ${token.slice(0, 8)}...`);
  });

  // --- Update ad (position, duration, name) ---
  socket.on('update_ad', (data) => {
    if (rateLimited()) return;
    const s = getState();
    if (!s || !data) return;
    const ad = findAd(s, data.ad_id);
    if (!ad) return;

    let changed = false;

    if (data.position && typeof data.position === 'object') {
      ad.position = {
        x: clampNum(data.position.x, 0, 1920),
        y: clampNum(data.position.y, 0, 1080),
        w: clampNum(data.position.w, 10, 1920),
        h: clampNum(data.position.h, 10, 1080),
      };
      changed = true;
    }
    if (typeof data.duration === 'number') {
      ad.duration = clampNum(data.duration, 0, 300000);
      changed = true;
    }
    if (typeof data.name === 'string') {
      ad.name = sanitizeString(data.name, ad.name, 100);
      changed = true;
    }

    if (changed) {
      broadcast('ad_updated', {
        ad_id: ad.id,
        position: ad.position,
        duration: ad.duration,
        name: ad.name,
      });
    }
  });

  // --- Show ad ---
  socket.on('show_ad', (data) => {
    if (rateLimited()) return;
    const s = getState();
    if (!s || !data) return;
    const ad = findAd(s, data.ad_id);
    if (!ad) return;

    // Clear any existing duration timer for this ad
    clearAdTimer(ad.id);

    ad.visible = true;
    broadcast('show_ad', { ad: { ...ad } });

    // If ad has a duration, auto-hide after it expires
    let duration = ad.duration || 0;
    if (ad.format === 'takeover' && (duration === 0 || duration > 5000)) duration = 5000;
    if (duration > 0) {
      adTimers.set(ad.id, setTimeout(() => {
        adTimers.delete(ad.id);
        ad.visible = false;
        io.to(room).emit('hide_ad', { ad_id: ad.id });
        console.log(`[Duration] Auto-hid "${ad.name}" after ${duration}ms`);
      }, duration));
    }
  });

  // --- Hide ad ---
  socket.on('hide_ad', (data) => {
    if (rateLimited()) return;
    const s = getState();
    if (!s || !data) return;
    const ad = findAd(s, data.ad_id);
    if (!ad) return;

    clearAdTimer(ad.id);
    ad.visible = false;
    broadcast('hide_ad', { ad_id: ad.id });
  });

  // --- Delete ad ---
  socket.on('delete_ad', (data) => {
    if (rateLimited()) return;
    const s = getState();
    if (!s || !data) return;

    const adId = data.ad_id;
    clearAdTimer(adId);
    const idx = s.ads.findIndex((a) => a.id === adId);
    if (idx === -1) return;

    s.ads.splice(idx, 1);

    // Clean up from rotation pool
    const poolIdx = s.rotation.pool.indexOf(adId);
    if (poolIdx !== -1) {
      s.rotation.pool.splice(poolIdx, 1);
      // Adjust currentIndex if needed
      if (s.rotation.currentIndex >= s.rotation.pool.length) {
        s.rotation.currentIndex = 0;
      }
      // Stop rotation if pool is now empty
      if (s.rotation.pool.length === 0 && s.rotation.active) {
        stopRotation(s);
        broadcast('rotation_stopped');
      }
    }

    broadcast('ad_deleted', { ad_id: adId });
    console.log(`[Ad] Deleted ${adId.slice(0, 8)}... from ${token.slice(0, 8)}...`);
  });

  // --- Set format (with auto-snap to preset) ---
  socket.on('set_format', (data) => {
    if (rateLimited()) return;
    const s = getState();
    if (!s || !data) return;
    const ad = findAd(s, data.ad_id);
    if (!ad) return;

    if (!VALID_FORMATS.has(data.format)) return;

    ad.format = data.format;

    // Auto-snap to recommended preset
    const presetKey = FORMAT_DEFAULT_PRESET[data.format];
    if (presetKey && PRESETS[presetKey]) {
      ad.position = { ...PRESETS[presetKey] };
    }

    broadcast('format_changed', {
      ad_id: ad.id,
      format: ad.format,
      position: ad.position,
    });
  });

  // --- Snap to preset (explicit) ---
  socket.on('snap_to_preset', (data) => {
    if (rateLimited()) return;
    const s = getState();
    if (!s || !data) return;
    const ad = findAd(s, data.ad_id);
    if (!ad) return;

    const preset = PRESETS[data.preset];
    if (!preset) return;

    ad.position = { ...preset };
    broadcast('ad_updated', {
      ad_id: ad.id,
      position: ad.position,
      duration: ad.duration,
      name: ad.name,
    });
  });

  // --- Panic: hide all ---
  socket.on('panic_hide_all', () => {
    if (rateLimited()) return;
    const s = getState();
    if (!s) return;

    s.ads.forEach((ad) => { clearAdTimer(ad.id); ad.visible = false; });
    stopRotation(s);

    broadcast('hide_all');
    console.log(`[Panic] All ads hidden for ${token.slice(0, 8)}...`);
  });

  // --- Start rotation ---
  socket.on('start_rotation', (data) => {
    if (rateLimited()) return;
    const s = getState();
    if (!s || !data) return;

    // Validate pool — only include existing ad IDs
    const validPool = (data.ad_ids || []).filter((id) =>
      s.ads.some((a) => a.id === id)
    );
    if (validPool.length === 0) {
      socket.emit('error_msg', { error: 'Rotation pool is empty' });
      return;
    }

    // Stop any existing rotation first
    stopRotation(s);

    s.rotation.active = true;
    s.rotation.interval = clampNum(data.interval, 5000, 1800000); // 5s to 30min
    s.rotation.pool = validPool;
    s.rotation.currentIndex = 0;

    // Show first ad immediately
    const firstAd = s.ads.find((a) => a.id === validPool[0]);
    if (firstAd) {
      firstAd.visible = true;
      broadcast('show_ad', { ad: { ...firstAd } });
    }

    // Schedule the cycle
    s.rotation.timerId = setTimeout(
      () => cycleRotation(token, room),
      firstAd?.duration || s.rotation.interval
    );

    broadcast('rotation_started', {
      interval: s.rotation.interval,
      pool: s.rotation.pool,
    });

    console.log(`[Rotation] Started for ${token.slice(0, 8)}... (${validPool.length} ads, ${s.rotation.interval}ms)`);
  });

  // --- Stop rotation ---
  socket.on('stop_rotation', () => {
    if (rateLimited()) return;
    const s = getState();
    if (!s) return;

    // Hide current rotating ad before stopping
    const currentAdId = s.rotation.pool[s.rotation.currentIndex];
    if (currentAdId) {
      const currentAd = s.ads.find((a) => a.id === currentAdId);
      if (currentAd) {
        currentAd.visible = false;
        broadcast('hide_ad', { ad_id: currentAd.id });
      }
    }

    stopRotation(s);
    broadcast('rotation_stopped');
    console.log(`[Rotation] Stopped for ${token.slice(0, 8)}...`);
  });

  // --- Detection forwarding (Person B integration) ---
  socket.on('detection_update', (data) => {
    // Forward as-is to overlay clients in the room — no state storage
    // No rate limit here since detection runs at frame rate
    io.to(room).emit('detection_update', data);
  });

  // --- Disconnect cleanup ---
  socket.on('disconnect', (reason) => {
    console.log(`[Socket] ${socket.role} disconnected (${reason}) from ${token.slice(0, 8)}...`);

    // Check if any clients remain in this room
    const roomSockets = io.sockets.adapter.rooms.get(room);
    if (!roomSockets || roomSockets.size === 0) {
      // No one left — stop rotation timer to prevent leaks
      const s = overlays.get(token);
      if (s && s.rotation.timerId) {
        clearTimeout(s.rotation.timerId);
        s.rotation.timerId = null;
        console.log(`[Cleanup] Stopped rotation timer for empty room ${token.slice(0, 8)}...`);
      }
    }
  });
});

// ============================================================
// Rotation cycle function (runs via setTimeout, outside socket scope)
// ============================================================

function cycleRotation(token, room) {
  const s = overlays.get(token);
  if (!s || !s.rotation.active) return;

  // Hide current ad
  const currentAdId = s.rotation.pool[s.rotation.currentIndex];
  const currentAd = currentAdId ? s.ads.find((a) => a.id === currentAdId) : null;
  if (currentAd) {
    currentAd.visible = false;
    io.to(room).emit('hide_ad', { ad_id: currentAd.id });
  }

  // Advance index
  s.rotation.currentIndex = (s.rotation.currentIndex + 1) % s.rotation.pool.length;

  // Show next ad
  const nextAdId = s.rotation.pool[s.rotation.currentIndex];
  const nextAd = nextAdId ? s.ads.find((a) => a.id === nextAdId) : null;
  if (nextAd) {
    nextAd.visible = true;
    io.to(room).emit('show_ad', { ad: { ...nextAd } });
  }

  // Schedule next cycle
  const duration = nextAd?.duration || s.rotation.interval;
  s.rotation.timerId = setTimeout(() => cycleRotation(token, room), duration);
}

// ============================================================
// Helpers
// ============================================================

function stopRotation(state) {
  if (state.rotation.timerId) {
    clearTimeout(state.rotation.timerId);
    state.rotation.timerId = null;
  }
  state.rotation.active = false;
}

function clampNum(val, min, max) {
  if (typeof val !== 'number' || isNaN(val)) return min;
  return Math.max(min, Math.min(max, Math.round(val)));
}

function sanitizeString(val, fallback, maxLen) {
  if (typeof val !== 'string' || val.trim().length === 0) return fallback;
  return val.trim().slice(0, maxLen);
}

// ============================================================
// Graceful shutdown — clean up all timers
// ============================================================

function shutdown() {
  console.log('\n[Server] Shutting down...');

  // Clear all rotation timers
  for (const [token, state] of overlays) {
    if (state.rotation.timerId) {
      clearTimeout(state.rotation.timerId);
      state.rotation.timerId = null;
    }
  }
  // Clear all ad duration timers
  for (const [id, t] of adTimers) { clearTimeout(t); }
  adTimers.clear();

  // Close socket.io then http server
  io.close(() => {
    server.close(() => {
      console.log('[Server] Closed.');
      process.exit(0);
    });
  });

  // Force exit after 5 seconds if graceful fails
  setTimeout(() => {
    console.error('[Server] Forced exit after timeout');
    process.exit(1);
  }, 5000);
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

// ============================================================
// Start
// ============================================================

server.listen(PORT, () => {
  console.log(`\n  OBS Ad Overlay Server`);
  console.log(`  ─────────────────────`);
  console.log(`  Running on:  http://localhost:${PORT}`);
  console.log(`  Dashboard:   http://localhost:${PORT}/dashboard.html`);
  console.log(`  Overlay:     http://localhost:${PORT}/overlay.html?token=<TOKEN>`);
  console.log(`  Get token:   GET http://localhost:${PORT}/api/token\n`);
});
