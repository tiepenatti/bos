const API_URL = import.meta.env.VITE_API_URL ?? ''

// ── DOM refs ──────────────────────────────────────────────────────────────────
const dropZone       = document.getElementById('drop-zone')
const fileInput      = document.getElementById('file-input')
const fileInfo       = document.getElementById('file-info')
const uploadBtn      = document.getElementById('upload-btn')
const uploadSection  = document.getElementById('upload-section')
const settingsSection= document.getElementById('settings-section')
const reprocessBtn   = document.getElementById('reprocess-btn')
const reprocessHint  = document.getElementById('reprocess-hint')
const progressSection= document.getElementById('progress-section')
const progressFill   = document.getElementById('progress-fill')
const progressLabel  = document.getElementById('progress-label')
const resultsSection = document.getElementById('results-section')
const summaryEl      = document.getElementById('summary')
const framesGrid     = document.getElementById('frames-grid')
const downloadZipBtn = document.getElementById('download-zip-btn')
const errorBanner    = document.getElementById('error-banner')

// Player refs
const playerEl       = document.getElementById('player')
const playerVideo    = document.getElementById('player-video')
const playerPrev     = document.getElementById('player-prev')
const playerNext     = document.getElementById('player-next')
const playerRateSel  = document.getElementById('player-rate-select')
const playerCounter  = document.getElementById('player-counter')

let selectedFile = null
let currentJobId = null

// ── Parameter controls ──────────────────────────────────────────────────────
// Maps a param id -> how to read it from its <input>/<select>.
const PARAM_IDS = [
  'method', 'detector', 'max_features', 'match_ratio',
  'flow_max_corners', 'flow_quality', 'flow_win_size',
  'ransac_thresh', 'gain', 'blur_ksize', 'threshold', 'morph_ksize',
  'overlay', 'colormap', 'normalize', 'num_pairs', 'frame_step', 'max_width',
]

// Live-update the <output> next to each range slider.
for (const id of PARAM_IDS) {
  const el = document.getElementById(`p-${id}`)
  const out = document.getElementById(`o-${id}`)
  if (!el) continue
  if (out) {
    const sync = () => {
      out.textContent = id === 'overlay'
        ? `${Math.round(Number(el.value) * 100)}%`
        : el.value
    }
    el.addEventListener('input', sync)
    sync()
  }
}

// Show only the controls relevant to the selected motion-estimation method.
const methodSelect = document.getElementById('p-method')
function syncMethodGroups() {
  const method = methodSelect.value
  document.querySelectorAll('.method-group').forEach((g) => {
    g.classList.toggle('hidden', g.dataset.method !== method)
  })
}
methodSelect.addEventListener('change', syncMethodGroups)
syncMethodGroups()

// ── Custom colormap dropdown ──────────────────────────────────────────────────
// A non-native dropdown so each option can preview its palette as a gradient.
// The chosen value is mirrored into the hidden #p-colormap input that
// collectParams() reads, so the rest of the code is unchanged.
const COLORMAPS = [
  { value: 'inferno', label: 'Inferno' },
  { value: 'magma',   label: 'Magma' },
  { value: 'turbo',   label: 'Turbo' },
  { value: 'jet',     label: 'Jet' },
  { value: 'viridis', label: 'Viridis' },
  { value: 'hot',     label: 'Hot' },
  { value: 'gray',    label: 'Grayscale' },
]

function initColormapSelect() {
  const root = document.getElementById('cmap-select')
  const hidden = document.getElementById('p-colormap')
  if (!root || !hidden) return

  root.innerHTML = `
    <button type="button" class="cmap-trigger" aria-haspopup="listbox" aria-expanded="false">
      <span class="cmap-swatch" data-cmap=""></span>
      <span class="cmap-name"></span>
      <span class="cmap-caret" aria-hidden="true">▾</span>
    </button>
    <ul class="cmap-menu" role="listbox"></ul>
  `
  const trigger = root.querySelector('.cmap-trigger')
  const menu = root.querySelector('.cmap-menu')
  const swatch = root.querySelector('.cmap-swatch')
  const nameEl = root.querySelector('.cmap-name')

  menu.innerHTML = COLORMAPS.map(
    (c) => `
    <li class="cmap-option" role="option" data-value="${c.value}" tabindex="-1">
      <span class="cmap-swatch" data-cmap="${c.value}"></span>
      <span class="cmap-name">${c.label}</span>
    </li>`
  ).join('')

  const setValue = (value) => {
    const cm = COLORMAPS.find((c) => c.value === value) ?? COLORMAPS[0]
    hidden.value = cm.value
    swatch.dataset.cmap = cm.value
    nameEl.textContent = cm.label
    menu.querySelectorAll('.cmap-option').forEach((o) => {
      o.setAttribute('aria-selected', String(o.dataset.value === cm.value))
    })
  }

  const closeMenu = () => {
    root.classList.remove('open')
    trigger.setAttribute('aria-expanded', 'false')
  }
  const openMenu = () => {
    root.classList.add('open')
    trigger.setAttribute('aria-expanded', 'true')
  }

  trigger.addEventListener('click', (e) => {
    e.stopPropagation()
    root.classList.contains('open') ? closeMenu() : openMenu()
  })
  menu.querySelectorAll('.cmap-option').forEach((opt) => {
    opt.addEventListener('click', () => {
      setValue(opt.dataset.value)
      closeMenu()
    })
  })
  document.addEventListener('click', (e) => {
    if (!root.contains(e.target)) closeMenu()
  })
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeMenu()
  })

  setValue(hidden.value || 'inferno')
}
initColormapSelect()

function collectParams() {
  const val = (id) => document.getElementById(`p-${id}`)
  return {
    method: val('method').value,
    detector: val('detector').value,
    max_features: Number(val('max_features').value),
    match_ratio: Number(val('match_ratio').value),
    flow_max_corners: Number(val('flow_max_corners').value),
    flow_quality: Number(val('flow_quality').value),
    flow_win_size: Number(val('flow_win_size').value),
    ransac_thresh: Number(val('ransac_thresh').value),
    gain: Number(val('gain').value),
    blur_ksize: Number(val('blur_ksize').value),
    threshold: Number(val('threshold').value),
    morph_ksize: Number(val('morph_ksize').value),
    overlay: Number(val('overlay').value),
    colormap: val('colormap').value,
    normalize: val('normalize').checked,
    num_pairs: Number(val('num_pairs').value),
    frame_step: Number(val('frame_step').value),
    max_width: Number(val('max_width').value),
  }
}

// ── File selection ────────────────────────────────────────────────────────────
dropZone.addEventListener('click', () => fileInput.click())
fileInput.addEventListener('change', () => setFile(fileInput.files[0]))

dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over') })
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'))
dropZone.addEventListener('drop', (e) => {
  e.preventDefault()
  dropZone.classList.remove('drag-over')
  const file = e.dataTransfer.files[0]
  if (file) setFile(file)
})

function setFile(file) {
  if (!file || !file.type.startsWith('video/')) {
    showError('Please select a valid video file.')
    return
  }
  selectedFile = file
  fileInfo.textContent = `${file.name} — ${formatBytes(file.size)}`
  fileInfo.classList.remove('hidden')
  uploadBtn.disabled = false
  settingsSection.classList.remove('hidden')
  clearError()
}

// ── Upload & process ──────────────────────────────────────────────────────────
uploadBtn.addEventListener('click', async () => {
  if (!selectedFile) return
  clearError()
  setBusy(true)

  showProgress('Uploading…', 10)
  progressSection.classList.remove('hidden')
  resultsSection.classList.add('hidden')

  try {
    // Upload the source, then process it.
    const formData = new FormData()
    formData.append('file', selectedFile)

    const uploadRes = await fetch(`${API_URL}/api/upload`, {
      method: 'POST',
      body: formData,
    })
    if (!uploadRes.ok) throw new Error(await uploadRes.text())
    const { job_id } = await uploadRes.json()
    currentJobId = job_id

    await runProcessing(job_id)
  } catch (err) {
    onProcessError(err)
  }
})

// ── Re-process the same source with the current settings ───────────────────────
reprocessBtn.addEventListener('click', async () => {
  if (!currentJobId) return
  clearError()
  setBusy(true)
  resultsSection.classList.add('hidden')
  progressSection.classList.remove('hidden')
  try {
    await runProcessing(currentJobId)
  } catch (err) {
    onProcessError(err)
  }
})

// Shared pipeline: (re)process an existing job, poll progress, render results.
// Re-processing replaces the previously generated video/frames server-side.
async function runProcessing(jobId) {
  showProgress('Starting…', 0)
  const processRes = await fetch(`${API_URL}/api/process/${jobId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(collectParams()),
  })
  if (!processRes.ok) throw new Error(await processRes.text())

  await pollUntilDone(jobId)

  const resultsRes = await fetch(`${API_URL}/api/results/${jobId}`)
  if (!resultsRes.ok) throw new Error(await resultsRes.text())
  const results = await resultsRes.json()

  showProgress('Done!', 100)
  await sleep(300)

  renderResults(results)
  // Now that a result exists, expose the re-process action.
  reprocessBtn.classList.remove('hidden')
  reprocessHint.classList.remove('hidden')
  setBusy(false)
}

function onProcessError(err) {
  progressSection.classList.add('hidden')
  uploadSection.classList.remove('hidden')
  setBusy(false)
  showError(`Error: ${err.message}`)
}

// Disable the action buttons while a job is in flight.
function setBusy(busy) {
  uploadBtn.disabled = busy || !selectedFile
  reprocessBtn.disabled = busy
}

// ── Status polling ────────────────────────────────────────────────────────────
async function pollUntilDone(jobId) {
  while (true) {
    let status
    try {
      const res = await fetch(`${API_URL}/api/status/${jobId}`)
      if (!res.ok) throw new Error(await res.text())
      status = await res.json()
    } catch {
      // Transient network / cold-start hiccup — retry shortly.
      await sleep(800)
      continue
    }

    if (status.state === 'error') {
      throw new Error(status.error || 'Processing failed')
    }
    if (status.state === 'done') {
      showProgress('Finalizing…', 100)
      return
    }

    const pct = Math.max(0, Math.min(100, status.percent ?? 0))
    if (status.state === 'queued') {
      showProgress('Queued…', 2)
    } else if (status.total) {
      showProgress(`Processing frame ${status.processed ?? 0} / ${status.total} · ${pct}%`, pct)
    } else {
      showProgress(`Processing… ${status.processed ?? 0} frames`, pct)
    }
    await sleep(500)
  }
}

// ── Results rendering ─────────────────────────────────────────────────────────
function renderResults(results) {
  progressSection.classList.add('hidden')
  resultsSection.classList.remove('hidden')

  // Summary stats
  summaryEl.innerHTML = [
    stat(results.frame_count, 'Frames'),
    stat(results.fps?.toFixed(2), 'FPS'),
    stat(`${results.width}×${results.height}`, 'Resolution'),
    stat(formatDuration(results.duration_seconds), 'Duration'),
    stat(results.extracted_frames?.length ?? 0, 'Extracted'),
  ].join('')

  // Download ZIP
  downloadZipBtn.onclick = () => {
    window.open(`${API_URL}/api/download/${currentJobId}`, '_blank')
  }

  // Full processed-video player
  initPlayer(results)

  // Frames
  framesGrid.innerHTML = ''
  for (const frame of results.extracted_frames ?? []) {
    const s = frame.stats ?? {}
    const alignBadge = s.aligned === false
      ? `<span class="badge warn" title="No reliable homography — showing raw diff">unaligned</span>`
      : `<span class="badge ok" title="${s.inliers ?? '?'} RANSAC inliers">aligned</span>`
    const card = document.createElement('div')
    card.className = 'frame-card'
    card.innerHTML = `
      <img src="${API_URL}/api/frames/${currentJobId}/${frame.index}" alt="Frame ${frame.index}" loading="lazy" />
      <div class="frame-label">
        <span>${frame.prev_index ?? '?'}→${frame.index} · ${formatTimestamp(frame.timestamp_ms)}</span>
        <a href="${API_URL}/api/frames/${currentJobId}/${frame.index}" download="schlieren_${frame.index}.jpg">↓</a>
      </div>
      <div class="frame-meta">${alignBadge}<span>${s.inliers ?? 0} inliers</span></div>
    `
    framesGrid.appendChild(card)
  }
}

// ── Processed-video player ────────────────────────────────────────────────────
let playerFps = 24

function initPlayer(results) {
  playerFps = results.output_fps || results.fps || 24

  if (!results.has_video) {
    playerEl.classList.add('hidden')
    return
  }
  playerEl.classList.remove('hidden')

  // Cache-bust so a re-processed job doesn't show the previous render.
  playerVideo.src = `${API_URL}/api/video/${currentJobId}?t=${Date.now()}`
  playerVideo.playbackRate = Number(playerRateSel.value) || 1
  playerVideo.load()
  updatePlayerCounter(results)
}

function updatePlayerCounter(results) {
  const total = results.processed_frames ?? 0
  const dur = total > 0 && playerFps ? (total / playerFps).toFixed(1) : '?'
  const setText = () => {
    const cur = Math.round((playerVideo.currentTime || 0) * playerFps)
    playerCounter.textContent = `frame ${Math.min(cur + 1, total)} / ${total} · ${dur}s`
  }
  playerVideo.ontimeupdate = setText
  setText()
}

function stepVideoFrame(delta) {
  playerVideo.pause()
  const dt = 1 / (playerFps || 24)
  const t = (playerVideo.currentTime || 0) + delta * dt
  playerVideo.currentTime = Math.max(0, Math.min(playerVideo.duration || t, t))
}

playerPrev.addEventListener('click', () => stepVideoFrame(-1))
playerNext.addEventListener('click', () => stepVideoFrame(1))
playerRateSel.addEventListener('change', () => {
  playerVideo.playbackRate = Number(playerRateSel.value) || 1
})

// ── Helpers ───────────────────────────────────────────────────────────────────
function showProgress(label, pct) {
  progressFill.style.width = `${pct}%`
  progressLabel.textContent = label
}

function showError(msg) {
  errorBanner.textContent = msg
  errorBanner.classList.remove('hidden')
}

function clearError() {
  errorBanner.classList.add('hidden')
  errorBanner.textContent = ''
}

function stat(value, label) {
  return `<div class="stat-card"><div class="stat-value">${value ?? '—'}</div><div class="stat-label">${label}</div></div>`
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`
}

function formatDuration(seconds) {
  if (seconds == null) return '—'
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return m > 0 ? `${m}m ${s}s` : `${s}s`
}

function formatTimestamp(ms) {
  if (ms == null) return ''
  const s = (ms / 1000).toFixed(2)
  return `${s}s`
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)) }
