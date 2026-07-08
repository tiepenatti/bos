const API_URL = import.meta.env.VITE_API_URL ?? ''

// ── DOM refs ──────────────────────────────────────────────────────────────────
const dropZone       = document.getElementById('drop-zone')
const fileInput      = document.getElementById('file-input')
const fileInfo       = document.getElementById('file-info')
const uploadBtn      = document.getElementById('upload-btn')
const uploadSection  = document.getElementById('upload-section')
const progressSection= document.getElementById('progress-section')
const progressFill   = document.getElementById('progress-fill')
const progressLabel  = document.getElementById('progress-label')
const resultsSection = document.getElementById('results-section')
const summaryEl      = document.getElementById('summary')
const framesGrid     = document.getElementById('frames-grid')
const downloadZipBtn = document.getElementById('download-zip-btn')
const errorBanner    = document.getElementById('error-banner')

// Player refs
const playerImg      = document.getElementById('player-img')
const playerCaption  = document.getElementById('player-caption')
const playerScrub    = document.getElementById('player-scrub')
const playerPrev     = document.getElementById('player-prev')
const playerNext     = document.getElementById('player-next')
const playerPlay     = document.getElementById('player-play')
const playerCounter  = document.getElementById('player-counter')
const playerFpsSel   = document.getElementById('player-fps-select')
const playerLoop     = document.getElementById('player-loop-check')

let selectedFile = null
let currentJobId = null

// ── Parameter controls ──────────────────────────────────────────────────────
// Maps a param id -> how to read it from its <input>/<select>.
const PARAM_IDS = [
  'method', 'detector', 'max_features', 'match_ratio',
  'flow_max_corners', 'flow_quality', 'flow_win_size',
  'ransac_thresh', 'gain', 'blur_ksize', 'threshold', 'morph_ksize',
  'overlay', 'colormap', 'normalize', 'num_pairs', 'frame_step',
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
  clearError()
}

// ── Upload & process ──────────────────────────────────────────────────────────
uploadBtn.addEventListener('click', async () => {
  if (!selectedFile) return
  uploadBtn.disabled = true
  clearError()

  showProgress('Uploading…', 10)
  uploadSection.classList.add('hidden')
  progressSection.classList.remove('hidden')
  resultsSection.classList.add('hidden')

  try {
    // 1. Upload
    const formData = new FormData()
    formData.append('file', selectedFile)

    const uploadRes = await fetch(`${API_URL}/api/upload`, {
      method: 'POST',
      body: formData,
    })
    if (!uploadRes.ok) throw new Error(await uploadRes.text())
    const { job_id } = await uploadRes.json()
    currentJobId = job_id

    // 2. Trigger processing with the chosen BOS parameters
    showProgress('Computing schlieren…', 40)
    const processRes = await fetch(`${API_URL}/api/process/${job_id}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(collectParams()),
    })
    if (!processRes.ok) throw new Error(await processRes.text())

    // 3. Poll for results
    showProgress('Fetching results…', 80)
    const resultsRes = await fetch(`${API_URL}/api/results/${job_id}`)
    if (!resultsRes.ok) throw new Error(await resultsRes.text())
    const results = await resultsRes.json()

    showProgress('Done!', 100)
    await sleep(400)

    renderResults(results)
  } catch (err) {
    progressSection.classList.add('hidden')
    uploadSection.classList.remove('hidden')
    uploadBtn.disabled = false
    showError(`Error: ${err.message}`)
  }
})

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

  // Frame-by-frame player
  initPlayer(results.extracted_frames ?? [])

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

// ── Frame-by-frame player ─────────────────────────────────────────────────────
let playerFrames = []
let playerIndex = 0
let playerTimer = null

function initPlayer(frames) {
  stopPlayback()
  playerFrames = frames
  playerIndex = 0

  if (!frames.length) {
    document.getElementById('player').classList.add('hidden')
    return
  }
  document.getElementById('player').classList.remove('hidden')

  // Preload all frame images so playback is smooth.
  for (const f of frames) {
    const img = new Image()
    img.src = frameUrl(f.index)
  }

  playerScrub.max = String(frames.length - 1)
  playerScrub.value = '0'
  showFrame(0)
}

function frameUrl(index) {
  return `${API_URL}/api/frames/${currentJobId}/${index}`
}

function showFrame(i) {
  if (!playerFrames.length) return
  playerIndex = (i + playerFrames.length) % playerFrames.length
  const f = playerFrames[playerIndex]
  playerImg.src = frameUrl(f.index)
  playerScrub.value = String(playerIndex)
  playerCounter.textContent = `${playerIndex + 1} / ${playerFrames.length}`
  const aligned = f.stats?.aligned === false ? 'unaligned' : 'aligned'
  playerCaption.textContent =
    `${f.prev_index ?? '?'}→${f.index} · ${formatTimestamp(f.timestamp_ms)} · ${aligned}`
}

function stepFrame(delta) {
  const next = playerIndex + delta
  // When stepping manually past the end without loop, clamp instead of wrapping.
  if (!playerLoop.checked && (next < 0 || next >= playerFrames.length)) {
    showFrame(Math.max(0, Math.min(playerFrames.length - 1, next)))
    return
  }
  showFrame(next)
}

function startPlayback() {
  if (!playerFrames.length) return
  const fps = Number(playerFpsSel.value) || 10
  playerPlay.textContent = '⏸ Pause'
  playerTimer = setInterval(() => {
    const atEnd = playerIndex >= playerFrames.length - 1
    if (atEnd && !playerLoop.checked) {
      stopPlayback()
      return
    }
    showFrame(playerIndex + 1)
  }, 1000 / fps)
}

function stopPlayback() {
  if (playerTimer) {
    clearInterval(playerTimer)
    playerTimer = null
  }
  playerPlay.textContent = '▶ Play'
}

function togglePlayback() {
  if (playerTimer) stopPlayback()
  else startPlayback()
}

playerPlay.addEventListener('click', togglePlayback)
playerPrev.addEventListener('click', () => { stopPlayback(); stepFrame(-1) })
playerNext.addEventListener('click', () => { stopPlayback(); stepFrame(1) })
playerScrub.addEventListener('input', () => { stopPlayback(); showFrame(Number(playerScrub.value)) })
playerFpsSel.addEventListener('change', () => { if (playerTimer) { stopPlayback(); startPlayback() } })

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
