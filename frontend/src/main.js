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

let selectedFile = null
let currentJobId = null

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

    // 2. Trigger processing
    showProgress('Processing with OpenCV…', 40)
    const processRes = await fetch(`${API_URL}/api/process/${job_id}`, { method: 'POST' })
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

  // Frames
  framesGrid.innerHTML = ''
  for (const frame of results.extracted_frames ?? []) {
    const card = document.createElement('div')
    card.className = 'frame-card'
    card.innerHTML = `
      <img src="${API_URL}/api/frames/${currentJobId}/${frame.index}" alt="Frame ${frame.index}" loading="lazy" />
      <div class="frame-label">
        <span>#${frame.index} · ${formatTimestamp(frame.timestamp_ms)}</span>
        <a href="${API_URL}/api/frames/${currentJobId}/${frame.index}" download="frame_${frame.index}.jpg">↓</a>
      </div>
    `
    framesGrid.appendChild(card)
  }
}

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
