import './style.css'
import './workspace.css'
import { api } from './api'
import { icon } from './icons'
import { analysisProgress, applicationTemplate, batchCard, escapeHtml, resultCard } from './ui'

const app = document.querySelector<HTMLDivElement>('#app')!
app.innerHTML = applicationTemplate()

let singleFile: File | null = null
let singlePreview = ''
let batchFiles: File[] = []

const allowedTypes = ['image/jpeg', 'image/png', 'image/bmp']

function validFiles(files: File[]) {
  const invalid = files.find(file => !allowedTypes.includes(file.type) || file.size > 10 * 1024 * 1024)
  if (invalid) { showToast('Use JPG, PNG, or BMP files smaller than 10 MB each.'); return false }
  return true
}

function size(bytes: number) { return bytes > 1048576 ? `${(bytes / 1048576).toFixed(1)} MB` : `${(bytes / 1024).toFixed(1)} KB` }
function showToast(message: string) { const toast = document.querySelector<HTMLElement>('#toast')!; toast.textContent = message; toast.classList.remove('hidden'); setTimeout(() => toast.classList.add('hidden'), 4300) }
function setButtonLoading(button: HTMLButtonElement, label: string, loading: boolean) { button.disabled = loading; button.innerHTML = loading ? `<i class="spinner"></i> ${label}` : label }

function selectSingle(file: File | null) {
  if (!file || !validFiles([file])) return
  singleFile = file
  if (singlePreview) URL.revokeObjectURL(singlePreview)
  singlePreview = URL.createObjectURL(file)
  const selected = document.querySelector<HTMLElement>('#single-selected')!
  selected.innerHTML = `<img src="${singlePreview}" alt="Selected image"><div><strong>${escapeHtml(file.name)}</strong><span>${size(file.size)} · Ready to analyze</span></div><button id="remove-single" aria-label="Remove image">${icon('close')}</button>`
  selected.classList.remove('hidden'); document.querySelector('#single-empty')!.classList.add('hidden')
  document.querySelector<HTMLButtonElement>('#single-analyze')!.disabled = false
  document.querySelector('#remove-single')!.addEventListener('click', event => { event.preventDefault(); resetSingle() })
}

function resetSingle() {
  singleFile = null; if (singlePreview) URL.revokeObjectURL(singlePreview); singlePreview = ''
  document.querySelector<HTMLInputElement>('#single-file')!.value = ''
  document.querySelector('#single-selected')!.classList.add('hidden'); document.querySelector('#single-empty')!.classList.remove('hidden')
  document.querySelector<HTMLButtonElement>('#single-analyze')!.disabled = true
}

async function analyzeSingle() {
  if (!singleFile) return
  const button = document.querySelector<HTMLButtonElement>('#single-analyze')!
  setButtonLoading(button, 'Running complete pipeline', true)
  const empty = document.querySelector<HTMLElement>('#single-empty-result')!
  empty.innerHTML = analysisProgress()
  try {
    const response = await api.analyze(singleFile)
    const audit = await api.submissionAudit(response.submission.submission_id)
    document.querySelector<HTMLElement>('#single-result')!.innerHTML = resultCard(response.result, audit.events)
    document.querySelector('#single-result')!.classList.remove('hidden'); empty.classList.add('hidden')
    document.querySelector('#single-badge')!.textContent = 'Analysis complete'
  } catch (error) {
    showToast(error instanceof Error ? error.message : 'Unable to reach the analysis service.')
    empty.innerHTML = `${icon('image')}<h4>Analysis unavailable</h4><p>Check that the Flask backend is running, then try again.</p>`
  } finally { setButtonLoading(button, `Analyze another image ${icon('arrow')}`, false) }
}

function selectBatch(files: File[]) {
  if (!files.length || !validFiles(files)) return
  if (files.length > 50) { showToast('A batch can contain at most 50 images.'); return }
  batchFiles = files
  const list = document.querySelector<HTMLElement>('#batch-list')!
  list.innerHTML = files.map((file, index) => `<div><span>${index + 1}</span><strong>${escapeHtml(file.name)}</strong><small>${size(file.size)}</small></div>`).join('')
  document.querySelector<HTMLButtonElement>('#batch-analyze')!.disabled = false
  document.querySelector('#batch-badge')!.textContent = `${files.length} image${files.length === 1 ? '' : 's'} selected`
}

async function analyzeBatch() {
  if (!batchFiles.length) return
  const button = document.querySelector<HTMLButtonElement>('#batch-analyze')!
  setButtonLoading(button, 'Processing batch and creating report', true)
  const empty = document.querySelector<HTMLElement>('#batch-empty-result')!
  empty.innerHTML = analysisProgress()
  try {
    const batch = await api.batch(batchFiles)
    const audit = await api.batchAudit(batch.batch_id)
    document.querySelector<HTMLElement>('#batch-result')!.innerHTML = batchCard(batch, audit.events, api.reportUrl(batch.batch_id))
    document.querySelector('#batch-result')!.classList.remove('hidden'); empty.classList.add('hidden')
    document.querySelector('#batch-badge')!.textContent = 'PDF report ready'
  } catch (error) {
    showToast(error instanceof Error ? error.message : 'The batch could not be processed.')
    empty.innerHTML = `${icon('files')}<h4>Batch analysis unavailable</h4><p>Check that the Flask backend and model assets are available, then try again.</p>`
  } finally { setButtonLoading(button, `Analyze batch & generate PDF ${icon('arrow')}`, false) }
}

function setupDropzone(zoneId: string, inputId: string, handler: (files: File[]) => void) {
  const zone = document.querySelector<HTMLElement>(zoneId)!; const input = document.querySelector<HTMLInputElement>(inputId)!
  input.addEventListener('change', () => handler(Array.from(input.files ?? [])))
  ;['dragenter', 'dragover'].forEach(event => zone.addEventListener(event, action => { action.preventDefault(); zone.classList.add('drag') }))
  ;['dragleave', 'drop'].forEach(event => zone.addEventListener(event, action => { action.preventDefault(); zone.classList.remove('drag') }))
  zone.addEventListener('drop', event => handler(Array.from(event.dataTransfer?.files ?? [])))
}

document.querySelectorAll<HTMLButtonElement>('.mode').forEach(button => button.addEventListener('click', () => {
  const batch = button.dataset.mode === 'batch'
  document.querySelectorAll('.mode').forEach(item => item.classList.toggle('active', item === button))
  document.querySelector('#single-workspace')!.classList.toggle('hidden', batch)
  document.querySelector('#batch-workspace')!.classList.toggle('hidden', !batch)
}))
setupDropzone('#single-dropzone', '#single-file', files => selectSingle(files[0] ?? null))
setupDropzone('#batch-dropzone', '#batch-files', selectBatch)
document.querySelector('#single-analyze')!.addEventListener('click', analyzeSingle)
document.querySelector('#batch-analyze')!.addEventListener('click', analyzeBatch)

api.health().then(() => { document.querySelector('#system-status')!.innerHTML = '<i></i>System online' }).catch(() => { document.querySelector('#system-status')!.textContent = 'Backend offline' })

const sectionLinks = Array.from(document.querySelectorAll<HTMLAnchorElement>('[data-section]'))
const observedSections = sectionLinks
  .map(link => document.getElementById(link.dataset.section ?? ''))
  .filter((section): section is HTMLElement => section !== null)

const sectionObserver = new IntersectionObserver(entries => {
  const visible = entries
    .filter(entry => entry.isIntersecting)
    .sort((left, right) => right.intersectionRatio - left.intersectionRatio)[0]
  if (!visible) return
  sectionLinks.forEach(link => link.classList.toggle('active', link.dataset.section === visible.target.id))
}, { rootMargin: '-22% 0px -58% 0px', threshold: [0.1, 0.35, 0.6] })

observedSections.forEach(section => sectionObserver.observe(section))
