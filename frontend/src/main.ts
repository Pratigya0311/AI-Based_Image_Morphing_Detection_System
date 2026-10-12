import './style.css'
import './workspace.css'
import './login.css'
import { api } from './api'
import { icon } from './icons'
import { loginTemplate } from './login'
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

function analysisFailure(error: unknown) {
  const message = error instanceof Error ? error.message.toLowerCase() : ''
  if (message.includes('no detectable face') || message.includes('face not detected')) {
    return { title: 'Face not detected', detail: 'We could not find one clear face in this image. Upload a well-lit, front-facing portrait with the face clearly visible.' }
  }
  if (message.includes('multiple faces')) {
    return { title: 'Multiple faces detected', detail: 'Please upload an image containing only one person for an accurate analysis.' }
  }
  if (message.includes('quality is too low') || message.includes('blurry')) {
    return { title: 'Image quality is too low', detail: 'Use a sharper, well-lit image with a clearly visible face, then try again.' }
  }
  if (message.includes('invalid image') || message.includes('could not be decoded')) {
    return { title: 'Image could not be processed', detail: 'Choose a valid JPG, PNG, or BMP image and try again.' }
  }
  return { title: 'Analysis could not be completed', detail: 'Please try again with a clear portrait image in a moment.' }
}

async function showAdminDashboard() {
  try {
    const [usersResponse, activityResponse] = await Promise.all([api.adminUsers(), api.adminActivity()])
    const users = usersResponse.users.map(user => `<tr><td>${escapeHtml(user.name)}</td><td>${escapeHtml(user.email)}</td><td>${escapeHtml(user.role)}</td><td>${new Date(user.last_signed_in_at).toLocaleString()}</td><td>${user.role === 'admin' ? 'Protected' : `<button class="remove-user" data-user-id="${user.id}">Remove</button>`}</td></tr>`).join('')
    const activity = activityResponse.events.slice(0, 12).map(event => `<li><strong>${escapeHtml(event.event_type.replaceAll('_', ' '))}</strong><span>${escapeHtml(event.actor_email ?? 'System')} · ${new Date(event.timestamp).toLocaleString()}</span></li>`).join('') || '<li><span>No activity recorded yet.</span></li>'
    const page = document.querySelector<HTMLElement>('#top')!
    page.innerHTML = `<section class="admin-page"><header><div><p class="eyebrow">ADMINISTRATOR AREA</p><h1>System administration</h1><p>Manage registered users and review recent protected activity.</p></div></header><section class="admin-grid"><article><h2>Registered users <span>${usersResponse.users.length}</span></h2><div class="admin-table-wrap"><table><thead><tr><th>Name</th><th>Email</th><th>Role</th><th>Last sign-in</th><th>Action</th></tr></thead><tbody>${users}</tbody></table></div></article><article><h2>Recent activity</h2><ol class="admin-activity">${activity}</ol></article></section></section>`
    document.querySelectorAll<HTMLAnchorElement>('[data-page]').forEach(link => link.classList.remove('active'))
    document.querySelector('#admin-nav-link')?.classList.add('active')
    document.querySelectorAll<HTMLAnchorElement>('.nav nav [data-page], .nav .brand').forEach(link => link.addEventListener('click', event => {
      event.preventDefault()
      window.location.hash = link.getAttribute('href')?.replace('#', '') || 'home'
      window.location.reload()
    }))
    document.querySelectorAll<HTMLButtonElement>('.remove-user').forEach(button => button.addEventListener('click', async () => {
      if (!window.confirm('Remove this user and their analysis history permanently?')) return
      try { await api.deleteUser(button.dataset.userId!); await showAdminDashboard() } catch (error) { window.alert(error instanceof Error ? error.message : 'Unable to remove user') }
    }))
  } catch (error) { showToast(error instanceof Error ? error.message : 'Unable to load the admin dashboard.') }
}

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
    const failure = analysisFailure(error)
    showToast(failure.title)
    empty.innerHTML = `${icon('image')}<h4>${failure.title}</h4><p>${failure.detail}</p>`
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
    showToast('Batch analysis could not be completed')
    empty.innerHTML = `${icon('files')}<h4>Batch analysis could not be completed</h4><p>Review your images and try again. Each image should be clear and contain one visible face.</p>`
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

function routeWorkspace() {
  const requestedRoute = window.location.hash.replace('#', '') || 'home'
  const route = requestedRoute === 'analyze' ? 'analyze' : 'home'
  const homeSections = ['.hero', '.project-credit', '.metrics', '.process']
  const analysis = document.querySelector<HTMLElement>('#analyze')!
  const isAnalysis = route === 'analyze'
  homeSections.forEach(selector => document.querySelector<HTMLElement>(selector)!.classList.toggle('hidden', isAnalysis))
  analysis.classList.toggle('hidden', !isAnalysis)
  document.querySelectorAll<HTMLAnchorElement>('[data-page]').forEach(link => link.classList.toggle('active', link.dataset.page === route))
  if (requestedRoute === 'how') window.setTimeout(() => document.querySelector('#how')?.scrollIntoView({ behavior: 'smooth' }), 0)
  else window.scrollTo({ top: 0, behavior: 'auto' })
}

window.addEventListener('hashchange', routeWorkspace)
routeWorkspace()

void api.profile().then(profile => {
  const initial = profile.user.name.trim().charAt(0).toUpperCase()
  const avatar = document.querySelector<HTMLElement>('#account-avatar')!
  const name = document.querySelector<HTMLElement>('#account-name')!
  avatar.textContent = initial; name.textContent = profile.user.name.split(' ')[0]
  if (profile.user.picture) avatar.style.backgroundImage = `url("${profile.user.picture}")`
  const menu = document.querySelector<HTMLElement>('#account-menu')!
  const lastSignedIn = profile.user.last_signed_in_at ? new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(profile.user.last_signed_in_at)) : 'Not available'
  menu.innerHTML = `<div class="account-info"><strong>${escapeHtml(profile.user.name)}</strong><span>${escapeHtml(profile.user.email)}</span><small>Last signed in: ${escapeHtml(lastSignedIn)}</small></div><div class="account-stats"><div><strong>${profile.stats.images_analyzed}</strong><span>Images analyzed</span></div><div><strong>${profile.stats.batches_created}</strong><span>Batches created</span></div><div><strong>${profile.stats.morphed_flags}</strong><span>Morph flags</span></div></div>${profile.user.role === 'admin' ? '<button class="account-secondary" id="admin-dashboard-button">Admin dashboard</button>' : ''}${profile.user.has_password ? '<button class="account-secondary" id="change-password-button">Change password</button><form class="account-password hidden" id="change-password-form"><input name="current_password" type="password" placeholder="Current password" required><input name="new_password" type="password" placeholder="New password (8+ characters)" minlength="8" required><button type="submit">Update password</button></form>' : '<p class="account-provider">Signed in with Google</p>'}<button id="logout-button">Sign out</button>`
  if (profile.user.role === 'admin') {
    const navigation = document.querySelector<HTMLElement>('.nav nav')!
    navigation.insertAdjacentHTML('beforeend', '<a href="#admin" id="admin-nav-link">Admin</a>')
    document.querySelector('#admin-nav-link')!.addEventListener('click', event => { event.preventDefault(); void showAdminDashboard() })
    if (window.location.hash === '#admin') void showAdminDashboard()
  }
  document.querySelector('#account-button')!.addEventListener('click', () => menu.classList.toggle('hidden'))
  document.querySelector('#admin-dashboard-button')?.addEventListener('click', () => { window.location.hash = 'admin'; void showAdminDashboard() })
  document.querySelector('#change-password-button')?.addEventListener('click', () => document.querySelector('#change-password-form')!.classList.toggle('hidden'))
  document.querySelector<HTMLFormElement>('#change-password-form')?.addEventListener('submit', async event => {
    event.preventDefault(); const form = event.currentTarget as HTMLFormElement; const values = new FormData(form)
    try { await api.changePassword(String(values.get('current_password') ?? ''), String(values.get('new_password') ?? '')); form.reset(); form.classList.add('hidden'); showToast('Password updated successfully.') } catch (error) { showToast(error instanceof Error ? error.message : 'Unable to update password.') }
  })
  document.querySelector('#logout-button')!.addEventListener('click', async () => { await api.logout(); window.location.reload() })
}).catch(() => {
  const resetToken = new URLSearchParams(window.location.search).get('reset_token') ?? ''
  const verifyToken = new URLSearchParams(window.location.search).get('verify_token') ?? ''
  if (verifyToken) {
    app.innerHTML = loginTemplate('Verifying your email address...')
    void api.verifyEmail(verifyToken).then(() => { window.history.replaceState({}, '', window.location.pathname); window.location.reload() }).catch(error => {
      app.innerHTML = loginTemplate(error instanceof Error ? error.message : 'Unable to verify this email link.')
    })
    return
  }
  app.innerHTML = loginTemplate('', resetToken)
  const message = document.querySelector<HTMLElement>('#auth-message')!
  const showMessage = (value: string, error = true) => { message.textContent = value; message.classList.toggle('error', error) }
  const formValue = (form: HTMLFormElement, name: string) => String(new FormData(form).get(name) ?? '')
  const showPanel = (panel: 'login' | 'register' | 'forgot') => {
    document.querySelectorAll<HTMLElement>('.local-auth').forEach(form => form.classList.toggle('hidden', form.id !== `${panel}-form`))
    document.querySelectorAll<HTMLButtonElement>('.auth-tab').forEach(tab => tab.classList.toggle('active', tab.dataset.auth === panel))
    message.textContent = ''
  }
  document.querySelectorAll<HTMLButtonElement>('.auth-tab').forEach(tab => tab.addEventListener('click', () => showPanel(tab.dataset.auth as 'login' | 'register')))
  document.querySelector('#forgot-password')?.addEventListener('click', () => showPanel('forgot'))
  document.querySelector('#back-to-login')?.addEventListener('click', () => showPanel('login'))
  document.querySelector<HTMLFormElement>('#login-form')?.addEventListener('submit', async event => {
    event.preventDefault(); const form = event.currentTarget as HTMLFormElement
    try { await api.login(formValue(form, 'email'), formValue(form, 'password')); window.location.reload() } catch (error) { showMessage(error instanceof Error ? error.message : 'Unable to sign in') }
  })
  document.querySelector<HTMLFormElement>('#register-form')?.addEventListener('submit', async event => {
    event.preventDefault(); const form = event.currentTarget as HTMLFormElement; const password = formValue(form, 'password')
    if (password !== formValue(form, 'confirm_password')) { showMessage('Passwords do not match'); return }
    try { const response = await api.register(formValue(form, 'name'), formValue(form, 'email'), password); if (response.authenticated) window.location.reload(); else showMessage(response.message ?? 'Check your email to verify your account.', false) } catch (error) { showMessage(error instanceof Error ? error.message : 'Unable to create account') }
  })
  document.querySelector<HTMLFormElement>('#forgot-form')?.addEventListener('submit', async event => {
    event.preventDefault(); const form = event.currentTarget as HTMLFormElement
    try { const response = await api.forgotPassword(formValue(form, 'email')); showMessage(response.message, false) } catch (error) { showMessage(error instanceof Error ? error.message : 'Unable to send reset link') }
  })
  document.querySelector<HTMLFormElement>('#reset-form')?.addEventListener('submit', async event => {
    event.preventDefault(); const form = event.currentTarget as HTMLFormElement; const password = formValue(form, 'password')
    if (password !== formValue(form, 'confirm_password')) { showMessage('Passwords do not match'); return }
    try { const response = await api.resetPassword(resetToken, password); showMessage(response.message, false); window.setTimeout(() => { window.history.replaceState({}, '', window.location.pathname); window.location.reload() }, 1400) } catch (error) { showMessage(error instanceof Error ? error.message : 'Unable to reset password') }
  })
  document.querySelector('#google-login')?.addEventListener('click', () => window.location.assign(api.loginUrl))
})
