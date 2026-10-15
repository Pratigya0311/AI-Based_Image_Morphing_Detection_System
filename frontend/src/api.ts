import type { AnalysisResponse, AuditEvent, BatchResponse } from './types'

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`, { credentials: 'include', ...options })
  const payload = await response.json() as T & { error?: string }
  if (!response.ok) throw new Error(payload.error ?? 'The request could not be completed.')
  return payload
}

export const api = {
  me: () => request<{ authenticated: boolean; user: { name: string; email: string; picture?: string } }>('/api/auth/me'),
  profile: () => request<{ user: { name: string; email: string; picture?: string; has_password?: boolean; role?: string; last_signed_in_at?: string }; stats: { images_analyzed: number; batches_created: number; morphed_flags: number } }>('/api/auth/profile'),
  loginUrl: `${apiBaseUrl}/api/auth/google`,
  login: (email: string, password: string) => request<{ authenticated: boolean }>('/api/auth/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ email, password }) }),
  register: (name: string, email: string, password: string) => request<{ authenticated?: boolean; message?: string }>('/api/auth/register', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, email, password }) }),
  forgotPassword: (email: string) => request<{ message: string }>('/api/auth/forgot-password', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ email }) }),
  resetPassword: (token: string, password: string) => request<{ message: string }>('/api/auth/reset-password', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ token, password }) }),
  verifyEmail: (token: string) => request<{ authenticated: boolean }>('/api/auth/verify-email', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ token }) }),
  changePassword: (currentPassword: string, newPassword: string) => request<{ message: string }>('/api/auth/change-password', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }) }),
  adminUsers: () => request<{ users: Array<{ id: string; name: string; email: string; role: string; created_at: string; last_signed_in_at: string }> }>('/api/auth/admin/users'),
  adminActivity: (offset = 0, limit = 12) => request<{ events: AuditEvent[]; has_more: boolean }>(`/api/auth/admin/activity?offset=${offset}&limit=${limit}`),
  deleteUser: (id: string) => request<{ message: string }>(`/api/auth/admin/users/${id}`, { method: 'DELETE' }),
  logout: () => request<{ authenticated: boolean }>('/api/auth/logout', { method: 'POST' }),
  health: () => request<{ status: string }>('/api/analysis/health'),
  analyze: (file: File) => { const form = new FormData(); form.append('image', file); return request<AnalysisResponse>('/api/analysis/analyze', { method: 'POST', body: form }) },
  batch: (files: File[]) => { const form = new FormData(); files.forEach(file => form.append('images', file)); form.append('generate_report', 'true'); return request<BatchResponse>('/api/analysis/batch', { method: 'POST', body: form }) },
  submissionAudit: (submissionId: string) => request<{ events: AuditEvent[] }>(`/api/analysis/audit/submission/${submissionId}`),
  batchAudit: (batchId: string) => request<{ events: AuditEvent[] }>(`/api/analysis/audit/batch/${batchId}`),
  reportUrl: (batchId: string) => `${apiBaseUrl}/api/analysis/report/${batchId}`,
}
