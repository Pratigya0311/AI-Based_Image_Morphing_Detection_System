import type { AnalysisResponse, AuditEvent, BatchResponse } from './types'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options)
  const payload = await response.json() as T & { error?: string }
  if (!response.ok) throw new Error(payload.error ?? 'The request could not be completed.')
  return payload
}

export const api = {
  me: () => request<{ authenticated: boolean; user: { name: string; email: string; picture?: string } }>('/api/auth/me'),
  profile: () => request<{ user: { name: string; email: string; picture?: string }; stats: { images_analyzed: number; batches_created: number; morphed_flags: number } }>('/api/auth/profile'),
  loginUrl: '/api/auth/google',
  logout: () => request<{ authenticated: boolean }>('/api/auth/logout', { method: 'POST' }),
  health: () => request<{ status: string }>('/api/analysis/health'),
  analyze: (file: File) => { const form = new FormData(); form.append('image', file); return request<AnalysisResponse>('/api/analysis/analyze', { method: 'POST', body: form }) },
  batch: (files: File[]) => { const form = new FormData(); files.forEach(file => form.append('images', file)); form.append('generate_report', 'true'); return request<BatchResponse>('/api/analysis/batch', { method: 'POST', body: form }) },
  submissionAudit: (submissionId: string) => request<{ events: AuditEvent[] }>(`/api/analysis/audit/submission/${submissionId}`),
  batchAudit: (batchId: string) => request<{ events: AuditEvent[] }>(`/api/analysis/audit/batch/${batchId}`),
  reportUrl: (batchId: string) => `/api/analysis/report/${batchId}`,
}
