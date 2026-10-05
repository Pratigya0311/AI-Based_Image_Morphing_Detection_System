export type AnalysisResult = {
  submission_id: string
  predicted_label: 'Genuine' | 'Morphed'
  confidence_percentage: number
  class_probabilities: { Genuine: number; Morphed: number }
  generated_at: string
}

export type AnalysisResponse = {
  submission: { submission_id: string; valid: boolean }
  result: AnalysisResult
}

export type BatchResponse = {
  batch_id: string
  successful_count: number
  failed_count: number
  report_path: string | null
  items: Array<{ filename: string; succeeded: boolean; result?: AnalysisResult; error?: string }>
}

export type AuditEvent = { event_type: string; outcome: string; timestamp: string }
