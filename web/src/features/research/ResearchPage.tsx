import DOMPurify from 'dompurify'
import { marked } from 'marked'
import { type FormEvent, useCallback, useEffect, useRef, useState } from 'react'
import {
  applyBackendEvent,
  applyReviewSubmitFailure,
  applyReviewSubmitStart,
  createHistoryTitle,
  createId,
  createResearchRun,
  getAssistantPlainText,
  getStepName,
  type BackendEvent,
  type ResearchRun,
  type ResearchStage,
  toAssistantUiMessages,
} from './researchRun'
import {
  buildResearchQuery,
  createDefaultTaskSettings,
  type TaskSettings,
  type TaskSettingsErrors,
  validateTaskSettings,
} from './researchSettings'
import { SettingsModal } from './SettingsModal'

const parseMarkdown = (content: string) =>
  DOMPurify.sanitize(marked.parse(content) as string)

const emptyErrors: TaskSettingsErrors = {
  year: '',
  keywords: '',
  paperLimit: '',
  customPrompt: '',
}

const cloneRun = (run: ResearchRun): ResearchRun => ({
  ...run,
  user: { ...run.user },
  assistant: {
    ...run.assistant,
    stages: run.assistant.stages.map((s) => ({ ...s })),
  },
})

const persistHistory = (run: ResearchRun) => {
  const content = getAssistantPlainText(run.assistant)
  if (!content) return
  const item = {
    id: run.assistant.id,
    title: createHistoryTitle(run.user.content || ''),
    query: run.user.content,
    status: run.assistant.status === 'error' ? 'failed' : 'completed',
    createdAt: run.user.timestamp,
    updatedAt: run.assistant.completedAt,
    content,
  }
  try {
    const saved = JSON.parse(localStorage.getItem('reportHistory') || '[]')
    const next = [item, ...saved.filter((r: { id: string }) => r.id !== item.id)].slice(0, 50)
    localStorage.setItem('reportHistory', JSON.stringify(next))
  } catch {
    // ignore
  }
}

const getCurrentStepName = (stages: ResearchStage[]): string => {
  const active = stages.find(
    (s) => s.status === 'preparing' || s.status === 'thinking' || s.status === 'generating'
  )
  if (active) {
    if (active.status === 'preparing') return `${getStepName(active.step)}准备中...`
    if (active.status === 'thinking') return `${getStepName(active.step)}思考中...`
    if (active.status === 'generating') return `${getStepName(active.step)}生成中...`
  }
  return '处理中...'
}

export function ResearchPage() {
  const [userInput, setUserInput] = useState('')
  const [taskSettings, setTaskSettings] = useState<TaskSettings>(() => createDefaultTaskSettings())
  const [errors, setErrors] = useState<TaskSettingsErrors>(emptyErrors)
  const [run, setRun] = useState<ResearchRun | null>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const sourceRef = useRef<EventSource | null>(null)
  const messagesEndRef = useRef<HTMLDivElement | null>(null)
  const inputRef = useRef<HTMLTextAreaElement | null>(null)

  const closeEventSource = useCallback(() => {
    sourceRef.current?.close()
    sourceRef.current = null
  }, [])

  const autoScroll = useCallback(() => {
    requestAnimationFrame(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    })
  }, [])

  useEffect(() => {
    autoScroll()
  }, [autoScroll, run])

  useEffect(() => {
    inputRef.current?.focus()
  }, [run?.assistant.id])

  useEffect(() => () => closeEventSource(), [closeEventSource])

  const finishRun = useCallback(() => {
    closeEventSource()
    setRun((current) =>
      current
        ? { ...cloneRun(current), isSubmitting: false, isReviewing: false, isReviewSubmitting: false }
        : current
    )
  }, [closeEventSource])

  const startResearch = useCallback(
    async (rawQuery: string) => {
      const query = rawQuery.trim()
      if (!query || run?.isSubmitting) return

      const validation = validateTaskSettings(taskSettings)
      setErrors(validation.errors)
      if (!validation.valid) return

      closeEventSource()
      const nextRun = createResearchRun(createId('user'), createId('assistant'), query)
      setRun(nextRun)
      setUserInput('')
      requestAnimationFrame(() => inputRef.current?.focus())

      const source = new EventSource(
        `/api/research?query=${encodeURIComponent(buildResearchQuery(query, taskSettings))}`
      )
      sourceRef.current = source

      source.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as BackendEvent
          setRun((current) => {
            if (!current) return current
            const draft = cloneRun(current)
            applyBackendEvent(draft, data)
            if (data.state === 'finished') {
              persistHistory(draft)
            }
            requestAnimationFrame(autoScroll)
            return draft
          })
        } catch {
          // ignore parse errors
        }
      }

      source.onerror = () => {
        setRun((current) => {
          if (!current?.isSubmitting) return current
          const draft = cloneRun(current)
          applyBackendEvent(draft, {
            step: 'connection',
            state: 'error',
            data: '连接后端服务失败，请确认服务已经启动。',
          })
          return draft
        })
        finishRun()
      }
    },
    [autoScroll, closeEventSource, finishRun, run?.isSubmitting, taskSettings]
  )

  const submitRequest = (event: FormEvent) => {
    event.preventDefault()
    void startResearch(userInput)
  }

  const stopCurrentRun = useCallback(async () => {
    closeEventSource()
    setRun((current) => {
      if (!current) return current
      const draft = cloneRun(current)
      draft.assistant.status = 'stopped'
      draft.assistant.statusText = '已停止'
      draft.assistant.completedAt = new Date().toISOString()
      draft.isSubmitting = false
      draft.isReviewing = false
      return draft
    })
  }, [closeEventSource])

  const clearConversation = () => {
    if (run?.isSubmitting) void stopCurrentRun()
    setRun(null)
    setUserInput('')
    requestAnimationFrame(() => inputRef.current?.focus())
  }

  const submitReviewInput = async () => {
    if (!run?.reviewInput.trim() || run.isReviewSubmitting) return
    const reviewText = run.reviewInput.trim()
    let snapshot: ReturnType<typeof applyReviewSubmitStart> | null = null

    setRun((current) => {
      if (!current) return current
      const draft = cloneRun(current)
      snapshot = applyReviewSubmitStart(draft, reviewText)
      return draft
    })

    try {
      const response = await fetch('/send_input', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input: reviewText }),
      })
      if (!response.ok) throw new Error(`审核提交失败: ${response.status}`)
    } catch {
      setRun((current) => {
        if (!current) return current
        const draft = cloneRun(current)
        if (snapshot) applyReviewSubmitFailure(draft, snapshot)
        return draft
      })
      window.alert('提交审核反馈失败，请检查后端服务。')
      return
    }

    setRun((current) =>
      current ? { ...cloneRun(current), isReviewSubmitting: false } : current
    )
  }

  const copyResult = async () => {
    if (!run) return
    const content = getAssistantPlainText(run.assistant)
    if (!content) return
    await navigator.clipboard.writeText(content)
  }

  const downloadResult = () => {
    if (!run) return
    const content = getAssistantPlainText(run.assistant)
    if (!content) return
    const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `research-report-${Date.now()}.md`
    a.click()
    URL.revokeObjectURL(url)
  }

  const isRunning = Boolean(run?.isSubmitting)
  const isDone = Boolean(run && !isRunning && !run.isReviewing && run.assistant.status === 'done')
  const hasActiveStage = run?.isSubmitting && run.assistant.stages.length > 0

  const messages = run ? toAssistantUiMessages(run) : []

  return (
    <div className="chat-page">
      {!run ? (
        <div className="chat-empty-state">
          <h1 className="welcome-text">今天想探索什么研究方向？</h1>
          <div className="main-input-wrapper">
            <form onSubmit={submitRequest} style={{ width: '100%' }}>
              <div className="chat-input-container">
                <textarea
                  ref={inputRef}
                  className="main-input"
                  rows={3}
                  value={userInput}
                  placeholder="例如：调研多模态大模型在医学影像诊断中的研究进展"
                  onChange={(e) => setUserInput(e.target.value)}
                />
                <button className="chat-send-btn" type="submit" disabled={!userInput.trim() || isRunning}>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="12" y1="19" x2="12" y2="5" />
                    <polyline points="5 12 12 5 19 12" />
                  </svg>
                </button>
              </div>
            </form>
          </div>
        </div>
      ) : (
        <>
          <div className="chat-input-bar">
            <span className="welcome-text">探索研究方向</span>
            <form onSubmit={submitRequest} style={{ display: 'flex', flex: 1, alignItems: 'center' }}>
              <div className="chat-input-container" style={{ flex: 1 }}>
                <textarea
                  ref={inputRef}
                  className="main-input"
                  rows={1}
                  value={userInput}
                  placeholder="继续补充研究要求..."
                  disabled={isRunning}
                  onChange={(e) => setUserInput(e.target.value)}
                />
                {isRunning ? (
                  <button className="chat-stop-btn" type="button" onClick={() => void stopCurrentRun()}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                      <rect x="6" y="6" width="12" height="12" rx="2" />
                    </svg>
                  </button>
                ) : (
                  <button className="chat-send-btn" type="submit" disabled={!userInput.trim()}>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="12" y1="19" x2="12" y2="5" />
                      <polyline points="5 12 12 5 19 12" />
                    </svg>
                  </button>
                )}
              </div>
            </form>
          </div>

          <div className="chat-messages">
            {messages.map((msg) => (
              <div key={msg.id} className={`message-bubble ${msg.role}`}>
                {msg.role === 'assistant' ? (
                  <div dangerouslySetInnerHTML={{ __html: parseMarkdown(msg.content) }} />
                ) : (
                  msg.content
                )}
                {msg.role === 'assistant' && isDone && (
                  <div className="message-actions">
                    <button className="message-action-btn" onClick={() => void copyResult()}>
                      📋 复制
                    </button>
                    <button className="message-action-btn" onClick={downloadResult}>
                      ⬇ 下载
                    </button>
                  </div>
                )}
              </div>
            ))}

            {isRunning && (
              <div className="message-status">
                {hasActiveStage ? getCurrentStepName(run.assistant.stages) : '正在连接研究代理...'}
              </div>
            )}

            {run.isReviewing && (
              <div className="review-inline">
                <textarea
                  value={run.reviewInput}
                  rows={4}
                  placeholder="输入审核意见或补充要求"
                  onChange={(e) =>
                    setRun((current) =>
                      current ? { ...cloneRun(current), reviewInput: e.target.value } : current
                    )
                  }
                />
                <button
                  className="review-submit-btn"
                  type="button"
                  disabled={!run.reviewInput.trim() || run.isReviewSubmitting}
                  onClick={() => void submitReviewInput()}
                >
                  提交审核反馈
                </button>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </>
      )}

      <div className="chat-footer">
        {run && (
          <button className="footer-btn" title="新建任务" onClick={clearConversation}>
            🆕
          </button>
        )}
        <button className="footer-btn" title="设置" onClick={() => setSettingsOpen(true)}>
          ⚙️
        </button>
      </div>

      {settingsOpen && (
        <SettingsModal
          settings={taskSettings}
          onSave={(settings) => {
            setTaskSettings(settings)
            const validation = validateTaskSettings(settings)
            setErrors(validation.errors)
            setSettingsOpen(false)
          }}
          onClose={() => setSettingsOpen(false)}
        />
      )}
    </div>
  )
}
