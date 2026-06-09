import DOMPurify from 'dompurify'
import { marked } from 'marked'
import { type FormEvent, type KeyboardEvent, type SyntheticEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AssistantRuntimeBridge } from './AssistantRuntimeBridge'
import {
  applyBackendEvent,
  applyReviewSubmitFailure,
  applyReviewSubmitStart,
  createHistoryTitle,
  createId,
  createResearchRun,
  getAssistantPlainText,
  getStagePlaceholder,
  type BackendEvent,
  type ResearchRun,
  toAssistantUiMessages
} from './researchRun'
import {
  buildResearchQuery,
  createDefaultTaskSettings,
  summarizeConstraints,
  type TaskSettings,
  type TaskSettingsErrors,
  validateTaskSettings
} from './researchSettings'

const promptExamples = [
  '调研多模态大模型在医学影像诊断中的研究进展',
  '生成一份 AI Agent 评测方法的学术综述',
  '比较近三年 RAG 与长上下文模型在论文写作中的应用',
  '分析自动化科研助手的发展趋势、局限性和未来方向'
]

const emptyErrors: TaskSettingsErrors = {
  year: '',
  keywords: '',
  paperLimit: '',
  customPrompt: ''
}

const cloneRun = (run: ResearchRun): ResearchRun => ({
  ...run,
  user: { ...run.user },
  assistant: {
    ...run.assistant,
    stages: run.assistant.stages.map((stage) => ({ ...stage }))
  }
})

const parseMarkdown = (content: string) => DOMPurify.sanitize(marked.parse(content) as string)

const getStageStateText = (status: string) => {
  const statusMap: Record<string, string> = {
    queued: '排队',
    preparing: '准备',
    thinking: '思考',
    generating: '生成',
    review: '审核',
    done: '完成',
    error: '异常'
  }
  return statusMap[status] || '运行'
}

const getAssistantStatusText = (status?: string, statusText?: string) => {
  if (statusText) return statusText
  const statusMap: Record<string, string> = {
    connecting: '正在连接研究代理',
    processing: '正在生成',
    review: '等待审核',
    done: '调研报告生成完成',
    error: '生成异常',
    stopped: '已停止'
  }
  return statusMap[status || ''] || '运行中'
}

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
    content
  }

  try {
    const saved = JSON.parse(localStorage.getItem('reportHistory') || '[]')
    const nextHistory = [item, ...saved.filter((record: { id: string }) => record.id !== item.id)].slice(0, 50)
    localStorage.setItem('reportHistory', JSON.stringify(nextHistory))
  } catch (error) {
    console.error('保存历史记录失败:', error)
  }
}

export function ResearchPage() {
  const [userInput, setUserInput] = useState('')
  const [taskSettings, setTaskSettings] = useState<TaskSettings>(() => createDefaultTaskSettings())
  const [errors, setErrors] = useState<TaskSettingsErrors>(emptyErrors)
  const [run, setRun] = useState<ResearchRun | null>(null)
  const [rightPanelCollapsed, setRightPanelCollapsed] = useState(false)
  const [copiedMessageId, setCopiedMessageId] = useState('')
  const [activeTooltip, setActiveTooltip] = useState({ text: '', left: 0, top: 0, transform: 'translate(-50%, -100%)' })
  const sourceRef = useRef<EventSource | null>(null)
  const messagesContainerRef = useRef<HTMLElement | null>(null)
  const composerInputRef = useRef<HTMLTextAreaElement | null>(null)

  const closeEventSource = useCallback(() => {
    sourceRef.current?.close()
    sourceRef.current = null
  }, [])

  const autoScroll = useCallback(() => {
    window.requestAnimationFrame(() => {
      const container = messagesContainerRef.current
      if (!container) return
      container.scrollTo({
        top: container.scrollHeight,
        behavior: 'smooth'
      })
    })
  }, [])

  useEffect(() => {
    autoScroll()
  }, [autoScroll, run])

  useEffect(() => {
    composerInputRef.current?.focus()
  }, [run?.assistant.id])

  useEffect(() => () => closeEventSource(), [closeEventSource])

  const finishRun = useCallback(() => {
    closeEventSource()
    setRun((current) => (current ? { ...cloneRun(current), isSubmitting: false, isReviewing: false, isReviewSubmitting: false } : current))
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
      window.requestAnimationFrame(() => composerInputRef.current?.focus())

      const source = new EventSource(`/api/research?query=${encodeURIComponent(buildResearchQuery(query, taskSettings))}`)
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
            window.requestAnimationFrame(autoScroll)
            return draft
          })
        } catch (error) {
          console.error('处理流式数据失败:', error)
        }
      }

      source.onerror = () => {
        setRun((current) => {
          if (!current?.isSubmitting) return current
          const draft = cloneRun(current)
          applyBackendEvent(draft, {
            step: 'connection',
            state: 'error',
            data: '连接后端服务失败，请确认服务已经启动。'
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
    if (run?.isSubmitting) {
      void stopCurrentRun()
    }
    setRun(null)
    setCopiedMessageId('')
    window.requestAnimationFrame(() => composerInputRef.current?.focus())
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
        body: JSON.stringify({ input: reviewText })
      })
      if (!response.ok) throw new Error(`审核提交失败: ${response.status}`)
    } catch (error) {
      console.error(error)
      setRun((current) => {
        if (!current) return current
        const draft = cloneRun(current)
        if (snapshot) {
          applyReviewSubmitFailure(draft, snapshot)
        }
        return draft
      })
      window.alert('提交审核反馈失败，请检查后端服务。')
      return
    }

    setRun((current) => (current ? { ...cloneRun(current), isReviewSubmitting: false } : current))
  }

  const copyAssistantOutput = async () => {
    if (!run) return
    const content = getAssistantPlainText(run.assistant)
    if (!content) return
    await navigator.clipboard.writeText(content)
    setCopiedMessageId(run.assistant.id)
    window.setTimeout(() => setCopiedMessageId(''), 1600)
  }

  const messages = useMemo(() => (run ? toAssistantUiMessages(run) : []), [run])
  const statusText = run?.isReviewing ? '等待人工审核' : run?.isSubmitting ? '报告生成中' : run ? '可继续补充' : '准备就绪'

  const formatTime = (timestamp?: string) => {
    if (!timestamp) return ''
    return new Date(timestamp).toLocaleTimeString('zh-CN', {
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const handleComposerKeydown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      void startResearch(userInput)
    }
  }

  const showTooltip = (trigger: HTMLElement, text: string) => {
    const rect = trigger.getBoundingClientRect()
    const estimatedWidth = Math.min(280, Math.max(200, window.innerWidth - 24))
    const center = rect.left + rect.width / 2
    const left = Math.min(Math.max(center, estimatedWidth / 2 + 12), window.innerWidth - estimatedWidth / 2 - 12)
    const shouldShowBelow = rect.top < 86
    setActiveTooltip({
      text,
      left,
      top: shouldShowBelow ? rect.bottom + 10 : rect.top - 10,
      transform: shouldShowBelow ? 'translate(-50%, 0)' : 'translate(-50%, -100%)'
    })
  }

  const handleTooltipEnter = (event: SyntheticEvent<HTMLElement>) => {
    const target = event.target
    if (!(target instanceof Element)) {
      setActiveTooltip((current) => ({ ...current, text: '' }))
      return
    }

    const trigger = target.closest('.help-tip')
    if (!(trigger instanceof HTMLElement) || !trigger.dataset.tooltip) {
      if (event.type === 'mouseover') setActiveTooltip((current) => ({ ...current, text: '' }))
      return
    }
    showTooltip(trigger, trigger.dataset.tooltip)
  }

  const hideTooltip = () => setActiveTooltip((current) => ({ ...current, text: '' }))

  return (
    <AssistantRuntimeBridge
      messages={messages}
      isRunning={Boolean(run?.isSubmitting)}
      isSendDisabled={Boolean(run?.isSubmitting)}
      onUserMessage={startResearch}
      onCancel={stopCurrentRun}
    >
      <div
        className={`report-workbench ${rightPanelCollapsed ? 'right-collapsed' : ''}`}
        onFocus={handleTooltipEnter}
        onBlur={hideTooltip}
        onMouseLeave={hideTooltip}
        onMouseOver={handleTooltipEnter}
        onScrollCapture={hideTooltip}
      >
        <section className="report-center">
          <header className="report-header">
            <div>
              <p className="eyebrow">Research Report Workflow</p>
              <h1>学术调研报告生成</h1>
            </div>
            <div className="header-actions">
              <span className={`status-pill ${run?.isSubmitting ? 'active' : ''} ${run?.isReviewing ? 'review' : ''}`}>
                <span />
                {statusText}
              </span>
              <button className="secondary-button" type="button" onClick={clearConversation}>
                新建任务
              </button>
              <button className="icon-button" type="button" onClick={() => setRightPanelCollapsed((value) => !value)}>
                {rightPanelCollapsed ? '展开约束' : '收起约束'}
              </button>
            </div>
          </header>

          <main ref={messagesContainerRef} className="report-main">
            {!run ? (
              <section className="start-panel">
                <div className="start-copy">
                  <h2>输入研究主题，生成结构化调研报告</h2>
                </div>
                <form className="topic-card" onSubmit={submitRequest}>
                  <label htmlFor="research-topic">研究主题或任务需求</label>
                  <textarea
                    id="research-topic"
                    ref={composerInputRef}
                    value={userInput}
                    rows={5}
                    placeholder="例如：调研多模态大模型在医学影像诊断中的研究进展，重点比较近三年的代表性方法、数据集、局限性和未来方向。"
                    onChange={(event) => setUserInput(event.target.value)}
                    onKeyDown={handleComposerKeydown}
                  />
                  <div className="topic-actions">
                    <div className="constraint-summary">{summarizeConstraints(taskSettings)}</div>
                    <button className="primary-button" type="submit" disabled={!userInput.trim()}>
                      开始生成报告
                    </button>
                  </div>
                </form>
                <div className="prompt-row">
                  {promptExamples.map((prompt) => (
                    <button key={prompt} type="button" onClick={() => setUserInput(prompt)}>
                      {prompt}
                    </button>
                  ))}
                </div>
              </section>
            ) : (
              <section className="run-panel">
                <article className="message-row user">
                  <div className="message-label">
                    研究主题 <span>{formatTime(run.user.timestamp)}</span>
                  </div>
                  <div className="user-request">{run.user.content}</div>
                </article>
                <article className="message-row assistant">
                  <div className="message-label">
                    生成过程 <span>{formatTime(run.assistant.timestamp)}</span>
                  </div>
                  <div className="assistant-run-card">
                    <div className={`assistant-status ${run.assistant.status}`}>
                      <span className="status-dot" />
                      <strong>{getAssistantStatusText(run.assistant.status, run.assistant.statusText)}</strong>
                    </div>
                    {run.assistant.stages.length > 0 ? (
                      <div className="stage-list">
                        {run.assistant.stages.map((stage, index) => (
                          <section key={stage.id} className={`stage-card ${stage.status}`}>
                            <button
                              className="stage-header"
                              type="button"
                              onClick={() => {
                                setRun((current) => {
                                  if (!current) return current
                                  const draft = cloneRun(current)
                                  const target = draft.assistant.stages.find((item) => item.id === stage.id)
                                  if (target) target.expanded = !target.expanded
                                  return draft
                                })
                              }}
                            >
                              <span className="stage-index">{index + 1}</span>
                              <span className="stage-title">{stage.title}</span>
                              <span className={`stage-state ${stage.status}`}>{getStageStateText(stage.status)}</span>
                            </button>
                            {stage.expanded && (
                              <div className="stage-content">
                                {stage.thinking && (
                                  <div className="thinking-box">
                                    <button
                                      type="button"
                                      onClick={() => {
                                        setRun((current) => {
                                          if (!current) return current
                                          const draft = cloneRun(current)
                                          const target = draft.assistant.stages.find((item) => item.id === stage.id)
                                          if (target) target.showThinking = !target.showThinking
                                          return draft
                                        })
                                      }}
                                    >
                                      <span>思考过程</span>
                                      <span>{stage.showThinking ? '收起' : '展开'}</span>
                                    </button>
                                    {stage.showThinking && <pre>{stage.thinking}</pre>}
                                  </div>
                                )}
                                {stage.content ? (
                                  <div className="markdown-body" dangerouslySetInnerHTML={{ __html: parseMarkdown(stage.content) }} />
                                ) : (
                                  <p className="muted-line">{getStagePlaceholder(stage.status)}</p>
                                )}
                              </div>
                            )}
                          </section>
                        ))}
                      </div>
                    ) : (
                      <div className="assistant-loading">
                        <span />
                        <span />
                        <span />
                      </div>
                    )}

                    {run.isReviewing && (
                      <div className="review-panel">
                        <div className="review-title">
                          <strong>人工审核节点</strong>
                          <span>后端流程正在等待人工输入</span>
                        </div>
                        <textarea
                          value={run.reviewInput}
                          rows={4}
                          placeholder="输入审核意见或补充要求"
                          onChange={(event) => setRun((current) => (current ? { ...cloneRun(current), reviewInput: event.target.value } : current))}
                        />
                        <button type="button" disabled={!run.reviewInput.trim() || run.isReviewSubmitting} onClick={() => void submitReviewInput()}>
                          提交审核反馈
                        </button>
                      </div>
                    )}

                    {run.assistant.status === 'done' && (
                      <div className="message-actions">
                        <button type="button" onClick={() => void copyAssistantOutput()}>
                          {copiedMessageId === run.assistant.id ? '已复制' : '复制报告内容'}
                        </button>
                      </div>
                    )}
                  </div>
                </article>
              </section>
            )}
          </main>

          {run && (
            <footer className="continue-composer">
              <form onSubmit={submitRequest}>
                <textarea
                  ref={composerInputRef}
                  value={userInput}
                  rows={1}
                  placeholder="继续补充研究要求，或要求调整报告侧重点"
                  disabled={run.isSubmitting}
                  onChange={(event) => setUserInput(event.target.value)}
                  onKeyDown={handleComposerKeydown}
                />
                {run.isSubmitting && (
                  <button className="danger-button" type="button" onClick={() => void stopCurrentRun()}>
                    停止
                  </button>
                )}
                <button className="primary-button" type="submit" disabled={!userInput.trim() || run.isSubmitting}>
                  继续生成
                </button>
              </form>
            </footer>
          )}
        </section>

        <TaskPanel
          collapsed={rightPanelCollapsed}
          settings={taskSettings}
          errors={errors}
          onExpand={() => setRightPanelCollapsed(false)}
          onCollapse={() => setRightPanelCollapsed(true)}
          onChange={(next) => {
            setTaskSettings(next)
            setErrors((current) => ({
              year: next.yearStart !== taskSettings.yearStart || next.yearEnd !== taskSettings.yearEnd ? '' : current.year,
              keywords: next.keywords !== taskSettings.keywords ? '' : current.keywords,
              paperLimit: next.paperLimit !== taskSettings.paperLimit ? '' : current.paperLimit,
              customPrompt: next.customPrompt !== taskSettings.customPrompt ? '' : current.customPrompt
            }))
          }}
          onValidate={(next) => setErrors(validateTaskSettings(next).errors)}
        />
        {activeTooltip.text && (
          <div
            className="global-tooltip"
            style={{
              left: `${activeTooltip.left}px`,
              top: `${activeTooltip.top}px`,
              transform: activeTooltip.transform
            }}
            role="tooltip"
          >
            {activeTooltip.text}
          </div>
        )}
      </div>
    </AssistantRuntimeBridge>
  )
}

function TaskPanel({
  collapsed,
  settings,
  errors,
  onExpand,
  onCollapse,
  onChange,
  onValidate
}: {
  collapsed: boolean
  settings: TaskSettings
  errors: TaskSettingsErrors
  onExpand: () => void
  onCollapse: () => void
  onChange: (settings: TaskSettings) => void
  onValidate: (settings: TaskSettings) => void
}) {
  const patch = (changes: Partial<TaskSettings>) => onChange({ ...settings, ...changes })

  return (
    <aside className={`task-panel ${collapsed ? 'collapsed' : ''}`} aria-label="当前任务约束配置">
      {collapsed ? (
        <button className="expand-task-button" type="button" onClick={onExpand}>
          约束
        </button>
      ) : (
        <>
          <header className="task-panel-header">
            <div>
              <p className="eyebrow">Task Constraints</p>
              <h2>任务约束</h2>
            </div>
            <button className="icon-only-button" type="button" title="收起右侧栏" onClick={onCollapse}>
              ›
            </button>
          </header>
          <div className="task-panel-body">
            <p className="task-panel-note">这些设置只作用于本次调研任务：年份、关键词与论文数量影响检索阶段；输出语言与自定义约束提示词影响生成阶段。</p>
            <section className={`constraint-card ${errors.year ? 'invalid' : ''}`}>
              <div className="constraint-title">
                <h3>年份范围</h3>
                <button className="help-tip" type="button" data-tooltip="用于限制论文检索的时间范围，会直接影响搜索结果。" aria-label="年份范围说明">
                  ?
                </button>
              </div>
              <div className="field-pair">
                <label>
                  <span>起始年份</span>
                  <input value={settings.yearStart} type="number" min="1900" placeholder="2021" onBlur={() => onValidate(settings)} onChange={(event) => patch({ yearStart: event.target.value })} />
                </label>
                <label>
                  <span>结束年份</span>
                  <input value={settings.yearEnd} type="number" min="1900" placeholder="2026" onBlur={() => onValidate(settings)} onChange={(event) => patch({ yearEnd: event.target.value })} />
                </label>
              </div>
              {errors.year && <p className="field-error">{errors.year}</p>}
            </section>
            <section className={`constraint-card ${errors.keywords ? 'invalid' : ''}`}>
              <div className="constraint-title">
                <h3>关键词</h3>
                <button className="help-tip" type="button" data-tooltip="该字段会参与论文检索语句构造，请用分号分隔多个关键词。" aria-label="关键词说明">
                  ?
                </button>
              </div>
              <textarea value={settings.keywords} rows={3} placeholder="示例：large language model; agent; reasoning" onBlur={() => onValidate(settings)} onChange={(event) => patch({ keywords: event.target.value })} />
              {errors.keywords && <p className="field-error">{errors.keywords}</p>}
            </section>
            <section className={`constraint-card ${errors.paperLimit ? 'invalid' : ''}`}>
              <div className="constraint-title">
                <h3>论文数量</h3>
                <button className="help-tip" type="button" data-tooltip="这是检索论文时获取结果的最大数量，会影响检索规模与后续处理开销。" aria-label="论文数量说明">
                  ?
                </button>
              </div>
              <label>
                <span>检索论文上限</span>
                <input value={settings.paperLimit} type="number" min="1" max="100" step="1" onBlur={() => onValidate(settings)} onChange={(event) => patch({ paperLimit: Number(event.target.value) })} />
              </label>
              {errors.paperLimit && <p className="field-error">{errors.paperLimit}</p>}
            </section>
            <section className="constraint-card">
              <div className="constraint-title">
                <h3>输出语言</h3>
                <button className="help-tip" type="button" data-tooltip="控制报告生成语言，只允许中文或英文，不支持自由输入。" aria-label="输出语言说明">
                  ?
                </button>
              </div>
              <select value={settings.outputLanguage} onChange={(event) => patch({ outputLanguage: event.target.value as TaskSettings['outputLanguage'] })}>
                <option>中文</option>
                <option>英文</option>
              </select>
            </section>
            <section className={`constraint-card ${errors.customPrompt ? 'invalid' : ''}`}>
              <div className="constraint-title">
                <h3>用户自定义约束提示词</h3>
                <button className="help-tip" type="button" data-tooltip="这部分会作为智能体系统提示词的一部分参与报告生成，适合填写高级写作与分析约束。" aria-label="用户自定义约束提示词说明">
                  ?
                </button>
              </div>
              <textarea
                value={settings.customPrompt}
                rows={7}
                placeholder="例如：强调方法对比；更关注局限性总结；写作风格偏学术综述。"
                onBlur={() => onValidate(settings)}
                onChange={(event) => patch({ customPrompt: event.target.value })}
              />
              {errors.customPrompt && <p className="field-error">{errors.customPrompt}</p>}
            </section>
          </div>
        </>
      )}
    </aside>
  )
}
