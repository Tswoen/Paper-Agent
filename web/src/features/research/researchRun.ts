export type BackendState = 'initializing' | 'thinking' | 'generating' | 'user_review' | 'completed' | 'error' | 'finished' | 'failed'

export type BackendEvent = {
  step?: string
  state: BackendState | string
  data: unknown
}

export type StageStatus = 'queued' | 'preparing' | 'thinking' | 'generating' | 'review' | 'done' | 'error'
export type AssistantStatus = 'connecting' | 'processing' | 'review' | 'done' | 'error' | 'stopped'

export type ResearchStage = {
  id: string
  step: string
  title: string
  status: StageStatus
  thinking: string
  content: string
  timestamp: string
  expanded: boolean
  showThinking: boolean
}

export type ResearchMessage = {
  id: string
  role: 'user' | 'assistant'
  content?: string
  replyTo?: string
  status?: AssistantStatus
  statusText?: string
  stages?: ResearchStage[]
  finalContent?: string
  timestamp: string
  completedAt?: string | null
}

export type ResearchRun = {
  user: ResearchMessage
  assistant: ResearchMessage & { role: 'assistant'; stages: ResearchStage[] }
  isSubmitting: boolean
  isReviewing: boolean
  isReviewSubmitting: boolean
  reviewInput: string
  currentActiveStep: string | null
}

export type ReviewSubmitSnapshot = {
  reviewInput: string
  assistantStatus?: AssistantStatus
  assistantStatusText?: string
  stageId?: string
  stage?: Pick<ResearchStage, 'content' | 'status' | 'title' | 'expanded'>
}

export type AssistantUiMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
}

export const createId = (prefix: string): string => `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`

export const createResearchRun = (userId: string, assistantId: string, query: string): ResearchRun => {
  const timestamp = new Date().toISOString()
  return {
    user: {
      id: userId,
      role: 'user',
      content: query,
      timestamp
    },
    assistant: {
      id: assistantId,
      role: 'assistant',
      replyTo: userId,
      status: 'connecting',
      statusText: '正在连接研究代理',
      stages: [],
      finalContent: '',
      timestamp,
      completedAt: null
    },
    isSubmitting: true,
    isReviewing: false,
    isReviewSubmitting: false,
    reviewInput: '',
    currentActiveStep: null
  }
}

export const stringifyData = (data: unknown): string => {
  if (data === null || data === undefined) return ''
  if (typeof data === 'string') return data
  if (typeof data === 'object' && 'content' in data && typeof data.content === 'string') return data.content
  if (typeof data === 'object' && 'text' in data && typeof data.text === 'string') return data.text
  return JSON.stringify(data, null, 2)
}

export const extractThinking = (data: unknown): string => {
  if (!data || typeof data !== 'object') return ''
  return 'thinking' in data && typeof data.thinking === 'string' ? data.thinking : ''
}

export const extractContent = (data: unknown): string => {
  if (!data || typeof data !== 'object') return ''
  return 'content' in data && typeof data.content === 'string' ? data.content : ''
}

export const getStepName = (step = ''): string => {
  const stepNames: Record<string, string> = {
    searching: '检索',
    reading: '阅读',
    analyzing: '分析',
    writing: '撰写',
    writing_director: '撰写规划',
    section_writing: '小节撰写',
    reporting: '报告',
    connection: '连接',
    error: '错误'
  }

  if (step.startsWith('section_writing')) {
    const matched = step.match(/section_writing_?(\d+)?$/)
    return matched?.[1] ? `撰写第 ${matched[1]} 部分` : '小节撰写'
  }

  return stepNames[step] || step || '流程'
}

const createStage = (run: ResearchRun, step = '流程'): ResearchStage => {
  const stage: ResearchStage = {
    id: createId('stage'),
    step,
    title: `${getStepName(step)}排队中`,
    status: 'queued',
    thinking: '',
    content: '',
    timestamp: new Date().toISOString(),
    expanded: true,
    showThinking: false
  }
  run.assistant.stages.push(stage)
  run.currentActiveStep = step
  return stage
}

export const ensureStage = (run: ResearchRun, step = '流程'): ResearchStage => {
  const reusable = run.assistant.stages.find((stage) => stage.step === step && stage.status !== 'done')
  if (reusable) {
    run.currentActiveStep = step
    return reusable
  }
  return createStage(run, step)
}

const updateAssistantStatus = (run: ResearchRun, status: AssistantStatus, statusText: string) => {
  run.assistant.status = status
  run.assistant.statusText = statusText
}

export const applyBackendEvent = (run: ResearchRun, event: BackendEvent): void => {
  const step = event.step || '流程'
  const stepName = getStepName(step)

  if (event.state === 'initializing') {
    const stage = ensureStage(run, step)
    stage.status = 'preparing'
    stage.title = `${stepName}准备中`
    stage.thinking += extractThinking(event.data)
    stage.content += extractContent(event.data)
    updateAssistantStatus(run, 'processing', `${stepName}准备中`)
    return
  }

  if (event.state === 'thinking') {
    const stage = ensureStage(run, step)
    stage.status = 'thinking'
    stage.title = `${stepName}思考中`
    stage.thinking += extractThinking(event.data) || stringifyData(event.data)
    stage.showThinking = true
    updateAssistantStatus(run, 'processing', `${stepName}思考中`)
    return
  }

  if (event.state === 'generating') {
    const stage = ensureStage(run, step)
    stage.status = 'generating'
    stage.title = `${stepName}生成中`
    stage.content += stringifyData(event.data)
    stage.expanded = true
    updateAssistantStatus(run, 'processing', `${stepName}生成中`)
    return
  }

  if (event.state === 'user_review') {
    const stage = ensureStage(run, step)
    stage.status = 'review'
    stage.title = `${stepName}等待审核`
    stage.expanded = true
    run.reviewInput = stringifyData(event.data)
    run.isReviewing = true
    updateAssistantStatus(run, 'review', '等待人工审核')
    return
  }

  if (event.state === 'completed') {
    const stage = ensureStage(run, step)
    stage.status = 'done'
    stage.title = `${stepName}完成`
    stage.content += stringifyData(event.data)
    updateAssistantStatus(run, 'processing', `${stepName}完成`)
    return
  }

  if (event.state === 'error' || event.state === 'failed') {
    const stage = ensureStage(run, step || 'error')
    stage.status = 'error'
    stage.title = `${getStepName(step)}异常`
    stage.content += stringifyData(event.data) || '当前步骤处理异常。'
    updateAssistantStatus(run, 'error', '生成异常')
    run.isSubmitting = false
    run.isReviewing = false
    return
  }

  if (event.state === 'finished') {
    run.assistant.status = 'done'
    run.assistant.statusText = '调研报告生成完成'
    run.assistant.completedAt = new Date().toISOString()
    run.assistant.finalContent = getAssistantPlainText(run.assistant)
    run.isSubmitting = false
    run.isReviewing = false
    run.isReviewSubmitting = false
    run.currentActiveStep = null
  }
}

export const getStagePlaceholder = (status: StageStatus): string => {
  if (status === 'thinking') return '正在组织调研思路...'
  if (status === 'generating') return '正在生成阶段内容...'
  if (status === 'review') return '等待人工审核反馈。'
  if (status === 'error') return '当前步骤暂无更多错误详情。'
  return '等待后端返回内容。'
}

const getReviewStage = (run: ResearchRun): ResearchStage | undefined =>
  run.assistant.stages.find((stage) => stage.status === 'review') || run.assistant.stages.at(-1)

export const applyReviewSubmitStart = (run: ResearchRun, reviewText: string): ReviewSubmitSnapshot => {
  const reviewStage = getReviewStage(run)
  const snapshot: ReviewSubmitSnapshot = {
    reviewInput: run.reviewInput,
    assistantStatus: run.assistant.status,
    assistantStatusText: run.assistant.statusText,
    stageId: reviewStage?.id,
    stage: reviewStage
      ? {
          content: reviewStage.content,
          status: reviewStage.status,
          title: reviewStage.title,
          expanded: reviewStage.expanded
        }
      : undefined
  }

  run.isReviewSubmitting = true
  run.isReviewing = false
  run.reviewInput = ''

  if (reviewStage) {
    reviewStage.content += `\n\n人工反馈：${reviewText}\n`
    reviewStage.status = 'generating'
    reviewStage.title = `${getStepName(reviewStage.step)}中`
    reviewStage.expanded = true
  }

  updateAssistantStatus(run, 'processing', `${reviewStage ? getStepName(reviewStage.step) : '流程'}中`)
  return snapshot
}

export const applyReviewSubmitFailure = (run: ResearchRun, snapshot: ReviewSubmitSnapshot): void => {
  const reviewStage = snapshot.stageId ? run.assistant.stages.find((stage) => stage.id === snapshot.stageId) : undefined
  if (reviewStage && snapshot.stage) {
    reviewStage.content = snapshot.stage.content
    reviewStage.status = snapshot.stage.status
    reviewStage.title = snapshot.stage.title
    reviewStage.expanded = snapshot.stage.expanded
  }

  run.reviewInput = snapshot.reviewInput
  run.isReviewing = true
  run.isReviewSubmitting = false
  run.assistant.status = 'review'
  run.assistant.statusText = '等待人工审核'
}

export const getAssistantPlainText = (message: ResearchRun['assistant']): string => {
  const stageContent = message.stages
    .map((stage) => stage.content?.trim())
    .filter(Boolean)
    .join('\n\n')
  return message.finalContent || stageContent
}

export const toAssistantUiMessages = (run: ResearchRun): AssistantUiMessage[] => [
  { id: run.user.id, role: 'user', content: run.user.content || '' },
  { id: run.assistant.id, role: 'assistant', content: getAssistantPlainText(run.assistant) }
]

export const createHistoryTitle = (query: string): string => (query.length > 28 ? `${query.slice(0, 28)}...` : query)
