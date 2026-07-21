import { describe, expect, it } from 'vitest'
import {
  applyBackendEvent,
  applyReviewSubmitFailure,
  applyReviewSubmitStart,
  createResearchRun,
  getAssistantPlainText,
  getStagePlaceholder,
  toAssistantUiMessages
} from './researchRun'

describe('research run reducer', () => {
  it('converts backend SSE events into expandable report stages', () => {
    const run = createResearchRun('user-1', 'assistant-1', '调研 RAG')

    applyBackendEvent(run, {
      step: 'searching',
      state: 'thinking',
      data: { thinking: 'search plan' }
    })
    applyBackendEvent(run, {
      step: 'searching',
      state: 'generating',
      data: 'found papers'
    })
    applyBackendEvent(run, {
      step: 'searching',
      state: 'completed',
      data: '\nsearch done'
    })

    expect(run.assistant.status).toBe('processing')
    expect(run.assistant.stages).toHaveLength(1)
    expect(run.assistant.stages[0]).toMatchObject({
      step: 'searching',
      status: 'done',
      title: '检索完成',
      thinking: 'search plan',
      content: 'found papers\nsearch done'
    })
  })

  it('marks the run as waiting for human review and submits review text into the stage', () => {
    const run = createResearchRun('user-1', 'assistant-1', '调研 Agent')

    applyBackendEvent(run, {
      step: 'searching',
      state: 'user_review',
      data: '请审核检索条件'
    })

    expect(run.isReviewing).toBe(true)
    expect(run.reviewInput).toBe('请审核检索条件')
    expect(run.assistant.statusText).toBe('等待人工审核')
    expect(run.assistant.stages[0].status).toBe('review')
  })

  it('builds assistant-ui compatible messages from the current run', () => {
    const run = createResearchRun('user-1', 'assistant-1', '调研多模态模型')
    applyBackendEvent(run, {
      step: 'writing',
      state: 'completed',
      data: '# 报告\n\n内容'
    })
    applyBackendEvent(run, { step: 'writing', state: 'finished', data: '' })

    expect(getAssistantPlainText(run.assistant)).toContain('# 报告')
    expect(toAssistantUiMessages(run)).toEqual([
      { id: 'user-1', role: 'user', content: '调研多模态模型' },
      { id: 'assistant-1', role: 'assistant', content: '# 报告\n\n内容' }
    ])
  })

  it('returns the same empty stage placeholder text as the Vue workflow', () => {
    expect(getStagePlaceholder('thinking')).toBe('正在组织调研思路...')
    expect(getStagePlaceholder('generating')).toBe('正在生成阶段内容...')
    expect(getStagePlaceholder('review')).toBe('等待人工审核反馈。')
    expect(getStagePlaceholder('error')).toBe('当前步骤暂无更多错误详情。')
    expect(getStagePlaceholder('queued')).toBe('等待后端返回内容。')
  })

  it('rolls back optimistic review submission when /send_input fails', () => {
    const run = createResearchRun('user-1', 'assistant-1', '调研 Agent')
    applyBackendEvent(run, {
      step: 'searching',
      state: 'user_review',
      data: '请审核检索条件'
    })
    const snapshot = applyReviewSubmitStart(run, '保留 query 条件')

    expect(run.isReviewing).toBe(false)
    expect(run.isReviewSubmitting).toBe(true)
    expect(run.reviewInput).toBe('')
    expect(run.assistant.stages[0]).toMatchObject({
      status: 'generating',
      title: '检索中'
    })
    expect(run.assistant.stages[0].content).toContain('人工反馈：保留 query 条件')

    applyReviewSubmitFailure(run, snapshot)

    expect(run.isReviewing).toBe(true)
    expect(run.isReviewSubmitting).toBe(false)
    expect(run.reviewInput).toBe('请审核检索条件')
    expect(run.assistant.status).toBe('review')
    expect(run.assistant.statusText).toBe('等待人工审核')
    expect(run.assistant.stages[0]).toMatchObject({
      content: '',
      status: 'review',
      title: '检索等待审核',
      expanded: true
    })
  })
})
