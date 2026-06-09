import { describe, expect, it } from 'vitest'
import { buildResearchQuery, createDefaultTaskSettings, validateTaskSettings } from './researchSettings'

describe('research task settings', () => {
  it('builds the backend query with all configured task constraints', () => {
    const settings = createDefaultTaskSettings()
    settings.yearStart = '2022'
    settings.yearEnd = '2026'
    settings.keywords = 'RAG; agent'
    settings.paperLimit = 12
    settings.outputLanguage = '英文'
    settings.customPrompt = 'Focus on limitations and benchmark comparison.'

    const query = buildResearchQuery('Survey autonomous research agents', settings)

    expect(query).toContain('Survey autonomous research agents')
    expect(query).toContain('年份范围：2022 到 2026')
    expect(query).toContain('论文检索关键词：RAG; agent')
    expect(query).toContain('获取论文结果最大数量：12')
    expect(query).toContain('输出语言：英文')
    expect(query).toContain('用户自定义约束提示词：Focus on limitations and benchmark comparison.')
  })

  it('rejects invalid year range, Chinese keyword separators, and out-of-range paper limit', () => {
    const settings = createDefaultTaskSettings()
    settings.yearStart = '2027'
    settings.yearEnd = '2024'
    settings.keywords = '大模型，检索'
    settings.paperLimit = 101

    const result = validateTaskSettings(settings, 2026)

    expect(result.valid).toBe(false)
    expect(result.errors.year).toBe('请输入 1900-2026 之间的四位年份。')
    expect(result.errors.keywords).toContain('英文分号')
    expect(result.errors.paperLimit).toContain('1-100')
  })
})
