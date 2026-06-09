export type OutputLanguage = '中文' | '英文'

export type TaskSettings = {
  yearStart: string
  yearEnd: string
  keywords: string
  paperLimit: number
  outputLanguage: OutputLanguage
  customPrompt: string
}

export type TaskSettingsErrors = {
  year: string
  keywords: string
  paperLimit: string
  customPrompt: string
}

export const createDefaultTaskSettings = (): TaskSettings => ({
  yearStart: '2021',
  yearEnd: '2026',
  keywords: '',
  paperLimit: 20,
  outputLanguage: '中文',
  customPrompt: ''
})

const emptyErrors = (): TaskSettingsErrors => ({
  year: '',
  keywords: '',
  paperLimit: '',
  customPrompt: ''
})

const normalizeYear = (value: unknown, currentYear: number): number | null => {
  const text = String(value ?? '').trim()
  if (!text) return null
  const year = Number(text)
  if (!Number.isInteger(year) || year < 1900 || year > currentYear) return null
  return year
}

export const validateTaskSettings = (settings: TaskSettings, currentYear = new Date().getFullYear()) => {
  const errors = emptyErrors()
  const start = normalizeYear(settings.yearStart, currentYear)
  const end = normalizeYear(settings.yearEnd, currentYear)

  if ((settings.yearStart && !start) || (settings.yearEnd && !end)) {
    errors.year = `请输入 1900-${currentYear} 之间的四位年份。`
  } else if (start && end && start > end) {
    errors.year = '起始年份不能晚于结束年份。'
  }

  const keywords = settings.keywords.trim()
  if (keywords) {
    if (/[，,、；]/.test(keywords)) {
      errors.keywords = '请使用英文分号分隔多个关键词，例如：large language model; agent; reasoning。'
    } else if (keywords.includes(';') && keywords.split(';').some((keyword) => !keyword.trim())) {
      errors.keywords = '分号前后都需要是有效关键词，请删除多余的分号。'
    }
  }

  const rawPaperLimit = String(settings.paperLimit ?? '').trim()
  const paperLimit = Number(rawPaperLimit)
  if (!rawPaperLimit) {
    errors.paperLimit = '请填写检索论文上限，建议输入 5-50 之间的整数。'
  } else if (!Number.isInteger(paperLimit)) {
    errors.paperLimit = '论文数量必须是整数，例如 20。'
  } else if (paperLimit < 1 || paperLimit > 100) {
    errors.paperLimit = '论文数量范围为 1-100，避免检索规模过小或处理开销过高。'
  }

  const customPrompt = settings.customPrompt.trim()
  if (customPrompt && customPrompt.length < 6) {
    errors.customPrompt = '自定义约束过短，请写成可执行的生成约束，例如“强调方法对比”。'
  } else if (customPrompt.length > 500) {
    errors.customPrompt = '自定义约束建议控制在 500 字以内，避免覆盖主要研究主题。'
  }

  return {
    valid: !errors.year && !errors.keywords && !errors.paperLimit && !errors.customPrompt,
    errors
  }
}

export const buildResearchQuery = (query: string, settings: TaskSettings): string => {
  const constraints: string[] = []
  if (settings.yearStart || settings.yearEnd) {
    constraints.push(`检索阶段 - 年份范围：${settings.yearStart || '不限'} 到 ${settings.yearEnd || '不限'}，用于限制论文搜索结果`)
  }
  if (settings.keywords.trim()) {
    constraints.push(`检索阶段 - 论文检索关键词：${settings.keywords.trim()}（按分号分隔）`)
  }
  constraints.push(`检索阶段 - 获取论文结果最大数量：${settings.paperLimit}`)
  constraints.push(`生成阶段 - 输出语言：${settings.outputLanguage}`)
  if (settings.customPrompt.trim()) {
    constraints.push(`生成阶段 - 用户自定义约束提示词：${settings.customPrompt.trim()}。该内容作为智能体系统提示词的一部分参与分析、写作和报告组织。`)
  }

  return `${query}\n\n本次调研报告生成约束：\n${constraints.map((item) => `- ${item}`).join('\n')}`
}

export const summarizeConstraints = (settings: TaskSettings): string => {
  const years = settings.yearStart || settings.yearEnd ? `${settings.yearStart || '不限'}-${settings.yearEnd || '不限'}` : '年份不限'
  const keywords = settings.keywords.trim() ? `，关键词 ${settings.keywords.trim()}` : ''
  return `约束：${years}${keywords}，检索上限 ${settings.paperLimit} 篇，${settings.outputLanguage}`
}
