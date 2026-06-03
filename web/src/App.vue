<template>
  <div
    class="report-workbench"
    :class="{ 'right-collapsed': rightPanelCollapsed }"
    @focusin="handleTooltipEnter"
    @focusout="hideTooltip"
    @mouseleave="hideTooltip"
    @mouseover="handleTooltipEnter"
    @scroll.capture="hideTooltip"
  >
    <section class="report-center">
      <header class="report-header">
        <div>
          <p class="eyebrow">Research Report Workflow</p>
          <h1>学术调研报告生成</h1>
        </div>
        <div class="header-actions">
          <span class="status-pill" :class="{ active: isSubmitting, review: isReviewing }">
            <span></span>
            {{ runStatusText }}
          </span>
          <button class="secondary-button" type="button" @click="clearConversation">
            新建任务
          </button>
          <button class="icon-button" type="button" @click="rightPanelCollapsed = !rightPanelCollapsed">
            {{ rightPanelCollapsed ? '展开约束' : '收起约束' }}
          </button>
        </div>
      </header>

      <main ref="messagesContainer" class="report-main">
        <section v-if="conversation.length === 0" class="start-panel">
          <div class="start-copy">
            <h2>输入研究主题，生成结构化调研报告</h2>
          </div>

          <form class="topic-card" @submit.prevent="submitRequest">
            <label for="research-topic">研究主题或任务需求</label>
            <textarea
              id="research-topic"
              ref="composerInput"
              v-model="userInput"
              rows="5"
              placeholder="例如：调研多模态大模型在医学影像诊断中的研究进展，重点比较近三年的代表性方法、数据集、局限性和未来方向。"
              :disabled="isSubmitting"
              @keydown="handleComposerKeydown"
            ></textarea>
            <div class="topic-actions">
              <div class="constraint-summary">
                {{ constraintSummary }}
              </div>
              <button class="primary-button" type="submit" :disabled="!userInput.trim() || isSubmitting">
                开始生成报告
              </button>
            </div>
          </form>

          <div class="prompt-row">
            <button
              v-for="prompt in promptExamples"
              :key="prompt"
              type="button"
              @click="selectPrompt(prompt)"
            >
              {{ prompt }}
            </button>
          </div>
        </section>

        <section v-else class="run-panel">
          <article
            v-for="message in conversation"
            :key="message.id"
            class="message-row"
            :class="message.role"
          >
            <div class="message-label">
              {{ message.role === 'user' ? '研究主题' : '生成过程' }}
              <span>{{ formatTime(message.timestamp) }}</span>
            </div>

            <div v-if="message.role === 'user'" class="user-request">
              {{ message.content }}
            </div>

            <div v-else class="assistant-run-card">
              <div class="assistant-status" :class="message.status">
                <span class="status-dot"></span>
                <strong>{{ getAssistantStatusText(message) }}</strong>
              </div>

              <div v-if="message.stages.length > 0" class="stage-list">
                <section
                  v-for="(stage, index) in message.stages"
                  :key="stage.id"
                  class="stage-card"
                  :class="stage.status"
                >
                  <button class="stage-header" type="button" @click="stage.expanded = !stage.expanded">
                    <span class="stage-index">{{ index + 1 }}</span>
                    <span class="stage-title">{{ stage.title }}</span>
                    <span class="stage-state" :class="stage.status">{{ getStageStateText(stage.status) }}</span>
                  </button>

                  <div v-show="stage.expanded" class="stage-content">
                    <div v-if="stage.thinking" class="thinking-box">
                      <button type="button" @click="stage.showThinking = !stage.showThinking">
                        <span>思考过程</span>
                        <span>{{ stage.showThinking ? '收起' : '展开' }}</span>
                      </button>
                      <pre v-show="stage.showThinking">{{ stage.thinking }}</pre>
                    </div>

                    <div
                      v-if="stage.content"
                      class="markdown-body"
                      v-html="parseMarkdown(stage.content)"
                    ></div>
                    <p v-else class="muted-line">{{ getStagePlaceholder(stage.status) }}</p>
                  </div>
                </section>
              </div>

              <div v-else class="assistant-loading">
                <span></span>
                <span></span>
                <span></span>
              </div>

              <div
                v-if="isReviewing && activeAssistantId === message.id"
                class="review-panel"
              >
                <div class="review-title">
                  <strong>人工审核节点</strong>
                  <span>后端流程正在等待人工输入</span>
                </div>
                <textarea
                  v-model="userReviewInput"
                  rows="4"
                  placeholder="输入审核意见或补充要求"
                ></textarea>
                <button type="button" :disabled="!userReviewInput.trim()" @click="submitReviewInput">
                  提交审核反馈
                </button>
              </div>

              <div v-if="message.status === 'done'" class="message-actions">
                <button type="button" @click="copyAssistantOutput(message)">
                  {{ copiedMessageId === message.id ? '已复制' : '复制报告内容' }}
                </button>
              </div>
            </div>
          </article>
        </section>
      </main>

      <footer v-if="conversation.length > 0" class="continue-composer">
        <form @submit.prevent="submitRequest">
          <textarea
            ref="composerInput"
            v-model="userInput"
            rows="1"
            placeholder="继续补充研究要求，或要求调整报告侧重点"
            :disabled="isSubmitting"
            @keydown="handleComposerKeydown"
          ></textarea>
          <button v-if="isSubmitting" class="danger-button" type="button" @click="stopCurrentRun">
            停止
          </button>
          <button class="primary-button" type="submit" :disabled="!userInput.trim() || isSubmitting">
            继续生成
          </button>
        </form>
      </footer>
    </section>

    <aside class="task-panel" :class="{ collapsed: rightPanelCollapsed }" aria-label="当前任务约束配置">
      <button
        v-if="rightPanelCollapsed"
        class="expand-task-button"
        type="button"
        @click="rightPanelCollapsed = false"
      >
        约束
      </button>

      <template v-else>
        <header class="task-panel-header">
          <div>
            <p class="eyebrow">Task Constraints</p>
            <h2>任务约束</h2>
          </div>
          <button class="icon-only-button" type="button" title="收起右侧栏" @click="rightPanelCollapsed = true">
            ›
          </button>
        </header>

        <div class="task-panel-body">
          <p class="task-panel-note">
            这些设置只作用于本次调研任务：年份、关键词与论文数量影响检索阶段；输出语言与自定义约束提示词影响生成阶段。
          </p>

          <section class="constraint-card" :class="{ invalid: yearError }">
            <div class="constraint-title">
              <h3>年份范围</h3>
              <button
                class="help-tip"
                type="button"
                data-tooltip="用于限制论文检索的时间范围，会直接影响搜索结果。"
                aria-label="年份范围说明"
              >
                ?
              </button>
            </div>
            <div class="field-pair">
              <label>
                <span>起始年份</span>
                <input
                  v-model="taskSettings.yearStart"
                  type="number"
                  inputmode="numeric"
                  min="1900"
                  :max="currentYear"
                  placeholder="2021"
                  @blur="validateYearRange"
                  @input="clearYearError"
                >
              </label>
              <label>
                <span>结束年份</span>
                <input
                  v-model="taskSettings.yearEnd"
                  type="number"
                  inputmode="numeric"
                  min="1900"
                  :max="currentYear"
                  placeholder="2026"
                  @blur="validateYearRange"
                  @input="clearYearError"
                >
              </label>
            </div>
            <p v-if="yearError" class="field-error">{{ yearError }}</p>
          </section>

          <section class="constraint-card" :class="{ invalid: keywordError }">
            <div class="constraint-title">
              <h3>关键词</h3>
              <button
                class="help-tip"
                type="button"
                data-tooltip="该字段会参与论文检索语句构造，请用分号分隔多个关键词。"
                aria-label="关键词说明"
              >
                ?
              </button>
            </div>
            <textarea
              v-model="taskSettings.keywords"
              rows="3"
              placeholder="示例：large language model; agent; reasoning"
              @blur="validateKeywords"
              @input="clearKeywordError"
            ></textarea>
            <p v-if="keywordError" class="field-error">{{ keywordError }}</p>
          </section>

          <section class="constraint-card" :class="{ invalid: paperLimitError }">
            <div class="constraint-title">
              <h3>论文数量</h3>
              <button
                class="help-tip"
                type="button"
                data-tooltip="这是检索论文时获取结果的最大数量，会影响检索规模与后续处理开销。"
                aria-label="论文数量说明"
              >
                ?
              </button>
            </div>
            <label>
              <span>检索论文上限</span>
              <input
                v-model.number="taskSettings.paperLimit"
                type="number"
                min="1"
                max="100"
                step="1"
                @blur="validatePaperLimit"
                @input="clearPaperLimitError"
              >
            </label>
            <p v-if="paperLimitError" class="field-error">{{ paperLimitError }}</p>
          </section>

          <section class="constraint-card">
            <div class="constraint-title">
              <h3>输出语言</h3>
              <button
                class="help-tip"
                type="button"
                data-tooltip="控制报告生成语言，只允许中文或英文，不支持自由输入。"
                aria-label="输出语言说明"
              >
                ?
              </button>
            </div>
            <select v-model="taskSettings.outputLanguage">
              <option>中文</option>
              <option>英文</option>
            </select>
          </section>

          <section class="constraint-card" :class="{ invalid: customPromptError }">
            <div class="constraint-title">
              <h3>用户自定义约束提示词</h3>
              <button
                class="help-tip"
                type="button"
                data-tooltip="这部分会作为智能体系统提示词的一部分参与报告生成，适合填写高级写作与分析约束。"
                aria-label="用户自定义约束提示词说明"
              >
                ?
              </button>
            </div>
            <textarea
              v-model="taskSettings.customPrompt"
              rows="7"
              placeholder="例如：强调方法对比；更关注局限性总结；写作风格偏学术综述。"
              @blur="validateCustomPrompt"
              @input="clearCustomPromptError"
            ></textarea>
            <p v-if="customPromptError" class="field-error">{{ customPromptError }}</p>
          </section>
        </div>
      </template>
    </aside>

    <Teleport to="body">
      <div
        v-if="activeTooltip.text"
        class="global-tooltip"
        :style="{
          left: `${activeTooltip.left}px`,
          top: `${activeTooltip.top}px`,
          transform: activeTooltip.transform
        }"
        role="tooltip"
      >
        {{ activeTooltip.text }}
      </div>
    </Teleport>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, reactive, ref } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'

const userInput = ref('')
const userReviewInput = ref('')
const isSubmitting = ref(false)
const isReviewing = ref(false)
const conversation = ref([])
const eventSource = ref(null)
const currentActiveStage = ref(null)
const activeSubStages = ref(new Map())
const activeAssistantMessage = ref(null)
const copiedMessageId = ref('')
const messagesContainer = ref(null)
const composerInput = ref(null)
const rightPanelCollapsed = ref(false)
const currentYear = new Date().getFullYear()
const keywordError = ref('')
const yearError = ref('')
const paperLimitError = ref('')
const customPromptError = ref('')
const activeTooltip = reactive({
  text: '',
  left: 0,
  top: 0,
  transform: 'translate(-50%, -100%)'
})

const taskSettings = reactive({
  yearStart: '2021',
  yearEnd: '2026',
  keywords: '',
  paperLimit: 20,
  outputLanguage: '中文',
  customPrompt: ''
})

const promptExamples = [
  '调研多模态大模型在医学影像诊断中的研究进展',
  '生成一份 AI Agent 评测方法的学术综述',
  '比较近三年 RAG 与长上下文模型在论文写作中的应用',
  '分析自动化科研助手的发展趋势、局限性和未来方向'
]

const activeAssistantId = computed(() => activeAssistantMessage.value?.id || '')

const runStatusText = computed(() => {
  if (isReviewing.value) return '等待人工审核'
  if (isSubmitting.value) return '报告生成中'
  return conversation.value.length > 0 ? '可继续补充' : '准备就绪'
})

const constraintSummary = computed(() => {
  const years = taskSettings.yearStart || taskSettings.yearEnd
    ? `${taskSettings.yearStart || '不限'}-${taskSettings.yearEnd || '不限'}`
    : '年份不限'
  const keywords = taskSettings.keywords.trim()
    ? `，关键词 ${taskSettings.keywords.trim()}`
    : ''
  return `约束：${years}${keywords}，检索上限 ${taskSettings.paperLimit} 篇，${taskSettings.outputLanguage}`
})

const parseMarkdown = (content) => {
  if (!content) return ''
  return DOMPurify.sanitize(marked.parse(content))
}

const createId = (prefix) => {
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

const submitRequest = () => {
  const query = userInput.value.trim()
  if (!query || isSubmitting.value) return
  if (!validateTaskSettings()) return

  closeEventSource()
  resetRuntimeState()

  const userMessage = {
    id: createId('user'),
    role: 'user',
    content: query,
    timestamp: new Date().toISOString()
  }

  const assistantMessage = {
    id: createId('assistant'),
    role: 'assistant',
    replyTo: userMessage.id,
    status: 'connecting',
    statusText: '正在连接研究代理',
    stages: [],
    finalContent: '',
    timestamp: new Date().toISOString(),
    completedAt: null
  }

  conversation.value.push(userMessage, assistantMessage)
  activeAssistantMessage.value = assistantMessage
  userInput.value = ''
  isSubmitting.value = true

  nextTick(() => {
    autoScroll()
    composerInput.value?.focus()
  })

  const source = new EventSource(`/api/research?query=${encodeURIComponent(buildResearchQuery(query))}`)
  eventSource.value = source

  source.onmessage = (event) => {
    try {
      handleBackendData(JSON.parse(event.data))
    } catch (error) {
      console.error('处理流式数据失败:', error)
    }
  }

  source.onerror = () => {
    if (!isSubmitting.value) return
    markConnectionError()
    finishProcessing()
  }
}

const buildResearchQuery = (query) => {
  const constraints = []
  if (taskSettings.yearStart || taskSettings.yearEnd) {
    constraints.push(`检索阶段 - 年份范围：${taskSettings.yearStart || '不限'} 到 ${taskSettings.yearEnd || '不限'}，用于限制论文搜索结果`)
  }
  if (taskSettings.keywords.trim()) {
    constraints.push(`检索阶段 - 论文检索关键词：${taskSettings.keywords.trim()}（按分号分隔）`)
  }
  constraints.push(`检索阶段 - 获取论文结果最大数量：${taskSettings.paperLimit}`)
  constraints.push(`生成阶段 - 输出语言：${taskSettings.outputLanguage}`)
  if (taskSettings.customPrompt.trim()) {
    constraints.push(`生成阶段 - 用户自定义约束提示词：${taskSettings.customPrompt.trim()}。该内容作为智能体系统提示词的一部分参与分析、写作和报告组织。`)
  }

  return `${query}\n\n本次调研报告生成约束：\n${constraints.map(item => `- ${item}`).join('\n')}`
}

const validateTaskSettings = () => {
  validateYearRange()
  validateKeywords()
  validatePaperLimit()
  validateCustomPrompt()
  return !yearError.value && !keywordError.value && !paperLimitError.value && !customPromptError.value
}

const validateYearRange = () => {
  const start = normalizeYear(taskSettings.yearStart)
  const end = normalizeYear(taskSettings.yearEnd)

  if ((taskSettings.yearStart && !start) || (taskSettings.yearEnd && !end)) {
    yearError.value = `请输入 1900-${currentYear} 之间的四位年份。`
    return
  }

  if (start && end && start > end) {
    yearError.value = '起始年份不能晚于结束年份。'
    return
  }

  yearError.value = ''
}

const clearYearError = () => {
  if (yearError.value) {
    yearError.value = ''
  }
}

const normalizeYear = (value) => {
  const text = String(value ?? '').trim()
  if (!text) return null
  const year = Number(text)
  if (!Number.isInteger(year) || year < 1900 || year > currentYear) return null
  return year
}

const validateKeywords = () => {
  const value = taskSettings.keywords.trim()
  if (!value) {
    keywordError.value = ''
    return
  }

  if (/[，,、；]/.test(value)) {
    keywordError.value = '请使用英文分号分隔多个关键词，例如：large language model; agent; reasoning。'
    return
  }

  if (value.includes(';')) {
    const hasEmptyKeyword = value.split(';').some(keyword => !keyword.trim())
    if (hasEmptyKeyword) {
      keywordError.value = '分号前后都需要是有效关键词，请删除多余的分号。'
      return
    }
  }

  keywordError.value = ''
}

const clearKeywordError = () => {
  if (keywordError.value) {
    keywordError.value = ''
  }
}

const validatePaperLimit = () => {
  const rawValue = String(taskSettings.paperLimit ?? '').trim()
  const value = Number(rawValue)

  if (!rawValue) {
    paperLimitError.value = '请填写检索论文上限，建议输入 5-50 之间的整数。'
    return
  }

  if (!Number.isInteger(value)) {
    paperLimitError.value = '论文数量必须是整数，例如 20。'
    return
  }

  if (value < 1 || value > 100) {
    paperLimitError.value = '论文数量范围为 1-100，避免检索规模过小或处理开销过高。'
    return
  }

  paperLimitError.value = ''
}

const clearPaperLimitError = () => {
  if (paperLimitError.value) {
    paperLimitError.value = ''
  }
}

const validateCustomPrompt = () => {
  const value = taskSettings.customPrompt.trim()

  if (!value) {
    customPromptError.value = ''
    return
  }

  if (value.length < 6) {
    customPromptError.value = '自定义约束过短，请写成可执行的生成约束，例如“强调方法对比”。'
    return
  }

  if (value.length > 500) {
    customPromptError.value = '自定义约束建议控制在 500 字以内，避免覆盖主要研究主题。'
    return
  }

  customPromptError.value = ''
}

const clearCustomPromptError = () => {
  if (customPromptError.value) {
    customPromptError.value = ''
  }
}

const handleBackendData = (backData) => {
  const { step, state, data } = backData
  const handlers = {
    initializing: () => handleInitializing(step, data),
    thinking: () => handleThinking(step, data),
    generating: () => handleGenerating(step, data),
    user_review: () => handleUserReview(step, data),
    completed: () => handleComplete(step, data),
    error: () => handleError(step, data),
    finished: () => handleFinish()
  }

  if (handlers[state]) {
    handlers[state]()
  } else {
    console.warn('未知流程状态:', state, 'step:', step)
  }
}

const handleInitializing = (step, data) => {
  const stage = createStage(step)
  stage.status = 'preparing'
  stage.title = `${getStepName(step)}准备中`
  stage.thinking += extractThinking(data)
  stage.content += extractContent(data)
  updateAssistantStatus('processing', `${getStepName(step)}准备中`)
  autoScroll()
}

const handleThinking = (step, data) => {
  const stage = ensureStage(step)
  stage.status = 'thinking'
  stage.title = `${getStepName(step)}思考中`
  stage.thinking += extractThinking(data) || stringifyData(data)
  stage.showThinking = true
  updateAssistantStatus('processing', `${getStepName(step)}思考中`)
  autoScroll()
}

const handleGenerating = (step, data) => {
  const stage = ensureStage(step)
  stage.status = 'generating'
  stage.title = `${getStepName(step)}生成中`
  stage.content += stringifyData(data)
  stage.expanded = true
  updateAssistantStatus('processing', `${getStepName(step)}生成中`)
  autoScroll()
}

const handleUserReview = (step, data) => {
  const stage = ensureStage(step)
  stage.status = 'review'
  stage.title = `${getStepName(step)}等待审核`
  stage.expanded = true
  userReviewInput.value = stringifyData(data)
  isReviewing.value = true
  updateAssistantStatus('review', '等待人工审核')
  autoScroll()
}

const handleComplete = (step, data) => {
  const stage = ensureStage(step)
  stage.status = 'done'
  stage.title = `${getStepName(step)}完成`
  stage.content += stringifyData(data)

  if (step?.startsWith('section_writing')) {
    activeSubStages.value.delete(step)
  } else if (currentActiveStage.value?.step === step) {
    currentActiveStage.value = null
  }

  updateAssistantStatus('processing', `${getStepName(step)}完成`)
  autoScroll()
}

const handleError = (step, data) => {
  const stage = ensureStage(step || 'error')
  stage.status = 'error'
  stage.title = `${getStepName(step)}异常`
  stage.content += stringifyData(data) || '当前步骤处理异常。'
  updateAssistantStatus('error', '生成异常')
  finishProcessing()
  autoScroll()
}

const handleFinish = () => {
  if (activeAssistantMessage.value) {
    activeAssistantMessage.value.status = 'done'
    activeAssistantMessage.value.statusText = '调研报告生成完成'
    activeAssistantMessage.value.completedAt = new Date().toISOString()
    activeAssistantMessage.value.finalContent = getAssistantPlainText(activeAssistantMessage.value)
    persistHistory(activeAssistantMessage.value)
  }
  finishProcessing()
  autoScroll()
}

const createStage = (step) => {
  const assistant = activeAssistantMessage.value
  if (!assistant) {
    throw new Error('没有可写入的助手消息')
  }

  const stage = {
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

  assistant.stages.push(stage)

  if (step?.startsWith('section_writing')) {
    activeSubStages.value.set(step, stage)
  } else {
    currentActiveStage.value = stage
  }

  return stage
}

const ensureStage = (step) => {
  if (step?.startsWith('section_writing') && activeSubStages.value.has(step)) {
    return activeSubStages.value.get(step)
  }

  if (currentActiveStage.value?.step === step) {
    return currentActiveStage.value
  }

  const assistant = activeAssistantMessage.value
  const reusableStage = assistant?.stages.find(stage => stage.step === step && stage.status !== 'done')
  if (reusableStage) {
    currentActiveStage.value = reusableStage
    return reusableStage
  }

  return createStage(step)
}

const updateAssistantStatus = (status, text) => {
  if (!activeAssistantMessage.value) return
  activeAssistantMessage.value.status = status
  activeAssistantMessage.value.statusText = text
}

const submitReviewInput = async () => {
  const reviewText = userReviewInput.value.trim()
  if (!reviewText) return

  try {
    const response = await fetch('/send_input', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ input: reviewText })
    })

    if (!response.ok) {
      throw new Error(`审核提交失败: ${response.status}`)
    }

    if (currentActiveStage.value) {
      currentActiveStage.value.content += `\n\n人工反馈：${reviewText}\n`
      currentActiveStage.value.status = 'generating'
      currentActiveStage.value.title = `${getStepName(currentActiveStage.value.step)}继续生成`
    }

    isReviewing.value = false
    userReviewInput.value = ''
    updateAssistantStatus('processing', '已提交审核反馈')
    autoScroll()
  } catch (error) {
    console.error(error)
    window.alert('提交审核反馈失败，请检查后端服务。')
  }
}

const stopCurrentRun = () => {
  if (activeAssistantMessage.value && isSubmitting.value) {
    activeAssistantMessage.value.status = 'stopped'
    activeAssistantMessage.value.statusText = '已停止'
    activeAssistantMessage.value.completedAt = new Date().toISOString()
  }
  finishProcessing()
}

const clearConversation = () => {
  if (isSubmitting.value) {
    stopCurrentRun()
  }
  conversation.value = []
  copiedMessageId.value = ''
  resetRuntimeState()
  nextTick(() => composerInput.value?.focus())
}

const closeEventSource = () => {
  eventSource.value?.close()
  eventSource.value = null
}

const finishProcessing = () => {
  closeEventSource()
  isSubmitting.value = false
  isReviewing.value = false
  currentActiveStage.value = null
  activeSubStages.value.clear()
}

const resetRuntimeState = () => {
  currentActiveStage.value = null
  activeSubStages.value.clear()
  activeAssistantMessage.value = null
  userReviewInput.value = ''
  isReviewing.value = false
}

const markConnectionError = () => {
  if (!activeAssistantMessage.value) return
  const stage = createStage('connection')
  stage.status = 'error'
  stage.title = '连接异常'
  stage.content = '连接后端服务失败，请确认服务已经启动。'
  activeAssistantMessage.value.status = 'error'
  activeAssistantMessage.value.statusText = '连接异常'
  activeAssistantMessage.value.completedAt = new Date().toISOString()
}

const stringifyData = (data) => {
  if (data === null || data === undefined) return ''
  if (typeof data === 'string') return data
  if (typeof data.content === 'string') return data.content
  if (typeof data.text === 'string') return data.text
  return JSON.stringify(data, null, 2)
}

const extractThinking = (data) => {
  if (!data || typeof data !== 'object') return ''
  return typeof data.thinking === 'string' ? data.thinking : ''
}

const extractContent = (data) => {
  if (!data || typeof data !== 'object') return ''
  return typeof data.content === 'string' ? data.content : ''
}

const getStepName = (step = '') => {
  const stepNames = {
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

const getStageStateText = (status) => {
  const statusMap = {
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

const getStagePlaceholder = (status) => {
  if (status === 'thinking') return '正在组织调研思路...'
  if (status === 'generating') return '正在生成阶段内容...'
  if (status === 'review') return '等待人工审核反馈。'
  if (status === 'error') return '当前步骤暂无更多错误详情。'
  return '等待后端返回内容。'
}

const getAssistantStatusText = (message) => {
  if (message.statusText) return message.statusText
  const statusMap = {
    connecting: '正在连接研究代理',
    processing: '正在生成',
    review: '等待审核',
    done: '调研报告生成完成',
    error: '生成异常',
    stopped: '已停止'
  }
  return statusMap[message.status] || '运行中'
}

const selectPrompt = (prompt) => {
  userInput.value = prompt
  nextTick(() => composerInput.value?.focus())
}

const handleComposerKeydown = (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    submitRequest()
  }
}

const handleTooltipEnter = (event) => {
  const target = event.target
  if (!(target instanceof Element)) {
    hideTooltip()
    return
  }

  const trigger = target.closest('.help-tip')
  if (!(trigger instanceof HTMLElement) || !trigger.dataset.tooltip) {
    if (event.type === 'mouseover') {
      hideTooltip()
    }
    return
  }

  showTooltip(trigger, trigger.dataset.tooltip)
}

const showTooltip = (trigger, text) => {
  const rect = trigger.getBoundingClientRect()
  const estimatedWidth = Math.min(280, Math.max(200, window.innerWidth - 24))
  const center = rect.left + rect.width / 2
  const left = Math.min(
    Math.max(center, estimatedWidth / 2 + 12),
    window.innerWidth - estimatedWidth / 2 - 12
  )
  const shouldShowBelow = rect.top < 86

  activeTooltip.text = text
  activeTooltip.left = left
  activeTooltip.top = shouldShowBelow ? rect.bottom + 10 : rect.top - 10
  activeTooltip.transform = shouldShowBelow ? 'translate(-50%, 0)' : 'translate(-50%, -100%)'
}

const hideTooltip = () => {
  activeTooltip.text = ''
}

const copyAssistantOutput = async (message) => {
  const content = getAssistantPlainText(message)
  if (!content) return
  await navigator.clipboard.writeText(content)
  copiedMessageId.value = message.id
  setTimeout(() => {
    if (copiedMessageId.value === message.id) {
      copiedMessageId.value = ''
    }
  }, 1600)
}

const getAssistantPlainText = (message) => {
  if (!message) return ''
  const stageContent = message.stages
    .map(stage => stage.content?.trim())
    .filter(Boolean)
    .join('\n\n')
  return message.finalContent || stageContent
}

const persistHistory = (assistantMessage) => {
  const userMessage = conversation.value.find(message => message.id === assistantMessage.replyTo)
  if (!userMessage) return

  const item = {
    id: assistantMessage.id,
    title: createHistoryTitle(userMessage.content),
    query: userMessage.content,
    status: assistantMessage.status === 'error' ? 'failed' : 'completed',
    createdAt: userMessage.timestamp,
    updatedAt: assistantMessage.completedAt,
    content: getAssistantPlainText(assistantMessage)
  }

  try {
    const saved = JSON.parse(localStorage.getItem('reportHistory') || '[]')
    const nextHistory = [item, ...saved.filter(record => record.id !== item.id)].slice(0, 50)
    localStorage.setItem('reportHistory', JSON.stringify(nextHistory))
  } catch (error) {
    console.error('保存历史记录失败:', error)
  }
}

const createHistoryTitle = (query) => {
  return query.length > 28 ? `${query.slice(0, 28)}...` : query
}

const formatTime = (timestamp) => {
  if (!timestamp) return ''
  return new Date(timestamp).toLocaleTimeString('zh-CN', {
    hour: '2-digit',
    minute: '2-digit'
  })
}

const autoScroll = () => {
  nextTick(() => {
    const container = messagesContainer.value
    if (!container) return
    container.scrollTo({
      top: container.scrollHeight,
      behavior: 'smooth'
    })
  })
}

onBeforeUnmount(() => {
  closeEventSource()
})
</script>

<style scoped>
.report-workbench {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 332px;
  height: 100%;
  min-height: 0;
  background: var(--pa-bg);
  color: var(--pa-text);
  transition: grid-template-columns 0.22s ease;
}

.report-workbench.right-collapsed {
  grid-template-columns: minmax(0, 1fr) 44px;
}

.report-center {
  display: grid;
  grid-template-rows: auto minmax(0, 1fr) auto;
  min-width: 0;
  min-height: 0;
  background: var(--pa-surface);
  border-right: 1px solid var(--pa-border);
}

.report-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  min-height: 72px;
  padding: 14px 22px;
  border-bottom: 1px solid var(--pa-border);
  background: var(--pa-surface);
}

.eyebrow {
  margin: 0 0 5px;
  color: var(--pa-primary);
  font-size: 0.73rem;
  font-weight: 800;
  text-transform: uppercase;
}

.report-header h1 {
  margin: 0;
  font-size: 1.25rem;
  line-height: 1.25;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-pill,
.assistant-status {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  min-height: 32px;
  padding: 0 11px;
  border-radius: 999px;
  background: var(--pa-surface-muted);
  color: var(--pa-text-muted);
  font-size: 0.82rem;
  font-weight: 800;
  white-space: nowrap;
}

.status-pill span,
.status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: currentColor;
}

.status-pill.active,
.assistant-status.processing,
.assistant-status.connecting {
  background: var(--pa-success-soft);
  color: var(--pa-success);
}

.status-pill.review,
.assistant-status.review {
  background: var(--pa-warning-soft);
  color: var(--pa-warning);
}

.assistant-status.error,
.assistant-status.stopped {
  background: var(--pa-danger-soft);
  color: var(--pa-danger);
}

.primary-button,
.secondary-button,
.icon-button,
.danger-button,
.icon-only-button,
.message-actions button,
.review-panel button {
  min-height: 34px;
  border-radius: 8px;
  font-weight: 800;
  cursor: pointer;
}

.primary-button {
  padding: 0 16px;
  border: 1px solid var(--pa-primary);
  background: var(--pa-primary);
  color: var(--pa-surface);
}

.primary-button:disabled,
.danger-button:disabled,
.review-panel button:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.secondary-button,
.icon-button,
.icon-only-button,
.message-actions button {
  padding: 0 12px;
  border: 1px solid var(--pa-border);
  background: var(--pa-surface);
  color: var(--pa-text-muted);
}

.secondary-button:hover,
.icon-button:hover,
.icon-only-button:hover,
.message-actions button:hover {
  background: var(--pa-surface-soft);
  color: var(--pa-text);
}

.danger-button {
  padding: 0 14px;
  border: 1px solid var(--pa-danger-soft);
  background: var(--pa-danger-soft);
  color: var(--pa-danger);
}

.report-main {
  min-height: 0;
  overflow-y: auto;
  background: var(--pa-surface);
}

.start-panel {
  display: grid;
  align-content: center;
  gap: 20px;
  max-width: 920px;
  min-height: 100%;
  margin: 0 auto;
  padding: 36px 28px;
}

.start-copy {
  display: grid;
  gap: 8px;
}

.start-copy h2 {
  margin: 0;
  font-size: 1.65rem;
  line-height: 1.25;
}

.topic-card {
  display: grid;
  gap: 10px;
  padding: 16px;
  border: 1px solid var(--pa-border);
  border-radius: 10px;
  background: var(--pa-surface);
  box-shadow: var(--pa-shadow);
}

.topic-card label {
  color: var(--pa-text-muted);
  font-size: 0.84rem;
  font-weight: 800;
}

textarea,
input,
select {
  width: 100%;
  border: 1px solid var(--pa-border);
  border-radius: 8px;
  outline: none;
  background: var(--pa-input-bg);
  color: var(--pa-text);
}

textarea {
  resize: none;
  line-height: 1.55;
}

.topic-card textarea {
  min-height: 150px;
  padding: 12px;
  font-size: 0.96rem;
}

textarea::placeholder,
input::placeholder {
  color: var(--pa-text-subtle);
}

textarea:focus,
input:focus,
select:focus {
  border-color: var(--pa-primary);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--pa-primary) 18%, transparent);
}

.topic-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.constraint-summary {
  overflow: hidden;
  color: var(--pa-text-subtle);
  font-size: 0.82rem;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.prompt-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.prompt-row button {
  min-height: 32px;
  padding: 0 12px;
  border: 1px solid var(--pa-border);
  border-radius: 999px;
  background: var(--pa-surface-soft);
  color: var(--pa-text-muted);
  cursor: pointer;
}

.prompt-row button:hover {
  color: var(--pa-text);
  border-color: var(--pa-border-strong);
}

.run-panel {
  display: grid;
  gap: 18px;
  max-width: 980px;
  margin: 0 auto;
  padding: 24px 28px 34px;
}

.message-row {
  display: grid;
  gap: 8px;
}

.message-label {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  color: var(--pa-text-subtle);
  font-size: 0.78rem;
  font-weight: 800;
}

.message-label span {
  font-weight: 600;
}

.user-request,
.assistant-run-card {
  border: 1px solid var(--pa-border);
  border-radius: 10px;
  background: var(--pa-surface);
}

.user-request {
  padding: 14px;
  color: var(--pa-text);
  line-height: 1.7;
  white-space: pre-wrap;
}

.assistant-run-card {
  display: grid;
  gap: 12px;
  padding: 14px;
}

.stage-list {
  display: grid;
  gap: 10px;
}

.stage-card {
  border: 1px solid var(--pa-border);
  border-radius: 8px;
  background: var(--pa-surface-soft);
  overflow: hidden;
}

.stage-header {
  display: grid;
  grid-template-columns: 30px minmax(0, 1fr) auto;
  align-items: center;
  gap: 10px;
  width: 100%;
  min-height: 44px;
  padding: 7px 10px;
  border: 0;
  background: transparent;
  color: var(--pa-text);
  text-align: left;
  cursor: pointer;
}

.stage-index {
  display: grid;
  place-items: center;
  width: 26px;
  height: 26px;
  border-radius: 7px;
  background: var(--pa-primary-soft);
  color: var(--pa-primary);
  font-size: 0.76rem;
  font-weight: 900;
}

.stage-title {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-weight: 800;
}

.stage-state {
  padding: 4px 8px;
  border-radius: 999px;
  background: var(--pa-surface-muted);
  color: var(--pa-text-muted);
  font-size: 0.72rem;
  font-weight: 900;
}

.stage-state.thinking,
.stage-state.generating,
.stage-state.preparing {
  background: var(--pa-success-soft);
  color: var(--pa-success);
}

.stage-state.review {
  background: var(--pa-warning-soft);
  color: var(--pa-warning);
}

.stage-state.error {
  background: var(--pa-danger-soft);
  color: var(--pa-danger);
}

.stage-content {
  padding: 0 12px 12px 50px;
}

.thinking-box {
  margin-bottom: 10px;
  border: 1px solid var(--pa-border);
  border-radius: 8px;
  background: var(--pa-surface);
  overflow: hidden;
}

.thinking-box button {
  display: flex;
  justify-content: space-between;
  width: 100%;
  min-height: 34px;
  padding: 0 10px;
  border: 0;
  background: var(--pa-surface-muted);
  color: var(--pa-text-muted);
  cursor: pointer;
}

.thinking-box pre {
  max-height: 240px;
  margin: 0;
  padding: 10px;
  overflow: auto;
  color: var(--pa-text);
  font-family: Consolas, 'Courier New', monospace;
  font-size: 0.84rem;
  line-height: 1.55;
  white-space: pre-wrap;
}

.muted-line {
  margin: 0;
  color: var(--pa-text-subtle);
  font-size: 0.9rem;
}

.assistant-loading {
  display: grid;
  gap: 8px;
}

.assistant-loading span {
  height: 10px;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--pa-surface-muted), var(--pa-primary-soft), var(--pa-surface-muted));
  background-size: 220% 100%;
  animation: shimmer 1.2s ease-in-out infinite;
}

.assistant-loading span:nth-child(2) {
  width: 80%;
}

.assistant-loading span:nth-child(3) {
  width: 52%;
}

@keyframes shimmer {
  from { background-position: 220% 0; }
  to { background-position: -220% 0; }
}

.review-panel {
  display: grid;
  gap: 10px;
  padding: 12px;
  border: 1px solid color-mix(in srgb, var(--pa-warning) 30%, var(--pa-border));
  border-radius: 8px;
  background: var(--pa-warning-soft);
}

.review-title {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  color: var(--pa-warning);
  font-size: 0.86rem;
}

.review-panel textarea {
  min-height: 90px;
  padding: 10px;
}

.review-panel button {
  justify-self: end;
  padding: 0 14px;
  border: 1px solid var(--pa-warning);
  background: var(--pa-warning);
  color: var(--pa-surface);
}

.message-actions {
  display: flex;
  justify-content: flex-end;
}

.message-actions button {
  border-color: var(--pa-border);
}

.continue-composer {
  padding: 12px 22px 16px;
  border-top: 1px solid var(--pa-border);
  background: var(--pa-surface);
}

.continue-composer form {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto auto;
  gap: 8px;
  max-width: 980px;
  margin: 0 auto;
}

.continue-composer textarea {
  min-height: 38px;
  max-height: 130px;
  padding: 8px 10px;
}

.task-panel {
  min-width: 0;
  min-height: 0;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
  background: var(--pa-surface-soft);
}

.task-panel.collapsed {
  display: grid;
  place-items: start center;
  padding-top: 14px;
}

.expand-task-button {
  writing-mode: vertical-rl;
  min-width: 30px;
  min-height: 78px;
  border: 1px solid var(--pa-border);
  border-radius: 8px;
  background: var(--pa-surface);
  color: var(--pa-text-muted);
  font-weight: 800;
  cursor: pointer;
}

.task-panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  min-height: 72px;
  padding: 14px;
  border-bottom: 1px solid var(--pa-border);
}

.task-panel-header h2 {
  margin: 0;
  font-size: 1.05rem;
}

.icon-only-button {
  width: 34px;
  padding: 0;
  font-size: 1.2rem;
}

.task-panel-body {
  display: grid;
  align-content: start;
  gap: 12px;
  min-height: 0;
  overflow-y: auto;
  padding: 14px;
}

.task-panel-note {
  margin: 0;
  padding: 11px 12px;
  border: 1px solid var(--pa-border);
  border-radius: 8px;
  background: var(--pa-primary-soft);
  color: var(--pa-primary-strong);
  font-size: 0.78rem;
  font-weight: 700;
  line-height: 1.55;
}

.constraint-card {
  display: grid;
  gap: 9px;
  padding: 13px;
  border: 1px solid var(--pa-border);
  border-radius: 8px;
  background: var(--pa-surface);
}

.constraint-card.invalid {
  border-color: color-mix(in srgb, var(--pa-danger) 45%, var(--pa-border));
}

.constraint-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.constraint-card h3 {
  margin: 0;
  font-size: 0.92rem;
}

.help-tip {
  display: inline-grid;
  place-items: center;
  width: 20px;
  height: 20px;
  border: 1px solid var(--pa-border);
  border-radius: 50%;
  background: var(--pa-surface-soft);
  color: var(--pa-text-muted);
  font-size: 0.75rem;
  font-weight: 900;
  cursor: help;
  transition: border-color 0.15s ease, background 0.15s ease, color 0.15s ease;
}

.help-tip:hover,
.help-tip:focus-visible {
  border-color: var(--pa-primary);
  background: var(--pa-primary-soft);
  color: var(--pa-primary);
}

.global-tooltip {
  position: fixed;
  z-index: 2147483647;
  width: max-content;
  max-width: min(280px, calc(100vw - 24px));
  padding: 8px 10px;
  border: 1px solid var(--pa-border);
  border-radius: 8px;
  background: var(--pa-text);
  color: var(--pa-surface);
  box-shadow: var(--pa-shadow);
  font-size: 0.76rem;
  font-weight: 700;
  line-height: 1.45;
  pointer-events: none;
  text-align: left;
  white-space: normal;
}

.field-error {
  margin: 0;
  font-size: 0.76rem;
  line-height: 1.48;
}

.field-error {
  color: var(--pa-danger);
  font-weight: 800;
}

.field-pair {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.constraint-card label {
  display: grid;
  gap: 6px;
}

.constraint-card label span {
  color: var(--pa-text-muted);
  font-size: 0.78rem;
  font-weight: 800;
}

.constraint-card input,
.constraint-card select {
  min-height: 36px;
  padding: 0 10px;
}

.constraint-card textarea {
  padding: 9px 10px;
}

.markdown-body {
  color: var(--pa-text);
  line-height: 1.72;
  word-break: break-word;
}

.markdown-body :deep(h1),
.markdown-body :deep(h2),
.markdown-body :deep(h3) {
  margin: 18px 0 10px;
  color: var(--pa-text);
  line-height: 1.32;
}

.markdown-body :deep(h1) {
  font-size: 1.28rem;
}

.markdown-body :deep(h2) {
  font-size: 1.12rem;
}

.markdown-body :deep(h3) {
  font-size: 1rem;
}

.markdown-body :deep(p) {
  margin: 0 0 12px;
}

.markdown-body :deep(ul),
.markdown-body :deep(ol) {
  margin: 0 0 12px;
  padding-left: 1.4rem;
}

.markdown-body :deep(code) {
  padding: 2px 5px;
  border-radius: 5px;
  background: var(--pa-surface-muted);
  color: var(--pa-warning);
  font-family: Consolas, 'Courier New', monospace;
}

.markdown-body :deep(pre) {
  overflow: auto;
  padding: 12px;
  border-radius: 8px;
  background: var(--pa-surface-muted);
  color: var(--pa-text);
}

.report-main::-webkit-scrollbar,
.task-panel-body::-webkit-scrollbar,
.thinking-box pre::-webkit-scrollbar {
  width: 8px;
}

.report-main::-webkit-scrollbar-thumb,
.task-panel-body::-webkit-scrollbar-thumb,
.thinking-box pre::-webkit-scrollbar-thumb {
  background: var(--pa-scrollbar);
  border-radius: 999px;
}

.report-main::-webkit-scrollbar-track,
.task-panel-body::-webkit-scrollbar-track,
.thinking-box pre::-webkit-scrollbar-track {
  background: transparent;
}
</style>
