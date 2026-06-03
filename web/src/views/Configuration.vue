<template>
  <div class="config-page">
    <header class="config-hero">
      <div class="hero-copy">
        <p class="eyebrow">Runtime Configuration</p>
        <h1>系统配置</h1>
        <p>
          集中管理 Provider、默认模型、智能体阶段模型和知识库嵌入模型，让 Paper-Agent 的运行参数有一个稳定入口。
        </p>
        <span v-if="modelsPath" class="path-pill">YAML: {{ modelsPath }}</span>
      </div>

      <div class="hero-actions">
        <button class="secondary-button" type="button" :disabled="isLoading" @click="loadSettings">
          {{ isLoading ? '加载中' : '重新加载' }}
        </button>
        <button v-if="draftMeta" class="secondary-button" type="button" @click="restoreDraft">
          恢复草稿
        </button>
        <button class="secondary-button" type="button" :disabled="isLoading" @click="saveDraft">
          保存草稿
        </button>
        <button class="primary-button" type="button" :disabled="isSaving || isLoading" @click="saveSettings">
          {{ isSaving ? '保存中' : '保存配置' }}
        </button>
      </div>
    </header>

    <section v-if="draftMeta" class="draft-banner">
      <div>
        <strong>检测到本地草稿</strong>
        <span>保存于 {{ formatDate(draftMeta.savedAt) }}，可恢复到当前表单继续编辑。</span>
      </div>
      <div class="draft-actions">
        <button type="button" @click="restoreDraft">恢复</button>
        <button type="button" @click="clearDraft">忽略</button>
      </div>
    </section>

    <section v-if="isLoading" class="center-state">
      <div class="spinner"></div>
      <p>正在读取模型配置...</p>
    </section>

    <template v-else>
      <section class="config-section provider-section">
        <header class="section-header">
          <div>
            <p class="section-kicker">01 Provider</p>
            <h2>模型 Provider</h2>
            <span>配置 OpenAI、SiliconFlow、DashScope、Ark 或自定义 OpenAI-compatible 服务。</span>
          </div>
          <button class="primary-button" type="button" @click="addProvider">
            添加 Provider
          </button>
        </header>

        <div class="provider-grid">
          <article v-for="provider in providers" :key="provider.localId" class="provider-card">
            <div class="card-top">
              <div>
                <span class="card-kicker">Provider</span>
                <h3>{{ provider.id || '未命名 Provider' }}</h3>
              </div>
              <button
                class="danger-button"
                type="button"
                :disabled="providers.length <= 1"
                @click="removeProvider(provider)"
              >
                删除
              </button>
            </div>

            <div class="form-grid">
              <label class="field">
                <span>
                  Provider ID
                  <button
                    class="hint-button"
                    type="button"
                    data-tooltip="Provider ID 会写入 YAML 顶层键，例如 siliconflow 或 ark，也会被模型配置引用。"
                    aria-label="Provider ID 说明"
                  >
                    ?
                  </button>
                </span>
                <input
                  v-model.trim="provider.id"
                  type="text"
                  placeholder="siliconflow"
                  @change="syncProviderReferences(provider)"
                >
              </label>

              <label class="field">
                <span>
                  Provider 类型
                  <button
                    class="hint-button"
                    type="button"
                    data-tooltip="用于减少输入错误。自定义服务可选择 OpenAI Compatible。"
                    aria-label="Provider 类型说明"
                  >
                    ?
                  </button>
                </span>
                <select v-model="provider.type" @change="handleProviderTypeChange(provider)">
                  <option v-for="type in providerTypeOptions" :key="type.value" :value="type.value">
                    {{ type.label }}
                  </option>
                </select>
              </label>

              <label class="field wide">
                <span>
                  API URL
                  <button
                    class="hint-button"
                    type="button"
                    data-tooltip="服务商的 OpenAI-compatible base_url，后端会传给模型客户端。"
                    aria-label="API URL 说明"
                  >
                    ?
                  </button>
                </span>
                <input v-model.trim="provider.base_url" type="url" placeholder="https://api.example.com/v1">
              </label>

              <label class="field wide">
                <span>
                  API Key
                  <button
                    class="hint-button"
                    type="button"
                    data-tooltip="可以填写真实 Key，也可以填写环境变量名，例如 OPENAI_API_KEY。测试时后端会自动解析环境变量。"
                    aria-label="API Key 说明"
                  >
                    ?
                  </button>
                </span>
                <div class="password-field">
                  <input
                    v-model.trim="provider.api_key"
                    :type="visibleProviderKeys[provider.localId] ? 'text' : 'password'"
                    placeholder="OPENAI_API_KEY 或 sk-..."
                    autocomplete="off"
                  >
                  <button type="button" @click="toggleProviderKey(provider.localId)">
                    {{ visibleProviderKeys[provider.localId] ? '隐藏' : '显示' }}
                  </button>
                </div>
              </label>
            </div>
          </article>
        </div>
      </section>

      <section class="config-section">
        <header class="section-header">
          <div>
            <p class="section-kicker">02 Defaults</p>
            <h2>默认模型配置</h2>
            <span>模块未指定专用模型时，将回退到这里配置的 LLM 或 Embedding。</span>
          </div>
        </header>

        <div class="model-grid default-grid">
          <article v-for="item in defaultModels" :key="item.key" class="model-card" :class="item.kind">
            <div class="model-card-header">
              <div>
                <span class="key-chip">{{ item.config.key }}</span>
                <h3>{{ item.label }}</h3>
                <p>{{ item.description }}</p>
              </div>
              <button class="test-button" type="button" :disabled="isTesting(item.config.key)" @click="testModel(item)">
                {{ isTesting(item.config.key) ? '测试中' : '测试连通性' }}
              </button>
            </div>

            <div class="model-fields">
              <label class="field">
                <span>
                  Provider
                  <button
                    class="hint-button"
                    type="button"
                    data-tooltip="选择该模型调用时使用的服务商，对应 YAML 中的 model-provider。"
                    aria-label="模型 Provider 说明"
                  >
                    ?
                  </button>
                </span>
                <select v-model="item.config.provider">
                  <option v-for="provider in providerOptions" :key="provider.value" :value="provider.value">
                    {{ provider.label }}
                  </option>
                </select>
              </label>

              <label class="field">
                <span>
                  模型名称
                  <button
                    class="hint-button"
                    type="button"
                    data-tooltip="服务商侧的模型 ID，例如 Qwen/Qwen3-32B、text-embedding-v4 或 doubao endpoint。"
                    aria-label="模型名称说明"
                  >
                    ?
                  </button>
                </span>
                <input v-model.trim="item.config.model" type="text" placeholder="model-name">
              </label>

              <label v-if="item.kind === 'embedding'" class="field">
                <span>
                  向量维度
                  <button
                    class="hint-button"
                    type="button"
                    data-tooltip="Embedding 返回向量维度，会影响检索、相似度计算和聚类。"
                    aria-label="向量维度说明"
                  >
                    ?
                  </button>
                </span>
                <input v-model.number="item.config.dimension" type="number" min="1" placeholder="1024">
              </label>
            </div>

            <p v-if="testResults[item.config.key]" class="test-result" :class="testResults[item.config.key].status">
              {{ testResults[item.config.key].message }}
              <span v-if="testResults[item.config.key].latency_ms">
                {{ testResults[item.config.key].latency_ms }} ms
              </span>
            </p>
          </article>
        </div>
      </section>

      <section class="config-section">
        <header class="section-header">
          <div>
            <p class="section-kicker">03 Agent Models</p>
            <h2>高级智能体模型配置</h2>
            <span>按智能体职责拆分模型，方便在能力、速度和成本之间做阶段化选择。</span>
          </div>
        </header>

        <div class="model-grid">
          <article v-for="item in agentModels" :key="item.key" class="model-card">
            <div class="model-card-header">
              <div>
                <span class="key-chip">{{ item.config.key }}</span>
                <h3>{{ item.label }}</h3>
                <p>{{ item.description }}</p>
              </div>
              <button class="test-button" type="button" :disabled="isTesting(item.config.key)" @click="testModel(item)">
                {{ isTesting(item.config.key) ? '测试中' : '测试连通性' }}
              </button>
            </div>

            <div class="model-fields">
              <label class="field">
                <span>
                  Provider
                  <button
                    class="hint-button"
                    type="button"
                    data-tooltip="该阶段模型使用的 Provider，与后端 YAML 配置键保持一致。"
                    aria-label="智能体 Provider 说明"
                  >
                    ?
                  </button>
                </span>
                <select v-model="item.config.provider">
                  <option v-for="provider in providerOptions" :key="provider.value" :value="provider.value">
                    {{ provider.label }}
                  </option>
                </select>
              </label>

              <label class="field">
                <span>
                  模型名称
                  <button
                    class="hint-button"
                    type="button"
                    data-tooltip="建议复杂分析阶段选择能力更强的模型，检索或简单规划阶段可选择更快模型。"
                    aria-label="智能体模型名称说明"
                  >
                    ?
                  </button>
                </span>
                <input v-model.trim="item.config.model" type="text" placeholder="model-name">
              </label>
            </div>

            <p v-if="testResults[item.config.key]" class="test-result" :class="testResults[item.config.key].status">
              {{ testResults[item.config.key].message }}
              <span v-if="testResults[item.config.key].latency_ms">
                {{ testResults[item.config.key].latency_ms }} ms
              </span>
            </p>
          </article>
        </div>
      </section>

      <section class="config-section">
        <header class="section-header">
          <div>
            <p class="section-kicker">04 Embeddings</p>
            <h2>嵌入模型配置</h2>
            <span>单独管理知识库、Chroma 向量库和论文聚类使用的向量化模型。</span>
          </div>
        </header>

        <div class="model-grid">
          <article v-for="item in embeddingModels" :key="item.key" class="model-card embedding">
            <div class="model-card-header">
              <div>
                <span class="key-chip">{{ item.config.key }}</span>
                <h3>{{ item.label }}</h3>
                <p>{{ item.description }}</p>
              </div>
              <button class="test-button" type="button" :disabled="isTesting(item.config.key)" @click="testModel(item)">
                {{ isTesting(item.config.key) ? '测试中' : '测试连通性' }}
              </button>
            </div>

            <div class="model-fields">
              <label class="field">
                <span>
                  Provider
                  <button
                    class="hint-button"
                    type="button"
                    data-tooltip="嵌入模型调用时使用的服务商，对应 YAML 中的 model-provider。"
                    aria-label="嵌入 Provider 说明"
                  >
                    ?
                  </button>
                </span>
                <select v-model="item.config.provider">
                  <option v-for="provider in providerOptions" :key="provider.value" :value="provider.value">
                    {{ provider.label }}
                  </option>
                </select>
              </label>

              <label class="field">
                <span>
                  模型名称
                  <button
                    class="hint-button"
                    type="button"
                    data-tooltip="嵌入模型 ID 会影响检索召回、相似度计算和聚类质量。"
                    aria-label="嵌入模型名称说明"
                  >
                    ?
                  </button>
                </span>
                <input v-model.trim="item.config.model" type="text" placeholder="embedding-model">
              </label>

              <label class="field">
                <span>
                  向量维度
                  <button
                    class="hint-button"
                    type="button"
                    data-tooltip="需要与实际模型返回维度和向量库配置匹配。"
                    aria-label="嵌入维度说明"
                  >
                    ?
                  </button>
                </span>
                <input v-model.number="item.config.dimension" type="number" min="1" placeholder="1024">
              </label>
            </div>

            <p v-if="testResults[item.config.key]" class="test-result" :class="testResults[item.config.key].status">
              {{ testResults[item.config.key].message }}
              <span v-if="testResults[item.config.key].latency_ms">
                {{ testResults[item.config.key].latency_ms }} ms
              </span>
            </p>
          </article>
        </div>
      </section>
    </template>

    <div v-if="toast.show" class="toast" :class="toast.type">
      {{ toast.message }}
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { configApi } from '../api/config'

const DRAFT_KEY = 'paper-agent-model-settings-draft'

const fallbackProviderTypes = [
  { value: 'openai', label: 'OpenAI', default_base_url: 'https://api.openai.com/v1' },
  { value: 'siliconflow', label: 'SiliconFlow', default_base_url: 'https://api.siliconflow.cn/v1' },
  { value: 'dashscope', label: 'DashScope', default_base_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1' },
  { value: 'ark', label: 'Ark', default_base_url: 'https://ark.cn-beijing.volces.com/api/v3' },
  { value: 'custom', label: 'OpenAI Compatible', default_base_url: '' }
]

const isLoading = ref(false)
const isSaving = ref(false)
const modelsPath = ref('')
const providers = ref([])
const providerTypes = ref([])
const defaultModels = ref([])
const agentModels = ref([])
const embeddingModels = ref([])
const draftMeta = ref(null)
const visibleProviderKeys = reactive({})
const testResults = reactive({})
const toast = ref({
  show: false,
  message: '',
  type: 'success'
})

const providerTypeOptions = computed(() => {
  return providerTypes.value.length > 0 ? providerTypes.value : fallbackProviderTypes
})

const providerOptions = computed(() => {
  return providers.value
    .filter(provider => provider.id)
    .map(provider => ({
      value: provider.id,
      label: `${provider.id} (${getProviderTypeLabel(provider.type)})`
    }))
})

const createLocalId = () => {
  return `provider-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

const normalizeProviders = (items = []) => {
  return items.map(provider => ({
    localId: provider.localId || createLocalId(),
    id: provider.id || '',
    previousId: provider.previousId || provider.id || '',
    type: provider.type || 'custom',
    base_url: provider.base_url || '',
    api_key: provider.api_key || ''
  }))
}

const normalizeModelItems = (items = []) => {
  return items.map(item => ({
    ...item,
    config: {
      key: item.config?.key || item.key,
      provider: item.config?.provider || '',
      model: item.config?.model || '',
      dimension: item.kind === 'embedding' ? normalizeDimension(item.config?.dimension) : undefined
    }
  }))
}

const normalizeDimension = (value) => {
  if (value === '' || value === null || value === undefined) return null
  const parsed = Number(value)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null
}

const applySettings = (settings) => {
  providers.value = normalizeProviders(settings.providers || [])
  providerTypes.value = settings.provider_types || []
  modelsPath.value = settings.models_path || ''
  defaultModels.value = normalizeModelItems(settings.default_models || [])
  agentModels.value = normalizeModelItems(settings.agent_models || [])
  embeddingModels.value = normalizeModelItems(settings.embedding_models || [])
  clearTestResults()
}

const loadSettings = async () => {
  isLoading.value = true
  try {
    const response = await configApi.getModelSettings()
    applySettings(response.data)
  } catch (error) {
    console.error('读取模型配置失败:', error)
    showToast(getErrorMessage(error, '读取模型配置失败'), 'error')
  } finally {
    isLoading.value = false
  }
}

const addProvider = () => {
  const type = 'custom'
  providers.value.push({
    localId: createLocalId(),
    id: suggestProviderId(type),
    previousId: '',
    type,
    base_url: '',
    api_key: ''
  })
}

const removeProvider = (provider) => {
  if (providers.value.length <= 1) return

  const removedId = provider.id
  providers.value = providers.value.filter(item => item.localId !== provider.localId)

  const fallbackProvider = providers.value[0]?.id || ''
  forEachModelItem(item => {
    if (item.config.provider === removedId) {
      item.config.provider = fallbackProvider
    }
  })
}

const suggestProviderId = (type) => {
  const base = type && type !== 'custom' ? type : 'custom'
  let candidate = base
  let index = 2

  while (providers.value.some(provider => provider.id === candidate)) {
    candidate = `${base}-${index}`
    index += 1
  }

  return candidate
}

const handleProviderTypeChange = (provider) => {
  const option = providerTypeOptions.value.find(item => item.value === provider.type)
  if (option?.default_base_url && !provider.base_url) {
    provider.base_url = option.default_base_url
  }
}

const syncProviderReferences = (provider) => {
  const oldId = provider.previousId
  const nextId = provider.id

  if (oldId && nextId && oldId !== nextId) {
    forEachModelItem(item => {
      if (item.config.provider === oldId) {
        item.config.provider = nextId
      }
    })
  }

  provider.previousId = nextId
}

const toggleProviderKey = (localId) => {
  visibleProviderKeys[localId] = !visibleProviderKeys[localId]
}

const getProviderTypeLabel = (type) => {
  return providerTypeOptions.value.find(item => item.value === type)?.label || type || 'Custom'
}

const forEachModelItem = (callback) => {
  ;[defaultModels.value, agentModels.value, embeddingModels.value].forEach(group => {
    group.forEach(callback)
  })
}

const toProviderPayload = (provider) => ({
  id: provider.id,
  type: provider.type,
  base_url: provider.base_url,
  api_key: provider.api_key
})

const toModelPayload = (item) => {
  const payload = {
    key: item.config.key,
    provider: item.config.provider,
    model: item.config.model
  }

  if (item.kind === 'embedding') {
    payload.dimension = normalizeDimension(item.config.dimension)
  }

  return payload
}

const buildPayload = () => ({
  providers: providers.value.map(toProviderPayload),
  default_models: defaultModels.value.map(toModelPayload),
  agent_models: agentModels.value.map(toModelPayload),
  embedding_models: embeddingModels.value.map(toModelPayload)
})

const saveDraft = () => {
  try {
    const draft = {
      savedAt: new Date().toISOString(),
      settings: buildPayload()
    }
    localStorage.setItem(DRAFT_KEY, JSON.stringify(draft))
    draftMeta.value = {
      savedAt: draft.savedAt
    }
    showToast('草稿已保存到本地', 'success')
  } catch (error) {
    console.error('保存草稿失败:', error)
    showToast('保存草稿失败，请检查浏览器存储权限', 'error')
  }
}

const restoreDraft = () => {
  const draft = readDraft()
  if (!draft) {
    showToast('没有可恢复的草稿', 'error')
    return
  }

  providers.value = normalizeProviders(draft.settings.providers || [])
  mergeModelPayload(defaultModels.value, draft.settings.default_models || [])
  mergeModelPayload(agentModels.value, draft.settings.agent_models || [])
  mergeModelPayload(embeddingModels.value, draft.settings.embedding_models || [])
  clearTestResults()
  showToast('已恢复本地草稿', 'success')
}

const clearDraft = () => {
  localStorage.removeItem(DRAFT_KEY)
  draftMeta.value = null
  showToast('已忽略本地草稿', 'success')
}

const readDraft = () => {
  try {
    const rawDraft = localStorage.getItem(DRAFT_KEY)
    if (!rawDraft) return null
    return JSON.parse(rawDraft)
  } catch (error) {
    console.error('读取草稿失败:', error)
    localStorage.removeItem(DRAFT_KEY)
    draftMeta.value = null
    return null
  }
}

const refreshDraftMeta = () => {
  const draft = readDraft()
  draftMeta.value = draft?.savedAt ? { savedAt: draft.savedAt } : null
}

const mergeModelPayload = (targetItems, payloadItems) => {
  const payloadByKey = new Map(payloadItems.map(item => [item.key, item]))
  targetItems.forEach(item => {
    const saved = payloadByKey.get(item.config.key)
    if (!saved) return
    item.config.provider = saved.provider || item.config.provider
    item.config.model = saved.model || item.config.model
    if (item.kind === 'embedding') {
      item.config.dimension = normalizeDimension(saved.dimension)
    }
  })
}

const saveSettings = async () => {
  isSaving.value = true
  try {
    const response = await configApi.saveModelSettings(buildPayload())
    applySettings(response.data.settings)
    showToast(response.data.message || '配置已保存', 'success')
  } catch (error) {
    console.error('保存模型配置失败:', error)
    showToast(getErrorMessage(error, '保存模型配置失败'), 'error')
  } finally {
    isSaving.value = false
  }
}

const testModel = async (item) => {
  const key = item.config.key
  const provider = providers.value.find(candidate => candidate.id === item.config.provider)

  if (!provider) {
    testResults[key] = {
      status: 'failed',
      message: '当前模型选择的 Provider 不存在'
    }
    return
  }

  testResults[key] = {
    status: 'running',
    message: '正在测试当前模型连通性...'
  }

  try {
    const response = await configApi.testModel({
      provider: toProviderPayload(provider),
      model: toModelPayload(item),
      model_type: item.kind === 'embedding' ? 'embedding' : 'llm'
    })
    testResults[key] = response.data
  } catch (error) {
    console.error('模型连通性测试失败:', error)
    testResults[key] = {
      status: 'failed',
      message: getErrorMessage(error, '模型连通性测试失败')
    }
  }
}

const isTesting = (key) => {
  return testResults[key]?.status === 'running'
}

const clearTestResults = () => {
  Object.keys(testResults).forEach(key => {
    delete testResults[key]
  })
}

const getErrorMessage = (error, fallback) => {
  const detail = error?.response?.data?.detail
  if (typeof detail === 'string') return detail
  if (detail?.message) return detail.message
  if (error?.response?.data?.message) return error.response.data.message
  return fallback
}

const showToast = (message, type = 'success') => {
  toast.value = {
    show: true,
    message,
    type
  }

  window.setTimeout(() => {
    toast.value.show = false
  }, 2600)
}

const formatDate = (dateString) => {
  if (!dateString) return '未知时间'
  return new Date(dateString).toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  })
}

onMounted(async () => {
  refreshDraftMeta()
  await loadSettings()
})
</script>

<style scoped>
.config-page {
  height: 100%;
  min-height: 0;
  overflow-y: auto;
  padding: 18px;
  background:
    radial-gradient(circle at top left, color-mix(in srgb, var(--pa-primary) 14%, transparent), transparent 34%),
    linear-gradient(135deg, var(--pa-bg), var(--pa-surface-soft));
  color: var(--pa-text);
}

.config-hero,
.config-section,
.draft-banner,
.center-state {
  border: 1px solid var(--pa-border);
  border-radius: 12px;
  background: color-mix(in srgb, var(--pa-surface) 94%, transparent);
  box-shadow: var(--pa-shadow);
}

.config-hero {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
  margin-bottom: 14px;
  padding: 20px;
}

.hero-copy {
  display: grid;
  gap: 8px;
  max-width: 760px;
}

.eyebrow,
.section-kicker,
.card-kicker {
  margin: 0;
  color: var(--pa-primary);
  font-size: 0.73rem;
  font-weight: 900;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.config-hero h1,
.section-header h2,
.provider-card h3,
.model-card h3 {
  margin: 0;
  color: var(--pa-text);
}

.config-hero h1 {
  font-size: clamp(1.6rem, 2.4vw, 2.2rem);
  line-height: 1.16;
}

.hero-copy p {
  margin: 0;
  color: var(--pa-text-muted);
  line-height: 1.65;
}

.path-pill,
.key-chip {
  display: inline-flex;
  align-items: center;
  width: fit-content;
  min-height: 26px;
  padding: 0 10px;
  border: 1px solid var(--pa-border);
  border-radius: 999px;
  background: var(--pa-surface-soft);
  color: var(--pa-text-muted);
  font-size: 0.76rem;
  font-weight: 800;
}

.hero-actions,
.draft-actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
}

.primary-button,
.secondary-button,
.danger-button,
.test-button,
.draft-actions button,
.password-field button {
  min-height: 36px;
  border-radius: 8px;
  font: inherit;
  font-weight: 850;
  cursor: pointer;
}

.primary-button {
  padding: 0 15px;
  border: 1px solid var(--pa-primary);
  background: var(--pa-primary);
  color: var(--pa-surface);
}

.secondary-button,
.test-button,
.draft-actions button,
.password-field button {
  padding: 0 12px;
  border: 1px solid var(--pa-border);
  background: var(--pa-surface);
  color: var(--pa-text-muted);
}

.danger-button {
  padding: 0 12px;
  border: 1px solid var(--pa-danger-soft);
  background: var(--pa-danger-soft);
  color: var(--pa-danger);
}

.primary-button:disabled,
.secondary-button:disabled,
.danger-button:disabled,
.test-button:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.secondary-button:hover,
.test-button:hover,
.draft-actions button:hover,
.password-field button:hover {
  border-color: var(--pa-border-strong);
  color: var(--pa-text);
}

.draft-banner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  margin-bottom: 14px;
  padding: 13px 16px;
}

.draft-banner div:first-child {
  display: grid;
  gap: 3px;
}

.draft-banner strong {
  color: var(--pa-warning);
}

.draft-banner span {
  color: var(--pa-text-muted);
  font-size: 0.86rem;
}

.center-state {
  display: grid;
  justify-items: center;
  align-content: center;
  gap: 12px;
  min-height: 360px;
  color: var(--pa-text-muted);
}

.spinner {
  width: 42px;
  height: 42px;
  border: 4px solid var(--pa-border);
  border-top-color: var(--pa-primary);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.config-section {
  display: grid;
  gap: 16px;
  margin-bottom: 14px;
  padding: 18px;
}

.section-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
}

.section-header div {
  display: grid;
  gap: 5px;
}

.section-header h2 {
  font-size: 1.18rem;
}

.section-header span {
  color: var(--pa-text-muted);
  font-size: 0.9rem;
  line-height: 1.6;
}

.provider-grid,
.model-grid {
  display: grid;
  gap: 14px;
}

.provider-grid {
  grid-template-columns: repeat(auto-fit, minmax(310px, 1fr));
}

.model-grid {
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
}

.default-grid {
  grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
}

.provider-card,
.model-card {
  display: grid;
  gap: 14px;
  padding: 15px;
  border: 1px solid var(--pa-border);
  border-radius: 10px;
  background: var(--pa-surface);
}

.provider-card {
  position: relative;
  overflow: hidden;
}

.provider-card::before {
  position: absolute;
  inset: 0 auto 0 0;
  width: 4px;
  background: linear-gradient(180deg, var(--pa-primary), var(--pa-success));
  content: '';
}

.card-top,
.model-card-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.card-top h3,
.model-card h3 {
  margin-top: 4px;
  font-size: 1rem;
}

.model-card-header p {
  margin: 8px 0 0;
  color: var(--pa-text-muted);
  font-size: 0.84rem;
  line-height: 1.55;
}

.model-card.embedding {
  background:
    linear-gradient(180deg, color-mix(in srgb, var(--pa-primary-soft) 42%, transparent), transparent 120px),
    var(--pa-surface);
}

.form-grid,
.model-fields {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.field {
  display: grid;
  gap: 7px;
  min-width: 0;
}

.field.wide {
  grid-column: 1 / -1;
}

.field > span {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--pa-text-muted);
  font-size: 0.8rem;
  font-weight: 850;
}

input,
select {
  width: 100%;
  min-height: 38px;
  border: 1px solid var(--pa-border);
  border-radius: 8px;
  outline: none;
  background: var(--pa-input-bg);
  color: var(--pa-text);
}

input {
  padding: 0 10px;
}

select {
  padding: 0 9px;
}

input::placeholder {
  color: var(--pa-text-subtle);
}

input:focus,
select:focus {
  border-color: var(--pa-primary);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--pa-primary) 18%, transparent);
}

.password-field {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 8px;
}

.hint-button {
  display: inline-grid;
  place-items: center;
  position: relative;
  width: 19px;
  height: 19px;
  border: 1px solid var(--pa-border);
  border-radius: 999px;
  background: var(--pa-surface-soft);
  color: var(--pa-text-muted);
  font-size: 0.72rem;
  font-weight: 900;
  cursor: help;
}

.hint-button::after {
  position: absolute;
  left: 50%;
  bottom: calc(100% + 8px);
  z-index: 30;
  width: max-content;
  max-width: 260px;
  padding: 8px 10px;
  border: 1px solid var(--pa-border);
  border-radius: 8px;
  background: var(--pa-text);
  color: var(--pa-surface);
  box-shadow: var(--pa-shadow);
  content: attr(data-tooltip);
  font-size: 0.76rem;
  font-weight: 700;
  line-height: 1.45;
  opacity: 0;
  pointer-events: none;
  text-align: left;
  transform: translate(-50%, 4px);
  transition: opacity 0.15s ease, transform 0.15s ease;
  white-space: normal;
}

.hint-button:hover::after,
.hint-button:focus-visible::after {
  opacity: 1;
  transform: translate(-50%, 0);
}

.test-result {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin: 0;
  padding: 9px 10px;
  border-radius: 8px;
  font-size: 0.8rem;
  font-weight: 800;
  line-height: 1.5;
}

.test-result.running {
  background: var(--pa-warning-soft);
  color: var(--pa-warning);
}

.test-result.success {
  background: var(--pa-success-soft);
  color: var(--pa-success);
}

.test-result.failed {
  background: var(--pa-danger-soft);
  color: var(--pa-danger);
}

.test-result span {
  flex: 0 0 auto;
  color: currentColor;
  opacity: 0.8;
}

.toast {
  position: fixed;
  top: 18px;
  right: 18px;
  z-index: 2000;
  max-width: min(420px, calc(100vw - 36px));
  padding: 12px 16px;
  border-radius: 10px;
  box-shadow: var(--pa-shadow);
  color: var(--pa-surface);
  font-size: 0.88rem;
  font-weight: 850;
}

.toast.success {
  background: var(--pa-success);
}

.toast.error {
  background: var(--pa-danger);
}

.config-page::-webkit-scrollbar {
  width: 8px;
}

.config-page::-webkit-scrollbar-thumb {
  background: var(--pa-scrollbar);
  border-radius: 999px;
}

.config-page::-webkit-scrollbar-track {
  background: transparent;
}

@media (max-width: 900px) {
  .config-hero,
  .section-header,
  .draft-banner {
    align-items: stretch;
    flex-direction: column;
  }

  .hero-actions,
  .draft-actions {
    justify-content: stretch;
  }

  .hero-actions button,
  .draft-actions button,
  .section-header .primary-button {
    width: 100%;
  }
}

@media (max-width: 640px) {
  .config-page {
    padding: 10px;
  }

  .config-hero,
  .config-section {
    padding: 14px;
  }

  .form-grid,
  .model-fields,
  .provider-grid,
  .model-grid,
  .default-grid {
    grid-template-columns: 1fr;
  }

  .card-top,
  .model-card-header {
    flex-direction: column;
  }

  .test-button,
  .danger-button {
    width: 100%;
  }
}
</style>
