<template>
  <div
    class="config-page"
    @focusin="handleTooltipEnter"
    @focusout="hideTooltip"
    @mouseleave="hideTooltip"
    @mouseover="handleTooltipEnter"
    @scroll="hideTooltip"
  >
    <header class="config-hero">
      <div class="hero-copy">
        <p class="eyebrow">Runtime Configuration</p>
        <h1>系统配置</h1>
      </div>

      <div class="hero-actions">
        <button class="secondary-button" type="button" :disabled="isLoading" @click="loadSettings">
          {{ isLoading ? '加载中' : '重新加载' }}
        </button>
        <button class="primary-button" type="button" :disabled="isSaving || isLoading" @click="saveSettings">
          {{ isSaving ? '保存中' : '保存配置' }}
        </button>
      </div>
    </header>

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

        <div class="provider-console">
          <aside class="provider-list-panel" aria-label="模型服务商列表">
            <button
              v-for="provider in providers"
              :key="provider.localId"
              class="provider-list-item"
              :class="{ active: selectedProviderLocalId === provider.localId }"
              type="button"
              @click="selectProvider(provider)"
            >
              <span class="provider-logo" :data-provider="provider.type">
                {{ getProviderInitial(provider) }}
              </span>
              <span class="provider-list-copy">
                <strong>{{ getProviderDisplayName(provider) }}</strong>
              </span>
            </button>
          </aside>

          <section v-if="selectedProvider" class="provider-detail-panel">
            <header class="provider-detail-header">
              <span class="provider-logo large" :data-provider="selectedProvider.type">
                {{ getProviderInitial(selectedProvider) }}
              </span>
              <div>
                <h3>{{ getProviderDisplayName(selectedProvider) }}</h3>
                <p>在服务商控制台获取 API 密钥，并配置 OpenAI-compatible 调用地址。</p>
              </div>
              <button
                class="provider-link-button"
                type="button"
                :disabled="!getProviderConsoleUrl(selectedProvider)"
                @click="openProviderConsole(selectedProvider)"
              >
                获取 API Key
              </button>
            </header>

            <div class="provider-detail-fields">
              <label class="field provider-field-wide">
                <span>
                  API Key
                  <button
                    class="hint-button"
                    type="button"
                    data-tooltip="这里只填写真实 Key；保存后后端会自动写入 .env，并在 models.yaml 中使用 Provider ID 对应的环境变量名。"
                    aria-label="API Key 说明"
                  >
                  </button>
                </span>
                <div class="password-field">
                  <input
                    v-model.trim="selectedProvider.api_key"
                    :type="visibleProviderKeys[selectedProvider.localId] ? 'text' : 'password'"
                    placeholder="sk-..."
                    autocomplete="off"
                  >
                  <button type="button" @click="toggleProviderKey(selectedProvider.localId)">
                    {{ visibleProviderKeys[selectedProvider.localId] ? '隐藏' : '显示' }}
                  </button>
                </div>
              </label>

              <label class="field provider-field-wide">
                <span>
                  API URL
                  <button
                    class="hint-button"
                    type="button"
                    data-tooltip="服务商的 OpenAI-compatible base_url，后端会传给模型客户端。"
                    aria-label="API URL 说明"
                  >
                  </button>
                </span>
                <input v-model.trim="selectedProvider.base_url" type="url" placeholder="https://api.example.com/v1">
              </label>

              <label class="field">
                <span>
                  Provider ID
                  <button
                    class="hint-button"
                    type="button"
                    data-tooltip="Provider ID 会写入 YAML 顶层键，并用于生成环境变量名，例如 siliconflow 对应 SILICONFLOW_API_KEY。"
                    aria-label="Provider ID 说明"
                  >
                  </button>
                </span>
                <input
                  v-model.trim="selectedProvider.id"
                  type="text"
                  placeholder="siliconflow"
                  @change="syncProviderReferences(selectedProvider)"
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
                  </button>
                </span>
                <select v-model="selectedProvider.type" @change="handleProviderTypeChange(selectedProvider)">
                  <option v-for="type in providerTypeOptions" :key="type.value" :value="type.value">
                    {{ type.label }}
                  </option>
                </select>
              </label>
            </div>

            <footer class="provider-detail-footer">
              <span>当前配置键：{{ selectedProvider.id || '未命名 Provider' }}</span>
              <button
                class="provider-remove-button"
                type="button"
                :disabled="providers.length <= 1"
                @click="removeProvider(selectedProvider)"
              >
                删除 Provider
              </button>
            </footer>
          </section>
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

        <div class="agent-table-card">
          <div class="agent-table-scroll">
            <table class="agent-model-table">
              <thead>
                <tr>
                  <th scope="col">智能体阶段</th>
                  <th scope="col">配置键名</th>
                  <th scope="col">
                    Provider
                    <button
                      class="hint-button"
                      type="button"
                      data-tooltip="该阶段模型使用的 Provider，与后端 YAML 配置键保持一致。"
                      aria-label="智能体 Provider 说明"
                    >
                    </button>
                  </th>
                  <th scope="col">
                    模型名称
                    <button
                      class="hint-button"
                      type="button"
                      data-tooltip="建议复杂分析阶段选择能力更强的模型，检索或简单规划阶段可选择更快模型。"
                      aria-label="智能体模型名称说明"
                    >
                    </button>
                  </th>
                  <th scope="col">连通性</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="item in agentModels" :key="item.key">
                  <td class="agent-stage-cell" data-label="智能体阶段">
                    <strong>{{ item.label }}</strong>
                    <span>{{ item.description }}</span>
                  </td>
                  <td class="agent-key-cell" data-label="配置键名">
                    <span class="key-chip">{{ item.config.key }}</span>
                  </td>
                  <td data-label="Provider">
                    <select v-model="item.config.provider" @change="handleModelProviderChange(item)">
                      <option v-for="provider in modelProviderOptions(item)" :key="provider.value" :value="provider.value">
                        {{ provider.label }}
                      </option>
                    </select>
                  </td>
                  <td data-label="模型名称">
                    <input
                      v-model.trim="item.config.model"
                      type="text"
                      :placeholder="getModelPlaceholder(item)"
                      :disabled="usesDefaultModel(item)"
                    >
                  </td>
                  <td class="agent-test-cell" data-label="连通性">
                    <button class="test-button" type="button" :disabled="isTesting(item.config.key)" @click="testModel(item)">
                      {{ isTesting(item.config.key) ? '测试中' : '测试' }}
                    </button>
                    <p
                      v-if="testResults[item.config.key]"
                      class="inline-test-result"
                      :class="testResults[item.config.key].status"
                    >
                      {{ testResults[item.config.key].message }}
                      <span v-if="testResults[item.config.key].latency_ms">
                        {{ testResults[item.config.key].latency_ms }} ms
                      </span>
                    </p>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
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
                  </button>
                </span>
                <select v-model="item.config.provider" @change="handleModelProviderChange(item)">
                  <option v-for="provider in modelProviderOptions(item)" :key="provider.value" :value="provider.value">
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
                  </button>
                </span>
                <input
                  v-model.trim="item.config.model"
                  type="text"
                  :placeholder="getModelPlaceholder(item)"
                  :disabled="usesDefaultModel(item)"
                >
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
                  </button>
                </span>
                <input
                  v-model.number="item.config.dimension"
                  type="number"
                  min="1"
                  :placeholder="getDimensionPlaceholder(item)"
                  :disabled="usesDefaultModel(item)"
                >
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
  </div>
</template>

<script setup>
import { computed, onActivated, reactive, ref } from 'vue'
import { configApi } from '../api/config'

const fallbackProviderTypes = [
  { value: 'openai', label: 'OpenAI', default_base_url: 'https://api.openai.com/v1' },
  { value: 'siliconflow', label: 'SiliconFlow', default_base_url: 'https://api.siliconflow.cn/v1' },
  { value: 'dashscope', label: 'DashScope', default_base_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1' },
  { value: 'ark', label: 'Ark', default_base_url: 'https://ark.cn-beijing.volces.com/api/v3' },
  { value: 'custom', label: 'OpenAI Compatible', default_base_url: '' }
]

const DEFAULT_CONFIG_VALUE = ''
const ENV_VAR_NAME_PATTERN = /^[A-Z_][A-Z0-9_]*$/
const defaultModelKeys = new Set(['default-model', 'default-embedding-model'])

const isLoading = ref(false)
const isSaving = ref(false)
const providers = ref([])
const selectedProviderLocalId = ref('')
const providerTypes = ref([])
const defaultModels = ref([])
const agentModels = ref([])
const embeddingModels = ref([])
const visibleProviderKeys = reactive({})
const testResults = reactive({})
const activeTooltip = reactive({
  text: '',
  left: 0,
  top: 0,
  transform: 'translate(-50%, -100%)'
})
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

const modelProviderOptions = (item) => {
  if (isDefaultModelItem(item)) {
    return providerOptions.value
  }

  return [
    { value: DEFAULT_CONFIG_VALUE, label: '默认配置' },
    ...providerOptions.value
  ]
}

const selectedProvider = computed(() => {
  return providers.value.find(provider => provider.localId === selectedProviderLocalId.value) || providers.value[0] || null
})

const providerConsoleLinks = {
  openai: 'https://platform.openai.com/api-keys',
  siliconflow: 'https://cloud.siliconflow.cn/account/ak',
  dashscope: 'https://dashscope.console.aliyun.com/apiKey',
  ark: 'https://console.volcengine.com/ark'
}

const createLocalId = () => {
  return `provider-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

const normalizeProviderApiKey = (provider) => {
  const rawApiKey = provider.api_key || ''
  const apiKeyEnv = provider.api_key_env || (ENV_VAR_NAME_PATTERN.test(rawApiKey) ? rawApiKey : '')

  return {
    apiKey: ENV_VAR_NAME_PATTERN.test(rawApiKey) ? '' : rawApiKey,
    apiKeyEnv
  }
}

const normalizeProviders = (items = []) => {
  return items.map(provider => {
    const { apiKey, apiKeyEnv } = normalizeProviderApiKey(provider)

    return {
      localId: provider.localId || createLocalId(),
      id: provider.id || '',
      previousId: provider.previousId || provider.id || '',
      type: provider.type || 'custom',
      base_url: provider.base_url || '',
      api_key: apiKey,
      api_key_env: apiKeyEnv
    }
  })
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
  ensureSelectedProvider()
  providerTypes.value = settings.provider_types || []
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
  const provider = {
    localId: createLocalId(),
    id: suggestProviderId(type),
    previousId: '',
    type,
    base_url: '',
    api_key: ''
  }

  providers.value.push(provider)
  selectedProviderLocalId.value = provider.localId
}

const removeProvider = (provider) => {
  if (providers.value.length <= 1) return

  const removedId = provider.id
  providers.value = providers.value.filter(item => item.localId !== provider.localId)
  if (selectedProviderLocalId.value === provider.localId) {
    selectedProviderLocalId.value = providers.value[0]?.localId || ''
  }

  const fallbackProvider = providers.value[0]?.id || ''
  forEachModelItem(item => {
    if (item.config.provider === removedId) {
      item.config.provider = isDefaultModelItem(item) ? fallbackProvider : DEFAULT_CONFIG_VALUE
      handleModelProviderChange(item)
    }
  })
}

const ensureSelectedProvider = () => {
  if (!providers.value.length) {
    selectedProviderLocalId.value = ''
    return
  }

  const exists = providers.value.some(provider => provider.localId === selectedProviderLocalId.value)
  if (!exists) {
    selectedProviderLocalId.value = providers.value[0].localId
  }
}

const selectProvider = (provider) => {
  selectedProviderLocalId.value = provider.localId
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

const getProviderDisplayName = (provider) => {
  if (!provider) return 'Provider'
  const typeLabel = getProviderTypeLabel(provider.type)
  return provider.id || typeLabel || 'Provider'
}

const getProviderInitial = (provider) => {
  const label = getProviderDisplayName(provider)
  return label.slice(0, 2).toUpperCase()
}

const getProviderConsoleUrl = (provider) => {
  if (!provider) return ''
  return providerConsoleLinks[provider.type] || ''
}

const openProviderConsole = (provider) => {
  const url = getProviderConsoleUrl(provider)
  if (!url) return
  window.open(url, '_blank', 'noopener,noreferrer')
}

const isDefaultModelItem = (item) => {
  return defaultModelKeys.has(item?.config?.key || item?.key)
}

const usesDefaultModel = (item) => {
  return !isDefaultModelItem(item) && !item.config.provider
}

const getDefaultModelKey = (kind) => {
  return kind === 'embedding' ? 'default-embedding-model' : 'default-model'
}

const getDefaultModelItem = (kind) => {
  const key = getDefaultModelKey(kind)
  return defaultModels.value.find(item => item.config.key === key) || null
}

const getModelPlaceholder = (item) => {
  if (!usesDefaultModel(item)) {
    return item.kind === 'embedding' ? 'embedding-model' : 'model-name'
  }

  const defaultModel = getDefaultModelItem(item.kind)?.config.model
  return defaultModel ? `使用默认配置：${defaultModel}` : '使用默认配置'
}

const getDimensionPlaceholder = (item) => {
  if (!usesDefaultModel(item)) {
    return '1024'
  }

  const defaultDimension = normalizeDimension(getDefaultModelItem(item.kind)?.config.dimension)
  return defaultDimension ? `使用默认维度：${defaultDimension}` : '1024'
}

const handleModelProviderChange = (item) => {
  if (!usesDefaultModel(item)) return

  item.config.model = ''
  if (item.kind === 'embedding') {
    item.config.dimension = null
  }
  delete testResults[item.config.key]
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
  const usesDefault = usesDefaultModel(item)
  const payload = {
    key: item.config.key,
    provider: usesDefault ? DEFAULT_CONFIG_VALUE : item.config.provider,
    model: usesDefault ? '' : item.config.model
  }

  if (item.kind === 'embedding') {
    payload.dimension = usesDefault ? null : normalizeDimension(item.config.dimension)
  }

  return payload
}

const getEffectiveModelPayload = (item) => {
  if (!usesDefaultModel(item)) {
    return toModelPayload(item)
  }

  const defaultItem = getDefaultModelItem(item.kind)
  return defaultItem ? toModelPayload(defaultItem) : toModelPayload(item)
}

const buildPayload = () => ({
  providers: providers.value.map(toProviderPayload),
  default_models: defaultModels.value.map(toModelPayload),
  agent_models: agentModels.value.map(toModelPayload),
  embedding_models: embeddingModels.value.map(toModelPayload)
})

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
  const modelPayload = getEffectiveModelPayload(item)
  const provider = providers.value.find(candidate => candidate.id === modelPayload.provider)

  if (!provider) {
    testResults[key] = {
      status: 'failed',
      message: usesDefaultModel(item) ? '默认配置的 Provider 不存在' : '当前模型选择的 Provider 不存在'
    }
    return
  }

  if (!provider.api_key || ENV_VAR_NAME_PATTERN.test(provider.api_key)) {
    testResults[key] = {
      status: 'failed',
      message: 'API Key 请填写真实密钥，不要填写环境变量名'
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
      model: modelPayload,
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

const handleTooltipEnter = (event) => {
  const target = event.target
  if (!(target instanceof Element)) {
    hideTooltip()
    return
  }

  const tooltipTarget = getTooltipTarget(target)
  if (!tooltipTarget) {
    if (event.type === 'mouseover') {
      hideTooltip()
    }
    return
  }

  showTooltip(tooltipTarget.trigger, tooltipTarget.text)
}

const getTooltipTarget = (target) => {
  const hintButton = target.closest('.hint-button')
  if (hintButton instanceof HTMLElement) {
    const text = hintButton.dataset.tooltip
    return text ? { trigger: hintButton, text } : null
  }

  if (!target.matches('input, select, textarea')) {
    return null
  }

  const field = target.closest('.field')
  const fieldHint = field?.querySelector('.hint-button')
  if (fieldHint instanceof HTMLElement && fieldHint.dataset.tooltip) {
    return { trigger: target, text: fieldHint.dataset.tooltip }
  }

  const cell = target.closest('td')
  const table = target.closest('table')
  if (cell instanceof HTMLTableCellElement && table) {
    const header = table.querySelectorAll('thead th')[cell.cellIndex]
    const tableHint = header?.querySelector('.hint-button')
    if (tableHint instanceof HTMLElement && tableHint.dataset.tooltip) {
      return { trigger: target, text: tableHint.dataset.tooltip }
    }
  }

  return null
}

const showTooltip = (trigger, text) => {
  const rect = trigger.getBoundingClientRect()
  const estimatedWidth = Math.min(320, Math.max(220, window.innerWidth - 32))
  const center = rect.left + rect.width / 2
  const left = Math.min(
    Math.max(center, estimatedWidth / 2 + 12),
    window.innerWidth - estimatedWidth / 2 - 12
  )

  const shouldShowBelow = rect.top < 84
  activeTooltip.text = text
  activeTooltip.left = left
  activeTooltip.top = shouldShowBelow ? rect.bottom + 10 : rect.top - 10
  activeTooltip.transform = shouldShowBelow ? 'translate(-50%, 0)' : 'translate(-50%, -100%)'
}

const hideTooltip = () => {
  activeTooltip.text = ''
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

onActivated(async () => {
  await loadSettings()
})
</script>

<style scoped>
.config-page {
  height: 100%;
  min-height: 0;
  overflow-y: auto;
  background:
    radial-gradient(circle at top left, color-mix(in srgb, var(--pa-primary) 14%, transparent), transparent 34%),
    linear-gradient(135deg, var(--pa-bg), var(--pa-surface-soft));
  color: var(--pa-text);
}

.config-section,
.center-state {
  border: 1px solid var(--pa-border);
  border-radius: 12px;
  background: color-mix(in srgb, var(--pa-surface) 94%, transparent);
  box-shadow: var(--pa-shadow);
}

.config-hero {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  min-height: 72px;
  padding: 14px 22px;
  border-bottom: 1px solid var(--pa-border);
  background: var(--pa-surface);
}

.hero-copy {
  display: grid;
  gap: 5px;
  min-width: 0;
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
.model-card h3 {
  margin: 0;
  color: var(--pa-text);
}

.config-hero h1 {
  font-size: 1.25rem;
  line-height: 1.25;
}

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

.hero-actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
}

.primary-button,
.secondary-button,
.danger-button,
.test-button,
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
.password-field button:hover {
  border-color: var(--pa-border-strong);
  color: var(--pa-text);
}

.center-state {
  display: grid;
  justify-items: center;
  align-content: center;
  gap: 12px;
  min-height: 360px;
  margin: 14px 18px;
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
  margin: 14px 18px;
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

.model-grid {
  display: grid;
  gap: 14px;
}

.model-grid {
  grid-template-columns: 1fr;
}

.default-grid {
  grid-template-columns: 1fr;
}

.model-card {
  display: grid;
  gap: 14px;
  padding: 15px;
  border: 1px solid var(--pa-border);
  border-radius: 10px;
  background: var(--pa-surface);
}

.model-card-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

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

.provider-console {
  display: grid;
  grid-template-columns: minmax(220px, 270px) minmax(0, 1fr);
  min-height: 360px;
  border: 1px solid var(--pa-border);
  border-radius: 12px;
  background: var(--pa-surface);
  overflow: hidden;
}

.provider-list-panel {
  display: grid;
  align-content: start;
  gap: 8px;
  padding: 12px;
  border-right: 1px solid var(--pa-border);
  background: color-mix(in srgb, var(--pa-surface-soft) 72%, var(--pa-surface));
}

.provider-list-item {
  display: grid;
  grid-template-columns: 34px minmax(0, 1fr);
  align-items: center;
  gap: 10px;
  width: 100%;
  min-height: 62px;
  padding: 9px 10px;
  border: 1px solid transparent;
  border-radius: 12px;
  background: transparent;
  color: var(--pa-text);
  text-align: left;
  cursor: pointer;
}

.provider-list-item:hover {
  background: var(--pa-surface);
}

.provider-list-item.active {
  border-color: var(--pa-border-strong);
  background: var(--pa-surface);
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--pa-text) 6%, transparent);
}

.provider-logo {
  display: grid;
  place-items: center;
  width: 28px;
  height: 28px;
  border-radius: 9px;
  background: linear-gradient(135deg, #56d6b2, #5e7cff 58%, #b975ff);
  color: #ffffff;
  font-size: 0.62rem;
  font-weight: 900;
  letter-spacing: -0.02em;
}

.provider-logo[data-provider='openai'] {
  background: #111827;
}

.provider-logo[data-provider='siliconflow'] {
  background: linear-gradient(135deg, #2f8da3, #7fc7d8);
}

.provider-logo[data-provider='dashscope'] {
  background: linear-gradient(135deg, #1f6f82, #86d7b8);
}

.provider-logo[data-provider='ark'] {
  background: linear-gradient(135deg, #2b5cab, #8fb2ff);
}

.provider-logo.large {
  width: 40px;
  height: 40px;
  border-radius: 12px;
  font-size: 0.78rem;
}

.provider-list-copy {
  display: block;
  min-width: 0;
}

.provider-list-copy strong {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.provider-list-copy strong {
  font-size: 0.92rem;
}

.provider-detail-panel {
  display: grid;
  align-content: start;
  gap: 22px;
  padding: 24px 30px 22px;
}

.provider-detail-header {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  align-items: start;
  gap: 13px;
}

.provider-detail-header h3 {
  margin: 0 0 6px;
  color: var(--pa-text);
  font-size: 1.24rem;
}

.provider-detail-header p {
  margin: 0;
  color: var(--pa-text-muted);
  font-size: 0.88rem;
  line-height: 1.55;
}

.provider-link-button {
  min-height: 32px;
  padding: 0 2px;
  border: 0;
  background: transparent;
  color: var(--pa-text-muted);
  font: inherit;
  font-size: 0.8rem;
  font-weight: 850;
  cursor: pointer;
}

.provider-link-button:hover {
  color: var(--pa-primary);
}

.provider-link-button:disabled {
  cursor: not-allowed;
  opacity: 0.42;
}

.provider-detail-fields {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px 16px;
}

.provider-field-wide {
  grid-column: 1 / -1;
}

.provider-detail-fields input,
.provider-detail-fields select {
  min-height: 44px;
  border-radius: 10px;
}

.provider-detail-fields .password-field button {
  min-height: 44px;
}

.provider-detail-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding-top: 6px;
  color: var(--pa-text-subtle);
  font-size: 0.78rem;
  font-weight: 800;
}

.provider-remove-button {
  min-height: 34px;
  padding: 0 11px;
  border: 1px solid transparent;
  border-radius: 8px;
  background: transparent;
  color: var(--pa-danger);
  font: inherit;
  font-weight: 850;
  cursor: pointer;
}

.provider-remove-button:hover {
  background: var(--pa-danger-soft);
}

.provider-remove-button:disabled {
  cursor: not-allowed;
  opacity: 0.42;
}

.agent-table-card {
  border: 1px solid var(--pa-border);
  border-radius: 10px;
  background: var(--pa-surface);
  overflow: hidden;
}

.agent-table-scroll {
  overflow-x: auto;
}

.agent-model-table {
  width: 100%;
  min-width: 920px;
  border-collapse: collapse;
}

.agent-model-table th,
.agent-model-table td {
  padding: 13px 14px;
  border-bottom: 1px solid var(--pa-border);
  text-align: left;
  vertical-align: top;
}

.agent-model-table th {
  background:
    linear-gradient(180deg, color-mix(in srgb, var(--pa-primary-soft) 66%, transparent), transparent),
    var(--pa-surface-soft);
  color: var(--pa-text-muted);
  font-size: 0.78rem;
  font-weight: 900;
  letter-spacing: 0.02em;
  white-space: nowrap;
}

.agent-model-table th .hint-button {
  margin-left: 5px;
  vertical-align: middle;
}

.agent-model-table tbody tr {
  transition: background 0.16s ease;
}

.agent-model-table tbody tr:hover {
  background: color-mix(in srgb, var(--pa-primary-soft) 24%, transparent);
}

.agent-model-table tbody tr:last-child td {
  border-bottom: 0;
}

.agent-model-table select,
.agent-model-table input {
  min-height: 36px;
}

.agent-stage-cell {
  width: 27%;
}

.agent-stage-cell strong {
  display: block;
  margin-bottom: 5px;
  color: var(--pa-text);
  font-size: 0.94rem;
}

.agent-stage-cell span {
  display: block;
  color: var(--pa-text-muted);
  font-size: 0.78rem;
  line-height: 1.5;
}

.agent-key-cell {
  width: 19%;
}

.agent-test-cell {
  min-width: 150px;
}

.agent-test-cell .test-button {
  width: fit-content;
  min-height: 34px;
}

.inline-test-result {
  display: grid;
  gap: 3px;
  max-width: 260px;
  margin: 0;
  margin-top: 8px;
  padding: 7px 9px;
  border-radius: 8px;
  font-size: 0.75rem;
  font-weight: 800;
  line-height: 1.45;
}

.inline-test-result.running {
  background: var(--pa-warning-soft);
  color: var(--pa-warning);
}

.inline-test-result.success {
  background: var(--pa-success-soft);
  color: var(--pa-success);
}

.inline-test-result.failed {
  background: var(--pa-danger-soft);
  color: var(--pa-danger);
}

.inline-test-result span {
  opacity: 0.78;
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

input:disabled,
select:disabled {
  cursor: not-allowed;
  opacity: 0.72;
}

.password-field {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 8px;
}

.hint-button {
  display: inline-grid;
  place-items: center;
  width: 19px;
  height: 19px;
  flex: 0 0 19px;
  overflow: hidden;
  border: 1px solid var(--pa-border);
  border-radius: 999px;
  background: var(--pa-surface-soft);
  color: var(--pa-text-muted);
  font-size: 0;
  font-weight: 900;
  line-height: 1;
  cursor: help;
}

.hint-button::before {
  content: '?';
  color: currentColor;
  font-size: 0.72rem;
  line-height: 1;
}

.hint-button:hover,
.hint-button:focus-visible {
  border-color: var(--pa-primary);
  color: var(--pa-primary);
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

.global-tooltip {
  position: fixed;
  z-index: 10000;
  width: max-content;
  max-width: min(320px, calc(100vw - 24px));
  padding: 9px 11px;
  border: 1px solid color-mix(in srgb, var(--pa-text) 12%, transparent);
  border-radius: 9px;
  background: var(--pa-text);
  color: var(--pa-surface);
  box-shadow: 0 14px 36px rgba(15, 23, 42, 0.22);
  font-size: 0.78rem;
  font-weight: 750;
  line-height: 1.5;
  pointer-events: none;
  text-align: left;
  white-space: normal;
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
  .section-header {
    align-items: stretch;
    flex-direction: column;
  }

  .hero-actions {
    justify-content: stretch;
  }

  .hero-actions button,
  .section-header .primary-button {
    width: 100%;
  }

  .provider-console {
    grid-template-columns: 1fr;
  }

  .provider-list-panel {
    grid-auto-flow: column;
    grid-auto-columns: minmax(190px, 1fr);
    overflow-x: auto;
    border-right: 0;
    border-bottom: 1px solid var(--pa-border);
  }
}

@media (max-width: 640px) {
  .config-hero,
  .config-section {
    padding: 14px;
  }

  .config-section,
  .center-state {
    margin: 10px;
  }

  .form-grid,
  .model-fields,
  .model-grid,
  .default-grid {
    grid-template-columns: 1fr;
  }

  .provider-detail-panel {
    padding: 18px;
  }

  .provider-detail-header {
    grid-template-columns: auto minmax(0, 1fr);
  }

  .provider-link-button {
    grid-column: 1 / -1;
    justify-self: start;
  }

  .provider-detail-fields {
    grid-template-columns: 1fr;
  }

  .provider-detail-footer {
    align-items: stretch;
    flex-direction: column;
  }

  .model-card-header {
    flex-direction: column;
  }

  .test-button,
  .danger-button {
    width: 100%;
  }
}
</style>
