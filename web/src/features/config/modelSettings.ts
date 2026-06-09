export type ProviderTypeOption = {
  value: string
  label: string
  default_base_url?: string
}

export type ProviderConfig = {
  localId: string
  id: string
  previousId: string
  type: string
  base_url: string
  api_key: string
  api_key_env: string
}

export type ModelKind = 'llm' | 'embedding'

export type ModelItem = {
  key: string
  label?: string
  description?: string
  kind: ModelKind
  config: {
    key: string
    provider: string
    model: string
    dimension?: number | null
  }
}

type RawModelItem = Omit<Partial<ModelItem>, 'config'> & {
  config?: Omit<Partial<ModelItem['config']>, 'dimension'> & {
    dimension?: unknown
  }
}

export type ModelSettingsResponse = {
  provider_types?: ProviderTypeOption[]
  providers?: Array<Partial<ProviderConfig>>
  default_models?: RawModelItem[]
  agent_models?: RawModelItem[]
  embedding_models?: RawModelItem[]
}

export type ModelSettingsState = {
  providerTypes: ProviderTypeOption[]
  providers: ProviderConfig[]
  defaultModels: ModelItem[]
  agentModels: ModelItem[]
  embeddingModels: ModelItem[]
}

export const fallbackProviderTypes: ProviderTypeOption[] = [
  { value: 'openai', label: 'OpenAI', default_base_url: 'https://api.openai.com/v1' },
  { value: 'siliconflow', label: 'SiliconFlow', default_base_url: 'https://api.siliconflow.cn/v1' },
  { value: 'dashscope', label: 'DashScope', default_base_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1' },
  { value: 'ark', label: 'Ark', default_base_url: 'https://ark.cn-beijing.volces.com/api/v3' },
  { value: 'custom', label: 'OpenAI Compatible', default_base_url: '' }
]

export const DEFAULT_CONFIG_VALUE = ''
export const ENV_VAR_NAME_PATTERN = /^[A-Z_][A-Z0-9_]*$/
export const defaultModelKeys = new Set(['default-model', 'default-embedding-model'])

export const createLocalId = (): string => `provider-${Date.now()}-${Math.random().toString(16).slice(2)}`

export const normalizeDimension = (value: unknown): number | null => {
  if (value === '' || value === null || value === undefined) return null
  const parsed = Number(value)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null
}

const normalizeProviderApiKey = (provider: Partial<ProviderConfig>) => {
  const rawApiKey = provider.api_key || ''
  const apiKeyEnv = provider.api_key_env || (ENV_VAR_NAME_PATTERN.test(rawApiKey) ? rawApiKey : '')
  return {
    apiKey: ENV_VAR_NAME_PATTERN.test(rawApiKey) ? '' : rawApiKey,
    apiKeyEnv
  }
}

export const normalizeProviders = (items: Array<Partial<ProviderConfig>> = []): ProviderConfig[] =>
  items.map((provider) => {
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

export const normalizeModelItems = (items: RawModelItem[] = []): ModelItem[] =>
  items.map((item) => {
    const key = item.config?.key || item.key || ''
    const kind: ModelKind = item.kind === 'embedding' ? 'embedding' : 'llm'
    return {
      key,
      label: item.label,
      description: item.description,
      kind,
      config: {
        key,
        provider: item.config?.provider || '',
        model: item.config?.model || '',
        dimension: kind === 'embedding' ? normalizeDimension(item.config?.dimension) : undefined
      }
    }
  })

export const applySettings = (settings: ModelSettingsResponse): ModelSettingsState => ({
  providerTypes: settings.provider_types || [],
  providers: normalizeProviders(settings.providers || []),
  defaultModels: normalizeModelItems(settings.default_models || []),
  agentModels: normalizeModelItems(settings.agent_models || []),
  embeddingModels: normalizeModelItems(settings.embedding_models || [])
})

export const isDefaultModelItem = (item: ModelItem): boolean => defaultModelKeys.has(item.config.key || item.key)
export const usesDefaultModel = (item: ModelItem): boolean => !isDefaultModelItem(item) && !item.config.provider

export const toProviderPayload = (provider: ProviderConfig) => ({
  id: provider.id,
  type: provider.type,
  base_url: provider.base_url,
  api_key: provider.api_key
})

export const toModelPayload = (item: ModelItem) => {
  const usesDefault = usesDefaultModel(item)
  const payload: { key: string; provider: string; model: string; dimension?: number | null } = {
    key: item.config.key,
    provider: usesDefault ? DEFAULT_CONFIG_VALUE : item.config.provider,
    model: usesDefault ? '' : item.config.model
  }

  if (item.kind === 'embedding') {
    payload.dimension = usesDefault ? null : normalizeDimension(item.config.dimension)
  }

  return payload
}

export const buildModelSettingsPayload = (state: ModelSettingsState) => ({
  providers: state.providers.map(toProviderPayload),
  default_models: state.defaultModels.map(toModelPayload),
  agent_models: state.agentModels.map(toModelPayload),
  embedding_models: state.embeddingModels.map(toModelPayload)
})

export const getProviderTypeLabel = (types: ProviderTypeOption[], type: string): string =>
  (types.length > 0 ? types : fallbackProviderTypes).find((item) => item.value === type)?.label || type || 'Custom'

export const suggestProviderId = (providers: ProviderConfig[], type: string): string => {
  const base = type && type !== 'custom' ? type : 'custom'
  let candidate = base
  let index = 2

  while (providers.some((provider) => provider.id === candidate)) {
    candidate = `${base}-${index}`
    index += 1
  }

  return candidate
}

const syncModelGroupProvider = (items: ModelItem[], oldId: string, nextId: string): ModelItem[] =>
  items.map((item) => (item.config.provider === oldId ? { ...item, config: { ...item.config, provider: nextId } } : item))

export const syncProviderReferences = (state: ModelSettingsState, oldId: string, nextId: string): ModelSettingsState => {
  if (!oldId || !nextId || oldId === nextId) return state
  return {
    ...state,
    defaultModels: syncModelGroupProvider(state.defaultModels, oldId, nextId),
    agentModels: syncModelGroupProvider(state.agentModels, oldId, nextId),
    embeddingModels: syncModelGroupProvider(state.embeddingModels, oldId, nextId)
  }
}

export const removeProviderFromState = (state: ModelSettingsState, localId: string): ModelSettingsState => {
  const removedProvider = state.providers.find((provider) => provider.localId === localId)
  if (!removedProvider || state.providers.length <= 1) return state

  const providers = state.providers.filter((provider) => provider.localId !== localId)
  const fallbackProvider = providers[0]?.id || ''

  const resetModel = (item: ModelItem): ModelItem => {
    if (item.config.provider !== removedProvider.id) return item
    if (isDefaultModelItem(item)) {
      return { ...item, config: { ...item.config, provider: fallbackProvider } }
    }
    return {
      ...item,
      config: {
        ...item.config,
        provider: DEFAULT_CONFIG_VALUE,
        model: '',
        dimension: item.kind === 'embedding' ? null : item.config.dimension
      }
    }
  }

  return {
    ...state,
    providers,
    defaultModels: state.defaultModels.map(resetModel),
    agentModels: state.agentModels.map(resetModel),
    embeddingModels: state.embeddingModels.map(resetModel)
  }
}
