import { type SyntheticEvent, useEffect, useMemo, useState } from 'react'
import { configApi } from '../../api/config'
import {
  applySettings,
  buildModelSettingsPayload,
  createLocalId,
  DEFAULT_CONFIG_VALUE,
  ENV_VAR_NAME_PATTERN,
  fallbackProviderTypes,
  getProviderTypeLabel,
  isDefaultModelItem,
  removeProviderFromState,
  suggestProviderId,
  syncProviderReferences,
  type ModelItem,
  type ModelSettingsState,
  type ProviderConfig,
  toProviderPayload,
  usesDefaultModel
} from './modelSettings'

type TestResult = {
  status: 'running' | 'success' | 'failed' | string
  message: string
  latency_ms?: number
}

const emptyState: ModelSettingsState = {
  providerTypes: [],
  providers: [],
  defaultModels: [],
  agentModels: [],
  embeddingModels: []
}

const providerConsoleLinks: Record<string, string> = {
  openai: 'https://platform.openai.com/api-keys',
  siliconflow: 'https://cloud.siliconflow.cn/account/ak',
  dashscope: 'https://dashscope.console.aliyun.com/apiKey',
  ark: 'https://console.volcengine.com/ark'
}

const getErrorMessage = (error: unknown, fallback: string) => {
  const candidate = error as { response?: { data?: { detail?: string | { message?: string }; message?: string } } }
  const detail = candidate.response?.data?.detail
  if (typeof detail === 'string') return detail
  if (detail?.message) return detail.message
  return candidate.response?.data?.message || fallback
}

export function ConfigurationPage() {
  const [state, setState] = useState<ModelSettingsState>(emptyState)
  const [selectedProviderLocalId, setSelectedProviderLocalId] = useState('')
  const [visibleProviderKeys, setVisibleProviderKeys] = useState<Record<string, boolean>>({})
  const [testResults, setTestResults] = useState<Record<string, TestResult>>({})
  const [isLoading, setIsLoading] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [toast, setToast] = useState<{ show: boolean; message: string; type: 'success' | 'error' }>({ show: false, message: '', type: 'success' })
  const [activeTooltip, setActiveTooltip] = useState({ text: '', left: 0, top: 0, transform: 'translate(-50%, -100%)' })

  const providerTypeOptions = state.providerTypes.length > 0 ? state.providerTypes : fallbackProviderTypes
  const selectedProvider = state.providers.find((provider) => provider.localId === selectedProviderLocalId) || state.providers[0] || null
  const providerOptions = useMemo(
    () =>
      state.providers
        .filter((provider) => provider.id)
        .map((provider) => ({
          value: provider.id,
          label: `${provider.id} (${getProviderTypeLabel(providerTypeOptions, provider.type)})`
        })),
    [providerTypeOptions, state.providers]
  )

  const showToast = (message: string, type: 'success' | 'error' = 'success') => {
    setToast({ show: true, message, type })
    window.setTimeout(() => setToast((current) => ({ ...current, show: false })), 2600)
  }

  const loadSettings = async () => {
    setIsLoading(true)
    try {
      const response = await configApi.getModelSettings()
      const next = applySettings(response.data)
      setState(next)
      setSelectedProviderLocalId(next.providers[0]?.localId || '')
      setTestResults({})
    } catch (error) {
      console.error('读取模型配置失败:', error)
      showToast(getErrorMessage(error, '读取模型配置失败'), 'error')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    void loadSettings()
  }, [])

  const updateProvider = (localId: string, changes: Partial<ProviderConfig>) => {
    setState((current) => {
      let nextState = current
      const providers = current.providers.map((provider) => {
        if (provider.localId !== localId) return provider
        const next = { ...provider, ...changes }
        if (changes.id && provider.previousId && provider.previousId !== changes.id) {
          nextState = syncProviderReferences(nextState, provider.previousId, changes.id)
        }
        if (changes.id) next.previousId = changes.id
        return next
      })
      return { ...nextState, providers }
    })
  }

  const addProvider = () => {
    const provider: ProviderConfig = {
      localId: createLocalId(),
      id: suggestProviderId(state.providers, 'custom'),
      previousId: '',
      type: 'custom',
      base_url: '',
      api_key: '',
      api_key_env: ''
    }
    setState((current) => ({ ...current, providers: [...current.providers, provider] }))
    setSelectedProviderLocalId(provider.localId)
  }

  const removeProvider = (provider: ProviderConfig) => {
    if (state.providers.length <= 1) return
    const nextState = removeProviderFromState(state, provider.localId)
    setState(nextState)
    setSelectedProviderLocalId(nextState.providers[0]?.localId || '')
  }

  const updateModelItem = (group: 'defaultModels' | 'agentModels' | 'embeddingModels', key: string, changes: Partial<ModelItem['config']>) => {
    setState((current) => ({
      ...current,
      [group]: current[group].map((item) => {
        if (item.config.key !== key) return item
        const nextConfig = { ...item.config, ...changes }
        if (!isDefaultModelItem(item) && changes.provider === DEFAULT_CONFIG_VALUE) {
          nextConfig.model = ''
          nextConfig.dimension = item.kind === 'embedding' ? null : undefined
        }
        return { ...item, config: nextConfig }
      })
    }))
    setTestResults((current) => {
      const next = { ...current }
      delete next[key]
      return next
    })
  }

  const saveSettings = async () => {
    setIsSaving(true)
    try {
      const response = await configApi.saveModelSettings(buildModelSettingsPayload(state))
      setState(applySettings(response.data.settings))
      showToast(response.data.message || '配置已保存')
    } catch (error) {
      console.error('保存模型配置失败:', error)
      showToast(getErrorMessage(error, '保存模型配置失败'), 'error')
    } finally {
      setIsSaving(false)
    }
  }

  const testModel = async (item: ModelItem) => {
    const modelPayload = usesDefaultModel(item) ? getEffectiveDefaultModel(item, state) : item
    const provider = state.providers.find((candidate) => candidate.id === modelPayload.config.provider)
    if (!provider) {
      setTestResults((current) => ({ ...current, [item.config.key]: { status: 'failed', message: '当前模型选择的 Provider 不存在' } }))
      return
    }
    if (!provider.api_key || ENV_VAR_NAME_PATTERN.test(provider.api_key)) {
      setTestResults((current) => ({ ...current, [item.config.key]: { status: 'failed', message: 'API Key 请填写真实密钥，不要填写环境变量名' } }))
      return
    }
    setTestResults((current) => ({ ...current, [item.config.key]: { status: 'running', message: '正在测试当前模型连通性...' } }))
    try {
      const response = await configApi.testModel({
        provider: toProviderPayload(provider),
        model: {
          key: modelPayload.config.key,
          provider: modelPayload.config.provider,
          model: modelPayload.config.model,
          dimension: modelPayload.kind === 'embedding' ? modelPayload.config.dimension ?? null : undefined
        },
        model_type: modelPayload.kind === 'embedding' ? 'embedding' : 'llm'
      })
      setTestResults((current) => ({ ...current, [item.config.key]: response.data }))
    } catch (error) {
      console.error('模型连通性测试失败:', error)
      setTestResults((current) => ({ ...current, [item.config.key]: { status: 'failed', message: getErrorMessage(error, '模型连通性测试失败') } }))
    }
  }

  const showTooltip = (trigger: HTMLElement, text: string) => {
    const rect = trigger.getBoundingClientRect()
    const estimatedWidth = Math.min(320, Math.max(220, window.innerWidth - 32))
    const center = rect.left + rect.width / 2
    const left = Math.min(Math.max(center, estimatedWidth / 2 + 12), window.innerWidth - estimatedWidth / 2 - 12)
    const shouldShowBelow = rect.top < 84
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

    const hintButton = target.closest('.hint-button')
    if (hintButton instanceof HTMLElement && hintButton.dataset.tooltip) {
      showTooltip(hintButton, hintButton.dataset.tooltip)
      return
    }

    if (target.matches('input, select, textarea')) {
      const field = target.closest('.field')
      const fieldHint = field?.querySelector('.hint-button')
      if (fieldHint instanceof HTMLElement && fieldHint.dataset.tooltip) {
        showTooltip(target as HTMLElement, fieldHint.dataset.tooltip)
        return
      }
    }

    if (event.type === 'mouseover') setActiveTooltip((current) => ({ ...current, text: '' }))
  }

  const hideTooltip = () => setActiveTooltip((current) => ({ ...current, text: '' }))

  return (
    <div className="config-page" onFocus={handleTooltipEnter} onBlur={hideTooltip} onMouseLeave={hideTooltip} onMouseOver={handleTooltipEnter} onScroll={hideTooltip}>
      <header className="config-hero">
        <div className="hero-copy">
          <p className="eyebrow">Runtime Configuration</p>
          <h1>系统配置</h1>
        </div>
        <div className="hero-actions">
          <button className="secondary-button" type="button" disabled={isLoading} onClick={() => void loadSettings()}>
            {isLoading ? '加载中' : '重新加载'}
          </button>
          <button className="primary-button" type="button" disabled={isSaving || isLoading} onClick={() => void saveSettings()}>
            {isSaving ? '保存中' : '保存配置'}
          </button>
        </div>
      </header>

      {isLoading ? (
        <section className="center-state">
          <div className="spinner" />
          <p>正在读取模型配置...</p>
        </section>
      ) : (
        <>
          <section className="config-section provider-section">
            <header className="section-header">
              <div>
                <p className="section-kicker">01 Provider</p>
                <h2>模型 Provider</h2>
                <span>配置 OpenAI、SiliconFlow、DashScope、Ark 或自定义 OpenAI-compatible 服务。</span>
              </div>
              <button className="primary-button" type="button" onClick={addProvider}>
                添加 Provider
              </button>
            </header>
            <div className="provider-console">
              <aside className="provider-list-panel" aria-label="模型服务商列表">
                {state.providers.map((provider) => (
                  <button
                    key={provider.localId}
                    className={`provider-list-item ${selectedProvider?.localId === provider.localId ? 'active' : ''}`}
                    type="button"
                    onClick={() => setSelectedProviderLocalId(provider.localId)}
                  >
                    <span className="provider-logo">{(provider.id || provider.type || 'P').slice(0, 2).toUpperCase()}</span>
                    <span className="provider-list-copy">
                      <strong>{provider.id || getProviderTypeLabel(providerTypeOptions, provider.type)}</strong>
                    </span>
                  </button>
                ))}
              </aside>
              {selectedProvider && (
                <section className="provider-detail-panel">
                  <header className="provider-detail-header">
                    <span className="provider-logo large">{(selectedProvider.id || selectedProvider.type || 'P').slice(0, 2).toUpperCase()}</span>
                    <div>
                      <h3>{selectedProvider.id || 'Provider'}</h3>
                      <p>在服务商控制台获取 API 密钥，并配置 OpenAI-compatible 调用地址。</p>
                    </div>
                    <button
                      className="provider-link-button"
                      type="button"
                      disabled={!providerConsoleLinks[selectedProvider.type]}
                      onClick={() => window.open(providerConsoleLinks[selectedProvider.type], '_blank', 'noopener,noreferrer')}
                    >
                      获取 API Key
                    </button>
                  </header>
                  <div className="provider-detail-fields">
                    <label className="field provider-field-wide">
                      <span>
                        API Key
                        <button className="hint-button" type="button" data-tooltip="这里只填写真实 Key；保存后后端会自动写入 .env，并在 models.yaml 中使用 Provider ID 对应的环境变量名。" aria-label="API Key 说明" />
                      </span>
                      <div className="password-field">
                        <input
                          value={selectedProvider.api_key}
                          type={visibleProviderKeys[selectedProvider.localId] ? 'text' : 'password'}
                          placeholder="sk-..."
                          autoComplete="off"
                          onChange={(event) => updateProvider(selectedProvider.localId, { api_key: event.target.value })}
                        />
                        <button type="button" onClick={() => setVisibleProviderKeys((current) => ({ ...current, [selectedProvider.localId]: !current[selectedProvider.localId] }))}>
                          {visibleProviderKeys[selectedProvider.localId] ? '隐藏' : '显示'}
                        </button>
                      </div>
                    </label>
                    <label className="field provider-field-wide">
                      <span>
                        API URL
                        <button className="hint-button" type="button" data-tooltip="服务商的 OpenAI-compatible base_url，后端会传给模型客户端。" aria-label="API URL 说明" />
                      </span>
                      <input value={selectedProvider.base_url} type="url" placeholder="https://api.example.com/v1" onChange={(event) => updateProvider(selectedProvider.localId, { base_url: event.target.value })} />
                    </label>
                    <label className="field">
                      <span>
                        Provider ID
                        <button className="hint-button" type="button" data-tooltip="Provider ID 会写入 YAML 顶层键，并用于生成环境变量名，例如 siliconflow 对应 SILICONFLOW_API_KEY。" aria-label="Provider ID 说明" />
                      </span>
                      <input value={selectedProvider.id} type="text" placeholder="siliconflow" onChange={(event) => updateProvider(selectedProvider.localId, { id: event.target.value })} />
                    </label>
                    <label className="field">
                      <span>
                        Provider 类型
                        <button className="hint-button" type="button" data-tooltip="用于减少输入错误。自定义服务可选择 OpenAI Compatible。" aria-label="Provider 类型说明" />
                      </span>
                      <select
                        value={selectedProvider.type}
                        onChange={(event) => {
                          const option = providerTypeOptions.find((item) => item.value === event.target.value)
                          updateProvider(selectedProvider.localId, {
                            type: event.target.value,
                            base_url: selectedProvider.base_url || option?.default_base_url || ''
                          })
                        }}
                      >
                        {providerTypeOptions.map((type) => (
                          <option key={type.value} value={type.value}>
                            {type.label}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                  <footer className="provider-detail-footer">
                    <span>当前配置键：{selectedProvider.id || '未命名 Provider'}</span>
                    <button className="provider-remove-button" type="button" disabled={state.providers.length <= 1} onClick={() => removeProvider(selectedProvider)}>
                      删除 Provider
                    </button>
                  </footer>
                </section>
              )}
            </div>
          </section>

          <ModelSection title="默认模型配置" kicker="02 Defaults" items={state.defaultModels} defaultModels={state.defaultModels} providerOptions={providerOptions} testResults={testResults} onChange={updateModelItem} onTest={testModel} group="defaultModels" />
          <ModelSection title="高级智能体模型配置" kicker="03 Agent Models" items={state.agentModels} defaultModels={state.defaultModels} providerOptions={providerOptions} testResults={testResults} onChange={updateModelItem} onTest={testModel} group="agentModels" />
          <ModelSection title="嵌入模型配置" kicker="04 Embeddings" items={state.embeddingModels} defaultModels={state.defaultModels} providerOptions={providerOptions} testResults={testResults} onChange={updateModelItem} onTest={testModel} group="embeddingModels" />
        </>
      )}

      {toast.show && <div className={`toast ${toast.type}`}>{toast.message}</div>}
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
  )
}

const getEffectiveDefaultModel = (item: ModelItem, state: ModelSettingsState): ModelItem => {
  if (!usesDefaultModel(item)) return item
  const defaultKey = item.kind === 'embedding' ? 'default-embedding-model' : 'default-model'
  return state.defaultModels.find((candidate) => candidate.config.key === defaultKey) || item
}

function ModelSection({
  title,
  kicker,
  items,
  defaultModels,
  providerOptions,
  testResults,
  onChange,
  onTest,
  group
}: {
  title: string
  kicker: string
  items: ModelItem[]
  defaultModels: ModelItem[]
  providerOptions: Array<{ value: string; label: string }>
  testResults: Record<string, TestResult>
  onChange: (group: 'defaultModels' | 'agentModels' | 'embeddingModels', key: string, changes: Partial<ModelItem['config']>) => void
  onTest: (item: ModelItem) => void
  group: 'defaultModels' | 'agentModels' | 'embeddingModels'
}) {
  return (
    <section className="config-section">
      <header className="section-header">
        <div>
          <p className="section-kicker">{kicker}</p>
          <h2>{title}</h2>
        </div>
      </header>
      <div className="model-grid">
        {items.map((item) => {
          const usesDefault = usesDefaultModel(item)
          const options = isDefaultModelItem(item) ? providerOptions : [{ value: DEFAULT_CONFIG_VALUE, label: '默认配置' }, ...providerOptions]
          const defaultModel = defaultModels.find((candidate) => candidate.config.key === (item.kind === 'embedding' ? 'default-embedding-model' : 'default-model'))
          const modelPlaceholder = usesDefault && defaultModel?.config.model ? `使用默认配置：${defaultModel.config.model}` : usesDefault ? '使用默认配置' : item.kind === 'embedding' ? 'embedding-model' : 'model-name'
          const dimensionPlaceholder =
            usesDefault && defaultModel?.config.dimension ? `使用默认维度：${defaultModel.config.dimension}` : usesDefault ? '使用默认维度' : '1024'
          return (
            <article key={item.config.key} className={`model-card ${item.kind}`}>
              <div className="model-card-header">
                <div>
                  <span className="key-chip">{item.config.key}</span>
                  <h3>{item.label || item.config.key}</h3>
                  {item.description && <p>{item.description}</p>}
                </div>
                <button className="test-button" type="button" disabled={testResults[item.config.key]?.status === 'running'} onClick={() => onTest(item)}>
                  {testResults[item.config.key]?.status === 'running' ? '测试中' : '测试连通性'}
                </button>
              </div>
              <div className="model-fields">
                <label className="field">
                  <span>
                    Provider
                    <button className="hint-button" type="button" data-tooltip="选择该模型调用时使用的服务商，对应 YAML 中的 model-provider。" aria-label="模型 Provider 说明" />
                  </span>
                  <select value={item.config.provider} onChange={(event) => onChange(group, item.config.key, { provider: event.target.value })}>
                    {options.map((provider) => (
                      <option key={provider.value || 'default'} value={provider.value}>
                        {provider.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="field">
                  <span>
                    模型名称
                    <button className="hint-button" type="button" data-tooltip="服务商侧的模型 ID，例如 Qwen/Qwen3-32B、text-embedding-v4 或 doubao endpoint。" aria-label="模型名称说明" />
                  </span>
                  <input
                    value={item.config.model}
                    type="text"
                    placeholder={modelPlaceholder}
                    disabled={usesDefault}
                    onChange={(event) => onChange(group, item.config.key, { model: event.target.value })}
                  />
                </label>
                {item.kind === 'embedding' && (
                  <label className="field">
                    <span>
                      向量维度
                      <button className="hint-button" type="button" data-tooltip="需要与实际模型返回维度和向量库配置匹配。" aria-label="嵌入维度说明" />
                    </span>
                    <input
                      value={item.config.dimension ?? ''}
                      type="number"
                      min="1"
                      placeholder={dimensionPlaceholder}
                      disabled={usesDefault}
                      onChange={(event) => onChange(group, item.config.key, { dimension: event.target.value ? Number(event.target.value) : null })}
                    />
                  </label>
                )}
              </div>
              {testResults[item.config.key] && (
                <p className={`test-result ${testResults[item.config.key].status}`}>
                  {testResults[item.config.key].message}
                  {testResults[item.config.key].latency_ms ? ` ${testResults[item.config.key].latency_ms} ms` : ''}
                </p>
              )}
            </article>
          )
        })}
      </div>
    </section>
  )
}
