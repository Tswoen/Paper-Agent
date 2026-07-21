import { useEffect, useState } from 'react'
import { configApi } from '../../api/config'
import type { TaskSettings } from './researchSettings'

type ModelConfig = {
  baseUrl: string
  apiKey: string
  modelName: string
}

type SettingsModalProps = {
  settings: TaskSettings
  onSave: (settings: TaskSettings) => void
  onClose: () => void
}

const PROVIDER_ID = 'datacenter'

export function SettingsModal({ settings, onSave, onClose }: SettingsModalProps) {
  const [modelConfig, setModelConfig] = useState<ModelConfig>({
    baseUrl: '',
    apiKey: '',
    modelName: '',
  })
  const [localSettings, setLocalSettings] = useState<TaskSettings>({ ...settings })

  useEffect(() => {
    configApi.getModelSettings().then((res) => {
      const data = res.data
      const providers = data.providers || []
      const existing = providers.find((p: { id: string }) => p.id === PROVIDER_ID)
      if (existing) {
        setModelConfig({
          baseUrl: existing.base_url || '',
          apiKey: existing.api_key || '',
          modelName: '',
        })
      }
      const defaults = data.default_models || []
      const defaultModel = defaults.find((m: { config: { key: string } }) => m.config?.key === 'default-model')
      if (defaultModel?.config?.model) {
        setModelConfig((prev) => ({ ...prev, modelName: defaultModel.config.model }))
      }
    }).catch(() => {
      // ignore - settings will be empty
    })
  }, [])

  const patchLocal = (changes: Partial<TaskSettings>) => {
    setLocalSettings((prev) => ({ ...prev, ...changes }))
  }

  const handleSave = async () => {
    try {
      const existing = (await configApi.getModelSettings()).data
      const otherProviders = (existing.providers || []).filter(
        (p: { id: string }) => p.id !== PROVIDER_ID
      )

      const payload = {
        providers: [
          ...otherProviders,
          {
            id: PROVIDER_ID,
            type: 'custom',
            base_url: modelConfig.baseUrl,
            api_key: modelConfig.apiKey,
          },
        ],
        default_models: [
          {
            key: 'default-model',
            provider: PROVIDER_ID,
            model: modelConfig.modelName,
          },
          {
            key: 'default-embedding-model',
            provider: PROVIDER_ID,
            model: modelConfig.modelName,
          },
        ],
        agent_models: [
          { key: 'search-model', provider: '', model: '' },
          { key: 'reading-model', provider: '', model: '' },
          { key: 'subanalyse-cluster-model', provider: '', model: '' },
          { key: 'subanalyse-deep-analyse-model', provider: '', model: '' },
          { key: 'subanalyse-global-analyse-model', provider: '', model: '' },
          { key: 'subwriting-writing-director-model', provider: '', model: '' },
          { key: 'subwriting-writing-model', provider: '', model: '' },
          { key: 'report-model', provider: '', model: '' },
        ],
        embedding_models: [
          { key: 'chroma-embedding-model', provider: '', model: '', dimension: null },
          { key: 'cluster-embedding-model', provider: '', model: '', dimension: null },
        ],
      }

      await configApi.saveModelSettings(payload)
      onSave(localSettings)
    } catch (err: unknown) {
      let msg = '保存设置失败，请检查后端服务。'
      if (err && typeof err === 'object' && 'response' in err) {
        const resp = (err as { response: { data?: { detail?: string } } }).response
        if (resp.data?.detail) {
          msg += `\n\n后端返回: ${resp.data.detail}`
        }
      }
      console.error('保存设置失败:', err)
      window.alert(msg)
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-container" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>设置</h2>
          <button className="modal-close-btn" onClick={onClose}>
            ✕
          </button>
        </div>

        <div className="modal-body">
          <div className="modal-section">
            <h3>模型配置</h3>
            <div className="modal-field">
              <label>Base URL</label>
              <input
                type="text"
                placeholder="https://api.example.com/v1"
                value={modelConfig.baseUrl}
                onChange={(e) => setModelConfig((prev) => ({ ...prev, baseUrl: e.target.value }))}
              />
            </div>
            <div className="modal-field">
              <label>API Key</label>
              <input
                type="password"
                placeholder="sk-..."
                value={modelConfig.apiKey}
                onChange={(e) => setModelConfig((prev) => ({ ...prev, apiKey: e.target.value }))}
              />
            </div>
            <div className="modal-field">
              <label>Model Name</label>
              <input
                type="text"
                placeholder="deepseek-v4-flash"
                value={modelConfig.modelName}
                onChange={(e) => setModelConfig((prev) => ({ ...prev, modelName: e.target.value }))}
              />
            </div>
          </div>

          <div className="modal-section">
            <h3>任务约束</h3>
            <div className="modal-field">
              <label>起始年份</label>
              <input
                type="number"
                min="1900"
                placeholder="2021"
                value={localSettings.yearStart}
                onChange={(e) => patchLocal({ yearStart: e.target.value })}
              />
            </div>
            <div className="modal-field">
              <label>结束年份</label>
              <input
                type="number"
                min="1900"
                placeholder="2026"
                value={localSettings.yearEnd}
                onChange={(e) => patchLocal({ yearEnd: e.target.value })}
              />
            </div>
            <div className="modal-field">
              <label>关键词（分号分隔）</label>
              <textarea
                rows={2}
                placeholder="large language model; agent; reasoning"
                value={localSettings.keywords}
                onChange={(e) => patchLocal({ keywords: e.target.value })}
              />
            </div>
            <div className="modal-field">
              <label>论文数量上限</label>
              <input
                type="number"
                min="1"
                max="100"
                value={localSettings.paperLimit}
                onChange={(e) => patchLocal({ paperLimit: Number(e.target.value) })}
              />
            </div>
            <div className="modal-field">
              <label>输出语言</label>
              <select
                value={localSettings.outputLanguage}
                onChange={(e) => patchLocal({ outputLanguage: e.target.value as TaskSettings['outputLanguage'] })}
              >
                <option>中文</option>
                <option>英文</option>
              </select>
            </div>
            <div className="modal-field">
              <label>自定义约束提示词</label>
              <textarea
                rows={3}
                placeholder="例如：强调方法对比；更关注局限性总结"
                value={localSettings.customPrompt}
                onChange={(e) => patchLocal({ customPrompt: e.target.value })}
              />
            </div>
          </div>
        </div>

        <div className="modal-footer">
          <button className="modal-cancel-btn" onClick={onClose}>
            取消
          </button>
          <button className="modal-save-btn" onClick={() => void handleSave()}>
            保存
          </button>
        </div>
      </div>
    </div>
  )
}
