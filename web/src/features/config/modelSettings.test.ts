import { describe, expect, it } from 'vitest'
import { applySettings, buildModelSettingsPayload, removeProviderFromState, suggestProviderId, syncProviderReferences } from './modelSettings'

describe('model settings state', () => {
  it('normalizes provider ids, default model items, and embedding dimensions', () => {
    const state = applySettings({
      providers: [{ id: 'siliconflow', type: 'siliconflow', base_url: 'https://api.siliconflow.cn/v1', api_key: 'SILICONFLOW_API_KEY' }],
      default_models: [{ key: 'default-model', label: '默认 LLM', kind: 'llm', config: { provider: 'siliconflow', model: 'Qwen3' } }],
      agent_models: [{ key: 'search-agent', label: '检索', kind: 'llm', config: { provider: '', model: '' } }],
      embedding_models: [{ key: 'knowledge-embedding', label: '知识库', kind: 'embedding', config: { provider: '', model: '', dimension: '' } }]
    })

    expect(state.providers[0]).toMatchObject({
      id: 'siliconflow',
      api_key: '',
      api_key_env: 'SILICONFLOW_API_KEY'
    })
    expect(state.defaultModels[0].config.key).toBe('default-model')
    expect(state.embeddingModels[0].config.dimension).toBeNull()
  })

  it('builds the save payload while preserving default-model fallbacks', () => {
    const state = applySettings({
      providers: [{ id: 'openai', type: 'openai', base_url: 'https://api.openai.com/v1', api_key: 'sk-test' }],
      default_models: [{ key: 'default-model', kind: 'llm', config: { provider: 'openai', model: 'gpt-4.1' } }],
      agent_models: [{ key: 'search-agent', kind: 'llm', config: { provider: '', model: '' } }],
      embedding_models: [{ key: 'knowledge-embedding', kind: 'embedding', config: { provider: '', model: '', dimension: null } }]
    })

    expect(buildModelSettingsPayload(state)).toEqual({
      providers: [{ id: 'openai', type: 'openai', base_url: 'https://api.openai.com/v1', api_key: 'sk-test' }],
      default_models: [{ key: 'default-model', provider: 'openai', model: 'gpt-4.1' }],
      agent_models: [{ key: 'search-agent', provider: '', model: '' }],
      embedding_models: [{ key: 'knowledge-embedding', provider: '', model: '', dimension: null }]
    })
  })

  it('suggests unique provider ids for newly added providers', () => {
    const state = applySettings({
      providers: [
        { id: 'custom', type: 'custom' },
        { id: 'custom-2', type: 'custom' },
        { id: 'openai', type: 'openai' }
      ]
    })

    expect(suggestProviderId(state.providers, 'custom')).toBe('custom-3')
    expect(suggestProviderId(state.providers, 'siliconflow')).toBe('siliconflow')
  })

  it('syncs renamed provider ids across default, agent, and embedding model references', () => {
    const state = applySettings({
      providers: [{ id: 'old-provider', type: 'custom' }],
      default_models: [{ key: 'default-model', kind: 'llm', config: { provider: 'old-provider', model: 'a' } }],
      agent_models: [{ key: 'search-agent', kind: 'llm', config: { provider: 'old-provider', model: 'b' } }],
      embedding_models: [{ key: 'knowledge-embedding', kind: 'embedding', config: { provider: 'old-provider', model: 'c', dimension: 1024 } }]
    })

    const next = syncProviderReferences(state, 'old-provider', 'new-provider')

    expect(next.defaultModels[0].config.provider).toBe('new-provider')
    expect(next.agentModels[0].config.provider).toBe('new-provider')
    expect(next.embeddingModels[0].config.provider).toBe('new-provider')
  })

  it('removes a provider and falls back model references like the Vue implementation', () => {
    const state = applySettings({
      providers: [
        { id: 'remove-me', type: 'custom' },
        { id: 'fallback', type: 'openai' }
      ],
      default_models: [{ key: 'default-model', kind: 'llm', config: { provider: 'remove-me', model: 'default' } }],
      agent_models: [{ key: 'search-agent', kind: 'llm', config: { provider: 'remove-me', model: 'agent' } }],
      embedding_models: [{ key: 'knowledge-embedding', kind: 'embedding', config: { provider: 'remove-me', model: 'embed', dimension: 1024 } }]
    })

    const next = removeProviderFromState(state, state.providers[0].localId)

    expect(next.providers.map((provider) => provider.id)).toEqual(['fallback'])
    expect(next.defaultModels[0].config.provider).toBe('fallback')
    expect(next.agentModels[0].config).toMatchObject({ provider: '', model: '' })
    expect(next.embeddingModels[0].config).toMatchObject({ provider: '', model: '', dimension: null })
  })
})
