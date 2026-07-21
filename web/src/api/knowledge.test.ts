import axios from 'axios'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { knowledgeApi } from './knowledge'

vi.mock('axios', () => {
  const api = {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
    interceptors: {
      response: {
        use: vi.fn()
      }
    }
  }

  return {
    default: {
      create: vi.fn(() => api),
      post: vi.fn()
    }
  }
})

const mockedAxios = vi.mocked(axios)
const api = mockedAxios.create()

describe('knowledgeApi compatibility surface', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('keeps the Vue knowledge database endpoints and parameter shapes', () => {
    knowledgeApi.getDatabases()
    knowledgeApi.createDatabase({ database_name: 'papers' })
    knowledgeApi.deleteDatabase('db-1')
    knowledgeApi.selectDatabase('db-2')
    knowledgeApi.getDatabaseInfo('db-3')
    knowledgeApi.updateDatabase('db-4', { description: 'updated' })

    expect(api.get).toHaveBeenCalledWith('/databases')
    expect(api.post).toHaveBeenCalledWith('/databases', { database_name: 'papers' })
    expect(api.delete).toHaveBeenCalledWith('/databases/db-1')
    expect(api.get).toHaveBeenCalledWith('/databases/select', { params: { db_id: 'db-2' } })
    expect(api.get).toHaveBeenCalledWith('/databases/db-3')
    expect(api.put).toHaveBeenCalledWith('/databases/db-4', { description: 'updated' })
  })

  it('keeps document, query-test, upload, and supported-types endpoints for future UI expansion', () => {
    const file = new File(['hello'], 'paper.txt')

    knowledgeApi.addDocuments('db-1', ['/tmp/paper.txt'], { content_type: 'file' })
    knowledgeApi.getDocumentInfo('db-1', 'doc-1')
    knowledgeApi.getDocumentBasicInfo('db-1', 'doc-2')
    knowledgeApi.getDocumentContent('db-1', 'doc-3')
    knowledgeApi.deleteDocument('db-1', 'doc-4')
    knowledgeApi.queryDatabase('db-1', 'agent', { top_k: 3 })
    knowledgeApi.uploadFile(file, 'db-1', true)
    knowledgeApi.getSupportedTypes()

    expect(api.post).toHaveBeenCalledWith('/databases/db-1/documents', {
      items: ['/tmp/paper.txt'],
      params: { content_type: 'file' }
    })
    expect(api.get).toHaveBeenCalledWith('/databases/db-1/documents/doc-1')
    expect(api.get).toHaveBeenCalledWith('/databases/db-1/documents/doc-2/basic')
    expect(api.get).toHaveBeenCalledWith('/databases/db-1/documents/doc-3/content')
    expect(api.delete).toHaveBeenCalledWith('/databases/db-1/documents/doc-4')
    expect(api.post).toHaveBeenCalledWith('/databases/db-1/query-test', { query: 'agent', meta: { top_k: 3 } })
    expect(mockedAxios.post).toHaveBeenCalledWith(
      '/knowledge/files/upload',
      expect.any(FormData),
      expect.objectContaining({
        headers: { 'Content-Type': 'multipart/form-data' },
        params: { db_id: 'db-1', allow_jsonl: true }
      })
    )
    expect(api.get).toHaveBeenCalledWith('/files/supported-types')
  })
})
