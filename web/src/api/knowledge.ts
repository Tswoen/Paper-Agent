import axios from 'axios'

const API_BASE_URL = '/knowledge'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json'
  }
})

api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error)
    return Promise.reject(error)
  }
)

export type KnowledgeDatabase = {
  id?: string
  db_id?: string
  name?: string
  description?: string
  kb_type?: string
  embedding_model?: string
  document_count?: number
  [key: string]: unknown
}

export const normalizeDatabase = <T extends KnowledgeDatabase>(database: T | null | undefined): (T & { id: string }) | null => {
  if (!database) return null
  const id = database.db_id || database.id
  return id ? ({ ...database, id } as T & { id: string }) : null
}

export const knowledgeApi = {
  getDatabases() {
    return api.get('/databases')
  },

  createDatabase(data: unknown) {
    return api.post('/databases', data)
  },

  deleteDatabase(dbId: string) {
    return api.delete(`/databases/${dbId}`)
  },

  selectDatabase(dbId: string) {
    return api.get('/databases/select', { params: { db_id: dbId } })
  },

  getDatabaseInfo(dbId: string) {
    return api.get(`/databases/${dbId}`)
  },

  updateDatabase(dbId: string, data: unknown) {
    return api.put(`/databases/${dbId}`, data)
  },

  addDocuments(dbId: string, items: unknown[], params: Record<string, unknown> = {}) {
    return api.post(`/databases/${dbId}/documents`, { items, params })
  },

  getDocumentInfo(dbId: string, docId: string) {
    return api.get(`/databases/${dbId}/documents/${docId}`)
  },

  getDocumentBasicInfo(dbId: string, docId: string) {
    return api.get(`/databases/${dbId}/documents/${docId}/basic`)
  },

  getDocumentContent(dbId: string, docId: string) {
    return api.get(`/databases/${dbId}/documents/${docId}/content`)
  },

  deleteDocument(dbId: string, docId: string) {
    return api.delete(`/databases/${dbId}/documents/${docId}`)
  },

  queryDatabase(dbId: string, query: string, meta: Record<string, unknown> = {}) {
    return api.post(`/databases/${dbId}/query-test`, { query, meta })
  },

  uploadFile(file: File, dbId: string | null = null, allowJsonl = false) {
    const formData = new FormData()
    formData.append('file', file)

    const params: Record<string, unknown> = {}
    if (dbId) params.db_id = dbId
    if (allowJsonl) params.allow_jsonl = true

    return axios.post(`${API_BASE_URL}/files/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      params
    })
  },

  getSupportedTypes() {
    return api.get('/files/supported-types')
  }
}

export default knowledgeApi
