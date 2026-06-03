import axios from 'axios'

const API_BASE_URL = '/config'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json'
  }
})

api.interceptors.response.use(
  response => response,
  error => {
    console.error('Config API Error:', error)
    return Promise.reject(error)
  }
)

export const configApi = {
  getModelSettings() {
    return api.get('/model-settings')
  },

  saveModelSettings(data) {
    return api.put('/model-settings', data)
  },

  testModel(data) {
    return api.post('/model-settings/test', data)
  }
}

export default configApi
