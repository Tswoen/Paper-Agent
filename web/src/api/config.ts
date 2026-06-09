import axios from 'axios'

const api = axios.create({
  baseURL: '/config',
  headers: {
    'Content-Type': 'application/json'
  }
})

api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('Config API Error:', error)
    return Promise.reject(error)
  }
)

export const configApi = {
  getModelSettings() {
    return api.get('/model-settings')
  },
  saveModelSettings(data: unknown) {
    return api.put('/model-settings', data)
  },
  testModel(data: unknown) {
    return api.post('/model-settings/test', data)
  }
}
