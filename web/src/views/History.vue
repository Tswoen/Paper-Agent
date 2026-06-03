<template>
  <div class="history-container">
    <header class="history-header">
      <div>
        <p class="eyebrow">Conversation Archive</p>
        <h1>历史报告</h1>
      </div>
      <button class="refresh-button" type="button" :disabled="isLoading" @click="loadHistory">
        {{ isLoading ? '刷新中' : '刷新' }}
      </button>
    </header>

    <section v-if="isLoading" class="center-state">
      <div class="spinner"></div>
      <p>正在加载历史报告...</p>
    </section>

    <section v-else-if="historyList.length === 0" class="empty-state">
      <div class="empty-mark">AI</div>
      <h2>还没有历史报告</h2>
      <p>完成一次对话式调研后，报告会自动保存在这里。</p>
      <button class="primary-button" type="button" @click="goToCreate">开始对话</button>
    </section>

    <section v-else class="history-list">
      <article v-for="item in historyList" :key="item.id" class="history-card">
        <div class="card-main">
          <div class="report-title">
            <span class="report-mark">{{ getTitleInitial(item.title) }}</span>
            <div>
              <h2>{{ item.title || '未命名报告' }}</h2>
              <span class="report-time">{{ formatDate(item.createdAt) }}</span>
            </div>
          </div>

          <p class="report-query">{{ item.query }}</p>

          <div class="report-preview" v-if="item.content">
            {{ item.content }}
          </div>
        </div>

        <div class="card-side">
          <span class="report-status" :class="item.status">
            {{ getStatusText(item.status) }}
          </span>
          <button class="text-button" type="button" @click="viewReport(item)">
            回到对话
          </button>
          <button class="danger-button" type="button" @click="deleteReport(item)">
            删除
          </button>
        </div>
      </article>
    </section>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const isLoading = ref(false)
const historyList = ref([])

const loadHistory = async () => {
  isLoading.value = true
  try {
    await new Promise(resolve => setTimeout(resolve, 250))
    const saved = localStorage.getItem('reportHistory')
    historyList.value = saved ? JSON.parse(saved) : []
  } catch (error) {
    console.error('加载历史报告失败:', error)
    historyList.value = []
  } finally {
    isLoading.value = false
  }
}

const getStatusText = (status) => {
  const statusMap = {
    completed: '已完成',
    processing: '处理中',
    failed: '失败',
    pending: '待处理'
  }
  return statusMap[status] || '未知'
}

const formatDate = (dateString) => {
  if (!dateString) return '未知时间'
  const date = new Date(dateString)
  return date.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  })
}

const getTitleInitial = (title = '') => {
  return title.trim().slice(0, 1).toUpperCase() || 'R'
}

const viewReport = (item) => {
  router.push({
    path: '/',
    query: { reportId: item.id }
  })
}

const deleteReport = (item) => {
  if (!window.confirm(`确定要删除报告"${item.title || '未命名报告'}"吗？此操作不可恢复。`)) {
    return
  }

  try {
    historyList.value = historyList.value.filter(history => history.id !== item.id)
    localStorage.setItem('reportHistory', JSON.stringify(historyList.value))
  } catch (error) {
    console.error('删除报告失败:', error)
    window.alert('删除失败，请重试。')
  }
}

const goToCreate = () => {
  router.push('/')
}

onMounted(() => {
  loadHistory()
})
</script>

<style scoped>
.history-container {
  height: 100%;
  min-height: 0;
  overflow-y: auto;
  padding: 6px;
  color: #17202a;
}

.history-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  margin-bottom: 16px;
  padding: 18px;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  background: #ffffff;
}

.eyebrow {
  margin: 0 0 5px;
  color: #0f766e;
  font-size: 0.76rem;
  font-weight: 900;
  text-transform: uppercase;
}

.history-header h1 {
  margin: 0;
  color: #111827;
  font-size: 1.45rem;
}

.refresh-button,
.primary-button,
.text-button,
.danger-button {
  min-height: 38px;
  border: 0;
  border-radius: 8px;
  font: inherit;
  font-weight: 800;
  cursor: pointer;
}

.refresh-button,
.primary-button {
  padding: 0 16px;
  background: #14b8a6;
  color: #042f2e;
}

.refresh-button:disabled {
  cursor: not-allowed;
  opacity: 0.6;
}

.center-state,
.empty-state {
  display: grid;
  justify-items: center;
  align-content: center;
  gap: 12px;
  min-height: calc(100% - 104px);
  padding: 32px;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  background: #ffffff;
  text-align: center;
}

.spinner {
  width: 42px;
  height: 42px;
  border: 4px solid #e5e7eb;
  border-top-color: #14b8a6;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.center-state p,
.empty-state p {
  margin: 0;
  color: #64748b;
}

.empty-mark {
  display: grid;
  place-items: center;
  width: 58px;
  height: 58px;
  border-radius: 8px;
  background: #17202a;
  color: #2dd4bf;
  font-weight: 900;
}

.empty-state h2 {
  margin: 4px 0 0;
  color: #111827;
  font-size: 1.35rem;
}

.history-list {
  display: grid;
  gap: 12px;
}

.history-card {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 140px;
  gap: 16px;
  padding: 16px;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  background: #ffffff;
  box-shadow: 0 12px 34px rgba(17, 24, 39, 0.06);
}

.card-main {
  min-width: 0;
}

.report-title {
  display: flex;
  gap: 12px;
  align-items: center;
}

.report-mark {
  display: grid;
  place-items: center;
  width: 42px;
  height: 42px;
  flex: 0 0 auto;
  border-radius: 8px;
  background: #ecfdf5;
  color: #0f766e;
  font-weight: 900;
}

.report-title h2 {
  margin: 0 0 4px;
  overflow: hidden;
  color: #111827;
  font-size: 1rem;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.report-time {
  color: #64748b;
  font-size: 0.8rem;
}

.report-query {
  margin: 14px 0 0;
  color: #334155;
  line-height: 1.6;
}

.report-preview {
  display: -webkit-box;
  margin-top: 10px;
  overflow: hidden;
  color: #64748b;
  font-size: 0.9rem;
  line-height: 1.55;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
}

.card-side {
  display: grid;
  align-content: start;
  gap: 8px;
}

.report-status {
  display: inline-grid;
  place-items: center;
  min-height: 30px;
  padding: 0 10px;
  border-radius: 8px;
  background: #eff6ff;
  color: #1d4ed8;
  font-size: 0.78rem;
  font-weight: 900;
}

.report-status.failed {
  background: #fef2f2;
  color: #b91c1c;
}

.report-status.processing,
.report-status.pending {
  background: #fff7ed;
  color: #c2410c;
}

.text-button {
  background: #eef7f4;
  color: #0f766e;
}

.danger-button {
  background: #fef2f2;
  color: #b91c1c;
}

@media (max-width: 720px) {
  .history-header,
  .history-card {
    grid-template-columns: 1fr;
  }

  .history-header {
    align-items: flex-start;
    flex-direction: column;
  }

  .refresh-button {
    width: 100%;
  }

  .card-side {
    grid-template-columns: 1fr 1fr;
  }

  .report-status {
    grid-column: 1 / -1;
  }
}
</style>
