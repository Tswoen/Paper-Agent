import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'

type HistoryItem = {
  id: string
  title?: string
  query?: string
  status?: string
  createdAt?: string
  content?: string
}

const getStatusText = (status?: string) => {
  const statusMap: Record<string, string> = {
    completed: '已完成',
    processing: '处理中',
    failed: '失败',
    pending: '待处理'
  }
  return statusMap[status || ''] || '未知'
}

const formatDate = (dateString?: string) => {
  if (!dateString) return '未知时间'
  return new Date(dateString).toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  })
}

export function HistoryPage() {
  const [isLoading, setIsLoading] = useState(false)
  const [historyList, setHistoryList] = useState<HistoryItem[]>([])
  const navigate = useNavigate()

  const loadHistory = async () => {
    setIsLoading(true)
    try {
      await new Promise((resolve) => window.setTimeout(resolve, 250))
      const saved = localStorage.getItem('reportHistory')
      setHistoryList(saved ? JSON.parse(saved) : [])
    } catch (error) {
      console.error('加载历史报告失败:', error)
      setHistoryList([])
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    void loadHistory()
  }, [])

  const deleteReport = (item: HistoryItem) => {
    if (!window.confirm(`确定要删除报告"${item.title || '未命名报告'}"吗？此操作不可恢复。`)) return
    try {
      const nextHistory = historyList.filter((history) => history.id !== item.id)
      setHistoryList(nextHistory)
      localStorage.setItem('reportHistory', JSON.stringify(nextHistory))
    } catch (error) {
      console.error('删除报告失败:', error)
      window.alert('删除失败，请重试。')
    }
  }

  return (
    <div className="history-container">
      <header className="history-header">
        <div>
          <p className="eyebrow">Conversation Archive</p>
          <h1>历史报告</h1>
        </div>
        <button className="refresh-button" type="button" disabled={isLoading} onClick={() => void loadHistory()}>
          {isLoading ? '刷新中' : '刷新'}
        </button>
      </header>

      {isLoading ? (
        <section className="center-state">
          <div className="spinner" />
          <p>正在加载历史报告...</p>
        </section>
      ) : historyList.length === 0 ? (
        <section className="empty-state">
          <div className="empty-mark">AI</div>
          <h2>还没有历史报告</h2>
          <p>完成一次对话式调研后，报告会自动保存在这里。</p>
          <button className="primary-button" type="button" onClick={() => navigate('/')}>
            开始对话
          </button>
        </section>
      ) : (
        <section className="history-list">
          {historyList.map((item) => (
            <article key={item.id} className="history-card">
              <div className="card-main">
                <div className="report-title">
                  <span className="report-mark">{(item.title || 'R').trim().slice(0, 1).toUpperCase()}</span>
                  <div>
                    <h2>{item.title || '未命名报告'}</h2>
                    <span className="report-time">{formatDate(item.createdAt)}</span>
                  </div>
                </div>
                <p className="report-query">{item.query}</p>
                {item.content && <div className="report-preview">{item.content}</div>}
              </div>
              <div className="card-side">
                <span className={`report-status ${item.status}`}>{getStatusText(item.status)}</span>
                <button className="text-button" type="button" onClick={() => navigate(`/?reportId=${encodeURIComponent(item.id)}`)}>
                  回到对话
                </button>
                <button className="danger-button" type="button" onClick={() => deleteReport(item)}>
                  删除
                </button>
              </div>
            </article>
          ))}
        </section>
      )}
    </div>
  )
}
