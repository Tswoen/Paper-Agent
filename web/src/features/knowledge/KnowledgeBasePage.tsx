import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { knowledgeApi, normalizeDatabase, type KnowledgeDatabase } from '../../api/knowledge'
import { CreateDatabaseModal } from './components/CreateDatabaseModal'
import { DatabaseCard } from './components/DatabaseCard'
import { FileUpload } from './components/FileUpload'
import { QueryTest, type QueryTestHandle } from './components/QueryTest'

type ToastState = {
  show: boolean
  message: string
  type: 'success' | 'error'
}

export function KnowledgeBasePage() {
  const navigate = useNavigate()
  const [databases, setDatabases] = useState<Array<KnowledgeDatabase & { id: string }>>([])
  const [selectedDatabaseId, setSelectedDatabaseId] = useState('')
  const [selectedDatabase, setSelectedDatabase] = useState<(KnowledgeDatabase & { id: string }) | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [toast, setToast] = useState<ToastState>({ show: false, message: '', type: 'success' })
  const queryTestRef = useRef<QueryTestHandle | null>(null)

  const showToast = (message: string, type: ToastState['type'] = 'success') => {
    setToast({ show: true, message, type })
    window.setTimeout(() => setToast((current) => ({ ...current, show: false })), 3000)
  }

  const loadDatabases = async () => {
    setIsLoading(true)
    try {
      const response = await knowledgeApi.getDatabases()
      const rawDatabases = response.data.databases || []
      setDatabases(rawDatabases.map((db: KnowledgeDatabase) => normalizeDatabase(db)).filter(Boolean))
    } catch (error) {
      console.error('加载知识库列表失败:', error)
      showToast('加载知识库列表失败', 'error')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    void loadDatabases()
  }, [])

  const handleSelectDatabase = (database: KnowledgeDatabase & { id: string }) => {
    if (selectedDatabaseId === database.id) {
      setSelectedDatabaseId('')
      setSelectedDatabase(null)
      showToast('已取消选择知识库')
      return
    }

    setSelectedDatabaseId(database.id)
    setSelectedDatabase(database)
    queryTestRef.current?.clearResults()
    showToast(`已选择知识库: ${database.name}`)
  }

  const handleDeleteDatabase = async (database: KnowledgeDatabase & { id: string }) => {
    if (!window.confirm(`确定要删除知识库"${database.name}"吗？此操作不可恢复。`)) return

    try {
      await knowledgeApi.deleteDatabase(database.db_id || database.id)
      if (selectedDatabaseId === database.id) {
        setSelectedDatabaseId('')
        setSelectedDatabase(null)
      }
      await loadDatabases()
      showToast('知识库删除成功')
    } catch (error) {
      console.error('删除知识库失败:', error)
      showToast('删除知识库失败', 'error')
    }
  }

  const handleCreateDatabase = async (data: unknown) => {
    try {
      const response = await knowledgeApi.createDatabase(data)
      await loadDatabases()
      showToast('知识库创建成功')
      return normalizeDatabase(response.data) || response.data
    } catch (error) {
      console.error('创建知识库失败:', error)
      showToast('创建知识库失败', 'error')
      throw error
    }
  }

  return (
    <div className="knowledge-base-container">
      <header className="knowledge-header">
        <h1>RAG知识库管理</h1>
        <div className="header-actions">
          <button className="primary-button" type="button" onClick={() => setShowCreateModal(true)}>
            + 创建知识库
          </button>
          <button className="secondary-button" type="button" onClick={() => navigate('/')}>
            返回主页
          </button>
        </div>
      </header>

      <div className="knowledge-layout">
        <div className="knowledge-main">
          <div className="section-header">
            <h2>知识库列表</h2>
            {selectedDatabase && (
              <div className="selected-info">
                <span>当前选中:</span>
                <strong>{selectedDatabase.name}</strong>
              </div>
            )}
          </div>
          {isLoading ? (
            <div className="center-state">
              <div className="spinner" />
              <p>加载知识库列表...</p>
            </div>
          ) : databases.length === 0 ? (
            <div className="empty-state">
              <div className="empty-mark">KB</div>
              <p>暂无知识库</p>
              <button className="primary-button" type="button" onClick={() => setShowCreateModal(true)}>
                创建第一个知识库
              </button>
            </div>
          ) : (
            <div className="databases-grid">
              {databases.map((database) => (
                <DatabaseCard
                  key={database.id}
                  database={database}
                  isSelected={selectedDatabaseId === database.id}
                  onSelect={handleSelectDatabase}
                  onDelete={handleDeleteDatabase}
                />
              ))}
            </div>
          )}
        </div>

        {selectedDatabase && (
          <aside className="knowledge-side-panel">
            <section className="panel-section">
              <h3>文件上传</h3>
              <FileUpload selectedDatabaseId={selectedDatabaseId} onUploadComplete={(file) => showToast(`文件 "${file.name}" 上传成功`)} onUploadError={(file) => showToast(`文件 "${file.name}" 上传失败: ${file.error}`, 'error')} />
            </section>
            <section className="panel-section">
              <h3>查询测试</h3>
              <QueryTest ref={queryTestRef} selectedDatabaseId={selectedDatabaseId} />
            </section>
          </aside>
        )}
      </div>

      <CreateDatabaseModal visible={showCreateModal} onVisibleChange={setShowCreateModal} onSubmit={handleCreateDatabase} />
      {toast.show && <div className={`toast ${toast.type}`}>{toast.message}</div>}
    </div>
  )
}
