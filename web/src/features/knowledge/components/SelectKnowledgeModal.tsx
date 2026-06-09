import { useEffect, useState } from 'react'
import { knowledgeApi, normalizeDatabase, type KnowledgeDatabase } from '../../../api/knowledge'

type SelectKnowledgeModalProps = {
  visible: boolean
  currentDatabaseId?: string
  onVisibleChange: (visible: boolean) => void
  onSelect: (database: (KnowledgeDatabase & { id: string }) | null) => void
  onCreateDatabase: () => void
}

export function SelectKnowledgeModal({ visible, currentDatabaseId = '', onVisibleChange, onSelect, onCreateDatabase }: SelectKnowledgeModalProps) {
  const [isLoading, setIsLoading] = useState(false)
  const [databases, setDatabases] = useState<Array<KnowledgeDatabase & { id: string }>>([])
  const [selectedDatabaseId, setSelectedDatabaseId] = useState(currentDatabaseId)

  useEffect(() => {
    setSelectedDatabaseId(currentDatabaseId)
  }, [currentDatabaseId])

  useEffect(() => {
    if (!visible) return
    const loadDatabases = async () => {
      setIsLoading(true)
      try {
        const response = await knowledgeApi.getDatabases()
        const rawDatabases = response.data.databases || []
        setDatabases(rawDatabases.map((db: KnowledgeDatabase) => normalizeDatabase(db)).filter(Boolean))
      } catch (error) {
        console.error('加载知识库列表失败:', error)
        setDatabases([])
      } finally {
        setIsLoading(false)
      }
    }
    void loadDatabases()
  }, [visible])

  if (!visible) return null

  const confirm = () => {
    onSelect(selectedDatabaseId ? databases.find((db) => db.id === selectedDatabaseId) || null : null)
    onVisibleChange(false)
  }

  return (
    <div className="modal-overlay" onClick={() => onVisibleChange(false)}>
      <div className="modal-container" onClick={(event) => event.stopPropagation()}>
        <header className="modal-header">
          <h2>选择知识库</h2>
          <button className="modal-close-button" type="button" onClick={() => onVisibleChange(false)}>
            ×
          </button>
        </header>
        <div className="modal-body">
          {isLoading ? (
            <div className="center-state">
              <div className="spinner" />
              <p>加载知识库列表...</p>
            </div>
          ) : databases.length === 0 ? (
            <div className="empty-state">
              <p>暂无可用知识库</p>
              <button className="primary-button" type="button" onClick={onCreateDatabase}>
                创建知识库
              </button>
            </div>
          ) : (
            <div className="database-list">
              {databases.map((database) => (
                <button
                  key={database.id}
                  className={`database-list-item ${selectedDatabaseId === database.id ? 'selected' : ''}`}
                  type="button"
                  onClick={() => setSelectedDatabaseId((current) => (current === database.id ? '' : database.id))}
                >
                  <span>
                    <strong>{database.name}</strong>
                    <small>{database.description || '暂无描述'}</small>
                  </span>
                  {selectedDatabaseId === database.id && <strong>✓</strong>}
                </button>
              ))}
            </div>
          )}
          <footer className="modal-footer">
            <button className="secondary-button" type="button" onClick={() => onVisibleChange(false)}>
              取消
            </button>
            <button className="primary-button" type="button" onClick={confirm}>
              确认选择
            </button>
          </footer>
        </div>
      </div>
    </div>
  )
}
