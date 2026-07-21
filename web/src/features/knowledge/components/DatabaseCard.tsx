import type { KnowledgeDatabase } from '../../../api/knowledge'

type DatabaseCardProps = {
  database: KnowledgeDatabase & { id: string }
  isSelected: boolean
  onSelect: (database: KnowledgeDatabase & { id: string }) => void
  onDelete: (database: KnowledgeDatabase & { id: string }) => void
}

export function DatabaseCard({ database, isSelected, onSelect, onDelete }: DatabaseCardProps) {
  return (
    <article className={`database-card ${isSelected ? 'selected' : ''}`} onClick={() => onSelect(database)}>
      <header className="database-card-header">
        <h3>{database.name}</h3>
        <button
          className="database-delete-button"
          type="button"
          title="删除知识库"
          onClick={(event) => {
            event.stopPropagation()
            onDelete(database)
          }}
        >
          ×
        </button>
      </header>
      <p className="database-description">{database.description || '暂无描述'}</p>
      <div className="database-info">
        <span>类型: {database.kb_type || 'chroma'}</span>
        {database.embedding_model && <span>模型: {database.embedding_model}</span>}
        {database.document_count !== undefined && <span>文档数: {database.document_count}</span>}
      </div>
      <footer className="database-card-footer">
        <button
          className={`database-select-button ${isSelected ? 'active' : ''}`}
          type="button"
          onClick={(event) => {
            event.stopPropagation()
            onSelect(database)
          }}
        >
          {isSelected ? '已选中' : '选择'}
        </button>
      </footer>
    </article>
  )
}
