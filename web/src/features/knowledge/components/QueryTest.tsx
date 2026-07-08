import { forwardRef, useImperativeHandle, useState } from 'react'
import { knowledgeApi } from '../../../api/knowledge'

type QueryResult = {
  content?: string
  score?: number | string
  metadata?: Record<string, unknown>
}

export type QueryTestHandle = {
  clearResults: () => void
}

type QueryTestProps = {
  selectedDatabaseId: string
}

export const QueryTest = forwardRef<QueryTestHandle, QueryTestProps>(function QueryTest({ selectedDatabaseId }, ref) {
  const [queryText, setQueryText] = useState('')
  const [isQuerying, setIsQuerying] = useState(false)
  const [queryResults, setQueryResults] = useState<QueryResult[]>([])
  const [queryError, setQueryError] = useState('')
  const [hasQueried, setHasQueried] = useState(false)

  const clearResults = () => {
    setQueryResults([])
    setQueryError('')
    setHasQueried(false)
  }

  useImperativeHandle(ref, () => ({ clearResults }))

  const handleQuery = async () => {
    if (!selectedDatabaseId || !queryText.trim()) return
    setIsQuerying(true)
    setQueryError('')
    setQueryResults([])
    try {
      const response = await knowledgeApi.queryDatabase(selectedDatabaseId, queryText, {})
      if (response.data.status === 'success') {
        setQueryResults(response.data.result || [])
      } else {
        setQueryError(response.data.message || '查询失败')
      }
    } catch (error) {
      const candidate = error as { response?: { data?: { detail?: string } }; message?: string }
      console.error('查询失败:', error)
      setQueryError(candidate.response?.data?.detail || candidate.message || '查询失败，请稍后重试')
    } finally {
      setIsQuerying(false)
      setHasQueried(true)
    }
  }

  const formatScore = (score: QueryResult['score']) => (typeof score === 'number' ? `${(score * 100).toFixed(2)}%` : score)

  return (
    <div className="query-test-container">
      <header className="query-header">
        <h3>知识库查询测试</h3>
        {!selectedDatabaseId && <p className="query-hint">请先选择一个知识库</p>}
      </header>
      <div className="query-input-section">
        <textarea
          value={queryText}
          placeholder="输入您的问题..."
          rows={3}
          disabled={!selectedDatabaseId || isQuerying}
          onChange={(event) => setQueryText(event.target.value)}
          onKeyDown={(event) => {
            if (event.ctrlKey && event.key === 'Enter') void handleQuery()
          }}
        />
        <button className="primary-button" type="button" disabled={!selectedDatabaseId || !queryText.trim() || isQuerying} onClick={() => void handleQuery()}>
          {isQuerying ? '查询中...' : '查询'}
        </button>
      </div>
      {(queryResults.length > 0 || queryError) && (
        <div className="query-results">
          {queryError ? (
            <div className="error-message">{queryError}</div>
          ) : (
            queryResults.map((result, index) => (
              <article key={index} className="result-item">
                <header>
                  <strong>#{index + 1}</strong>
                  <span>相似度: {formatScore(result.score)}</span>
                </header>
                <p>{result.content}</p>
                {result.metadata && (
                  <div className="result-meta">
                    {Object.entries(result.metadata).map(([key, value]) => (
                      <span key={key}>
                        {key}: {String(value)}
                      </span>
                    ))}
                  </div>
                )}
              </article>
            ))
          )}
        </div>
      )}
      {hasQueried && !isQuerying && queryResults.length === 0 && !queryError && (
        <div className="empty-state">
          <p>暂无查询结果</p>
        </div>
      )}
    </div>
  )
})
