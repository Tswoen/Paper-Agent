import { type FormEvent, useEffect, useState } from 'react'

type CreateDatabaseModalProps = {
  visible: boolean
  onVisibleChange: (visible: boolean) => void
  onSubmit: (data: { database_name: string; description: string }) => Promise<unknown>
}

export function CreateDatabaseModal({ visible, onVisibleChange, onSubmit }: CreateDatabaseModalProps) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [nameError, setNameError] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  useEffect(() => {
    if (visible) {
      setName('')
      setDescription('')
      setNameError('')
    }
  }, [visible])

  if (!visible) return null

  const validateName = () => {
    if (!name.trim()) {
      setNameError('知识库名称不能为空')
      return false
    }
    if (name.length < 2) {
      setNameError('知识库名称至少需要2个字符')
      return false
    }
    if (name.length > 50) {
      setNameError('知识库名称不能超过50个字符')
      return false
    }
    setNameError('')
    return true
  }

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    if (!validateName()) return
    setIsSubmitting(true)
    try {
      await onSubmit({ database_name: name, description })
      onVisibleChange(false)
    } catch (error) {
      console.error('创建知识库失败:', error)
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="modal-overlay" onClick={() => onVisibleChange(false)}>
      <div className="modal-container" onClick={(event) => event.stopPropagation()}>
        <header className="modal-header">
          <h2>创建知识库</h2>
          <button className="modal-close-button" type="button" onClick={() => onVisibleChange(false)}>
            ×
          </button>
        </header>
        <form className="modal-body" onSubmit={submit}>
          <label className="field">
            <span>知识库名称 *</span>
            <input value={name} className={nameError ? 'error' : ''} placeholder="请输入知识库名称" onBlur={validateName} onChange={(event) => setName(event.target.value)} />
            {nameError && <span className="field-error">{nameError}</span>}
          </label>
          <label className="field">
            <span>描述</span>
            <textarea value={description} placeholder="请输入知识库描述（可选）" rows={3} onChange={(event) => setDescription(event.target.value)} />
          </label>
          <footer className="modal-footer">
            <button className="secondary-button" type="button" onClick={() => onVisibleChange(false)}>
              取消
            </button>
            <button className="primary-button" type="submit" disabled={isSubmitting}>
              {isSubmitting ? '创建中...' : '创建'}
            </button>
          </footer>
        </form>
      </div>
    </div>
  )
}
