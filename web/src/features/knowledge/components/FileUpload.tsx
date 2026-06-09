import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react'
import { knowledgeApi } from '../../../api/knowledge'

export type UploadQueueItem = {
  id: string
  file: File
  name: string
  size: number
  status: 'pending' | 'uploading' | 'success' | 'failed'
  progress: number
  error: string | null
}

export type FileUploadHandle = {
  clearQueue: () => void
}

type FileUploadProps = {
  selectedDatabaseId: string
  onUploadComplete?: (file: UploadQueueItem) => void
  onUploadError?: (file: UploadQueueItem) => void
}

export const FileUpload = forwardRef<FileUploadHandle, FileUploadProps>(function FileUpload({ selectedDatabaseId, onUploadComplete, onUploadError }, ref) {
  const [isDragOver, setIsDragOver] = useState(false)
  const [uploadQueue, setUploadQueue] = useState<UploadQueueItem[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [supportedTypes, setSupportedTypes] = useState<string[]>([])
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const uploadQueueRef = useRef<UploadQueueItem[]>([])
  const isUploadingRef = useRef(false)

  const acceptTypes = useMemo(() => supportedTypes.map((type) => `${type}`).join(','), [supportedTypes])

  useImperativeHandle(ref, () => ({
    clearQueue: () => updateQueue([])
  }))

  useEffect(() => {
    const loadSupportedTypes = async () => {
      try {
        const response = await knowledgeApi.getSupportedTypes()
        setSupportedTypes(response.data.file_types || [])
      } catch (error) {
        console.error('加载支持的文件类型失败:', error)
        setSupportedTypes(['pdf', 'docx', 'txt', 'md'])
      }
    }
    void loadSupportedTypes()
  }, [])

  const updateQueue = (next: UploadQueueItem[] | ((current: UploadQueueItem[]) => UploadQueueItem[])) => {
    setUploadQueue((current) => {
      const resolved = typeof next === 'function' ? next(current) : next
      uploadQueueRef.current = resolved
      return resolved
    })
  }

  const updateQueueItem = (id: string, patch: Partial<UploadQueueItem>) => {
    updateQueue((current) => current.map((item) => (item.id === id ? { ...item, ...patch } : item)))
  }

  const uploadFile = async (item: UploadQueueItem) => {
    updateQueueItem(item.id, { status: 'uploading', progress: 0, error: null })

    try {
      const response = await knowledgeApi.uploadFile(item.file, selectedDatabaseId)
      const filePath = response.data.file_path
      await knowledgeApi.addDocuments(selectedDatabaseId, [filePath], { content_type: 'file' })
      const completedItem = { ...item, status: 'success' as const, progress: 100, error: null }
      updateQueueItem(item.id, completedItem)
      onUploadComplete?.(completedItem)
    } catch (error) {
      const candidate = error as { response?: { data?: { detail?: string } }; message?: string }
      const failedItem = {
        ...item,
        status: 'failed' as const,
        error: candidate.response?.data?.detail || candidate.message || '上传失败'
      }
      updateQueueItem(item.id, failedItem)
      onUploadError?.(failedItem)
    }
  }

  const processQueue = async () => {
    if (isUploadingRef.current) return
    isUploadingRef.current = true
    setIsUploading(true)
    try {
      for (let index = 0; index < uploadQueueRef.current.length; index += 1) {
        const item = uploadQueueRef.current[index]
        if (item?.status === 'pending' || item?.status === 'failed') {
          await uploadFile(item)
        }
      }
    } finally {
      isUploadingRef.current = false
      setIsUploading(false)
    }
  }

  const addFilesToQueue = (files: File[]) => {
    const items = files.map((file) => ({
      id: `${file.name}-${file.lastModified}-${file.size}-${crypto.randomUUID()}`,
      file,
      name: file.name,
      size: file.size,
      status: 'pending' as const,
      progress: 0,
      error: null
    }))
    updateQueue((current) => [...current, ...items])
    void processQueue()
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return `${Math.round((bytes / Math.pow(k, i)) * 100) / 100} ${sizes[i]}`
  }

  const getStatusText = (status: UploadQueueItem['status']) => {
    const statusMap = {
      pending: '等待中',
      uploading: '上传中',
      success: '成功',
      failed: '失败'
    }
    return statusMap[status]
  }

  return (
    <div className="file-upload-container">
      <div
        className={`upload-zone ${isDragOver ? 'drag-over' : ''} ${!selectedDatabaseId ? 'disabled' : ''}`}
        onDragOver={(event) => {
          event.preventDefault()
          if (selectedDatabaseId) setIsDragOver(true)
        }}
        onDragLeave={(event) => {
          event.preventDefault()
          setIsDragOver(false)
        }}
        onDrop={(event) => {
          event.preventDefault()
          if (!selectedDatabaseId) return
          setIsDragOver(false)
          addFilesToQueue(Array.from(event.dataTransfer.files))
        }}
        onClick={() => selectedDatabaseId && fileInputRef.current?.click()}
      >
        <div className="upload-icon">FILE</div>
        <p>{selectedDatabaseId ? '拖拽文件到此处或点击上传' : '请先选择知识库'}</p>
        {selectedDatabaseId && <p className="upload-hint">支持的文件类型: {supportedTypes.join(', ')}</p>}
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept={acceptTypes}
          hidden
          onChange={(event) => {
            addFilesToQueue(Array.from(event.target.files || []))
            event.target.value = ''
          }}
        />
      </div>
      {uploadQueue.length > 0 && (
        <div className="upload-progress">
          <h3>上传队列</h3>
          {uploadQueue.map((file, index) => (
            <div key={`${file.name}-${index}`} className="file-item">
              <span>{file.name}</span>
              <span>{formatFileSize(file.size)}</span>
              <strong className={`status-badge ${file.status}`}>{getStatusText(file.status)}</strong>
              {file.status === 'uploading' && <span>{file.progress}%</span>}
            </div>
          ))}
          <button className="secondary-button" type="button" disabled={isUploading} onClick={() => updateQueue([])}>
            清空队列
          </button>
        </div>
      )}
    </div>
  )
})
