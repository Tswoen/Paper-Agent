import { AssistantRuntimeProvider, type AppendMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import type { PropsWithChildren } from 'react'
import { useMemo } from 'react'
import type { AssistantUiMessage } from './researchRun'

type AssistantRuntimeBridgeProps = PropsWithChildren<{
  messages: AssistantUiMessage[]
  isRunning: boolean
  isSendDisabled: boolean
  onUserMessage: (text: string) => Promise<void> | void
  onCancel: () => Promise<void> | void
}>

const getAppendMessageText = (message: AppendMessage): string => {
  if (typeof message.content === 'string') return message.content
  if (Array.isArray(message.content)) {
    return message.content
      .map((part) => ('text' in part && typeof part.text === 'string' ? part.text : ''))
      .join('')
      .trim()
  }
  return ''
}

export function AssistantRuntimeBridge({
  messages,
  isRunning,
  isSendDisabled,
  onUserMessage,
  onCancel,
  children
}: AssistantRuntimeBridgeProps) {
  const adapter = useMemo(
    () => ({
      messages,
      isRunning,
      isSendDisabled,
      unstable_capabilities: {
        copy: true
      },
      convertMessage: (message: AssistantUiMessage) => ({
        id: message.id,
        role: message.role,
        content: message.content
      }),
      onNew: async (message: AppendMessage) => {
        const text = getAppendMessageText(message)
        if (text) {
          await onUserMessage(text)
        }
      },
      onCancel: async () => {
        await onCancel()
      }
    }),
    [isRunning, isSendDisabled, messages, onCancel, onUserMessage]
  )

  const runtime = useExternalStoreRuntime(adapter)

  return <AssistantRuntimeProvider runtime={runtime}>{children}</AssistantRuntimeProvider>
}
