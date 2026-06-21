import type { BackendEvent, ThreadMessage } from "./types";
import { uid } from "./utils";

function findActiveAssistant(messages: ThreadMessage[], turnId?: string): ThreadMessage | undefined {
  if (!turnId) {
    return undefined;
  }

  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const current = messages[index];
    if (current.role === "assistant" && current.turn_id === turnId) {
      return current;
    }
  }

  return undefined;
}

function ensureAssistant(messages: ThreadMessage[], event: BackendEvent): ThreadMessage {
  const existing = findActiveAssistant(messages, event.turn_id);
  if (existing) {
    return existing;
  }

  const assistant: ThreadMessage = {
    id: uid(),
    role: "assistant",
    content: "",
    reasoning: "",
    is_streaming: true,
    turn_id: event.turn_id ?? null,
  };
  messages.push(assistant);
  return assistant;
}

export function applyBackendEvents(baseMessages: ThreadMessage[], events: BackendEvent[]): ThreadMessage[] {
  const messages: ThreadMessage[] = baseMessages.map((message) => ({
    ...message,
    reasoning: message.reasoning ?? "",
  }));

  for (const event of events) {
    if (event.event === "message") {
      if (event.role === "user") {
        const exists = messages.some((item) => item.role === "user" && item.turn_id === event.turn_id);
        if (!exists) {
          messages.push({
            id: event.id ?? uid(),
            role: "user",
            content: String(event.content ?? ""),
            reasoning: "",
            media: Array.isArray(event.media) ? event.media : [],
            turn_id: event.turn_id ?? null,
          });
        }
        continue;
      }

      const assistant = ensureAssistant(messages, event);
      assistant.content = String(event.content ?? assistant.content ?? "");
      continue;
    }

    if (event.event === "reasoning_delta") {
      // 中文注释：把后端返回的推理增量合并到同一条 assistant 消息，便于前端连续展示。
      const assistant = ensureAssistant(messages, event);
      assistant.reasoning = `${assistant.reasoning ?? ""}${String(event.content ?? event.delta ?? "")}`;
      assistant.reasoning_streaming = true;
      continue;
    }

    if (event.event === "delta") {
      // 中文注释：正文内容也按轮次累加，避免每个 delta 都变成一条独立消息。
      const assistant = ensureAssistant(messages, event);
      assistant.content = `${assistant.content ?? ""}${String(event.content ?? event.delta ?? "")}`;
      assistant.is_streaming = true;
      continue;
    }

    if (event.event === "reasoning_end") {
      const assistant = findActiveAssistant(messages, event.turn_id);
      if (assistant) {
        assistant.reasoning_streaming = false;
      }
      continue;
    }

    if (event.event === "stream_end" || event.event === "turn_end") {
      const assistant = findActiveAssistant(messages, event.turn_id);
      if (assistant) {
        assistant.is_streaming = false;
      }
    }
  }

  return messages;
}
