import type {
  SessionArtifact,
  SessionRuntimeEvent,
  SessionThread,
  SessionTimelineSnapshot,
  StoredSessionEvent,
  UISessionMessage,
} from "../types/sessions";

function createMessage(partial: Partial<UISessionMessage>): UISessionMessage {
  return {
    id: partial.id ?? crypto.randomUUID(),
    role: partial.role ?? "assistant",
    kind: partial.kind ?? "message",
    content: partial.content ?? "",
    reasoning: partial.reasoning ?? "",
    isStreaming: partial.isStreaming ?? false,
    reasoningStreaming: partial.reasoningStreaming ?? false,
    media: partial.media ?? [],
    toolEvents: partial.toolEvents ?? [],
    artifactRefs: partial.artifactRefs ?? [],
    turnId: partial.turnId ?? null,
    createdAt: partial.createdAt ?? null,
  };
}

export class SessionStreamAggregator {
  private messages: UISessionMessage[] = [];
  private artifacts: SessionArtifact[] = [];
  private isStreaming = false;
  private runStartedAt: string | null = null;
  private streamError: SessionRuntimeEvent | null = null;
  private status = "created";
  private activeAssistantId: string | null = null;

  /** 以线程快照重建时间线状态，保证历史和实时渲染口径一致。 */
  hydrate(thread: SessionThread) {
    this.messages = [];
    this.artifacts = [...thread.artifacts];
    this.isStreaming = thread.status === "running" || thread.has_pending_tool_calls;
    this.runStartedAt = thread.run_started_at;
    this.streamError = null;
    this.status = thread.status;
    this.activeAssistantId = null;

    if (thread.events.length > 0) {
      for (const storedEvent of thread.events) {
        this.apply(this.normalizeStoredEvent(storedEvent));
      }
    }

    // 中文注释：兼容历史老数据，如果事件流为空，则退回到消息表重建最基础时间线。
    if (this.messages.length === 0) {
      for (const message of thread.messages) {
        this.messages.push(
          createMessage({
            id: message.id,
            role: message.role,
            kind: message.kind ?? "message",
            content: message.content,
            reasoning: message.reasoning ?? "",
            media: message.media ?? [],
            turnId: message.turn_id ?? null,
            createdAt: message.created_at,
          }),
        );
      }
    }
  }

  /** 乐观插入一条用户消息，提升发送时的响应感知。 */
  addOptimisticUserMessage(content: string, turnId: string) {
    this.messages.push(
      createMessage({
        role: "user",
        content,
        turnId,
        createdAt: new Date().toISOString(),
      }),
    );
    this.isStreaming = true;
    this.status = "running";
  }

  /** 应用一条实时运行事件并刷新聚合状态。 */
  apply(event: SessionRuntimeEvent) {
    switch (event.event) {
      case "message":
        this.applyMessageEvent(event);
        return;
      case "delta":
        this.ensureActiveAssistant(event).content += event.content ?? event.delta ?? "";
        this.ensureAssistantStreaming(event);
        return;
      case "reasoning_delta":
        this.ensureActiveAssistant(event).reasoning += event.content ?? event.delta ?? "";
        this.ensureActiveAssistant(event).reasoningStreaming = true;
        this.ensureAssistantStreaming(event);
        return;
      case "reasoning_end":
        this.ensureActiveAssistant(event).reasoningStreaming = false;
        return;
      case "artifact":
        this.applyArtifactEvent(event);
        return;
      case "status":
        this.status = event.status ?? this.status;
        this.runStartedAt = event.run_started_at ?? this.runStartedAt;
        this.isStreaming = event.status === "running";
        return;
      case "error":
        this.streamError = event;
        this.messages.push(
          createMessage({
            role: "system",
            kind: "error",
            content: event.message ?? event.content ?? "运行失败",
            turnId: event.turn_id ?? null,
            createdAt: event.timestamp ?? new Date().toISOString(),
          }),
        );
        this.status = "failed";
        this.isStreaming = false;
        this.activeAssistantId = null;
        return;
      case "turn_end":
        this.status = event.status ?? this.status;
        this.isStreaming = false;
        if (this.activeAssistantId) {
          const assistant = this.activeAssistant();
          if (assistant) {
            assistant.isStreaming = false;
            assistant.reasoningStreaming = false;
          }
        }
        this.activeAssistantId = null;
        return;
      default:
        return;
    }
  }

  /** 返回当前前端可直接消费的时间线快照。 */
  snapshot(): SessionTimelineSnapshot {
    return {
      messages: [...this.messages],
      artifacts: [...this.artifacts],
      isStreaming: this.isStreaming,
      runStartedAt: this.runStartedAt,
      streamError: this.streamError,
      status: this.status,
    };
  }

  /** 将落库事件转换为与 SSE 对齐的统一结构。 */
  private normalizeStoredEvent(event: StoredSessionEvent): SessionRuntimeEvent {
    const metadata = event.metadata ?? {};
    if (event.event_type === "status_change") {
      return {
        event: "status",
        session_key: String(metadata.session_key ?? ""),
        status: String(metadata.status ?? event.content ?? "created"),
        run_started_at: (metadata.run_started_at as string | null | undefined) ?? null,
        turn_id: typeof metadata.turn_id === "string" ? metadata.turn_id : undefined,
        timestamp: String(metadata.timestamp ?? event.created_at),
      };
    }

    return {
      ...(metadata as SessionRuntimeEvent),
      event: event.event_type,
      session_key: String(metadata.session_key ?? ""),
      content: String(
        metadata.content ??
          metadata.message ??
          event.content ??
          "",
      ),
      timestamp: String(metadata.timestamp ?? event.created_at),
      stream_seq: Number(metadata.stream_seq ?? event.seq_no),
    };
  }

  /** 处理 message 事件，包括用户消息、助手消息与进度卡片。 */
  private applyMessageEvent(event: SessionRuntimeEvent) {
    const kind = event.kind ?? "message";
    const role = event.role ?? "assistant";

    if (kind === "progress" || kind === "tool" || kind === "tool_hint") {
      this.messages.push(
        createMessage({
          role: "system",
          kind,
          content: event.content ?? "",
          turnId: event.turn_id ?? null,
          createdAt: event.timestamp ?? new Date().toISOString(),
        }),
      );
      this.isStreaming = true;
      this.status = "running";
      return;
    }

    if (role === "user") {
      const existing = this.messages.find(
        (message) =>
          message.role === "user"
          && message.turnId === (event.turn_id ?? null)
          && message.content === (event.content ?? ""),
      );
      if (!existing) {
        this.messages.push(
          createMessage({
            role: "user",
            content: event.content ?? "",
            media: event.media ?? [],
            turnId: event.turn_id ?? null,
            createdAt: event.timestamp ?? new Date().toISOString(),
          }),
        );
      }
      return;
    }

    const assistant = this.ensureActiveAssistant(event);
    assistant.content = event.content ?? assistant.content;
    assistant.media = [...assistant.media, ...(event.media ?? [])];
    assistant.createdAt = event.timestamp ?? assistant.createdAt;
    assistant.isStreaming = true;
    this.isStreaming = true;
    this.status = "running";
  }

  /** 处理 artifact 事件，并把产物挂到当前 assistant 消息下。 */
  private applyArtifactEvent(event: SessionRuntimeEvent) {
    const artifact = event.artifact;
    if (!artifact || typeof artifact !== "object") {
      return;
    }
    const artifactId = String((artifact as Record<string, unknown>).id ?? (artifact as Record<string, unknown>).artifact_id ?? "");
    if (artifactId && !this.artifacts.some((item) => item.id === artifactId)) {
      this.artifacts.push({
        id: artifactId,
        artifact_type: String((artifact as Record<string, unknown>).artifact_type ?? "artifact"),
        name: String((artifact as Record<string, unknown>).name ?? "artifact"),
        path: String((artifact as Record<string, unknown>).path ?? ""),
        size: Number((artifact as Record<string, unknown>).size ?? 0),
        created_at: String((artifact as Record<string, unknown>).created_at ?? event.timestamp ?? new Date().toISOString()),
        metadata: ((artifact as Record<string, unknown>).metadata as Record<string, unknown> | undefined) ?? {},
      });
    }
    this.ensureActiveAssistant(event).artifactRefs.push(artifact);
  }

  /** 确保当前存在一个用于承接流式增量的 assistant 卡片。 */
  private ensureActiveAssistant(event: SessionRuntimeEvent): UISessionMessage {
    const existing = this.activeAssistant();
    if (existing) {
      return existing;
    }
    const message = createMessage({
      role: "assistant",
      isStreaming: true,
      turnId: event.turn_id ?? null,
      createdAt: event.timestamp ?? new Date().toISOString(),
    });
    this.messages.push(message);
    this.activeAssistantId = message.id;
    return message;
  }

  /** 把当前活动 assistant 标记为流式输出中。 */
  private ensureAssistantStreaming(event: SessionRuntimeEvent) {
    const assistant = this.ensureActiveAssistant(event);
    assistant.isStreaming = true;
    this.isStreaming = true;
    this.status = "running";
  }

  /** 获取当前处于流式拼接中的 assistant 消息。 */
  private activeAssistant(): UISessionMessage | undefined {
    if (!this.activeAssistantId) {
      return undefined;
    }
    return this.messages.find((message) => message.id === this.activeAssistantId);
  }
}
