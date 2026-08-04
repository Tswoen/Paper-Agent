import type {
  RuntimeDetailContent,
  SessionArtifact,
  SessionRuntimeEvent,
  SessionThread,
  SessionTimelineSnapshot,
  StoredSessionEvent,
  UIRuntimeTimelineEvent,
  UISessionMessage,
} from "../types/sessions";
import { createRandomId } from "./random-id";

function createMessage(partial: Partial<UISessionMessage>): UISessionMessage {
  return {
    id: partial.id ?? createRandomId("message"),
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

function createRuntimeEvent(partial: Partial<UIRuntimeTimelineEvent> & { id: string }): UIRuntimeTimelineEvent {
  return {
    id: partial.id,
    parentId: partial.parentId ?? null,
    type: partial.type ?? "runtime_event",
    title: partial.title ?? "执行事件",
    status: partial.status ?? "running",
    showContent: partial.showContent ?? "正在处理",
    detailContent: partial.detailContent ?? null,
    metadata: partial.metadata ?? {},
    createdAt: partial.createdAt ?? null,
    updatedAt: partial.updatedAt ?? null,
    completedAt: partial.completedAt ?? null,
    children: partial.children ?? [],
    isCollapsed: partial.isCollapsed ?? false,
    resumeAvailable: partial.resumeAvailable ?? false,
    recoveryStatus: partial.recoveryStatus ?? null,
    nextPosition: partial.nextPosition ?? null,
    completed: partial.completed ?? null,
    total: partial.total ?? null,
    inputTokens: partial.inputTokens ?? 0,
    outputTokens: partial.outputTokens ?? 0,
    raw: partial.raw ?? ({ event: "runtime_event", session_key: "" } as SessionRuntimeEvent),
  };
}

const FINISHED_STATUSES = new Set(["completed", "failed", "cancelled", "skipped"]);

export class SessionStreamAggregator {
  private messages: UISessionMessage[] = [];
  private runtimeEvents: UIRuntimeTimelineEvent[] = [];
  private runtimeEventMap = new Map<string, UIRuntimeTimelineEvent>();
  private artifacts: SessionArtifact[] = [];
  private isStreaming = false;
  private runStartedAt: string | null = null;
  private streamError: SessionRuntimeEvent | null = null;
  private status = "created";
  private activeAssistantId: string | null = null;
  private activeNodeKey: string | null = null;

  /** 用线程快照重建时间线状态，保证历史回放和实时展示走同一条路。 */
  hydrate(thread: SessionThread) {
    this.messages = [];
    // 中文注释：执行过程现在统一放在 runtimeEvents 里，前端只按 id 更新事件，不再追加旧 node 行。
    this.runtimeEvents = [];
    this.runtimeEventMap = new Map();
    this.artifacts = [...thread.artifacts];
    this.isStreaming = thread.status === "running" || thread.status === "cancel_requested" || thread.has_pending_tool_calls;
    this.runStartedAt = thread.run_started_at;
    this.streamError = null;
    this.status = thread.status;
    this.activeAssistantId = null;
    this.activeNodeKey = null;

    if (thread.events.length > 0) {
      for (const storedEvent of thread.events) {
        this.apply(this.normalizeStoredEvent(storedEvent));
      }
    }

    // 中文注释：兼容没有事件流的老数据，如果历史里没有消息事件，就从消息表重建基础聊天内容。
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

  /** 乐观插入一条用户消息，减少提交时的等待感。 */
  addOptimisticUserMessage(content: string, turnId: string) {
    this.clearResumeMarkers();
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

  /** 应用一条运行事件，并把它折叠成前端可直接渲染的快照。 */
  apply(event: SessionRuntimeEvent) {
    switch (event.event) {
      case "runtime_event":
        this.applyRuntimeEvent(event);
        return;
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
        if ("run_started_at" in event) {
          this.runStartedAt = event.run_started_at ?? null;
        }
        this.isStreaming = event.status === "running" || event.status === "cancel_requested";
        // 中文注释：只有正在运行或正在停止时保留忙碌状态，其他状态都要结束“正在生成中”。
        if (event.status && event.status !== "running" && event.status !== "cancel_requested") {
          this.runStartedAt = null;
          this.stopActiveAssistantStreaming();
          this.activeNodeKey = null;
        }
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
        this.activeNodeKey = null;
        this.stopActiveAssistantStreaming();
        return;
      case "turn_end":
        this.status = event.status ?? this.status;
        if (event.status === "cancelled") {
          // 用户主动停止后，后端不会再为每个正在运行的子节点单独发送结束事件。
          // 这里统一补上取消状态，避免整体已经停止但节点卡片仍显示“处理中”。
          this.markUnfinishedRuntimeEventsCancelled(event);
        }
        this.isStreaming = false;
        this.activeNodeKey = null;
        this.stopActiveAssistantStreaming();
        return;
      default:
        return;
    }
  }

  /** 返回当前时间线快照。 */
  snapshot(): SessionTimelineSnapshot {
    return {
      messages: [...this.messages],
      runtimeEvents: this.runtimeEvents.map((event) => this.cloneRuntimeEvent(event)),
      activeNodeKey: this.activeNodeKey,
      artifacts: [...this.artifacts],
      isStreaming: this.isStreaming,
      runStartedAt: this.runStartedAt,
      streamError: this.streamError,
      status: this.status,
    };
  }

  /** 把历史事件表里的记录转成和 SSE 对齐的统一结构。 */
  private normalizeStoredEvent(event: StoredSessionEvent): SessionRuntimeEvent {
    const metadata = event.metadata ?? {};
    if (event.event_type === "status_change") {
      const normalized: SessionRuntimeEvent = {
        event: "status",
        session_key: String(metadata.session_key ?? ""),
        status: String(metadata.status ?? event.content ?? "created"),
        turn_id: typeof metadata.turn_id === "string" ? metadata.turn_id : undefined,
        timestamp: String(metadata.timestamp ?? event.created_at),
      };
      if ("run_started_at" in metadata) {
        normalized.run_started_at = (metadata.run_started_at as string | null | undefined) ?? null;
      }
      return normalized;
    }

    return {
      ...(metadata as SessionRuntimeEvent),
      event: event.event_type,
      session_key: String(metadata.session_key ?? ""),
      content: String(metadata.content ?? metadata.message ?? event.content ?? ""),
      timestamp: String(metadata.timestamp ?? event.created_at),
      stream_seq: Number(metadata.stream_seq ?? event.seq_no),
    };
  }

  /** 处理普通 message 事件，包括用户消息、助手消息和旧版 progress 卡片。 */
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

  /** 按 runtime_event.id 更新执行过程；同一个 id 永远只显示一个事件。 */
  private applyRuntimeEvent(event: SessionRuntimeEvent) {
    const eventId = String(event.id ?? "").trim();
    if (!eventId) {
      return;
    }

    const item = this.ensureRuntimeEvent(eventId);
    this.updateRuntimeEvent(item, event);

    if (item.parentId) {
      const parent = this.ensureRuntimeEvent(item.parentId);
      this.attachChild(parent, item);
    } else {
      this.attachRoot(item);
    }
    // 每次子卡片变化后都重新汇总，父卡片始终等于所有子卡片之和。
    this.recalculateTokenTotals();

    if (item.status === "failed" && this.hasRecoveryCheckpoint(item)) {
      this.markLatestResumeEvent(item);
    }

    if (item.status === "failed") {
      this.status = "failed";
      this.isStreaming = false;
      this.activeNodeKey = null;
      this.stopActiveAssistantStreaming();
      return;
    }

    if (item.status === "running") {
      this.status = "running";
      this.isStreaming = true;
      this.activeNodeKey = this.nodeKeyFromRuntimeEvent(item);
      return;
    }

    if (item.status === "completed") {
      // 中文注释：某个事件完成不代表整个工作流完成，所以这里只更新事件本身，最终状态仍等 turn_end。
      this.status = this.status === "created" ? "running" : this.status;
    }
  }

  /** 找到或创建一个执行事件，子事件先到时也能先占位。 */
  private ensureRuntimeEvent(id: string): UIRuntimeTimelineEvent {
    const existing = this.runtimeEventMap.get(id);
    if (existing) {
      return existing;
    }
    const item = createRuntimeEvent({
      id,
      title: "执行事件",
      showContent: "等待事件详情",
      raw: { event: "runtime_event", session_key: "" },
    });
    this.runtimeEventMap.set(id, item);
    return item;
  }

  /** 用后端新事件覆盖旧显示内容，metadata 则保留旧字段并用新字段覆盖。 */
  private updateRuntimeEvent(item: UIRuntimeTimelineEvent, event: SessionRuntimeEvent) {
    const metadata = event.metadata ?? {};
    item.parentId = typeof event.parent_id === "string" && event.parent_id ? event.parent_id : null;
    item.type = event.type ?? item.type;
    item.title = event.title ?? item.title;
    item.status = event.status ?? item.status;
    item.showContent = event.show_content ?? event.message ?? event.content ?? item.showContent;
    item.detailContent = normalizeDetailContent(event.detail_content ?? item.detailContent);
    item.metadata = { ...item.metadata, ...metadata };
    item.createdAt = event.created_at ?? event.timestamp ?? item.createdAt;
    item.updatedAt = event.updated_at ?? event.timestamp ?? item.updatedAt;
    item.completedAt = event.completed_at ?? (FINISHED_STATUSES.has(item.status) ? item.updatedAt : item.completedAt);
    item.isCollapsed = item.status === "completed";
    item.recoveryStatus = typeof metadata.recovery_status === "string" ? metadata.recovery_status : item.recoveryStatus;
    item.nextPosition = typeof metadata.next_position === "number" ? metadata.next_position : item.nextPosition;
    item.completed = typeof metadata.completed === "number" ? metadata.completed : item.completed;
    item.total = typeof metadata.total === "number" ? metadata.total : item.total;
    if (typeof event.input_tokens === "number") {
      item.inputTokens = Math.max(0, event.input_tokens);
    } else if (typeof metadata.input_tokens === "number") {
      item.inputTokens = Math.max(0, metadata.input_tokens);
    }
    if (typeof event.output_tokens === "number") {
      item.outputTokens = Math.max(0, event.output_tokens);
    } else if (typeof metadata.output_tokens === "number") {
      item.outputTokens = Math.max(0, metadata.output_tokens);
    }
    item.raw = event;
  }

  /** 叶子卡片使用模型返回值，父卡片只显示当前所有子卡片的合计。 */
  private recalculateTokenTotals() {
    const update = (event: UIRuntimeTimelineEvent): { input: number; output: number } => {
      if (event.children.length === 0) {
        return { input: event.inputTokens, output: event.outputTokens };
      }
      const total = event.children.reduce(
        (sum, child) => {
          const childTotal = update(child);
          return {
            input: sum.input + childTotal.input,
            output: sum.output + childTotal.output,
          };
        },
        { input: 0, output: 0 },
      );
      event.inputTokens = total.input;
      event.outputTokens = total.output;
      return total;
    };

    for (const event of this.runtimeEvents) {
      update(event);
    }
  }

  /**
   * 将运行树中还没有结束的节点统一标记为已取消。
   * 已完成、已失败或已经跳过的节点不改动，保证用户仍能看到中断前已经完成的结果。
   */
  private markUnfinishedRuntimeEventsCancelled(event: SessionRuntimeEvent) {
    const timestamp = event.timestamp ?? new Date().toISOString();
    const update = (item: UIRuntimeTimelineEvent) => {
      if (item.status === "running" || item.status === "pending" || item.status === "cancel_requested") {
        item.status = "cancelled";
        item.showContent = "已停止";
        item.updatedAt = timestamp;
        item.completedAt = timestamp;
        item.metadata = {
          ...item.metadata,
          cancellation_reason: "user_requested",
        };
        item.raw = {
          ...item.raw,
          event: "runtime_event",
          status: "cancelled",
          message: "任务已按用户请求停止",
          timestamp,
        };
      }

      for (const child of item.children) {
        update(child);
      }
    };

    for (const item of this.runtimeEvents) {
      update(item);
    }
    this.recalculateTokenTotals();
  }

  /** 把子事件挂到父事件下面；如果已经挂过，就只保持原位置，不重复插入。 */
  private attachChild(parent: UIRuntimeTimelineEvent, child: UIRuntimeTimelineEvent) {
    this.runtimeEvents = this.runtimeEvents.filter((event) => event.id !== child.id);
    if (!parent.children.some((event) => event.id === child.id)) {
      parent.children.push(child);
    }
    this.attachRoot(parent);
  }

  /** 把没有父级的事件放到根列表。 */
  private attachRoot(item: UIRuntimeTimelineEvent) {
    if (!this.runtimeEvents.some((event) => event.id === item.id)) {
      this.runtimeEvents.push(item);
    }
  }

  /** 判断失败事件里是否带有后端保存的恢复位置。 */
  private hasRecoveryCheckpoint(event: UIRuntimeTimelineEvent) {
    const checkpoint = event.metadata.checkpoint;
    return Boolean(checkpoint && typeof checkpoint === "object");
  }

  /** 把最近一次可恢复失败标出来，旧失败点先隐藏按钮，避免用户不知道该点哪一个。 */
  private markLatestResumeEvent(event: UIRuntimeTimelineEvent) {
    this.clearResumeMarkers();
    event.resumeAvailable = true;
  }

  /** 清掉旧的继续按钮，避免恢复已经开始后还显示旧失败入口。 */
  private clearResumeMarkers() {
    for (const item of this.runtimeEventMap.values()) {
      // 中文注释：这里只隐藏按钮，不清掉 completed/total 等进度数字；这些数字也是事件自己的元数据。
      item.resumeAvailable = false;
    }
  }

  /** 处理 artifact 事件，并把产物挂到当前助手消息下面。 */
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

  /** 确保当前存在一张可以承接流式输出的 assistant 卡片。 */
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

  /** 把当前 assistant 标记成正在流式输出。 */
  private ensureAssistantStreaming(event: SessionRuntimeEvent) {
    const assistant = this.ensureActiveAssistant(event);
    assistant.isStreaming = true;
    this.isStreaming = true;
    this.status = "running";
  }

  /** 失败或结束时，同时关掉整体状态和单条消息状态，避免界面残留“正在生成中”。 */
  private stopActiveAssistantStreaming() {
    const assistant = this.activeAssistant();
    if (assistant) {
      assistant.isStreaming = false;
      assistant.reasoningStreaming = false;
    }
    this.activeAssistantId = null;
  }

  /** 取出当前正在拼接中的 assistant 消息。 */
  private activeAssistant(): UISessionMessage | undefined {
    if (!this.activeAssistantId) {
      return undefined;
    }
    return this.messages.find((message) => message.id === this.activeAssistantId);
  }

  /** 从 runtime_event 的 metadata 里取节点 key，供顶部运行状态做轻量提示。 */
  private nodeKeyFromRuntimeEvent(item: UIRuntimeTimelineEvent) {
    const nodeKey = item.metadata.node_key;
    return typeof nodeKey === "string" ? nodeKey : null;
  }

  /** 复制执行事件树，避免 Vue 组件意外修改聚合器内部状态。 */
  private cloneRuntimeEvent(event: UIRuntimeTimelineEvent): UIRuntimeTimelineEvent {
    return {
      ...event,
      metadata: { ...event.metadata },
      inputTokens: event.inputTokens,
      outputTokens: event.outputTokens,
      children: event.children.map((child) => this.cloneRuntimeEvent(child)),
    };
  }
}

function normalizeDetailContent(value: RuntimeDetailContent | undefined): RuntimeDetailContent {
  if (value === undefined) {
    return null;
  }
  if (value === null || typeof value === "string") {
    return value;
  }
  return { ...value };
}
