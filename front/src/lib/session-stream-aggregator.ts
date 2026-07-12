import type {
  SessionArtifact,
  SessionRuntimeEvent,
  SessionThread,
  SessionTimelineSnapshot,
  StoredSessionEvent,
  UINodeTimelineEntry,
  UINodeTimelineGroup,
  UINodeTimelineStatus,
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

const NODE_EVENT_LABELS: Record<string, string> = {
  node_started: "开始",
  node_progress: "进行中",
  node_completed: "完成",
  node_failed: "失败",
};

export class SessionStreamAggregator {
  private messages: UISessionMessage[] = [];
  private nodeGroups: UINodeTimelineGroup[] = [];
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
    // 中文注释：节点过程单独放在 nodeGroups 里，不再混进普通消息列表，避免时间线越跑越长。
    this.nodeGroups = [];
    this.artifacts = [...thread.artifacts];
    this.isStreaming = thread.status === "running" || thread.has_pending_tool_calls;
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

    // 中文注释：兼容老数据，如果历史里还没有完整事件流，就退回消息表重建最基础的展示。
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
        this.isStreaming = event.status === "running";
        // 中文注释：只要后端明确告诉我们不是运行中，就要把卡片上的“正在生成中”也关掉。
        if (event.status && event.status !== "running") {
          this.runStartedAt = null;
          this.stopActiveAssistantStreaming();
          this.activeNodeKey = null;
        }
        return;
      case "node_started":
      case "node_progress":
      case "node_completed":
      case "node_failed":
        this.applyNodeEvent(event);
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
      nodeGroups: this.nodeGroups.map((group) => ({
        ...group,
        entries: group.entries.map((entry) => ({ ...entry })),
      })),
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

  /** 把节点级事件归到同一个节点组里，避免每一步都变成一张很大的消息卡片。 */
  private applyNodeEvent(event: SessionRuntimeEvent) {
    const group = this.ensureNodeGroup(event);
    const entry = this.createNodeEntry(event);
    group.entries.push(entry);
    group.latestMessage = entry.message || entry.label;

    if (event.event === "node_started" && !group.startedAt) {
      group.startedAt = entry.timestamp;
    }

    if (event.event === "node_failed") {
      if (this.hasRecoveryCheckpoint(event)) {
        this.markLatestResumeGroup(group, event);
      }
      this.updateNodeGroup(group, "failed", entry.timestamp, false);
      this.status = "failed";
      this.isStreaming = false;
      this.activeNodeKey = null;
      this.stopActiveAssistantStreaming();
      return;
    }

    if (event.event === "node_completed") {
      this.updateNodeGroup(group, "completed", entry.timestamp, true);
      if (this.activeNodeKey === group.nodeKey) {
        this.activeNodeKey = null;
      }
      // 中文注释：节点完成不代表整个工作流结束，所以这里不关闭整体运行状态，等待 turn_end 来最终收尾。
      this.status = "running";
      this.isStreaming = true;
      return;
    }

    this.updateNodeGroup(group, "running", null, false);
    this.status = "running";
    this.isStreaming = true;
    this.activeNodeKey = group.nodeKey;
  }

  /** 判断失败事件里是否带有后端保存的恢复位置。 */
  private hasRecoveryCheckpoint(event: SessionRuntimeEvent) {
    return Boolean(event.checkpoint && typeof event.checkpoint === "object");
  }

  /** 把最近一次可恢复失败标出来，旧失败点先隐藏按钮，避免用户不知道该点哪一个。 */
  private markLatestResumeGroup(group: UINodeTimelineGroup, event: SessionRuntimeEvent) {
    this.clearResumeMarkers();
    group.resumeAvailable = true;
    group.recoveryStatus = event.recovery_status ?? null;
    group.nextPosition = typeof event.next_position === "number" ? event.next_position : null;
    group.completed = typeof event.completed === "number" ? event.completed : null;
    group.total = typeof event.total === "number" ? event.total : null;
  }

  /** 清掉旧的继续按钮，避免恢复已经开始后还显示旧失败入口。 */
  private clearResumeMarkers() {
    for (const item of this.nodeGroups) {
      item.resumeAvailable = false;
      item.recoveryStatus = null;
      item.nextPosition = null;
      item.completed = null;
      item.total = null;
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

  /** 找到或创建当前节点对应的展示分组。 */
  private ensureNodeGroup(event: SessionRuntimeEvent): UINodeTimelineGroup {
    const nodeKey = event.node_key ?? event.node_title ?? "workflow_node";
    const turnKey = event.turn_id ?? event.run_id ?? "session";
    const groupId = `${turnKey}:${nodeKey}`;
    const existing = this.nodeGroups.find((group) => group.id === groupId);
    if (existing) {
      return existing;
    }

    const group: UINodeTimelineGroup = {
      id: groupId,
      nodeKey,
      nodeTitle: event.node_title ?? event.node_key ?? "工作流节点",
      status: "running",
      startedAt: event.timestamp ?? null,
      completedAt: null,
      latestMessage: event.message ?? event.content ?? "节点开始执行",
      isCollapsed: false,
      resumeAvailable: false,
      recoveryStatus: null,
      nextPosition: null,
      completed: null,
      total: null,
      entries: [],
    };
    this.nodeGroups.push(group);
    return group;
  }

  /** 把后端的一条节点事件整理成列表里的单行记录。 */
  private createNodeEntry(event: SessionRuntimeEvent): UINodeTimelineEntry {
    const label = NODE_EVENT_LABELS[event.event] ?? "更新";
    return {
      id: event.event_id ?? `${event.turn_id ?? event.run_id ?? "event"}:${event.node_key ?? event.node_title ?? "node"}:${event.event}:${event.stream_seq ?? createRandomId("node-event")}`,
      event: event.event,
      label,
      message: event.message ?? event.content ?? label,
      stage: event.stage ?? null,
      timestamp: event.timestamp ?? null,
      raw: event,
    };
  }

  /** 更新节点组的状态；完成的节点默认折叠，失败和运行中的节点默认展开。 */
  private updateNodeGroup(
    group: UINodeTimelineGroup,
    status: UINodeTimelineStatus,
    completedAt: string | null,
    isCollapsed: boolean,
  ) {
    group.status = status;
    group.completedAt = completedAt ?? group.completedAt;
    group.isCollapsed = isCollapsed;
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
}
