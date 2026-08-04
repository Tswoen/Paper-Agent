export type SessionStatus =
  | "created"
  | "running"
  | "cancel_requested"
  | "cancelled"
  | "completed"
  | "failed";

export interface SessionSummary {
  key: string;
  title: string;
  preview: string;
  updated_at: string;
  status: SessionStatus;
  run_started_at: string | null;
}

export interface SessionMessageRecord {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  created_at: string;
  reasoning?: string;
  media?: Array<Record<string, unknown>>;
  kind?: string;
  turn_id?: string | null;
  turn_phase?: string | null;
  turn_seq?: number | null;
}

export interface StoredSessionEvent {
  id: string;
  event_type: string;
  content: string;
  created_at: string;
  seq_no: number;
  metadata: Record<string, unknown>;
}

export interface SessionArtifact {
  id: string;
  artifact_type: string;
  name: string;
  path: string;
  size: number;
  created_at: string;
  metadata: Record<string, unknown>;
}

export interface SessionThread {
  key: string;
  title: string;
  status: SessionStatus | string;
  messages: SessionMessageRecord[];
  events: StoredSessionEvent[];
  artifacts: SessionArtifact[];
  has_pending_tool_calls: boolean;
  run_started_at: string | null;
}

export interface SessionListPayload {
  sessions: SessionSummary[];
}

export interface SessionCreatePayload {
  session: SessionSummary;
}

export interface SessionThreadPayload extends SessionThread {}

export interface SessionRunAccepted {
  session_key: string;
  run_id: string;
  turn_id: string;
  status: "accepted";
  stream_url: string;
}

/** 用户可以在提交主题时附带的检索和阅读限制。空字段不会写入请求。 */
export interface SessionConstraints {
  year_from?: number;
  year_to?: number;
  max_results?: number;
  deep_read_limit?: number;
  excluded_terms?: string[];
  sources?: string[];
}

export interface SessionRunStartPayload {
  content?: string;
  turn_id?: string;
  resume_from_last_checkpoint?: boolean;
  constraints?: SessionConstraints;
}

export type RuntimeDetailContent = string | Record<string, unknown> | null;

export interface SessionRuntimeEvent {
  event: string;
  session_key: string;
  chat_id?: string;
  run_id?: string;
  turn_id?: string;
  timestamp?: string;
  stream_seq?: number;
  event_id?: string;
  role?: "user" | "assistant" | "system";
  kind?: string;
  content?: string;
  delta?: string;
  status?: string;
  run_started_at?: string | null;
  step?: string;
  id?: string;
  parent_id?: string | null;
  type?: string;
  title?: string;
  show_content?: string;
  detail_content?: RuntimeDetailContent;
  created_at?: string | null;
  updated_at?: string | null;
  completed_at?: string | null;
  node_key?: string;
  node_title?: string;
  stage?: string;
  media?: Array<Record<string, unknown>>;
  message?: string;
  artifact?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  keywords?: string[];
  sources?: string[];
  max_results?: number;
  raw_paper_count?: number;
  selected_paper_count?: number;
  artifact_count?: number;
  search_halted?: boolean;
  checkpoint?: Record<string, unknown>;
  recovery_status?: string;
  next_position?: number;
  completed?: number;
  total?: number;
  input_tokens?: number;
  output_tokens?: number;
}

export interface UISessionMessage {
  id: string;
  role: "user" | "assistant" | "system";
  kind: string;
  content: string;
  reasoning: string;
  isStreaming: boolean;
  reasoningStreaming: boolean;
  media: Array<Record<string, unknown>>;
  toolEvents: SessionRuntimeEvent[];
  artifactRefs: Array<Record<string, unknown>>;
  turnId: string | null;
  createdAt: string | null;
}

export interface UIRuntimeTimelineEvent {
  id: string;
  parentId: string | null;
  type: string;
  title: string;
  status: string;
  showContent: string;
  detailContent: RuntimeDetailContent;
  metadata: Record<string, unknown>;
  createdAt: string | null;
  updatedAt: string | null;
  completedAt: string | null;
  children: UIRuntimeTimelineEvent[];
  isCollapsed: boolean;
  resumeAvailable: boolean;
  recoveryStatus: string | null;
  nextPosition: number | null;
  completed: number | null;
  total: number | null;
  inputTokens: number;
  outputTokens: number;
  raw: SessionRuntimeEvent;
}

export interface SessionTimelineSnapshot {
  messages: UISessionMessage[];
  runtimeEvents: UIRuntimeTimelineEvent[];
  activeNodeKey: string | null;
  artifacts: SessionArtifact[];
  isStreaming: boolean;
  runStartedAt: string | null;
  streamError: SessionRuntimeEvent | null;
  status: string;
}
