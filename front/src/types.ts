export type RuntimeCapabilities = Record<string, boolean | string | number | null | undefined>;

export interface BootstrapPayload {
  expires_in: number;
  api_base: string;
  runtime_surface?: string;
  runtime_capabilities?: RuntimeCapabilities;
}

export interface SessionSummary {
  key: string;
  title: string;
  created_at?: string;
  updated_at?: string;
  preview?: string;
  run_started_at?: string | null;
  workspace_scope?: Record<string, unknown> | null;
}

export interface ThreadMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  kind?: string;
  is_streaming?: boolean;
  reasoning?: string;
  reasoning_streaming?: boolean;
  tool_events?: Array<Record<string, unknown>>;
  file_edits?: Array<Record<string, unknown>>;
  media?: Array<Record<string, unknown>>;
  turn_id?: string | null;
  turn_phase?: string | null;
  turn_seq?: number | null;
  created_at?: string;
}

export interface SettingsAgent {
  name: string;
  label?: string;
  provider: string;
  model?: string;
  model_name?: string;
  temperature?: number | null;
  reasoning_effort?: "none" | "low" | "medium" | "high" | string;
  max_tokens?: number | null;
  context_window_tokens?: number | null;
}

export interface SettingsProvider {
  name: string;
  label?: string;
  configured?: boolean;
  auth_type?: string;
  api_key_required?: boolean;
  api_key_hint?: string | null;
  api_key_env?: string | null;
  api_base?: string | null;
  default_api_base?: string | null;
  model_selectable?: boolean;
  provider_type?: string;
  backend?: string;
  oauth_login_supported?: boolean;
}

export interface ProviderType {
  name: string;
  label?: string;
  backend?: string;
  default_api_base?: string | null;
  api_key_required?: boolean;
}

export interface SettingsPayload {
  agent?: SettingsAgent;
  active_agent?: string;
  agents: SettingsAgent[];
  providers: SettingsProvider[];
  provider_types: ProviderType[];
}

export interface SessionListPayload {
  sessions: SessionSummary[];
}

export interface SessionCreatePayload {
  session: SessionSummary;
}

export interface ThreadPayload {
  key: string;
  messages: ThreadMessage[];
  workspace_scope?: Record<string, unknown> | null;
  has_pending_tool_calls?: boolean;
}

export interface BackendEvent {
  event: string;
  role?: "user" | "assistant" | "system";
  content?: string;
  delta?: string;
  turn_id?: string;
  id?: string;
  media?: Array<Record<string, unknown>>;
  [key: string]: unknown;
}

export interface MessageSubmitPayload {
  session_key: string;
  turn_id: string;
  events: BackendEvent[];
  thread: ThreadPayload;
}
