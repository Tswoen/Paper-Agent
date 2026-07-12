<script setup lang="ts">
import {
  Bot,
  FileStack,
  LoaderCircle,
  SearchCheck,
  Sparkles,
  TriangleAlert,
  UserRound,
} from "lucide-vue-next";

import StatusPill from "../StatusPill.vue";
import RuntimeEventTree from "./RuntimeEventTree.vue";
import { createRandomId } from "../../lib/random-id";
import type {
  SessionArtifact,
  SessionTimelineSnapshot,
  UISessionMessage,
} from "../../types/sessions";

defineProps<{
  title: string;
  snapshot: SessionTimelineSnapshot | null;
  loading: boolean;
}>();

const emit = defineEmits<{
  resume: [];
}>();

function messageIcon(message: UISessionMessage) {
  if (message.role === "user") {
    return UserRound;
  }
  if (message.kind === "progress") {
    return SearchCheck;
  }
  if (message.kind === "error") {
    return TriangleAlert;
  }
  return Bot;
}

function statusTone(status: string) {
  if (status === "completed" || status === "success") {
    return "success";
  }
  if (status === "running" || status === "pending") {
    return "warning";
  }
  if (status === "failed" || status === "error") {
    return "danger";
  }
  return "neutral";
}

/** 将产物字节数格式化为更易读的单位。 */
function formatSize(size: number) {
  if (size >= 1024 * 1024) {
    return `${(size / (1024 * 1024)).toFixed(1)} MB`;
  }
  if (size >= 1024) {
    return `${(size / 1024).toFixed(1)} KB`;
  }
  return `${size} B`;
}

/** 使用本地时间渲染消息时间戳。 */
function formatTime(value: string | null) {
  if (!value) {
    return "";
  }
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(value));
}

/** 为 artifact 事件里的松散对象补齐最小展示字段。 */
function artifactLabel(artifact: Record<string, unknown>) {
  return String(artifact.name ?? artifact.path ?? "artifact");
}

function artifactPath(artifact: Record<string, unknown>) {
  return String(artifact.path ?? "");
}

function artifactId(artifact: Record<string, unknown>) {
  return String(artifact.id ?? artifact.artifact_id ?? artifact.path ?? createRandomId("artifact"));
}

function artifactTime(artifact: Record<string, unknown>) {
  return String(artifact.created_at ?? "");
}

function hasTimelineContent(snapshot: SessionTimelineSnapshot | null) {
  return Boolean(snapshot && (snapshot.messages.length || snapshot.runtimeEvents.length || snapshot.artifacts.length));
}
</script>

<template>
  <section class="session-timeline-card">
    <div class="session-timeline-head">
      <div>
        <span class="eyebrow">Execution Flow</span>
        <h2>{{ title || "执行过程" }}</h2>
      </div>
      <StatusPill
        :tone="statusTone(snapshot?.status ?? 'created')"
        :label="snapshot?.status ?? 'created'"
      />
    </div>

    <div v-if="loading" class="session-empty">
      <LoaderCircle class="spinning" :size="18" />
      <span>正在载入线程快照…</span>
    </div>

    <div v-else-if="!hasTimelineContent(snapshot)" class="session-empty session-empty-large">
      <Sparkles :size="18" />
      <span>输入一个论文主题，工作流会在这里实时展开。</span>
    </div>

    <div v-else class="session-timeline-scroll">
      <div v-if="snapshot?.messages.length" class="session-timeline-stream">
        <article
          v-for="message in snapshot.messages"
          :key="message.id"
          class="session-message-card"
          :data-role="message.role"
          :data-kind="message.kind"
        >
          <div class="session-message-icon">
            <component :is="messageIcon(message)" :size="16" />
          </div>

          <div class="session-message-body">
            <div class="session-message-meta">
              <strong>
                {{
                  message.role === "user"
                    ? "用户主题"
                    : message.kind === "progress"
                      ? "执行进度"
                      : message.kind === "error"
                        ? "执行错误"
                        : "系统助手"
                }}
              </strong>
              <span>{{ formatTime(message.createdAt) }}</span>
            </div>

            <pre class="session-message-content">{{ message.content }}</pre>

            <details
              v-if="message.reasoning"
              class="session-reasoning-card"
              :open="message.reasoningStreaming"
            >
              <summary>查看 reasoning 过程</summary>
              <pre>{{ message.reasoning }}</pre>
            </details>

            <div v-if="message.isStreaming" class="session-streaming-hint">
              <LoaderCircle class="spinning" :size="14" />
              <span>正在生成中</span>
            </div>

            <div v-if="message.artifactRefs.length" class="session-artifact-inline">
              <span class="eyebrow">Artifacts</span>
              <div class="session-artifact-grid">
                <article
                  v-for="artifact in message.artifactRefs"
                  :key="artifactId(artifact)"
                  class="session-artifact-item"
                >
                  <div class="session-artifact-item-head">
                    <FileStack :size="14" />
                    <strong>{{ artifactLabel(artifact) }}</strong>
                  </div>
                  <span>{{ artifactPath(artifact) }}</span>
                  <span>{{ formatTime(artifactTime(artifact)) }}</span>
                </article>
              </div>
            </div>
          </div>
        </article>
      </div>

      <section v-if="snapshot?.runtimeEvents.length" class="runtime-event-section">
        <div class="runtime-event-section-head">
          <h3>执行过程</h3>
          <p>同一个事件会在原位置更新，子事件按照后端传来的 parent_id 自动缩进展示。</p>
        </div>
        <RuntimeEventTree :events="snapshot.runtimeEvents" @resume="emit('resume')" />
      </section>

      <section v-if="snapshot?.artifacts.length" class="session-artifact-panel">
        <div class="session-artifact-panel-head">
          <div>
            <span class="eyebrow">Artifacts</span>
            <h3>本会话产物</h3>
          </div>
        </div>
        <div class="session-artifact-grid">
          <article
            v-for="artifact in snapshot.artifacts"
            :key="artifact.id"
            class="session-artifact-item"
          >
            <div class="session-artifact-item-head">
              <FileStack :size="14" />
              <strong>{{ artifact.name }}</strong>
            </div>
            <span>{{ artifact.path }}</span>
            <span>{{ formatSize(artifact.size) }} · {{ formatTime(artifact.created_at) }}</span>
          </article>
        </div>
      </section>
    </div>
  </section>
</template>
