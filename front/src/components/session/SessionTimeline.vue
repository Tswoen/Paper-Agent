<script setup lang="ts">
import {
  Bot,
  CheckCircle2,
  CircleDotDashed,
  FileStack,
  LoaderCircle,
  SearchCheck,
  Sparkles,
  TriangleAlert,
  UserRound,
} from "lucide-vue-next";

import StatusPill from "../StatusPill.vue";
import { createRandomId } from "../../lib/random-id";
import type {
  SessionArtifact,
  SessionTimelineSnapshot,
  UINodeTimelineEntry,
  UINodeTimelineGroup,
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
  if (status === "completed") {
    return "success";
  }
  if (status === "running") {
    return "warning";
  }
  if (status === "failed") {
    return "danger";
  }
  return "neutral";
}

function nodeIcon(group: UINodeTimelineGroup) {
  if (group.status === "completed") {
    return CheckCircle2;
  }
  if (group.status === "failed") {
    return TriangleAlert;
  }
  return CircleDotDashed;
}

function nodeStatusLabel(status: UINodeTimelineGroup["status"]) {
  if (status === "completed") {
    return "已完成";
  }
  if (status === "failed") {
    return "执行失败";
  }
  return "执行中";
}

function nodeEventTone(entry: UINodeTimelineEntry) {
  if (entry.event === "node_completed") {
    return "success";
  }
  if (entry.event === "node_failed") {
    return "danger";
  }
  if (entry.event === "node_started") {
    return "warning";
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
  return Boolean(snapshot && (snapshot.messages.length || snapshot.nodeGroups.length || snapshot.artifacts.length));
}

/** 中文注释：把恢复进度整理成一句短提示，只告诉用户可以继续，不展示复杂的恢复数据。 */
function resumeHint(group: UINodeTimelineGroup) {
  if (group.completed !== null && group.total !== null) {
    return `已完成 ${group.completed}/${group.total}，可从失败位置继续`;
  }
  if (group.nextPosition !== null) {
    return `将从第 ${group.nextPosition + 1} 项附近继续`;
  }
  return "可从上次失败位置继续";
}
</script>

<template>
  <section class="session-timeline-card">
    <div class="session-timeline-head">
      <div>
        <span class="eyebrow">Live Timeline</span>
        <h2>{{ title || "会话时间线" }}</h2>
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

      <section v-if="snapshot?.nodeGroups.length" class="session-node-groups">
        <div class="session-node-groups-head">
          <span class="eyebrow">Node Timeline</span>
          <p>每个节点的执行过程会放在一起，完成后自动收起，方便只关注正在运行的节点。</p>
        </div>

        <!-- 中文注释：这里用浏览器自带的 details 折叠能力，不额外保存展开状态，结构更简单。 -->
        <details
          v-for="group in snapshot.nodeGroups"
          :key="group.id"
          class="session-node-group"
          :data-status="group.status"
          :open="!group.isCollapsed"
        >
          <summary class="session-node-group-summary">
            <span class="session-node-icon">
              <component :is="nodeIcon(group)" :size="15" />
            </span>
            <span class="session-node-title-block">
              <strong>{{ group.nodeTitle }}</strong>
              <small>{{ group.latestMessage }}</small>
            </span>
            <span class="session-node-meta">
              <StatusPill :tone="statusTone(group.status)" :label="nodeStatusLabel(group.status)" />
              <span>{{ group.entries.length }} 步</span>
              <span>{{ formatTime(group.completedAt ?? group.startedAt) }}</span>
              <button
                v-if="group.resumeAvailable"
                class="button compact session-resume-button"
                type="button"
                @click.stop.prevent="emit('resume')"
              >
                继续执行
              </button>
            </span>
          </summary>

          <p v-if="group.resumeAvailable" class="session-resume-hint">{{ resumeHint(group) }}</p>

          <ol class="session-node-event-list">
            <li
              v-for="entry in group.entries"
              :key="entry.id"
              class="session-node-event"
              :data-tone="nodeEventTone(entry)"
            >
              <div class="session-node-event-mark">{{ entry.label }}</div>
              <div class="session-node-event-body">
                <div class="session-node-event-meta">
                  <strong>{{ entry.stage || "节点更新" }}</strong>
                  <span>{{ formatTime(entry.timestamp) }}</span>
                </div>
                <p>{{ entry.message }}</p>
              </div>
            </li>
          </ol>
        </details>
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
