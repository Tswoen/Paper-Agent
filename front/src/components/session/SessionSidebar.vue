<script setup lang="ts">
import { LoaderCircle, MessageSquarePlus, Trash2 } from "lucide-vue-next";

import StatusPill from "../StatusPill.vue";
import type { SessionSummary } from "../../types/sessions";

defineProps<{
  sessions: SessionSummary[];
  selectedKey: string;
  loading: boolean;
  creating: boolean;
}>();

const emit = defineEmits<{
  create: [];
  select: [sessionKey: string];
  remove: [sessionKey: string];
}>();

function statusTone(status: SessionSummary["status"]) {
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

/** 将更新时间格式化为简洁可读的本地时间。 */
function formatUpdatedAt(value: string) {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}
</script>

<template>
  <section class="session-sidebar-card">
    <div class="session-sidebar-head">
      <div>
        <span class="eyebrow">Session Index</span>
        <h2>会话列表</h2>
      </div>
      <button class="button primary compact" type="button" :disabled="creating" @click="emit('create')">
        <MessageSquarePlus :size="16" />
        新建会话
      </button>
    </div>

    <div v-if="loading" class="session-empty">
      <LoaderCircle class="spinning" :size="18" />
      <span>正在加载会话列表…</span>
    </div>

    <div v-else-if="!sessions.length" class="session-empty">
      <span>还没有会话，先新建一个工作台开始检索。</span>
    </div>

    <div v-else class="session-list">
      <button
        v-for="session in sessions"
        :key="session.key"
        class="session-list-item"
        :class="{ active: session.key === selectedKey }"
        type="button"
        @click="emit('select', session.key)"
      >
        <div class="session-list-main">
          <div class="session-list-row">
            <strong>{{ session.title }}</strong>
            <StatusPill :tone="statusTone(session.status)" :label="session.status" />
          </div>
          <p>{{ session.preview || "等待输入主题" }}</p>
          <div class="session-list-row session-list-meta">
            <span>{{ formatUpdatedAt(session.updated_at) }}</span>
            <button
              class="icon-button ghost"
              type="button"
              aria-label="删除会话"
              @click.stop="emit('remove', session.key)"
            >
              <Trash2 :size="14" />
            </button>
          </div>
        </div>
      </button>
    </div>
  </section>
</template>
