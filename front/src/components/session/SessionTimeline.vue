<script setup lang="ts">
import { LoaderCircle, Sparkles } from "lucide-vue-next";

import StatusPill from "../StatusPill.vue";
import RuntimeEventTree from "./RuntimeEventTree.vue";
import type { SessionTimelineSnapshot } from "../../types/sessions";

defineProps<{
  title: string;
  snapshot: SessionTimelineSnapshot | null;
  loading: boolean;
}>();

const emit = defineEmits<{
  resume: [];
}>();

function statusTone(status: string) {
  if (status === "completed" || status === "success") {
    return "success";
  }
  if (status === "running" || status === "pending" || status === "cancel_requested") {
    return "warning";
  }
  if (status === "cancelled") {
    return "neutral";
  }
  if (status === "failed" || status === "error") {
    return "danger";
  }
  return "neutral";
}

function hasTimelineContent(snapshot: SessionTimelineSnapshot | null) {
  // 中文注释：这个组件现在只展示“执行过程”这棵树。
  // 消息内容和产物文件仍然会被前端保存，但它们不再显示在这里，所以不能再用它们判断页面是否有可见内容。
  return Boolean(snapshot?.runtimeEvents.length);
}
</script>

<template>
  <section class="session-timeline-card">
    <div class="session-timeline-head">
      <div>
        <span class="eyebrow">Execution Flow</span>
        <h2>执行流程</h2>
        <p class="session-timeline-subtitle">任务已顺利完成，以下是各步骤的执行详情</p>
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
      <section v-if="snapshot?.runtimeEvents.length" class="runtime-event-section">
        <div class="runtime-event-section-head">
          <h3>执行过程</h3>
          <p>同一个事件会在原位置更新，子事件按照后端传来的 parent_id 自动缩进展示。</p>
        </div>
        <RuntimeEventTree :events="snapshot.runtimeEvents" @resume="emit('resume')" />
      </section>
    </div>
  </section>
</template>
