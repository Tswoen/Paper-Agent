<script setup lang="ts">
import {
  CheckCircle2,
  ChevronRight,
  CircleDot,
  LoaderCircle,
  TriangleAlert,
} from "lucide-vue-next";

import StatusPill from "../StatusPill.vue";
import type { UIRuntimeTimelineEvent } from "../../types/sessions";

defineOptions({
  name: "RuntimeEventTree",
});

defineProps<{
  events: UIRuntimeTimelineEvent[];
  depth?: number;
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

function statusLabel(status: string) {
  if (status === "running") {
    return "处理中";
  }
  if (status === "cancel_requested") {
    return "正在停止";
  }
  if (status === "cancelled") {
    return "已停止";
  }
  if (status === "completed") {
    return "完成";
  }
  if (status === "failed") {
    return "失败";
  }
  if (status === "pending") {
    return "等待中";
  }
  if (status === "skipped") {
    return "已跳过";
  }
  return status || "未知";
}

function statusIcon(status: string) {
  if (status === "completed" || status === "success") {
    return CheckCircle2;
  }
  if (status === "failed" || status === "error") {
    return TriangleAlert;
  }
  if (status === "running" || status === "pending" || status === "cancel_requested") {
    return LoaderCircle;
  }
  return CircleDot;
}

/** 中文注释：这里统一选一个最能代表事件当前状态的时间，避免每行出现多个时间把界面挤乱。 */
function eventTime(event: UIRuntimeTimelineEvent) {
  return event.completedAt ?? event.updatedAt ?? event.createdAt;
}

/** 使用本地时间渲染事件时间戳。 */
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

/** 中文注释：detailContent 可以是字符串，也可以是对象；对象格式化后更方便用户展开查看。 */
function formatDetailContent(value: UIRuntimeTimelineEvent["detailContent"]) {
  if (!value) {
    return "";
  }
  if (typeof value === "string") {
    return value;
  }
  return JSON.stringify(value, null, 2);
}

function hasDetail(event: UIRuntimeTimelineEvent) {
  if (!event.detailContent) {
    return false;
  }
  if (typeof event.detailContent === "string") {
    return Boolean(event.detailContent.trim());
  }
  return Object.keys(event.detailContent).length > 0;
}

/** 中文注释：把恢复进度整理成一句短提示，只告诉用户可以继续，不展示复杂的恢复数据。 */
function resumeHint(event: UIRuntimeTimelineEvent) {
  if (event.completed !== null && event.total !== null) {
    return `已完成 ${event.completed}/${event.total}，可从失败位置继续`;
  }
  if (event.nextPosition !== null) {
    return `将从第 ${event.nextPosition + 1} 项附近继续`;
  }
  return "可从上次失败位置继续";
}
</script>

<template>
  <ol class="runtime-event-tree" :style="{ '--event-depth': depth ?? 0 }">
    <li
      v-for="event in events"
      :key="event.id"
      class="runtime-event-item"
      :data-status="event.status"
    >
      <details
        v-if="event.children.length"
        class="runtime-event-branch"
        :open="!event.isCollapsed"
      >
        <summary class="runtime-event-row">
          <span class="runtime-event-toggle">
            <ChevronRight :size="14" />
          </span>
          <span class="runtime-event-dot">
            <component
              :is="statusIcon(event.status)"
              :size="14"
              :class="{ spinning: event.status === 'running' }"
            />
          </span>
          <span class="runtime-event-copy">
            <span class="runtime-event-title-line">
              <strong>{{ event.title }}</strong>
              <StatusPill :tone="statusTone(event.status)" :label="statusLabel(event.status)" />
            </span>
            <span class="runtime-event-show">{{ event.showContent }}</span>
          </span>
          <span class="runtime-event-meta">
            <time class="runtime-event-time">{{ formatTime(eventTime(event)) }}</time>
            <span class="runtime-event-tokens">输入 {{ event.inputTokens }} · 输出 {{ event.outputTokens }}</span>
          </span>
        </summary>

        <details v-if="hasDetail(event)" class="runtime-event-detail">
          <summary>查看详情</summary>
          <pre>{{ formatDetailContent(event.detailContent) }}</pre>
        </details>

        <p v-if="event.resumeAvailable" class="runtime-event-resume-hint">{{ resumeHint(event) }}</p>
        <button
          v-if="event.resumeAvailable"
          class="button compact runtime-event-resume-button"
          type="button"
          @click.stop.prevent="emit('resume')"
        >
          继续执行
        </button>

        <RuntimeEventTree
          class="runtime-event-children"
          :events="event.children"
          :depth="(depth ?? 0) + 1"
          @resume="emit('resume')"
        />
      </details>

      <div v-else class="runtime-event-row runtime-event-row-leaf">
        <span class="runtime-event-toggle" aria-hidden="true"></span>
        <span class="runtime-event-dot">
          <component
            :is="statusIcon(event.status)"
            :size="14"
            :class="{ spinning: event.status === 'running' }"
          />
        </span>
        <span class="runtime-event-copy">
          <span class="runtime-event-title-line">
            <strong>{{ event.title }}</strong>
            <StatusPill :tone="statusTone(event.status)" :label="statusLabel(event.status)" />
          </span>
          <span class="runtime-event-show">{{ event.showContent }}</span>
          <details v-if="hasDetail(event)" class="runtime-event-detail">
            <summary>查看详情</summary>
            <pre>{{ formatDetailContent(event.detailContent) }}</pre>
          </details>
          <p v-if="event.resumeAvailable" class="runtime-event-resume-hint">{{ resumeHint(event) }}</p>
          <button
            v-if="event.resumeAvailable"
            class="button compact runtime-event-resume-button"
            type="button"
            @click.stop.prevent="emit('resume')"
          >
            继续执行
          </button>
        </span>
        <span class="runtime-event-meta">
          <time class="runtime-event-time">{{ formatTime(eventTime(event)) }}</time>
          <span class="runtime-event-tokens">输入 {{ event.inputTokens }} · 输出 {{ event.outputTokens }}</span>
        </span>
      </div>
    </li>
  </ol>
</template>
