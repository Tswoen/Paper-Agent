<script setup lang="ts">
import { LoaderCircle, PauseCircle, SendHorizonal } from "lucide-vue-next";

defineProps<{
  modelValue: string;
  running: boolean;
  sending: boolean;
  statusText: string;
}>();

const emit = defineEmits<{
  "update:modelValue": [value: string];
  submit: [];
}>();

/** 统一处理 Ctrl/Command + Enter 快捷发送。 */
function onKeydown(event: KeyboardEvent) {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    emit("submit");
  }
}
</script>

<template>
  <section class="session-composer-card">
    <div class="session-composer-head">
      <div>
        <span class="eyebrow">Topic Workspace</span>
        <h2>输入主题并启动工作流</h2>
      </div>
      <div class="session-composer-status">
        <LoaderCircle v-if="running || sending" class="spinning" :size="14" />
        <span>{{ statusText }}</span>
      </div>
    </div>

    <textarea
      class="session-composer-textarea"
      :value="modelValue"
      rows="5"
      placeholder="输入论文主题、研究问题或综述方向，系统会启动对应工作流"
      :disabled="running || sending"
      @input="emit('update:modelValue', ($event.target as HTMLTextAreaElement).value)"
      @keydown="onKeydown"
    />

    <div class="session-composer-actions">
      <button class="button secondary" type="button" disabled>
        <PauseCircle :size="16" />
        停止运行
      </button>
      <button class="button primary" type="button" :disabled="running || sending || !modelValue.trim()" @click="emit('submit')">
        <SendHorizonal :size="16" />
        启动工作流
      </button>
    </div>
  </section>
</template>
