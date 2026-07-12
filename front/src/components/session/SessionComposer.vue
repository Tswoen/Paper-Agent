<script setup lang="ts">
import { computed } from "vue";
import { LoaderCircle, SendHorizonal } from "lucide-vue-next";

const props = withDefaults(
  defineProps<{
    modelValue: string;
    running: boolean;
    sending: boolean;
    statusText: string;
    variant?: "default" | "welcome";
    heading?: string;
    helperText?: string;
    placeholder?: string;
    rows?: number;
  }>(),
  {
    variant: "default",
    heading: "今天想探索什么研究方向？",
    helperText: "",
    placeholder: "输入研究主题或调研任务...",
    rows: 1,
  },
);

const emit = defineEmits<{
  "update:modelValue": [value: string];
  submit: [];
}>();

/**
 * 中文注释：这个输入框有两种样子。
 * 默认样子放在时间线底部，欢迎样子放在页面中间，但它们用的是同一套发送逻辑，避免以后维护两份重复代码。
 */
const isWelcomeVariant = computed(() => props.variant === "welcome");

/** 中文注释：统一处理 Ctrl/Command + Enter 快捷发送，方便键盘操作时不用移动鼠标。 */
function onKeydown(event: KeyboardEvent) {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    emit("submit");
  }
}
</script>

<template>
  <section class="session-composer-card" :data-variant="props.variant" :aria-busy="running || sending">
    <div v-if="isWelcomeVariant" class="session-composer-intro">
      <span class="eyebrow">New Research Session</span>
      <h2>{{ props.heading }}</h2>
    </div>

    <div class="session-composer-input-row">
      <textarea
        class="session-composer-textarea"
        :value="modelValue"
        :rows="props.rows"
        :placeholder="props.placeholder"
        :disabled="running || sending"
        @input="emit('update:modelValue', ($event.target as HTMLTextAreaElement).value)"
        @keydown="onKeydown"
      />

      <button
        class="button primary compact session-composer-send"
        type="button"
        :disabled="running || sending || !modelValue.trim()"
        :title="running || sending ? props.statusText : '发送，Ctrl / ⌘ + Enter'"
        @click="emit('submit')"
      >
        <LoaderCircle v-if="running || sending" class="spinning" :size="16" />
        <SendHorizonal v-else :size="16" />
      </button>
    </div>

    <p v-if="isWelcomeVariant && props.helperText" class="session-composer-helper">
      {{ props.helperText }}
    </p>
  </section>
</template>
