<script setup lang="ts">
import { computed, ref } from "vue";
import { ChevronDown, Filter, LoaderCircle, SendHorizonal } from "lucide-vue-next";

import type { SessionConstraints } from "../../types/sessions";

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
    constraints: SessionConstraints;
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
  "update:constraints": [value: SessionConstraints];
  submit: [];
}>();

/**
 * 中文注释：这个输入框有两种样子。
 * 默认样子放在时间线底部，欢迎样子放在页面中间，但它们用的是同一套发送逻辑，避免以后维护两份重复代码。
 */
const isWelcomeVariant = computed(() => props.variant === "welcome");
const constraintsExpanded = ref(false);
const hasConstraints = computed(() => {
  const value = props.constraints;
  return Boolean(
    value.year_from !== undefined ||
      value.year_to !== undefined ||
      value.max_results !== undefined ||
      value.deep_read_limit !== undefined ||
      value.excluded_terms?.length ||
      value.sources?.length,
  );
});

const excludedTermsText = computed(() => (props.constraints.excluded_terms ?? []).join(", "));
const sourceOptions = [
  { value: "openalex", label: "OpenAlex" },
  { value: "arxiv", label: "arXiv" },
  { value: "semantic_scholar", label: "Semantic Scholar" },
];

/** 只更新一个约束字段，避免直接修改父组件传入的对象。 */
function updateConstraint<K extends keyof SessionConstraints>(key: K, value: SessionConstraints[K] | undefined) {
  const next = { ...props.constraints };
  if (value === undefined || (Array.isArray(value) && value.length === 0)) {
    delete next[key];
  } else {
    next[key] = value;
  }
  emit("update:constraints", next);
}

/** 数字输入框为空时删除约束，否则提交一个真正的数字。 */
function updateNumberConstraint(key: "year_from" | "year_to" | "max_results" | "deep_read_limit", event: Event) {
  const raw = (event.target as HTMLInputElement).value.trim();
  const value = raw ? Number(raw) : undefined;
  updateConstraint(key, value !== undefined && Number.isFinite(value) ? value : undefined);
}

/** 排除词支持中英文逗号和换行，提交时统一成字符串数组。 */
function updateExcludedTerms(event: Event) {
  const value = (event.target as HTMLInputElement).value;
  const terms = value
    .split(/[,，\n]/)
    .map((term) => term.trim())
    .filter(Boolean);
  updateConstraint("excluded_terms", terms);
}

function toggleSource(source: string, event: Event) {
  const checked = (event.target as HTMLInputElement).checked;
  const sources = new Set(props.constraints.sources ?? []);
  if (checked) {
    sources.add(source);
  } else {
    sources.delete(source);
  }
  updateConstraint("sources", Array.from(sources));
}

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

    <div class="session-composer-constraints">
      <button
        class="session-composer-constraints-toggle"
        type="button"
        :aria-expanded="constraintsExpanded"
        @click="constraintsExpanded = !constraintsExpanded"
      >
        <Filter :size="15" />
        <span>检索与阅读约束</span>
        <span v-if="hasConstraints" class="session-composer-constraints-badge">已设置</span>
        <ChevronDown :size="15" :class="{ rotated: constraintsExpanded }" />
      </button>

      <div v-if="constraintsExpanded" class="session-composer-constraints-panel">
        <div class="session-composer-constraint-grid">
          <label class="session-composer-constraint-field">
            <span>最早年份</span>
            <input
              class="field"
              type="number"
              min="1900"
              max="2100"
              placeholder="不限"
              :value="constraints.year_from ?? ''"
              :disabled="running || sending"
              @input="updateNumberConstraint('year_from', $event)"
            />
          </label>
          <label class="session-composer-constraint-field">
            <span>最晚年份</span>
            <input
              class="field"
              type="number"
              min="1900"
              max="2100"
              placeholder="不限"
              :value="constraints.year_to ?? ''"
              :disabled="running || sending"
              @input="updateNumberConstraint('year_to', $event)"
            />
          </label>
          <label class="session-composer-constraint-field">
            <span>检索结果数</span>
            <input
              class="field"
              type="number"
              min="1"
              max="200"
              placeholder="默认"
              :value="constraints.max_results ?? ''"
              :disabled="running || sending"
              @input="updateNumberConstraint('max_results', $event)"
            />
          </label>
          <label class="session-composer-constraint-field">
            <span>深度阅读数</span>
            <input
              class="field"
              type="number"
              min="0"
              max="200"
              placeholder="默认全部"
              :value="constraints.deep_read_limit ?? ''"
              :disabled="running || sending"
              @input="updateNumberConstraint('deep_read_limit', $event)"
            />
          </label>
        </div>

        <label class="session-composer-constraint-field session-composer-constraint-wide">
          <span>排除关键词</span>
          <input
            class="field"
            type="text"
            placeholder="用逗号分隔，例如：survey, medical"
            :value="excludedTermsText"
            :disabled="running || sending"
            @input="updateExcludedTerms"
          />
        </label>

        <fieldset class="session-composer-source-field">
          <legend>检索来源</legend>
          <label v-for="option in sourceOptions" :key="option.value" class="session-composer-source-option">
            <input
              type="checkbox"
              :checked="constraints.sources?.includes(option.value)"
              :disabled="running || sending"
              @change="toggleSource(option.value, $event)"
            />
            <span>{{ option.label }}</span>
          </label>
          <small>未选择时使用全部可用来源</small>
        </fieldset>
      </div>
    </div>

    <p v-if="isWelcomeVariant && props.helperText" class="session-composer-helper">
      {{ props.helperText }}
    </p>
  </section>
</template>
