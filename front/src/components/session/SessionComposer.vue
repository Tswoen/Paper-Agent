<script setup lang="ts">
import { computed, nextTick, ref } from "vue";
import {
  ChevronDown,
  FileText,
  Filter,
  Lightbulb,
  LoaderCircle,
  SendHorizonal,
  Sparkles,
} from "lucide-vue-next";

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

interface TopicSuggestion {
  label: string;
  prompt: string;
}

const isWelcomeVariant = computed(() => props.variant === "welcome");
const constraintsExpanded = ref(false);
const textareaElement = ref<HTMLTextAreaElement | null>(null);

const quickTopics: TopicSuggestion[] = [
  { label: "多模态大模型的最新进展", prompt: "调研多模态大模型的最新进展、代表性工作与未来趋势" },
  { label: "检索增强生成(RAG)应用研究", prompt: "调研检索增强生成(RAG)在实际应用中的研究进展与关键挑战" },
  { label: "智能体协作机制综述", prompt: "梳理多智能体协作机制的研究现状、常见架构与评估方法" },
  { label: "推荐系统冷启动问题分析", prompt: "分析推荐系统冷启动问题的主流解决方案、适用条件与研究机会" },
];
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

/**
 * 中文说明：标题中的“研究方向”单独使用强调色，既贴近启动页的视觉重点，
 * 也不需要把标题文案拆散写在父页面中，之后调整标题时仍只改一个地方。
 */
const headingParts = computed(() => {
  const emphasis = "研究方向";
  const emphasisIndex = props.heading.indexOf(emphasis);
  if (emphasisIndex < 0) {
    return { before: props.heading, emphasis: "", after: "" };
  }
  return {
    before: props.heading.slice(0, emphasisIndex),
    emphasis,
    after: props.heading.slice(emphasisIndex + emphasis.length),
  };
});

/** 只更新一个检索设置，避免直接修改父页面传入的对象。 */
function updateConstraint<K extends keyof SessionConstraints>(key: K, value: SessionConstraints[K] | undefined) {
  const next = { ...props.constraints };
  if (value === undefined || (Array.isArray(value) && value.length === 0)) {
    delete next[key];
  } else {
    next[key] = value;
  }
  emit("update:constraints", next);
}

/** 数字输入框为空时清除该项设置，否则只提交有效的数字。 */
function updateNumberConstraint(key: "year_from" | "year_to" | "max_results" | "deep_read_limit", event: Event) {
  const raw = (event.target as HTMLInputElement).value.trim();
  const value = raw ? Number(raw) : undefined;
  updateConstraint(key, value !== undefined && Number.isFinite(value) ? value : undefined);
}

/** 排除词支持中英文逗号和换行，提交前统一整理成关键词列表。 */
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

/**
 * 中文说明：快捷主题只帮助用户快速补全研究任务，最终仍由用户在输入框中确认和编辑，
 * 不会自动发送，避免误触后直接启动一次调研。
 */
async function applyPrompt(prompt: string) {
  emit("update:modelValue", prompt);
  await nextTick();
  textareaElement.value?.focus();
}

/** 统一处理 Ctrl/Command + Enter，方便键盘输入后直接开始任务。 */
function onKeydown(event: KeyboardEvent) {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    emit("submit");
  }
}
</script>

<template>
  <section class="session-composer-card" :data-variant="props.variant" :aria-busy="running || sending">
    <template v-if="isWelcomeVariant">
      <header class="session-welcome-heading">
        <span class="session-welcome-badge"><Sparkles :size="15" />新建研究会话</span>
        <h2>
          {{ headingParts.before }}<strong v-if="headingParts.emphasis">{{ headingParts.emphasis }}</strong>{{ headingParts.after }}
        </h2>
        <p v-if="props.helperText">{{ props.helperText }}</p>
      </header>

      <div class="session-welcome-prompt-card">
        <div class="session-welcome-input-row">
          <FileText class="session-welcome-input-icon" :size="24" aria-hidden="true" />
          <textarea
            ref="textareaElement"
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
            :title="running || sending ? props.statusText : '开始调研'
            "
            aria-label="开始调研"
            @click="emit('submit')"
          >
            <LoaderCircle v-if="running || sending" class="spinning" :size="22" />
            <SendHorizonal v-else :size="22" />
          </button>
        </div>

        <div v-if="constraintsExpanded" class="session-composer-constraints-panel session-welcome-constraints-panel">
          <div class="session-composer-constraint-grid">
            <label class="session-composer-constraint-field">
              <span>最早年份</span>
              <input class="field" type="number" min="1900" max="2100" placeholder="不限" :value="constraints.year_from ?? ''" :disabled="running || sending" @input="updateNumberConstraint('year_from', $event)" />
            </label>
            <label class="session-composer-constraint-field">
              <span>最晚年份</span>
              <input class="field" type="number" min="1900" max="2100" placeholder="不限" :value="constraints.year_to ?? ''" :disabled="running || sending" @input="updateNumberConstraint('year_to', $event)" />
            </label>
            <label class="session-composer-constraint-field">
              <span>检索结果数</span>
              <input class="field" type="number" min="1" max="200" placeholder="默认" :value="constraints.max_results ?? ''" :disabled="running || sending" @input="updateNumberConstraint('max_results', $event)" />
            </label>
            <label class="session-composer-constraint-field">
              <span>深度阅读数</span>
              <input class="field" type="number" min="0" max="200" placeholder="默认全部" :value="constraints.deep_read_limit ?? ''" :disabled="running || sending" @input="updateNumberConstraint('deep_read_limit', $event)" />
            </label>
          </div>

          <label class="session-composer-constraint-field session-composer-constraint-wide">
            <span>排除关键词</span>
            <input class="field" type="text" placeholder="用逗号分隔，例如：survey, medical" :value="excludedTermsText" :disabled="running || sending" @input="updateExcludedTerms" />
          </label>

          <fieldset class="session-composer-source-field">
            <legend>检索来源</legend>
            <label v-for="option in sourceOptions" :key="option.value" class="session-composer-source-option">
              <input type="checkbox" :checked="constraints.sources?.includes(option.value)" :disabled="running || sending" @change="toggleSource(option.value, $event)" />
              <span>{{ option.label }}</span>
            </label>
          </fieldset>
        </div>

        <div class="session-welcome-quick-start">
          <span class="session-welcome-quick-label"><Sparkles :size="17" />快捷开始</span>
          <button v-for="topic in quickTopics" :key="topic.label" class="session-welcome-topic-button" type="button" @click="applyPrompt(topic.prompt)">
            <FileText :size="14" />
            <span>{{ topic.label }}</span>
          </button>
          <button
            class="session-welcome-settings-button"
            type="button"
            title="检索设置"
            aria-label="检索设置"
            :aria-expanded="constraintsExpanded"
            @click="constraintsExpanded = !constraintsExpanded"
          >
            <Filter :size="16" />
          </button>
        </div>
      </div>

      <p class="session-welcome-inspiration"><Lightbulb :size="17" />不知道写什么？试试这些热门主题</p>
    </template>

    <template v-else>
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
          :title="running || sending ? props.statusText : '发送，Ctrl / Command + Enter'"
          @click="emit('submit')"
        >
          <LoaderCircle v-if="running || sending" class="spinning" :size="16" />
          <SendHorizonal v-else :size="16" />
        </button>
      </div>

      <div class="session-composer-constraints">
        <button class="session-composer-constraints-toggle" type="button" :aria-expanded="constraintsExpanded" @click="constraintsExpanded = !constraintsExpanded">
          <Filter :size="15" />
          <span>检索与阅读约束</span>
          <span v-if="hasConstraints" class="session-composer-constraints-badge">已设置</span>
          <ChevronDown :size="15" :class="{ rotated: constraintsExpanded }" />
        </button>

        <div v-if="constraintsExpanded" class="session-composer-constraints-panel">
          <div class="session-composer-constraint-grid">
            <label class="session-composer-constraint-field">
              <span>最早年份</span>
              <input class="field" type="number" min="1900" max="2100" placeholder="不限" :value="constraints.year_from ?? ''" :disabled="running || sending" @input="updateNumberConstraint('year_from', $event)" />
            </label>
            <label class="session-composer-constraint-field">
              <span>最晚年份</span>
              <input class="field" type="number" min="1900" max="2100" placeholder="不限" :value="constraints.year_to ?? ''" :disabled="running || sending" @input="updateNumberConstraint('year_to', $event)" />
            </label>
            <label class="session-composer-constraint-field">
              <span>检索结果数</span>
              <input class="field" type="number" min="1" max="200" placeholder="默认" :value="constraints.max_results ?? ''" :disabled="running || sending" @input="updateNumberConstraint('max_results', $event)" />
            </label>
            <label class="session-composer-constraint-field">
              <span>深度阅读数</span>
              <input class="field" type="number" min="0" max="200" placeholder="默认全部" :value="constraints.deep_read_limit ?? ''" :disabled="running || sending" @input="updateNumberConstraint('deep_read_limit', $event)" />
            </label>
          </div>

          <label class="session-composer-constraint-field session-composer-constraint-wide">
            <span>排除关键词</span>
            <input class="field" type="text" placeholder="用逗号分隔，例如：survey, medical" :value="excludedTermsText" :disabled="running || sending" @input="updateExcludedTerms" />
          </label>

          <fieldset class="session-composer-source-field">
            <legend>检索来源</legend>
            <label v-for="option in sourceOptions" :key="option.value" class="session-composer-source-option">
              <input type="checkbox" :checked="constraints.sources?.includes(option.value)" :disabled="running || sending" @change="toggleSource(option.value, $event)" />
              <span>{{ option.label }}</span>
            </label>
            <small>未选择时使用全部可用来源</small>
          </fieldset>
        </div>
      </div>
    </template>
  </section>
</template>
