<template>
  <div class="app-container">
    <h1>智能调研报告生成</h1>
    <div class="input-section">
      <textarea 
        v-model="userInput" 
        placeholder="请输入您想调研的问题..."
        rows="5"
      ></textarea>  
      <button 
        @click="submitRequest" 
        :disabled="isSubmitting"
      >
        {{ isSubmitting ? '处理中...' : '提交' }}
      </button>
    </div>

    <div class="progress-section" v-if="steps.length > 0">
      <div 
        v-for="(step, index) in steps" 
        :key="index" 
        class="step-card"
        :class="{ 'error': step.error, 'show': step.show }"
      >
        <!-- 假设StepCard组件已经定义，并接受step作为prop -->
        <StepCard :step="step" />
      </div>
    </div>

    <div class="report-section" v-if="reportContent">
      <h2>最终报告</h2>
      <textarea 
        readonly 
        v-model="reportContent" 
        rows="20"
      ></textarea>
      <button @click="copyReport">复制报告</button>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick, onBeforeUnmount } from 'vue'

const userInput = ref('')
const isSubmitting = ref(false)
const steps = ref([])
const reportContent = ref('')
const eventSource = ref(null)


// deepseek修改后
const submitRequest = () => {
  if (!userInput.value.trim()) return
  
  isSubmitting.value = true
  steps.value = []
  reportContent.value = ''
  
  // 初始化SSE连接
  eventSource.value = new EventSource(`/api/research?query=${encodeURIComponent(userInput.value)}`)

  const addStep = (name, content, isError = false) => {
    const stepElement = {
      name,
      content,
      error: isError,
      timestamp: new Date().toISOString(),
      show: false
    };
    steps.value.push(stepElement);
    nextTick(() => {
      stepElement.show = true;
      autoScroll();
    });
  };

const autoScroll = () => {
  nextTick(() => {
    const progressSection = document.querySelector('.progress-section')
    if (progressSection) {
      progressSection.scrollTop = progressSection.scrollHeight
    }
  })
}

  const finishProcessing = () => {
    isSubmitting.value = false;
    eventSource.value?.close();
  };

  // 状态处理器工厂函数
  const createStateHandlers = () => ({
    processing: ({ step, data }) => {
      const stepProcessors = {
        searching: () => addStep('搜索', data?.content || '正在搜索论文...'),
        reading: () => addStep('阅读', data?.content || '解析论文内容中...'),
        analyzing: () => addStep('分析', data?.content || '生成分析报告中...'),
        writing: () => addStep('撰写', data?.content || '整合最终报告中...')
      };
      stepProcessors[step]?.();
    },
    completed: ({ data }) => {
      reportContent.value = data.report_markdown;
      finishProcessing();
    },
    error: ({ error }) => {
      addStep('错误', error?.message || '发生未知错误', true);
      finishProcessing();
    }
  });

  // 统一状态处理入口
  const handleExecutionState = (stateData) => {
    const handlers = createStateHandlers();
    const handler = handlers[stateData.state];
    if (handler) {
      handler(stateData);
    } else {
      console.warn('未知状态:', stateData.state);
    }
  };

  eventSource.value.onmessage = (event) => {
    try {
      const stateData = JSON.parse(event.data);
      handleExecutionState(stateData);
    } catch (error) {
      console.error('Error processing event:', error);
    }
  };
    
  eventSource.value.onerror = () => {
    addStep('错误', '连接服务器失败', true);
    finishProcessing();
  };
}


const copyReport = () => {
  if (!reportContent.value) return
  navigator.clipboard.writeText(reportContent.value)
}

onBeforeUnmount(() => {
  if (eventSource.value) {
    eventSource.value.close()
  }
})
</script>

<style>
.app-container {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
}

.input-section {
  margin-bottom: 20px;
}

.input-section textarea {
  width: 100%;
  padding: 10px;
  margin-bottom: 10px;
}

.progress-section {
  margin-bottom: 20px;
}

.step-card {
  transition: all 0.3s ease;
  opacity: 0;
  transform: translateY(20px);
  border: 1px solid #ddd;
  border-radius: 5px;
  padding: 15px;
  margin-bottom: 15px;
}

.step-card.show {
  opacity: 1;
  transform: translateY(0);
}

.step-card[data-type='AGENT_LOG'] {
  border-left: 4px solid #1890ff;
  background: #f0f9ff;
}

.step-card.error {
  border-color: #ff4d4f;
  color: #ff4d4f;
}

.step-card pre {
  white-space: pre-wrap;
  font-family: monospace;
  margin: 0;
}

.report-section textarea {
  width: 100%;
  padding: 10px;
}
</style>