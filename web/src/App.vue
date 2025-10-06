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

    <div class="progress-section" v-if="steps.length > 0" id="steps-container">
      <div 
        v-for="(step, index) in steps" 
        :key="index" 
        class="step-card"
        :class="{ 'error': step.isError, 'processing': step.isProcessing, 'show': step.show }"
      >
        <div class="step-header">
          <div class="step-title">
            {{ step.title }}
            <!-- 加载动画：当 isProcessing 为 true 时显示旋转圈圈 -->
            <span class="loading-spinner" v-if="step.isProcessing">⟳</span>
          </div>
          <div class="step-time">{{ new Date(step.timestamp).toLocaleTimeString() }}</div>
        </div>
        
        <!-- 思考部分 - 可展开/收缩 -->
        <div class="thinking-section" v-if="step.thinking">
          <div class="thinking-header" @click="step.showThinking = !step.showThinking">
            <span class="thinking-title">思考过程</span>
            <span class="toggle-icon" :class="{ 'expanded': step.showThinking }">
              ▼
            </span>
          </div>
          <div class="thinking-content" v-show="step.showThinking">
            {{ step.thinking }}
          </div>
        </div>
        
        <!-- 响应内容部分 -->
        <div class="step-content markdown-body" v-if="step.content" v-html="parseMarkdown(step.content)"></div>
      </div>
    </div>

	<!-- 人工审核部分 -->
	<div class="review-panel" v-if="isReviewing">
		<h2>AI 人工审核面板</h2>

		<div>
			<p>⚠️ 系统正在等待人工审核输入...</p>
			<textarea
				v-model="userReviewInput"
				placeholder="请进行人工审核..."
				rows="4"
				class="input-box"
			></textarea>
			<button @click="submitReviewInput" class="btn">提交审核结果</button>
		</div>
	</div>

	<!-- 最终报告部分 -->
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
import { marked } from 'marked'  // 引入marked库
import DOMPurify from 'dompurify';  // 引入安全过滤插件


const userInput = ref('')
const userReviewInput = ref('')
const isSubmitting = ref(false)
const steps = ref([])
const reportContent = ref('')
const eventSource = ref(null)
const isReviewing = ref(false);

const currentActiveStep = ref(null); // 使用 ref 确保响应式

// Markdown解析方法，增加安全过滤
const parseMarkdown = (content) => {
  if (!content) return '';
  // 1. 先用 marked 将 Markdown 解析为 HTML
  const html = marked.parse(content);
  // 2. 用 DOMPurify 过滤危险 HTML 内容（防止 XSS）
  return DOMPurify.sanitize(html);
};


// === 提交人工审核输入 ===
const submitReviewInput = async () => {
  if (!userReviewInput.value.trim()) {
    alert("请输入审核意见！");
    return;
  }
  try {
    // const res = await fetch("/send_input", {
    const res = await fetch("http://localhost:8000/send_input", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: userReviewInput.value }),
    });
    console.log("提交审核输出:", res.status);
    if (res.status != 200) {
      alert( "提交失败");
      return;
    }
    currentActiveStep.value.content = userReviewInput.value + "\n";
    isReviewing.value = false;
  } catch (err) {
    console.error(err);
    alert("提交失败，请检查网络连接");
  }
  return;
};

// deepseek修改后
const submitRequest = () => {
  if (!userInput.value.trim()) return

  // 存储当前活跃的步骤（用于后续更新complete/error状态）
  isSubmitting.value = true
  steps.value = []
  reportContent.value = ''
  
  // 初始化SSE连接
  eventSource.value = new EventSource(`/api/research?query=${encodeURIComponent(userInput.value)}`)

  eventSource.value.onmessage = (event) => {
    try {
      const backData  = JSON.parse(event.data);
      console.log('Received data:', backData );
      handleBackendData(backData); // 统一处理后端数据
    } catch (error) {
      console.error('Error processing event:', error);
    }
  };
    
  eventSource.value.onerror = () => {
    addStep('错误', '连接服务器失败', true);
    finishProcessing();
  };


    // 统一处理后端数据
  const handleBackendData = (backData) => {
    const { step, state, data } = backData;
    const handlers = {
      initializing: () => handleInitializing(step, data),
      thinking: () => handleThinking(step, data),
      generating: () => handleGenerating(step, data),
      user_review: () => handleUserReview(step, data),
      completed: () => handleComplete(step, data),
      error: () => handleError(step, data),
      finished: () => handleFinish()
    };
 
    if (handlers[state]) {
      handlers[state]();
    } else {
      console.warn('Unknown state:', state, 'in step:', step);
    }
  };

  // 处理「阶段正在处理」状态：新增步骤，标记为活跃状态
  const handleInitializing = (step, data) => {
    // 创建处理中步骤（带加载动画标识）
    const stepElement = {
      step,
      title: `${getStepName(step)}阶段正在初始化`,
      thinking: data?.thinking || '',
      content: data?.content || '',
      isProcessing: false, // 用于渲染加载动画（圈圈）
      isError: false,
      timestamp: new Date().toISOString(),
      show: false,
      showThinking: true // 控制思考部分的展开状态
    };
    steps.value.push(stepElement);
    currentActiveStep.value = stepElement; // 记录当前活跃步骤，供后续更新

    nextTick(() => {
      stepElement.show = true;
      autoScroll();
    });
  };

      // 处理「阶段正在处理」状态：新增步骤，标记为活跃状态
  const handleThinking = (step, data) => {
    if (!currentActiveStep.value || currentActiveStep.value.step !== step) {
      console.warn(`No active step found for completed step: ${step}`);
      return;
    }
    // 更新步骤内容
    currentActiveStep.value.isProcessing = true;
    currentActiveStep.value.title = `${getStepName(step)}思考中`;
    // 处理思考部分和内容部分
    if (data) {
      currentActiveStep.value.thinking += data;
    }
    autoScroll();
  };

      // 处理「阶段正在思考」状态：新增步骤，标记为活跃状态
  const handleGenerating = (step, data) => {
    if (!currentActiveStep.value || currentActiveStep.value.step !== step) {
      console.warn(`No active step found for completed step: ${step}`);
      return;
    }
    // 更新步骤内容
    currentActiveStep.value.isProcessing = true;
    currentActiveStep.value.title = `${getStepName(step)}生成中`;
    // 处理思考部分和内容部分
    if (data) {
      currentActiveStep.value.content += data;
    }
    autoScroll();
  };

        // 处理「人工审核」状态
  const handleUserReview = (step, data) => {
    if (!currentActiveStep.value || currentActiveStep.value.step !== step) {
      console.warn(`No active step found for completed step: ${step}`);
      return;
    }
    // 更新步骤内容
	userReviewInput.value = data
	isReviewing.value = true;
    autoScroll();
  };

  // 处理「阶段完成」状态：更新当前活跃步骤的内容
  const handleComplete = (step, data) => {
    if (!currentActiveStep.value || currentActiveStep.value.step !== step) {
      console.warn(`No active step found for completed step: ${step}`);
      return;
    }
    // 更新步骤内容（关闭加载动画，显示结果）
    currentActiveStep.value.isProcessing = false;
    currentActiveStep.value.title = `${getStepName(step)}处理完成`;
    if (data) {
      currentActiveStep.value.content += data;
    }
    currentActiveStep.value = null; // 清除活跃状态，等待下一阶段
    autoScroll();
  };

  // 处理「阶段出错」状态：更新当前活跃步骤的内容（标记错误）
  const handleError = (step, data) => {
    if (!currentActiveStep.value || currentActiveStep.value.step !== step) {
      console.warn(`No active step found for error step: ${step}`);
      return;
    }
    // 更新步骤内容（关闭加载动画，显示错误信息）
    currentActiveStep.value.isProcessing = false;
    currentActiveStep.value.isError = true;
    currentActiveStep.value.title = `${getStepName(step)}处理异常`;
    
    // 处理错误信息
    if (data) {
      currentActiveStep.value.content += data;
    }
    
    currentActiveStep.value = null; // 清除活跃状态
    autoScroll();
  };

  // 处理「所有阶段完成」状态
  const handleFinish = () => {
    // 新增最终完成步骤
    const finishStep = {
      step: 'finish',
      title: '所有阶段处理完成',
      content: '流程已结束',
      thinking: '',
      isProcessing: false,
      isError: false,
      timestamp: new Date().toISOString(),
      show: false,
      showThinking: false
    };
    steps.value.push(finishStep);
    nextTick(() => {
      finishStep.show = true;
      autoScroll();
      finishProcessing(); // 关闭连接
    });
  };

  // 辅助函数：获取阶段中文名称
  const getStepName = (step) => {
    const stepNames = {
      searching: '搜索',
      reading: '阅读',
      analyzing: '分析',
      writing: '撰写',
      writing_director: '撰写指导',
      section_writing: '撰写小节',
      reporting: '报告',

      // 扩展其他阶段
    };
    return stepNames[step] || step;
  };

  // 结束流程
  const finishProcessing = () => {
    isSubmitting.value = false;
    eventSource.value?.close();
  };

  // 自动滚动到最新步骤
  const autoScroll = () => {
    // 实现滚动逻辑（例如滚动到容器底部）
    const container = document.getElementById('steps-container');
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
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

<style scoped>
.app-container {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  color: #333;
}

h1 {
  text-align: center;
  color: #2c3e50;
  margin-bottom: 30px;
  font-weight: 600;
}

.input-section {
  margin-bottom: 20px;
  background: #f8f9fa;
  padding: 20px;
  border-radius: 10px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

.input-section textarea {
  width: 96%;
  padding: 15px;
  margin-bottom: 15px;
  border: 1px solid #e1e5e9;
  border-radius: 8px;
  font-size: 16px;
  transition: border-color 0.3s;
  resize: vertical;
}

.input-section textarea:focus {
  outline: none;
  border-color: #3498db;
  box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
}

.input-section button {
  background: #3498db;
  color: white;
  border: none;
  padding: 12px 25px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 16px;
  font-weight: 500;
  transition: background 0.3s;
}

.input-section button:hover:not(:disabled) {
  background: #2980b9;
}

.input-section button:disabled {
  background: #bdc3c7;
  cursor: not-allowed;
}

.progress-section {
  margin-bottom: 20px;
  max-height: 500px;
  overflow-y: auto;
  padding-right: 5px;
}

.step-card {
  transition: all 0.3s ease;
  opacity: 0;
  transform: translateY(20px);
  border-radius: 10px;
  padding: 0;
  margin-bottom: 20px;
  background: white;
  box-shadow: 0 3px 15px rgba(0, 0, 0, 0.08);
  overflow: hidden;
}

.step-card.show {
  opacity: 1;
  transform: translateY(0);
}

.step-card.error {
  border-left: 5px solid #e74c3c;
}

.step-card.processing {
  border-left: 5px solid #3498db;
}

.step-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px 20px;
  background: #f8f9fa;
  border-bottom: 1px solid #e9ecef;
}

.step-title {
  font-weight: 600;
  font-size: 16px;
  color: #2c3e50;
  display: flex;
  align-items: center;
}

.loading-spinner {
  display: inline-block;
  margin-left: 8px;
  animation: spin 1s linear infinite;
  color: #3498db;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.step-time {
  font-size: 14px;
  color: #7f8c8d;
}

.thinking-section {
  border-bottom: 1px solid #e9ecef;
}

.thinking-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 20px;
  cursor: pointer;
  background: #f1f8ff;
  transition: background 0.2s;
}

.thinking-header:hover {
  background: #e3f2fd;
}

.thinking-title {
  font-weight: 500;
  color: #3498db;
}

.toggle-icon {
  transition: transform 0.3s;
  color: #7f8c8d;
}

.toggle-icon.expanded {
  transform: rotate(180deg);
}

.thinking-content {
  padding: 15px 20px;
  background: #f8fafc;
  color: #4a5568;
  font-size: 14px;
  line-height: 1.5;
  white-space: pre-wrap;
  border-top: 1px solid #e2e8f0;
}

.step-content {
  padding: 20px;
  white-space: pre-wrap;
  line-height: 1.6;
  color: #2d3748;
}

.report-section {
  margin-top: 30px;
  background: #f8f9fa;
  padding: 20px;
  border-radius: 10px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

.report-section h2 {
  margin-top: 0;
  color: #2c3e50;
  font-weight: 600;
}

.report-section textarea {
  width: 100%;
  padding: 15px;
  margin-bottom: 15px;
  border: 1px solid #e1e5e9;
  border-radius: 8px;
  font-size: 16px;
  resize: vertical;
}

.report-section button {
  background: #27ae60;
  color: white;
  border: none;
  padding: 12px 25px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 16px;
  font-weight: 500;
  transition: background 0.3s;
}

.report-section button:hover {
  background: #219653;
}

/* 滚动条样式 */
.progress-section::-webkit-scrollbar {
  width: 6px;
}

.progress-section::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 10px;
}

.progress-section::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 10px;
}

.progress-section::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}

/* 引入Markdown样式（可根据需要调整） */
.markdown-body {
  box-sizing: border-box;
  min-width: 200px;
  max-width: 980px;
  margin: 0 auto;
  padding: 16px;
  line-height: 1.5;
  word-break: break-word;
}

.markdown-body h1,
.markdown-body h2,
.markdown-body h3 {
  margin-top: 24px;
  margin-bottom: 16px;
  font-weight: 600;
  line-height: 1.25;
}

.markdown-body p {
  margin-top: 0;
  margin-bottom: 16px;
}

.markdown-body ul,
.markdown-body ol {
  padding-left: 2em;
  margin-top: 0;
  margin-bottom: 16px;
}

.markdown-body strong {
  font-weight: 600;
}

.markdown-body em {
  font-style: italic;
}

.markdown-body a {
  color: #0366d6;
  text-decoration: none;
}

.markdown-body a:hover {
  text-decoration: underline;
}

.markdown-body code {
  padding: 0.2em 0.4em;
  margin: 0;
  font-size: 85%;
  background-color: rgba(27, 31, 35, 0.05);
  border-radius: 3px;
}

.review-panel {
  max-width: 600px;
  margin: 60px auto;
  text-align: center;
  padding: 20px;
  border-radius: 12px;
  background: #f8f9fa;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
}
.btn {
  background-color: #409eff;
  color: white;
  padding: 10px 16px;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  margin-top: 10px;
}
.btn:hover {
  background-color: #66b1ff;
}
.input-box {
  width: 100%;
  margin-top: 10px;
  padding: 10px;
  border-radius: 8px;
  border: 1px solid #ccc;
  resize: vertical;
}

</style>