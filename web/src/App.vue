<template>
  <div class="app-container">
    <h1>智能调研报告生成</h1>
    
    <div class="input-section">
      <textarea 
        v-model="userInput" 
        placeholder="请输入您想调研的问题..."
        rows="5"
        class="input-textarea"
      ></textarea>  
      
      <div class="input-actions">
        <button 
          class="btn-submit"
          @click="submitRequest" 
          :disabled="isSubmitting"
        >
          {{ isSubmitting ? '处理中...' : '提交' }}
        </button>
        
        <div class="knowledge-selector">
          <button 
            class="btn-select-knowledge"
            @click="showSelectModal = true"
          >
            {{ selectedDatabase ? `📚 ${selectedDatabase.name}` : '选择知识库' }}
          </button>
        </div>
      </div>
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
    
    <!-- 知识库选择模态框 -->
    <SelectKnowledgeModal
      v-model:visible="showSelectModal"
      :current-database-id="selectedDatabase?.id || ''"
      @select="handleSelectDatabase"
      @create-database="handleCreateDatabase"
    />
  </div>
</template>

<script setup>
import { ref, nextTick, onBeforeUnmount } from 'vue'
import { useRouter } from 'vue-router'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import SelectKnowledgeModal from './components/SelectKnowledgeModal.vue'
import { knowledgeApi } from './api/knowledge'

const router = useRouter()

const userInput = ref('')
const userReviewInput = ref('')
const isSubmitting = ref(false)
const steps = ref([])
const reportContent = ref('')
const eventSource = ref(null)
const isReviewing = ref(false)
const showSelectModal = ref(false)
const selectedDatabase = ref(null)

const currentActiveStep = ref(null); // 使用 ref 确保响应式
const activeSubSteps = ref(new Map()); // 追踪writing步骤的活跃子块（key为section_id）

// Markdown解析方法，增加安全过滤
const parseMarkdown = (content) => {
  if (!content) return '';
  // 1. 先用 marked 将 Markdown 解析为 HTML
  const html = marked.parse(content);
  // 2. 用 DOMPurify 过滤危险 HTML 内容（防止 XSS）
  return DOMPurify.sanitize(html);
};

const scrollToLatest = () => {
  nextTick(() => {
    const container = document.getElementById('steps-container');
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  });
};

const addStep = (title, content = '', isError = false) => {
  const stepElement = {
    step: isError ? 'error' : 'message',
    title,
    thinking: '',
    content,
    isProcessing: false,
    isError,
    timestamp: new Date().toISOString(),
    show: false,
    showThinking: false,
  };
  steps.value.push(stepElement);
  nextTick(() => {
    stepElement.show = true;
    scrollToLatest();
  });
};


// === 提交人工审核输入 ===
const submitReviewInput = async () => {
  if (!userReviewInput.value.trim()) {
    alert("请输入审核意见！");
    return;
  }
  try {
    const res = await fetch("/send_input", {
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

  const dbId = selectedDatabase.value ? selectedDatabase.value.id : ''

  isSubmitting.value = true
  steps.value = []
  reportContent.value = ''
  
  // 初始化SSE连接
  eventSource.value = new EventSource(`/api/research?query=${encodeURIComponent(userInput.value)}`)

  //   let apiUrl = `/api/research?query=${encodeURIComponent(userInput.value)}`
  // if (dbId) {
  //   apiUrl += `&db_id=${encodeURIComponent(dbId)}`
  // }
  
  // eventSource.value = new EventSource(apiUrl)

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
      thinking: typeof data === 'string' ? '' : data?.thinking || '',
      content: typeof data === 'string' ? data : data?.content || '',
      isProcessing: false, // 用于渲染加载动画（圈圈）
      isError: false,
      timestamp: new Date().toISOString(),
      show: false,
      showThinking: true, // 控制思考部分的展开状态
    };
    steps.value.push(stepElement);
    if (step.startsWith("section_writing")) {
      // writing步骤添加子块数组
      activeSubSteps.value.set(step, stepElement);
    }else{
      currentActiveStep.value = stepElement; // 记录当前活跃步骤，供后续更新
    }

    nextTick(() => {
      stepElement.show = true;
      autoScroll();
    });
  };

      // 处理「阶段正在处理」状态：新增步骤，标记为活跃状态
  const handleThinking = (step, data) => {
    if(step.startsWith("section_writing")){
      currentActiveStep.value = activeSubSteps.value.get(step)
    }
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
    if(step.startsWith("section_writing")){
      currentActiveStep.value = activeSubSteps.value.get(step)
    }
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
    if(step.startsWith("section_writing")){
      currentActiveStep.value = activeSubSteps.value.get(step)
    }
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
    if(step.startsWith("section_writing")){
      currentActiveStep.value = activeSubSteps.value.get(step)
    }
    if (!currentActiveStep.value || currentActiveStep.value.step !== step) {
      console.warn(`No active step found for error step: ${step}`);
      addStep(`${getStepName(step)}处理异常`, data || '流程处理异常', true);
      finishProcessing();
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
    finishProcessing();
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
      start: '流程',

      // 扩展其他阶段
    };
    if(step.startsWith("section_writing")){
      const stepParts = step.split("_");
      const partNum = stepParts[stepParts.length - 1]; // 用 length-1 取最后一个元素
      stepNames[step] = "撰写第" + partNum + "部分";
    }
    return stepNames[step] || step;
  };

  // 结束流程
  const finishProcessing = () => {
    isSubmitting.value = false;
    eventSource.value?.close();
    activeSubSteps.value.clear(); // 清理活跃子块Map
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

const handleSelectDatabase = async (database) => {
  if (!database) {
    selectedDatabase.value = null
    try {
      await knowledgeApi.selectDatabase('')
      console.log('已取消选择知识库')
    } catch (error) {
      console.error('取消选择知识库失败:', error)
      alert('取消选择知识库失败，请重试')
    }
    return
  }

  try {
    await knowledgeApi.selectDatabase(database.id)
    selectedDatabase.value = database
    console.log(`已选择知识库: ${database.name} (ID: ${database.id})`)
  } catch (error) {
    console.error('选择知识库失败:', error)
    alert('选择知识库失败，请重试')
  }
}

const handleCreateDatabase = () => {
  router.push('/knowledge')
}

const goToKnowledgeBase = () => {
  router.push('/knowledge')
}

onBeforeUnmount(() => {
  if (eventSource.value) {
    eventSource.value.close()
  }
})
</script>

<style scoped>
.app-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  color: #333;
  width: 100%;
  box-sizing: border-box;
}

h1 {
  text-align: center;
  color: #2c3e50;
  margin-bottom: 30px;
  font-weight: 600;
  font-size: clamp(24px, 3vw, 32px);
}

.input-section {
  margin-bottom: 20px;
  background: #f8f9fa;
  padding: clamp(15px, 3vw, 25px);
  border-radius: 10px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
  max-width: 100%;
  width: 100%;
  box-sizing: border-box;
}

.input-textarea {
  width: 100%;
  padding: clamp(12px, 2vw, 18px);
  margin-bottom: clamp(10px, 2vw, 20px);
  border: 1px solid #e1e5e9;
  border-radius: 8px;
  font-size: clamp(14px, 1.5vw, 16px);
  line-height: 1.5;
  transition: border-color 0.3s, box-shadow 0.3s;
  resize: vertical;
  font-family: inherit;
  box-sizing: border-box;
  min-height: 120px;
  max-height: 300px;
}

.input-textarea:focus {
  outline: none;
  border-color: #3498db;
  box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.15);
}

.input-textarea::placeholder {
  color: #adb5bd;
}

.input-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: clamp(10px, 2vw, 20px);
  flex-wrap: wrap;
}

.btn-submit {
  padding: clamp(10px, 1.5vw, 14px) clamp(20px, 4vw, 30px);
  background: #3498db;
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: clamp(14px, 1.5vw, 16px);
  font-weight: 500;
  transition: background 0.3s, transform 0.2s;
  flex-shrink: 0;
  min-width: fit-content;
}

.btn-submit:hover:not(:disabled) {
  background: #2980b9;
  transform: translateY(-1px);
}

.btn-submit:active:not(:disabled) {
  transform: translateY(0);
}

.btn-submit:disabled {
  background: #bdc3c7;
  cursor: not-allowed;
  transform: none;
}

.knowledge-selector {
  display: flex;
  gap: clamp(8px, 1.5vw, 12px);
  align-items: center;
  flex-shrink: 0;
  flex-wrap: wrap;
}

.btn-select-knowledge {
  padding: clamp(8px, 1.2vw, 12px) clamp(16px, 3vw, 24px);
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: clamp(12px, 1.2vw, 14px);
  font-weight: 500;
  transition: all 0.2s;
  white-space: nowrap;
  min-width: fit-content;
}

.btn-select-knowledge {
  background: #e3f2fd;
  color: #3498db;
  border: 2px solid #3498db;
}

.btn-select-knowledge:hover {
  background: #3498db;
  color: white;
  transform: translateY(-1px);
}

@media (max-width: 1200px) {
  .input-section {
    padding: 18px;
  }
  
  .input-textarea {
    padding: 14px;
    font-size: 15px;
  }
}

@media (max-width: 992px) {
  .input-section {
    padding: 16px;
  }
  
  .input-textarea {
    padding: 13px;
    font-size: 15px;
    min-height: 100px;
  }
}

@media (max-width: 768px) {
  .app-container {
    padding: 15px;
    max-width: 100%;
  }
  
  .input-section {
    padding: 15px;
    border-radius: 8px;
  }
  
  .input-actions {
    flex-direction: column;
    align-items: stretch;
    gap: 12px;
  }
  
  .btn-submit {
    width: 100%;
    padding: 12px 20px;
    font-size: 15px;
  }
  
  .knowledge-selector {
    width: 100%;
    justify-content: center;
    gap: 10px;
  }
  
  .btn-select-knowledge {
    flex: 1;
    text-align: center;
    padding: 10px 16px;
    font-size: 13px;
  }
  
  .input-textarea {
    padding: 12px;
    font-size: 14px;
    min-height: 80px;
  }
  
  .progress-section,
  .report-section {
    max-width: 100%;
  }
}

@media (max-width: 480px) {
  .app-container {
    padding: 10px;
  }
  
  h1 {
    font-size: 24px;
    margin-bottom: 20px;
  }
  
  .input-section {
    padding: 12px;
  }
  
  .input-textarea {
    padding: 10px;
    font-size: 14px;
    min-height: 70px;
    max-height: 200px;
  }
  
  .btn-submit {
    padding: 10px 16px;
    font-size: 14px;
  }
  
  .btn-select-knowledge {
    padding: 8px 12px;
    font-size: 12px;
  }
}

@media (min-width: 1400px) {
  .app-container {
    max-width: 1000px;
  }
}

.progress-section {
  margin: 0 auto 20px;
  max-width: 1000px;
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
  margin: 30px auto;
  max-width: 1000px;
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
