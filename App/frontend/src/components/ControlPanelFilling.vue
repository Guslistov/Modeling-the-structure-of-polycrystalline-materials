<template>
  <div class="control-panel">
    <div class="file-upload">
      <input type="file" ref="fileInput" @change="handleFileUpload" hidden>
      <button @click="triggerFileInput">Загрузить файл</button>
      <div class="file-name">{{ fileName || "Файл не выбран" }}</div>
    </div>

    <div class="control-row">
      <label class="control-label">Модель:</label>
      <select v-model="selectedModel" class="control-input">
        <option v-for="model in models" :key="model" :value="model">
          {{ model }}
        </option>
      </select>
    </div>

    <div class="control-row">
      <label class="control-label">Шаг сдвига скользящего окна:</label>
      <input 
        type="number" 
        v-model.number="shiftStep" 
        min="0"
        class="input-step"
      >
    </div>

    <div class="control-row">
      <label class="control-label">XSize:</label>
      <input 
        type="number" 
        v-model.number="XStep" 
        min="0"
        class="input-step"
      >
    </div>

    <div class="control-row">
      <label class="control-label">YSize:</label>
      <input 
        type="number" 
        v-model.number="YStep" 
        min="0"
        class="input-step"
      >
    </div>
    <button 
      class="process-button"
      :class="{ 'blinking': isProcessing }"
      @click="start_process_resized"
      :disabled="isProcessing || !original_file"
    >
      Заполнить чересстрочные пропуски
    </button>

    <div class="download-section">
      <input 
        type="text" 
        v-model="downloadFileName" 
        placeholder="Имя файла"
        class="input-download-filename"
      >
      <button 
        @click="downloadFile"
        :disabled="!processedImage"
      >
        Скачать файл
      </button>
    </div>
  </div>
</template>

<script setup>

import { ref, onMounted } from 'vue';

const props = defineProps({
  onFileUpload: Function,
  onFileProcess: Function,
  onFileProcess_reconstraction: Function,
  activeTab: Number
});

const emit = defineEmits(['file-uploaded', 'processing-complete', 'download-file', 'new-message']);

const fileInput = ref(null);
const fileName = ref('');
const models = ref([]);
const selectedModel = ref(null);
const original_image = ref(null);
const original_file = ref(null);
const processedImage = ref(null);
const downloadFileName = ref(null);
const error = ref(null);
const shiftStep = ref(64);

const XStep = ref(2);
const YStep = ref(2);

const isProcessing = ref(false);

const triggerFileInput = () => fileInput.value.click();

onMounted(async () => {
  try {
    const res = await fetch('/api/models');
    const data = await res.json();
    models.value = data.models;
    if (models.value.length > 0) {
      selectedModel.value = models.value[0];
    }
  } catch (error) {
    console.error('Failed to fetch models:', error);
  }
});

const handleFileUpload = async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  
  fileName.value = file.name;
  error.value = null;
  emit('new-message', error, props.activeTab);
  
  try {
    const result = await props.onFileUpload(file);
    original_image.value = result.original_image;
    original_file.value = result.original_file;
    XStep.value = 2;//result.step_x;
    YStep.value = 2;//result.step_y;
    emit('file-uploaded', result, props.activeTab);
    processedImage.value = null;
    error.value = 'Файл загружен без ошибок';
    emit('new-message', error, props.activeTab);
  } catch (err) {
    error.value = 'Ошибка при загрузке файла';
    emit('new-message', error, props.activeTab);
    console.error('Upload error:', err);
  }
};

const start_process_resized = async () => {
  if (!original_file.value) {
    error.value = 'Сначала загрузите файл';
    return;
  }
  
  isProcessing.value = true;
  error.value = "функция заполнения чересстрочных пропусков запущена...";
  emit('new-message', error, props.activeTab);

  try {
    const result = await props.onFileProcess_reconstraction({
      file_name: original_file.value.split('/').pop(),
      model_name: selectedModel.value,
      window_padding: shiftStep.value,
      step_y: YStep.value,
      step_x: XStep.value,
      resized: true
    });
    
    processedImage.value = result.processed_image;
    downloadFileName.value = fileName.value.split('.').slice(0, -1).join('.') + "_new" + ".ctf";
    emit('processing-complete', result, props.activeTab);
    error.value = null;
    emit('new-message', error, props.activeTab);
  } catch (err) {
    error.value = err.message || 'Ошибка при обработке файла';
    emit('new-message', error, props.activeTab);
    console.error('Processing error details:', err.response?.data || err);
  } finally {
    isProcessing.value = false;
  }
};

const downloadFile = () => {
  if (!processedImage.value) return;
  emit('download-file', {filename: downloadFileName.value}, props.activeTab);
};
</script>
  
<style scoped>
.control-panel {
  width: 340px;
  background-color: #f0f0f0;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 15px;
  border-right: 1px solid #ccc;
}
.control-row {
  display: flex;
  align-items: center;
  gap: 10px;
}
.control-label {
  text-align: left;
  font-size: 16px;
}


.file-upload {
  display: flex;
  align-items: center;
  gap: 5px;
}
.file-name {
  font-size: 16px;
  color: #666;
  word-break: break-all;
  padding-left: 10px;
}

.input-step {
  width: 64px;
}

.process-button {
  margin-top: 10px;
  background-color: #33AD41;
}
.process-button:hover {
  background-color: #298b35;
}
.process-button:disabled {
  background-color: #cccccc;
  cursor: not-allowed;
}

.input-download-filename {
  width: 173px;
}

.blinking {
  animation: blinking 1s infinite;
}

@keyframes blinking {
  0%, 100% {
    filter: brightness(1);
  }
  50% {
    filter: brightness(0.8);
  }
}
</style>