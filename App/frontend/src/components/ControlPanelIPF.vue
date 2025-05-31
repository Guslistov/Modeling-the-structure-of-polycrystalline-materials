<template>
  <div class="control-panel">
    <div class="file-upload">
      <input type="file" ref="fileInput" @change="handleFileUpload" hidden>
      <button @click="triggerFileInput">Загрузить файл</button>
      <div class="file-name">{{ fileName || "Файл не выбран" }}</div>
    </div>

    <div class="control-row radio-group">
      <label class="control-label">Направление:</label>
      <label class="radio-option" :class="{ active: axis === 'X' }">
        <input type="radio" v-model="axis" value="X" @change="emitAxisChange">
        <span>X</span>
      </label>
      <label class="radio-option" :class="{ active: axis === 'Y' }">
        <input type="radio" v-model="axis" value="Y" @change="emitAxisChange">
        <span>Y</span>
      </label>
      <label class="radio-option" :class="{ active: axis === 'Z' }">
        <input type="radio" v-model="axis" value="Z" @change="emitAxisChange">
        <span>Z</span>
      </label>
    </div>

    <button 
      class="process-button" 
      @click="createIPFMap"
      :disabled="!original_file"
    >
      Построить карту
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

import { ref } from 'vue';

const props = defineProps({
  onFileUpload: Function,
  onFileProcess: Function,
  onFileProcess_reconstraction: Function,
  activeTab: Number
});

const emit = defineEmits(['file-uploaded', 'processing-complete', 'download-file', 'axis-change', 'new-message']);

const fileInput = ref(null);
const fileName = ref('');
const original_image = ref(null);
const original_file = ref(null);
const processedImage = ref(null);
const downloadFileName = ref(null);
const error = ref(null);

const isProcessing = ref(false);

const axis = ref('Z')

const emitAxisChange = () => {
  emit('axis-change', axis.value)
}

const triggerFileInput = () => fileInput.value.click();

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
    emit('file-uploaded', result, props.activeTab);
    processedImage.value = null; // Сбрасываем предыдущий результат
  } catch (err) {
    error.value = 'Ошибка при загрузке файла';
    emit('new-message', error, props.activeTab);
    console.error('Upload error:', err);
  }
};

const createIPFMap = async () => {
  if (!original_file.value) {
    error.value = 'Сначала загрузите файл';
    return;
  }
  
  isProcessing.value = true;
  error.value = null;
  emit('new-message', error, props.activeTab);

  let type = 2
  if (axis.value == "X") {
    type = 0
  }
  else if (axis.value == "Y") {
    type = 1
  }
  else if (axis.value == "Z") {
    type = 2
  }
  
  try {
    const result = await props.onFileProcess({
      file_name: original_file.value.split('/').pop(),
      IPF_type: type
    });
    
    processedImage.value = result.processed_image;
    downloadFileName.value = fileName.value.split('.').slice(0, -1).join('.') + "_IPF-" + axis.value + ".png";
    emit('processing-complete', result, props.activeTab);
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


.control-row.radio-group {
  display: flex;
  border-radius: 8px;
  padding: 0px;
  margin-top: 8px;
}
.radio-option {
  flex: 1;
  text-align: center;
  padding: 8px 12px;
  cursor: pointer;
  border-radius: 6px;
  transition: all 0.2s ease;
  position: relative;
}
.radio-option input[type="radio"] {
  position: absolute;
  opacity: 0;
  width: 0;
  height: 0;
}
.radio-option span {
  display: block;
  font-weight: 500;
  color: #6b7280;
  transition: all 0.2s ease;
}
.radio-option.active {
  background: white;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}
.radio-option.active span {
  color: #3b82f6;
  font-weight: 600;
}
.radio-option:hover:not(.active) {
  background: rgba(255, 255, 255, 0.7);
}


.input-download-filename {
  width: 173px;
}
</style>