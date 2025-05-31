<template>
  <div class="main-content">
    <keep-alive>
      <component
        :is="currentControlPanel"
        :activeTab="activeTab"
        :onFileProcess="processFile"
        :onFileProcess_reconstraction="processFile_reconstraction"
        :onFileUpload="uploadFile"
        @file-uploaded="(result, tabIndex) => handleOriginalFile(result, tabIndex)"
        @processing-complete="(result, tabIndex) => handleProcessedFile(result, tabIndex)"
        @download-file="(fileData, tabIndex) => downloadProcessedFile(fileData, tabIndex)"
        @new-message="(msg, tabIndex) => handleNewMessage(msg, tabIndex)"
      />
    </keep-alive>
    <div class="right-section">
      <VisualizationArea 
        :originalImage="currentOriginalImage"
        :processedImage="currentProcessedImage"
        :originalTitle="originalTitle"
        :processedTitle="processedTitle"
      />
      <StatsPanel 
        :error="currentErrorMessage"
      />
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import { computed } from 'vue'
import axios from 'axios';

const props = defineProps({ activeTab: Number })

import ControlPanelReconstruction from './ControlPanelReconstruction.vue';
import ControlPanelFilling from './ControlPanelFilling.vue';
import ControlPanelIPF from './ControlPanelIPF.vue';
const components = [
  ControlPanelReconstruction,
  ControlPanelFilling,
  ControlPanelIPF
]
const currentControlPanel = computed(() => components[props.activeTab])


import VisualizationArea from './VisualizationArea.vue';
const originalTitles = [
  'Углы Эйлера: до',          // для первой вкладки
  'Углы Эйлера: до',      // для второй вкладки
  'Углы Эйлера'       // для третьей вкладки
]
const processedTitles = [
  'Углы Эйлера: после',       // для первой вкладки
  'Углы Эйлера: после',   // для второй вкладки
  'IPF-карта'            // для третьей вкладки
]
const originalTitle = computed(() => originalTitles[props.activeTab])
const processedTitle = computed(() => processedTitles[props.activeTab])

const currentOriginalImage = computed(() => originalFiles.value[props.activeTab].original_image);
//const currentOriginalFile = computed(() => originalFiles.value[props.activeTab].original_file);
//const currentImageTimestamp = computed(() => originalFiles.value[props.activeTab].imageTimestamp);
const currentProcessedImage = computed(() => processedFiles.value[props.activeTab].processed_image);
//const currentProcessedFile = computed(() => processedFiles.value[props.activeTab].processed_file);
//const currentImageTimestamp = computed(() => processedFiles.value[props.activeTab].imageTimestamp);
const currentErrorMessage = computed(() => errorMessages.value[props.activeTab]);

import StatsPanel from './StatsPanel.vue';

const original_image = ref(null);
const original_file = ref(null);
const processed_image = ref(null);
const processed_file = ref(null);
const error_message = ref(null);
const imageTimestamp = ref(Date.now());

const originalFiles = ref([
  { original_file: null, original_image: null, imageTimestamp: null }, // Вкладка 0
  { original_file: null, original_image: null, imageTimestamp: null }, // Вкладка 1
  { original_file: null, original_image: null, imageTimestamp: null }, // Вкладка 2
]);
const processedFiles = ref([
  { processed_file: null, processed_image: null, imageTimestamp: null }, // Вкладка 0
  { processed_file: null, processed_image: null, imageTimestamp: null }, // Вкладка 1
  { processed_file: null, processed_image: null, imageTimestamp: null }, // Вкладка 2
]);
const errorMessages = ref([null, null, null]); // Ошибки по вкладкам

const uploadFile = async (file) => {
  if (!file.name.toLowerCase().endsWith('.ctf')) {
    throw new Error('Пожалуйста, выберите EBSD с расширением .ctf');
  } 
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const response = await axios.post(
      import.meta.env.VITE_API_BASE_URL + '/upload/',
      formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    
    return { original_file: response.data.original_file, original_image: response.data.original_image, step_x: response.data.step_x, step_y: response.data.step_y};
  } catch (err) {
    throw new Error(err.response?.data?.detail || 'Не удалось загрузить');
  }
};

const processFile = async ({ file_name, IPF_type}) => {  // Деструктуризация объекта
  try {
    const response = await axios.post(
      import.meta.env.VITE_API_BASE_URL + '/process/',
      { file_name, IPF_type },  // Ключ должен совпадать с ожидаемым на сервере
      {
        headers: {
          'Content-Type': 'application/json'
        }
      }
    );
    return { processed_file: response.data.processed, processed_image: response.data.processed };
  } catch (error) {
    console.error('Processing error:', error.response?.data);
    throw new Error(error.response?.data?.detail || 'Не удалось обработать изображение');
  }
};

const processFile_reconstraction = async ({ file_name, model_name, window_padding, step_y, step_x, resized}) => { 
  try {
    const response = await axios.post(
      import.meta.env.VITE_API_BASE_URL + '/process1/',
      { file_name, model_name, window_padding, step_y, step_x, resized},
      {
        headers: {
          'Content-Type': 'application/json'
        }
      }
    );
    return { processed_file: response.data.processed_file, processed_image: response.data.processed_image};
  } catch (error) {
    console.error('Processing error:', error.response?.data);
    throw new Error(error.response?.data?.detail || 'Не удалось восстановить ориентации');
  }
};

const handleOriginalFile = (fileData, tabIndex) => {
  const tabData_O = originalFiles.value[tabIndex];

  // Очищаем предыдущий Blob URL если используется
  if (tabData_O.original_image?.startsWith('blob:')) {
    URL.revokeObjectURL(tabData_O.original_image);
  }

  tabData_O.original_file = fileData.original_file;

  // Добавляем временную метку к URL
  const timestamp = Date.now();
  tabData_O.original_image = fileData.original_image.includes('?')
    ? `${fileData.original_image}&t=${timestamp}`
    : `${fileData.original_image}?t=${timestamp}`;

  // Reset processed file/image for this tab
  const tabData_P = processedFiles.value[tabIndex];
  tabData_P.processed_file = null;
  tabData_P.processed_image = null;
  tabData_P.imageTimestamp = null; // Optional: reset timestamp for processed image
};

const handleProcessedFile = (fileData, tabIndex) => {
  const tabData_P = processedFiles.value[tabIndex];

  tabData_P.processed_file = fileData.processed_file;
  
  // Очищаем предыдущий Blob URL если используется
  if (tabData_P.processed_image?.startsWith('blob:')) {
    URL.revokeObjectURL(tabData_P.processed_image);
  }

  // Добавляем временную метку к URL
  const timestamp = Date.now();
  tabData_P.processed_image = fileData.processed_image.includes('?')
    ? `${fileData.processed_image}&t=${timestamp}`
    : `${fileData.processed_image}?t=${timestamp}`;
  tabData_P.imageTimestamp = timestamp;
};

const handleNewMessage = (fileData, tabIndex) => {
  errorMessages.value[tabIndex] = fileData.value;
};

const downloadProcessedFile = async (fileData, tabIndex) => {
  const processedFile = processedFiles.value[tabIndex].processed_file;
  if (!processedFile) return;
  try {
    // Вариант 1: Blob URL
    if (processedFile.startsWith('blob:')) {
      const link = document.createElement('a');
      link.href = processedFile;
      link.download = fileData.filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      return;
    }

    // Вариант 2: Запрос к серверу
    const response = await axios.get(processedFile, {
      responseType: 'blob'
    });

    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.download = fileData.filename;
    document.body.appendChild(link);
    link.click();

    setTimeout(() => {
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    }, 100);
  } catch (error) {
    console.error('Download error:', error);
    // Можно добавить отображение ошибки пользователю
  }
};

</script>
<style scoped>
.main-content {
  display: flex;
  flex: 1;
  height: calc(100vh - 120px); /* Учитываем высоту header */
}

.right-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0; /* Важно для корректного сжатия */
}
</style>