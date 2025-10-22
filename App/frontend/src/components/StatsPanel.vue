<template>
    <div class="stats-panel">
      <h3>Сообщения</h3>
      <div v-if="error" class="error-message">{{ error }}</div>
      <div v-else class="error-message">...</div>
    </div>
  </template>
  
  <script setup>
  defineProps({
    error: String
  });
  </script>
  
<style scoped>
.stats-panel {
    padding: 15px;
    background-color: #f5f5f5;
    border-radius: 4px;
    border: 1px solid #ddd;
}

.stats-panel h3 {
    margin-bottom: 10px;
}

.stats-panel p {
    margin-bottom: 100px;
    font-size: 14px;
}

.error-message {
  color: rgb(90, 72, 255);
  margin-top: 10px;
  font-size: 16px;
}
</style>