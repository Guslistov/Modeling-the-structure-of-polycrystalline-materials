<template>
    <div class="image-viewer">
      <label>{{ title }}</label>
      <img 
        v-if="imageUrl" 
        :src="imageUrl" 
        alt="Data image"
        class="data-image"
      >
      <div v-else class="placeholder">
        Нет данных
      </div>
    </div>
  </template>
  
<script setup>
defineProps({
    title: String,
    imageUrl: String
});
</script>

<style scoped>
.image-viewer {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.data-image {
    width: 100%;
    height: calc(100% - 25px);
    object-fit: contain;
    border: 1px dashed #ccc;
    color: #999;
    background-color: #f9f9f9;
}
.placeholder {
    width: 100%;
    height: calc(100% - 25px);
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px dashed #ccc;
    color: #999;
    background-color: #f9f9f9;
}
label{
  font-size: 16pt;
  text-align: center;
  font-weight: bold;
}
</style>