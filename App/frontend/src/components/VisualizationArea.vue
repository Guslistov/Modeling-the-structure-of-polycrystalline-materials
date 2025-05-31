<template>
  <div class="visualization-area">
    <div class="images-container">
      <ImageViewer 
        :title="originalTitle" 
        :imageUrl="originalImage" 
      />
      <ImageViewer 
        :title="processedTitle" 
        :imageUrl="processedImage" 
      />
    </div>
  </div>
</template>

<script setup>
import ImageViewer from './ImageViewer.vue';

defineProps({
  originalImage: String,
  processedImage: String,
  originalTitle: {
    type: String,
    default: 'До'
  },
  processedTitle: {
    type: String,
    default: 'После'
  }
});
</script>
  
  <style scoped>
  .visualization-area {
    flex: 1;
    display: flex;
    flex-direction: column;
    padding: 20px;
    
  }
  .images-container {
    display: flex;
    gap: 20px;
    flex: 1;
    max-height: 600px;
  }
  </style>