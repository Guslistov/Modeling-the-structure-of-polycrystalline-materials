<template>
  <div id="app">
    <AppHeader :activeTab="activeTab" @tab-change="onTabChange" />
    <main>
      <MainContent :activeTab="activeTab" />
    </main>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import AppHeader from './components/header.vue';
import MainContent from './components/MainContent.vue';
const activeTab = ref(0)

function onTabChange(idx) {
  activeTab.value = idx
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
  font-family: Arial, sans-serif;
}

select, input {
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  height: 40px;
}


button {
  padding: 10px;
  background-color: #073B4C;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
}
button:hover {
  opacity: 0.9;
}
button:disabled {
  background-color: #cccccc;
  cursor: not-allowed;
}

</style>