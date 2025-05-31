<template>
  <header class="header">
    <div class="logo">EBSD: Euler Fix</div>
    <nav class="nav">
      <a
        v-for="(tab, idx) in tabs"
        :key="tab"
        href="#"
        class="nav-link"
        :class="{ active: idx === activeTab }"
        @click.prevent="$emit('tab-change', idx)"
      >
        {{ tab }}
      </a>
    </nav>
  </header>
</template>

<script setup>
const props = defineProps({
  activeTab: Number
})

const tabs = [
  'Реконструкция углов Эйлера',
  'Заполнение чересстрочных пропусков',
  'Построение IPF-карт'
]
</script>
  
  <style scoped>
  .header {
    display: flex;
    width: 100%;
    height: 100px;
    background-color: #fff;
  }
  
  .logo {
    width: 340px;
    height: 100%;
    background: linear-gradient(to right, #118AB2, #073B4C);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 24px;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: Arial, sans-serif;
  }
  
  .nav {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    flex-grow: 1;
    background-color: #118AB2;
    padding-left: 40px;
  }
  
  .nav-link {
    color: white;
    text-decoration: none;
    font-family: Arial, sans-serif;
    font-size: 18px;
    margin-right: 20px;
  }
  
  .nav-link:last-child {
    margin-right: 0;
  }

  .nav-link.active {

  border-bottom: 3px solid #fff;
}
  </style>