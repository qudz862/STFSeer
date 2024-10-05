<script setup>
import { ref, reactive, computed, watch, onMounted, onUpdated } from "vue"
import { useInforStore } from '@/stores/infor.js'
import getData from '@/services/index.js'
import * as d3 from 'd3'
import { valColorScheme_blue, valColorScheme_red, valColorScheme_fire } from '@/data/index.js'

const inforStore = useInforStore()

// watch (() => inforStore.cur_filtered_events, (oldValue, newValue) => {
//   console.log(inforStore.cur_filtered_events);
// })

const view_id = (feat_id) => `-${feat_id}`

</script>

<template>
  <div v-for="(item, index) in inforStore.cur_filtered_events" :key="index">
    {{ item.step }} {{ item.area_event }} {{ item.intensity_event }} {{ item.move_event }}
  </div>
</template>

<style scoped>

</style>