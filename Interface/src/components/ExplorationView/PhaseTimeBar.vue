<script setup>
import { ref, reactive, computed, watch, onMounted, onUpdated } from "vue"
import { useInforStore } from '@/stores/infor.js'
import * as d3 from 'd3'
import { valColorScheme_blue, valColorScheme_red, valColorScheme_fire, valColorScheme_double} from '@/data/index.js'

const props = defineProps({
  phase_id: Number,
  sorted_phase_id: Number,
  phase_data: Object,
})

const inforStore = useInforStore()

const view_id = (view_str, view_id) => `${view_str}_${view_id}`

onMounted(() => {

})

onUpdated(() => {
  
})


</script>

<template>
  <div>
    <svg :id="view_id('phase', phase_id)"></svg>
  </div>
  
</template>

<style scoped>

</style>