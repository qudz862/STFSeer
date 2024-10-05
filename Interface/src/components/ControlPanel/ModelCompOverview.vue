<script setup>
import { ref, reactive, computed, watch, onMounted, onUpdated } from "vue"
import { useInforStore } from '@/stores/infor.js'
import * as d3 from 'd3'
import { valColorScheme_blue, valColorScheme_red, valColorScheme_fire, valColorScheme_double} from '@/data/index.js'

const props = defineProps({
  model_name: String,
  model_type: String,
  step_error: Object,
  model_parameters: Object
})

const inforStore = useInforStore()

onMounted(() => {

})

onUpdated(() => {

})

watch (() => inforStore.multi_step_err_infor, (oldValue, newValue) => {
  
})

function drawModelCompOverview() {
  d3.select('#model_comp_overview').selectAll('*').remove()
  let margin_left = 10, margin_right = 10, margin_top = 18, margin_bottom = 10
  let main_w = 300, main_h = 300
  let svg_w = main_w + margin_left + margin_right
  let svg_h = main_h + margin_top + margin_bottom
  let svg = d3.select('#model_comp_overview')
    .attr('width', svg_w)
    .attr('height', svg_h)
    .append("g")
    .attr("transform", `translate(${margin_left},${margin_top})`)
  
}

</script>

<template>
  <div>
    <svg id="model_comp_overview" width="300" height="300"></svg>
  </div>
  
</template>

<style scoped>

</style>