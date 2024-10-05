<script setup>
import { ref, reactive, computed, watch, onMounted, onUpdated } from "vue"
import { useInforStore } from '@/stores/infor.js'
import getData from '@/services/index.js'
import * as d3 from 'd3'
import $ from 'jquery'
import { valColorScheme_blue, valColorScheme_fire, valColorScheme_red } from '@/data/index.js'
import PhasesPCP from './PhasesPCP.vue'
import EventsPCP from './EventsPCP.vue'
import PhaseCard from './PhaseCard.vue'
import PhasePredCard from './PhasePredCard.vue'
import EventsTest from './EventsTest.vue'
import PhaseEventSeq from './PhaseEventSeq.vue'

const inforStore = useInforStore()

let show_indicators = ref(false)
function deepCopyArray(arr) {
  return JSON.parse(JSON.stringify(arr));
}

watch (() => inforStore.cur_filtered_phases_indices, (oldValue, newValue) => {
  let tmp_indices = deepCopyArray(inforStore.cur_filtered_phases_indices)
  inforStore.sorted_phase_ids = sortPhases(cur_sort_attr.value, cur_order.value, tmp_indices)
  // inforStore.sorted_phase_ids = tmp_indices
  // inforStore.sorted_phase_ids = Array.from({ length: inforStore.cur_filtered_phases.length }, (_, index) => index);
  // inforStore.cur_filtered_events = inforStore.cur_filtered_phases.reduce((accu, obj) => accu.concat(obj.evolution_events), [])
  drawCntLegend()
  //&&&&&&&&&&&&&&&&&
  //新加
  drawErrLegend()
})




let global_space_mae_range = [0, 0]
watch (() => inforStore.cur_phase_indicators, (oldValue, newValue) => {
  global_space_mae_range = inforStore.cur_phase_indicators[0].global_space_mae_range
  drawErrLegend()
})

const phaseSvgID = (index) => `phase-${index}`

watch (() => inforStore.cur_phase_id, (oldValue, newValue) => {
  inforStore.sel_event_record = -1
  if (inforStore.cur_phase_id == -1) {
    // 清楚phase选中框和phase details
    $('.phase-block').removeClass('selected')
    d3.select('#st-layout-phase').selectAll('*').remove()
    d3.select('#time-event-bar').selectAll('*').remove()
    $('.feature-panel').css('display', 'none')
  } else {
    if (!(`${inforStore.cur_phase_sorted_id}` in inforStore.event_records)) inforStore.event_records[`${inforStore.cur_phase_sorted_id}`] = []
    // 清楚旧选中框
    $('.phase-block').removeClass('selected')
    // 添加新选中框
    let phase_block_id = '#' + view_id('phase-block', inforStore.cur_phase_id)
    $(phase_block_id).addClass('selected')
    $('.feature-panel').css('display', 'block')
  }
})

const sort_attrs = ref(['Temporal', 'life_span', 'mean_intensity', 'max_value', 'mean_focus_grids', 'phase_POD', 'phase_FAR', 'phase_RMSE'])
const cur_sort_attr = ref('Temporal')
const orders = ref(['Ascending', 'Descending'])
const cur_order = ref('Ascending')
// const temporal_attrs = ref(['Pollution_cnt', 'Temporal_intensity'])
// const cur_temporal_attr = ref(['time_pollution_cnt'])

const sortPhases = (attr, order, cur_ids) => {
  let attr_data
  if (attr == 'phase_POD' || attr == 'phase_FAR' || attr == 'phase_RMSE')
    attr_data = inforStore.st_phase_events.phases[inforStore.cur_focused_model].map(item => item[attr])
  else if (attr == 'Temporal')
    attr_data = inforStore.st_phase_events.phases[inforStore.cur_focused_model].map(item => item.focus_phase_id)
  else if (attr == 'Multi_Accuracy')
    attr_data = inforStore.st_phase_events.phases[inforStore.cur_focused_model].map(item => item.multi_accuracy)
  // else if (attr == 'Focus_Level')
  //   attr_data = inforStore.cur_phase_indicators.map(item => item.mean_level)

  let pollution_attrs = ['life_span', 'mean_intensity', 'max_value', 'mean_focus_grids']
  if (pollution_attrs.includes(attr)) {
    attr_data = inforStore.st_phase_events.phases[inforStore.cur_focused_model].map(item => item[attr])
  }

  cur_ids.sort(function (a, b) {
    if (order == 'Ascending')
      return attr_data[a] - attr_data[b]
    else 
      return attr_data[b] - attr_data[a]
  })
  
  // for (let i = 0; i < inforStore.st_phase_events.phases[inforStore.cur_focused_model].length; ++i) {
  //   let svg_id = '#' + phaseSvgID(i)
  //   d3.select(svg_id).selectAll("*").remove()
  // }
  return cur_ids
}

const onSortAttrChange = (index) => {
  cur_sort_attr.value = sort_attrs.value[index]
  inforStore.sorted_phase_ids = sortPhases(cur_sort_attr.value, cur_order.value, inforStore.sorted_phase_ids)
}
const onOrderChange = (index) => {
  cur_order.value = orders.value[index]
  inforStore.sorted_phase_ids = sortPhases(cur_sort_attr.value, cur_order.value, inforStore.sorted_phase_ids)
}

function drawCntLegend() {
  d3.select('#cnt-legend').selectAll('*').remove()
  let margin_left = 20, margin_right = 10, margin_top = 0, margin_bottom = 0
  let main_w = 120, main_h = 12
  let svg_w = main_w + margin_left + margin_right
  let svg_h = main_h + margin_top + margin_bottom
  let legend_svg = d3.select('#cnt-legend')
    .attr('width', svg_w)
    .attr('height', svg_h)
  let legend = legend_svg.append('g')
    .attr('transform', `translate(${margin_left}, ${margin_top})`)
  // let legendScale = d3.scaleLinear()
  //     .domain([0, main_w])
      //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
      //原来：
      // .c([valColorScheme_blue[0], valColorScheme_blue[valColorScheme_blue.length - 1]])
      // .range(['#fff','#4a1486'])
  let legendScale = d3.scaleQuantize()
    .domain([0, main_w])
    .range(['#edf8e9', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#005a32'])
  
    //   .domain([0, inforStore.err_abs_extreme_th])

  legend.selectAll('rect')
    .data(Array(main_w).fill(1))
    .join('rect')
      .attr('x', (d,i) => i)
      .attr('y', 0)
      .attr('width', 1)
      .attr('height', 12)
      .attr('fill', (d,i) => legendScale(i))
  legend.append('text')
    .attr('x', -4)
    .attr('y', 11)
    .attr('text-anchor', 'end')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('0')
  legend.append('text')
    .attr('x', main_w+4)
    .attr('y', 11)
    .attr('text-anchor', 'start')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('1')
}

function drawErrLegend() {
  d3.select('#err-legend').selectAll('*').remove()
  let margin_left = 24, margin_right = 28, margin_top = 2, margin_bottom = 2
  let main_w = 120, main_h = 12
  let svg_w = main_w + margin_left + margin_right
  let svg_h = main_h + margin_top + margin_bottom
  let legend_svg = d3.select('#err-legend')
    .attr('width', svg_w)
    .attr('height', svg_h)

 //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  let stack_colors = ['#b2182b','#d6604d','#f4a582','#fddbc7','#e0e0e0','#bababa','#878787','#4d4d4d']
  let legend = legend_svg.append('g')
    .attr('transform', `translate(${margin_left}, ${margin_top})`)
  let legendScale = d3.scaleQuantize()
    .domain([0, stack_colors.length])
    .range(stack_colors)

  legend.selectAll('rect')
    .data(d3.range(0, stack_colors.length+1))
    .join('rect')
      .attr('x', (d,i) => i*12+4)
      .attr('y', 0)
      .attr('width', 12)
      .attr('height', 12)
      .attr('fill', (d,i) => legendScale(i))

  legend.append('text')
    .attr('x', 0)
    .attr('y', 10)
    .attr('text-anchor', 'end')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text(`low`)
  legend.append('text')
    .attr('x', main_w-3)
    .attr('y', 10)
    .attr('text-anchor', 'start')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text(`high`)
}

const view_id = (view_str, view_id) => `${view_str}_${view_id}`

const phase_update_flag = (item) => {
  if (inforStore.cur_phase_indicators.length > 0) {
    return inforStore.cur_phase_indicators[item].ACC
  } else {
    return 0.000001
  }
}

const drawPredFlag = () => {
  return Object.keys(inforStore.phases_indicators).length > 0
}

const pcp_attrs_ready = () => {
  return Object.keys(inforStore.phase_pcp_dims).length > 0
}

let viewed_shown = ref(false)
let unviewed_shown = ref(false)
let phase_collection_shown = ref(false)

function showViewed() {
  viewed_shown.value = !viewed_shown.value
}

function showUnviewed() {
  unviewed_shown.value = !unviewed_shown.value
}

function showPhaseCollection() {
  phase_collection_shown.value = !phase_collection_shown.value
}

watch (() => viewed_shown.value, (oldValue, newValue) => {
  if (viewed_shown.value) {
    inforStore.sorted_phase_ids = inforStore.browsed_phases
  } else {
    let tmp_indices = deepCopyArray(inforStore.cur_filtered_phases_indices)
    inforStore.sorted_phase_ids = sortPhases(cur_sort_attr.value, cur_order.value, tmp_indices)
  }
})

watch (() => unviewed_shown.value, (oldValue, newValue) => {
  if (unviewed_shown.value) {
    inforStore.sorted_phase_ids = inforStore.cur_filtered_phases_indices.filter(item => !inforStore.browsed_phases.includes(item))
  } else {
    let tmp_indices = deepCopyArray(inforStore.cur_filtered_phases_indices)
    inforStore.sorted_phase_ids = sortPhases(cur_sort_attr.value, cur_order.value, tmp_indices)
  }
})

watch (() => phase_collection_shown.value, (oldValue, newValue) => {
  if (phase_collection_shown.value) {
    inforStore.sorted_phase_ids = inforStore.phase_collections
  } else {
    let tmp_indices = deepCopyArray(inforStore.cur_filtered_phases_indices)
    inforStore.sorted_phase_ids = sortPhases(cur_sort_attr.value, cur_order.value, tmp_indices)
  }
})

function savePhaseCollection() {
  getData(inforStore, 'save_phase_collections', inforStore.cur_sel_data, JSON.stringify(inforStore.phase_collections), JSON.stringify(inforStore.dataset_configs))
}

</script>

<template>
  <div class="models-container">
    <div class="title-layer">
      <div class="title">Phase View</div>
      <!-- <div class="params-control">
        <label class="form-label"><span class="attr-title">Min_Length: </span></label>
        <input class="form-control" type="text" v-model="inforStore.phase_params.min_length"> 
      </div> -->
      <div class="data-dropdown">
        <label class="form-label"><span class="attr-title">Sorted by: </span></label>
        <button class="btn dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false">{{ cur_sort_attr }}</button>
        <ul class="dropdown-menu">
          <li v-for="(item, index) in sort_attrs" :value="item" @click="onSortAttrChange(index)" class='dropdown-item' :key="index">
            <div class="li-data-name">{{ item }}</div>
          </li>
        </ul>
      </div>
      <div class="data-dropdown">
        <label class="form-label"><span class="attr-title">Order: </span></label>
        <button class="btn dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false">{{ cur_order }}</button>
        <ul class="dropdown-menu">
          <li v-for="(item, index) in orders" :value="item" @click="onOrderChange(index)" class='dropdown-item' :key="index">
            <div class="li-data-name">{{ item }}</div>
          </li>
        </ul>
      </div>
      <!-- <div class="data-dropdown">
        <label class="form-label"><span class="attr-title">Temporal_Attr: </span></label>
        <button class="btn dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false">{{ cur_order }}</button>
        <ul class="dropdown-menu">
          <li v-for="(item, index) in orders" :value="item" @click="onOrderChange(index)" class='dropdown-item' :key="index">
            <div class="li-data-name">{{ item }}</div>
          </li>
        </ul>
      </div> -->

      <div class="collection-control" style="font-weight:700" @click="showViewed()" :showing="viewed_shown">
        <span class="iconfont" style="font-size: 20px; margin-right: 3px;">&#xe624;</span> <span>Viewed ({{inforStore.browsed_phases.length}})</span> 
      </div>

      <div class="collection-control" style="font-weight:700" @click="showUnviewed()" :showing="unviewed_shown">
        <span class="iconfont" style="font-size: 20px; margin-right: 3px;">&#xe626;</span> <span>Unviewed ({{inforStore.cur_filtered_phases.length - inforStore.browsed_phases.length}})</span> 
      </div>

      <div class="collection-control" style="font-weight:700" @click="showPhaseCollection()" :showing="phase_collection_shown">
        <span class="iconfont" style="font-size: 22px; margin-right: 3px;">&#xe625;</span> <span>Collection ({{inforStore.phase_collections.length}})</span> 
      </div>
      <div class="iconfont" id='save-subgroup-collection' @click="savePhaseCollection()">&#xe63c;</div>

      <div class="data-dropdown">
        <label class="form-label"><span class="attr-title">Property Interval: </span></label><svg class="phase-legend" id="err-legend" width="10" height="10"></svg>
      </div>
      <div class="data-dropdown">
        <label class="form-label"><span class="attr-title">Focus_Val: </span></label><svg class="phase-legend" id="cnt-legend" width="10" height="10"></svg>
      </div>
    </div>

    <div class="phases-module">
      <PhasesPCP v-if="pcp_attrs_ready()" />
      <div class="phases-container">
        <div v-for="(item, index) in inforStore.sorted_phase_ids" class="phase-block" :id="view_id('phase-block', index)" :key="index">
          <PhaseCard :phase_id="item" :cur_phase_index="index" :phase_data="inforStore.st_phase_events.phases[inforStore.cur_focused_model][item]" />
        </div>
      </div>
    </div>
    
    <!-- <EventsPCP /> -->
    <!-- <PhaseEventSeq /> -->
    <!-- <EventsTest /> -->
  </div>

</template>

<style scoped>
.models-container {
  /* width: 1080px; */
  width: 1604px;
  height: 380px;
  border: solid 1px #c2c5c5;
  border-radius: 6px;
  /* padding: 1px; */
  margin: 2px;
  overflow-y: auto;
}

.title-layer {
  /* position: absolute; */
  z-index: 80;
  width: 1580px;
  height: 20px;
  text-align: left;
  padding-left: 12px;
  /* background-color: #6c757d; */
  /* color: #fff; */
  margin-top: 10px;
  margin-bottom: 10px;
  /* font: 700 16px "Microsort Yahei"; */
  display: flex;
  align-items: center;
  /* justify-content: space-between; */
}

.title {
  font: 700 20px "Arial";
  /* letter-spacing: 1px; */
  color: #333;
}

.params-control {
  display: flex;
  margin-left: 20px;
}

.params-control .form-control {
  width: 40px;
  height: 20px !important;
  /* width: 120px; */
  padding: 0px 0px !important;
  margin-left: 4px;
  border: none;
  border-bottom: solid 1px #9c9c9c;
  border-radius: 0;
  font-size: 14px;
  text-align: center;
  color:#1a73e8;
  /* text-align: left; */
  /* overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis; */
}

.form-label {
  margin-bottom: 0;
  font-weight: 700;
}

.data-dropdown {
  margin-left: 20px;
}

.data-dropdown .dropdown-toggle {
  width: 120px !important;
  height: 30px;
  padding: 0px 2px -30px 0px !important;
  margin-bottom: 10px;
  border-bottom: solid 1px #9c9c9c;
  color:#1a73e8;
  border-radius: 0;
  font-size: 14px;
  /* text-align: left; */
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.dropdown-toggle::after {
    margin-left: 0.6em !important;
}

.dropdown-item {
  border-bottom: solid 1px #cecece;
  font-size: 14px;
  max-width: 280px;
  cursor: pointer;
  white-space: normal;
}

.dropdown-item:hover {
  background-color: #cecece;
}

.li-data-name {
    font-size: 14px;
}

.li-data-description {
    font-size: 12px;
    color: #777;
}

.phases-container {
  /* margin-left: 10px; */
  margin-top: -2px;
  white-space: nowrap;
  overflow-x: auto;
}

.phase-block {
  display: inline-block;
  margin: 2px;
  /* border: solid 1px #999; */
}

.phase-block:hover {
  cursor: pointer;
}

.selected {
  border: solid 2px #1a73e8;
}

.collection-control {
  margin-left: 24px;
  display: flex;
  align-items: center;
}

.collection-control[showing=true] {
  color: #0097A7;
}

#save-subgroup-collection:hover,
.collection-control:hover {
  cursor: pointer;
  /* color: #0097A7; */
  color: #1a73e8;
}
#save-subgroup-collection {
  font-size: 20px;
  margin-left:8px;
  font-weight: 700;
}
</style>