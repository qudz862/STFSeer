<script setup>
import { ref, reactive, computed, watch, onMounted, onUpdated } from "vue"
import { useInforStore } from '@/stores/infor.js'
import getData from '@/services/index.js'
import * as d3 from 'd3'
import { valColorScheme_blue, valColorScheme_red, valColorScheme_fire } from '@/data/index.js'
import EventsPCP from './EventsPCP.vue'

const inforStore = useInforStore()

let cur_sel_record = ref(-1)

function selEventRecord(event_index) {
  if (cur_sel_record.value == event_index) {
    cur_sel_record.value = -1
    inforStore.sel_event_record = -1
  }
  else {
    cur_sel_record.value = event_index
    inforStore.sel_event_record = event_index
  }
  console.log('inforStore.sel_event_record', inforStore.sel_event_record);
}

</script>

<template>
  <div style="display: flex; ">
    <div v-for="(item, index) in inforStore.event_records[`${inforStore.cur_phase_sorted_id}`]" :key="index"  @click="selEventRecord(index)" class="record-container" :chosen="cur_sel_record==index">
      <div>Notes: {{ item.notes }}</div>
      <div class="horizontal-line"></div>
      <div class="event-record">
        <div>
          <div>timestamp: {{ item.left.timestamp }}</div>
          <div>model: {{ item.left.model}}</div>
          <div>type: {{ item.left.type}}</div>
          <div v-if="item.left.type=='feature'">features: {{ item.left.features}}</div>
          <div v-if="item.left.type=='error'">features: {{ item.left.focus_metric}}</div>
        </div>
        <div class="seg-line"></div>
        <div>
          <div>timestamp: {{ item.right.timestamp}}</div>
          <div>model: {{ item.right.model}}</div>
          <div>type: {{ item.right.type}}</div>
          <div v-if="item.right.type=='feature'">features: {{ item.right.features}}</div>
          <div v-if="item.right.type=='error'">features: {{ item.right.focus_metric}}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.events-container {
  /* width: 1600px; */
  width: 1304px !important;
  height: 564px;
  border: solid 1px #c2c5c5;
  border-radius: 6px;
  /* padding: 1px; */
  margin: 2px;
  margin-left: 286px;
  /* overflow-y: auto; */
}

.title-layer {
  /* position: absolute; */
  z-index: 80;
  width: 700px;
  height: 20px;
  text-align: left;
  padding-left: 12px;
  /* background-color: #6c757d; */
  /* color: #fff; */
  margin-top: 10px;
  margin-bottom: 10px;
  /* font: 700 16px "Microsort Yahei"; */
  /* font: 700 20px "Arial"; */
  /* letter-spacing: 1px; */
  /* color: #333; */
  display: flex;
  align-items: center;
  /* justify-content: space-between; */
}

.title {
  font: 700 20px "Arial";
  /* letter-spacing: 1px; */
  color: #333;
}

.attr-title {
  font-weight: 700;
}

.seg-line {
  width: 1px;
  background-color: #cecece;;
  /* border: solid 1px #cecece; */
  margin-top: 6px;
  margin-bottom: 6px;
  margin-left: 6px;
  margin-right: 6px;
}

.record-container {
  border: solid 1px #bcbcbc;
  border-radius: 6px;
  margin-left: 6px;
  margin-right: 2px;
  margin-bottom: 8px;
  padding: 4px 6px;
  cursor: pointer;
  transition: box-shadow 0.3s ease; /* 添加过渡效果 */  
}

.record-container[chosen=true] {
  border: solid 1px #333;
}

.event-record {
  display: flex;
  /* justify-content: space-between; */
  /* align-items: center; */
}

.record-container:hover {
  box-shadow: 0 0 5px 2px rgba(170, 170, 170, 0.5); /* 鼠标滑过时的阴影 */
}

.horizontal-line {
  border-bottom: 1px solid #cecece; /* 分割线颜色和粗细 */
  margin-top: 3px; /* 可选，增加一些下边距以避免内容过于紧凑 */
  margin-bottom: 3px; /* 可选，增加一些下边距以避免内容过于紧凑 */
  /* margin-left: 2px;
  margin-right: 3px; */
}


</style>
