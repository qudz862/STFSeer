<script setup>
import { ref, reactive, computed, watch, onMounted, onUpdated } from "vue"
import { useInforStore } from '@/stores/infor.js'
import getData from '@/services/index.js'
import * as d3 from 'd3'
import { valColorScheme_blue, valColorScheme_red, valColorScheme_fire } from '@/data/index.js'
import EventsPCP from './EventsPCP.vue'
import EventRecords from './EventRecords.vue'

const inforStore = useInforStore()

</script>

<template>
  <div class="events-container">
    <div class="title-layer">
      <div class="title">Event Selection</div>
    </div>
    <div class="type-sel-region">
      <span class="attr-title">Evolution Types: </span>
      <div style="display: flex;">
        <div class="form-check" v-for="(item, index) in inforStore.event_types" :key="index" style="margin-left: 16px;">
          <input class="form-check-input" type="checkbox" :value="item" v-model="inforStore.sel_event_types">
          <label class="form-check-label" style="margin-left: -2px;"> {{ item }} </label>
        </div>
      </div>
    </div>
    <EventsPCP v-if="(Object.keys(inforStore.cur_event_pcp_dims).length != 0) && inforStore.cur_phase_id!=-1" />
    <div class="horizontal-line"></div>
    <EventRecords v-if="(`${inforStore.cur_phase_sorted_id}` in inforStore.event_records) && inforStore.event_records[`${inforStore.cur_phase_sorted_id}`].length > 0"/>
  </div>

</template>

<style scoped>
.events-container {
  /* width: 1600px; */
  width: 1304px !important;
  height: 474px;
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

.type-sel-region{
  margin-left: 14px;
  display: flex;
  margin-bottom: 10px;
}

.horizontal-line {
  border-bottom: 1px solid #cecece; /* 分割线颜色和粗细 */
  margin-top: 6px; /* 可选，增加一些下边距以避免内容过于紧凑 */
  margin-bottom: 6px; /* 可选，增加一些下边距以避免内容过于紧凑 */
  margin-left: 22px;
  margin-right: 22px;
}
</style>