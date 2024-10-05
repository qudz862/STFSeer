<script setup>
import Control from "./Control.vue"
import Subsets from "./Subsets.vue"
import TemporalPhases from "./TemporalPhases.vue"
import Exploration from "./Exploration.vue"
import InstanceDetail from './InstanceDetail.vue'
import EventsView from './EventsView.vue'
import { useInforStore } from '@/stores/infor.js'

const inforStore = useInforStore()
document.addEventListener('contextmenu', function(e){
	e.preventDefault();
})

function onCloseBtnClick() {
  inforStore.cur_phase_id = -1
  inforStore.cur_phase_sorted_id = -1
}
</script>

<template>
  <div class="global-container">
    <Control />
    <div>
      <div class="local-container">
        <Subsets v-show="inforStore.cur_interface_type == 'Subgroup'" />
        <TemporalPhases v-show="inforStore.cur_interface_type == 'Phase&Event'" />

      </div>
      <!-- <EventsView v-show="inforStore.cur_interface_type == 'Phase&Event'" /> -->
      <!-- <div v-show="(inforStore.cur_interface_type == 'Phase&Event') && (inforStore.cur_phase_id != -1)" id="overlay"></div>
      <div v-show="(inforStore.cur_interface_type == 'Phase&Event') && (inforStore.cur_phase_id != -1)" id="popup"> -->
        <div v-show="inforStore.cur_interface_type == 'Phase&Event'">
          <Exploration v-if="inforStore.cur_phase_id != -1" />
        </div>
        
        <!-- <button id="closeButton" class="iconfont" @click="onCloseBtnClick()">&#xe620;</button> -->
      <!-- </div> -->
    </div>
    <!-- <InstanceDetail /> -->
  </div>
</template>


<style>
.global-container {
  display: flex;
}

.local-container {
  display: flex;
}

#popup {
  position: fixed;           /* 固定定位，元素相对于浏览器窗口 */
  top: 50%;                  /* 距离顶部 50% */
  left: 50%;                 /* 距离左侧 50% */
  transform: translate(-50%, -50%); /* 使用 transform 让元素的中心点位于正中央 */
  background-color: white;   /* 设置背景色，通常弹出层背景为白色 */
  padding: 20px;             /* 内边距，增加弹出层的可视区域 */
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* 添加阴影使弹出层有悬浮感 */
  border-radius: 10px;       /* 圆角边框 */
  z-index: 100;             /* 设置 z-index 使弹出层在最上层 */
  text-align: center;        /* 居中对齐内容 */
}

#overlay {
  position: fixed;           /* 固定定位，覆盖整个屏幕 */
  top: 0;
  left: 0;
  width: 100vw;              /* 覆盖全屏宽度 */
  height: 100vh;             /* 覆盖全屏高度 */
  background-color: rgba(0, 0, 0, 0.5); /* 半透明背景 */
  z-index: 99;              /* 弹出层背景在弹出层下方 */
}

#closeButton {
  color: #333;
  font-size: 18px;
  border: none;
  background: none;

  position: absolute;
  top: 10px; /* 距离顶部 10px */
  right: 10px; /* 距离右边 10px */
  background-color: transparent; /* 透明背景 */
  border: none;
  font-size: 20px; /* 字体大小 */
  cursor: pointer; /* 鼠标悬停时变成指针 */
}

#closeButton:hover {
  color: #0097A7;
}
</style>
