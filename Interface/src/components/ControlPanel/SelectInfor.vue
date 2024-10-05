<script setup>
import { ref, onMounted, watch } from "vue"
import getData from '@/services/index.js'
import { useInforStore } from '@/stores/infor.js'
import DataInfor from './DataInfor.vue'
import ModelInfor from './ModelInfor.vue'
// import SubsetConfig from "./SubsetConfig.vue";

const inforStore = useInforStore()


onMounted(() => {
  
})

// const onTaskCardClick = card_index => {
//   if (card_index == -1) {
//     sel_task_id.value = -1
//     sel_task_name.value = "Choose Task"
//     cur_data_names.value = []
//     inforStore.cur_sel_task = ""
//   }
//   else {
//     if (sel_task_id.value != -1) {
//       sel_data_id.value = -1
//       sel_data_name = ref('Choose Data')
//       sel_model_id.value = -1
//       sel_model_names = ref('Choose Models')
//     }
//     sel_task_id.value = card_index
//     sel_task_name.value = Object.keys(inforStore.existed_task_data_model)[card_index]
//     inforStore.cur_sel_task = sel_task_name.value
//     cur_data_names.value = Object.keys(inforStore.existed_task_data_model[sel_task_name.value])
//   }
// }

// watch (() => inforStore.cur_model_parameters, (oldValue, newValue) => {
//   console.log(inforStore.cur_model_parameters);
// })

// watch (() => cur_model_names.value, (oldValue, newValue) => {
//   getData(inforStore, 'cur_model_infor', sel_task_name.value, sel_data_name.value, JSON.stringify(inforStore.cur_sel_time), JSON.stringify(cur_model_names.value))
// })

const onSelInterfaceType = type => {
  inforStore.cur_interface_type = type
}

</script>

<template>
  <div class='outer-container'>
    <div class="title-layer">
      <span>Control Panel</span>
      <div class="interface-dropdown">
        <button class="btn dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false">{{ inforStore.cur_interface_type }}</button>
        <ul class="dropdown-menu">
          <li v-for="(value, index) in inforStore.interface_type" :key="index" :value="value" @click="onSelInterfaceType(value)" class='dropdown-item'>
            <div class="li-task-name">{{ value }}</div>
          </li>
        </ul>
      </div>
    </div>
    <!-- <div class="select-block">
      <span class="iconfont task-icon select-icon">&#xe63f;</span>
      <div class="task-dropdown">
        <button class="btn dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false">{{ sel_task_name }}</button>
        <ul class="dropdown-menu" id="dropdown-choose-task">
          <li v-for="(key, index) in Object.keys(inforStore.existed_task_data_model)" :value="key" @click="onTaskCardClick(index)" class='dropdown-item'>
            <div class="li-task-name">{{ key }}</div>
          </li>
        </ul>
      </div>
    </div> -->

    <div class="data-infor">
      <DataInfor />
    </div>

    <ModelInfor />
  </div>
</template>

<style scoped>
.outer-container {
  width: 308px;
  height: 1076px;
  /* height: 1372px; */
  border: solid 1px #c2c5c5;
  border-radius: 6px;
  /* padding: 1px; */
  margin: 2px;
  overflow-y: auto;
}

.select-block {
  display: flex;
  margin-left: 10px;
}

.title-layer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  /* position: absolute; */
  z-index: 80;
  width: 302px;
  height: 20px;
  text-align: left;
  padding-left: 12px;
  /* background-color: #6c757d; */
  /* color: #fff; */
  margin-top: 10px;
  margin-bottom: 9px;
  /* font: 700 16px "Microsort Yahei"; */
  font: 700 20px "Arial";
  /* letter-spacing: 1px; */
  color: #333;
}

.interface-dropdown .dropdown-toggle{
  color: #1a73e8;
  /* margin-left: 40px; */
}

.dropdown-toggle::after {
    margin-left: 0.6em !important;
}

.dropdown-item {
  border-bottom: solid 1px #cecece;
  font-size: 14px;
  max-width: 480px;
  cursor: pointer;
  white-space: normal;
}

.dropdown-item:hover {
  background-color: #cecece;
}

.select-icon {
  display: flex;
  width: 24px;
  justify-content: center;
}

.sub-title {
  display: flex;
  /* font-size: 14px; */
  align-items: center;
}


</style>