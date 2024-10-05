<script setup>
import { ref, reactive, computed, watch, onMounted, onUpdated, getCurrentInstance } from "vue";
import getData from '@/services/index.js'
import { useInforStore } from '@/stores/infor.js'
import $ from 'jquery'

const inforStore = useInforStore()

// 地图相关设置
let scope_num

onMounted(() => {
  // draw_th_sketch()
  $('#get-model-button').on('click', () => {
    // getData(inforStore, 'cur_model_infor', inforStore.cur_sel_data, inforStore.cur_sel_model, JSON.stringify(inforStore.cur_sel_failure_rules), inforStore.cur_sel_scope_th)
  })
})

const onAnalysisModeClick = (cur_mode) => {
  inforStore.analysis_type = cur_mode
  
}

</script>

<template>
  <div class="config-seg-line"></div>
  <div class="sub-title"><span class="iconfont title-icon">&#xe60a;</span> Subset Mining Configs</div>
  
  <!-- 设置时间范围的阈值 -->
  <!-- <div class="th-form-row">
    <div class="params-control">
      <label class="form-label"><span class="attr-title">Time_Scope_Size: </span></label>
      <input class="form-control" type="text" v-model="inforStore.cur_sel_scope_th"> 
    </div>
  </div> -->
  
  <div class="th-form-row">
    <label class="form-label"><span class="attr-title">Analysis_Mode: </span></label>
    <div class="analysis-type-dropdown">
      <button class="btn dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false">{{ inforStore.analysis_type }}</button>
      <ul class="dropdown-menu" id="dropdown-choose-task">
        <li @click="onAnalysisModeClick('Error-Guided')" value="Error-Guided" class='dropdown-item'>
          <div class="li-data-name">Error-Guided</div>
          <div class="li-data-description">Mining subsets Guided by error variation.</div>
        </li>
        <li @click="onAnalysisModeClick('Purity-Guided')" value="Purity-Guided" class='dropdown-item'>
          <div class="li-data-name">Purity-Guided</div>
          <div class="li-data-description">Mining subsets Guided by purity variation.</div>
        </li>
      </ul>
    </div>
  </div>

  <div class="th-form-row">
    <div class="params-control">
      <label class="form-label"><span class="attr-title">Range_Sup_TH: </span></label>
      <input class="form-control" type="text" v-model="inforStore.range_params.sup_th"> 
    </div>
  </div>
  <div class="th-form-row">
    <div class="params-control">
      <label class="form-label"><span class="attr-title">Range_Step_Len: </span></label>
      <input class="form-control" type="text" v-model="inforStore.range_params.step_len"> 
    </div>
  </div>
  <div class="th-form-row">
    <div class="params-control">
      <label class="form-label"><span class="attr-title">Range_Var_TH: </span></label>
      <input class="form-control" type="text" v-model="inforStore.range_params.div_th">
    </div>
  </div>
  <div class="th-form-row">
    <div class="params-control">
      <label class="form-label"><span class="attr-title">Subset_Sup_TH: </span></label>
      <input class="form-control" type="text" v-model="inforStore.subset_params.frequent_sup_th">
    </div>
  </div>
  <div class="th-form-row">
    <div class="params-control">
      <label class="form-label"><span class="attr-title">Subset_Err_TH: </span></label>
      <input class="form-control" type="text" v-model="inforStore.subset_params.err_diff_th">
    </div>
  </div>
  <div class="data_infor_title">
    <button class="btn btn-primary" id="get-model-button">Error Exploration</button>
  </div>
  
  
</template>

<style scoped>
.sub-title {
  width: 200px;
  display: flex;
  align-items: center;
  margin-left: 10px;
  font-size: 15px;
  font-weight: 700;
  margin-top: 8px;
  margin-bottom: 4px;
}

.title-icon {
  display: flex;
  width: 24px;
  margin-top: -2px;
  justify-content: center;
  margin-right: 4px;
  font-weight: 400;
}

.th-form-row {
  width: 252px;
  height: 26px;
  margin-left: 30px;
  margin-bottom: 3px;
  display: flex;
  font-size: 14px;
  /* justify-content: center; */
  align-items: center;
}

.params-control {
  width: 180px;
  height: 24px;
  display: flex;
  align-items: center;
}
.params-control .form-control {
  width: 90px;
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

.th-form-row .attr-title {
  display: inline-block;
  width: 130px;
  font-weight: 700;
}
.form-label {
  margin-bottom: 0;
}

.config-seg-line {
  height: 1px;
  width: 274px;
  background-color: #bcbcbc;
  margin: 0 auto;
  margin-top: 8px;
  margin-bottom: 8px;
}

.data_infor_title {
  margin-top: 6px;
  margin-left: 10px;
  font-size: 14px;
  width: 270px;
  display: flex;
  justify-content: space-between;
}

#get-model-button {
  margin: 0 auto;
  margin-top: 1px;
  width: 220px;
  height: 32px;
  padding: 2px 0px;
  font-size: 14px;
  font-weight: 700;
  border: solid 1px #9a9a9a;
  border-radius: 16px;
  color: #333;
  background-color: #fff;
}

#get-model-button:hover {
  border-color: #1a73e8;
  color: #1a73e8;
}

.dropdown-toggle {
  width: 106px !important;
  /* height: 30px; */
  /* width: 120px; */
  padding: 0px 2px 0 2px !important;
  border-bottom: solid 1px #9c9c9c;
  border-radius: 0;
  font-size: 14px;
  color: #1a73e8;
  /* text-align: left; */
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.dropdown-toggle::after {
    margin-left: 0.3em !important;
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

.li-model-name,
.li-data-name {
    font-size: 14px;
}

.li-model-description,
.li-data-description {
    font-size: 12px;
    color: #777;
}

/* .config-dropdown {
  margin-right: 20px !important;
} */
</style>