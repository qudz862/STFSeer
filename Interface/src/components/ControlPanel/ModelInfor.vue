<script setup>
import { ref, reactive, computed, watch, onMounted, onUpdated } from "vue"
import { useInforStore } from '@/stores/infor.js'
import * as d3 from 'd3'
import { valColorScheme_blue, valColorScheme_red } from '@/data/index.js'
import getData from '@/services/index.js'
import $ from 'jquery'
import "ion-rangeslider/css/ion.rangeSlider.min.css"; // 引入样式
import "ion-rangeslider/js/ion.rangeSlider.min.js"; // 引入脚本
import { distance } from "turf"
import ModelCard from './ModelCard.vue'
import ErrorDistribution from './ErrorDistribution.vue'

const inforStore = useInforStore()

let err_setting = ref({})
let err_th = ref(0)
let scope_num
let focus_type = ref('Greater than')
let focus_th = ref(0)

let sel_start_time = ref("")
let sel_end_time = ref("")

let new_config_name = ref("Medium-range-segment")
let new_config_description = ref("")

let failure_config = ref([
  {val_min: "", val_max: "", err_th: ""}
])

let output_range
let rule_num = ref(1)

let cur_error_configs = ref({})
// let sel_error_config = ref("Fine-range-segment")

function add_button_click() {
  let index = $('.add_button').index(this);
  let new_rule
  if (index == failure_config.value.length - 1) {
    new_rule = {val_min: "", val_max: output_range.max, err_th: ""}
    failure_config.value[index].val_max = ""
  }
  else {
    new_rule = {val_min: failure_config.value[index].val_max, val_max: "", err_th: ""}
  }
  failure_config.value.splice(index+1, 0, new_rule);

  rule_num.value += 1
}

function sub_button_click() {
  let index = $('.sub_button').index(this);
  if (index == rule_num.value-1) {
    failure_config.value[rule_num.value-2].val_max = output_range.max
  }
  failure_config.value.splice(index, 1)
  failure_config.value[index].val_min = failure_config.value[index].val_max
  
  rule_num.value -= 1
}

watch (() => inforStore.cur_sel_window_size, (oldValue, newValue) => {
  let output_window = inforStore.cur_sel_window_size['Forecast Steps']
  let forecast_scope_config = inforStore.dataset_configs.forecast_scope
  let cur_forecast_scopes = []
  if (forecast_scope_config['scope_mode'] == 'range') {
    for (let start = 0; start < output_window; start += forecast_scope_config.step_unit) {
      let end = start + forecast_scope_config.step_unit
      if (end > output_window) end = output_window
      cur_forecast_scopes.push([start, end])
    }
  } else if (forecast_scope_config['scope_mode'] == 'sample') {
    if (forecast_scope_config['sample_method'] == 'linear') {
      let samples = Array.from({ length: forecast_scope_config.sample_num }, (_, i) => i * (output_window - 1) / (forecast_scope_config.sample_num - 1));
      cur_forecast_scopes = samples.map(sample => [Math.round(sample), Math.round(sample) + 1]);
    } else if (forecast_scope_config.sample_method === 'exponential') {
      let expIntervals = Array.from({ length: forecast_scope_config.sample_num }, (_, i) => (i / (forecast_scope_config.sample_num - 1)) ** 2);
      let samples = expIntervals.map(interval => interval * (output_window - 1));
      cur_forecast_scopes = samples.map(sample => [Math.round(sample), Math.round(sample) + 1]);
    }
  }
  inforStore.forecast_scopes = cur_forecast_scopes
  scope_num = inforStore.forecast_scopes.length
})

onUpdated(() => {
  // console.log('updated!', rule_num.value);
  $('.add_button').off('click')
  $('.add_button').on('click', add_button_click)

  $('.sub_button').off('click')
  $('.sub_button').on('click', sub_button_click)
})

const tmp_focused_scope = ref('')
watch (() => inforStore.cur_sel_window_size, (oldValue, newValue) => {
  tmp_focused_scope.value = `1-${inforStore.cur_sel_window_size['Forecast Steps']}`
})

watch (() => inforStore.cur_focused_scope, (oldValue, newValue) => {
  // 改变关注scope后，需要重新计算phases_indicators
  // getData(inforStore, 'phases_indicators',  inforStore.cur_sel_data, inforStore.cur_baseline_model, JSON.stringify(inforStore.dataset_configs.focus_levels), JSON.stringify(inforStore.dataset_configs.event_params), JSON.stringify(inforStore.cur_focused_scope))
  // 需要重新计算phase_details
  // getData(inforStore, 'phase_details', inforStore.cur_sel_model, inforStore.cur_phase_sorted_id, JSON.stringify(inforStore.dataset_configs.focus_levels), inforStore.dataset_configs.focus_th, JSON.stringify(inforStore.cur_focused_scope))
})

watch (() => inforStore.cur_data_infor, (oldValue, newValue) => {
  cur_error_configs.value = inforStore.error_configs[inforStore.cur_sel_task]
  output_range = inforStore.cur_data_infor.threshold.output_range
  failure_config.value[0].val_min = output_range.min
  failure_config.value[0].val_max = output_range.max
  err_setting.value = inforStore.cur_data_infor.threshold.err_th
  // err_th.value = err_setting.value.default
  // inforStore.cur_sel_err_th = err_setting.value.default
  focus_type.value = inforStore.cur_data_infor.threshold.focus_type_default
  focus_th.value = inforStore.cur_data_infor.threshold.focus_th_default
  inforStore.sel_error_config = "Medium-range-segment"
})

onMounted(() => {
  
})

function getModelInfor(model_name) {
  let model_type = model_name.split('-')[0]
  // console.log(model_type);
  return inforStore.model_infor[model_type].Introduction
}

const onForeStepSel = (index) => {
  inforStore.cur_sel_window_size = inforStore.cur_window_sizes[index]
  let cur_key = Object.keys(inforStore.existed_task_data_model[inforStore.cur_sel_task][inforStore.cur_sel_data].forecast_steps)[index]
  inforStore.cur_model_names = inforStore.existed_task_data_model[inforStore.cur_sel_task][inforStore.cur_sel_data].forecast_steps[cur_key]

  getData(inforStore, 'input_output_data', inforStore.cur_sel_data, inforStore.cur_sel_window_size['Input Steps'], inforStore.cur_sel_window_size['Forecast Steps'], inforStore.cur_sel_window_size['Lead Steps'])
}

const parseForecastSteps = (window_obj) => {
  if (window_obj == 'Choose Forecast Steps') return window_obj
  else return `Input: ${window_obj['Input Steps']};  Forecast: ${window_obj['Forecast Steps']}; Lead: ${window_obj['Lead Steps']}`
}

const getModelType = model_name => model_name.split('-')[0]
const onModelCardClick = (model_name, type) => {
  if (type == 'focused') {
    inforStore.cur_focused_model = model_name
    inforStore.cur_sel_models[0] = model_name
    if (!(model_name in inforStore.subset_collections)) {
      inforStore.subset_collections[model_name] = []
    }
  }
  if (type == 'baseline') {
    inforStore.cur_baseline_model = model_name
    inforStore.cur_sel_models[1] = model_name
    if (!(model_name in inforStore.subset_collections)) {
      inforStore.subset_collections[model_name] = []
    }
  }
}

const onFocusedScopeChange = () => {
  let scope_start = parseInt(tmp_focused_scope.value.split('-')[0])
  let scope_end = parseInt(tmp_focused_scope.value.split('-')[1])
  inforStore.cur_focused_scope = [scope_start, scope_end]
}

watch (() => inforStore.model_err_mtx, (oldValue, newValue) => {
  drawErrorMatrix()
})

function drawErrorMatrix() {
  let err_mtx = inforStore.model_err_mtx.err_mtx
  let err_mtx_vals = Object.values(err_mtx)
  // console.log(err_mtx_vals);
  let residual_bin_edges_all = inforStore.model_err_mtx.residual_bin_edges_all
  let outlier_pos_cnts = inforStore.model_err_mtx.outlier_pos_cnts
  let outlier_neg_cnts = inforStore.model_err_mtx.outlier_neg_cnts
  let outlier_pos_cnts_vals = Object.values(outlier_pos_cnts)
  let outlier_neg_cnts_vals = Object.values(outlier_neg_cnts)
  let err_config = inforStore.cur_sel_failure_rules
  d3.select('#model-err-mtx').selectAll("*").remove()
  let margin_left = 10, margin_right = 10, margin_top = 10, margin_bottom = 10
  let main_w = 260
  let cell_w = main_w / (scope_num)
  let cell_h = 50
  let main_h = (inforStore.dataset_configs.focus_levels.length-1) * cell_h
  let svg_w = main_w + margin_left + margin_right
  let svg_h = main_h + margin_top + margin_bottom
  let svg = d3.select('#model-err-mtx')
    .attr('width', svg_w)
    .attr('height', svg_h)
    .append("g")
    .attr("transform", `translate(${margin_left},${margin_top})`)
  
  let glyph_cells = svg.append('g').attr('class', 'glyph-cells')
  // let radiusScale = d3.scaleLinear()
  //   .domain([-global_err_max_abs, global_err_max_abs])
  //   .range([0, row_h*0.45])
  // const pie = d3.pie()
  //   .value(d => 1)
  //   .sort(null);
  const model_color = d3.scaleOrdinal(d3.schemeCategory10);
  let outlier_w = 10
  for (let i = 0; i < inforStore.dataset_configs.focus_levels.length-1; ++i) {
    for (let j = 0; j < scope_num; ++j) {
      let xScale = d3.scaleLinear()
        .domain([-err_config[i].err_th, err_config[i].err_th])
        .range([outlier_w, cell_w-outlier_w])
      let glyph_cell = glyph_cells.append('g')
        .attr('class', `glyph-cell-${i}-${j}`)
        .attr('transform', `translate(${j*cell_w}, ${i*cell_h})`)
      glyph_cell.append('rect')
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', cell_w)
        .attr('height', cell_h)
        .attr('fill', 'none')
        .attr('stroke', '#cecece')
      // 绘制box plot试试
      let cell_max_cnt = 0
      for (let k = 0; k < err_mtx_vals.length; ++k) {
        let cur_err_hist = err_mtx_vals[k][i][j]
        let cur_outlier_pos_cnts = outlier_pos_cnts_vals[k][i][j]
        let cur_outlier_neg_cnts = outlier_neg_cnts_vals[k][i][j]
        let cur_max_cnt = Math.max(...cur_err_hist, cur_outlier_pos_cnts, cur_outlier_neg_cnts)
        if (cur_max_cnt > cell_max_cnt) cell_max_cnt = cur_max_cnt
      }
      for (let k = 0; k < err_mtx_vals.length; ++k) {
        let cur_err_hist = err_mtx_vals[k][i][j]
        let yScale = d3.scaleLinear()
          .domain([0, cell_max_cnt])
          .range([cell_h, 2])
        let hist_area = d3.area()
          .curve(d3.curveMonotoneX)
          .x((d,index) => xScale(residual_bin_edges_all[i][index]))
          .y0(cell_h)
          .y1((d,i) => yScale(d))
        let hist_line = d3.line()
          .curve(d3.curveMonotoneX)
          .x((d,index) => xScale(residual_bin_edges_all[i][index]))
          .y((d,i) => yScale(d))
        // 用路径绘制直方图
        // glyph_cell.append("path")
        //   .datum(cur_err_hist)
        //   .attr("class", "histogram")
        //   .attr("d", hist_area)
        //   .attr('fill', model_color(k))
        //   .attr('opacity', 0.4)
        glyph_cell.append("path")
          .datum(cur_err_hist)
          .attr("class", "histogram")
          .attr("d", hist_line)
          .attr('stroke', model_color(k))
          .attr('fill', 'none')
        glyph_cell.append('line')
          .attr('x1', xScale(0))
          .attr('x2', xScale(0))
          .attr('y1', cell_h)
          .attr('y2', 2)
          .attr('stroke', 'red')
          .attr('stroke-width', 1)
          // .attr('opacity', 0.4)
        glyph_cell.append('line')
          .attr('x1', 0)
          .attr('x2', outlier_w)
          .attr('y1', yScale(outlier_neg_cnts_vals[k][i][j]))
          .attr('y2', yScale(outlier_neg_cnts_vals[k][i][j]))
          .attr('stroke', model_color(k))
          .attr('stroke-width', 2)
        glyph_cell.append('line')
          .attr('x1', cell_w-outlier_w)
          .attr('x2', cell_w)
          .attr('y1', yScale(outlier_pos_cnts_vals[k][i][j]))
          .attr('y2', yScale(outlier_pos_cnts_vals[k][i][j]))
          .attr('stroke', model_color(k))
          .attr('stroke-width', 2)
      }
    }
  }
}

const onRulesChange = () => {
  // console.log(failure_config.value);
  inforStore.cur_sel_failure_rules = failure_config.value
}

const max_val_input = index => {
  failure_config.value[index+1].val_min = failure_config.value[index].val_max
  inforStore.cur_sel_failure_rules = failure_config.value
}

const onErrorConfigClick = key => {
  inforStore.sel_error_config = key
  new_config_name.value = key
}

const onSaveConfigClick = () => {
  getData(inforStore, 'save_error_config', inforStore.cur_sel_task, new_config_name.value, encodeURI(new_config_description.value), JSON.stringify(failure_config.value), inforStore.cur_sel_scope_th)
}

const onConfigNameInput = () => {
  // console.log(new_config_name.value);
}

const getModels = () => {
  // 获取模型参数
  if (inforStore.cur_baseline_model.length > 0) getData(inforStore, 'model_parameters', inforStore.cur_sel_data, JSON.stringify(inforStore.cur_sel_models), inforStore.cur_focused_model, inforStore.cur_baseline_model)
  else getData(inforStore, 'model_parameters', inforStore.cur_sel_data, JSON.stringify(inforStore.cur_sel_models), inforStore.cur_focused_model, 'none')
}

watch (() => inforStore.model_parameters, (oldValue, newValue) => {
  // 获取模型的err_mtx
  // getData(inforStore, 'model_err_mtx', inforStore.cur_sel_data, JSON.stringify(inforStore.cur_sel_models), JSON.stringify(inforStore.cur_sel_failure_rules), inforStore.cur_sel_scope_th)
  getData(inforStore, 'process_model_preds', inforStore.cur_sel_data, JSON.stringify(inforStore.cur_sel_models), inforStore.dataset_configs.focus_th, JSON.stringify(inforStore.dataset_configs.focus_levels))
  
  
  // 获取模型在各阶段的指标
  // getData(inforStore, 
  //   'phases_indicators',
  //   inforStore.cur_sel_data, 
  //   JSON.stringify(inforStore.cur_sel_models), 
  //   JSON.stringify(inforStore.dataset_configs.focus_levels), 
  //   JSON.stringify(inforStore.dataset_configs.event_params), 
  //   JSON.stringify(inforStore.cur_focused_scope))
})

watch (() => inforStore.process_preds_state, (oldValue, newValue) => {
  getData(inforStore, 'error_distributions',  inforStore.cur_sel_data, JSON.stringify(inforStore.cur_sel_models), JSON.stringify(inforStore.dataset_configs), JSON.stringify(inforStore.forecast_scopes), 'evaluation')
})

watch (() => inforStore.error_distributions, (oldValue, newValue) => {
  getData(inforStore, 'st_phase_events', inforStore.cur_sel_data, JSON.stringify(inforStore.cur_sel_models), inforStore.dataset_configs.focus_th, JSON.stringify(inforStore.dataset_configs.phase_params), JSON.stringify(inforStore.dataset_configs.focus_levels), JSON.stringify(inforStore.dataset_configs.event_params), JSON.stringify(inforStore.forecast_scopes))
  // 获取所有阶段的子集
  if (inforStore.cur_baseline_model.length > 0) {
    getData(inforStore, 'find_point_slices', inforStore.cur_sel_data, inforStore.cur_baseline_model, inforStore.cur_focused_model, JSON.stringify(inforStore.model_parameters[inforStore.cur_focused_model]), JSON.stringify(inforStore.forecast_scopes))
  } else {
    getData(inforStore, 'find_point_slices', inforStore.cur_sel_data, 'none', inforStore.cur_focused_model, JSON.stringify(inforStore.model_parameters[inforStore.cur_focused_model]), JSON.stringify(inforStore.forecast_scopes))
  }
  
})

watch (() => inforStore.cached_slice_id, (oldValue, newValue) => {
  getData(inforStore, 'slices_indices', inforStore.cur_sel_data, inforStore.cur_focused_model, inforStore.cached_slice_id)
})

</script>

<template>
  <!-- <div class="params-control">
    <label class="form-label"><span class="attr-title">Multi_Levels: </span></label>
    <input class="form-control" id="form-multi-level" type="text" v-model="inforStore.dataset_configs.focus_levels">
  </div> -->
  <!-- <div class="err-plot-title">
    <span>Focused Scope</span>
  </div> -->
  <!-- <div @click="onModelCardClick('compare')">Compare Models</div> -->

  <!-- model card中展示的信息该是什么呢？再想想~ -->
  <!-- <div v-if="Object.keys(inforStore.multi_step_err_infor).length > 0">
    <ModelCard v-for="(value, key, index) in inforStore.multi_step_err_infor" :key="index" :model_name="key" :model_type="getModelType(key)" :step_error="value" :model_parameters="inforStore.model_parameters[key]" @click="onModelCardClick(key)" />
  </div> -->
  <div class="select-block" style="margin-top: -2px">
    <span class="iconfont model-icon select-icon">&#xe6b5;</span>
    <div class="data-dropdown">
      <button class="btn dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false">{{ parseForecastSteps(inforStore.cur_sel_window_size) }}</button>
      <ul class="dropdown-menu">
        <li v-for="(item, index) in inforStore.cur_window_sizes" :value="item" @click="onForeStepSel(index)" class='dropdown-item' :key="index">
          <div class="li-data-name"><span>Input Steps: {{item['Input Steps']}}</span>; <span>Forecast Steps: {{item['Forecast Steps']}}</span>; <span>Lead Steps: {{ item['Lead Steps'] }}</span></div>
        </li>
      </ul>
    </div>
  </div>
  <div class="select-block" style="margin-top: -6px">
    <span class="iconfont model-icon select-icon">&#xe7ee;</span>
    <div class="data-dropdown">
      <button class="btn dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false">
        <span v-if="inforStore.cur_focused_model.length == 0">Select Focused Model</span>
        <span v-else>{{ inforStore.cur_focused_model }}</span>
      </button>
      <ul class="dropdown-menu">
        <li v-for="(item, index) in inforStore.cur_model_names" :value="item" @click="onModelCardClick(item, 'focused')" class='dropdown-item' :key="index">
          <div class="li-data-name">{{ item }}</div>
          <div class="li-data-description">{{ getModelInfor(item) }}</div>
        </li>
      </ul>
    </div>
  </div>

  <div class="select-block" style="margin-top: -6px">
    <span class="iconfont model-icon select-icon">&#xe7ee;</span>
    <div class="data-dropdown">
      <button class="btn dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false">
        <span v-if="inforStore.cur_baseline_model.length == 0">Select Baseline Model</span>
        <span v-else>{{ inforStore.cur_baseline_model }}</span>
      </button>
      <ul class="dropdown-menu">
        <li v-for="(item, index) in inforStore.cur_model_names" :value="item" @click="onModelCardClick(item, 'baseline')" class='dropdown-item' :key="index">
          <div class="li-data-name">{{ item }}</div>
          <div class="li-data-description">{{ getModelInfor(item) }}</div>
        </li>
      </ul>
    </div>
  </div>
  <div class="select-block" style="margin-top: -6px">
    <button class="btn btn-primary" id="get-model-button" @click="getModels()">Explore Model Performance</button>
  </div>
  <!-- <div class="config-seg-line"></div> -->
  <!-- <div class="params-control">
    <label class="form-label"><span class="attr-title">Focused_Scope: </span></label>
    <input class="form-control" id="form-focused-scope" type="text" v-model="tmp_focused_scope" @change="onFocusedScopeChange()">
  </div>
  <div class="params-control">
    <label class="form-label"><span class="attr-title">Focus_Target_TH: </span></label>
    <input class="form-control" id="form-target-th" type="text" v-model="inforStore.dataset_configs.focus_th">
  </div> -->
  <!-- <div v-if="inforStore.cur_focused_model.length > 0">
    <ModelCard :model_name="inforStore.cur_focused_model" :model_type="getModelType(inforStore.cur_focused_model)" :step_error="value" :model_parameters="inforStore.model_parameters[key]" />
  </div>
  <div v-if="inforStore.cur_baseline_model.length > 0">
    <ModelCard :model_name="inforStore.cur_focused_model" :model_type="getModelType(inforStore.cur_baseline_model)" :step_error="value" :model_parameters="inforStore.model_parameters[key]" />
  </div> -->
  <!-- <svg id="model-err-mtx"></svg> -->
  <ErrorDistribution />
  
</template>

<style scoped>
.select-block {
  display: flex;
  margin-left: 10px;
  /* margin-bottom: -3px; */
  /*margin-top: -3px; */
}

.model-icon {
  font-size: 26px;
  margin-right: 6px;
}

.select-icon {
  display: flex;
  width: 24px;
  justify-content: center;
}

.params-control {
  display: flex;
  margin-left: 10px;
  margin-bottom: 8px;
}

.params-control .form-control {
  width: 120px;
  height: 20px !important;
  /* width: 120px; */
  padding: 0px 0px !important;
  margin-left: 8px;
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

#form-multi-level {
  width: 182px;
}
#form-target-th {
  width: 80px;
}
#form-focused-scope {
  width: 80px;
}

.data-dropdown .dropdown-toggle {
  color: #1a73e8;
  width: 240px !important;
  height: 24px !important;
  /* width: 120px; */
  padding: 0px 2px 0 4px !important;
  border-bottom: solid 1px #9c9c9c;
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
  max-width: 480px;
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

#err-dis {
  display: block;
  margin-left: 10px;
}

.err-plot-title {
  margin-left: 10px;
  margin-right: 10px;
  font-weight: 700;
  display: flex;
  justify-content: space-between;
}

.focus-span {
  color: #1a73e8;
  font-weight: 400;
}

.axis text {
  transform: rotate(-45deg);
  /* text-anchor: end; */
}

.th-form-row,
.config-load-row {
  width: 276px;
  height: 26px;
  /* margin-left: 10px; */
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  font-size: 14px;
  justify-content: space-between;
  align-items: center;
}

.time-scope-control {
  width: 180px;
  height: 24px;
  display: flex;
  align-items: center;
}

.config-load-row .form-control {
  width: 168px;
  height: 22px !important;
  /* width: 120px; */
  padding: 2px 4px !important;
  margin-left: 4px;
  border: none;
  border-bottom: solid 1px #9c9c9c;
  border-radius: 0;
  font-size: 14px;
  text-align: center;
  color:#1a73e8;
  /* text-align: left; */
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.time-scope-control .form-control {
  width: 40px;
  height: 20px !important;
  /* width: 120px; */
  padding: 0px 0px !important;
  margin-left: 4px;
  margin-top: 6px;
  border: none;
  border-bottom: solid 1px #9c9c9c;
  border-radius: 0;
  font-size: 14px;
  font-weight: 700;
  text-align: center;
  color:#1a73e8;
  /* text-align: left; */
  /* overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis; */
}

.failure_rule {
  width: 276px;
  font-size: 14px;
  text-align: center;
  padding: 4px !important;
  margin: 0 auto;
}

.failure_rule tr {
  display: flex !important;
  justify-content: space-between;
}

.failure_rule th, 
.failure_rule td {
  /* display: inline-block; */
  padding: 3px !important;
}

.failure_rule td {
  border-bottom-width: 0;
  height: 34px;
}

.failure_rule th:first-child,
.failure_rule td:first-child {
  width: 130px;
}
.failure_rule th:nth-child(2),
.failure_rule td:nth-child(2) {
  width: 84px;
}

.failure_rule th:nth-child(3),
.failure_rule td:nth-child(3) {
  width: 62px;
}

.rule-range {
  display: flex;
  justify-content: space-around;
  align-items: center;
}

.err_th {
  display: flex;
  justify-content: center;
  align-items: center;
  /* margin: 0 auto; */
}

.rule-range .form-control,
.err_th .form-control {
  width: 48px;
  /* height: 18px; */
  font-size: 14px;
  padding: 2px 6px;
  text-align: center;
}

.add_rule {
  display: flex;
  justify-content: space-around;
  align-items: center;
}

.add_rule .add_button,
.add_rule .sub_button {
  font-size: 20px;
  cursor: pointer;
}

.add_rule .add_button:hover,
.add_rule .sub_button:hover {
  color: #1a73e8;
}

/* .add_rule .sub_button:first-of-type:hover {
  color: #212529;
  cursor: auto;
} */

.th-form-row label {
  margin-top: 8px;
  display: block;
}

/* .dropdown-menu,
.dropdown-item {
  font-size: 14px;
  width: 130px !important;
  cursor: pointer;
  white-space: normal;
} */


.config-seg-line {
  height: 1px;
  width: 280px;
  background-color: #bcbcbc;
  margin: 0 auto;
  margin-top: 8px;
  margin-bottom: 8px;
}

#save-cur-config {
  margin-top: 6px;
  font-size: 14px;
  font-weight: 700;
  color: #777;
  text-decoration: underline;
}
#save-cur-config:hover {
  color: #1a73e8;
  cursor: pointer;
}

#get-model-button {
  margin: 0 auto;
  margin-top: 0px;
  margin-bottom: 1px;
  width: 260px;
  height: 32px;
  /* padding: 1px 0px; */
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

.error-config-title {
  display: flex;
  /* justify-content: space-between; */
  align-items: center;
}

.error-config-title span:first-child {
  margin-right: 6px;
}

.save-config-description {
  font-size: 14px;
}

.li-config-description {
  font-size: 12px;
  color: #777;
}
.save-config-name {
  margin-top: 6px;
  font-size: 14px;
  display: flex;
  align-items: center;
  margin-bottom: 6px;
}

.save-config-name .form-control {
  width: 190px !important;
  height: 22px;
  /* width: 120px; */
  margin-left: 4px;
  padding: 0px 2px 0 4px !important;
  border: none;
  border-bottom: solid 1px #9c9c9c;
  border-radius: 0;
  font-size: 14px;
  text-align: center;
}

.save-config-row {
  margin-top: 6px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.save-config-state {  
  font-size: 14px;
  /* font-weight: 700; */
  color: #157347;
}

#save-config-btn {
  font-size: 14px;
  padding: 3px 6px !important;
}

.level-config-icon {
  margin-left: 10px;
  font-size: 20px;
  color: #777;
}
.level-config-icon:hover {
  cursor: pointer;
  color:#1a73e8;
}

</style>