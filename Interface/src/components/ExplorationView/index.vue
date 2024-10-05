<script setup>
import { ref, reactive, computed, watch, onMounted, onUpdated } from "vue"
import { useInforStore } from '@/stores/infor.js'
import getData from '@/services/index.js'
import * as d3 from 'd3'
import $ from 'jquery'
import { valColorScheme_blue, valColorScheme_red, valColorScheme_fire } from '@/data/index.js'
import SubsetList from '@/components/SubsetList/index.vue'
import EventsView from '@/components/EventsView/index.vue'
import html2canvas from 'html2canvas'

const inforStore = useInforStore()
let global_time_id = ref(0)
let global_time_id_left = ref(0)
let global_time_id_right = ref(0)
let cur_sel_step = ref(0)
let cur_hover_loc_left = ref(-1)
let cur_hover_loc_right = ref(-1)
let cur_hover_loc = ref(-1)
let cur_left_hover_entity = ref(-1)
let cur_right_hover_entity = ref(-1)
let cur_outer_indicator = ref('PSFC')
let cur_inner_indicator = ref('RH')
let view_linked = ref(true)
let view_form_types = ref(['feature', 'error'])
let left_form_type = ref('feature')
let right_form_type = ref('error')
let left_form_model = ref(inforStore.cur_focused_model)
let right_form_model = ref(inforStore.cur_focused_model)
let step_opts = ref(['previous', 'current'])
let sel_step_left = ref('current')
let sel_step_right = ref('current')
let cur_step_left = ref(0)
let cur_step_right = ref(0)
let cur_focus_fore_step = ref(-1)

const axis_attrs = ref(['Ground Truth', 'Prediction', 'Residual', 'target_val', 'temporal_state_vals', 'space_diff_state_vals', 'space_comp_state_vals'])
let bin_color_types = ref(['Count', 'Ground Truth', 'Prediction', 'Residual', 'target_val', 'temporal_state_vals', 'space_diff_state_vals', 'space_comp_state_vals'])

let intervalId_phase
let intervalId_subset

let inspect_modes = ['error', 'prediction']
let cur_inspect_mode = ref('prediction')
let pred_infors = ['']
let err_indicators = ['multi_indicators', 'residual', 'binary', 'level']
let cur_err_indicators = ref('multi_indicators')
// let focus_fore_steps = ref([1,1])
let cur_event_start = 0
let cur_sel_event_step = 0

// watch (() => inforStore.cur_data_infor, (oldValue, newValue) => {
//   focus_fore_steps.value = [1, inforStore.cur_data_infor.time.output_window]
// })
let cur_temporal_focus_cnt = ref([])
let cur_temporal_focus_locs = ref([])

const view_id = (str, attr) => `${str}-${attr}`

watch (() => left_form_model.value, (oldValue, newValue) => {
  if (left_form_type.value == 'feature') {
    drawSTLayout_Feature(left_form_model.value, sel_step_left.value, 'left')
  } else if (left_form_type.value == 'error') {
    drawSTLayout_Phase(left_form_model.value, sel_step_left.value, 'left')
  }
})

watch (() => right_form_model.value, (oldValue, newValue) => {
  if (right_form_type.value == 'feature') {
    drawSTLayout_Feature(right_form_model.value, sel_step_right.value, 'right')
  } else if (right_form_type.value == 'error') {
    drawSTLayout_Phase(right_form_model.value, sel_step_right.value, 'right')
  }
})

watch (() => left_form_type.value, (oldValue, newValue) => {
  if (left_form_type.value == 'feature') {
    drawSTLayout_Feature(inforStore.cur_sel_model, sel_step_left.value, 'left')
  } else if (left_form_type.value == 'error') {
    drawSTLayout_Phase(inforStore.cur_sel_model, sel_step_left.value, 'left')
  }
})
watch (() => sel_step_left.value, (oldValue, newValue) => {
  if (sel_step_left.value == 'current') {
    global_time_id_left.value = global_time_id.value
    cur_step_left.value = cur_sel_step.value
  }
  if (sel_step_left.value == 'previous') {
    cur_step_left.value = cur_sel_step.value-1
    global_time_id_left.value = global_time_id.value-1
  }
  if (left_form_type.value == 'feature') {
    drawSTLayout_Feature(inforStore.cur_sel_model, sel_step_left.value, 'left')
  } else if (left_form_type.value == 'error') {
    drawSTLayout_Phase(inforStore.cur_sel_model, sel_step_left.value, 'left')
  }
})
watch (() => right_form_type.value, (oldValue, newValue) => {
  if (right_form_type.value == 'feature') {
    drawSTLayout_Feature(inforStore.cur_sel_model, sel_step_right.value, 'right')
  } else if (right_form_type.value == 'error') {
    drawSTLayout_Phase(inforStore.cur_sel_model, sel_step_right.value, 'right')
  }
})
watch (() => sel_step_right.value, (oldValue, newValue) => {
  if (sel_step_right.value == 'current') {
    global_time_id_right.value = global_time_id.value
    cur_step_right.value = cur_sel_step.value
  }
  if (sel_step_right.value == 'previous') {
    global_time_id_right.value = global_time_id.value-1
    cur_step_right.value = cur_sel_step.value-1
  }
  if (right_form_type.value == 'feature') {
    drawSTLayout_Feature(inforStore.cur_sel_model, sel_step_right.value, 'right')
  } else if (right_form_type.value == 'error') {
    drawSTLayout_Phase(inforStore.cur_sel_model, sel_step_right.value, 'right')
  }
})

watch (() => inforStore.cur_subset_st_indices, (oldValue, newValue) => {
  if (inforStore.cur_subset_st_indices.length == 0) {
    // cur_temporal_focus_cnt.value.map(() => 0)
    // cur_temporal_focus_locs.value.map(() => [])
    let sel_phase_id = inforStore.sel_phase_details[inforStore.cur_focused_model].sel_phase_id
    // console.log('sel_phase_id', sel_phase_id, inforStore.sel_phase_details);
    let phase_infor = inforStore.st_phase_events.phases[inforStore.cur_focused_model][sel_phase_id]
    cur_temporal_focus_cnt.value = Array(parseInt(phase_infor.life_span)).fill(0)
    cur_temporal_focus_locs.value = Array(parseInt(phase_infor.life_span)).fill().map(() => [])
  } 
  else {
    let time_cnt = Array(cur_temporal_focus_cnt.value.length).fill(0)
    let time_locs = Array(cur_temporal_focus_locs.value.length).fill().map(() => [])
    for (let i = 0; i < inforStore.cur_subset_st_indices.length; ++i) {
      let cur_time_index = inforStore.cur_subset_st_indices[i][0]
      // if (cur_time_index >= time_locs.length) console.log('cur_time_index', cur_time_index, i);
      time_cnt[cur_time_index] += 1
      time_locs[cur_time_index].push(inforStore.cur_subset_st_indices[i][1])
    }
    cur_temporal_focus_cnt.value = time_cnt
    cur_temporal_focus_locs.value = time_locs
  }
  drawTimeEventBar()
  drawViews()
})

watch (() => inforStore.cur_sel_event_steps, (oldValue, newValue) => {
  drawTimeEventBar()
})




onUpdated(() => {
  // if ((inforStore.cur_detail_type == 'phase') && (inforStore.cur_phase_sorted_id != -1)) {
  //   // drawSTLayout_Phase()
  // }
  // else if (inforStore.cur_detail_type == 'subset') drawSubsetDetails(inforStore.sel_subset_details)
})

let pre_phase_id = -1
let global_residuals_range = []
let global_wind_max = 0

function drawViews() {
  if (left_form_type.value == 'feature') {
    drawSTLayout_Feature(inforStore.cur_sel_model, sel_step_left.value, 'left')
  } else if (left_form_type.value == 'error') {
    drawSTLayout_Phase(inforStore.cur_sel_model, sel_step_left.value, 'left')
  }
  if (right_form_type.value == 'feature') {
    drawSTLayout_Feature(inforStore.cur_sel_model, sel_step_right.value, 'right')
  } else if (right_form_type.value == 'error') {
    drawSTLayout_Phase(inforStore.cur_sel_model, sel_step_right.value, 'right')
  }
}

let loc_coords_x
let loc_coords_y
let grid_borders
let grid_points_x
let grid_points_y
let white_rate
let loc_x_scale
let loc_y_scale
let grid_points, line

watch (() => inforStore.sel_phase_details, (oldValue, newValue) => {
  console.log('sel_phase', inforStore.sel_phase_details);
  if (cur_hover_loc.value != -1) cur_hover_loc.value = -1
  let sel_phase_id = inforStore.sel_phase_details[inforStore.cur_focused_model].sel_phase_id
  let phase_infor = inforStore.st_phase_events.phases[inforStore.cur_focused_model][sel_phase_id]
  cur_temporal_focus_cnt.value = Array(parseInt(phase_infor.life_span)).fill(0)
  cur_temporal_focus_locs.value = Array(parseInt(phase_infor.life_span)).fill().map(() => [])
  inforStore.cur_detail_type = 'phase'
  if (sel_phase_id != pre_phase_id) {
    cur_sel_step.value = 0
    cur_event_start = 0
    cur_sel_event_step = 0
    pre_phase_id = sel_phase_id
  }
  
  global_time_id.value = parseInt(cur_sel_step.value) + parseInt(phase_infor.start)
  if (sel_step_left.value == 'current') {
    global_time_id_left.value = global_time_id.value
    cur_step_left.value = cur_sel_step.value
  }
  if (sel_step_left.value == 'previous') {
    cur_step_left.value = cur_sel_step.value-1
    global_time_id_left.value = global_time_id.value-1
  }
  if (sel_step_right.value == 'current') {
    global_time_id_right.value = global_time_id.value
    cur_step_right.value = cur_sel_step.value
  }
  if (sel_step_right.value == 'previous') {
    cur_step_right.value = cur_sel_step.value-1
    global_time_id_right.value = global_time_id.value-1
  }

  loc_coords_x = inforStore.cur_data_infor.space.loc_list.map(item => item.geometry.coordinates[0])
  loc_coords_y = inforStore.cur_data_infor.space.loc_list.map(item => item.geometry.coordinates[1])
  grid_borders = inforStore.cur_data_infor.space.grid_borders
  // console.log(grid_borders);
  grid_points = grid_borders.map(item => item['border_points'])
  // let grid_points_end = grid_borders.map(item => item['border_points'].slice(1, 5))
  grid_points_x = grid_points.map(item => item.map(e => e[0])).flat()
  grid_points_y = grid_points.map(item => item.map(e => e[1])).flat()
  
  white_rate = 0.05
  loc_x_scale = d3.scaleLinear()
        .domain([Math.min(...grid_points_x), Math.max(...grid_points_x)])
        .range([space_layout_h*white_rate, space_layout_h*(1-white_rate)])
  loc_y_scale = d3.scaleLinear()
        .domain([Math.min(...grid_points_y), Math.max(...grid_points_y)])
        .range([space_layout_h*(1-white_rate), space_layout_h*white_rate])
  line = d3.line()
    .x(d => loc_x_scale(d[0])) // x 坐标映射
    .y(d => loc_y_scale(d[1])); // y 坐标映射

  // 启动轮询，每隔一定时间调用一次轮询函数
  // intervalId_phase = setInterval(askForDrawPhase, 100);
  if (document.getElementById('st-layout-right')){
    let all_features = inforStore.feature_infor.input.split(', ')
    let U_index = all_features.indexOf('U')
    let V_index = all_features.indexOf('V')
    let phase_wind = inforStore.cur_phase_data.phase_raw_data
    let global_U = [].concat(...phase_wind.map(arr => [].concat(...arr.map(item => item[U_index]))))
    let global_V = [].concat(...phase_wind.map(arr => [].concat(...arr.map(item => item[V_index]))))
    let loc_U_extent = d3.extent(global_U)
    let loc_V_extent = d3.extent(global_V)
    global_wind_max = Math.max(-loc_U_extent[0], loc_U_extent[1], -loc_V_extent[0], loc_V_extent[1])

    let global_residuals = [].concat(...inforStore.sel_phase_details[inforStore.cur_focused_model].st_residuals.map(arr => [].concat(...arr)))
    let tmp_range = d3.extent(global_residuals)
    let global_err_abs_max = (Math.round(d3.max([-tmp_range[0], tmp_range[1]]) * 100) / 100).toFixed(2)
    global_residuals_range = [-global_err_abs_max, global_err_abs_max]

    drawViews()
    drawTimeEventBar()
  }
  // drawLevelConfusionSpace(inforStore.sel_phase_details)
  // console.log(inforStore.sel_phase_details)
})

let other_indicators = ref(['POD/FAR', 'Multi_accuracy'])
let other_focused_indicators = ref([])
watch (() => inforStore.cur_timeline_indicator, (oldValue, newValue) => {
  if (inforStore.cur_timeline_indicator == 'Residual_abs') other_indicators.value = ['POD/FAR', 'Multi_accuracy']
  if (inforStore.cur_timeline_indicator == 'POD' || inforStore.cur_timeline_indicator == 'FAR') other_indicators.value = ['Multi_accuracy', 'Residual_abs']
  if (inforStore.cur_timeline_indicator == 'Multi_accuracy') other_indicators.value = ['POD/FAR', 'Residual_abs']
  other_focused_indicators.value = []
  drawViews()
  drawTimeEventBar()
})

watch (() => other_focused_indicators.value, (oldValue, newValue) => {
  if (left_form_type.value == 'error') drawSTLayout_Phase(inforStore.cur_sel_model, sel_step_left.value, 'left')
  if (right_form_type.value == 'error') drawSTLayout_Phase(inforStore.cur_sel_model, sel_step_right.value, 'right')
})

watch (() => cur_outer_indicator.value, (oldValue, newValue) => {
  if (left_form_type.value == 'feature') drawSTLayout_Feature(inforStore.cur_sel_model, sel_step_left.value, 'left')
  if (right_form_type.value == 'feature') drawSTLayout_Feature(inforStore.cur_sel_model, sel_step_right.value, 'right')
})

watch (() => cur_inner_indicator.value, (oldValue, newValue) => {
  if (left_form_type.value == 'feature') drawSTLayout_Feature(inforStore.cur_sel_model, sel_step_left.value, 'left')
  if (right_form_type.value == 'feature') drawSTLayout_Feature(inforStore.cur_sel_model, sel_step_right.value, 'right')
})

watch (() => inforStore.sel_subset_details, (oldValue, newValue) => {
  inforStore.cur_detail_type = 'subset'
  // intervalId_subset = setInterval(() => askForDrawSubset(inforStore.sel_subset_details), 100);
  if (document.getElementById('join-dist-space'))
    drawSubsetDetails(inforStore.sel_subset_details)
  // console.log(inforStore.sel_subset_details)
})

watch (() => cur_sel_step.value, (oldValue, newValue) => {
  let sel_phase_id = inforStore.sel_phase_details[inforStore.cur_focused_model].sel_phase_id
  let phase_infor = inforStore.st_phase_events.phases[inforStore.cur_focused_model][sel_phase_id]
  global_time_id.value = parseInt(cur_sel_step.value) + parseInt(phase_infor.start)
  if (sel_step_left.value == 'current') {
    global_time_id_left.value = global_time_id.value
    cur_step_left.value = cur_sel_step.value
  }
  if (sel_step_left.value == 'previous') {
    cur_step_left.value = cur_sel_step.value-1
    global_time_id_left.value = global_time_id.value-1
  }
  if (sel_step_right.value == 'current') {
    global_time_id_right.value = global_time_id.value
    cur_step_right.value = cur_sel_step.value
  }
  if (sel_step_right.value == 'previous') {
    cur_step_right.value = cur_sel_step.value-1
    global_time_id_right.value = global_time_id.value-1
  }
  drawViews()
  console.log('cur_hover_loc.value', cur_hover_loc.value);
  if (cur_hover_loc.value != -1) {
    drawLocResidualInfor(cur_hover_loc.value, cur_sel_step.value)
    // drawLocTimeContext(cur_hover_loc.value, cur_sel_step.value)
    // drawLocFeatureInfor(cur_hover_loc.value, cur_sel_step.value)
    getData(inforStore, 'loc_instance_infor', JSON.stringify(inforStore.cur_sel_models),  inforStore.cur_phase_sorted_id, cur_sel_step.value, cur_hover_loc.value)
  }
  if (cur_hover_entity.value != -1) {
    drawEntityAreaError(cur_hover_entity.value, cur_sel_step.value)
    drawEntityMoveError(cur_hover_entity.value, cur_sel_step.value)
    drawEntityIntensityError(cur_hover_entity.value, cur_sel_step.value)
  }
})

function getDistance(p1, p2, x_scale, y_scale) {
  let dx = x_scale(p2[0]) - x_scale(p1[0]);
  let dy = y_scale(p2[1]) - y_scale(p1[1]);
  return Math.sqrt(Math.pow(dx, 2) + Math.pow(dy, 2));
}

let cur_transform_state = {k: 1, x: 0, y: 0}
let transform_state_left = {k: 1, x: 0, y: 0}
let transform_state_right = {k: 1, x: 0, y: 0}
let zoom_func = d3.zoom()
    .scaleExtent([0.1, 10])
    .on('zoom', handleZoom);
function handleZoom(e) {
  cur_transform_state = e.transform
  transform_state_left = e.transform
  transform_state_right = e.transform
  d3.selectAll('.space-layout-g').attr('transform', cur_transform_state)
}
let zoom_func_left = d3.zoom()
    .scaleExtent([0.1, 10])
    .on('zoom', handleZoom_left);
function handleZoom_left(e) {
  transform_state_left = e.transform
  cur_transform_state = e.transform
  d3.select('#space-layout-g-left').attr('transform', transform_state_left)
}
let zoom_func_right = d3.zoom()
    .scaleExtent([0.1, 10])
    .on('zoom', handleZoom_right);
function handleZoom_right(e) {
  transform_state_right = e.transform;
  cur_transform_state = e.transform
  d3.select('#space-layout-g-right').attr('transform', transform_state_right)
}
watch (() => view_linked.value, (oldValue, newValue) => {
  if (view_linked.value) {
    // d3.select('#st-layout-feature').call(zoom_func_left.transform, d3.zoomIdentity)
    // d3.select('#st-layout-phase').call(zoom_func_right.transform, d3.zoomIdentity)
    // console.log(d3.zoomIdentity);
    transform_state_left = {
      k: cur_transform_state.k,
      x: cur_transform_state.x,
      y: cur_transform_state.y
    }
    transform_state_right = {
      k: cur_transform_state.k,
      x: cur_transform_state.x,
      y: cur_transform_state.y
    }
    d3.select('#st-layout-left').call(zoom_func_left.transform, cur_transform_state)
    d3.select('#st-layout-right').call(zoom_func_right.transform, cur_transform_state)
    d3.select('#st-layout-left').call(zoom_func)
    d3.select('#st-layout-right').call(zoom_func)
  } else {
    d3.select('#st-layout-left').call(zoom_func.transform, transform_state_left)
    d3.select('#st-layout-right').call(zoom_func.transform, transform_state_right)
    d3.select('#st-layout-left').call(zoom_func_left)
    d3.select('#st-layout-right').call(zoom_func_right)
  }
})

let cur_phase_time_str_left = ref('')
let cur_phase_time_str_right = ref('')
let space_layout_w = 760, space_layout_h = 640
function drawSTLayout_Feature(model_sel, step_sel, view_sel) {
  let svg_id
  let stamp_strs = inforStore.st_phase_events.time_strs
  let cur_model = inforStore.cur_focused_model
  if (view_sel == 'left') {
    svg_id = '#st-layout-left'  
    cur_model = left_form_model.value
  } else if (view_sel == 'right') {
    svg_id = '#st-layout-right'
    cur_model = right_form_model.value
  }
  
  let phase_id = inforStore.sel_phase_details[cur_model].sel_phase_id
  let phase_list = inforStore.st_phase_events.phases[cur_model]
  let cur_step, cur_global_time_id
  if (step_sel == 'current') {
    cur_step = Number(cur_sel_step.value)
    cur_global_time_id = global_time_id.value
  } else if (step_sel == 'previous') {
    cur_step = Number(cur_sel_step.value) - 1
    cur_global_time_id = global_time_id.value - 1
  }
  let stamp_id = phase_list[phase_id].start + cur_step
  if (view_sel == 'left') {
    cur_phase_time_str_left.value = stamp_strs[stamp_id]
  } else if (view_sel == 'right') {
    cur_phase_time_str_right.value = stamp_strs[stamp_id]
  }
  
  d3.select(svg_id).selectAll('*').remove()
  let sel_phase_id = inforStore.sel_phase_details[cur_model].sel_phase_id
  let margin_left = 10, margin_right = 10
  let st_layout_w = space_layout_w + margin_left + margin_right
  let st_layout_h = space_layout_h - 5

  let st_layout_svg = d3.select(svg_id)
    .attr('class', 'st-layout-svg')
    .attr('width', st_layout_w)
    .attr('height', st_layout_h)
  
  let cell_len = getDistance(grid_borders[0].border_points[0], grid_borders[0].border_points[1], loc_x_scale, loc_y_scale)
  let space_layout_g = st_layout_svg.append('g')
    .attr('id', 'space-layout-g-left')
    .attr('class', 'space-layout-g')
    .attr('transform', () => {
      if (view_linked.value) return cur_transform_state
      else return transform_state_left
    })
    .attr('transform', `translate(50, -20)`)
  let grid_borders_g = space_layout_g.append('g')
    .attr('transform', `translate(${margin_left+space_layout_w*0.08}, 4) scale(0.85)`)
    .on('mousedown', (e) => {
      let border_path = d3.select(e.target)
      while (border_path.attr('loc_id') == null) {
        border_path = d3.select(border_path.node().parentNode)
      }
      let target_id = parseInt(border_path.attr('loc_id'))
      if (e.button === 2) {
        if (target_id == cur_hover_loc.value) cur_hover_loc.value = -1
        else if (target_id > -1) cur_hover_loc.value = target_id
      }
      if (e.button === 0) {
        let cur_entities 
        if (view_sel == 'left') cur_entities = phase_list[phase_id].focus_entities[cur_step_left.value]
        else if (view_sel == 'right') cur_entities = phase_list[phase_id].focus_entities[cur_step_right.value]

        let target_entity_id = -1
        for (let i = 0; i < cur_entities.length; ++i) {
          if (cur_entities[i].loc_ids.includes(target_id)) {
            target_entity_id = i
            break
          }
        }
        if (view_sel == 'left') {
          if (cur_left_hover_entity.value == target_entity_id) cur_left_hover_entity.value = -1
          else cur_left_hover_entity.value = target_entity_id
        }
        else if (view_sel == 'right') {
          if (cur_right_hover_entity.value == target_entity_id) cur_right_hover_entity.value = -1
          else cur_right_hover_entity.value = target_entity_id
        }
      }
    })
    // .on('mouseover', (e) => {
    //   let border_path = d3.select(e.target)
    //   while (border_path.attr('loc_id') == null) {
    //     border_path = d3.select(border_path.node().parentNode)
    //   }
    //   let target_id = parseInt(border_path.attr('loc_id'))

    //   if (view_sel == 'left') cur_hover_loc_left.value = target_id
    //   else if (view_sel == 'right') cur_hover_loc_right.value = target_id
    // })
    // .on('mouseout', (e) => {
    //   if (view_sel == 'left') cur_hover_loc_left.value = -1
    //   else if (view_sel == 'right') cur_hover_loc_right.value = -1
    //   // d3.select(`#loc-tooltip-${view_sel}`)
    //   //   .style("opacity", 0);
    // })
  let pollu_grid_borders_g = space_layout_g.append('g')
    .attr('transform', `translate(${margin_left+space_layout_w*0.08}, 4) scale(0.85)`)
    // .on('mouseover', (e) => {
    //   let border_path = d3.select(e.target)
    //   while (border_path.attr('loc_id') == null) {
    //     border_path = d3.select(border_path.node().parentNode)
    //   }
    //   let target_id = parseInt(border_path.attr('loc_id'))
    //   if (view_sel == 'left') cur_hover_loc_left.value = target_id
    //   else if (view_sel == 'right') cur_hover_loc_right.value = target_id
    // })
    // .on('mouseout', (e) => {
    //   if (view_sel == 'left') cur_hover_loc_left.value = -1
    //   else if (view_sel == 'right') cur_hover_loc_right.value = -1
    //   // d3.select(`#loc-tooltip-${view_sel}`)
    //   //   .style("opacity", 0);
    // })
  let level_id_list = []
  for (let i = 0; i < inforStore.dataset_configs.focus_levels.length-1; ++i) {
    level_id_list.push(i)
  }
  
  // 获取features的数据
  let all_features = inforStore.feature_infor.input.split(', ')
  let outer_index = all_features.indexOf(cur_outer_indicator.value)
  let inner_index = all_features.indexOf(cur_inner_indicator.value)
  let loc_outer_data = inforStore.cur_phase_data.phase_raw_data[cur_sel_step.value].map(item => item[outer_index])
  let loc_inner_data = inforStore.cur_phase_data.phase_raw_data[cur_sel_step.value].map(item => item[inner_index])
  let U_index = all_features.indexOf('U')
  let V_index = all_features.indexOf('V')
  let loc_U = inforStore.cur_phase_data.phase_raw_data[cur_sel_step.value].map(item => item[U_index])
  let loc_V = inforStore.cur_phase_data.phase_raw_data[cur_sel_step.value].map(item => item[V_index])
  let windScale = d3.scaleLinear()
    .domain([-global_wind_max, global_wind_max])
    .range([-cell_len*0.30, cell_len*0.30])
  let outerColor = d3.scaleQuantize()
    .domain(d3.extent(loc_outer_data))
    // .range(['#eff3ff', '#c6dbef', '#9ecae1', '#6baed6', '#3182bd', '#08519c'])
    // .range(['#8e0152','#c51b7d','#de77ae','#f1b6da','#fde0ef'])
    // .range(['#edf8e9', '#c7e9c0', '#a1d99b', '#74c476', '#31a354', '#006d2c'])
    // .range(['#E0F2F1', '#B2DFDB', '#80CBC4', '#4DB6AC', '#26A69A', '#009688'])
    .range(['#ECEFF1', '#CFD8DC', '#B0BEC5', '#90A4AE', '#78909C'])

    // .range(['#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c'])

  let innerColor = d3.scaleQuantize()
    .domain(d3.extent(loc_inner_data))
    // .range(['#edf8e9', '#c7e9c0', '#a1d99b', '#74c476', '#31a354', '#006d2c'])
    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    //原方案:
    // .range(['#feedde', '#fdd0a2', '#fdae6b', '#fd8d3c', '#e6550d', '#a63603'])
    // .range(['#f6e8c3', '#dfc27d', '#bf812d', '#8c510a', '#543005'])
    .range(['#eec4dc', '#e5a0c6', '#e27bb1', '#e2619f', '#e44b8d'])
    
    // .range(['#E0F2F1', '#B2DFDB', '#80CBC4', '#4DB6AC', '#26A69A', '#009688'])
    // .range(['#FCE4EC','#F8BBD0','#F48FB1','#F06292','#EC407A', 'E91E63'])
    // .range(['#8e0152','#c51b7d','#de77ae','#f1b6da','#fde0ef'])
    // .range(['#490024', '#69003D', '#8E2459', '#A6496D', '#BA697D', '#CE8E92', '#E3AEA6', '#F3CAB6'])
    
  let borders_g = grid_borders_g.append('g')
  borders_g.selectAll('path')
    .data(grid_points)
    .join('path')
      .attr('class', 'left-loc-border')
      .attr('id', (d,i) => `left_loc-${i}`)
      .attr('loc_id', (d,i) => i)
      .attr("d", line)
      .attr("fill", (d, i) => outerColor(loc_outer_data[i]))
      .attr("stroke", "#999")
      .attr("stroke-width", 1)
      .attr("opacity", 0.9)
  let pollu_grid_points = grid_points.filter(function(value, index) {
    let cur_val = inforStore.cur_phase_data.phase_raw_data[cur_step][index][0]
    return (cur_val >= inforStore.dataset_configs.focus_th)
  })
  pollu_grid_borders_g.selectAll('path')
    .data(pollu_grid_points)
    .join('path')
      .attr('id', (d,i) => `left_pollu_loc-${i}`)
      .attr('loc_id', (d,i) => i)
      .attr("d", line) // 应用生成器
      .attr("fill", 'none')
      .attr("stroke", (d,i) => '#000')
      .attr("stroke-width", 2)
      .style("stroke-linejoin", "round")
      .style("stroke-linecap", "round")
      .attr("opacity", 0.9)
  let mark_grid = space_layout_g.append('g')
    .attr('transform', `translate(${margin_left+space_layout_w*0.08}, 4) scale(0.85)`)
    .append('path')
      .attr('id', `${view_sel}-mark-grid`)
      .datum([])
      .attr("d", line) // 应用生成器
      .attr("fill", 'none')
      .attr("stroke", 'red')
      .attr("stroke-width", 4)
      .style("stroke-linejoin", "round")
      .style("stroke-linecap", "round")
  let mark_entity = space_layout_g.append('g')
    .attr('transform', `translate(${margin_left+space_layout_w*0.08}, 4) scale(0.85)`)
    .attr('id', `${view_sel}-mark-entity`)
  
  let subset_marks_g = space_layout_g.append('g')
    .attr('transform', `translate(${margin_left+space_layout_w*0.08}, 4) scale(0.85)`)
  let subset_marks = subset_marks_g.selectAll('circle')
    .data(cur_temporal_focus_locs.value[cur_step])
    .join('circle')
      .attr('transform', (d,i) => `translate(${loc_x_scale(loc_coords_x[d])-cell_len*0.33}, ${loc_y_scale(loc_coords_y[d])-cell_len*0.41}) scale(2)`)
      .attr('cx', 0).attr('cy', 0)
      .attr('r', 2)
      .attr('fill', '#333')
      .attr('stroke', 'none')
  // let s_mark_d = 
  
  // let initialTransform = d3.zoomIdentity;
  // st_layout_svg.call(zoom_func.transform, initialTransform);
  if (view_linked.value) st_layout_svg.call(zoom_func)
  else st_layout_svg.call(zoom_func_left)
  // let transformState = d3.zoomIdentity;
  if (inforStore.sel_phase_details[cur_model].phase_pred_val[cur_step][0].length == 0) return
  let glyphs_g = grid_borders_g.append('g')
    // .attr('transform', `translate(${margin_left+space_layout_w*0.08}, 4) scale(0.85)`)
  for (let i = 0; i < loc_outer_data.length; ++i) {
    let single_glyphs_g = glyphs_g.append('g')
    let single_glyph = single_glyphs_g.append('g')
      .attr('transform', `translate(${loc_x_scale(loc_coords_x[i])}, ${loc_y_scale(loc_coords_y[i])})`)
      .attr('loc_id', i)
    // 计算风速的大小和角度（注意：计算中U表示纬向风速，V表示经向风速）
    let windSpeed = Math.sqrt(loc_U[i] * loc_U[i] + loc_V[i] * loc_V[i]); // 风速大小
    let windAngle = Math.atan2(-loc_U[i], loc_V[i]) * (180 / Math.PI); // 风向角度
    // let arrowPath = "M0,-5L10,0L0,5Z"; // 箭头的形状
    // let arrowPath = "M0,-4L8,0L0,4Z"; // 箭头的形状
    // let arrowPath = "M0,-3L6,0L0,3L2,0Z"; // 箭头的形状
    let arrowPath = "M0,-4L8,0L0,4L3,0Z"; // 箭头的形状
    let arrow_g = single_glyph.append('g')
    arrow_g.append('circle')
      .attr('x', 0).attr('y', 0)
      .attr('r', cell_len*0.32).attr('fill', '#333')
      .attr('fill', '#fff')
      // .attr('fill', innerColor(loc_inner_data[i]))
      .attr('stroke', innerColor(loc_inner_data[i]))
      .attr('stroke-width', 5)
    arrow_g.append('circle')
      .attr('x', 0).attr('y', 0)
      .attr('r', 2)
      .attr('fill', '#333')
      // .attr('fill', innerColor(loc_inner_data[i]))
    arrow_g.append('line')
      .attr('x1', 0).attr('y1', 0)
      .attr('x2', windScale(loc_V[i])).attr('y2', -windScale(loc_U[i]))
      // .attr('stroke', innerColor(loc_inner_data[i]))
      .attr('stroke', '#333')
    arrow_g.append("path")
      .attr("d", arrowPath)
      .attr("transform", `translate(${windScale(loc_V[i])},${-windScale(loc_U[i])}) rotate(${windAngle}) translate(-4,0)`)
      // .attr("transform", `rotate(${windAngle}) scale(1.5) translate(-3,0)`)
      .style("fill", "#333")
      // .attr('fill', innerColor(loc_inner_data[i]))
  }


  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  
  // const legendRect = st_layout_svg.selectAll("legendRect")
  //       .data(d3.range(d3.min(loc_inner_data), d3.max(loc_inner_data), (d3.max(loc_inner_data) - d3.min(loc_inner_data)) / 12)) // 使用比例尺范围内的数据创建矩形
  //       .join('rect')
  //       .attr('x', (d,i) => 60+i*9)
  //       .attr('y', 17)
  //       .attr('width', 9)
  //       .attr('height', 12)
  //       .attr("fill", d => innerColor(d)); // 根据比例尺填充颜色
  
  // const legendRect_out = st_layout_svg.selectAll("legendRect")
  //       .data(d3.range(d3.min(loc_outer_data), d3.max(loc_outer_data), (d3.max(loc_outer_data) - d3.min(loc_outer_data)) / 12)) // 使用比例尺范围内的数据创建矩形
  //       .join('rect')
  //       .attr('x', (d,i) => 60+i*9)
  //       .attr('y', 17)
  //       .attr('width', 9)
  //       .attr('height', 12)
  //       .attr("fill", d => outerColor(d)); // 根据比例尺填充颜色
  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  let resi_legend_len = 80
  let resiLegendScale = d3.scaleQuantize()
  // let resiLegendScale = d3.scaleSequential(d3.interpolateBrBG)
    .domain([0, resi_legend_len])
    // .range(['#8e0152','#c51b7d','#de77ae','#f1b6da','#fde0ef']);
    // .range(['#edf8e9', '#bae4b3', '#74c476', '#31a354', '#006d2c'])
    .range(['#ECEFF1', '#CFD8DC', '#B0BEC5', '#90A4AE', '#78909C'])

  let resiLegendScale_inner = d3.scaleQuantize()
  // let resiLegendScale = d3.scaleSequential(d3.interpolateBrBG)
    .domain([0, resi_legend_len])
    // .range(['#8e0152','#c51b7d','#de77ae','#f1b6da','#fde0ef']);
    // .range(['#edf8e9', '#bae4b3', '#74c476', '#31a354', '#006d2c'])
    // .range(['#f6e8c3', '#dfc27d', '#bf812d', '#8c510a', '#543005'])
    .range(['#eec4dc', '#e5a0c6', '#e27bb1', '#e2619f', '#e44b8d'])

  let residual_legend = st_layout_svg.append('g')
    .attr('transform', 'translate(60, 24)')
  let residual_legend_inner = st_layout_svg.append('g')
    .attr('transform', 'translate(60, 18)')
  residual_legend_inner.selectAll('rect')
  .data(Array(resi_legend_len).fill(1))
    .join('rect')
      .attr('x', (d,i) => i)
      .attr('y', 0)
      .attr('width', 1)
      .attr('height', 12)
      .attr('fill', (d,i) => resiLegendScale_inner(i))

  residual_legend.selectAll('rect')
    .data(Array(resi_legend_len).fill(1))
    .join('rect')
      .attr('x', (d,i) => i)
      .attr('y', 30)
      .attr('width', 1)
      .attr('height', 12)
      .attr('fill', (d,i) => resiLegendScale(i))
  
  residual_legend.append('text')
    .attr('x', resi_legend_len * 0.5)
    .attr('y', -12)
    .attr('text-anchor', 'middle')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text(cur_inner_indicator.value)

  residual_legend.append('text')
    .attr('x', -4)
    .attr('y', 5)
    .attr('text-anchor', 'end')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text(d3.min(loc_inner_data))

  residual_legend.append('text')
    .attr('x', resi_legend_len+4)
    .attr('y', 5)
    .attr('text-anchor', 'start')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text(d3.max(loc_inner_data))

  residual_legend.append('text')
    .attr('x', resi_legend_len * 0.5)
    .attr('y', 23)
    .attr('text-anchor', 'middle')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text(cur_outer_indicator.value)
  residual_legend.append('text')
    .attr('x', -4)
    .attr('y', 40)
    .attr('text-anchor', 'end')
    .style('font-size', '12px')
    .attr('fill', '#333')
    // .text(`${local_err_range[0]}`)
    .text(`${d3.min(loc_outer_data)}`)
  residual_legend.append('text')
    .attr('x', resi_legend_len+4)
    .attr('y', 40)
    .attr('text-anchor', 'start')
    .style('font-size', '12px')
    .attr('fill', '#333')
    // .text(`${local_err_range[1]}`)
    .text(`${d3.max(loc_outer_data)}`)
}

function drawSTLayout_Phase(model_sel, step_sel, view_sel) {
  let svg_id
  let stamp_strs = inforStore.st_phase_events.time_strs
  let cur_model = inforStore.cur_focused_model
  if (view_sel == 'left') {
    svg_id = '#st-layout-left'  
    cur_model = left_form_model.value
  } else if (view_sel == 'right') {
    svg_id = '#st-layout-right'
    cur_model = right_form_model.value
  }
  
  let phase_id = inforStore.sel_phase_details[cur_model].sel_phase_id
  let phase_list = inforStore.st_phase_events.phases[cur_model]
  let cur_step
  if (step_sel == 'current') cur_step = Number(cur_sel_step.value)
  if (step_sel == 'previous') cur_step = Number(cur_sel_step.value)-1
  let stamp_id = phase_list[phase_id].start + cur_step
  if (view_sel == 'left') {
    cur_phase_time_str_left.value = stamp_strs[stamp_id]
  } else if (view_sel == 'right') {
    cur_phase_time_str_right.value = stamp_strs[stamp_id]
  }
  
  d3.select(svg_id).selectAll('*').remove()
  let sel_phase_id = inforStore.sel_phase_details[cur_model].sel_phase_id
  let margin_left = 10, margin_right = 10
  let st_layout_w = space_layout_w + margin_left + margin_right
  let st_layout_h = space_layout_h - 5

  let st_layout_svg = d3.select(svg_id)
    .attr('class', 'st-layout-svg')
    .attr('width', st_layout_w)
    .attr('height', st_layout_h)
  
  let points_x_extent = d3.extent(grid_points_x)
  let points_y_extent = d3.extent(grid_points_y)
  let loc_x_scale = d3.scaleLinear()
        .domain(points_x_extent)
        .range([space_layout_h*white_rate, space_layout_h*(1-white_rate)])
  let loc_y_scale = d3.scaleLinear()
        .domain(points_y_extent)
        .range([space_layout_h*(1-white_rate), space_layout_h*white_rate])
  let space_layout_g = st_layout_svg.append('g')
    .attr('id', 'space-layout-g-right')
    .attr('class', 'space-layout-g')
    // .attr('transform', `translate(${margin_left}, -30)`)
    .attr('transform', () => {
      if (view_linked.value) return cur_transform_state
      else return transform_state_right
    })
    .attr('transform', `translate(50, -20)`)
  // let points_x_range = points_x_extent[1] - points_x_extent[0]
  // let points_y_range = points_y_extent[1] - points_y_extent[0]
  let grid_borders_g = space_layout_g.append('g')
    .attr('transform', `translate(${margin_left+space_layout_w*0.08}, 4) scale(0.85)`)
    .on('mousedown', (e) => {
      inforStore.ins_inspect_type = 'phase'
      $('.overlay-container').css('display', 'flex')
      let border_path = d3.select(e.target)
      while (border_path.attr('loc_id') == null) {
        border_path = d3.select(border_path.node().parentNode)
      }
      let target_id = parseInt(border_path.attr('loc_id'))
      inforStore.cur_sel_condition.loc_id = target_id
      let tmp_stamp_obj = {}
      tmp_stamp_obj[String(stamp_id)] = {}
      tmp_stamp_obj[String(stamp_id)]['stamp_id'] = stamp_id
      tmp_stamp_obj[String(stamp_id)]['timestamp'] = stamp_strs[stamp_id]
      // tmp_stamp_obj[String(stamp_id)]['fore_step'] = Array.from({ length: binary_labels[i].length }, (_, index) => index)
      tmp_stamp_obj[String(stamp_id)]['binary_label'] = binary_labels[target_id]
      tmp_stamp_obj[String(stamp_id)]['multi_label'] = multi_labels[target_id]
      tmp_stamp_obj[String(stamp_id)]['residual'] = residuals[target_id]
      inforStore.tile_resi_edge_range = local_err_range
      inforStore.cur_sel_stamp_objs_single = tmp_stamp_obj

      while (border_path.attr('loc_id') == null) {
        border_path = d3.select(border_path.node().parentNode)
      }
      console.log(e);
      if (e.button === 2) {
        if (target_id == cur_hover_loc.value) cur_hover_loc.value = -1
        else if (target_id > -1) cur_hover_loc.value = target_id
      }
      if (e.button === 0) {
        let cur_entities 
        if (view_sel == 'left') cur_entities = phase_list[phase_id].focus_entities[cur_step_left.value]
        else if (view_sel == 'right') cur_entities = phase_list[phase_id].focus_entities[cur_step_right.value]
        
        let target_entity_id = -1
        for (let i = 0; i < cur_entities.length; ++i) {
          if (cur_entities[i].loc_ids.includes(target_id)) {
            target_entity_id = i
            break
          }
        }
        if (view_sel == 'left') {
          if (cur_left_hover_entity.value == target_entity_id) cur_left_hover_entity.value = -1
          else cur_left_hover_entity.value = target_entity_id
        }
        else if (view_sel == 'right') {
          if (cur_right_hover_entity.value == target_entity_id) cur_right_hover_entity.value = -1
          else cur_right_hover_entity.value = target_entity_id
        }
      }

      // if (view_sel == 'left') cur_hover_loc_left.value = target_id
      // else if (view_sel == 'right') cur_hover_loc_right.value = target_id
      // getData(inforStore, 'instance_seqs', stamp_id, target_id, inforStore.cur_focused_scope)
    })
    // .on('mouseover', (e) => {
    //   let border_path = d3.select(e.target)
    //   while (border_path.attr('loc_id') == null) {
    //     border_path = d3.select(border_path.node().parentNode)
    //   }
    //   let target_id = parseInt(border_path.attr('loc_id'))
    //   if (view_sel == 'left') cur_hover_loc_left.value = target_id
    //   else if (view_sel == 'right') cur_hover_loc_right.value = target_id
    // })
    // .on('mouseout', (e) => {
    //   if (view_sel == 'left') cur_hover_loc_left.value = -1
    //   else if (view_sel == 'right') cur_hover_loc_right.value = -1
    //   // d3.select(`#loc-tooltip-${view_sel}`)
    //   //   .style("opacity", 0);
    // })
  
  let pollu_grid_borders_g = space_layout_g.append('g')
    .attr('transform', `translate(${margin_left+space_layout_w*0.08}, 4) scale(0.85)`)
    // .on('mouseover', (e) => {
    //   let border_path = d3.select(e.target)
    //   while (border_path.attr('loc_id') == null) {
    //     border_path = d3.select(border_path.node().parentNode)
    //   }
    //   let target_id = parseInt(border_path.attr('loc_id'))
    //   if (view_sel == 'left') cur_hover_loc_left.value = target_id
    //   else if (view_sel == 'right') cur_hover_loc_right.value = target_id
    // })
    // .on('mouseout', (e) => {
    //   if (view_sel == 'left') cur_hover_loc_left.value = -1
    //   else if (view_sel == 'right') cur_hover_loc_right.value = -1
    //   // d3.select(`#loc-tooltip-${view_sel}`)
    //   //   .style("opacity", 0);
    // })
  line = d3.line()
    .x(d => loc_x_scale(d[0])) // x 坐标映射
    .y(d => loc_y_scale(d[1])); // y 坐标映射
  let level_id_list = []
  for (let i = 0; i < inforStore.dataset_configs.focus_levels.length-1; ++i) {
    level_id_list.push(i)
  }
  let boxColorLevel = d3.scaleOrdinal()
    .domain(level_id_list)
    // .range(['#fcfbfd', '#efedf5', '#dadaeb', '#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#4a1486'])
    .range(['#f2f0f7', '#cbc9e2', '#9e9ac8', '#756bb1', '#54278f'])
  let valColor = d3.scaleLinear()
    // .domain([inforStore.dataset_configs.focus_th, 500])
    .domain([0, 500])
    .range(['#fff', '#4a1486'])
  let boxColorPOD = d3.scaleOrdinal()
    .domain([-2, -1, 1, 2])
    .range([valColorScheme_blue[3], '#cecece', '#333', valColorScheme_fire[3]])
  let boxColorFAR = d3.scaleOrdinal()
    .domain([-1, 0, 1])
    .range([valColorScheme_blue[3], '#cecece', valColorScheme_fire[3]])
  let borders_g = grid_borders_g.append('g')
  borders_g.selectAll('path')
    .data(grid_points)
    .join('path')
      .attr('class', 'right-loc-border')
      .attr('id', (d,i) => `right_loc-${i}`)
      .attr('loc_id', (d,i) => i)
      .attr("d", line) // 应用生成器
      // .attr("fill", (d, i) => boxColorLevel(inforStore.phase_details_infor.phase_raw_level[cur_step][i]))
      .attr("fill", (d, i) => valColor(inforStore.phase_details_infor.phase_raw_val[cur_step][i]))
      .attr("stroke", "#999")
      .attr("stroke-width", 1)
      .attr("opacity", 0.9)
  let pollu_grid_points = grid_points.filter(function(value, index) {
    // let cur_level = inforStore.phase_details_infor.phase_raw_level[cur_step][index]
    // let cur_th = inforStore.dataset_configs.focus_levels[cur_level]
    let cur_val = inforStore.cur_phase_data.phase_raw_data[cur_step][index][0]
    return (cur_val >= inforStore.dataset_configs.focus_th)
  })
  pollu_grid_borders_g.selectAll('path')
    .data(pollu_grid_points)
    .join('path')
      .attr('id', (d,i) => `right_pollu_loc-${i}`)
      .attr('loc_id', (d,i) => i)
      .attr("d", line) // 应用生成器
      .attr("fill", 'none')
      .attr("stroke", (d,i) => '#000')
      .attr("stroke-width", 2)
      .style("stroke-linejoin", "round")
      .style("stroke-linecap", "round")
      .attr("opacity", 0.9)
  let mark_grid = space_layout_g.append('g')
    .attr('transform', `translate(${margin_left+space_layout_w*0.08}, 4) scale(0.85)`)
    .append('path')
      .attr('id', `${view_sel}-mark-grid`)
      .datum([])
      .attr("d", line) // 应用生成器
      .attr("fill", 'none')
      .attr("stroke", 'red')
      .attr("stroke-width", 4)
      .style("stroke-linejoin", "round")
      .style("stroke-linecap", "round")
  let mark_entity = space_layout_g.append('g')
    .attr('transform', `translate(${margin_left+space_layout_w*0.08}, 4) scale(0.85)`)
    .attr('id', `${view_sel}-mark-entity`)
  let cell_len = getDistance(grid_borders[0].border_points[0], grid_borders[0].border_points[1], loc_x_scale, loc_y_scale)
  let subset_marks_g = space_layout_g.append('g')
    .attr('transform', `translate(${margin_left+space_layout_w*0.08}, 4) scale(0.85)`)
  let subset_marks = subset_marks_g.selectAll('circle')
    .data(cur_temporal_focus_locs.value[cur_step])
    .join('circle')
      .attr('transform', (d,i) => `translate(${loc_x_scale(loc_coords_x[d])-cell_len*0.33}, ${loc_y_scale(loc_coords_y[d])-cell_len*0.41})`)
      .attr('cx', 0).attr('cy', 0)
      .attr('r', 4)
      .attr('fill', valColorScheme_red[1])
      .attr('stroke', 'none')

  if (view_linked.value) st_layout_svg.call(zoom_func)
  else st_layout_svg.call(zoom_func_left)
  // let transformState = d3.zoomIdentity;
  if (inforStore.sel_phase_details[cur_model].phase_pred_val[cur_step][0].length == 0) return
  let glyphs_g = grid_borders_g.append('g')
    // .attr('transform', `translate(${margin_left+space_layout_w*0.08}, 4) scale(0.85)`)
  let binary_labels = inforStore.sel_phase_details[cur_model].st_binary_label[cur_step]
  let multi_labels = inforStore.sel_phase_details[cur_model].st_multi_label[cur_step]
  let residuals = inforStore.sel_phase_details[cur_model].st_residuals[cur_step]
  let binary_pred = inforStore.sel_phase_details[cur_model].phase_pred_flag[cur_step]
  let level_pred = inforStore.sel_phase_details[cur_model].phase_pred_level[cur_step]
  let val_pred = inforStore.sel_phase_details[cur_model].phase_pred_val[cur_step]

  let cur_focus_indicator
  if (inforStore.cur_timeline_indicator == 'Residual_abs') {
    cur_focus_indicator = residuals
  } else if (inforStore.cur_timeline_indicator == 'POD' || inforStore.cur_timeline_indicator == 'FAR') {
    cur_focus_indicator = binary_pred
  } else if (inforStore.cur_timeline_indicator == 'Multi_accuracy') {
    cur_focus_indicator = multi_labels
  }
  
  let pie = d3.pie();
  pie.sort(null);
  let forecast_step_num = residuals[0].length
  let arcData = pie(Array(forecast_step_num).fill(1))
  let local_err_max = d3.max(residuals.map(item => d3.max(item)))
  let local_err_min = d3.min(residuals.map(item => d3.min(item)))
  let local_err_abs_max = d3.max([-local_err_min, local_err_max])
  let local_err_range = [-local_err_abs_max.toFixed(2), local_err_abs_max.toFixed(2)]
  // console.log(local_err_range);
  //$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
  //原方案
  let residualsColorScale = d3.scaleSequential(d3.interpolateRdBu)
    .domain(global_residuals_range)
    // .domain(local_err_range)

  // let residualsColorScale = d3.scaleQuantize().range(['#080882', '#181d8d', '#283198', '#3946a3', '#495bae', '#5970b9', '#6984c3', '#7999ce', '#8aaed9', '#9ac2e4', '#aad7ef'])


  // let residualsColorScale = d3.scaleSequential(d3.interpolateBrBG)
    
  if (other_focused_indicators.value.length == 0) {
    let singleIndicatorArc = d3.arc()
      .innerRadius(0)
      .outerRadius(12)
    let eventIndicatorArc = d3.arc()
      .innerRadius(15)
      .outerRadius(20)
    for (let i = 0; i < residuals.length; ++i) {
      let single_glyphs_g = glyphs_g.append('g')
      let single_ring = single_glyphs_g.append('g')
        .attr('class', '.binary-pred-ring')
        .attr('loc_id', i)
        .attr('transform', `translate(${loc_x_scale(loc_coords_x[i])}, ${loc_y_scale(loc_coords_y[i])})`)
          .selectAll('path')
          .data(arcData)
          .join("path")
            .attr('d', singleIndicatorArc)
            .attr('stroke', '#999')
      let grid_flags = inforStore.sel_phase_details[cur_model].phase_pred_flag[cur_step][i]
      if (grid_flags.some(value => value === true) || inforStore.phase_details_infor.phase_raw_flag[cur_step][i]) {
        let event_ring = single_glyphs_g.append('g')
        .attr('class', '.event-pred-ring')
        .attr('loc_id', i)
        .attr('transform', `translate(${loc_x_scale(loc_coords_x[i])}, ${loc_y_scale(loc_coords_y[i])})`)
          .selectAll('path')
          .data(arcData)
          .join("path")
            .attr('d', eventIndicatorArc)
            .attr('stroke', '#999')
            .attr('fill', (d,j) => {
              if (val_pred[i][j] >= inforStore.dataset_configs.focus_th) return '#333'
              else return '#cecece'
            })
      }
      

      if (inforStore.cur_timeline_indicator == 'Residual_abs') {
        single_ring.attr('fill', (d,j) => residualsColorScale(cur_focus_indicator[i][j]))
      } else if (inforStore.cur_timeline_indicator == 'POD' || inforStore.cur_timeline_indicator == 'FAR') {
        single_ring.attr('fill', (d,j) => {
          if (cur_focus_indicator[i][j] == 1) return '#333'
          else if (cur_focus_indicator[i][j] == 0) return '#cecece'
        })
      } else if (inforStore.cur_timeline_indicator == 'Multi_accuracy') {
        single_ring.attr('fill', (d,j) => {
          if (cur_focus_indicator[i][j] == 0) return '#cecece'
          else if (cur_focus_indicator[i][j] == 1) return valColorScheme_fire[3]
          else if (cur_focus_indicator[i][j] == -1) return valColorScheme_blue[3]
          else return '#cecece'
        })
      }
    }
  }

  if (other_focused_indicators.value.length == 1) {
    let other_indicator
    if (other_focused_indicators.value[0] == 'POD/FAR') other_indicator = binary_pred
    if (other_focused_indicators.value[0] == 'Multi_accuracy') other_indicator = multi_labels
    if (other_focused_indicators.value[0] == 'Residual_abs') other_indicator = residuals
    
    let doubleIndicatorArc_1 = d3.arc()
      .innerRadius(0)
      .outerRadius(12)
    let doubleIndicatorArc_2 = d3.arc()
      .innerRadius(16)
      .outerRadius(24)
    for (let i = 0; i < residuals.length; ++i) {
      let glyphs_1_g = glyphs_g.append('g')
      let glyphs_1_ring = glyphs_1_g.append('g')
        .attr('class', '.glyphs-1-ring')
        .attr('loc_id', i)
        .attr('transform', `translate(${loc_x_scale(loc_coords_x[i])}, ${loc_y_scale(loc_coords_y[i])})`)
          .selectAll('path')
          .data(arcData)
          .join("path")
            .attr('d', doubleIndicatorArc_1)
            .attr('stroke', '#999')
      if (inforStore.cur_timeline_indicator == 'Residual_abs') {
        glyphs_1_ring.attr('fill', (d,j) => residualsColorScale(cur_focus_indicator[i][j]))
      } else if (inforStore.cur_timeline_indicator == 'POD' || inforStore.cur_timeline_indicator == 'FAR') {
        glyphs_1_ring.attr('fill', (d,j) => {
          if (cur_focus_indicator[i][j] == 1) return '#333'
          else if (cur_focus_indicator[i][j] == 0) return '#cecece'
        })
      } else if (inforStore.cur_timeline_indicator == 'Multi_accuracy') {
        glyphs_1_ring.attr('fill', (d,j) => {
          if (cur_focus_indicator[i][j] == 0) return '#cecece'
          else if (cur_focus_indicator[i][j] == 1) return valColorScheme_fire[3]
          else if (cur_focus_indicator[i][j] == -1) return valColorScheme_blue[3]
          else return '#cecece'
        })
      }
      let glyphs_2_g = glyphs_g.append('g')
      let glyphs_2_ring = glyphs_2_g.append('g')
        .attr('class', '.glyphs-2-ring')
        .attr('loc_id', i)
        .attr('transform', `translate(${loc_x_scale(loc_coords_x[i])}, ${loc_y_scale(loc_coords_y[i])})`)
          .selectAll('path')
          .data(arcData)
          .join("path")
            .attr('d', doubleIndicatorArc_2)
            .attr('stroke', '#999')
      if (other_focused_indicators.value[0] == 'Residual_abs') {
        glyphs_2_ring.attr('fill', (d,j) => residualsColorScale(other_indicator[i][j]))
      } else if (other_focused_indicators.value[0] == 'POD/FAR') {
        glyphs_2_ring.attr('fill', (d,j) => {
          if (other_indicator[i][j] == 1) return '#333'
          else if (other_indicator[i][j] == 0) return '#cecece'
        })
      } else if (other_focused_indicators.value[0] == 'Multi_accuracy') {
        glyphs_2_ring.attr('fill', (d,j) => {
          if (other_indicator[i][j] == 0) return '#cecece'
          else if (other_indicator[i][j] == 1) return valColorScheme_fire[3]
          else if (other_indicator[i][j] == -1) return valColorScheme_blue[3]
          else return '#cecece'
        })
      }
    }
  }

  if (other_focused_indicators.value.length == 2) {
    let other_indicator_1, other_indicator_2
    if (other_focused_indicators.value[0] == 'POD/FAR') other_indicator_1 = binary_pred
    if (other_focused_indicators.value[0] == 'Multi_accuracy') other_indicator_1 = multi_labels
    if (other_focused_indicators.value[0] == 'Residual_abs') other_indicator_1 = residuals
    if (other_focused_indicators.value[1] == 'POD/FAR') other_indicator_2 = binary_pred
    if (other_focused_indicators.value[1] == 'Multi_accuracy') other_indicator_2 = multi_labels
    if (other_focused_indicators.value[1] == 'Residual_abs') other_indicator_2 = residuals
    
    let doubleIndicatorArc_1 = d3.arc()
      .innerRadius(0)
      .outerRadius(8)
    let doubleIndicatorArc_2 = d3.arc()
      .innerRadius(10)
      .outerRadius(16)
    let doubleIndicatorArc_3 = d3.arc()
      .innerRadius(18)
      .outerRadius(24)
    
    for (let i = 0; i < residuals.length; ++i) {
      let glyphs_1_g = glyphs_g.append('g')
      let glyphs_1_ring = glyphs_1_g.append('g')
        .attr('class', '.glyphs-1-ring')
        .attr('loc_id', i)
        .attr('transform', `translate(${loc_x_scale(loc_coords_x[i])}, ${loc_y_scale(loc_coords_y[i])})`)
          .selectAll('path')
          .data(arcData)
          .join("path")
            .attr('d', doubleIndicatorArc_1)
            .attr('stroke', '#999')
      if (inforStore.cur_timeline_indicator == 'Residual_abs') {
        glyphs_1_ring.attr('fill', (d,j) => residualsColorScale(cur_focus_indicator[i][j]))
      } else if (inforStore.cur_timeline_indicator == 'POD' || inforStore.cur_timeline_indicator == 'FAR') {
        glyphs_1_ring.attr('fill', (d,j) => {
          if (cur_focus_indicator[i][j] == 1) return '#333'
          else if (cur_focus_indicator[i][j] == 0) return '#cecece'
        })
      } else if (inforStore.cur_timeline_indicator == 'Multi_accuracy') {
        glyphs_1_ring.attr('fill', (d,j) => {
          if (cur_focus_indicator[i][j] == 0) return '#cecece'
          else if (cur_focus_indicator[i][j] == 1) return valColorScheme_fire[3]
          else if (cur_focus_indicator[i][j] == -1) return valColorScheme_blue[3]
          else return '#cecece'
        })
      }
      let glyphs_2_g = glyphs_g.append('g')
      let glyphs_2_ring = glyphs_2_g.append('g')
        .attr('class', '.glyphs-3-ring')
        .attr('loc_id', i)
        .attr('transform', `translate(${loc_x_scale(loc_coords_x[i])}, ${loc_y_scale(loc_coords_y[i])})`)
          .selectAll('path')
          .data(arcData)
          .join("path")
            .attr('d', doubleIndicatorArc_2)
            .attr('stroke', '#999')
      if (other_focused_indicators.value[0] == 'Residual_abs') {
        glyphs_2_ring.attr('fill', (d,j) => residualsColorScale(other_indicator_1[i][j]))
      } else if (other_focused_indicators.value[0] == 'POD/FAR') {
        glyphs_2_ring.attr('fill', (d,j) => {
          if (other_indicator_1[i][j] == 1) return '#333'
          else if (other_indicator_1[i][j] == 0) return '#cecece'
        })
      } else if (other_focused_indicators.value[0] == 'Multi_accuracy') {
        glyphs_2_ring.attr('fill', (d,j) => {
          if (other_indicator_1[i][j] == 0) return '#cecece'
          else if (other_indicator_1[i][j] == 1) return valColorScheme_fire[3]
          else if (other_indicator_1[i][j] == -1) return valColorScheme_blue[3]
          else return '#cecece'
        })
      }
      let glyphs_3_g = glyphs_g.append('g')
      let glyphs_3_ring = glyphs_3_g.append('g')
        .attr('class', '.glyphs-3-ring')
        .attr('loc_id', i)
        .attr('transform', `translate(${loc_x_scale(loc_coords_x[i])}, ${loc_y_scale(loc_coords_y[i])})`)
          .selectAll('path')
          .data(arcData)
          .join("path")
            .attr('d', doubleIndicatorArc_3)
            .attr('stroke', '#999')
      if (other_focused_indicators.value[1] == 'Residual_abs') {
        glyphs_3_ring.attr('fill', (d,j) => residualsColorScale(other_indicator_2[i][j]))
      } else if (other_focused_indicators.value[1] == 'POD/FAR') {
        glyphs_3_ring.attr('fill', (d,j) => {
          if (other_indicator_2[i][j] == 1) return '#333'
          else if (other_indicator_2[i][j] == 0) return '#cecece'
        })
      } else if (other_focused_indicators.value[1] == 'Multi_accuracy') {
        glyphs_3_ring.attr('fill', (d,j) => {
          if (other_indicator_2[i][j] == 0) return '#cecece'
          else if (other_indicator_2[i][j] == 1) return valColorScheme_fire[3]
          else if (other_indicator_2[i][j] == -1) return valColorScheme_blue[3]
          else return '#cecece'
        })
      }
    }
  }

  let resi_legend_len = 120
  let legend_y_shift = 24, legend_x_shift_1 = 60, legend_x_shift_2 = 60
  let cur_x_shift = 60

  let pollutionValScale = d3.scaleLinear()
      .domain([0, resi_legend_len])
      .range(['#fff', '#4a1486'])
  let pollution_val_legend = st_layout_svg.append('g')
    .attr('transform', `translate(230, ${space_layout_h-25})`)
  let x_shift_pollu_val = 840
  pollution_val_legend.selectAll('rect')
    .data(Array(resi_legend_len).fill(1))
    .join('rect')
      .attr('x', (d,i) => i+x_shift_pollu_val)
      .attr('y', 0)
      .attr('width', 1)
      .attr('height', 12)
      .attr('fill', (d,i) => pollutionValScale(i))
  pollution_val_legend.append('text')
    .attr('x', x_shift_pollu_val-60)
    .attr('y', 11)
    .attr('text-anchor', 'middle')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('Focus Value')
  pollution_val_legend.append('text')
    .attr('x', -4+x_shift_pollu_val)
    .attr('y', 11)
    .attr('text-anchor', 'end')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('0')
  pollution_val_legend.append('text')
    .attr('x', resi_legend_len+4+x_shift_pollu_val)
    .attr('y', 11)
    .attr('text-anchor', 'start')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('500')
  
  let indicatorScale = d3.scaleLinear()
      .domain([0, resi_legend_len])
      //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
      //原来：
      // .range(valColorScheme_fire)
      .range(['#4292c6',"#fd8d3c"])

  let indicator_legend = st_layout_svg.append('g')
    .attr('transform', `translate(430, ${space_layout_h-25})`)
  let x_shift_indicator = 920
  indicator_legend.selectAll('rect')
    .data(Array(resi_legend_len).fill(1))
    .join('rect')
      .attr('x', (d,i) => i+x_shift_indicator)
      .attr('y', 0)
      .attr('width', 1)
      .attr('height', 12)
      .attr('fill', (d,i) => indicatorScale(i))
  indicator_legend.append('text')
    .attr('x', x_shift_indicator-60)
    .attr('y', 11)
    .attr('text-anchor', 'middle')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text(`${inforStore.cur_timeline_indicator}`)
  indicator_legend.append('text')
    .attr('x', -4+x_shift_indicator)
    .attr('y', 11)
    .attr('text-anchor', 'end')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('0')
  indicator_legend.append('text')
    .attr('x', resi_legend_len+4+x_shift_indicator)
    .attr('y', 11)
    .attr('text-anchor', 'start')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text(() => {
      if (inforStore.cur_timeline_indicator == 'Residual_abs')
        return `${parseFloat(inforStore.sel_phase_details[cur_model].temporal_residuals_abs_max.toFixed(2))}`
      else if (inforStore.cur_timeline_indicator == 'Multi_accuracy')
        return 1
      else if (inforStore.cur_timeline_indicator == 'POD')
        return 1
      else if (inforStore.cur_timeline_indicator == 'FAR')
        return 1
    })

  if (inforStore.cur_timeline_indicator.includes('Residual_abs') || other_focused_indicators.value.includes('Residual_abs')) {
    let resiLegendScale = d3.scaleSequential(d3.interpolateRdBu)
      .domain([0, resi_legend_len]);
    // d3.scaleQuantize()
    // .range(['#080882', '#181d8d', '#283198', '#3946a3', '#495bae', '#5970b9', '#6984c3', '#7999ce', '#8aaed9', '#9ac2e4', '#aad7ef'])
      
    let residualsColorScale = d3.scaleSequential(d3.interpolateRdBu)
      .domain(global_residuals_range)
      
    let residual_legend = st_layout_svg.append('g')
      .attr('transform', `translate(${cur_x_shift}, ${legend_y_shift})`)
    residual_legend.selectAll('rect')
      .data(Array(resi_legend_len).fill(1))
      .join('rect')
        .attr('x', (d,i) => i)
        .attr('y', 15)
        .attr('width', 1)
        .attr('height', 12)
        .attr('fill', (d,i) => resiLegendScale(i))
    residual_legend.append('text')
      .attr('x', resi_legend_len * 0.5)
      .attr('y', 7)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .attr('fill', '#333')
      .text('Residual')
    residual_legend.append('text')
      .attr('x', -4)
      .attr('y', 26)
      .attr('text-anchor', 'end')
      .style('font-size', '12px')
      .attr('fill', '#333')
      // .text(`${local_err_range[0]}`)
      .text(`${global_residuals_range[0]}`)
    residual_legend.append('text')
      .attr('x', resi_legend_len+4)
      .attr('y', 26)
      .attr('text-anchor', 'start')
      .style('font-size', '12px')
      .attr('fill', '#333')
      // .text(`${local_err_range[1]}`)
      .text(`${global_residuals_range[1]}`)
    cur_x_shift += resi_legend_len+85
  }
  if (inforStore.cur_timeline_indicator.includes('POD/FAR') || other_focused_indicators.value.includes('POD/FAR')) {
    let binary_label_legend = st_layout_svg.append('g')
      .attr('transform', `translate(${cur_x_shift}, ${legend_y_shift})`)
    let label_block_w = 60, box_w = 12
    binary_label_legend.append('text')
      .attr('x', 134)
      .attr('y', -8)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .attr('fill', '#333')
      .text('Dichotomous Label')
    binary_label_legend.append('rect')
      .attr('x', box_w+30).attr('y', 0)
      .attr('width', box_w)
      .attr('height', box_w)
      .attr('fill', '#333') // neg correct
    binary_label_legend.append('text')
      .attr('x', box_w*2 + 33)
      .attr('y', 11)
      .attr('text-anchor', 'start')
      .style('font-size', '12px')
      .attr('fill', '#333')
      .text('Forecast Yes')
    binary_label_legend.append('rect')
      .attr('x', box_w*2+123).attr('y', 0)
      .attr('width', box_w)
      .attr('height', box_w)
      .attr('fill', '#cecece') // false alarm
    binary_label_legend.append('text')
      .attr('x', box_w*3 + 126)
      .attr('y', 11)
      .attr('text-anchor', 'start')
      .style('font-size', '12px')
      .attr('fill', '#333')
      .text('Forecast No')
    cur_x_shift += 300
  }
  if (inforStore.cur_timeline_indicator.includes('Multi_accuracy') || other_focused_indicators.value.includes('Multi_accuracy')) {
    let multi_label_legend = st_layout_svg.append('g')
      .attr('transform', `translate(${cur_x_shift}, ${legend_y_shift})`)
    let label_block_w = 60, box_w = 12
    multi_label_legend.append('text')
      .attr('x', 90)
      .attr('y', -8)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .attr('fill', '#333')
      .text('Multi-category Label')
    multi_label_legend.append('rect')
      .attr('x', 0).attr('y', 0)
      .attr('width', box_w)
      .attr('height', box_w)
      .attr('fill', '#cecece')
    multi_label_legend.append('text')
      .attr('x', box_w + 3)
      .attr('y', 11)
      .attr('text-anchor', 'start')
      .style('font-size', '12px')
      .attr('fill', '#333')
      .text('Accurate')  // 估计准确 
    multi_label_legend.append('rect')
      .attr('x', box_w + 62).attr('y', 0)
      .attr('width', box_w)
      .attr('height', box_w)
      .attr('fill', valColorScheme_fire[3])
    multi_label_legend.append('text')
      .attr('x', box_w*2 + 65)
      .attr('y', 11)
      .attr('text-anchor', 'start')
      .style('font-size', '12px')
      .attr('fill', '#333')
      .text('Higher')
    multi_label_legend.append('rect')
      .attr('x', box_w*2 + 110).attr('y', 0)
      .attr('width', box_w)
      .attr('height', box_w)
      .attr('fill', valColorScheme_blue[3])
    multi_label_legend.append('text')
      .attr('x', box_w*3 + 113)
      .attr('y', 11)
      .attr('text-anchor', 'start')
      .style('font-size', '12px')
      .attr('fill', '#333')
      .text('Lower')
  }
}

let cur_hover_event = ref([-1, -1])
let event_prev_loc_ids = ref([])
let event_cur_loc_ids = ref([])

watch (() => cur_hover_event.value, (oldValue, newValue) => {
  drawTimeEventBar()
})

function drawTimeEventBar() {
  d3.select('#time-event-bar').selectAll('*').remove()
  let sel_phase_id = inforStore.sel_phase_details[inforStore.cur_focused_model].sel_phase_id
  let space_layout_w = 1530, space_layout_h = 176, time_bar_h = 10
  let margin_left = 50, margin_right = 10
  let st_layout_w = space_layout_w + margin_left + margin_right
  let st_layout_h = space_layout_h
  let st_layout_svg = d3.select('#time-event-bar')
    .attr('width', st_layout_w)
    .attr('height', st_layout_h)
    .on("mouseout", (event) => {
        event_cur_loc_ids.value = []
        event_prev_loc_ids.value = []
        cur_hover_event.value = [-1, -1]
      })
  let white_rate = 0.06
  let phase_len = inforStore.st_phase_events.phases[inforStore.cur_focused_model][sel_phase_id].end - inforStore.st_phase_events.phases[inforStore.cur_focused_model][sel_phase_id].start
  let step_width = (space_layout_w-20) / phase_len
  
  // 绘制事件bar，略缩图
  let events_scope = 20
  let cell_w = 70, cell_h = 70
  let evevt_bar_y_shift = 130
  if (loc_coords_x.length > 100) {
    events_scope = 9
    cell_w = 80
    cell_h = 80
    evevt_bar_y_shift += 10
  }
  let cur_event_end = cur_event_start + events_scope
  
  let events_bar_g = st_layout_svg.append('g')
    .attr('transform', `translate(${margin_left+30}, ${space_layout_h-evevt_bar_y_shift})`)
  let cell_x_scale = d3.scaleLinear()
        .domain([Math.min(...loc_coords_x), Math.max(...loc_coords_x)])
        .range([cell_w*(white_rate), cell_w*(1-white_rate)])
  let cell_y_scale = d3.scaleLinear()
        .domain([Math.min(...loc_coords_y), Math.max(...loc_coords_y)])
        .range([cell_h*(1-white_rate), cell_h*white_rate])
  let cellValColor = d3.scaleOrdinal()
    .domain(d3.range(inforStore.dataset_configs.focus_levels.length))
    .range(['#fcfbfd', '#efedf5', '#dadaeb', '#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#4a1486'])

  let phase_raw_data = inforStore.cur_phase_data.phase_raw_data.slice(cur_event_start, cur_event_end)
  let phase_raw_level = inforStore.phase_details_infor.phase_raw_level.slice(cur_event_start, cur_event_end)
  let phase_raw_infor = []
  for (let i = 0; i < phase_raw_data.length; i++) {
    phase_raw_infor.push([])
    for (let j = 0; j < phase_raw_data[i].length; ++j) {
      phase_raw_infor[i].push({
        'val': phase_raw_data[i][j][0],
        'level': phase_raw_level[i][j]
      })
    }
  }

  for (let i = 0; i < phase_raw_infor.length; ++i) {
    let data_cells_tmp = events_bar_g.append('g')
        .attr('transform', `translate(${i*(cell_w+5)}, 0)`)
        .style('cursor', 'pointer')
        .attr('id', () => `data-cell-${i}`)
        .attr('class', `data-cells`)
        .on('click', (e,d) => {
          cur_sel_event_step = d3.select(e.target).attr('cur_event_step')
          cur_sel_step.value = cur_event_start + parseInt(cur_sel_event_step)
          data_cells_tmp.selectAll('rect').attr('stroke', '#bababa')
          data_cells_tmp.select(`#data-cell-${cur_sel_event_step}`)
            .attr('stroke', '#333')
          let event_header_g = d3.selectAll('.event-header')
          for (let j = 0; j < event_header_g._groups[0].length; ++j) {
            let cur_event_header = d3.select(event_header_g._groups[0][j])
            cur_event_header.select('rect')
              .attr('fill', () => {
              if (cur_event_header.attr('event-step') == cur_sel_event_step) return '#333'
              else return '#cecece'
            })
            cur_event_header.select('text')
              .attr('fill', () => {
                if (cur_event_header.attr('event-step') == cur_sel_event_step) return '#fff'
                else return '#333'
              })
          }
          
          d3.select('#cur_step_mark').attr('x', cur_sel_event_step*step_width)
        })
    data_cells_tmp.append('rect')
      .attr('cur_event_step', () => i)
      .attr('x', 0).attr('y', 0)
      .attr('width', cell_w).attr('height', cell_h)
      .attr('fill', '#fff')
      .attr('stroke', () => {
        if (i == cur_sel_event_step) return '#333'
        else return '#bababa'
      })
    data_cells_tmp.selectAll('circle')
      .data(phase_raw_infor[i])
      .join('circle')
        .attr('cx', (d,j) => cell_x_scale(loc_coords_x[j]))
        .attr('cy', (d,j) => cell_y_scale(loc_coords_y[j]))
        .attr('loc_id', (d,j) => j)
        .attr('r', 2)
        .attr('fill', (d,j) => cellValColor(d.level))
        .attr('stroke', (d,j) => {
          if ((i == cur_hover_event.value[0]) && event_cur_loc_ids.value.includes(j)) return '#0097A7'
          if ((i > 0) && (cur_hover_event.value[0] == i+1) && event_prev_loc_ids.value.includes(j)) return '#0097A7'
          if (d.val >= inforStore.dataset_configs.focus_th) return '#333'
          else return 'none'
        })
  }

  let cell_header_h = 14
  let cur_phase_id = inforStore.cur_phase_sorted_id
  let phase_events = inforStore.st_phase_events.phases[inforStore.cur_focused_model][cur_phase_id].phase_events

  let cur_events = []
  for (let i = 0; i < events_scope; ++i) {
    cur_events.push([])
    for (let j = 0; j < phase_events.length; ++j) {
      if (phase_events[j][1] == i+cur_event_start) {
        cur_events[i].push(phase_events[j])
      }
      // if (phase_events[j][1] > i+cur_event_start) break
    }
  }
  
  // 建立事件header
  let focus_entities = inforStore.st_phase_events.phases[inforStore.cur_focused_model][cur_phase_id].focus_entities
  let events_headers_g = st_layout_svg.append('g')
    .attr('transform', `translate(${margin_left+30}, ${space_layout_h-evevt_bar_y_shift})`)
  for (let i = 0; i < cur_events.length; ++i) {
    let event_headers_g = events_headers_g.append('g')
      .attr('transform', () => `translate(${i*(cell_w+5)}, 0)`)
      .attr('event-step', () => i)
    for (let j = 0; j < cur_events[i].length; ++j) {
      let event_header_g = event_headers_g.append('g')
          .attr('class', 'event-header')
          .attr('event-step', () => i)
          .style('cursor', 'pointer')
          .attr('transform', () => `translate(0, ${-(j+1)*cell_header_h})`)
          .on("mouseover", (e,d) => {
            let pre_entities = [], cur_entities = []
            if (cur_events[i][j][2] && Array.isArray(cur_events[i][j][2])) {
              for (let entity_id of cur_events[i][j][2]) {
                pre_entities.push(focus_entities[cur_events[i][j][1]-1][entity_id])
              }
            } else if (Number.isInteger(cur_events[i][j][2])) {
              pre_entities = [focus_entities[cur_events[i][j][1]-1][cur_events[i][j][2]]]
            }
            if (cur_events[i][j][3] && Array.isArray(cur_events[i][j][3])) {
              for (let entity_id of cur_events[i][j][3]) {
                cur_entities.push(focus_entities[cur_events[i][j][1]][entity_id])
              }
            } else if (Number.isInteger(cur_events[i][j][3])) {
              cur_entities = [focus_entities[cur_events[i][j][1]][cur_events[i][j][3]]]
            }

            let prev_loc_ids = [], cur_loc_ids = []
            for (let entity of pre_entities) {
              prev_loc_ids = prev_loc_ids.concat(entity.loc_ids)  
            }
            for (let entity of cur_entities) {
              cur_loc_ids = cur_loc_ids.concat(entity.loc_ids)  
            }
            event_cur_loc_ids.value = cur_loc_ids
            event_prev_loc_ids.value = prev_loc_ids
            cur_hover_event.value = [i, j]
          })
      event_header_g.append('rect')
        .attr('x', -0.5).attr('y', 0)
        .attr('width', cell_w+1).attr('height', cell_header_h)
        .attr('fill', () => {
          if ((cur_hover_event.value[0] == i) && (cur_hover_event.value[1] == j)) return '#0097A7'
          if (cur_events[i][j][1]-cur_event_start == cur_sel_event_step) return '#333'
          else return '#cecece'
        })
      event_header_g.append('text')
        .attr('x', cell_w/2).attr('y', 10)
        .attr('text-anchor', 'middle')
        .style('font-size', '9px')
        .attr('fill', () => {
          if ((cur_hover_event.value[0] == i) && (cur_hover_event.value[1] == j)) return '#fff'
          if (cur_events[i][j][1]-cur_event_start == cur_sel_event_step) return '#fff'
          else return '#333'
        })
        .text(() => cur_events[i][j][4].type)
    }
  }
  
  let time_bar_g = st_layout_svg.append('g')
    .attr('transform', `translate(${margin_left+30}, ${space_layout_h-54})`)
  let time_axis_bar = time_bar_g.append('g')
  let temporal_ids = []
  for (let i = 0; i < phase_len; i++) {
    temporal_ids.push(i);
  }
  let temporal_err_scale = d3.scaleLinear()
    .domain([0, inforStore.sel_phase_details[inforStore.cur_focused_model].temporal_residuals_abs_max])
    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    //原来：
    .range(valColorScheme_fire)
    // .range(['#4292c6','#fd8d3c'])


  // let temporal_err_scale = d3.scaleLinear()
  // let temporal_err_scale = d3.scaleSequential(d3.interpolateGreens)
  //   .domain([0, inforStore.sel_phase_details.temporal_residuals_abs_max])
    // .range(['#eee', '#000'])
    // .range(valColorScheme_fire)
  let temporal_rate_scale = d3.scaleQuantize()
    .domain([0, 1])
    .range(valColorScheme_fire)
  // let temporal_rate_scale = d3.scaleLinear()
  // let temporal_rate_scale = d3.scaleSequential(d3.interpolateGreens)
  //   .domain([0, 1])
    // .range(['#eee', '#000'])
    // .range(valColorScheme_fire)
  // let time_x_scale = d3.scaleLinear()
  //   .domain(temporal_ids)
  //   .range([0, space_layout_w])
  let phase_infor = inforStore.st_phase_events.phases[inforStore.cur_focused_model][sel_phase_id]
  let scope_size = inforStore.cur_focused_scope[1] - inforStore.cur_focused_scope[0] + 1
  let focusColormap = d3.scaleSequential(d3.interpolateGreys)
    .domain([0, d3.max(cur_temporal_focus_cnt.value) / (loc_coords_x.length * scope_size)])
  time_axis_bar.append('text')
    .attr('x', -4)
    .attr('y', 8)
    .attr('text-anchor', 'end')
    .style('font-size', '11px')
    .attr('fill', '#333')
    .text('Focus_cnt')
  time_axis_bar.append('g').attr('id', 'focus-time-axis')
    .selectAll('rect')
    .data(temporal_ids)
    .join('rect')
      .attr('id', d => `focus_unit-${d}`)
      .attr('x', d => d * step_width)
      .attr('y', 0)
      .attr('width', step_width)
      .attr('height', time_bar_h)
      .attr('fill', d => {
        // console.log(cur_temporal_focus_cnt.value[d], loc_coords_x.length * scope_size);
        if (inforStore.cur_sel_event_steps.length == 0) return '#fff'
        else if (inforStore.cur_sel_event_steps.includes(d)) return '#333'
        else return '#fff'
      })
      .attr('stroke', '#999')
      .on('click', (e) => {
        let target_id = d3.select(e.target).attr('id').split('-')[1]
        // console.log(inforStore.sel_phase_details.time_pod[target_id], inforStore.sel_phase_details.time_far[target_id]);
        cur_event_start = Math.max(0, Math.min(temporal_ids.length-events_scope, target_id));
        cur_event_end = cur_event_start + events_scope
        cur_sel_event_step = target_id - cur_event_start
        d3.select('#cur_step_mark').attr('x', cur_sel_event_step*step_width)
        phase_raw_infor = []
        // if (inforStore.sel_phase_details.phase_raw_level)
        phase_raw_data = inforStore.cur_phase_data.phase_raw_data.slice(cur_event_start, cur_event_end)
        phase_raw_level = inforStore.phase_details_infor.phase_raw_level.slice(cur_event_start, cur_event_end)
        
        for (let i = 0; i < phase_raw_data.length; i++) {
          phase_raw_infor.push([])
          for (let j = 0; j < phase_raw_data[i].length; ++j) {
            phase_raw_infor[i].push({
              'val': phase_raw_data[i][j][0],
              'level': phase_raw_level[i][j]
            })
          }
        }
        time_bar_g.select('#start-slider')
          .attr('transform', `translate(${cur_event_start * step_width}, ${time_bar_h*4+3})`)
        let data_cells_tmp = events_bar_g.selectAll('g')
          .data(phase_raw_infor)
          .join('g')
        for (let i = 0; i < phase_raw_infor.length; ++i) {
          d3.select(data_cells_tmp._groups[0][i]).selectAll('circle')
            .data(phase_raw_infor[i])
            .join('circle')
              .attr('cx', (d,j) => cell_x_scale(loc_coords_x[j]))
              .attr('cy', (d,j) => cell_y_scale(loc_coords_y[j]))
              .attr('loc_id', (d,j) => j)
              .attr('r', 2)
              .attr('fill', (d,j) => cellValColor(d.level))
              .attr('stroke', (d,j) => {
                if ((i == cur_hover_event.value[0]) && event_cur_loc_ids.value.includes(j)) return '#0097A7'
                if ((i > 0) && (cur_hover_event.value[0] == i+1) && event_prev_loc_ids.value.includes(j)) return '#0097A7'
                if (d.val >= inforStore.dataset_configs.focus_th) return '#333'
                else return 'none'
              })
        }
        data_cells_tmp.selectAll('rect')
          .attr('stroke', '#bababa')
        data_cells_tmp.select(`#data-cell-${cur_sel_event_step}`)
          .attr('stroke', '#333')
        cur_sel_step.value = target_id
        cur_events = []
        for (let i = 0; i < events_scope; ++i) {
          cur_events.push([])
          for (let j = 0; j < phase_events.length; ++j) {
            if (phase_events[j][1] == i+cur_event_start) {
              cur_events[i].push(phase_events[j])
            }
            // if (phase_events[j][1] > i+cur_event_start) break
          }
        }
        events_headers_g.selectAll('*').remove()
        for (let i = 0; i < cur_events.length; ++i) {
          let event_headers_g = events_headers_g.append('g')
            .attr('transform', () => `translate(${i*(cell_w+5)}, 0)`)
            .attr('event-step', () => i)
          for (let j = 0; j < cur_events[i].length; ++j) {
            let event_header_g = event_headers_g.append('g')
                .attr('class', 'event-header')
                .attr('event-step', () => i)
                .style('cursor', 'pointer')
                .attr('transform', () => `translate(0, ${-(j+1)*cell_header_h})`)
                .on("mouseover", (e,d) => {
                  let pre_entities = [], cur_entities = []
                  if (cur_events[i][j][2] && Array.isArray(cur_events[i][j][2])) {
                    for (let entity_id of cur_events[i][j][2]) {
                      pre_entities.push(focus_entities[cur_events[i][j][1]-1][entity_id])
                    }
                  } else if (Number.isInteger(cur_events[i][j][2])) {
                    pre_entities = [focus_entities[cur_events[i][j][1]-1][cur_events[i][j][2]]]
                  }
                  if (cur_events[i][j][3] && Array.isArray(cur_events[i][j][3])) {
                    for (let entity_id of cur_events[i][j][3]) {
                      cur_entities.push(focus_entities[cur_events[i][j][1]][entity_id])
                    }
                  } else if (Number.isInteger(cur_events[i][j][3])) {
                    cur_entities = [focus_entities[cur_events[i][j][1]][cur_events[i][j][3]]]
                  }

                  let prev_loc_ids = [], cur_loc_ids = []
                  for (let entity of pre_entities) {
                    prev_loc_ids = prev_loc_ids.concat(entity.loc_ids)  
                  }
                  for (let entity of cur_entities) {
                    cur_loc_ids = cur_loc_ids.concat(entity.loc_ids)  
                  }
                  event_cur_loc_ids.value = cur_loc_ids
                  event_prev_loc_ids.value = prev_loc_ids
                  cur_hover_event.value = [i, j]
                })
            event_header_g.append('rect')
              .attr('x', -0.5).attr('y', 0)
              .attr('width', cell_w+1).attr('height', cell_header_h)
              .attr('fill', () => {
                if ((cur_hover_event.value[0] == i) && (cur_hover_event.value[1] == j)) return '#0097A7'
                if (cur_events[i][j][1]-cur_event_start == cur_sel_event_step) return '#333'
                else return '#cecece'
              })
            event_header_g.append('text')
              .attr('x', cell_w/2).attr('y', 10)
              .attr('text-anchor', 'middle')
              .style('font-size', '9px')
              .attr('fill', () => {
                if ((cur_hover_event.value[0] == i) && (cur_hover_event.value[1] == j)) return '#fff'
                if (cur_events[i][j][1]-cur_event_start == cur_sel_event_step) return '#fff'
                else return '#333'
              })
              .text(() => cur_events[i][j][4].type)
          }
        }

      })

  // let cntColormap = d3.scaleLinear()
  //   .domain([0, 1])
  //   .range([valColorScheme_blue[0], valColorScheme_blue[valColorScheme_blue.length - 1]])
  let cntLenMap = d3.scaleLinear()
    .domain([0, 1])
    .range([time_bar_h*2, 0])
    // .range(['#eee', '#000'])
  // let valColor = d3.scaleQuantile()
  //   .domain([0, 500])
  //   .range(['#fcfbfd', '#efedf5', '#dadaeb', '#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#4a1486'])
  let valColor = d3.scaleLinear()
    // .domain([inforStore.dataset_configs.focus_th, 500])
    .domain([0, 500])
    .range(['#fff', '#4a1486'])
    // .range(['#efedf5', '#4a1486'])
    // .range(['#eee', '#000'])
  // let valColor = d3.scaleSequential(d3.interpolateBlues)
  //   .domain([0, 500])
  time_axis_bar.append('text')
    .attr('x', -4)
    .attr('y', 24)
    .attr('text-anchor', 'end')
    .style('font-size', '11px')
    .attr('fill', '#333')
    .text('Foucs_val')
  time_axis_bar.append('g').selectAll('rect')
    .data(temporal_ids)
    .join('rect')
      .attr('id', d => `cnt_unit-${d}`)
      .attr('x', d => d * step_width)
      // .attr('y', d => time_bar_h + cntLenMap(phase_infor.time_focus_cnt[d]) + 1)
      .attr('y', d => time_bar_h + cntLenMap(phase_infor.time_focus_cnt[d])/2 + 1)
      .attr('width', step_width)
      .attr('height', d => time_bar_h*2-cntLenMap(phase_infor.time_focus_cnt[d]))
      .attr('fill', d => {
        // if (phase_infor.time_focus_cnt[d] == 0) return '#fff'
        // else return cntColormap(phase_infor.time_focus_cnt[d])
        return valColor(phase_infor.time_focus_val[d])
      })
      .attr('stroke', '#999')
      .on('click', (e) => {
        let target_id = d3.select(e.target).attr('id').split('-')[1]
        // console.log(inforStore.sel_phase_details.time_pod[target_id], inforStore.sel_phase_details.time_far[target_id]);
        cur_event_start = Math.max(0, Math.min(temporal_ids.length-events_scope, target_id));
        cur_event_end = cur_event_start + events_scope
        cur_sel_event_step = target_id - cur_event_start
        d3.select('#cur_step_mark').attr('x', cur_sel_event_step*step_width)
        phase_raw_infor = []
        
        phase_raw_data = inforStore.cur_phase_data.phase_raw_data.slice(cur_event_start, cur_event_end)
        phase_raw_level = inforStore.phase_details_infor.phase_raw_level.slice(cur_event_start, cur_event_end)
        
        for (let i = 0; i < phase_raw_data.length; i++) {
          phase_raw_infor.push([])
          for (let j = 0; j < phase_raw_data[i].length; ++j) {
            phase_raw_infor[i].push({
              'val': phase_raw_data[i][j][0],
              'level': phase_raw_level[i][j]
            })
          }
        }
        time_bar_g.select('#start-slider')
          .attr('transform', `translate(${cur_event_start * step_width}, ${time_bar_h*4+3})`)
        let data_cells_tmp = events_bar_g.selectAll('g')
          .data(phase_raw_infor)
          .join('g')
        for (let i = 0; i < phase_raw_infor.length; ++i) {
          d3.select(data_cells_tmp._groups[0][i]).selectAll('circle')
            .data(phase_raw_infor[i])
            .join('circle')
              .attr('cx', (d,j) => cell_x_scale(loc_coords_x[j]))
              .attr('cy', (d,j) => cell_y_scale(loc_coords_y[j]))
              .attr('loc_id', (d,j) => j)
              .attr('r', 2)
              .attr('fill', (d,j) => cellValColor(d.level))
              .attr('stroke', (d,j) => {
                if ((i == cur_hover_event.value[0]) && event_cur_loc_ids.value.includes(j)) return '#0097A7'
                if ((i > 0) && (cur_hover_event.value[0] == i+1) && event_prev_loc_ids.value.includes(j)) return '#0097A7'
                if (d.val >= inforStore.dataset_configs.focus_th) return '#333'
                else return 'none'
              })
        }
        data_cells_tmp.selectAll('rect')
          .attr('stroke', '#bababa')
        data_cells_tmp.select(`#data-cell-${cur_sel_event_step}`)
          .attr('stroke', '#333')
        cur_sel_step.value = target_id
        cur_events = []
        for (let i = 0; i < events_scope; ++i) {
          cur_events.push([])
          for (let j = 0; j < phase_events.length; ++j) {
            if (phase_events[j][1] == i+cur_event_start) {
              cur_events[i].push(phase_events[j])
            }
            // if (phase_events[j][1] > i+cur_event_start) break
          }
        }
        events_headers_g.selectAll('*').remove()
        for (let i = 0; i < cur_events.length; ++i) {
          let event_headers_g = events_headers_g.append('g')
            .attr('transform', () => `translate(${i*(cell_w+5)}, 0)`)
            .attr('event-step', () => i)
          for (let j = 0; j < cur_events[i].length; ++j) {
            let event_header_g = event_headers_g.append('g')
                .attr('class', 'event-header')
                .attr('event-step', () => i)
                .style('cursor', 'pointer')
                .attr('transform', () => `translate(0, ${-(j+1)*cell_header_h})`)
                .on("mouseover", (e,d) => {
                  let pre_entities = [], cur_entities = []
                  if (cur_events[i][j][2] && Array.isArray(cur_events[i][j][2])) {
                    for (let entity_id of cur_events[i][j][2]) {
                      pre_entities.push(focus_entities[cur_events[i][j][1]-1][entity_id])
                    }
                  } else if (Number.isInteger(cur_events[i][j][2])) {
                    pre_entities = [focus_entities[cur_events[i][j][1]-1][cur_events[i][j][2]]]
                  }
                  if (cur_events[i][j][3] && Array.isArray(cur_events[i][j][3])) {
                    for (let entity_id of cur_events[i][j][3]) {
                      cur_entities.push(focus_entities[cur_events[i][j][1]][entity_id])
                    }
                  } else if (Number.isInteger(cur_events[i][j][3])) {
                    cur_entities = [focus_entities[cur_events[i][j][1]][cur_events[i][j][3]]]
                  }

                  let prev_loc_ids = [], cur_loc_ids = []
                  for (let entity of pre_entities) {
                    prev_loc_ids = prev_loc_ids.concat(entity.loc_ids)  
                  }
                  for (let entity of cur_entities) {
                    cur_loc_ids = cur_loc_ids.concat(entity.loc_ids)  
                  }
                  event_cur_loc_ids.value = cur_loc_ids
                  event_prev_loc_ids.value = prev_loc_ids
                  cur_hover_event.value = [i, j]
                })
            event_header_g.append('rect')
              .attr('x', -0.5).attr('y', 0)
              .attr('width', cell_w+1).attr('height', cell_header_h)
              .attr('fill', () => {
                if ((cur_hover_event.value[0] == i) && (cur_hover_event.value[1] == j)) return '#0097A7'
                if (cur_events[i][j][1]-cur_event_start == cur_sel_event_step) return '#333'
                else return '#cecece'
              })
            event_header_g.append('text')
              .attr('x', cell_w/2).attr('y', 10)
              .attr('text-anchor', 'middle')
              .style('font-size', '9px')
              .attr('fill', () => {
                if ((cur_hover_event.value[0] == i) && (cur_hover_event.value[1] == j)) return '#fff'
                if (cur_events[i][j][1]-cur_event_start == cur_sel_event_step) return '#fff'
                else return '#333'
              })
              .text(() => cur_events[i][j][4].type)
          }
        }

      })
  
  time_axis_bar.append('text')
    .attr('x', -4)
    .attr('y', 41)
    .attr('text-anchor', 'end')
    .style('font-size', '11px')
    .attr('fill', '#333')
    .text(inforStore.cur_timeline_indicator)
  time_axis_bar.append('g').selectAll('rect')
    .data(temporal_ids)
    .join('rect')
      .attr('id', d => `resi_unit-${d}`)
      .attr('x', d => d * step_width)
      .attr('y', time_bar_h*3 + 2)
      .attr('width', step_width)
      .attr('height', time_bar_h)
      .attr('fill', d => {
        if (inforStore.cur_timeline_indicator == 'Residual_abs') {
          let val = inforStore.sel_phase_details[inforStore.cur_focused_model].temporal_residuals_abs[d]
          if (val >= inforStore.dataset_configs.focus_th) return inforStore.extreme_err_color_scale(d)
          else return inforStore.mild_err_color_scale(d)
        }
        else if (inforStore.cur_timeline_indicator == 'Multi_accuracy') {
          let val = inforStore.sel_phase_details[inforStore.cur_focused_model].time_multi_accuracy[d]
          if (val >= inforStore.dataset_configs.focus_th) return inforStore.extreme_err_color_scale(d)
          else return inforStore.mild_err_color_scale(d)
        }
        else if (inforStore.cur_timeline_indicator == 'POD') {
          let val = inforStore.sel_phase_details[inforStore.cur_focused_model].time_pod[d]
          if (val >= inforStore.dataset_configs.focus_th) return inforStore.extreme_err_color_scale(d)
          else return inforStore.mild_err_color_scale(d)
        }
        else if (inforStore.cur_timeline_indicator == 'FAR') {
          let val = inforStore.sel_phase_details[inforStore.cur_focused_model].time_far[d]
          if (val >= inforStore.dataset_configs.focus_th) return inforStore.extreme_err_color_scale(d)
          else return inforStore.mild_err_color_scale(d)
        }
      })
      .attr('stroke', '#999')
      .on('click', (e) => {
        let target_id = d3.select(e.target).attr('id').split('-')[1]
        cur_event_start = Math.max(0, Math.min(temporal_ids.length-events_scope, target_id));
        cur_event_end = cur_event_start + events_scope
        cur_sel_event_step = target_id - cur_event_start
        d3.select('#cur_step_mark').attr('x', cur_sel_event_step*step_width)
        phase_raw_data = inforStore.cur_phase_data.phase_raw_data.slice(cur_event_start, cur_event_end)
        phase_raw_level = inforStore.phase_details_infor.phase_raw_level.slice(cur_event_start, cur_event_end)
        phase_raw_infor = []
        for (let i = 0; i < phase_raw_data.length; i++) {
          phase_raw_infor.push([])
          for (let j = 0; j < phase_raw_data[i].length; ++j) {
            phase_raw_infor[i].push({
              'val': phase_raw_data[i][j][0],
              'level': phase_raw_level[i][j]
            })
          }
        }
        time_bar_g.select('#start-slider')
          .attr('transform', `translate(${cur_event_start * step_width}, ${time_bar_h*4+3})`)
        let data_cells_tmp = events_bar_g.selectAll('g')
          .data(phase_raw_infor)
          .join('g')
        for (let i = 0; i < phase_raw_infor.length; ++i) {
          d3.select(data_cells_tmp._groups[0][i]).selectAll('circle')
            .data(phase_raw_infor[i])
            .join('circle')
              .attr('cx', (d,j) => cell_x_scale(loc_coords_x[j]))
              .attr('cy', (d,j) => cell_y_scale(loc_coords_y[j]))
              .attr('loc_id', (d,j) => j)
              .attr('r', 2)
              .attr('fill', (d,j) => cellValColor(d.level))
              .attr('stroke', (d,j) => {
                if ((i == cur_hover_event.value[0]) && event_cur_loc_ids.value.includes(j)) return '#0097A7'
                if ((i > 0) && (cur_hover_event.value[0] == i+1) && event_prev_loc_ids.value.includes(j)) return '#0097A7'
                if (d.val >= inforStore.dataset_configs.focus_th) return '#333'
                else return 'none'
              })
        }
        data_cells_tmp.selectAll('rect')
          .attr('stroke', '#bababa')
        data_cells_tmp.select(`#data-cell-${cur_sel_event_step}`)
          .attr('stroke', '#333')
        cur_sel_step.value = target_id

        cur_events = []
        for (let i = 0; i < events_scope; ++i) {
          cur_events.push([])
          for (let j = 0; j < phase_events.length; ++j) {
            if (phase_events[j][1] == i+cur_event_start) {
              cur_events[i].push(phase_events[j])
            }
            // if (phase_events[j][1] > i+cur_event_start) break
          }
        }
        events_headers_g.selectAll('*').remove()
        for (let i = 0; i < cur_events.length; ++i) {
          let event_headers_g = events_headers_g.append('g')
            .attr('transform', () => `translate(${i*(cell_w+5)}, 0)`)
            .attr('event-step', () => i)
          for (let j = 0; j < cur_events[i].length; ++j) {
            let event_header_g = event_headers_g.append('g')
                .attr('class', 'event-header')
                .attr('event-step', () => i)
                .style('cursor', 'pointer')
                .attr('transform', () => `translate(0, ${-(j+1)*cell_header_h})`)
                .on("mouseover", (e,d) => {
                  let pre_entities = [], cur_entities = []
                  if (cur_events[i][j][2] && Array.isArray(cur_events[i][j][2])) {
                    for (let entity_id of cur_events[i][j][2]) {
                      pre_entities.push(focus_entities[cur_events[i][j][1]-1][entity_id])
                    }
                  } else if (Number.isInteger(cur_events[i][j][2])) {
                    pre_entities = [focus_entities[cur_events[i][j][1]-1][cur_events[i][j][2]]]
                  }
                  if (cur_events[i][j][3] && Array.isArray(cur_events[i][j][3])) {
                    for (let entity_id of cur_events[i][j][3]) {
                      cur_entities.push(focus_entities[cur_events[i][j][1]][entity_id])
                    }
                  } else if (Number.isInteger(cur_events[i][j][3])) {
                    cur_entities = [focus_entities[cur_events[i][j][1]][cur_events[i][j][3]]]
                  }

                  let prev_loc_ids = [], cur_loc_ids = []
                  for (let entity of pre_entities) {
                    prev_loc_ids = prev_loc_ids.concat(entity.loc_ids)  
                  }
                  for (let entity of cur_entities) {
                    cur_loc_ids = cur_loc_ids.concat(entity.loc_ids)  
                  }
                  event_cur_loc_ids.value = cur_loc_ids
                  event_prev_loc_ids.value = prev_loc_ids
                  cur_hover_event.value = [i, j]
                })
            event_header_g.append('rect')
              .attr('x', -0.5).attr('y', 0)
              .attr('width', cell_w+1).attr('height', cell_header_h)
              .attr('fill', () => {
                if ((cur_hover_event.value[0] == i) && (cur_hover_event.value[1] == j)) return '#0097A7'
                if (cur_events[i][j][1]-cur_event_start == cur_sel_event_step) return '#333'
                else return '#cecece'
              })
            event_header_g.append('text')
              .attr('x', cell_w/2).attr('y', 10)
              .attr('text-anchor', 'middle')
              .style('font-size', '9px')
              .attr('fill', () => {
                if ((cur_hover_event.value[0] == i) && (cur_hover_event.value[1] == j)) return '#fff'
                if (cur_events[i][j][1]-cur_event_start == cur_sel_event_step) return '#fff'
                else return '#333'
              })
              .text(() => cur_events[i][j][4].type)
          }
        }

      })
  
  let slider_h = 8
  let slider = time_bar_g.append('g')
    .attr('id', 'start-slider')
    .attr('class', 'slider')
    .attr('transform', `translate(${cur_event_start*step_width}, ${time_bar_h*4+3})`)
    .attr('cursor', 'ew-resize')
  slider.append('rect')
    .attr('x', 0)
    .attr('y', 0)
    .attr('width', step_width*events_scope)
    .attr('height', slider_h)
    .attr('fill', '#333')
    .attr('stroke', 'none')
    .attr('opactiy', 0.2)
  slider.append('rect')
    .attr('id', 'cur_step_mark')
    .attr('x', cur_sel_event_step*step_width)
    .attr('y', 0)
    .attr('width', step_width)
    .attr('height', slider_h)
    .attr('fill', '#cecece')
    .attr('stroke', '#fff')
  // 定义拖拽
  let acc_dx = 0
  let data_cells
  const slider_drag = d3.drag().on("start", function(event) {}).on("drag", function(event) {
    // const x = Math.max(0, Math.min(temporal_ids.length-events_scope, event.x));
    // cur_event_start = Math.round(x / step_width) * step_width
    // cur_event_end = cur_event_start + events_scope
    // time_axis_bar.selectAll('.slider')
    //   .attr('x', cur_event_start*step_width)
    // data_cells.attr('transform', (d,i) => `translate(${i*(cell_w+5)}, 0)`)
    let curSliderX = cur_event_start * step_width
    acc_dx += event.dx
    if (Math.abs(acc_dx) > step_width) {
      let newSliderStep = Math.round((parseInt(curSliderX) + acc_dx) / step_width)
      cur_event_start = Math.max(0, Math.min(temporal_ids.length-events_scope, newSliderStep));
      cur_event_end = cur_event_start + events_scope
      // console.log(curSliderX, acc_dx, (parseInt(curSliderX) + acc_dx), Math.max(0, Math.min(temporal_ids.length-events_scope, newSliderStep)), temporal_ids.length-events_scope,cur_event_start)
      phase_raw_data = inforStore.cur_phase_data.phase_raw_data.slice(cur_event_start, cur_event_end)
      phase_raw_level = inforStore.phase_details_infor.phase_raw_level.slice(cur_event_start, cur_event_end)
      phase_raw_infor = []
      for (let i = 0; i < phase_raw_data.length; i++) {
        phase_raw_infor.push([])
        for (let j = 0; j < phase_raw_data[i].length; ++j) {
          phase_raw_infor[i].push({
            'val': phase_raw_data[i][j][0],
            'level': phase_raw_level[i][j]
          })
        }
      }
      time_bar_g.select('#start-slider')
        .attr('transform', `translate(${cur_event_start * step_width}, ${time_bar_h*4+3})`)
        // .attr('x', cur_event_start * step_width)
      // data_cells.selectAll('circle').remove()
      data_cells = events_bar_g.selectAll('g')
        .data(phase_raw_infor)
        .join('g')
      for (let i = 0; i < phase_raw_infor.length; ++i) {
        d3.select(data_cells._groups[0][i]).selectAll('circle')
          .data(phase_raw_infor[i])
          .join('circle')
            .attr('cx', (d,j) => cell_x_scale(loc_coords_x[j]))
            .attr('cy', (d,j) => cell_y_scale(loc_coords_y[j]))
            .attr('loc_id', (d,j) => j)
            .attr('r', 2)
            .attr('fill', (d,j) => cellValColor(d.level))
            .attr('stroke', (d,j) => {
              if ((i == cur_hover_event.value[0]) && event_cur_loc_ids.value.includes(j)) return '#0097A7'
              if ((i > 0) && (cur_hover_event.value[0] == i+1) && event_prev_loc_ids.value.includes(j)) return '#0097A7'
              if (d.val >= inforStore.dataset_configs.focus_th) return '#333'
              else return 'none'
            })
      }
      acc_dx = 0

      cur_events = []
        for (let i = 0; i < events_scope; ++i) {
          cur_events.push([])
          for (let j = 0; j < phase_events.length; ++j) {
            if (phase_events[j][1] == i+cur_event_start) {
              cur_events[i].push(phase_events[j])
            }
            // if (phase_events[j][1] > i+cur_event_start) break
          }
        }
        events_headers_g.selectAll('*').remove()
        for (let i = 0; i < cur_events.length; ++i) {
          let event_headers_g = events_headers_g.append('g')
            .attr('transform', () => `translate(${i*(cell_w+5)}, 0)`)
            .attr('event-step', () => i)
          for (let j = 0; j < cur_events[i].length; ++j) {
            let event_header_g = event_headers_g.append('g')
                .attr('class', 'event-header')
                .attr('event-step', () => i)
                .style('cursor', 'pointer')
                .attr('transform', () => `translate(0, ${-(j+1)*cell_header_h})`)
                .on("mouseover", (e,d) => {
                  let pre_entities = [], cur_entities = []
                  if (cur_events[i][j][2] && Array.isArray(cur_events[i][j][2])) {
                    for (let entity_id of cur_events[i][j][2]) {
                      pre_entities.push(focus_entities[cur_events[i][j][1]-1][entity_id])
                    }
                  } else if (Number.isInteger(cur_events[i][j][2])) {
                    pre_entities = [focus_entities[cur_events[i][j][1]-1][cur_events[i][j][2]]]
                  }
                  if (cur_events[i][j][3] && Array.isArray(cur_events[i][j][3])) {
                    for (let entity_id of cur_events[i][j][3]) {
                      cur_entities.push(focus_entities[cur_events[i][j][1]][entity_id])
                    }
                  } else if (Number.isInteger(cur_events[i][j][3])) {
                    cur_entities = [focus_entities[cur_events[i][j][1]][cur_events[i][j][3]]]
                    console.log('cur_entities', cur_entities);
                  } else {
                    console.log('cur_events[i][j][3]', cur_events[i][j][3]);
                  }

                  let prev_loc_ids = [], cur_loc_ids = []
                  for (let entity of pre_entities) {
                    prev_loc_ids = prev_loc_ids.concat(entity.loc_ids)  
                  }
                  for (let entity of cur_entities) {
                    cur_loc_ids = cur_loc_ids.concat(entity.loc_ids)  
                  }
                  event_cur_loc_ids.value = cur_loc_ids
                  event_prev_loc_ids.value = prev_loc_ids
                  cur_hover_event.value = [i, j]
                })
            event_header_g.append('rect')
              .attr('x', -0.5).attr('y', 0)
              .attr('width', cell_w+1).attr('height', cell_header_h)
              .attr('fill', () => {
                if ((cur_hover_event.value[0] == i) && (cur_hover_event.value[1] == j)) return '#0097A7'
                if (cur_events[i][j][1]-parseInt(cur_event_start) == parseInt(cur_sel_event_step)) return '#333'
                else return '#cecece'
              })
            event_header_g.append('text')
              .attr('x', cell_w/2).attr('y', 10)
              .attr('text-anchor', 'middle')
              .style('font-size', '9px')
              .attr('fill', () => {
                if ((cur_hover_event.value[0] == i) && (cur_hover_event.value[1] == j)) return '#fff'
                if (cur_events[i][j][1]-parseInt(cur_event_start) == parseInt(cur_sel_event_step)) return '#fff'
                else return '#333'
              })
              .text(() => cur_events[i][j][4].type)
          }
        }

        cur_sel_step.value = parseInt(cur_event_start) + parseInt(cur_sel_event_step)
    }
  }).on("end", function(event) {
    data_cells.selectAll('rect')
      .attr('stroke', '#bababa')
    data_cells.select(`#data-cell-${cur_sel_event_step}`)
      .attr('stroke', '#333')
    let event_header_g = d3.selectAll('.event-header')
      for (let i = 0; i < event_header_g._groups[0].length; ++i) {
        let cur_event_header = d3.select(event_header_g._groups[0][i])
        cur_event_header.select('rect')
          .attr('fill', () => {
            if (cur_event_header.attr('event-step') == cur_sel_event_step) return '#333'
            else return '#cecece'
          })
        cur_event_header.select('text')
          .attr('fill', () => {
            if (cur_event_header.attr('event-step') == cur_sel_event_step) return '#fff'
            else return '#333'
          })
      }
    // cur_sel_event_step = 0
  })

  time_bar_g.select('#start-slider').call(slider_drag)
  drawTimeBarLegends()
}

function onTimelineIndicatorSel(item) {
  inforStore.cur_timeline_indicator = item
  // console.log(inforStore.cur_focus_indicator);
  // drawTimeEventBar()
}
function onOuterIndicatorSel(item) {
  cur_outer_indicator.value = item
}
function onInnerIndicatorSel(item) {
  cur_inner_indicator.value = item
}

function onLinkBtnClick() {
  view_linked.value = !view_linked.value
}

function onLeftFormTypeSel(item) {
  left_form_type.value = item
}
function onRightFormTypeSel(item) {
  right_form_type.value = item
}

function onLeftFormModelSel(item) {
  left_form_model.value = item
}
function onRightFormModelSel(item) {
  right_form_model.value = item
}

function onLeftStepTypeSel(item) {
  sel_step_left.value = item
}
function onRightStepTypeSel(item) {
  sel_step_right.value = item
}

// function onFormBtnHover(label) {
//   console.log('hover!!');
//   let region_id = `#${label}-form-collapse`
//   $(region_id).addClass('show')
// }
// function onFormBtnout(label) {
//   let region_id = `#${label}-form-collapse`
//   $(region_id).removeClass('show')
// } 
function num_fix_2(val) {
  return parseFloat(val.toFixed(2));
}

function drawTimeBarLegends() {
  d3.select('#time-bar-legends').selectAll('*').remove()
  let svg = d3.select('#time-bar-legends')
    .attr('width', 150)
    .attr('height', 70)
  let resi_legend_len = 100
  let pollutionValScale = d3.scaleLinear()
      .domain([0, resi_legend_len])
      .range(['#fff', '#4a1486'])
  let pollution_val_legend = svg.append('g')
    .attr('transform', `translate(8, 20)`)
  pollution_val_legend.selectAll('rect')
    .data(Array(resi_legend_len).fill(1))
    .join('rect')
      .attr('x', (d,i) => 2+i)
      .attr('y', 16)
      .attr('width', 1)
      .attr('height', 12)
      .attr('fill', (d,i) => pollutionValScale(i))
  pollution_val_legend.append('text')
    .attr('x', (resi_legend_len+4)/2)
    .attr('y', 11)
    .attr('text-anchor', 'middle')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('Focus Value')
  pollution_val_legend.append('text')
    .attr('x', 0)
    .attr('y', 27)
    .attr('text-anchor', 'end')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('0')
  pollution_val_legend.append('text')
    .attr('x', resi_legend_len+4)
    .attr('y', 27)
    .attr('text-anchor', 'start')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('500')
  
  let indicatorScale = d3.scaleLinear()
      .domain([0, resi_legend_len])
      //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
      //原来：
      // .range(valColorScheme_fire)
      .range(['#4292c6',"#fd8d3c"])



  // let indicator_legend = svg.append('g')
  //   .attr('transform', `translate(8, 36)`)
  // indicator_legend.selectAll('rect')
  //   .data(Array(resi_legend_len).fill(1))
  //   .join('rect')
  //     .attr('x', (d,i) => i+2)
  //     .attr('y', 16)
  //     .attr('width', 1)
  //     .attr('height', 12)
  //     .attr('fill', (d,i) => indicatorScale(i))
  // indicator_legend.append('text')
  //   .attr('x', (resi_legend_len+4)/2)
  //   .attr('y', 11)
  //   .attr('text-anchor', 'middle')
  //   .style('font-size', '12px')
  //   .attr('fill', '#333')
  //   .text(`${inforStore.cur_timeline_indicator}`)
  // indicator_legend.append('text')
  //   .attr('x', 0)
  //   .attr('y', 27)
  //   .attr('text-anchor', 'end')
  //   .style('font-size', '12px')
  //   .attr('fill', '#333')
  //   .text('0')
  // indicator_legend.append('text')
  //   .attr('x', resi_legend_len+4)
  //   .attr('y', 27)
  //   .attr('text-anchor', 'start')
  //   .style('font-size', '12px')
  //   .attr('fill', '#333')
  //   .text(() => {
  //     if (inforStore.cur_timeline_indicator == 'Residual_abs')
  //       return `${parseFloat(inforStore.sel_phase_details[inforStore.cur_focused_model].temporal_residuals_abs_max.toFixed(2))}`
  //     else if (inforStore.cur_timeline_indicator == 'Multi_accuracy')
  //       return 1
  //     else if (inforStore.cur_timeline_indicator == 'POD')
  //       return 1
  //     else if (inforStore.cur_timeline_indicator == 'FAR')
  //       return 1
  //   })
}

watch (() => cur_focus_fore_step.value, (oldValue, newValue) => {
  drawLocResidualInfor(cur_hover_loc.value, cur_sel_step.value)
  drawLocTimeContext(cur_hover_loc.value, cur_sel_step.value)
})


let cur_hover_entity = ref(-1)
watch (() => cur_left_hover_entity.value, (oldValue, newValue) => {
  cur_hover_entity.value = cur_left_hover_entity.value
  console.log('cur_left_hover_entity.value', cur_left_hover_entity.value);
  let left_mark_entity = d3.select('#left-mark-entity')
  let right_mark_entity = d3.select('#right-mark-entity')
  if (cur_left_hover_entity.value == -1) {
    left_mark_entity.selectAll('*').remove()
    right_mark_entity.selectAll('*').remove()
  } else {
    left_mark_entity.selectAll('*').remove()
    right_mark_entity.selectAll('*').remove()
    let left_entity = inforStore.st_phase_events.phases[left_form_model.value][inforStore.cur_phase_sorted_id].focus_entities[cur_step_left.value][cur_left_hover_entity.value]
    console.log(left_entity);
    for (let i = 0; i < left_entity.loc_ids.length; ++i) {
      left_mark_entity.append('path')
        .datum(grid_points[left_entity.loc_ids[i]])
        .attr("d", line) // 应用生成器
        .attr("fill", 'none')
        .attr("stroke", valColorScheme_red[1])
        .attr("stroke-width", 2)
        .style("stroke-linejoin", "round")
        .style("stroke-linecap", "round")
    }
    if (sel_step_left.value == sel_step_right.value) {
      for (let i = 0; i < left_entity.loc_ids.length; ++i) {
        right_mark_entity.append('path')
          .datum(grid_points[left_entity.loc_ids[i]])
          .attr("d", line) // 应用生成器
          .attr("fill", 'none')
          .attr("stroke", valColorScheme_red[1])
          .attr("stroke-width", 2)
          .style("stroke-linejoin", "round")
          .style("stroke-linecap", "round")
      }
    }
  }
})

watch (() => cur_right_hover_entity.value, (oldValue, newValue) => {
  cur_hover_entity.value = cur_right_hover_entity.value
  console.log('cur_right_hover_entity.value', cur_right_hover_entity.value);
  let left_mark_entity = d3.select('#left-mark-entity')
  let right_mark_entity = d3.select('#right-mark-entity')
  if (cur_right_hover_entity.value == -1) {
    left_mark_entity.selectAll('*').remove()
    right_mark_entity.selectAll('*').remove()
  } else {
    left_mark_entity.selectAll('*').remove()
    right_mark_entity.selectAll('*').remove()
    // let left_entity = inforStore.st_phase_events.phases[left_form_model.value][inforStore.cur_phase_sorted_id].focus_entities[cur_step_left.value][cur_left_hover_entity.value]
    let right_entity = inforStore.st_phase_events.phases[right_form_model.value][inforStore.cur_phase_sorted_id].focus_entities[cur_step_right.value][cur_right_hover_entity.value]
    for (let i = 0; i < right_entity.loc_ids.length; ++i) {
      right_mark_entity.append('path')
        .datum(grid_points[right_entity.loc_ids[i]])
        .attr("d", line) // 应用生成器
        .attr("fill", 'none')
        .attr("stroke", valColorScheme_red[1])
        .attr("stroke-width", 2)
        .style("stroke-linejoin", "round")
        .style("stroke-linecap", "round")
    }
    if (sel_step_left.value == sel_step_right.value) {
      for (let i = 0; i < right_entity.loc_ids.length; ++i) {
        left_mark_entity.append('path')
          .datum(grid_points[right_entity.loc_ids[i]])
          .attr("d", line) // 应用生成器
          .attr("fill", 'none')
          .attr("stroke", valColorScheme_red[1])
          .attr("stroke-width", 2)
          .style("stroke-linejoin", "round")
          .style("stroke-linecap", "round")
      }
    }
  }
})

watch (() => cur_hover_entity.value, (oldValue, newValue) => {
  if (!document.getElementById('entity-area-error')) {
    setTimeout(() => {
      drawEntityAreaError(cur_hover_entity.value, cur_sel_step.value)
      drawEntityMoveError(cur_hover_entity.value, cur_sel_step.value)
      drawEntityIntensityError(cur_hover_entity.value, cur_sel_step.value)
    }, 200)
  } else {
    drawEntityAreaError(cur_hover_entity.value, cur_sel_step.value)
    drawEntityMoveError(cur_hover_entity.value, cur_sel_step.value)
    drawEntityIntensityError(cur_hover_entity.value, cur_sel_step.value)
  }
})

watch (() => cur_hover_loc.value, (oldValue, newValue) => {
  if (cur_hover_loc.value == -1) {
    d3.select('#left-mark-grid').datum([])
    d3.select('#left-mark-grid').attr('d', '')
    d3.select('#right-mark-grid').datum([])
    d3.select('#right-mark-grid').attr('d', '')
    d3.select('#loc-residual-infor').selectAll('*').remove()
    d3.select('#loc-time-context').selectAll('*').remove()
  } else {
    d3.select('#right-mark-grid')
      .datum(grid_points[cur_hover_loc.value])
      .attr("d", line) // 应用生成器
      .attr("fill", 'none')
      .attr("stroke", valColorScheme_red[1])
      .attr("stroke-width", 2)
      .style("stroke-linejoin", "round")
      .style("stroke-linecap", "round")
    d3.select('#left-mark-grid')
      .datum(grid_points[cur_hover_loc.value])
      .attr("d", line) // 应用生成器
      .attr("fill", 'none')
      .attr("stroke", valColorScheme_red[1])
      .attr("stroke-width", 2)
      .style("stroke-linejoin", "round")
      .style("stroke-linecap", "round")
    // drawLocResidualInfor(cur_hover_loc_right.value, cur_sel_step.value)
    // drawLocTimeContext(cur_hover_loc_right.value, cur_sel_step.value)
    // intervalId_phase = setInterval(askForDrawPhase, 100);
    if (!document.getElementById('loc-residual-infor')) {
      setTimeout(() => {
        drawLocResidualInfor(cur_hover_loc.value, cur_sel_step.value)
      }, 200)
    } else {
      drawLocResidualInfor(cur_hover_loc.value, cur_sel_step.value)
    }
    
    getData(inforStore, 'loc_instance_infor', JSON.stringify(inforStore.cur_sel_models), inforStore.cur_phase_sorted_id, cur_sel_step.value, cur_hover_loc.value)
  }
  
})

watch (() => inforStore.loc_instance_infor, (oldValue, newValue) => {
  drawLocTimeContext(cur_hover_loc.value, cur_sel_step.value)
  for (let i = 1; i < inforStore.dataset_configs.features.length; ++i) {
    drawLocFeatureInfor(cur_hover_loc.value, cur_sel_step.value, i)
  }
  
})

function drawLocResidualInfor(loc_id, step) {
  let margin_left = 28, margin_right = 16, margin_top = 17, margin_bottom = 18
  let main_w = 240, main_h = 94
  let svg_w = main_w + margin_left + margin_right
  let svg_h = main_h + margin_top + margin_bottom

  let gt_val = inforStore.phase_details_infor.phase_raw_val[step][loc_id]
  let residuals = inforStore.sel_phase_details[inforStore.cur_focused_model].st_residuals[step][loc_id]
  let pred_vals = inforStore.sel_phase_details[inforStore.cur_focused_model].phase_pred_val[step][loc_id]
  let pred_vals_comp

  let val_limit = inforStore.dataset_configs.focus_th
  if (d3.max(pred_vals) > val_limit) val_limit = d3.max(pred_vals)
  if (gt_val > val_limit) val_limit = gt_val
  if (inforStore.cur_baseline_model.length > 0) {
    pred_vals_comp = inforStore.sel_phase_details[inforStore.cur_baseline_model].phase_pred_val[step][loc_id]
    if (d3.max(pred_vals_comp) > val_limit) val_limit = d3.max(pred_vals_comp)
  }

  let y_tick_num = 1
  for (let i = 0; i < inforStore.dataset_configs.focus_levels.length; ++i) {
    if (val_limit < inforStore.dataset_configs.focus_levels[i]) {
      y_tick_num = i
      break
    }
  }

  d3.select('#loc-residual-infor').selectAll('*').remove()
  let svg = d3.select('#loc-residual-infor')
    .attr('width', svg_w)
    .attr('height', svg_h)
  
  let main_g = svg.append('g')
    .attr('transform', `translate(${margin_left}, ${margin_top})`)
  let xScale = d3.scaleLinear()
    .domain([0, inforStore.dataset_configs.output_window])
    .range([0, main_w])
  let yScale = d3.scaleLinear()
    .domain([0, val_limit+10])
    .range([main_h, 0])
  let resScale = d3.scaleSequential(d3.interpolateRdBu)
    .domain(global_residuals_range)
  let xAxis = d3.axisBottom(xScale)
    .ticks(inforStore.dataset_configs.output_window.length)
    .tickFormat(d => `${d}`)
  let yAxis = d3.axisLeft(yScale)
    .tickValues(inforStore.dataset_configs.focus_levels)
    .ticks(y_tick_num)
    .tickFormat((d,i) => `${inforStore.dataset_configs.focus_levels[i]}`)
  let xAxis_g = svg.append("g")
    .attr("transform", `translate(${margin_left}, ${margin_top+main_h})`) // 将X轴移至底部
    .call(xAxis);
  let yAxis_g = svg.append("g")
    .attr("transform", `translate(${margin_left}, ${margin_top})`) // 将X轴移至底部
    .call(yAxis);
  
  let bin_w = xScale(1)-xScale(0)
  if (inforStore.cur_baseline_model.length > 0) {
    bin_w /= 2
  }
  let hist = main_g.append('g')
  hist.selectAll('rect')
    .data(pred_vals)
    .join('rect')
      .attr('fore_step', (d,i) => i)
      .attr('x', (d,i) => xScale(i) + 0.1 * bin_w)
      .attr('y', (d,i) => yScale(d))
      .attr('width', (d,i) => bin_w * 0.8)
      .attr('height', (d,i) => (main_h-yScale(d)))
      .attr('fill', (d,i) => '#999')
      .attr('stroke', (d,i) => {
        if (cur_focus_fore_step.value == i) return '#0097A7'
        else return 'none'
      })
      // .attr('fill', (d,i) => resScale(residuals[i]))
      .on('click', (e,d) => {
        let cur_fore_step = pred_vals.indexOf(d)
        if (cur_focus_fore_step.value == cur_fore_step) cur_focus_fore_step.value = -1
        else cur_focus_fore_step.value = cur_fore_step
      })

  if (inforStore.cur_baseline_model.length > 0) {
    let pattern_defs = svg.append("defs")
    let err_pattern = pattern_defs.append("pattern")
      .attr("id", "err_pattern")
      .attr("patternUnits", "userSpaceOnUse")
      .attr("width", 5)
      .attr("height", 5)
    err_pattern.append("line")
      .attr("x1", 0)
      .attr("y1", 0)
      .attr("x2", 5)
      .attr("y2", 5)
      .attr("stroke", "#999")
      .attr("stroke-width", 1)

    let hist_comp = main_g.append('g')
      .attr('transform', `translate(${bin_w}, 0)`)
    hist_comp.selectAll('rect')
      .data(pred_vals_comp)
      .join('rect')
        .attr('fore_step', (d,i) => i)
        .attr('x', (d,i) => xScale(i) + 0.1 * bin_w)
        .attr('y', (d,i) => yScale(d))
        .attr('width', (d,i) => bin_w * 0.8)
        .attr('height', (d,i) => (main_h-yScale(d)))
        .attr('fill', (d,i) => 'url(#err_pattern)')
        .attr('stroke', (d,i) => {
          if (cur_focus_fore_step.value == i) return '#0097A7'
          else return 'none'
        })
        // .attr('fill', (d,i) => resScale(residuals[i]))
        .on('click', (e,d) => {
          let cur_fore_step = pred_vals_comp.indexOf(d)
          if (cur_focus_fore_step.value == cur_fore_step) cur_focus_fore_step.value = -1
          else cur_focus_fore_step.value = cur_fore_step
        })
  }
  
  let limit_lines = main_g.append('g')
  limit_lines.append('line')
    .attr('x1', 0).attr('x2', main_w)
    .attr('y1', yScale(inforStore.dataset_configs.focus_th)).attr('y2', yScale(inforStore.dataset_configs.focus_th))
    .attr('stroke', '#0097A7')
    .attr('stroke-dasharray', '5,5')
  limit_lines.append('line')
    .attr('x1', 0).attr('x2', main_w)
    .attr('y1', yScale(gt_val)).attr('y2', yScale(gt_val))
    .attr('stroke', '#0097A7')
  
  svg.append('text')
    .attr('x', margin_left)
    .attr('y', 12)
    .attr('text-anchor', 'middle')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('Focus_val')
}

function drawLocTimeContext(loc_id, step) {
  let margin_left = 28, margin_right = 16, margin_top = 17, margin_bottom = 18
  let main_w = 240, main_h = 94
  let svg_w = main_w + margin_left + margin_right
  let svg_h = main_h + margin_top + margin_bottom

  console.log(inforStore.loc_instance_infor.prev_truth);
  let gt_vals_context = inforStore.loc_instance_infor.prev_truth.map(item => item[0])
  let real_step_num = inforStore.dataset_configs.output_window

  let val_limit = inforStore.dataset_configs.focus_th
  if (d3.max(gt_vals_context) > val_limit) val_limit = d3.max(gt_vals_context)
  let pred_vals, pred_vals_comp
  if (cur_focus_fore_step.value != -1) {
    pred_vals = inforStore.loc_instance_infor.preds_with_prev[inforStore.cur_focused_model][cur_focus_fore_step.value]
    if (d3.max(pred_vals) > val_limit) val_limit = d3.max(pred_vals)
    if (inforStore.cur_baseline_model.length > 0) {
      pred_vals_comp = inforStore.loc_instance_infor.preds_with_prev[inforStore.cur_baseline_model][cur_focus_fore_step.value]
      if (d3.max(pred_vals_comp) > val_limit) val_limit = d3.max(pred_vals_comp)
    }
  }
  

  let y_tick_num = 1
  for (let i = 0; i < inforStore.dataset_configs.focus_levels.length; ++i) {
    if (val_limit < inforStore.dataset_configs.focus_levels[i]) {
      y_tick_num = i
      break
    }
  }

  d3.select('#loc-time-context').selectAll('*').remove()
  let svg = d3.select('#loc-time-context')
    .attr('width', svg_w)
    .attr('height', svg_h)
  
  let main_g = svg.append('g')
    .attr('transform', `translate(${margin_left}, ${margin_top})`)
  let xScale = d3.scaleLinear()
    .domain([0, real_step_num])
    .range([0, main_w])
  let yScale = d3.scaleLinear()
    .domain([0, val_limit+10])
    .range([main_h, 0])
  let resScale = d3.scaleSequential(d3.interpolateRdBu)
    .domain(global_residuals_range)
  let xAxis = d3.axisBottom(xScale)
    .ticks(real_step_num)
    .tickFormat(d => {
      if (d == real_step_num) return 'pred'
      else return `p${real_step_num-d}`
    })
  let yAxis = d3.axisLeft(yScale)
    .tickValues(inforStore.dataset_configs.focus_levels)
    .ticks(y_tick_num)
    .tickFormat((d,i) => `${inforStore.dataset_configs.focus_levels[i]}`)
  let xAxis_g = svg.append("g")
    .attr("transform", `translate(${margin_left}, ${margin_top+main_h})`) // 将X轴移至底部
    .call(xAxis);
  let yAxis_g = svg.append("g")
    .attr("transform", `translate(${margin_left}, ${margin_top})`) // 将X轴移至底部
    .call(yAxis);
  
  let bin_w = xScale(1)-xScale(0)

  let limit_lines = main_g.append('g')
  limit_lines.append('line')
    .attr('x1', 0).attr('x2', main_w)
    .attr('y1', yScale(inforStore.dataset_configs.focus_th)).attr('y2', yScale(inforStore.dataset_configs.focus_th))
    .attr('stroke', '#999')
    .attr('stroke-dasharray', '5,5')
  
  const line = d3.line()
    .x((d,i) => xScale(i))
    .y((d,i) => yScale(d))

  let time_line = main_g.append('g')
  time_line.append('path')
      .datum(gt_vals_context)
      .attr('d', line)  
      .attr('fill', 'none')
      .attr('stroke', '#999')
  let time_context_g = main_g.append('g')
  time_context_g.selectAll('circle')
    .data(gt_vals_context)
    .join('circle')
      .attr('cx', (d,i) => xScale(i))
      .attr('cy', (d,i) => yScale(d))
      .attr('r', 2.5)
      .attr('fill', '#999')
  if (cur_focus_fore_step.value != -1) {
    // console.log('pred_vals', pred_vals);
    const pred_line = d3.line()
      .x((d,i) => xScale(i+inforStore.dataset_configs.output_window - cur_focus_fore_step.value))
      .y((d,i) => yScale(d))
    let focus_pred = main_g.append('g')
    focus_pred.append('path')
      .datum(pred_vals)
      .attr('d', pred_line)  
      .attr('fill', 'none')
      .attr('stroke', '#0097A7')
    focus_pred.selectAll('circle')
      .data(pred_vals)
      .join('circle')
        .attr('cx', (d,i) => xScale(i + inforStore.dataset_configs.output_window - cur_focus_fore_step.value))
        .attr('cy', (d,i) => yScale(d))
        .attr('r', 2.5)
        .attr('fill', '#0097A7')
    if (inforStore.cur_baseline_model.length > 0) {
      let focus_pred_comp = main_g.append('g')
      focus_pred_comp.append('path')
        .datum(pred_vals_comp)
        .attr('d', pred_line)  
        .attr('fill', 'none')
        .attr('stroke', '#0097A7')
        .attr('stroke-dasharray', '5,5')
      focus_pred_comp.selectAll('circle')
        .data(pred_vals_comp)
        .join('circle')
          .attr('cx', (d,i) => xScale(i + inforStore.dataset_configs.output_window - cur_focus_fore_step.value))
          .attr('cy', (d,i) => yScale(d))
          .attr('r', 2.5)
          .attr('fill', '#0097A7')
    }
  }
  svg.append('text')
    .attr('x', margin_left)
    .attr('y', 12)
    .attr('text-anchor', 'middle')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('Focus_val')
}

function drawLocFeatureInfor(loc_id, step, feat_id) {
  let margin_left = 36, margin_right = 16, margin_top = 17, margin_bottom = 18
  let main_w = 240, main_h = 94
  let svg_w = main_w + margin_left + margin_right
  let svg_h = main_h + margin_top + margin_bottom

  let gt_vals_context = inforStore.loc_instance_infor.prev_truth.map(item => item[feat_id])
  let real_step_num = inforStore.dataset_configs.output_window

  let svg_id = `loc-feature-${feat_id}`
  d3.select(`#${svg_id}`).selectAll('*').remove()
  let svg = d3.select(`#${svg_id}`)
    .attr('width', svg_w)
    .attr('height', svg_h)
  
  let main_g = svg.append('g')
    .attr('transform', `translate(${margin_left}, ${margin_top})`)
  let xScale = d3.scaleLinear()
    .domain([0, real_step_num])
    .range([0, main_w])
  
  let yScale = d3.scaleLinear()
    .domain([d3.min(gt_vals_context), d3.max(gt_vals_context)])
    .range([main_h, 0])
  let resScale = d3.scaleSequential(d3.interpolateRdBu)
    .domain(global_residuals_range)
  let xAxis = d3.axisBottom(xScale)
    .ticks(real_step_num)
    .tickFormat(d => {
      if (d == real_step_num) return 'pred'
      else return `p${real_step_num-d}`
    })
  let yAxis = d3.axisLeft(yScale)
    // .tickValues(inforStore.dataset_configs.focus_levels)
    .ticks(4)
    .tickFormat((d,i) => {
      if (d > 1) return d3.format("~s")(d)
      else return d
      })
  let xAxis_g = svg.append("g")
    .attr("transform", `translate(${margin_left}, ${margin_top+main_h})`) // 将X轴移至底部
    .call(xAxis);
  let yAxis_g = svg.append("g")
    .attr("transform", `translate(${margin_left}, ${margin_top})`) // 将X轴移至底部
    .call(yAxis);

  let bin_w = xScale(1)-xScale(0)

  // let limit_lines = main_g.append('g')
  // limit_lines.append('line')
  //   .attr('x1', 0).attr('x2', main_w)
  //   .attr('y1', yScale(inforStore.dataset_configs.focus_th)).attr('y2', yScale(inforStore.dataset_configs.focus_th))
  //   .attr('stroke', '#999')
  //   .attr('stroke-dasharray', '5,5')
  
  const line = d3.line()
    .x((d,i) => xScale(i))
    .y((d,i) => yScale(d))

  let time_line = main_g.append('g')
  time_line.append('path')
      .datum(gt_vals_context)
      .attr('d', line)  
      .attr('fill', 'none')
      .attr('stroke', '#999')
  let time_context_g = main_g.append('g')
  time_context_g.selectAll('circle')
    .data(gt_vals_context)
    .join('circle')
      .attr('cx', (d,i) => xScale(i))
      .attr('cy', (d,i) => yScale(d))
      .attr('r', 2.5)
      .attr('fill', '#999')
  svg.append('text')
    .attr('x', margin_left)
    .attr('y', 12)
    .attr('text-anchor', 'middle')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text(inforStore.dataset_configs.features[feat_id])
}

let recordLayerShow = ref(false)
function onRecordIconClick() {
  recordLayerShow.value = !recordLayerShow.value
}

watch (() => inforStore.sel_event_record, (oldValue, newValue) => {
  if (inforStore.sel_event_record == -1) return 
  let cur_record = inforStore.event_records[`${inforStore.cur_phase_sorted_id}`][inforStore.sel_event_record]
  cur_sel_step.value = cur_record.step
  cur_hover_loc.value = cur_record.hover_loc
  cur_hover_entity.value = cur_record.hover_entity
  cur_event_notes.value = cur_record.notes
  cur_sel_event_step = cur_record.event_step

  cur_phase_time_str_left.value = cur_record.left.timestamp
  cur_phase_time_str_right.value = cur_record.right.timestamp

  left_form_model.value = cur_record.left.model
  right_form_model.value = cur_record.right.model

  left_form_type.value = cur_record.left.type
  right_form_type.value = cur_record.right.type

  sel_step_left.value = cur_record.left.step
  sel_step_right.value = cur_record.right.step

  if (left_form_type.value == 'feature') {
    cur_inner_indicator.value = cur_record.left.features[0]
    cur_outer_indicator.value = cur_record.left.features[1]
  }
  if (right_form_type.value == 'feature') {
    cur_inner_indicator.value = cur_record.right.features[1]
    cur_outer_indicator.value = cur_record.right.features[1]
  }
  if (left_form_type.value == 'error') {
    inforStore.cur_timeline_indicator = cur_record.left.focus_metric
  }
  if (right_form_type.value == 'error') {
    inforStore.cur_timeline_indicator = cur_record.right.focus_metric
  }

  drawTimeEventBar()
  inforStore.sel_event_record = -1
})


let cur_event_notes = ref("")
function onSaveRecordClick() {
  let cur_record = {}
  cur_record.left = {}
  cur_record.right = {}

  cur_record.notes = cur_event_notes.value
  cur_record.step = Number(cur_sel_step.value)
  cur_record.focus_loc = cur_hover_loc.value
  cur_record.event_step = cur_sel_event_step

  cur_record.left.timestamp = cur_phase_time_str_left.value
  cur_record.right.timestamp = cur_phase_time_str_right.value

  cur_record.left.model = left_form_model.value
  cur_record.right.model = right_form_model.value

  cur_record.left.type = left_form_type.value
  cur_record.right.type = right_form_type.value

  cur_record.left.step = sel_step_left.value
  cur_record.right.step = sel_step_right.value

  cur_record.hover_loc = cur_hover_loc.value
  cur_record.hover_entity = cur_hover_entity.value

  if (left_form_type.value == 'feature') {
    cur_record.left.features = [cur_inner_indicator.value, cur_outer_indicator.value]
  }
  if (right_form_type.value == 'feature') {
    cur_record.right.features = [cur_inner_indicator.value, cur_outer_indicator.value]
  }
  if (left_form_type.value == 'error') {
    cur_record.left.focus_metric = inforStore.cur_timeline_indicator
  }
  if (right_form_type.value == 'error') {
    cur_record.right.focus_metric = inforStore.cur_timeline_indicator
  }
  cur_event_notes.value = ""
  console.log(inforStore.event_records);
  inforStore.event_records[`${inforStore.cur_phase_sorted_id}`].push(cur_record)
  // inforStore.event_records.push(cur_record)
}

function drawEntityAreaError(entity_id, step) {
  let margin_left = 36, margin_right = 16, margin_top = 17, margin_bottom = 18
  let main_w = 230, main_h = 94
  let svg_w = main_w + margin_left + margin_right
  let svg_h = main_h + margin_top + margin_bottom

  let eval_objs = inforStore.st_phase_events.phases[inforStore.cur_focused_model][inforStore.cur_phase_sorted_id].entity_metrics[step]
  let eval_objs_comnp
  let area_errors = []
  let area_errors_comp = []
  // let entity_loc_ids = inforStore.st_phase_events.phases[inforStore.cur_focused_model][inforStore.cur_phase_sorted_id].focus_entities[step][entity_id].loc_ids
  for (let i = 0; i < eval_objs.length; ++i) {
    for (let j = 0; j < eval_objs[i].length; ++j) {
      if (eval_objs[i][j].truth_entity_ids.includes(entity_id)) {
        area_errors.push(eval_objs[i][j].area_error[1])
      }
    }
  }

  if (inforStore.cur_baseline_model.length > 0) {
    eval_objs_comnp = inforStore.st_phase_events.phases[inforStore.cur_baseline_model][inforStore.cur_phase_sorted_id].entity_metrics[step]
    for (let i = 0; i < eval_objs_comnp.length; ++i) {
      for (let j = 0; j < eval_objs_comnp[i].length; ++j) {
        if (eval_objs_comnp[i][j].truth_entity_ids.includes(entity_id)) {
          area_errors_comp.push(eval_objs_comnp[i][j].area_error[1])
        }
      }
    }
  }

  let max_error = d3.max([...area_errors, ...area_errors_comp])
  let min_error = d3.min([...area_errors, ...area_errors_comp])
  
  d3.select('#entity-area-error').selectAll('*').remove()
  let svg = d3.select('#entity-area-error')
    .attr('width', svg_w)
    .attr('height', svg_h)
  
  let main_g = svg.append('g')
    .attr('transform', `translate(${margin_left}, ${margin_top})`)
  let xScale = d3.scaleLinear()
    .domain([1, inforStore.dataset_configs.output_window])
    .range([0, main_w])
  let yScale = d3.scaleLinear()
    .domain([min_error-1, max_error+1])
    .range([main_h, 0])
  
  let ticks_num = 1
  if (Math.floor(max_error - min_error) <= 5) ticks_num = Math.floor(max_error - min_error)
  else if (Math.floor(max_error - min_error) <= 16) ticks_num = Math.floor((max_error - min_error) / 2)
  else ticks_num = 8

  let xAxis = d3.axisBottom(xScale)
    .ticks(inforStore.dataset_configs.output_window.length)
    .tickFormat(d => `${d}`)
  let yAxis = d3.axisLeft(yScale)
    .ticks(ticks_num)
    .tickFormat(d => `${d.toFixed(0)}`)
  let xAxis_g = svg.append("g")
    .attr("transform", `translate(${margin_left}, ${margin_top+main_h})`) // 将X轴移至底部
    .call(xAxis);
  let yAxis_g = svg.append("g")
    .attr("transform", `translate(${margin_left}, ${margin_top})`) // 将X轴移至底部
    .call(yAxis);
  
  let bin_w = xScale(1)-xScale(0)
  if (inforStore.cur_baseline_model.length > 0) {
    bin_w /= 2
  }
  
  let limit_lines = main_g.append('g')
  if (max_error * min_error <= 0) {
    limit_lines.append('line')
    .attr('x1', 0).attr('x2', main_w)
    .attr('y1', yScale(0)).attr('y2', yScale(0))
    .attr('stroke', '#999')
    .attr('stroke-dasharray', '5,5')
  }
  
  const line = d3.line()
    .x((d,i) => xScale(i+1))
    .y((d,i) => yScale(d))

  let focus_pred = main_g.append('g')
  focus_pred.append('path')
    .datum(area_errors)
    .attr('d', line)  
    .attr('fill', 'none')
    .attr('stroke', '#0097A7')
  focus_pred.selectAll('circle')
    .data(area_errors)
    .join('circle')
      .attr('cx', (d,i) => xScale(i+1))
      .attr('cy', (d,i) => yScale(d))
      .attr('r', 2.5)
      .attr('fill', '#0097A7')
  if (inforStore.cur_baseline_model.length > 0) {
    let focus_pred_comp = main_g.append('g')
    focus_pred_comp.append('path')
      .datum(area_errors_comp)
      .attr('d', line)  
      .attr('fill', 'none')
      .attr('stroke', '#0097A7')
      .attr('stroke-dasharray', '5,5')
    focus_pred_comp.selectAll('circle')
      .data(area_errors_comp)
      .join('circle')
        .attr('cx', (d,i) => xScale(i+1))
        .attr('cy', (d,i) => yScale(d))
        .attr('r', 2.5)
        .attr('fill', 'none')
        .attr('stroke', '#0097A7')
  }
  
  svg.append('text')
    .attr('x', margin_left)
    .attr('y', 12)
    .attr('text-anchor', 'middle')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('area_error')
}

function drawEntityIntensityError(entity_id, step) {
  let margin_left = 36, margin_right = 16, margin_top = 17, margin_bottom = 18
  let main_w = 230, main_h = 94
  let svg_w = main_w + margin_left + margin_right
  let svg_h = main_h + margin_top + margin_bottom

  let eval_objs = inforStore.st_phase_events.phases[inforStore.cur_focused_model][inforStore.cur_phase_sorted_id].entity_metrics[step]
  let eval_objs_comp
  let area_errors = []
  let area_errors_comp = []
  // let entity_loc_ids = inforStore.st_phase_events.phases[inforStore.cur_focused_model][inforStore.cur_phase_sorted_id].focus_entities[step][entity_id].loc_ids
  for (let i = 0; i < eval_objs.length; ++i) {
    for (let j = 0; j < eval_objs[i].length; ++j) {
      if (eval_objs[i][j].truth_entity_ids.includes(entity_id)) {
        area_errors.push(eval_objs[i][j].intensity_error[1])
      }
    }
  }

  if (inforStore.cur_baseline_model.length > 0) {
    eval_objs_comp = inforStore.st_phase_events.phases[inforStore.cur_baseline_model][inforStore.cur_phase_sorted_id].entity_metrics[step]
    for (let i = 0; i < eval_objs_comp.length; ++i) {
      for (let j = 0; j < eval_objs_comp[i].length; ++j) {
        if (eval_objs_comp[i][j].truth_entity_ids.includes(entity_id)) {
          area_errors_comp.push(eval_objs_comp[i][j].intensity_error[1])
        }
      }
    }
  }

  let max_error = d3.max([...area_errors, ...area_errors_comp])
  let min_error = d3.min([...area_errors, ...area_errors_comp])
  
  d3.select('#entity-intensity-error').selectAll('*').remove()
  let svg = d3.select('#entity-intensity-error')
    .attr('width', svg_w)
    .attr('height', svg_h)
  
  let main_g = svg.append('g')
    .attr('transform', `translate(${margin_left}, ${margin_top})`)
  let xScale = d3.scaleLinear()
    .domain([1, inforStore.dataset_configs.output_window])
    .range([0, main_w])
  let yScale = d3.scaleLinear()
    .domain([min_error-1, max_error+1])
    .range([main_h, 0])
  
  let xAxis = d3.axisBottom(xScale)
    .ticks(inforStore.dataset_configs.output_window.length)
    .tickFormat(d => `${d}`)
  let yAxis = d3.axisLeft(yScale)
    .ticks(8)
    .tickFormat(d => `${d.toFixed(0)}`)
  let xAxis_g = svg.append("g")
    .attr("transform", `translate(${margin_left}, ${margin_top+main_h})`) // 将X轴移至底部
    .call(xAxis);
  let yAxis_g = svg.append("g")
    .attr("transform", `translate(${margin_left}, ${margin_top})`) // 将X轴移至底部
    .call(yAxis);
  
  let bin_w = xScale(1)-xScale(0)
  if (inforStore.cur_baseline_model.length > 0) {
    bin_w /= 2
  }
  
  let limit_lines = main_g.append('g')
  if (max_error * min_error <= 0) {
    limit_lines.append('line')
    .attr('x1', 0).attr('x2', main_w)
    .attr('y1', yScale(0)).attr('y2', yScale(0))
    .attr('stroke', '#999')
    .attr('stroke-dasharray', '5,5')
  }
  
  const line = d3.line()
    .x((d,i) => xScale(i+1))
    .y((d,i) => yScale(d))

  let focus_pred = main_g.append('g')
  focus_pred.append('path')
    .datum(area_errors)
    .attr('d', line)  
    .attr('fill', 'none')
    .attr('stroke', '#0097A7')
  focus_pred.selectAll('circle')
    .data(area_errors)
    .join('circle')
      .attr('cx', (d,i) => xScale(i+1))
      .attr('cy', (d,i) => yScale(d))
      .attr('r', 2.5)
      .attr('fill', '#0097A7')
  if (inforStore.cur_baseline_model.length > 0) {
    let focus_pred_comp = main_g.append('g')
    focus_pred_comp.append('path')
      .datum(area_errors_comp)
      .attr('d', line)  
      .attr('fill', 'none')
      .attr('stroke', '#0097A7')
      .attr('stroke-dasharray', '5,5')
    focus_pred_comp.selectAll('circle')
      .data(area_errors_comp)
      .join('circle')
        .attr('cx', (d,i) => xScale(i+1))
        .attr('cy', (d,i) => yScale(d))
        .attr('r', 2.5)
        .attr('fill', 'none')
        .attr('stroke', '#0097A7')
  }
  
  svg.append('text')
    .attr('x', margin_left)
    .attr('y', 12)
    .attr('text-anchor', 'middle')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('intensity_err')
}

function drawEntityMoveError(entity_id, step) {
  let margin_left = 36, margin_right = 16, margin_top = 17, margin_bottom = 18
  let main_w = 230, main_h = 94
  let svg_w = main_w + margin_left + margin_right
  let svg_h = main_h + margin_top + margin_bottom

  let eval_objs = inforStore.st_phase_events.phases[inforStore.cur_focused_model][inforStore.cur_phase_sorted_id].entity_metrics[step]
  let eval_objs_comnp
  let area_errors = []
  let area_errors_comp = []
  // let entity_loc_ids = inforStore.st_phase_events.phases[inforStore.cur_focused_model][inforStore.cur_phase_sorted_id].focus_entities[step][entity_id].loc_ids
  for (let i = 0; i < eval_objs.length; ++i) {
    for (let j = 0; j < eval_objs[i].length; ++j) {
      if (eval_objs[i][j].truth_entity_ids.includes(entity_id)) {
        area_errors.push(eval_objs[i][j].shift_error)
      }
    }
  }

  if (inforStore.cur_baseline_model.length > 0) {
    eval_objs_comnp = inforStore.st_phase_events.phases[inforStore.cur_baseline_model][inforStore.cur_phase_sorted_id].entity_metrics[step]
    for (let i = 0; i < eval_objs_comnp.length; ++i) {
      for (let j = 0; j < eval_objs_comnp[i].length; ++j) {
        if (eval_objs_comnp[i][j].truth_entity_ids.includes(entity_id)) {
          area_errors_comp.push(eval_objs_comnp[i][j].shift_error)
        }
      }
    }
  }

  console.log('area_errors', area_errors, area_errors_comp);

  let max_error = d3.max([...area_errors, ...area_errors_comp])
  let min_error = d3.min([...area_errors, ...area_errors_comp])
  
  d3.select('#entity-move-error').selectAll('*').remove()
  let svg = d3.select('#entity-move-error')
    .attr('width', svg_w)
    .attr('height', svg_h)
  
  let main_g = svg.append('g')
    .attr('transform', `translate(${margin_left}, ${margin_top})`)
  let xScale = d3.scaleLinear()
    .domain([1, inforStore.dataset_configs.output_window])
    .range([0, main_w])
  let yScale = d3.scaleLinear()
    .domain([min_error-1, max_error+1])
    .range([main_h, 0])

  let xAxis = d3.axisBottom(xScale)
    .ticks(inforStore.dataset_configs.output_window.length)
    .tickFormat(d => `${d}`)
  let yAxis = d3.axisLeft(yScale)
    .ticks(4)
    .tickFormat(d => `${d.toFixed(1)}`)
  let xAxis_g = svg.append("g")
    .attr("transform", `translate(${margin_left}, ${margin_top+main_h})`) // 将X轴移至底部
    .call(xAxis);
  let yAxis_g = svg.append("g")
    .attr("transform", `translate(${margin_left}, ${margin_top})`) // 将X轴移至底部
    .call(yAxis);
  
  let bin_w = xScale(1)-xScale(0)
  if (inforStore.cur_baseline_model.length > 0) {
    bin_w /= 2
  }
  
  let limit_lines = main_g.append('g')
  if (max_error * min_error <= 0) {
    limit_lines.append('line')
    .attr('x1', 0).attr('x2', main_w)
    .attr('y1', yScale(0)).attr('y2', yScale(0))
    .attr('stroke', '#999')
    .attr('stroke-dasharray', '5,5')
  }
  
  const line = d3.line()
    .x((d,i) => xScale(i+1))
    .y((d,i) => yScale(d))
    .defined(function(d) { return d !== null; })

  let focus_pred = main_g.append('g')
  focus_pred.append('path')
    .datum(area_errors)
    .attr('d', line)  
    .attr('fill', 'none')
    .attr('stroke', '#0097A7')
  focus_pred.selectAll('circle')
    .data(area_errors)
    .join('circle')
      .attr('cx', (d,i) => xScale(i+1))
      .attr('cy', (d,i) => yScale(d))
      .attr('r', 2.5)
      .attr('fill', (d) => {
        if (d > 0) return '#0097A7'
        else return 'none'
      })
  if (inforStore.cur_baseline_model.length > 0) {
    let focus_pred_comp = main_g.append('g')
    focus_pred_comp.append('path')
      .datum(area_errors_comp)
      .attr('d', line)  
      .attr('fill', 'none')
      .attr('stroke', '#0097A7')
      .attr('stroke-dasharray', '5,5')
    focus_pred_comp.selectAll('circle')
      .data(area_errors_comp)
      .join('circle')
        .attr('cx', (d,i) => xScale(i+1))
        .attr('cy', (d,i) => yScale(d))
        .attr('r', 2.5)
        .attr('fill', 'none')
        .attr('stroke', (d) => {
          if (d > 0) return '#0097A7'
          else return 'none'
        })
  }
  
  svg.append('text')
    .attr('x', margin_left)
    .attr('y', 12)
    .attr('text-anchor', 'middle')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('shift_error')
}

</script>

<template>
  <div class="models-container">
    <div ref="reference" class="title-layer">
      <div style="width: 200px">Event Exploration</div>
      <div class="left-form-region">
        <div v-if="(inforStore.cur_detail_type == 'phase') && (inforStore.cur_phase_sorted_id != -1)" class="sel-title">Type: </div>
        <div v-if="(inforStore.cur_detail_type == 'phase') && (inforStore.cur_phase_sorted_id != -1)" class="data-dropdown">
          <button class="btn dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false">{{ left_form_type }}</button>
          <ul class="dropdown-menu">
            <li v-for="(item, index) in view_form_types" :value="item" @click="onLeftFormTypeSel(item)" class='dropdown-item' :key="index">
              <div class="li-data-name">{{ item }}</div>
            </li>
          </ul>
        </div>

        <div v-if="(inforStore.cur_detail_type == 'phase') && (inforStore.cur_phase_sorted_id != -1)" class="sel-title">Model: </div>
        <div v-if="(inforStore.cur_detail_type == 'phase') && (inforStore.cur_phase_sorted_id != -1)" class="data-dropdown">
          <button class="btn dropdown-toggle" style="width: 140px !important" type="button" data-bs-toggle="dropdown" aria-expanded="false">{{ left_form_model }}</button>
          <ul class="dropdown-menu">
            <li v-for="(item, index) in inforStore.cur_sel_models" :value="item" @click="onLeftFormModelSel(item)" class='dropdown-item' :key="index">
              <div class="li-data-name">{{ item }}</div>
            </li>
          </ul>
        </div>

        <div v-if="(inforStore.cur_detail_type == 'phase') && (inforStore.cur_phase_sorted_id != -1)" class="sel-title">Step: </div>
        <div v-if="(inforStore.cur_detail_type == 'phase') && (inforStore.cur_phase_sorted_id != -1)" class="data-dropdown">
          <button class="btn dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false">{{ sel_step_left }}</button>
          <ul class="dropdown-menu">
            <li v-for="(item, index) in step_opts" :value="item" @click="onLeftStepTypeSel(item)" class='dropdown-item' :key="index">
              <div class="li-data-name">{{ item }}</div>
            </li>
          </ul>
        </div>
        <div id="left-form-btn" class="iconfont" data-bs-toggle="collapse" data-bs-target="#left-form-collapse" aria-controls="left-form-collapse">&#xe8ca;</div>
        <!-- <div id="left-form-btn" class="iconfont" data-bs-toggle="collapse" data-bs-target="#collapseWidthExample" aria-expanded="false" aria-controls="collapseWidthExample" @mouseover="onFormBtnHover('left')" @mouseout="onFormBtnout('left')">&#xe8ca;</div> -->
      </div>
      <div v-if="(inforStore.cur_detail_type == 'phase') && (inforStore.cur_phase_sorted_id != -1)" class="right-form-region">
        <div class="right-normal-forms">
          <div class="sel-title">Type: </div>
          <div class="data-dropdown">
            <button class="btn dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false">{{ right_form_type }}</button>
            <ul class="dropdown-menu">
              <li v-for="(item, index) in view_form_types" :value="item" @click="onRightFormTypeSel(item)" class='dropdown-item' :key="index">
                <div class="li-data-name">{{ item }}</div>
              </li>
            </ul>
          </div>

          <div class="sel-title">Model: </div>
          <div class="data-dropdown">
            <button class="btn dropdown-toggle" style="width: 140px !important" type="button" data-bs-toggle="dropdown" aria-expanded="false">{{ right_form_model }}</button>
            <ul class="dropdown-menu">
              <li v-for="(item, index) in inforStore.cur_sel_models" :value="item" @click="onRightFormModelSel(item)" class='dropdown-item' :key="index">
                <div class="li-data-name">{{ item }}</div>
              </li>
            </ul>
          </div>

          <div class="sel-title">Step: </div>
          <div class="data-dropdown">
            <button class="btn dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false">{{ sel_step_right }}</button>
            <ul class="dropdown-menu">
              <li v-for="(item, index) in step_opts" :value="item" @click="onRightStepTypeSel(item)" class='dropdown-item' :key="index">
                <div class="li-data-name">{{ item }}</div>
              </li>
            </ul>
          </div>
          <div id="right-form-btn" class="iconfont" data-bs-toggle="collapse" data-bs-target="#right-form-collapse" aria-controls="right-form-collapse">&#xe8ca;</div>
        </div>
        <div class="module-btn-region">
          <button class="btn btn-primary module-btn" type="button" data-bs-toggle="collapse" data-bs-target="#event-module"  aria-controls="event-module">Event Selection</button>
          <!-- <button class="btn btn-primary module-btn" type="button" data-bs-toggle="collapse" data-bs-target="#subset-module"  aria-controls="subset-module">Subgroup Module</button> -->
        </div>
      </div>
    </div>
    <div v-if="(inforStore.cur_detail_type == 'phase') && (inforStore.cur_phase_sorted_id != -1)" class="collapse" id="left-form-collapse">
      <div v-if="left_form_type=='feature'" class="form-row">
        <div class="sel-title">Outer indicator: </div>
        <div class="data-dropdown">
          <button class="btn dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false">{{ cur_outer_indicator }}</button>
          <ul class="dropdown-menu">
            <li v-for="(item, index) in inforStore.input_feats" :value="item" @click="onOuterIndicatorSel(item)" class='dropdown-item' :key="index">
              <div class="li-data-name">{{ item }}</div>
            </li>
          </ul>
        </div>
      </div>
      <div v-if="left_form_type=='feature'" class="form-row">
        <div class="sel-title">Inner indicator: </div>
        <div class="data-dropdown">
          <button class="btn dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false">{{ cur_inner_indicator }}</button>
          <ul class="dropdown-menu">
            <li v-for="(item, index) in inforStore.input_feats" :value="item" @click="onInnerIndicatorSel(item)" class='dropdown-item' :key="index">
              <div class="li-data-name">{{ item }}</div>
            </li>
          </ul>
        </div>
      </div>
      <div v-if="left_form_type=='error'" class="form-row">
        <div class="sel-title">Focused indicator: </div>
        <div class="data-dropdown">
          <button class="btn dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false">{{ inforStore.cur_timeline_indicator }}</button>
          <ul class="dropdown-menu">
            <li v-for="(item, index) in inforStore.indicators_list" :value="item" @click="onTimelineIndicatorSel(item)" class='dropdown-item' :key="index">
              <div class="li-data-name">{{ item }}</div>
            </li>
          </ul>
        </div>
      </div>
      <div v-if="left_form_type=='error'" class="form-row">
        <div class="form-check" v-for="(item, index) in other_indicators" :key="index">
          <input class="form-check-input" type="checkbox" :value="item" v-model="other_focused_indicators">
          <label class="form-check-label">{{ item }}</label>
        </div>
      </div>
    </div>
    <div v-if="(inforStore.cur_detail_type == 'phase') && (inforStore.cur_phase_sorted_id != -1)" class="collapse" id="right-form-collapse">
      <div v-if="right_form_type=='feature'" class="form-row">
        <div class="sel-title">Outer indicator: </div>
        <div class="data-dropdown">
          <button class="btn dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false">{{ cur_outer_indicator }}</button>
          <ul class="dropdown-menu">
            <li v-for="(item, index) in inforStore.input_feats" :value="item" @click="onOuterIndicatorSel(item)" class='dropdown-item' :key="index">
              <div class="li-data-name">{{ item }}</div>
            </li>
          </ul>
        </div>
      </div>
      <div v-if="right_form_type=='feature'" class="form-row">
        <div class="sel-title">Inner indicator: </div>
        <div class="data-dropdown">
          <button class="btn dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false">{{ cur_inner_indicator }}</button>
          <ul class="dropdown-menu">
            <li v-for="(item, index) in inforStore.input_feats" :value="item" @click="onInnerIndicatorSel(item)" class='dropdown-item' :key="index">
              <div class="li-data-name">{{ item }}</div>
            </li>
          </ul>
        </div>
      </div>
      <div v-if="right_form_type=='error'" class="form-row">
        <div class="sel-title">Focused indicator: </div>
        <div class="data-dropdown">
          <button class="btn dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false">{{ inforStore.cur_timeline_indicator }}</button>
          <ul class="dropdown-menu">
            <li v-for="(item, index) in inforStore.indicators_list" :value="item" @click="onTimelineIndicatorSel(item)" class='dropdown-item' :key="index">
              <div class="li-data-name">{{ item }}</div>
            </li>
          </ul>
        </div>
      </div>
      <div v-if="right_form_type=='error'" class="form-row">
        <div class="form-check" v-for="(item, index) in other_indicators" :key="index">
          <input class="form-check-input" type="checkbox" :value="item" v-model="other_focused_indicators">
          <label class="form-check-label">{{ item }}</label>
        </div>
      </div>
    </div>
    <div class="collapse" id="event-module">
      <EventsView />
    </div>
    <!-- <div class="collapse" id="subset-module">
      <SubsetList />
    </div> -->
    <div id="exploration-block">
      <div class="st-layout-container">
        <!-- <svg id="st-layout-feature"></svg> -->
        <svg id="st-layout-left"></svg>
        <!-- <div class="seg-line"></div> -->
        <div class="loc-infor-region">
          <div>
            <div v-if="cur_hover_entity!=-1" style="text-align:center; display: flex; justify-content: center;">  
              <div style="margin-right: 20px;"><span class="tooltip-title">Area:</span> <span class="tooltip-val">{{ inforStore.st_phase_events.phases[inforStore.cur_focused_model][inforStore.cur_phase_sorted_id].focus_entities[cur_sel_step][cur_hover_entity].area }}</span></div>
              <div><span class="tooltip-title">Intensity:</span> <span class="tooltip-val">{{ inforStore.st_phase_events.phases[inforStore.cur_focused_model][inforStore.cur_phase_sorted_id].focus_entities[cur_sel_step][cur_hover_entity].intensity.toFixed(2) }}</span></div>
            </div>
            <div v-if="cur_hover_entity!=-1" class="horizontal-line"></div>
            <svg v-if="cur_hover_entity!=-1" id="entity-area-error"></svg>
            <svg v-if="cur_hover_entity!=-1" id="entity-intensity-error"></svg>
            <svg v-if="cur_hover_entity!=-1" id="entity-move-error"></svg>
          </div>
        <!-- *********************************************************************************** -->
          <div v-if="cur_hover_loc==-1" class="seg-line"></div>
          <div>
            <div v-if="cur_hover_loc!=-1" style="text-align:center;">
              <!-- <span class="tooltip-title">Location:</span> -->
              <span class="tooltip-val">{{ inforStore.loc_regions[cur_hover_loc] }} {{ inforStore.cur_data_infor.space.loc_list[cur_hover_loc].geometry.coordinates }}
            </span></div>
            <div v-if="cur_hover_loc!=-1" class="horizontal-line"></div>
            <svg v-if="cur_hover_loc!=-1" id="loc-residual-infor"></svg>
            <div v-if="cur_hover_loc!=-1" class="horizontal-line"></div>
            <svg v-if="cur_hover_loc!=-1" id="loc-time-context"></svg>
            <div v-if="cur_hover_loc!=-1">
              <div v-for="(item, index) in inforStore.dataset_configs.features" :key="index">
                <svg v-if="index > 0" :id="view_id('loc-feature', index)"></svg>
                <div v-if="cur_hover_loc!=-1" class="horizontal-line"></div>
              </div>
            </div>
          </div>
        </div>
        
        <!-- <svg id="st-layout-phase"></svg> -->
        <svg id="st-layout-right"></svg>
      </div>
      <svg id="time-event-bar"></svg>
      <div v-if="inforStore.cur_phase_sorted_id != -1" class="cur_stamp-row-left">
        <div class="title">Timestamp: </div>
        <div class="cur_stamp">{{ cur_phase_time_str_left }}</div>
      </div>
      <div v-if="inforStore.cur_phase_sorted_id != -1" class="cur_stamp-row-right">
        <div class="title">Timestamp: </div>
        <div class="cur_stamp">{{ cur_phase_time_str_right }}</div>
      </div>
      <div v-if="inforStore.cur_phase_sorted_id != -1" id="float-buttons">
        <div id="link-view-btn" :linked="view_linked ? 'yes': 'no'" @click="onLinkBtnClick()">
          <div id="link-view-text">Link Views</div>
          <div id="link-view-icon" class="iconfont">&#xe891;</div>
        </div>
        <Popper placement="top">
          <div id="record-btn" :showing="recordLayerShow" @click="onRecordIconClick()">
            <div id="record-text">Record</div>
            <div id="record-icon" class="iconfont">&#xe627;</div>
          </div>
          <template #content>
            <div style="width: 300px;">
              <textarea class="form-control" rows="3" v-model="cur_event_notes" placeholder="Notes..."></textarea>
              <div style="display:flex; justify-content: flex-end;">
                <button id="save-record-btn" class="btn btn-primary" @click="onSaveRecordClick()">Record</button>
              </div>
            </div>
          </template>
        </Popper>
      </div>
      <div class="timebar-legends-container">
        <svg id="time-bar-legends" width=1 height=1></svg>
      </div>
      <!-- <div v-if="inforStore.cur_phase_sorted_id != -1" class="feature-panel-title">Features: </div> -->
      <!-- <div class="feature-panel">
        <div v-for="(item, index) in inforStore.feature_infor.input.split(', ')" :key="index"><span v-if="cur_hover_loc != -1">{{item}}: {{inforStore.cur_data_infor.raw_data[global_time_id][cur_hover_loc][index]}}</span></div>
      </div> -->
    </div>
    <!-- <div id="subset-detail-tooltip"></div> -->
    <div v-if="inforStore.cur_detail_type == 'phase'" id="loc-tooltip-left">
      <div v-if="cur_hover_loc!=-1"><span class="tooltip-title">Location:</span> <span class="tooltip-val">{{ inforStore.cur_data_infor.space.loc_list[cur_hover_loc].geometry.coordinates }} {{ inforStore.loc_regions[cur_hover_loc] }}</span></div>
      <div class="horizontal-line"></div>
      <div v-if="left_form_type == 'feature'">
        <div v-if="cur_hover_loc!=-1">
          <span class="tooltip-title">Pollutants:</span> <span class="tooltip-item" v-for="(item, index) in inforStore.type_feats.Pollutants" :key="index"><span class="tooltip-title">{{item}}-</span><span class="tooltip-val">{{inforStore.cur_phase_data.phase_raw_data[cur_step_left][cur_hover_loc][index]}}</span></span>
        </div>
        <div v-if="cur_hover_loc!=-1">
          <span class="tooltip-title">Weather:</span> <span class="tooltip-item" v-for="(item, index) in inforStore.type_feats.Weather" :key="index"><span class="tooltip-title">{{item}}-</span><span class="tooltip-val">{{inforStore.cur_phase_data.phase_raw_data[cur_step_left][cur_hover_loc][6+index]}}</span></span>
        </div>
        <!-- <div v-if="cur_hover_loc!=-1">
          <span class="tooltip-title">Space:</span> <span class="tooltip-item" v-for="(item, index) in inforStore.type_feats.Space" :key="index"><span class="tooltip-title">{{item}}-</span><span class="tooltip-val">{{inforStore.cur_phase_data.phase_raw_data[cur_step_left][cur_hover_loc][13+index]}}</span></span>
        </div> -->
      </div>
      <div v-if="left_form_type == 'error'">
        <div v-if="cur_hover_loc!=-1"><span class="tooltip-title">Truth:</span> <span class="tooltip-val">{{inforStore.cur_phase_data.phase_raw_data[cur_step_left][cur_hover_loc][0]}}</span></div>
        <div><span class="tooltip-title">Residuals:</span></div>
        <div class="tooltip-row-grids" v-if="cur_hover_loc!=-1">
          <span class="tooltip-item" v-for="(item, index) in inforStore.sel_phase_details[inforStore.cur_focused_model].st_residuals[cur_step_left][cur_hover_loc]" :key="index"><span class="tooltip-title">fore_step {{index}}:</span> <span>{{ num_fix_2(inforStore.sel_phase_details[inforStore.cur_focused_model].phase_pred_val[cur_step_left][cur_hover_loc][index]) }}</span><span v-if="inforStore.cur_baseline_model.length>0">/{{ num_fix_2(inforStore.sel_phase_details[inforStore.cur_baseline_model].phase_pred_val[cur_step_left][cur_hover_loc][index]) }}</span> (<span class="tooltip-val">{{ num_fix_2(item) }}</span><span v-if="inforStore.cur_baseline_model.length>0" class="tooltip-val">/{{ num_fix_2(inforStore.sel_phase_details[inforStore.cur_baseline_model].st_residuals[cur_step_left][cur_hover_loc][index]) }}</span>)</span>
        </div>
      </div>
    </div>
    <div v-if="inforStore.cur_detail_type == 'phase'" id="loc-tooltip-right">
      <div v-if="cur_hover_loc!=-1"><span class="tooltip-title">Location:</span> <span class="tooltip-val">{{ inforStore.cur_data_infor.space.loc_list[cur_hover_loc].geometry.coordinates }} {{ inforStore.loc_regions[cur_hover_loc] }}</span></div>
      <div class="horizontal-line"></div>
      <div v-if="right_form_type == 'feature'">
        <div v-if="cur_hover_loc!=-1">
          <span class="tooltip-title">Pollutants:</span> <span class="tooltip-item" v-for="(item, index) in inforStore.type_feats.Pollutants" :key="index"><span class="tooltip-title">{{item}}-</span><span class="tooltip-val">{{inforStore.cur_phase_data.phase_raw_data[cur_step_right][cur_hover_loc][index]}}</span></span>
        </div>
        <div v-if="cur_hover_loc!=-1">
          <span class="tooltip-title">Weather:</span> <span class="tooltip-item" v-for="(item, index) in inforStore.type_feats.Weather" :key="index"><span class="tooltip-title">{{item}}-</span><span class="tooltip-val">{{inforStore.cur_phase_data.phase_raw_data[cur_step_right][cur_hover_loc][6+index]}}</span></span>
        </div>
        <!-- <div v-if="cur_hover_loc!=-1">
          <span class="tooltip-title">Space:</span> <span class="tooltip-item" v-for="(item, index) in inforStore.type_feats.Space" :key="index"><span class="tooltip-title">{{item}}-</span><span class="tooltip-val">{{inforStore.cur_phase_data.phase_raw_data[cur_step_right][cur_hover_loc][13+index]}}</span></span>
        </div> -->
      </div>
      <div v-if="right_form_type == 'error'">
        <div v-if="cur_hover_loc!=-1"><span class="tooltip-title">Truth:</span> <span class="tooltip-val">{{inforStore.cur_phase_data.phase_raw_data[cur_step_right][cur_hover_loc][0]}}</span></div>
        <div><span class="tooltip-title">Residuals:</span></div>
        <div class="tooltip-row-grids" v-if="cur_hover_loc!=-1">
          <span class="tooltip-item" v-for="(item, index) in inforStore.sel_phase_details[inforStore.cur_focused_model].st_residuals[cur_step_right][cur_hover_loc]" :key="index"><span class="tooltip-title">fore_step {{index}}:</span> <span>{{ num_fix_2(inforStore.sel_phase_details[inforStore.cur_focused_model].phase_pred_val[cur_step_right][cur_hover_loc][index]) }}</span><span v-if="inforStore.cur_baseline_model.length>0">/{{ num_fix_2(inforStore.sel_phase_details[inforStore.cur_baseline_model].phase_pred_val[cur_step_right][cur_hover_loc][index]) }}</span> (<span class="tooltip-val">{{ num_fix_2(item) }}</span><span v-if="inforStore.cur_baseline_model.length>0" class="tooltip-val">/{{ num_fix_2(inforStore.sel_phase_details[inforStore.cur_baseline_model].st_residuals[cur_step_right][cur_hover_loc][index]) }}</span>)</span>
        </div>
      </div>
    </div>
  </div>

</template>

<style scoped>
.models-container {
  width: 1604px;
  /* width: 860px; */
  height: 692px;
  border: solid 1px #c2c5c5;
  border-radius: 6px;
  /* padding: 1px; */
  margin: 2px;
  overflow-y: auto;
}

.title-layer {
  /* position: absolute; */
  z-index: 80;
  width: 1600px;
  height: 20px;
  text-align: left;
  padding-left: 12px;
  /* background-color: #6c757d; */
  /* color: #fff; */
  margin-top: 10px;
  margin-bottom: 4px;
  /* font: 700 16px "Microsort Yahei"; */
  font: 700 20px "Arial";
  /* letter-spacing: 1px; */
  color: #333;
  display: flex;
  align-items: center;
  justify-content: flex-start;
}

#exploration-block {
  position: relative;
  /* width: 850px;
  height: 780px; */
}

.st-layout-container {
  display: flex;
  justify-content: space-around;
  /* align-items: center; */
}

.left-form-region {
  width: 626px;
  height: 20px;
  display: flex;
  align-items: center;
  /* border-right: solid 2px #cecece; */
}

.right-form-region {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 800px;
}

.right-normal-forms {
  display: flex;
  align-items: center;
  margin-left: 14px;
}

.error-form-region {
  display: flex;
  align-items: center;
}

.model-btn-region {
  display: flex;
  align-items: center;
}

.module-btn {
  font-size: 14px;
  padding: 2px 4px 2px 4px;
  height: 26px;
}

.seg-line {
  width: 1px;
  border: solid 1px #cecece;
  margin-top: 40px;
  height: 375px;
}

.feature-panel {
  position: absolute; /* 或者使用 fixed，根据需求选择 */
  top: 14.5%; /* 垂直居中 */
  left: 3%; /* 水平居中 */
  background-color: #fff;
}
.feature-panel-title {
  position: absolute; /* 或者使用 fixed，根据需求选择 */
  top: 12%; /* 垂直居中 */
  left: 3%; /* 水平居中 */
  background-color: #fff;
  font-weight: 700;
  font-size: 14px;
}

.timebar-legends-container {
  position: absolute;
  top: 66.5%; /* 垂直居中 */
  left: 0.5%; /* 水平居中 */
  /* color: #cecece; */
  /* text-align: center; */
}

#float-buttons {
  position: absolute;
  top: 67%; /* 垂直居中 */
  left: 45.55%; /* 水平居中 */
  display: flex;
}

#save-record-btn {
  margin-top: 8px;
  font-size: 14px;
  padding: 2px 7px;
}

#record-btn {
  margin-left: 10px;
  text-align: center;
  color: #ababab;
}

#record-btn[showing=true] {
  color: #1a73e8;
}


#record-btn:hover {
  cursor: pointer;
}

#record-text {
  font-size: 14px;
}

#record-icon {
  font-size: 20px;
  margin-top: -6px;
}

#link-view-btn {
  color: #ababab;
  text-align: center;
}
#link-view-btn:hover {
  cursor: pointer;
}
#link-view-btn[linked='yes'] {
  color: #1a73e8;
}
#link-view-btn[linked='no'] {
  color: #ababab;
}
#link-view-icon {
  display: block;
  margin-top: -16px;
  font-size: 32px;
}
#link-view-text {
  font-size: 14px;
}

.cur_stamp-row-left {
  position: absolute; /* 或者使用 fixed，根据需求选择 */
  top: 11%; /* 垂直居中 */
  left: 1.2%; /* 水平居中 */
  display: flex;
  background-color: #fff;
  font-size: 14px;
}

.cur_stamp-row-right {
  position: absolute; /* 或者使用 fixed，根据需求选择 */
  top: 11%; /* 垂直居中 */
  left: 60%; /* 水平居中 */
  display: flex;
  background-color: #fff;
  font-size: 14px;
}

.cur_stamp-row-left .title,
.cur_stamp-row-right .title {
  font-weight: 700;
}

.cur_stamp-row-left .cur_stamp,
.cur_stamp-row-right .cur_stamp {
  margin-left: 5px;
  color: #1a73e8;
}

.sel-title {
  margin-left: 16px;
  font-size: 14px;
  font-weight: 700;
  font-family: "Helvetica Neue", Helvetica, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "微软雅黑", Arial, sans-serif;
  color: #515A6E;
}

.form-check {
  font-size: 14px;
  margin-left: 16px;
  display: flex;
  align-items: center;
}

.form-check-label {
  font-weight: 14px;
  font-weight: 400;
  margin-left: 6px;
  margin-top: 2px;
}

.data-dropdown .dropdown-toggle {
  width: 100px !important;
  height: 24px;
  /* width: 120px; */
  padding: 0px 2px 0 4px;
  /* padding-bottom: -10px; */
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

.select-config-row {
  display: flex;
  justify-content: space-around;
  align-items: center;
}

#loc-tooltip-left,
#loc-tooltip-right {
  width: 440px;
  position: absolute;
  padding: 10px;
  background-color: #fff;
  border: 1px solid #999;
  border-radius: 5px;
  pointer-events: none;
  opacity: 0;
}

#time-event-bar {
  margin-top: -164px;
}

#subset-module {
  position: absolute;
  background-color: #fff;
  z-index:9999;
  height: 628px;
  padding: 0;
  margin-left: -2px
}
#event-module {
  position: absolute;
  background-color: #fff;
  z-index:9999;
  padding: 0;
  margin-left: -2px
}


.form-row {
  display: flex
}

#left-form-collapse, 
#right-form-collapse {
  z-index: 999;
  position: absolute;
  margin-left: 480px;
  width: 280px;
  border: solid 1px #cecece;
  border-radius: 5px;
  padding: 8px 4px 12px 4px;
  background-color: #fff;
}

#right-form-collapse {
  margin-left: 1080px;
  width: 290px;
}

#left-form-btn,
#right-form-btn {
  margin-left: 16px;
  font-size: 22px;
  color: #999;
  cursor: pointer;
}
#left-form-btn:hover,
#right-form-btn:hover {
  color: #1a73e8;
}

.module-btn-region {
  width: 120px;
  display: flex;
  justify-content: space-around;
}

.tooltip-title {
  font-size: 14px;
  font-weight: 700;
  color: #333;
}
.tooltip-val {
  font-size: 14px;
  color: #1a73e8;
}
.tooltip-item {
  margin-right: 10px;
  flex-basis: 45%;
}
.horizontal-line {
  border-bottom: 1px solid #cecece; /* 分割线颜色和粗细 */
  margin-top: 6px; /* 可选，增加一些下边距以避免内容过于紧凑 */
  margin-bottom: 6px; /* 可选，增加一些下边距以避免内容过于紧凑 */
}
.tooltip-row-grids {
  display: flex;
  flex-wrap: wrap;
}

.loc-infor-region {
  display: flex;
  justify-content: center;
  margin-top: 10px;
  width: 360px !important;
  height: 420px;
  overflow-x: auto;
  overflow-y: auto;
}

.event-header:hover {
  background-color: #333;
}

/* #loc-residual-infor,
#loc-time-context {
  margin-left: -100px;
  z-index: 999;
} */
</style>