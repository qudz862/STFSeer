<script setup>
import { ref, reactive, computed, watch, onMounted, onUpdated } from "vue"
import { useInforStore } from '@/stores/infor.js'
import getData from '@/services/index.js'
import * as d3 from 'd3'
import $ from 'jquery'
import { valColorScheme_blue, valColorScheme_red, valColorScheme_fire } from '@/data/index.js'

const inforStore = useInforStore()
let global_time_id = ref(0)
let cur_sel_step = ref(0)
let cur_temporal_focus_cnt = ref([])
let cur_temporal_focus_locs = ref([])

let cur_transform_state = {k: 1, x: 0, y: 0}
let zoom_func = d3.zoom()
    .scaleExtent([0.1, 10])
    .on('zoom', handleZoom);
function handleZoom(e) {
  cur_transform_state = e.transform
  d3.select('#space-layout-g').attr('transform', cur_transform_state)
}

onMounted(() => {
  if (inforStore.month_id = -1) {
    // 绘制全局
    drawGlobalSTLayout()
    d3.select('#st-layout').call(zoom_func.transform, cur_transform_state)
    d3.select('#st-layout').call(zoom_func)
  } else {
    drawTimeBar()
    drawSTLayout()
    d3.select('#st-layout').call(zoom_func.transform, cur_transform_state)
    d3.select('#st-layout').call(zoom_func)
  }
})

onUpdated(() => {

})

watch (() => cur_sel_step.value, (oldVlaue, newValue) => {
  let month_id = inforStore.month_id
  let month_infor = inforStore.st_phase_events.phases_list[month_id]
  global_time_id.value = parseInt(cur_sel_step.value) + parseInt(month_infor.start)
  drawSTLayout()
})
let grid_points, line

let cur_phase_time_str = ref('')

function drawGlobalSTLayout() {

}

function drawSTLayout() {
  let month_id = inforStore.month_id
  let phase_list = inforStore.st_phase_events.phases_list
  let cur_step
  
  let stamp_id = phase_list[phase_id].start + cur_step
  let stamp_strs = inforStore.st_phase_events.time_strs
  
  let svg_id = '#st-layout'
  d3.select(svg_id).selectAll('*').remove()

  let space_layout_w = 870, space_layout_h = 870  
  let margin_left = 10, margin_right = 10
  let st_layout_w = space_layout_w + margin_left + margin_right
  let st_layout_h = space_layout_h - 5
  let st_layout_svg = d3.select(svg_id)
    .attr('class', 'st-layout-svg')
    .attr('width', st_layout_w)
    .attr('height', st_layout_h)

  // 获取空间位置
  let loc_coords_x = inforStore.cur_data_infor.space.loc_list.map(item => item.geometry.coordinates[0])
  let loc_coords_y = inforStore.cur_data_infor.space.loc_list.map(item => item.geometry.coordinates[1])
  let grid_borders = inforStore.cur_data_infor.space.grid_borders
  grid_points = grid_borders.map(item => item['border_points'])
  let grid_points_x = grid_points.map(item => item.map(e => e[0])).flat()
  let grid_points_y = grid_points.map(item => item.map(e => e[1])).flat()
  
  let white_rate = 0.05
  let points_x_extent = d3.extent(grid_points_x)
  let points_y_extent = d3.extent(grid_points_y)
  let loc_x_scale = d3.scaleLinear()
        .domain(points_x_extent)
        .range([space_layout_w*white_rate, space_layout_w*(1-white_rate)])
  let loc_y_scale = d3.scaleLinear()
        .domain(points_y_extent)
        .range([space_layout_h*(1-white_rate), space_layout_h*white_rate])
  let space_layout_g = st_layout_svg.append('g')
    .attr('id', 'space-layout-g')
    .attr('class', 'space-layout-g')
    // .attr('transform', `translate(${margin_left}, -30)`)
    .attr('transform', () => {
      if (view_linked.value) return cur_transform_state
      else return transform_state_right
    })
  // let points_x_range = points_x_extent[1] - points_x_extent[0]
  // let points_y_range = points_y_extent[1] - points_y_extent[0]
  let grid_borders_g = space_layout_g.append('g')
    .attr('transform', `translate(${margin_left+space_layout_w*0.08}, 4) scale(0.85)`)
    .on('click', (e) => {
      // 点击空间点，弹出tooltip显示信息
      
    })
    .on('mouseover', (e) => {
      // 鼠标划过时，改变网格效果
    })
    .on('mouseout', (e) => {
      // 鼠标划出时，恢复网格效果
    })
  
  let pollu_grid_borders_g = space_layout_g.append('g')
    .attr('transform', `translate(${margin_left+space_layout_w*0.08}, 4) scale(0.85)`)

  line = d3.line()
    .x(d => loc_x_scale(d[0])) // x 坐标映射
    .y(d => loc_y_scale(d[1])); // y 坐标映射
  let level_id_list = []
  for (let i = 0; i < inforStore.cur_val_bins.length-1; ++i) {
    level_id_list.push(i)
  }
  let boxColorLevel = d3.scaleOrdinal()
    .domain(level_id_list)
    .range(['#fcfbfd', '#efedf5', '#dadaeb', '#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#4a1486'])
  let borders_g = grid_borders_g.append('g')
  borders_g.selectAll('path')
    .data(grid_points)
    .join('path')
      .attr('class', 'right-loc-border')
      .attr('id', (d,i) => `right_loc-${i}`)
      .attr('loc_id', (d,i) => i)
      .attr("d", line) // 应用生成器
      .attr("fill", (d, i) => boxColorLevel(inforStore.sel_phase_details.phase_raw_level[cur_step][i]))
      .attr("stroke", "#999")
      .attr("stroke-width", 1)
      .attr("opacity", 0.9)
  let pollu_grid_points = grid_points.filter(function(value, index) {
    // let cur_level = inforStore.sel_phase_details.phase_raw_level[cur_step][index]
    // let cur_th = inforStore.cur_val_bins[cur_level]
    let cur_val = inforStore.sel_phase_details.phase_raw_val[cur_step][index]
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
  
  // 极值、异常的网格可以突出显示
  
  if (inforStore.sel_phase_details.phase_pred_val[cur_step][0].length == 0) return
  let glyphs_g = grid_borders_g.append('g')
    // .attr('transform', `translate(${margin_left+space_layout_w*0.08}, 4) scale(0.85)`)
  let cur_aq_vals = inforStore.month_aq_vals[cur_step]
  // 绘制空气质量glyph，可以雷达图为基础
  // xxxxxxxxxxx

  // 绘制legend
  
}

function drawTimeBar() {
  d3.select('#time-event-bar').selectAll('*').remove()
  let month_id = inforStore.month_id
  let space_layout_w = 1000, space_layout_h = 50, time_bar_h = 10
  let margin_left = 50, margin_right = 10
  let st_layout_w = space_layout_w + margin_left + margin_right
  let st_layout_h = space_layout_h + time_bar_h*3 - 20
  let st_layout_svg = d3.select('#time-event-bar')
    .attr('width', st_layout_w)
    .attr('height', st_layout_h)
  let loc_coords_x = inforStore.cur_data_infor.space.loc_list.map(item => item.geometry.coordinates[0])
  let loc_coords_y = inforStore.cur_data_infor.space.loc_list.map(item => item.geometry.coordinates[1])
  let month_len = 30
  let step_width = (space_layout_w-20) / phase_len
  
  let month_raw_data = inforStore.month_raw_data
  let phase_raw_level = inforStore.sel_phase_details.phase_raw_level.slice(cur_event_start, cur_event_end)
  let phase_raw_infor = []
  for (let i = 0; i < phase_raw_data.length; i++) {
    phase_raw_infor.push([])
    for (let j = 0; j < phase_raw_data[i].length; ++j) {
      phase_raw_infor[i].push({
        'val': phase_raw_data[i][j],
        'level': phase_raw_level[i][j]
      })
    }
  }

  let time_bar_g = st_layout_svg.append('g')
    .attr('transform', `translate(${margin_left+30}, ${space_layout_h-58})`)
  let time_axis_bar = time_bar_g.append('g')
  let temporal_ids = []
  for (let i = 0; i < phase_len; i++) {
    temporal_ids.push(i);
  }
  
  let temporal_rate_scale = d3.scaleQuantize()
    .domain([0, 1])
    .range(valColorScheme_fire)
  
  let month_infor = inforStore.st_phase_events.phases_list[month_id]
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
        // 如果是对于异常和极值类型，则异常和极值所在的时间步，显示为黑色，否则为白色
        // xxxxxxxxxxx
      })
      .attr('stroke', '#999')
      .on('click', (e) => {
        let target_id = d3.select(e.target).attr('id').split('-')[1]
        // console.log(inforStore.sel_phase_details.time_pod[target_id], inforStore.sel_phase_details.time_far[target_id]);
        phase_raw_infor = []
        for (let i = 0; i < phase_raw_data.length; i++) {
          phase_raw_infor.push([])
          for (let j = 0; j < phase_raw_data[i].length; ++j) {
            phase_raw_infor[i].push({
              'val': phase_raw_data[i][j],
              'level': phase_raw_level[i][j]
            })
          }
        }
        time_bar_g.select('#start-slider')
          .attr('transform', `translate(${cur_event_start * step_width}, ${time_bar_h*4+3})`)
        
          // 更新所查看的时间步
        cur_sel_step.value = target_id
      })

  
  let cntLenMap = d3.scaleLinear()
    .domain([0, 1])
    .range([time_bar_h*2, 0])
  
  let valColor = d3.scaleLinear()
    // .domain([inforStore.dataset_configs.focus_th, 500])
    .domain([0, 500])
    .range(['#fff', '#4a1486'])
  
  time_axis_bar.append('text')
    .attr('x', -4)
    .attr('y', 24)
    .attr('text-anchor', 'end')
    .style('font-size', '11px')
    .attr('fill', '#333')
    .text('Pollution')
  time_axis_bar.append('g').selectAll('rect')
    .data(temporal_ids)
    .join('rect')
      .attr('id', d => `cnt_unit-${d}`)
      .attr('x', d => d * step_width)
      // .attr('y', d => time_bar_h + cntLenMap(phase_infor.time_pollution_cnt_all[d]) + 1)
      .attr('y', d => time_bar_h + cntLenMap(phase_infor.time_pollution_cnt_all[d])/2 + 1)
      .attr('width', step_width)
      .attr('height', d => time_bar_h*2-cntLenMap(phase_infor.time_pollution_cnt_all[d]))
      .attr('fill', d => {
        return valColor(phase_infor.time_pollution_event_val[d])
      })
      .attr('stroke', '#999')
      .on('click', (e) => {
        let target_id = d3.select(e.target).attr('id').split('-')[1]
        // console.log(inforStore.sel_phase_details.time_pod[target_id], inforStore.sel_phase_details.time_far[target_id]);
        
        phase_raw_infor = []
        for (let i = 0; i < phase_raw_data.length; i++) {
          phase_raw_infor.push([])
          for (let j = 0; j < phase_raw_data[i].length; ++j) {
            phase_raw_infor[i].push({
              'val': phase_raw_data[i][j],
              'level': phase_raw_level[i][j]
            })
          }
        }
        time_bar_g.select('#start-slider')
          .attr('transform', `translate(${cur_event_start * step_width}, ${time_bar_h*4+3})`)
        cur_sel_step.value = target_id
      })
  
  let slider_h = 8
  time_bar_g.append('rect')
    .attr('id', 'start-slider')
    .attr('x', cur_sel_step.value*step_width)
    .attr('y', time_bar_h*3)
    .attr('width', step_width)
    .attr('height', slider_h)
    .attr('fill', '#333')
    .attr('stroke', 'none')
  
  // 定义拖拽
  let acc_dx = 0
  const slider_drag = d3.drag().on("start", function(event) {}).on("drag", function(event) {
    let curSliderX = cur_event_start * step_width
    acc_dx += event.dx
    if (Math.abs(acc_dx) > step_width) {
      phase_raw_infor = []
      for (let i = 0; i < phase_raw_data.length; i++) {
        phase_raw_infor.push([])
        for (let j = 0; j < phase_raw_data[i].length; ++j) {
          phase_raw_infor[i].push({
            'val': phase_raw_data[i][j],
            'level': phase_raw_level[i][j]
          })
        }
      }
      time_bar_g.select('#start-slider')
        .attr('transform', `translate(${cur_event_start * step_width}, ${time_bar_h*4+3})`)
      acc_dx = 0
      cur_sel_step.value = parseInt(cur_event_start) + parseInt(cur_sel_event_step)
    }
  }).on("end", function(event) {
    
  })

  time_bar_g.select('#start-slider').call(slider_drag)

  let resi_legend_len = 120
  
  let pollutionValScale = d3.scaleLinear()
      .domain([0, resi_legend_len])
      .range(['#fff', '#4a1486'])
  let pollution_val_legend = st_layout_svg.append('g')
    .attr('transform', `translate(230, ${space_layout_h-5})`)
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
    .text('Pollution Value')
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
}
</script>

<template>
  <div class="models-container">
    <div ref="reference" class="title-layer">
      Exploration View
      <div class="right-form-region">
        <div class="right-normal-forms">        
        </div>
      </div>
    </div>
    <div class="exploration-block">
      <div class="st-layout-container">
        <svg id="st-layout"></svg>
      </div>
      <svg id="time-event-bar"></svg>
      <div v-if="inforStore.cur_phase_sorted_id != -1" class="cur_stamp-row-right">
        <div class="title">Timestamp: </div>
        <div class="cur_stamp">{{ cur_phase_time_str_right }}</div>
      </div>
    </div>
  </div>

</template>

<style scoped>
.models-container {
  width: 1610px;
  /* width: 860px; */
  height: 836px;
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
  margin-bottom: 10px;
  /* font: 700 16px "Microsort Yahei"; */
  font: 700 20px "Arial";
  /* letter-spacing: 1px; */
  color: #333;
  display: flex;
  align-items: center;
  justify-content: flex-start;
}

.exploration-block {
  position: relative;
  /* width: 850px;
  height: 780px; */
}

.st-layout-container {
  display: flex;
  justify-content: space-around;
  /* align-items: center; */
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
  margin-top: 30px;
  height: 532px;
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

.cur_stamp-row-left {
  position: absolute; /* 或者使用 fixed，根据需求选择 */
  top: 5.2%; /* 垂直居中 */
  left: 1.2%; /* 水平居中 */
  display: flex;
  background-color: #fff;
  font-size: 14px;
}

.cur_stamp-row-right {
  position: absolute; /* 或者使用 fixed，根据需求选择 */
  top: 5.2%; /* 垂直居中 */
  left: 51.5%; /* 水平居中 */
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



.select-config-row {
  display: flex;
  justify-content: space-around;
  align-items: center;
}


#time-event-bar {
  margin-top: -164px;
}



.form-row {
  display: flex
}

</style>