<script setup>
import { ref, reactive, computed, watch, onMounted, onUpdated } from "vue"
import { useInforStore } from '@/stores/infor.js'
import * as d3 from 'd3'
import { valColorScheme_blue, valColorScheme_red, valColorScheme_fire, valColorScheme_double} from '@/data/index.js'

const props = defineProps({
  phase_id: Number
})

const inforStore = useInforStore()

const view_id = (view_str, model_name, phase_id) => `${view_str}_${model_name}_${phase_id}`

// watch (() => inforStore.phases_indicators, (oldValue, newValue) => {
//   // inforStore.cur_phase_indicators = inforStore.phases_indicators[inforStore.cur_sel_model]
//   console.log(inforStore.phases_indicators);
//   if (Object.keys(inforStore.phases_indicators).length == 1) {
//     inforStore.cur_phase_indicators = inforStore.phases_indicators[inforStore.cur_focused_model]
//     drawPhasesErrors()
//   }
// })

onMounted(() => {
  // console.log('mount ~~~~~~~');
  // drawPhasesErrors()
  if (Object.keys(inforStore.phases_indicators).length > 0) {
    // inforStore.cur_phase_indicators = inforStore.phases_indicators[inforStore.cur_focused_model]
    drawPhasesErrors()
  }
})

onUpdated(() => {
  // console.log('update ~~~~~~~');
  if (Object.keys(inforStore.phases_indicators).length > 0) {
    // inforStore.cur_phase_indicators = inforStore.phases_indicators[inforStore.cur_focused_model]
    drawPhasesErrors()
  }
})

let svg_h = 84
let svg_w = 150
let time_h = 14
let dist_h = 78
let err_h = 78
let dist_h_pure = 50

const drawPhasesErrors = () => {
  let phase_errors = inforStore.phases_indicators[inforStore.cur_focused_model][props.phase_id]
  let phase_baseline_errors = 'null'
  if (inforStore.phases_indicators.hasOwnProperty(inforStore.cur_baseline_model)) {
    phase_baseline_errors = inforStore.phases_indicators[inforStore.cur_baseline_model][props.phase_id]
  }
  let cur_svg_id = '#' + view_id('phase-pred', 'a', props.phase_id)
  d3.select(cur_svg_id).selectAll('*').remove()
  let cur_svg = d3.select(cur_svg_id)
    .attr('width', svg_w)
    .attr('height', svg_h)
  let phase_err_g = cur_svg.append('g')
    .attr('transform', `translate(0, 5)`)
  phase_err_g.append('rect')
    .attr('x', 0).attr('y', 0)
    .attr('width', svg_w)
    .attr('height', err_h)
    .attr('fill', 'none')
    .attr('stroke', '#333')
    .attr('stroke-width', 1)
  
  // 绘制pod饼图
  let pod_g = phase_err_g.append('g')
    .attr('transform', `translate(${svg_w*0.13}, ${err_h/4})`)
  let pie = d3.pie();
  pie.sort(null);

  // 定义斜条纹图案
  cur_svg.append("defs")
    .append("pattern")
    .attr("id", "stripes")
    .attr("patternUnits", "userSpaceOnUse")
    .attr("width", 5)
    .attr("height", 5)
    .append("line")
    .attr("x1", 0)
    .attr("y1", 5)
    .attr("x2", 5)
    .attr("y2", 0)
    .attr("stroke", "#3182bd")
    .attr("stroke-width", 1.5)

  let podArcData, podColorList, farArcData, farColorList
  if (phase_baseline_errors == 'null') {
    podArcData = pie([phase_errors.POD, 0, (1-phase_errors.POD)])
    podColorList = [valColorScheme_fire[4], 'url(#stripes)', '#cecece']
  } else if (phase_baseline_errors.POD >= phase_errors.POD) {
    podArcData = pie([phase_errors.POD, phase_baseline_errors.POD-phase_errors.POD, (1-phase_baseline_errors.POD)])
    podColorList = [valColorScheme_fire[4], 'url(#stripes)', '#cecece']
  } else if (phase_baseline_errors.POD < phase_errors.POD) {
    podArcData = pie([phase_baseline_errors.POD, phase_errors.POD-phase_baseline_errors.POD, (1-phase_errors.POD)])
    podColorList = ['url(#stripes)', valColorScheme_fire[4], '#cecece']
  }
  if (phase_baseline_errors == 'null') {
    farArcData = pie([phase_errors.FAR, 0, (1-phase_errors.FAR)])
    farColorList = [valColorScheme_fire[4], 'url(#stripes)', '#cecece']
  } else if (phase_baseline_errors.FAR >= phase_errors.FAR) {
    farArcData = pie([phase_errors.FAR, phase_baseline_errors.FAR-phase_errors.FAR, (1-phase_baseline_errors.FAR)])
    farColorList = [valColorScheme_fire[4], 'url(#stripes)', '#cecece']
  } else if (phase_baseline_errors.FAR < phase_errors.FAR) {
    farArcData = pie([phase_baseline_errors.FAR, phase_errors.FAR-phase_baseline_errors.FAR, (1-phase_errors.FAR)])
    farColorList = ['url(#stripes)', valColorScheme_fire[4], '#cecece']
  }
  
  let arc = d3.arc()
    .innerRadius(0)
    .outerRadius(svg_w*0.08);
  let podArcs = pod_g.selectAll("g")
      .data(podArcData)
      .join("g")
  podArcs.append("path")
      .attr("d", arc)
      .attr("fill", (d, i) => podColorList[i]);
    
  let far_g = phase_err_g.append('g')
    .attr('transform', `translate(${svg_w*0.37}, ${err_h*0.26})`)
  let farArcs = far_g.selectAll("g")
      .data(farArcData)
      .join("g")
  farArcs.append("path")
      .attr("d", arc)
      .attr("fill", (d, i) => farColorList[i]);
  
  // 绘制multi accuracy的柱状图
  let multi_accuracy_g = phase_err_g.append('g')
    .attr('transform', `translate(${svg_w*0.06}, ${err_h*0.53})`)
  let binScale = d3.scaleLinear()
    .domain([0, inforStore.dataset_configs.focus_levels.length-1])
    .range([0, svg_w*0.38])
  let yScale = d3.scaleLinear()
    .domain([0, 1])
    .range([0, err_h * 0.3])
  let binAxis = d3.axisBottom(binScale)
    .ticks(inforStore.dataset_configs.focus_levels.length)
    .tickSize(3)
    .tickFormat((d) => inforStore.dataset_configs.focus_levels[d])

  // 绘制bar的背景
  // multi_accuracy_g.append('g').selectAll('rect')
  //   .data(phase_errors.multi_accuracy_list)
  //   .join('rect')
  //     .attr('x', (d,i) => binScale(i))
  //     .attr('y', 0)
  //     .attr('width', binScale(1)-binScale(0))
  //     .attr('height', (d,i) => yScale(1))
  //     .attr('fill', '#cecece')
  // // 绘制multi accuracy的bar
  // multi_accuracy_g.append('g').selectAll('rect')
  //   .data(phase_errors.multi_accuracy_list)
  //   .join('rect')
  //     .attr('x', (d,i) => binScale(i))
  //     .attr('y', (d,i) => yScale(1-d))
  //     .attr('width', binScale(1)-binScale(0))
  //     .attr('height', (d,i) => yScale(d))
  //     .attr('fill', (d,i) => pieColorList[0])
  // multi_accuracy_g.append("g")
  //   .attr("class", "multi-accuracy-axis")
  //   .attr("transform", `translate(0, ${err_h*0.3})`) // 将X轴移至底部
  //   .call(binAxis);
  
  // 中间分割线
  cur_svg.append('line')
    .attr('x1', svg_w/2)
    .attr('y1', 8)
    .attr('x2', svg_w/2)
    .attr('y2', svg_w/2-6)
    .attr('stroke', '#cecece')
    .attr('stroke-width', 1)
    
  // 误差绝对值空间分布
  let loc_g = cur_svg.append('g')
    .attr('transform', `translate(${svg_w/2}, 2)`)
  let loc_coords_x = inforStore.cur_data_infor.space.loc_list.map(item => item.geometry.coordinates[0])
  let loc_coords_y = inforStore.cur_data_infor.space.loc_list.map(item => item.geometry.coordinates[1])
  let white_rate = 0.1
  let loc_x_scale = d3.scaleLinear()
        .domain([Math.min(...loc_coords_x), Math.max(...loc_coords_x)])
        .range([svg_w/2*(white_rate), svg_w/2*(1-white_rate)])
  let loc_y_scale = d3.scaleLinear()
        .domain([Math.min(...loc_coords_y), Math.max(...loc_coords_y)])
        .range([svg_w/2*(1-white_rate), svg_w/2*white_rate])
  let colormap = d3.scaleQuantize()
    .domain(phase_errors.global_space_mae_range)
    .range(valColorScheme_fire)
  
  loc_g.selectAll('circle')
    .data(phase_errors.space_abs_residual)
    .join('circle')
      .attr('cx', (d,i) => loc_x_scale(loc_coords_x[i]))
      .attr('cy', (d,i) => loc_y_scale(loc_coords_y[i]))
      .attr('r', 1.8)
      .attr('fill', (d,i) => {
        if (d == 0) return '#cecece'
        else return colormap(d)
      })
      .attr('stroke', 'none')
  d3.selectAll('.multi-accuracy-axis > .tick > text').remove()
}

</script>

<template>
  <div>
    <svg :id="view_id('phase-pred', 'a', phase_id)"></svg>
  </div>
  
</template>

<style scoped>

</style>