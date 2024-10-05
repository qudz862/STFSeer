<script setup>
import { ref, reactive, computed, watch, onMounted, onUpdated } from "vue"
import { useInforStore } from '@/stores/infor.js'
import getData from '@/services/index.js'
import * as d3 from 'd3'
import { valColorScheme_blue, valColorScheme_red, valColorScheme_fire, valColorScheme_double} from '@/data/index.js'

const props = defineProps({
  phase_id: Number,
  cur_phase_index: Number,
  phase_data: Object,
})

const inforStore = useInforStore()

const view_id = (view_str, model_name) => `${view_str}_${model_name}`

let normalize_metrics = ref([])
let value_metrics = ref([])
let level_metrics = ref([])

// function processMetricType() {
//   let metrics = inforStore.dataset_configs.phase_params.error_metrics
//   for (let metric of metrics) {
//     if (value_metrics)
//   }
// }

onMounted(() => {

  // drawPhases()
  drawPhaseTimeInfor()
  drawPhaseSpaceInfor()
  drawBinaryMetrics('phase_POD')
  drawBinaryMetrics('phase_FAR')
  drawLevelMetrics('focus_level_accuracy')

  // drawPhaseResHist('normal')
  // drawPhaseResHist('extreme_pos')
  // drawPhaseResHist('extreme_neg')
})

onUpdated(() => {
  drawPhaseTimeInfor()
  drawPhaseSpaceInfor()
  drawBinaryMetrics('phase_POD')
  drawBinaryMetrics('phase_FAR')
  drawLevelMetrics('focus_level_accuracy')

  // drawPhaseResHist('normal')
  // drawPhaseResHist('extreme_pos')
  // drawPhaseResHist('extreme_neg')
  // drawPhases()
})


const drawPhases = () => {
  let svg_h = 92, svg_w = 150, time_h = 14, dist_h = 78, dist_h_pure = 50
  let colorScale = d3.scaleLinear()
    .domain([0, inforStore.cur_data_infor.space.loc_list.length])
    .range(["#e7fbff","#08306b"])
  // let colorScale = d3.scaleQuantize()
  //   .domain([0, inforStore.cur_data_infor.space.loc_list.length])
  //   .range(valColorScheme_blue)
  let cntScale = d3.scaleLinear()
      .domain([0, inforStore.cur_data_infor.space.loc_list.length])
      .range([0, dist_h_pure])
  let binScale = d3.scaleLinear()
    .domain([0, inforStore.dataset_configs.focus_levels.length-1])
    .range([20, svg_w/2-20])
  let binAxis = d3.axisBottom(binScale)
    .ticks(inforStore.dataset_configs.focus_levels.length)
    .tickSize(3)
    .tickFormat((d) => inforStore.dataset_configs.focus_levels[d])
  let timeScale = d3.scaleLinear()
    .domain([0, inforStore.st_phase_events.time_strs.length])
    .range([0, svg_w])
  
  let phase_infor = props.phase_data
  // console.log(phase_infor);
  let cur_svg_id = '#' + view_id('phase', props.phase_id)
  d3.select(cur_svg_id).selectAll('*').remove()
  let cur_svg = d3.select(cur_svg_id)
    .attr('width', svg_w).attr('height', svg_h)

  cur_svg.append('rect')
    .attr('x', 0).attr('y', time_h)
    .attr('width', svg_w)
    .attr('height', dist_h)
    .attr('fill', 'none')
    .attr('stroke', '#333')
    .attr('stroke-width', 1)

  // 绘制target_val直方图
  let hist_block = cur_svg.append('g')
    .attr('class', 'phase-hist-block')
    .attr('transform', `translate(0, ${time_h+4})`)

  hist_block.append('g').selectAll('rect')
    .data(phase_infor.focus_level_hist)
    .join('rect')
      .attr('x', (d,i) => binScale(i))
      .attr('y', (d,i) => dist_h_pure - cntScale(d))
      .attr('width', binScale(1)-binScale(0))
      .attr('height', (d,i) => cntScale(d))
      .attr('fill', (d,i) => {
        if (inforStore.dataset_configs.focus_levels[i] < inforStore.dataset_configs.focus_th)
          return '#cecece'
        else return valColorScheme_blue[3]
      })
  let hist_x_axis = hist_block.append("g")
    .attr("class", "hist-axis")
    .attr("transform", `translate(0, ${dist_h_pure})`) // 将X轴移至底部
    .call(binAxis);
  // 展示空间污染
  let loc_g = cur_svg.append('g')
    .attr('transform', `translate(${svg_w/2}, ${time_h+2})`)
  let loc_coords_x = inforStore.cur_data_infor.space.loc_list.map(item => item.geometry.coordinates[0])
  let loc_coords_y = inforStore.cur_data_infor.space.loc_list.map(item => item.geometry.coordinates[1])
  let white_rate = 0.1
  let loc_x_scale = d3.scaleLinear()
        .domain([Math.min(...loc_coords_x), Math.max(...loc_coords_x)])
        .range([svg_w/2*(white_rate), svg_w/2*(1-white_rate)])
  let loc_y_scale = d3.scaleLinear()
        .domain([Math.min(...loc_coords_y), Math.max(...loc_coords_y)])
        .range([svg_w/2*(1-white_rate), svg_w/2*white_rate])
  let colormap = d3.scaleLinear()
    .domain([0, 1])
    .range([valColorScheme_blue[0], valColorScheme_blue[valColorScheme_blue.length -1]])
  let opacityScale = d3.scaleLinear()
    .domain([0, 1])
    .range([0.2, 1])
  // loc_g.append('rect')
  //   .attr('x', 0)
  //   .attr('y', 0)
  //   .attr('width', svg_w/2)
  //   .attr('height', svg_w/2)
  //   .attr('fill', 'none')
  //   .attr('stroke', '#cecece')
  //   .attr('stroke-width', 1)
  cur_svg.append('line')
    .attr('x1', svg_w/2)
    .attr('y1', time_h + 8)
    .attr('x2', svg_w/2)
    .attr('y2', time_h + svg_w/2-6)
    .attr('stroke', '#cecece')
    .attr('stroke-width', 1)
  loc_g.selectAll('circle')
    .data(phase_infor.space_focus_cnt)
    .join('circle')
      .attr('cx', (d,i) => loc_x_scale(loc_coords_x[i]))
      .attr('cy', (d,i) => loc_y_scale(loc_coords_y[i]))
      .attr('r', 1.8)
      .attr('fill', (d,i) => {
        if (d == 0) return '#cecece'
        else return colormap(d)
      })
      .attr('stroke', 'none')
      // .attr('opacity', (d,i) => {
      //   if (d == 0) return 0.5
      //   else return opacityScale(d)
      // })

  // 绘制时序污染情况
  let time_g = cur_svg.append('g')
    .attr('transform', `translate(0, ${time_h+2})`)
  let time_segs = new Array(phase_infor.time_focus_cnt.length).fill(1)
  const pie = d3.pie().value(d => d)
  const pieData = pie(time_segs)
  const range_index_scale = d3.scaleLinear()
    .domain([0, 1])
    .range([valColorScheme_blue[0], valColorScheme_blue[valColorScheme_blue.length-1]])
  const arc = d3.arc()
    .innerRadius(svg_w*0.20*(1-white_rate))
    .outerRadius(svg_w*0.25*(1-white_rate))
  const paths = time_g.selectAll("path")
    .data(pieData)
    .enter()
    .append("path")
      .attr('transform', `translate(${svg_w * 0.25}, ${0.25*svg_w})`)
      .attr("d", arc)
      .attr("fill", (d, i) => range_index_scale(phase_infor.time_focus_cnt[i]))
      // .attr("stroke", '#cecece')
      // .attr("stroke-width", 0.5)
  d3.selectAll('.hist-axis > .tick > text').remove()
}

function timeFocusCntCompact(time_focus_cnt) {
  if (time_focus_cnt.length > 90) {
    // 计算 down_rate 并对数组进行下采样
    const downRate = Math.ceil(time_focus_cnt.length / 90);
    const timeFocusCnt = [];
    for (let i = 0; i < time_focus_cnt.length; i += downRate) {
      const group = time_focus_cnt.slice(i, i + downRate);  // 取出每 n 个数值
      const sum = group.reduce((acc, val) => acc + val, 0);  // 计算这一组的总和
      timeFocusCnt.push(sum / group.length);  // 计算均值并存入结果数组
    }
    return timeFocusCnt
  } 
  return time_focus_cnt;
}

function drawPhaseTimeInfor() {
  let main_h = 80, main_w = 80, margin_left = 2, margin_right = 2, margin_bottom = 2, margin_top = 2
  let dist_h_pure = main_h / 2
  let svg_h = main_h + margin_bottom + margin_top
  let svg_w = main_w + margin_left + margin_right
  let phase_infor = props.phase_data
  
  let cur_svg_id = '#' + view_id('phase-time', props.phase_id)
  d3.select(cur_svg_id).selectAll('*').remove()
  let cur_svg = d3.select(cur_svg_id)
    .attr('width', svg_w).attr('height', svg_h)

  // 绘制level hist
  let cntScale = d3.scaleLinear()
      .domain([0, inforStore.cur_data_infor.space.loc_list.length])
      .range([0, dist_h_pure])
  let binScale = d3.scaleLinear()
    .domain([0, inforStore.dataset_configs.focus_levels.length])
    .range([main_w*0.3, main_w*0.7])
  let binAxis = d3.axisBottom(binScale)
    .ticks(inforStore.dataset_configs.focus_levels.length)
    .tickSize(3)
    .tickFormat((d) => inforStore.dataset_configs.focus_levels[d])
  let hist_block = cur_svg.append('g')
    .attr('class', 'phase-hist-block')
    .attr('transform', `translate(${margin_left}, ${margin_top+main_h*0.15})`)
  hist_block.append('g').selectAll('rect')
    .data(phase_infor.focus_level_hist)
    .join('rect')
      .attr('x', (d,i) => binScale(i))
      .attr('y', (d,i) => dist_h_pure - cntScale(d))
      .attr('width', binScale(1)-binScale(0))
      .attr('height', (d,i) => cntScale(d))
      .attr('fill', (d,i) => {
        if (inforStore.dataset_configs.focus_levels[i] < inforStore.dataset_configs.focus_th)
          return '#cecece'
        else return '#0097A7'
      })
  let hist_x_axis = hist_block.append("g")
    .attr("class", "hist-axis")
    .attr("transform", `translate(0, ${dist_h_pure})`) // 将X轴移至底部
    .call(binAxis);

  let time_g = cur_svg.append('g')
    .attr('transform', `translate(${margin_left}, ${margin_top})`)
  let compact_time_focus_cnt = timeFocusCntCompact(phase_infor.time_focus_cnt)
  let time_segs = new Array(compact_time_focus_cnt.length).fill(1)
  const pie = d3.pie().value(d => d)
  const pieData = pie(time_segs)
  // const range_index_scale = d3.scaleLinear()
  //   .domain([0, 1])
    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    //原来：
    // .range([valColorScheme_blue[0], valColorScheme_blue[valColorScheme_blue.length-1]])
    // .range(['#fff', '#4a1486'])
  let range_index_scale = d3.scaleQuantize()
    .domain([0, 1])
    .range(['#edf8e9', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#005a32'])


  const arc = d3.arc()
    .innerRadius(main_w*0.32)
    .outerRadius(main_w*0.41)
  const paths = time_g.selectAll("path")
    .data(pieData)
    .enter()
    .append("path")
      .attr('transform', `translate(${main_w * 0.5}, ${0.5*main_w})`)
      .attr("d", arc)
      .attr("fill", (d, i) => range_index_scale(phase_infor.time_focus_cnt[i]))
      // .attr("stroke", '#cecece')
      // .attr("stroke-width", 0.5)
  d3.selectAll('.hist-axis > .tick > text').remove()
}

function drawPhaseSpaceInfor() {
  let main_h = 80, main_w = 80, margin_left = 2, margin_right = 2, margin_bottom = 2, margin_top = 2
  let svg_h = main_h + margin_bottom + margin_top
  let svg_w = main_w + margin_left + margin_right
  let phase_infor = props.phase_data
  
  let cur_svg_id = '#' + view_id('phase-space', props.phase_id)
  d3.select(cur_svg_id).selectAll('*').remove()
  let cur_svg = d3.select(cur_svg_id)
    .attr('width', svg_w).attr('height', svg_h)
  let loc_g = cur_svg.append('g')
    .attr('transform', `translate(${margin_left}, ${margin_top})`)
  // loc_g.append('rect')
  //   .attr('x', 0).attr('y', 0)
  //   .attr('width', main_w)
  //   .attr('height', main_h)
  //   .attr('fill', 'none')
  //   .attr('stroke', '#999')
  //   .attr('stroke-width', 1)
  let loc_coords_x = inforStore.cur_data_infor.space.loc_list.map(item => item.geometry.coordinates[0])
  let loc_coords_y = inforStore.cur_data_infor.space.loc_list.map(item => item.geometry.coordinates[1])
  let white_rate = 0.1
  let loc_x_scale = d3.scaleLinear()
        .domain([Math.min(...loc_coords_x), Math.max(...loc_coords_x)])
        .range([main_w*white_rate, main_w*(1-white_rate)])
  let loc_y_scale = d3.scaleLinear()
        .domain([Math.min(...loc_coords_y), Math.max(...loc_coords_y)])
        .range([main_h*(1-white_rate), main_h*white_rate])
  
  
  // let colormap = d3.scaleLinear()
  //   .domain([0, 1])
    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    //原来：
   // .range([valColorScheme_blue[0], valColorScheme_blue[valColorScheme_blue.length -1]])
    // .range(['#fff', '#4a1486'])
  let colormap = d3.scaleQuantize()
    .domain([0, 1])
    .range(['#edf8e9', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#005a32'])




  loc_g.append('g').selectAll('circle')
    .data(phase_infor.space_focus_cnt)
    .join('circle')
      .attr('cx', (d,i) => loc_x_scale(loc_coords_x[i]))
      .attr('cy', (d,i) => loc_y_scale(loc_coords_y[i]))
      .attr('r', 1.8)
      .attr('fill', (d,i) => {
        if (d == 0) return '#cecece'
        else return colormap(d)
      })
      .attr('stroke', 'none')
}

function drawBinaryMetrics(metric) {
  let metric_val, metric_val_comp
  if (inforStore.cur_baseline_model.length > 0) {
    metric_val = inforStore.st_phase_events.phases[inforStore.cur_baseline_model][props.phase_id][metric]
    metric_val_comp = inforStore.st_phase_events.phases[inforStore.cur_focused_model][props.phase_id][metric]
  } else {
    metric_val = inforStore.st_phase_events.phases[inforStore.cur_focused_model][props.phase_id][metric]
  }

  let main_h = 30, main_w = 30, margin_left = 1, margin_right = 1, margin_bottom = 1, margin_top = 1
  let svg_h = main_h + margin_bottom + margin_top
  let svg_w = main_w + margin_left + margin_right
  
  let cur_svg_id = '#' + view_id(`phase-${metric}`, props.phase_id)
  d3.select(cur_svg_id).selectAll('*').remove()
  let cur_svg = d3.select(cur_svg_id)
    .attr('width', svg_w).attr('height', svg_h)
  let metric_g = cur_svg.append('g')
    .attr('transform', `translate(${margin_left+main_w/2}, ${margin_top+main_h/2})`)
  let pie = d3.pie();
  pie.sort(null);

  let podArcData = pie([metric_val, (1-metric_val)])
  let podColorList = ['#0097A7' , '#cecece']
  if (inforStore.cur_baseline_model.length > 0) {
    podColorList = ['#999' , '#cecece']
  }
  let arc = d3.arc()
    .innerRadius(0)
    .outerRadius(main_w*0.45);
  let metricArcs = metric_g.selectAll("g")
      .data(podArcData)
      .join("g")
  metricArcs.append("path")
      .attr("d", arc)
      .attr("fill", (d, i) => podColorList[i]);
  if (inforStore.cur_baseline_model.length > 0) {
    let metric_g_comp = cur_svg.append('g')
      .attr('transform', `translate(${margin_left+main_w/2}, ${margin_top+main_h/2})`)
    let podArcData_comp = pie([metric_val_comp, (1-metric_val_comp)])
    let podColorList_comp = ['#0097A7' , 'none']
    let arc_comp = d3.arc()
      .innerRadius(0)
      .outerRadius(main_w*0.3);
    let metricArcs_comp = metric_g_comp.selectAll("g")
        .data(podArcData_comp)
        .join("g")
    metricArcs_comp.append("path")
      .attr("d", arc_comp)
      .attr("fill", (d, i) => podColorList_comp[i]);
  }
}

function drawLevelMetrics(metric) {
  let metric_val, metric_val_comp
  if (inforStore.cur_baseline_model.length > 0) {
    metric_val = inforStore.st_phase_events.phases[inforStore.cur_baseline_model][props.phase_id][metric]
    metric_val_comp = inforStore.st_phase_events.phases[inforStore.cur_focused_model][props.phase_id][metric]
  } else {
    metric_val = inforStore.st_phase_events.phases[inforStore.cur_focused_model][props.phase_id][metric]
  }

  let main_h = 30, main_w = 90, margin_left = 1, margin_right = 1, margin_bottom = 3, margin_top = 0
  let svg_h = main_h + margin_bottom + margin_top
  let svg_w = main_w + margin_left + margin_right
  let cur_svg_id = '#' + view_id(`phase-${metric}`, props.phase_id)
  d3.select(cur_svg_id).selectAll('*').remove()
  let cur_svg = d3.select(cur_svg_id)
    .attr('width', svg_w).attr('height', svg_h)

  let multi_accuracy_g = cur_svg.append('g')
    .attr('transform', `translate(${margin_left}, ${margin_top})`)
  let binScale = d3.scaleLinear()
    .domain([0, inforStore.dataset_configs.focus_levels.length-1])
    .range([0, main_w])
  let yScale = d3.scaleLinear()
    .domain([0, 1])
    .range([0, main_h])
  let binAxis = d3.axisBottom(binScale)
    .ticks(inforStore.dataset_configs.focus_levels.length)
    .tickSize(3)
    .tickFormat((d) => inforStore.dataset_configs.focus_levels[d])
  let bin_width = binScale(1)-binScale(0)
  multi_accuracy_g.append('g').selectAll('rect')
    .data(metric_val)
    .join('rect')
      .attr('x', (d,i) => binScale(i))
      .attr('y', 0)
      .attr('width', bin_width-0.5)
      .attr('height', (d,i) => yScale(1))
      .attr('fill', '#cecece')
    
  // 绘制multi accuracy的bar
  multi_accuracy_g.append('g').selectAll('rect')
    .data(metric_val)
    .join('rect')
      .attr('x', (d,i) => binScale(i))
      .attr('y', (d,i) => yScale(1-d))
      .attr('width', bin_width-0.5)
      .attr('height', (d,i) => yScale(d))
      .attr('fill', (d,i) => {
        if (inforStore.cur_baseline_model.length > 0) return '#999'
        else return '#0097A7'
      })
  multi_accuracy_g.append("g")
    .attr("class", "multi-accuracy-axis")
    .attr("transform", `translate(0, ${main_h})`) // 将X轴移至底部
    .call(binAxis);

  if (inforStore.cur_baseline_model.length > 0) {
    multi_accuracy_g.append('g').selectAll('rect')
    .data(metric_val_comp)
    .join('rect')
      .attr('x', (d,i) => binScale(i)+bin_width*0.25)
      .attr('y', (d,i) => yScale(1-d))
      .attr('width', bin_width*0.5)
      .attr('height', (d,i) => yScale(d))
      .attr('fill', (d,i) => '#0097A7')
      .attr('stroke', '#666')
  }

}

function drawPhaseResHist(err_type) {
  let hist_type, val_bins, error_hists, error_hists_comp
  if (err_type == 'normal') {
    val_bins = inforStore.error_distributions.all_residual_bins
    hist_type = 'residual_hist'
  }
  else if (err_type == 'extreme_pos') {
    val_bins = inforStore.error_distributions.all_pos_extreme_bins
    hist_type = 'pos_extreme_hist'
  }
  else if (err_type == 'extreme_neg') {
    val_bins = inforStore.error_distributions.all_neg_extreme_bins
    hist_type = 'neg_extreme_hist'
  }

  let main_h = 80, main_w = 100, margin_left = 2, margin_right = 2, margin_bottom = 2, margin_top = 2
  let svg_h = main_h + margin_bottom + margin_top
  let svg_w = main_w + margin_left + margin_right  
  
  if (inforStore.cur_baseline_model.length > 0) {
    error_hists = inforStore.st_phase_events.phases[inforStore.cur_baseline_model][props.phase_id].res_distribution[hist_type]
    error_hists_comp = inforStore.st_phase_events.phases[inforStore.cur_focused_model][props.phase_id].res_distribution[hist_type]
  } else {
    error_hists = inforStore.st_phase_events.phases[inforStore.cur_focused_model][props.phase_id].res_distribution[hist_type]
  }

  let hist_max = d3.max(error_hists)
  if (inforStore.cur_baseline_model.length > 0) {
    let all_hists = [...error_hists, ...error_hists_comp]
    hist_max = d3.max(all_hists)
  }

  let cur_svg_id = '#' + view_id(`phase-${err_type}`, props.phase_id)
  d3.select(cur_svg_id).selectAll('*').remove()
  let svg = d3.select(cur_svg_id)
    .attr('width', svg_w).attr('height', svg_h)
  let xScale = d3.scaleLinear()
    .domain([0, val_bins.length-1])
    .range([0, main_w])
  let yScale = d3.scaleLinear()
    .domain([0, hist_max])
    .range([0, main_h])
  let err_hist = svg.append('g')
    .attr('transform', `translate(${margin_left}, ${margin_top})`)
  let bin_width = xScale(1) - xScale(0)
  let width_rate = 1.0
  err_hist.selectAll('rect')
    .data(error_hists)
    .join('rect')
      .attr('x', (d,i) => xScale(i)+bin_width*0.05)
      .attr('y', (d,i) => main_h - yScale(d))
      .attr('width', width_rate * bin_width*0.9)
      .attr('height', (d,i) => yScale(d))
      .attr('fill', (d,i) => '#bcbcbc')
      .attr('stroke', 'none')
  
  if (inforStore.cur_baseline_model.length > 0) {
    let err_hist_comp = svg.append('g')
      .attr("transform", `translate(${margin_left+3},${margin_top})`);
    err_hist_comp.selectAll('rect')
      .data(error_hists_comp)
      .join('rect')
        .attr('x', (d,i) => xScale(i)+bin_width*0.05)
        .attr('y', (d,i) => main_h - yScale(d))
        .attr('width', width_rate * bin_width*0.9 - 6)
        .attr('height', (d,i) => yScale(d))
        .attr('fill', '#0097A7')
        .attr('stroke', '#666')
  }
  
  let zero_line = err_hist.append('line')
    .attr('x1', xScale(val_bins.indexOf(0)))
    .attr('x2', xScale(val_bins.indexOf(0)))
    .attr('y1', 0)
    .attr('y2', main_h)
    .attr('stroke', valColorScheme_red[1])
    .attr('stroke-dasharray', '4,4')
}

const onPhaseSelect = (phase_id, cur_phase_index) => {
  if (inforStore.cur_phase_id == cur_phase_index) {
    inforStore.cur_phase_id = -1
    inforStore.cur_phase_sorted_id = -1
  } else {
    inforStore.cur_phase_id = cur_phase_index
    inforStore.cur_phase_sorted_id = phase_id

    if (!inforStore.browsed_phases.includes(phase_id)) inforStore.browsed_phases.push(phase_id)

    getData(inforStore, 'phase_data', phase_id)
    
    getData(inforStore, 'phase_details', JSON.stringify(inforStore.cur_sel_models), phase_id, JSON.stringify(inforStore.dataset_configs.focus_levels), inforStore.dataset_configs.focus_th, JSON.stringify(inforStore.forecast_scopes), inforStore.cur_focused_scope)
  }
}

// const onPhaseSelect = (phase_id, cur_phase_index) => {
//   if (inforStore.cur_phase_id == cur_phase_index) {
//     inforStore.cur_phase_id = -1
//     inforStore.cur_phase_sorted_id = -1
//   } else {
//     inforStore.cur_phase_id = cur_phase_index
//     inforStore.cur_phase_sorted_id = phase_id

//     if (!inforStore.browsed_phases.includes(phase_id)) inforStore.browsed_phases.push(phase_id)

//     getData(inforStore, 'phase_data', phase_id)
    
//     getData(inforStore, 'phase_details', JSON.stringify(inforStore.cur_sel_models), phase_id, JSON.stringify(inforStore.dataset_configs.focus_levels), inforStore.dataset_configs.focus_th, JSON.stringify(inforStore.forecast_scopes), inforStore.cur_focused_scope)
//   }
// }

function CollectPhase(phase_id) {
  let index = inforStore.phase_collections.indexOf(phase_id);
  if (index > -1) {
    inforStore.phase_collections.splice(index, 1);
  } else {
    inforStore.phase_collections.push(phase_id)
  }
}

function processPhaseDateRange(startTime, endTime) {
    // 将时间字符串按年份、月份、日期、小时分割
    const startArr = [
    startTime.slice(0, 4),   // 年份
    startTime.slice(4, 6),   // 月份
    startTime.slice(6, 8),   // 日期
    startTime.slice(8, 10)   // 小时
  ];
  
  const endArr = [
    endTime.slice(0, 4),     // 年份
    endTime.slice(4, 6),     // 月份
    endTime.slice(6, 8),     // 日期
    endTime.slice(8, 10)     // 小时
  ];

  let result = "";

  // 检查年份是否相同
  if (startArr[0] === endArr[0]) {
    // 如果年份相同，再检查月份是否相同
    if (startArr[1] === endArr[1]) {
      // 年份和月份都相同，则保留年份和月份一次
      result = `${startArr[0]}-${startArr[1]} `;
    } else {
      // 年份相同但月份不同，只保留年份
      result = `${startArr[0]} `;
    }
  } 

  // 如果年份不同，什么都不保留，直接格式化为完整的两个时间段
  if (startArr[0] !== endArr[0]) {
    result = `${startArr[0]}${startArr[1]}${startArr[2]}${startArr[3]}~${endArr[0]}${endArr[1]}${endArr[2]}${endArr[3]}`;
  } else {
    // 保留相同的年份或年份和月份，并拼接日期和小时
    if (startArr[1] === endArr[1]) {
      result += `${startArr[2]}T${startArr[3]}~${endArr[2]}T${endArr[3]}`;
    } else {
      result += `${startArr[1]}-${startArr[2]}T${startArr[3]}~${endArr[1]}-${endArr[2]}T${endArr[3]}`;
    }
  }

  return result;
}

</script>

<template>
  <div class="phase-card">
    <div class="title-row"> 
        <span class="iconfont phase-action-icon" :browsed="inforStore.cur_phase_id!=cur_phase_index && inforStore.browsed_phases.includes(cur_phase_index)" :chosen="inforStore.cur_phase_id==cur_phase_index" @click="onPhaseSelect(phase_id, cur_phase_index)">&#xe624;</span>
        <div class="date-range-str">{{ processPhaseDateRange(phase_data['start_date'], phase_data['end_date']) }}</div>
        <span class="iconfont phase-action-icon" :chosen="inforStore.phase_collections.includes(phase_id)" @click="CollectPhase(phase_id)">&#xe61c;</span>
    </div>
    <div class="phase-time-infor">
      <Popper placement="right">
        <svg :id="view_id('phase-time', phase_id)"></svg>
        <template #content>
          <div class="attrs-block">
            <div><span class="attr-title">Life Span:</span> <span class="attr-val">{{ phase_data.life_span }}</span></div>
            <div><span class="attr-title">𝜇(Duration):</span> <span class="attr-val">{{ Math.round(phase_data.mean_duration) }}</span></div>
            <div><span class="attr-title">𝜇(Intensity):</span> <span class="attr-val">{{ Math.round(phase_data.mean_intensity) }}</span></div>
            <div><span class="attr-title">Max_Value:</span> <span class="attr-val">{{ Math.round(phase_data.max_value) }}</span></div>
          </div>
        </template>
      </Popper>
      <div class="vertical-seg"></div>
      <Popper placement="right">
        <svg :id="view_id('phase-space', phase_id)"></svg>
        <template #content>
          <div class="attrs-block">
            <div><span class="attr-title">𝜇(Focus_grids):</span> <span class="attr-val">{{ Math.round(phase_data.mean_focus_grids) }}</span></div>
            <div><span class="attr-title">Max_Grids:</span> <span class="attr-val">{{ Math.round(phase_data.max_step_focus_grids) }}</span></div>
            <div><span class="attr-title">𝜇(Move_dis):</span> <span class="attr-val">{{ Math.round(phase_data.mean_move_distance) }}</span></div>
          </div>
        </template>
      </Popper>
    </div>
    <!-- <div class="phase-space-infor">
      <svg :id="view_id('phase-space', phase_id)"></svg>
      <div class="vertical-seg"></div>
      <div class="attrs-block">
        <div><span class="attr-title">𝜇(Focus_grids):</span> <span class="attr-val">{{ Math.round(phase_data.mean_focus_grids) }}</span></div>
        <div><span class="attr-title">Max_Grids:</span> <span class="attr-val">{{ Math.round(phase_data.max_step_focus_grids) }}</span></div>
        <div><span class="attr-title">𝜇(Move_dis):</span> <span class="attr-val">{{ Math.round(phase_data.mean_move_distance) }}</span></div>
      </div>
    </div> -->
    <div class="horizontal-seg"></div>
    <div class="metrics-container">
      <!-- <svg :id="view_id('phase-normal', phase_id)"></svg>
      <svg :id="view_id('phase-extreme_pos', phase_id)"></svg>
      <svg :id="view_id('phase-extreme_neg', phase_id)"></svg> -->
      <div class="metric-blcok">
        <div class="attr-title">POD</div>
        <svg :id="view_id('phase-phase_POD', phase_id)"></svg>
      </div>
      <div class="metric-blcok">
        <div class="attr-title">FAR</div>
        <svg :id="view_id('phase-phase_FAR', phase_id)"></svg>
      </div>
      <div class="metric-blcok">
        <div class="attr-title">Level Accuracy</div>
        <svg :id="view_id('phase-focus_level_accuracy', phase_id)"></svg>
      </div>
    </div>
    
    <!-- <svg :id="view_id('phase', phase_id)"></svg> -->
  </div>
  
</template>

<style scoped>

.phase-card {
  border: solid 1px #bcbcbc;
  border-radius: 6px;
  margin-left: 2px;
  margin-top: 4px;
  margin-bottom: 1px;
  padding: 1px 3px;
  cursor: pointer;
  transition: box-shadow 0.3s ease; /* 添加过渡效果 */  
}

.phase-card:hover {
  box-shadow: 0 0 5px 2px rgba(170, 170, 170, 0.5); /* 鼠标滑过时的阴影 */
}

.title-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: solid 1px #999;
  margin-bottom: 3px;
  max-width: 220px;
}

.date-range-str {
  font-weight: 700;
  max-width: 160;
  text-overflow: ellipsis;
}

.phase-action-icon {
  font-size: 18px;
  color: #acacac;
  margin: 0 2px;
  padding-bottom: -10px;
  font-weight: 400 !important;
}
.phase-action-icon:hover {
  cursor: pointer;
  color: #0097A7;
}
.phase-action-icon[chosen=true] {
  color: #0097A7;
}

.phase-action-icon[browsed=true] {
  color: #6a51a3;
}

.phase-time-infor,
.phase-space-infor {
  display: flex;
  /* justify-content: space-between; */
}

.vertical-seg {
  width:1px;
  /* height: 60px; */
  background-color: #bcbcbc;
  margin-top: 6px;
  margin-bottom: 6px;
  margin-left: 8px;
  margin-right: 4px;
}

.horizontal-seg {
  height:1px;
  /* height: 60px; */
  background-color: #bcbcbc;
  margin-top: 2px;
  margin-bottom: 4px;
  margin-left: 8px;
  margin-right: 8px;
}

.attrs-block {
  display: flex;
  flex-direction: column;
  justify-content: center;
  /* display: grid; */
}

.attr-title {
  font-weight: 700;
}

.attr-val {
  color: #1a73e8;
}

.metrics-container {
  display: flex;
  justify-content: space-around;
  margin-bottom: 4px;
}

.metric-blcok {
  display: flex;
  flex-direction: column;
  align-items: center;
}
</style>