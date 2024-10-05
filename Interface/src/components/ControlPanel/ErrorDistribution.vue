<script setup>
import { ref, reactive, computed, watch, onMounted, onUpdated } from "vue"
import { useInforStore } from '@/stores/infor.js'
import * as d3 from 'd3'
import { valColorScheme_blue, valColorScheme_red, valColorScheme_fire, valColorScheme_double} from '@/data/index.js'

const inforStore = useInforStore()

const view_id = (view_str, label) => `${view_str}-${label}`

watch (() => inforStore.error_distributions, (oldValue, newValue) => {
  console.log('error_distributions', inforStore.error_distributions);


    // let mid_bins = inforStore.error_distributions.all_mid_bins
    // let pos_extreme_bins = inforStore.error_distributions.all_pos_extreme_bins
    // let neg_extreme_bins = inforStore.error_distributions.all_neg_extreme_bins
    // let hist_bins = inforStore.error_distributions.all_residual_bins
    // inforStore.err_abs_extreme_th = Math.max(Math.abs(mid_bins[0]), mid_bins[mid_bins.length-1])
    
    // inforStore.mild_err_color_scale = d3.scaleSequential(d3.interpolateBlues)
    //   .domain([0, inforStore.err_abs_extreme_th])
    // inforStore.extreme_err_color_scale = d3.scaleSequential(d3.interpolateYlOrRd)
    //   .domain([inforStore.err_abs_extreme_th, Math.max(Math.abs(neg_extreme_bins[0]), pos_extreme_bins[pos_extreme_bins.length-1])])

  let mid_bins = inforStore.error_distributions.all_mid_bins
  let pos_extreme_bins = inforStore.error_distributions.all_pos_extreme_bins
  let neg_extreme_bins = inforStore.error_distributions.all_neg_extreme_bins
  let hist_bins = inforStore.error_distributions.all_residual_bins
  inforStore.err_abs_extreme_th = Math.max(Math.abs(mid_bins[0]), mid_bins[mid_bins.length-1])
  
  let blueScale = d3.scaleSequential()
    .domain([0,1])
    .interpolator(d3.interpolateBlues)
  let colorScale = d3.scaleSequential()
    .domain([0,1])
    .interpolator(d3.interpolateYlOrRd)
  inforStore.mild_err_color_scale = d3.scaleSequential()
    .domain([0, inforStore.err_abs_extreme_th])
    .interpolator(d3.interpolate(
      blueScale(0.0),
      blueScale(0.7)
    ));
  inforStore.extreme_err_color_scale = d3.scaleSequential()
    .domain([inforStore.err_abs_extreme_th, Math.max(Math.abs(neg_extreme_bins[0]), pos_extreme_bins[pos_extreme_bins.length-1])])
    .interpolator(d3.interpolate(
        colorScale(0.4),
        colorScale(1.0)
    ));
  drawErrDisLegend()
  drawScopeErrBoxPlotEval()
  drawErrHistEval(inforStore.cur_focused_scope)
  console.log('err_abs_extreme_th', inforStore.err_abs_extreme_th);
})

watch (() => inforStore.cur_focused_scope, (oldValue, newValue) => {
  drawErrDisLegend()
  drawScopeErrBoxPlotEval()
  drawErrHistEval(inforStore.cur_focused_scope)
})

onMounted(() => {
})

onUpdated(() => {
  drawErrDisLegend()
  // drawScopeErrBoxPlotEval()
  // drawErrHistEval(inforStore.cur_focused_scope)
})

function drawScopeErrBoxPlotEval() {
  //$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
  //原方案  
  const bar_color =  '#0097A7';

  //方案1
  // onst bar_color =  '#CC79A7';

  //$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

  const disPlotConfig = {
    width: 276,
    height: 182,
    left_padding: 32,
    right_padding: 4,
    top_padding: 20,
    bottom_padding: 12
  }
  let err_dis_svg = d3.select(`#scope-err-boxes`)
    .attr('width', disPlotConfig.width)
    .attr('height', disPlotConfig.height)
  err_dis_svg.selectAll('*').remove()
  let scope_err_infor_list = inforStore.error_distributions.boxes_infor
  let val_bins = inforStore.error_distributions.all_residual_bins
  let mid_bins = inforStore.error_distributions.all_mid_bins
  
  let mild_pos_outliers_nums = []
  let mild_neg_outliers_nums = []
  let extreme_pos_outliers_nums = []
  let extreme_neg_outliers_nums = []
  for (let key in scope_err_infor_list) {
    mild_pos_outliers_nums = mild_pos_outliers_nums.concat(scope_err_infor_list[key].map(item => item.mild_pos_outliers_num))
    mild_neg_outliers_nums = mild_neg_outliers_nums.concat(scope_err_infor_list[key].map(item => item.mild_neg_outliers_num))
    extreme_pos_outliers_nums = extreme_pos_outliers_nums.concat(scope_err_infor_list[key].map(item => item.extreme_pos_outliers_num))
    extreme_neg_outliers_nums = extreme_neg_outliers_nums.concat(scope_err_infor_list[key].map(item => item.extreme_neg_outliers_num))
  }
  let mild_outliers_nums = [...mild_pos_outliers_nums, ...mild_neg_outliers_nums]
  let extreme_outliers_nums = [...extreme_pos_outliers_nums, ...extreme_neg_outliers_nums]
  let mild_outliers_num_range = d3.extent(mild_outliers_nums)
  let extreme_outliers_num_range = d3.extent(extreme_outliers_nums)
  let text_h = 18

  let xScale = d3.scaleLinear()
    .domain([0.4, inforStore.forecast_scopes.length+0.6])
    .range([disPlotConfig.left_padding, disPlotConfig.width-disPlotConfig.right_padding])
  
  let step_width = xScale(1.4) - xScale(0.4)
  if (Object.keys(scope_err_infor_list).length > 1) {
    step_width /= 2
  }

  err_dis_svg.append('text')
    .attr('x', 0)
    .attr('y', 12)
    .attr('text-anchor', 'start')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('Residuals')
  err_dis_svg.append('text')
    .attr('x', disPlotConfig.width/2+disPlotConfig.left_padding/2-2)
    .attr('y', disPlotConfig.height-2)
    .attr('text-anchor', 'middle')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('Forecast step')

  let xAxis = d3.axisBottom(xScale)
    .ticks(inforStore.forecast_scopes.length)
    .tickFormat((d,i) => `${inforStore.forecast_scopes[i][0]}~${inforStore.forecast_scopes[i][1]}`)
  let xAxis_g = err_dis_svg.append("g")
    .attr("transform", `translate(0, ${disPlotConfig.height-text_h-disPlotConfig.bottom_padding})`) // 将X轴移至底部
    .call(xAxis);
  let yScale = d3.scaleLinear()
    .domain([0, val_bins.length-1])
    .range([disPlotConfig.height-text_h-disPlotConfig.top_padding-disPlotConfig.bottom_padding, 0])
  let range_height = yScale(0) - yScale(1)
  let yMidScale = d3.scaleLinear()
    .domain([mid_bins[1], mid_bins[mid_bins.length-2]])
    .range([yScale(2), yScale(val_bins.length-3)])
  let yAxis = d3.axisLeft(yScale)
    .ticks(val_bins.length)
    .tickFormat(d => val_bins[d])
  let yAxis_g = err_dis_svg.append("g")
    .attr("transform", `translate(${disPlotConfig.left_padding}, ${disPlotConfig.top_padding})`) // 将X轴移至底部
    .call(yAxis);
  // yAxis_g.selectAll("text")
  //   .attr("transform", "rotate(-45)")
    // .style("text-anchor", "middle"); // 设置文本锚点位置
  let mildRScale = d3.scaleLinear()
    .domain(mild_outliers_num_range)
    .range([0.18, 0.42])
  let extremeRScale = d3.scaleLinear()
    .domain(extreme_outliers_num_range)
    .range([0.18, 0.42])

  let pattern_defs = err_dis_svg.append("defs")
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
  
  let err_highlight_pattern = pattern_defs.append("pattern")
    .attr("id", "err_highlight_pattern")
    .attr("patternUnits", "userSpaceOnUse")
    .attr("width", 5)
    .attr("height", 5)
  err_highlight_pattern.append("line")
    .attr("x1", 0)
    .attr("y1", 0)
    .attr("x2", 5)
    .attr("y2", 5)
    .attr("stroke", "#0097A7")
    .attr("stroke-width", 1)
  
  for (let k = 0; k < Object.keys(scope_err_infor_list).length; ++k) {
    let key = Object.keys(scope_err_infor_list)[k]
    for (let i = 1; i < scope_err_infor_list[key].length+1; ++i) {
      let step_err_infor = scope_err_infor_list[key][i-1]
      let step_err_box = err_dis_svg.append('g').attr('class', 'step-err-box')
      if (Object.keys(scope_err_infor_list).length > 1)
      step_err_box.attr('transform', `translate(${(k-0.5)*step_width}, ${disPlotConfig.top_padding})`)
      else step_err_box.attr('transform', `translate(0, ${disPlotConfig.top_padding})`)
      // 绘制箱线图
      step_err_box.append('line')
        .attr('x1', xScale(i))
        .attr('x2', xScale(i))
        .attr('y1', () => {
          if (step_err_infor.lower_whisker < val_bins[2]) return yMidScale(val_bins[2])
          else return yMidScale(step_err_infor.lower_whisker)
        })
        .attr('y2', () => {
          if (step_err_infor.upper_whisker > val_bins[val_bins.length - 3]) return yMidScale(val_bins[val_bins.length - 3])
          else return yMidScale(step_err_infor.upper_whisker)
        })
        .attr('stroke', '#999')
      step_err_box.append('rect')
        .attr('x', d => xScale(i) - 0.4 * step_width)
        .attr('y', d => yMidScale(step_err_infor.percentiles[2]))
        .attr('width', d => step_width * 0.8)
        .attr('height', d => yMidScale(step_err_infor.percentiles[0]) - yMidScale(step_err_infor.percentiles[2]))
        .attr('fill', d => {
          if (k == 0) {
            if (inforStore.cur_focused_scope != i) return '#cecece'
            else return '#0097A7'
          } else if (k == 1) {
            if (inforStore.cur_focused_scope != i) return 'url(#err_pattern)'
            else return 'url(#err_highlight_pattern)'
          }
        })
        .attr('stroke', '#999')
        .style('cursor', 'pointer')
        .on('click', (e,d) => {
          if (inforStore.cur_focused_scope == i) inforStore.cur_focused_scope = 0
          else inforStore.cur_focused_scope = i
        })
      step_err_box.append('line')
        .attr('x1', d => xScale(i) - 0.4 * step_width)
        .attr('x2', d => xScale(i) + 0.4 * step_width)
        .attr('y1', d => yMidScale(step_err_infor.percentiles[1]))
        .attr('y2', d => yMidScale(step_err_infor.percentiles[1]))
        .attr('stroke', d => {
          if (inforStore.cur_focused_scope != i) return '#333'
          else return '#fff'
        })
      step_err_box.append('line')
        .attr('x1', d => xScale(i) - 0.4 * step_width)
        .attr('x2', d => xScale(i) + 0.4 * step_width)
        .attr('y1', d => {
          if (step_err_infor.lower_whisker < val_bins[2]) return yMidScale(val_bins[2])
          else return yMidScale(step_err_infor.lower_whisker)
        })
        .attr('y2', d => {
          if (step_err_infor.lower_whisker < val_bins[2]) return yMidScale(val_bins[2])
          else return yMidScale(step_err_infor.lower_whisker)
        })
        .attr('stroke', d => {
          if (inforStore.cur_focused_scope != i) return '#333'
          // else return '#3182bd'
          else return '#0097A7'
        })
      step_err_box.append('line')
        .attr('x1', d => xScale(i) - 0.4 * step_width)
        .attr('x2', d => xScale(i) + 0.4 * step_width)
        .attr('y1', d => {
          if (step_err_infor.upper_whisker > val_bins[val_bins.length - 3]) return yMidScale(val_bins[val_bins.length - 3])
          else return yMidScale(step_err_infor.upper_whisker)
        })
        .attr('y2', d => {
          if (step_err_infor.upper_whisker > val_bins[val_bins.length - 3]) return yMidScale(val_bins[val_bins.length - 3])
          else return yMidScale(step_err_infor.upper_whisker)
        })
        .attr('stroke', d => {
          if (inforStore.cur_focused_scope != i) return '#333'
          // else return '#3182bd'
          else return '#0097A7'
        })
      
      // 绘制异常点
      let outlier_color = '#999'
      let outlier_stroke_width = 1
      step_err_box.append('circle')
        .attr('cx', d => xScale(i))
        .attr('cy', d => (yScale(1) + range_height / 2))
        .attr('r', d => extremeRScale(step_err_infor.extreme_neg_outliers_num)*range_height)
        .attr('fill', d => {
          if (inforStore.cur_focused_scope != i) return outlier_color
          // else return '#3182bd'
          else return '#0097A7'
        })
        .attr('stroke', 'none')
      step_err_box.append('circle')
        .attr('cx', d => xScale(i))
        .attr('cy', d => (yScale(2) + range_height / 2))
        .attr('r', d => mildRScale(step_err_infor.mild_neg_outliers_num)*range_height - outlier_stroke_width/2)
        .attr('fill', '#fff')
        .attr('stroke', d => {
          if (inforStore.cur_focused_scope != i) return outlier_color
          // else return '#3182bd'
          else return '#0097A7'
        })
        .attr('stroke-width', outlier_stroke_width)
      step_err_box.append('circle')
        .attr('cx', d => xScale(i))
        .attr('cy', d => (yScale(val_bins.length-1) + range_height / 2))
        .attr('r', d => extremeRScale(step_err_infor.extreme_pos_outliers_num)*range_height)
        .attr('fill', d => {
          if (inforStore.cur_focused_scope != i) return outlier_color
          // else return '#3182bd'
          else return '#0097A7'
        })
        .attr('stroke', 'none')
      step_err_box.append('circle')
        .attr('cx', d => xScale(i))
        .attr('cy', d => (yScale(val_bins.length-2) + range_height / 2))
        .attr('r', d => mildRScale(step_err_infor.mild_pos_outliers_num)*range_height - outlier_stroke_width/2)
        .attr('fill', '#fff')
        .attr('stroke', d => {
          if (inforStore.cur_focused_scope != i) return outlier_color
          // else return '#3182bd'
          else return '#0097A7'
        })
        .attr('stroke-width', outlier_stroke_width)
    }
  }
  // 绘制异常标记线
  err_dis_svg.append('line')
    .attr('x1', d => xScale(0.4))
    .attr('x2', d => xScale(inforStore.forecast_scopes.length+0.6))
    .attr('y1', d => yScale(val_bins.length-3)+disPlotConfig.top_padding)
    .attr('y2', d => yScale(val_bins.length-3)+disPlotConfig.top_padding)
    .attr('stroke', '#ef3b2c')
    .attr('stroke-width', 1)
    .attr('stroke-dasharray', '5,5')
  err_dis_svg.append('line')
    .attr('x1', d => xScale(0.4))
    .attr('x2', d => xScale(inforStore.forecast_scopes.length+0.6))
    .attr('y1', d => yScale(val_bins.length/2)+disPlotConfig.top_padding)
    .attr('y2', d => yScale(val_bins.length/2)+disPlotConfig.top_padding)
    .attr('stroke', '#ef3b2c')
    .attr('stroke-width', 1)
    .attr('stroke-dasharray', '5,5')
  err_dis_svg.append('line')
    .attr('x1', d => xScale(0.4))
    .attr('x2', d => xScale(inforStore.forecast_scopes.length+0.6))
    .attr('y1', d => yScale(2)+disPlotConfig.top_padding)
    .attr('y2', d => yScale(2)+disPlotConfig.top_padding)
    .attr('stroke', '#ef3b2c')
    .attr('stroke-width', 1)
    .attr('stroke-dasharray', '5,5')
  
}


function drawMidHist(bin_edges, hists, scope_id) {
  console.log('hists', hists);
  let all_hists = []
  for (let key in hists) {
    all_hists = all_hists.concat(hists[key][scope_id])
  }

  const histSort = [...all_hists].sort((a, b) => b - a);
  const valid_histSort = histSort.filter(element => element !== 0);
  const threshold = d3.quantile(valid_histSort, 0.5);
  let thBinId = 0;
  for (let i = 0; i < valid_histSort.length - 1; i++) {
      if (valid_histSort[i] > threshold && valid_histSort[i + 1] <= threshold) {
          thBinId = i;
          break;
      }
  }
  
  d3.select('#mid-dis').selectAll("*").remove()
  let margin_left = 33, margin_right = 22, margin_top = 17, margin_bottom = 18
  let main_w = 240, main_h = 134
  let text_h = 18
  let svg_w = main_w + margin_left + margin_right
  let svg_h = main_h + margin_top + margin_bottom
  let svg = d3.select('#mid-dis')
    .attr('width', svg_w)
    .attr('height', svg_h)

  let all_process_rate = []
  let all_used_cnts = []
  let process_rate = {}
  let used_cnts = {}
  for (let k = 0; k < Object.keys(hists).length; ++k) {
    let key = Object.keys(hists)[k]
    process_rate[key] = Array(hists[key][scope_id].length).fill(0)
    // 自适应调整倍数
    used_cnts[key] = [...hists[key][scope_id]]; // 复制 hist 数组
    // all_used_cnts = all_used_cnts.concat(used_cnts[key])
    if (valid_histSort[0] / valid_histSort[thBinId + 1] >= 5) {
        // 识别频次最高的 bin
        const maxFreq = valid_histSort[thBinId + 1];
        for (let i = 0; i < hists[key][scope_id].length; i++) {
            if (hists[key][scope_id][i] > threshold) { // high_freq_bins 的逻辑替换
                // 计算自适应调整倍数，使调整后频次不超过 maxFreq
                const adjustmentFactor = hists[key][scope_id][i] / maxFreq
                // const adjustmentFactor = Math.floor(range_cnts[i] / maxFreq)
                process_rate[key][i] = Math.floor(adjustmentFactor * 10) / 10
                all_process_rate.push(process_rate[key][i])
                used_cnts[key][i] = Math.floor(hists[key][scope_id][i] / adjustmentFactor)
                all_used_cnts.push(used_cnts[key][i])
            }
        }
    }
  }

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
    .attr("stroke", "#666")
    .attr("stroke-width", 1)
  let err_highlight_pattern = pattern_defs.append("pattern")
    .attr("id", "err_highlight_pattern")
    .attr("patternUnits", "userSpaceOnUse")
    .attr("width", 5)
    .attr("height", 5)
  err_highlight_pattern.append("line")
    .attr("x1", 0)
    .attr("y1", 0)
    .attr("x2", 5)
    .attr("y2", 5)
    .attr("stroke", "#0097A7")
    .attr("stroke-width", 1)

  // define scale
  let xScale = d3.scaleLinear()
    .domain([0, bin_edges.length-1])
    .range([0, main_w])
  let yScale = d3.scaleLinear()
    .domain([0, d3.max(all_used_cnts)])
    .range([main_h-text_h, 0])
  let rateScale = d3.scaleLinear()
    .domain([0, d3.max(all_process_rate)*1.1])
    .range([main_h-text_h, 0])

  let xAxis = d3.axisBottom(xScale)
    .ticks(bin_edges.length)
    .tickFormat((d) => bin_edges[d])
  const yAxis = d3.axisLeft(yScale)
    .ticks(5)
    .tickFormat(d3.format("~s"))
  let rateAxis = d3.axisRight(rateScale)
      // .tickValues(process_rate)
      .ticks(5)
  
  svg.append('text')
    .attr('x', 4)
    .attr('y', 12)
    .attr('text-anchor', 'start')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('#predictions')
  svg.append('text')
    .attr('x', svg_w/2+margin_left/2-4)
    .attr('y', svg_h-6)
    .attr('text-anchor', 'middle')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('Residual Range')
  svg.append('text')
    .attr('x', svg_w-1)
    .attr('y', 12)
    .attr('text-anchor', 'end')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('Multiples')
  let hist_plot = svg.append('g').attr('class', 'hist-plot')
    .attr("transform", `translate(${margin_left},${margin_top})`)
  let xAxis_g = hist_plot.append("g")
    .attr("transform", `translate(0, ${main_h-text_h})`)
    .call(xAxis);
  // xAxis_g.selectAll("text")
  //   .attr("transform", "translate(8,6) rotate(45)")
  //   .style("text-anchor", "middle"); // 设置文本锚点位置
  let yAxis_g = hist_plot.append("g")
    // .attr("transform", `translate(0, 0)`)
    .call(yAxis);
  if (valid_histSort[0] / valid_histSort[thBinId + 1] >= 5) {
    let rateAxis_g = hist_plot.append("g")
      .attr("transform", `translate(${main_w},0)`)
      .call(rateAxis)
  }

  let bin_width = xScale(1) - xScale(0)
  if (Object.keys(hists).length > 1) {
    bin_width /= 2
  }

  for (let k = 0; k < Object.keys(hists).length; ++k) {
    let key = Object.keys(hists)[k]
    let bins_g = hist_plot.append('g')
    bins_g.selectAll('rect')
      .data(used_cnts[key])
      .join('rect')
        .attr('transform', `translate(${k*bin_width}, 0)`)
        .attr('x', (d,i) => xScale(i)+1*(1-k))
        .attr('y', (d,i) => yScale(d))
        .attr('bin_id', (d,i) => i)
        .attr('width', (d,i) => bin_width-1)
        .attr('height', (d,i) => main_h-text_h-yScale(d))
        .attr('fill', (d,i) => {
          if (k == 0) {
            if (i <= 1 || i >= bin_edges.length-3) {
              return '#0097A7'
            } else {
              return "#999"
            }
          } else {
            if (i <= 1 || i >= bin_edges.length-3) {
              return 'url(#err_highlight_pattern)'
            } else {
              return "url(#err_pattern)"
            }
          }
        })
        .attr('opacity', (d,i) => {
          if (k == 0) {
            return 0.8
          } else {
            return 1
          }
        })
        .on('mouseover', (e,d) => {
          cur_mid_hover_val.value = hists[key][scope_id][d3.select(e.target).attr('bin_id')]
          d3.select('#mid-hist-tooltip')
            .style('left', e.clientX+'px')
            .style('top', e.clientY+'px')
            .style('opacity', 1)
        })
        .on('mouseout', (e,d) => {
          cur_mid_hover_val.value = -1
          d3.select('#mid-hist-tooltip')
            .style('opacity', 0)
        })
    if (valid_histSort[0] / valid_histSort[thBinId + 1] >= 5) {
      let rete_marks = hist_plot.append('g')
      rete_marks.append('g').selectAll('line')
        .data(process_rate[key])
        .join('line')
          .attr('x1', (d,i) => xScale(i+0.5*k)+1*(1-k)).attr('x2', (d,i) => xScale(i+0.5*k)+bin_width-1+1*(1-k))
          .attr('y1', (d,i) => rateScale(d)).attr('y2', (d,i) => rateScale(d))
          .attr('stroke', (d,i) => {
            if (k == 0) {
              if (d>0) {
                // if (i < 2 || i > bin_edges.length-3) return '#fff'
                // else return '#666'
                return '#fff'
              } 
              else return 'none'
            } else {
              if (d>0) return '#333'
              else return 'none'
            }
          })
      rete_marks.append('g').selectAll('text')
        .data(process_rate[key])
        .join('text')
          .attr('x', (d,i) => xScale(i+0.5*k)+1*(1-k)+(bin_width-1)/2)
          .attr('y', (d,i) => rateScale(d)+4)
          .attr('text-anchor', 'middle')
          .attr('fill', (d,i) => {
            if (k == 0) {
              if (d>0) {
                // if (i < 2 || i > bin_edges.length-3) return '#fff'
                // else return '#666'
                return '#fff'
              } 
              else return 'none'
            } else {
              if (d>0) return '#333'
              else return 'none'
            }
          })
          .text('x')
    }
  }
}

function drawExtremeHist(bin_edges, hists, scope_id, polarity) {
  let all_hists = []
  for (let key in hists) {
    all_hists = all_hists.concat(hists[key][scope_id])
  }

  const histSort = [...all_hists].sort((a, b) => b - a);
  const valid_histSort = histSort.filter(element => element !== 0);
  const threshold = d3.quantile(valid_histSort, 0.5);
  let thBinId = 0;
  for (let i = 0; i < valid_histSort.length - 1; i++) {
      if (valid_histSort[i] > threshold && valid_histSort[i + 1] <= threshold) {
          thBinId = i;
          break;
      }
  }

  let all_process_rate = []
  let all_used_cnts = []
  let process_rate = {}
  let used_cnts = {}
  for (let k = 0; k < Object.keys(hists).length; ++k) {
    let key = Object.keys(hists)[k]
    process_rate[key] = Array(hists[key][scope_id].length).fill(0)
    // 自适应调整倍数
    used_cnts[key] = [...hists[key][scope_id]]; // 复制 hist 数组
    // all_used_cnts = all_used_cnts.concat(used_cnts[key])
    if (valid_histSort[0] / valid_histSort[thBinId + 1] >= 5) {
        // 识别频次最高的 bin
        const maxFreq = valid_histSort[thBinId + 1];
        for (let i = 0; i < hists[key][scope_id].length; i++) {
            if (hists[key][scope_id][i] > threshold) { // high_freq_bins 的逻辑替换
                // 计算自适应调整倍数，使调整后频次不超过 maxFreq
                const adjustmentFactor = hists[key][scope_id][i] / maxFreq
                // const adjustmentFactor = Math.floor(range_cnts[i] / maxFreq)
                process_rate[key][i] = Math.floor(adjustmentFactor * 10) / 10
                all_process_rate.push(process_rate[key][i])
                used_cnts[key][i] = Math.floor(hists[key][scope_id][i] / adjustmentFactor)
                all_used_cnts.push(used_cnts[key][i])
            }
        }
    }
  }
  
  let svg_id = `extreme-${polarity}-dis`
  d3.select(`#${svg_id}`).selectAll("*").remove()
  let margin_left = 32, margin_right = 26, margin_top = 17, margin_bottom = 17
  let main_w = 240, main_h = 134
  let text_h = 18
  let svg_w = main_w + margin_left + margin_right
  let svg_h = main_h + margin_top + margin_bottom
  let svg = d3.select(`#${svg_id}`)
    .attr('width', svg_w)
    .attr('height', svg_h)
  
  let pattern_defs = svg.append("defs")
  let err_highlight_pattern = pattern_defs.append("pattern")
    .attr("id", "err_highlight_pattern")
    .attr("patternUnits", "userSpaceOnUse")
    .attr("width", 5)
    .attr("height", 5)
  err_highlight_pattern.append("line")
    .attr("x1", 0)
    .attr("y1", 0)
    .attr("x2", 5)
    .attr("y2", 5)
    .attr("stroke", "#0097A7")
    .attr("stroke-width", 1)

  // define scale
  let xScale = d3.scaleLinear()
    .domain([0, bin_edges.length-1])
    .range([0, main_w])
  let yScale = d3.scaleLinear()
    .domain([0, d3.max(all_used_cnts)])
    .range([main_h-text_h, 0])
  let rateScale = d3.scaleLinear()
    .domain([0, d3.max(all_process_rate)*1.1])
    .range([main_h-text_h, 0])

  let xAxis = d3.axisBottom(xScale)
    .ticks(bin_edges.length)
    .tickFormat((d) => bin_edges[d])
  const yAxis = d3.axisLeft(yScale)
    .ticks(5)
    .tickFormat(d3.format("~s"))
  let rateAxis = d3.axisRight(rateScale)
      // .tickValues(process_rate)
      .ticks(5)
  
  svg.append('text')
    .attr('x', 4)
    .attr('y', 12)
    .attr('text-anchor', 'start')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('#Predictions')
  svg.append('text')
    .attr('x', svg_w/2+margin_left/2-4)
    .attr('y', svg_h-6)
    .attr('text-anchor', 'middle')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('Residual Range')
  svg.append('text')
    .attr('x', svg_w-1)
    .attr('y', 12)
    .attr('text-anchor', 'end')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('Multiples')
    
  let hist_plot = svg.append('g').attr('class', 'hist-plot')
    .attr("transform", `translate(${margin_left},${margin_top})`)
  let xAxis_g = hist_plot.append("g")
    .attr("transform", `translate(0, ${main_h-text_h})`)
    .call(xAxis);
  // xAxis_g.selectAll("text")
  //   .attr("transform", "translate(8,6) rotate(45)")
  //   .style("text-anchor", "middle"); // 设置文本锚点位置
  let yAxis_g = hist_plot.append("g")
    // .attr("transform", `translate(0, 0)`)
    .call(yAxis);
  if (valid_histSort[0] / valid_histSort[thBinId + 1] >= 5) {
    let rateAxis_g = hist_plot.append("g")
      .attr("transform", `translate(${main_w},0)`)
      .call(rateAxis)
  }

  let bin_width = xScale(1) - xScale(0)
  if (Object.keys(hists).length > 1) {
    bin_width /= 2
  }

  for (let k = 0; k < Object.keys(hists).length; ++k) {
    let key = Object.keys(hists)[k]
    let bins_g = hist_plot.append('g')

    bins_g.selectAll('rect')
      .data(used_cnts[key])
      .join('rect')
        .attr('transform', `translate(${k*bin_width}, 0)`)
        .attr('x', (d,i) => xScale(i)+1*(1-k))
        .attr('y', (d,i) => yScale(d))
        .attr('bin_id', (d,i) => i)
        .attr('width', (d,i) => bin_width-1)
        .attr('height', (d,i) => main_h-text_h-yScale(d))
        .attr('fill', (d,i) => {
          if (k == 0) {
            return "#0097A7"
          } else {
            return "url(#err_highlight_pattern)"
          }
        })
        .attr('opacity', (d,i) => {
          if (k == 0) {
            return 0.8
          } else {
            return 1
          }
        })
        .on('mouseover', (e,d) => {
          if (polarity == 'pos') {
            cur_pos_extreme_hover_val.value = hists[key][scope_id][d3.select(e.target).attr('bin_id')]
            d3.select('#pos-extreme-tooltip')
              .style('left', e.clientX+'px')
              .style('top', e.clientY+'px')
              .style('opacity', 1)
          } else {
            cur_neg_extreme_hover_val.value = hists[key][scope_id][d3.select(e.target).attr('bin_id')]
            d3.select('#neg-extreme-tooltip')
              .style('left', e.clientX+'px')
              .style('top', e.clientY+'px')
              .style('opacity', 1)
          }
        })
        .on('mouseout', (e,d) => {
          if (polarity == 'pos') {
            cur_pos_extreme_hover_val.value = -1
            d3.select('#pos-extreme-tooltip')
              .style('opacity', 0)
          } else {
            cur_neg_extreme_hover_val.value = -1
            d3.select('#neg-extreme-tooltip')
              .style('opacity', 0)
          }
        })
    
    if (valid_histSort[0] / valid_histSort[thBinId + 1] >= 5) {
      let rete_marks = hist_plot.append('g')
      rete_marks.append('g').selectAll('line')
        .data(process_rate[key])
        .join('line')
          .attr('x1', (d,i) => xScale(i+0.5*k)+1*(1-k)).attr('x2', (d,i) => xScale(i+0.5*k)+bin_width-1+1*(1-k))
          .attr('y1', (d,i) => rateScale(d)).attr('y2', (d,i) => rateScale(d))
          .attr('stroke', (d,i) => {
            if (k == 0) {
              if (d>0) {
                // if (i < 2 || i > bin_edges.length-3) return '#fff'
                // else return '#666'
                return '#fff'
              } 
              else return 'none'
            } else {
              if (d>0) return '#333'
              else return 'none'
            }
          })
      rete_marks.append('g').selectAll('text')
        .data(process_rate[key])
        .join('text')
          .attr('x', (d,i) => xScale(i+0.5*k)+1*(1-k)+(bin_width-1)/2)
          .attr('y', (d,i) => rateScale(d)+4)
          .attr('text-anchor', 'middle')
          .attr('fill', (d,i) => {
            if (k == 0) {
              if (d>0) {
                // if (i < 2 || i > bin_edges.length-3) return '#fff'
                // else return '#666'
                return '#fff'
              } 
              else return 'none'
            } else {
              if (d>0) return '#333'
              else return 'none'
            }
          })
          .text('x')
    }
  }
}

function drawOriginalMidHist(bin_edges, hists, scope_id) {
  let all_hists = []
  for (let key in hists) {
    all_hists = all_hists.concat(hists[key][scope_id])
  }

  const histSort = [...all_hists].sort((a, b) => b - a);
  const valid_histSort = histSort.filter(element => element !== 0);
  const threshold = d3.quantile(valid_histSort, 0.5);
  let thBinId = 0;
  for (let i = 0; i < valid_histSort.length - 1; i++) {
      if (valid_histSort[i] > threshold && valid_histSort[i + 1] <= threshold) {
          thBinId = i;
          break;
      }
  }
  
  d3.select('#mid-dis').selectAll("*").remove()
  let margin_left = 33, margin_right = 22, margin_top = 17, margin_bottom = 18
  let main_w = 240, main_h = 134
  let text_h = 18
  let svg_w = main_w + margin_left + margin_right
  let svg_h = main_h + margin_top + margin_bottom
  let svg = d3.select('#mid-dis')
    .attr('width', svg_w)
    .attr('height', svg_h)

  let all_process_rate = []
  let all_used_cnts = []
  let process_rate = {}
  let used_cnts = {}
  for (let k = 0; k < Object.keys(hists).length; ++k) {
    let key = Object.keys(hists)[k]
    process_rate[key] = Array(hists[key][scope_id].length).fill(0)
    // 自适应调整倍数
    used_cnts[key] = [...hists[key][scope_id].slice(-3)]; // 复制 hist 数组
    all_used_cnts = all_used_cnts.concat(used_cnts[key])
  }

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
    .attr("stroke", "#666")
    .attr("stroke-width", 1)
  let err_highlight_pattern = pattern_defs.append("pattern")
    .attr("id", "err_highlight_pattern")
    .attr("patternUnits", "userSpaceOnUse")
    .attr("width", 5)
    .attr("height", 5)
  err_highlight_pattern.append("line")
    .attr("x1", 0)
    .attr("y1", 0)
    .attr("x2", 5)
    .attr("y2", 5)
    .attr("stroke", "#0097A7")
    .attr("stroke-width", 1)

  // define scale
  let xScale = d3.scaleLinear()
    .domain([0, bin_edges.slice(-4).length-1])
    .range([0, main_w])
  let yScale = d3.scaleLinear()
    .domain([0, d3.max(all_used_cnts)])
    .range([main_h-text_h, 0])
  let rateScale = d3.scaleLinear()
    .domain([0, d3.max(all_process_rate)*1.1])
    .range([main_h-text_h, 0])

  let xAxis = d3.axisBottom(xScale)
    .ticks(bin_edges.slice(-4).length)
    .tickFormat((d) => bin_edges.slice(-4)[d])
  const yAxis = d3.axisLeft(yScale)
    .ticks(5)
    .tickFormat(d3.format("~s"))
  let rateAxis = d3.axisRight(rateScale)
      // .tickValues(process_rate)
      .ticks(5)
  
  svg.append('text')
    .attr('x', 4)
    .attr('y', 12)
    .attr('text-anchor', 'start')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('#predictions')
  svg.append('text')
    .attr('x', svg_w/2+margin_left/2-4)
    .attr('y', svg_h-6)
    .attr('text-anchor', 'middle')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('Residual Range')

  let hist_plot = svg.append('g').attr('class', 'hist-plot')
    .attr("transform", `translate(${margin_left},${margin_top})`)
  let xAxis_g = hist_plot.append("g")
    .attr("transform", `translate(0, ${main_h-text_h})`)
    .call(xAxis);
  // xAxis_g.selectAll("text")
  //   .attr("transform", "translate(8,6) rotate(45)")
  //   .style("text-anchor", "middle"); // 设置文本锚点位置
  let yAxis_g = hist_plot.append("g")
    // .attr("transform", `translate(0, 0)`)
    .call(yAxis);

  let bin_width = xScale(1) - xScale(0)
  if (Object.keys(hists).length > 1) {
    bin_width /= 2
  }

  for (let k = 0; k < Object.keys(hists).length; ++k) {
    let key = Object.keys(hists)[k]
    let bins_g = hist_plot.append('g')
    bins_g.selectAll('rect')
      .data(used_cnts[key])
      .join('rect')
        .attr('transform', `translate(${k*bin_width}, 0)`)
        .attr('x', (d,i) => xScale(i)+1*(1-k))
        .attr('y', (d,i) => yScale(d))
        .attr('bin_id', (d,i) => i)
        .attr('width', (d,i) => bin_width-1)
        .attr('height', (d,i) => main_h-text_h-yScale(d))
        .attr('fill', (d,i) => {
          if (k == 0) {
            if (i >= bin_edges.slice(-4).length-3) {
              return '#0097A7'
            } else {
              return "#999"
            }
          } else {
            if (i >= bin_edges.slice(-4).length-3) {
              return 'url(#err_highlight_pattern)'
            } else {
              return "url(#err_pattern)"
            }
          }
        })
        .attr('opacity', (d,i) => {
          if (k == 0) {
            return 0.8
          } else {
            return 1
          }
        })
        .on('mouseover', (e,d) => {
          cur_mid_hover_val.value = hists[key][scope_id][d3.select(e.target).attr('bin_id')]
          d3.select('#mid-hist-tooltip')
            .style('left', e.clientX+'px')
            .style('top', e.clientY+'px')
            .style('opacity', 1)
        })
        .on('mouseout', (e,d) => {
          cur_mid_hover_val.value = -1
          d3.select('#mid-hist-tooltip')
            .style('opacity', 0)
        })
  }
}

function drawOriginalExtremeHist(bin_edges, hists, scope_id, polarity) {
  let all_hists = []
  for (let key in hists) {
    all_hists = all_hists.concat(hists[key][scope_id])
  }

  const histSort = [...all_hists].sort((a, b) => b - a);
  const valid_histSort = histSort.filter(element => element !== 0);
  const threshold = d3.quantile(valid_histSort, 0.5);
  let thBinId = 0;
  for (let i = 0; i < valid_histSort.length - 1; i++) {
      if (valid_histSort[i] > threshold && valid_histSort[i + 1] <= threshold) {
          thBinId = i;
          break;
      }
  }

  let all_process_rate = []
  let all_used_cnts = []
  let process_rate = {}
  let used_cnts = {}
  for (let k = 0; k < Object.keys(hists).length; ++k) {
    let key = Object.keys(hists)[k]
    process_rate[key] = Array(hists[key][scope_id].length).fill(0)
    // 自适应调整倍数
    used_cnts[key] = [...hists[key][scope_id]]; // 复制 hist 数组
    all_used_cnts = all_used_cnts.concat(used_cnts[key])
  }
  
  let svg_id = `extreme-${polarity}-dis`
  d3.select(`#${svg_id}`).selectAll("*").remove()
  let margin_left = 32, margin_right = 26, margin_top = 17, margin_bottom = 18
  let main_w = 240, main_h = 134
  let text_h = 18
  let svg_w = main_w + margin_left + margin_right
  let svg_h = main_h + margin_top + margin_bottom
  let svg = d3.select(`#${svg_id}`)
    .attr('width', svg_w)
    .attr('height', svg_h)
  
  let pattern_defs = svg.append("defs")
  let err_highlight_pattern = pattern_defs.append("pattern")
    .attr("id", "err_highlight_pattern")
    .attr("patternUnits", "userSpaceOnUse")
    .attr("width", 5)
    .attr("height", 5)
  err_highlight_pattern.append("line")
    .attr("x1", 0)
    .attr("y1", 0)
    .attr("x2", 5)
    .attr("y2", 5)
    .attr("stroke", "#0097A7")
    .attr("stroke-width", 1)

  // define scale
  let xScale = d3.scaleLinear()
    .domain([0, bin_edges.length-1])
    .range([0, main_w])
  let yScale = d3.scaleLinear()
    .domain([0, d3.max(all_used_cnts)])
    .range([main_h-text_h, 0])
  let rateScale = d3.scaleLinear()
    .domain([0, d3.max(all_process_rate)*1.1])
    .range([main_h-text_h, 0])

  let xAxis = d3.axisBottom(xScale)
    .ticks(bin_edges.length)
    .tickFormat((d) => bin_edges[d])
  const yAxis = d3.axisLeft(yScale)
    .ticks(5)
    .tickFormat(d3.format("~s"))
  let rateAxis = d3.axisRight(rateScale)
      // .tickValues(process_rate)
      .ticks(5)
  
  svg.append('text')
    .attr('x', 4)
    .attr('y', 12)
    .attr('text-anchor', 'start')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('#Predictions')
  svg.append('text')
    .attr('x', svg_w/2+margin_left/2-4)
    .attr('y', svg_h-6)
    .attr('text-anchor', 'middle')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('Residual Range')
    
  let hist_plot = svg.append('g').attr('class', 'hist-plot')
    .attr("transform", `translate(${margin_left},${margin_top})`)
  let xAxis_g = hist_plot.append("g")
    .attr("transform", `translate(0, ${main_h-text_h})`)
    .call(xAxis);
  // xAxis_g.selectAll("text")
  //   .attr("transform", "translate(8,6) rotate(45)")
  //   .style("text-anchor", "middle"); // 设置文本锚点位置
  let yAxis_g = hist_plot.append("g")
    // .attr("transform", `translate(0, 0)`)
    .call(yAxis);
  // if (valid_histSort[0] / valid_histSort[thBinId + 1] >= 5) {
  //   let rateAxis_g = hist_plot.append("g")
  //     .attr("transform", `translate(${main_w},0)`)
  //     .call(rateAxis)
  // }

  let bin_width = xScale(1) - xScale(0)
  if (Object.keys(hists).length > 1) {
    bin_width /= 2
  }

  for (let k = 0; k < Object.keys(hists).length; ++k) {
    let key = Object.keys(hists)[k]
    let bins_g = hist_plot.append('g')

    bins_g.selectAll('rect')
      .data(used_cnts[key])
      .join('rect')
        .attr('transform', `translate(${k*bin_width}, 0)`)
        .attr('x', (d,i) => xScale(i)+1*(1-k))
        .attr('y', (d,i) => yScale(d))
        .attr('bin_id', (d,i) => i)
        .attr('width', (d,i) => bin_width-1)
        .attr('height', (d,i) => main_h-text_h-yScale(d))
        .attr('fill', (d,i) => {
          if (k == 0) {
            return "#0097A7"
          } else {
            return "url(#err_highlight_pattern)"
          }
        })
        .attr('opacity', (d,i) => {
          if (k == 0) {
            return 0.8
          } else {
            return 1
          }
        })
        .on('mouseover', (e,d) => {
          if (polarity == 'pos') {
            cur_pos_extreme_hover_val.value = hists[key][scope_id][d3.select(e.target).attr('bin_id')]
            d3.select('#pos-extreme-tooltip')
              .style('left', e.clientX+'px')
              .style('top', e.clientY+'px')
              .style('opacity', 1)
          } else {
            cur_neg_extreme_hover_val.value = hists[key][scope_id][d3.select(e.target).attr('bin_id')]
            d3.select('#neg-extreme-tooltip')
              .style('left', e.clientX+'px')
              .style('top', e.clientY+'px')
              .style('opacity', 1)
          }
        })
        .on('mouseout', (e,d) => {
          if (polarity == 'pos') {
            cur_pos_extreme_hover_val.value = -1
            d3.select('#pos-extreme-tooltip')
              .style('opacity', 0)
          } else {
            cur_neg_extreme_hover_val.value = -1
            d3.select('#neg-extreme-tooltip')
              .style('opacity', 0)
          }
        })
  }
}

function drawLogMidHist(bin_edges, hists, scope_id) {
  let all_hists = []
  for (let key in hists) {
    all_hists = all_hists.concat(hists[key][scope_id])
  }

  const histSort = [...all_hists].sort((a, b) => b - a);
  const valid_histSort = histSort.filter(element => element !== 0);
  const threshold = d3.quantile(valid_histSort, 0.5);
  let thBinId = 0;
  for (let i = 0; i < valid_histSort.length - 1; i++) {
      if (valid_histSort[i] > threshold && valid_histSort[i + 1] <= threshold) {
          thBinId = i;
          break;
      }
  }
  
  d3.select('#mid-dis').selectAll("*").remove()
  let margin_left = 33, margin_right = 22, margin_top = 17, margin_bottom = 18
  let main_w = 240, main_h = 134
  let text_h = 18
  let svg_w = main_w + margin_left + margin_right
  let svg_h = main_h + margin_top + margin_bottom
  let svg = d3.select('#mid-dis')
    .attr('width', svg_w)
    .attr('height', svg_h)

  let all_process_rate = []
  let all_used_cnts = []
  let process_rate = {}
  let used_cnts = {}
  for (let k = 0; k < Object.keys(hists).length; ++k) {
    let key = Object.keys(hists)[k]
    process_rate[key] = Array(hists[key][scope_id].length).fill(0)
    // 自适应调整倍数
    used_cnts[key] = [...hists[key][scope_id]]
    all_used_cnts = all_used_cnts.concat(used_cnts[key])
    // if (valid_histSort[0] / valid_histSort[thBinId + 1] >= 5) {
    //     // 识别频次最高的 bin
    //     const maxFreq = valid_histSort[thBinId + 1];
    //     for (let i = 0; i < hists[key][scope_id].length; i++) {
    //         if (hists[key][scope_id][i] > threshold) { // high_freq_bins 的逻辑替换
    //             // 计算自适应调整倍数，使调整后频次不超过 maxFreq
    //             const adjustmentFactor = hists[key][scope_id][i] / maxFreq
    //             // const adjustmentFactor = Math.floor(range_cnts[i] / maxFreq)
    //             process_rate[key][i] = Math.floor(adjustmentFactor * 10) / 10
    //             all_process_rate.push(process_rate[key][i])
    //             used_cnts[key][i] = Math.floor(hists[key][scope_id][i] / adjustmentFactor)
    //             all_used_cnts.push(used_cnts[key][i])
    //         }
    //     }
    // }
  }

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
    .attr("stroke", "#666")
    .attr("stroke-width", 1)
  let err_highlight_pattern = pattern_defs.append("pattern")
    .attr("id", "err_highlight_pattern")
    .attr("patternUnits", "userSpaceOnUse")
    .attr("width", 5)
    .attr("height", 5)
  err_highlight_pattern.append("line")
    .attr("x1", 0)
    .attr("y1", 0)
    .attr("x2", 5)
    .attr("y2", 5)
    .attr("stroke", "#0097A7")
    .attr("stroke-width", 1)

  // define scale
  let xScale = d3.scaleLinear()
    .domain([0, bin_edges.length-1])
    .range([0, main_w])
  let yScale = d3.scaleLog()
    .domain([1, d3.max(all_used_cnts)])
    .range([main_h-text_h, 0])
  // 创建 y 轴
  let rateScale = d3.scaleLinear()
    .domain([0, d3.max(all_process_rate)*1.1])
    .range([main_h-text_h, 0])

  let xAxis = d3.axisBottom(xScale)
    .ticks(bin_edges.length)
    .tickFormat((d) => bin_edges[d])
  const yAxis = d3.axisLeft(yScale)
    .ticks(5)
    .tickFormat((d) => {
      const expData = Math.exp(d)
      return d3.format("~s")(d)
    })
  let rateAxis = d3.axisRight(rateScale)
      // .tickValues(process_rate)
      .ticks(5)
  
  svg.append('text')
    .attr('x', 4)
    .attr('y', 12)
    .attr('text-anchor', 'start')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('#predictions')
  svg.append('text')
    .attr('x', svg_w/2+margin_left/2-4)
    .attr('y', svg_h-6)
    .attr('text-anchor', 'middle')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('Residual Range')
  // svg.append('text')
  //   .attr('x', svg_w-1)
  //   .attr('y', 12)
  //   .attr('text-anchor', 'end')
  //   .style('font-size', '12px')
  //   .attr('fill', '#333')
  //   .text('Multiples')
  let hist_plot = svg.append('g').attr('class', 'hist-plot')
    .attr("transform", `translate(${margin_left},${margin_top})`)
  let xAxis_g = hist_plot.append("g")
    .attr("transform", `translate(0, ${main_h-text_h})`)
    .call(xAxis);
  // xAxis_g.selectAll("text")
  //   .attr("transform", "translate(8,6) rotate(45)")
  //   .style("text-anchor", "middle"); // 设置文本锚点位置
  let yAxis_g = hist_plot.append("g")
    // .attr("transform", `translate(0, 0)`)
    .call(yAxis);
  // if (valid_histSort[0] / valid_histSort[thBinId + 1] >= 5) {
  //   let rateAxis_g = hist_plot.append("g")
  //     .attr("transform", `translate(${main_w},0)`)
  //     .call(rateAxis)
  // }

  let bin_width = xScale(1) - xScale(0)
  if (Object.keys(hists).length > 1) {
    bin_width /= 2
  }

  for (let k = 0; k < Object.keys(hists).length; ++k) {
    let key = Object.keys(hists)[k]
    let bins_g = hist_plot.append('g')
    bins_g.selectAll('rect')
      .data(used_cnts[key])
      .join('rect')
        .attr('transform', `translate(${k*bin_width}, 0)`)
        .attr('x', (d,i) => xScale(i)+1*(1-k))
        .attr('y', (d,i) => yScale(d))
        .attr('bin_id', (d,i) => i)
        .attr('width', (d,i) => bin_width-1)
        .attr('height', (d,i) => main_h-text_h-yScale(d))
        .attr('fill', (d,i) => {
          if (k == 0) {
            if (i <= 1 || i >= bin_edges.length-3) {
              return '#0097A7'
            } else {
              return "#999"
            }
          } else {
            if (i <= 1 || i >= bin_edges.length-3) {
              return 'url(#err_highlight_pattern)'
            } else {
              return "url(#err_pattern)"
            }
          }
        })
        .attr('opacity', (d,i) => {
          if (k == 0) {
            return 0.8
          } else {
            return 1
          }
        })
        .on('mouseover', (e,d) => {
          cur_mid_hover_val.value = hists[key][scope_id][d3.select(e.target).attr('bin_id')]
          d3.select('#mid-hist-tooltip')
            .style('left', e.clientX+'px')
            .style('top', e.clientY+'px')
            .style('opacity', 1)
        })
        .on('mouseout', (e,d) => {
          cur_mid_hover_val.value = -1
          d3.select('#mid-hist-tooltip')
            .style('opacity', 0)
        })
    // if (valid_histSort[0] / valid_histSort[thBinId + 1] >= 5) {
    //   let rete_marks = hist_plot.append('g')
    //   rete_marks.append('g').selectAll('line')
    //     .data(process_rate[key])
    //     .join('line')
    //       .attr('x1', (d,i) => xScale(i+0.5*k)+1*(1-k)).attr('x2', (d,i) => xScale(i+0.5*k)+bin_width-1+1*(1-k))
    //       .attr('y1', (d,i) => rateScale(d)).attr('y2', (d,i) => rateScale(d))
    //       .attr('stroke', (d,i) => {
    //         if (k == 0) {
    //           if (d>0) {
    //             // if (i < 2 || i > bin_edges.length-3) return '#fff'
    //             // else return '#666'
    //             return '#fff'
    //           } 
    //           else return 'none'
    //         } else {
    //           if (d>0) return '#333'
    //           else return 'none'
    //         }
    //       })
    //   rete_marks.append('g').selectAll('text')
    //     .data(process_rate[key])
    //     .join('text')
    //       .attr('x', (d,i) => xScale(i+0.5*k)+1*(1-k)+(bin_width-1)/2)
    //       .attr('y', (d,i) => rateScale(d)+4)
    //       .attr('text-anchor', 'middle')
    //       .attr('fill', (d,i) => {
    //         if (k == 0) {
    //           if (d>0) {
    //             // if (i < 2 || i > bin_edges.length-3) return '#fff'
    //             // else return '#666'
    //             return '#fff'
    //           } 
    //           else return 'none'
    //         } else {
    //           if (d>0) return '#333'
    //           else return 'none'
    //         }
    //       })
    //       .text('x')
    // }
  }
}

function drawLogExtremeHist(bin_edges, hists, scope_id, polarity) {
  let all_hists = []
  for (let key in hists) {
    all_hists = all_hists.concat(hists[key][scope_id])
  }

  const histSort = [...all_hists].sort((a, b) => b - a);
  const valid_histSort = histSort.filter(element => element !== 0);
  const threshold = d3.quantile(valid_histSort, 0.5);
  let thBinId = 0;
  for (let i = 0; i < valid_histSort.length - 1; i++) {
      if (valid_histSort[i] > threshold && valid_histSort[i + 1] <= threshold) {
          thBinId = i;
          break;
      }
  }

  let all_process_rate = []
  let all_used_cnts = []
  let process_rate = {}
  let used_cnts = {}
  for (let k = 0; k < Object.keys(hists).length; ++k) {
    let key = Object.keys(hists)[k]
    process_rate[key] = Array(hists[key][scope_id].length).fill(0)
    // 自适应调整倍数
    used_cnts[key] = [...hists[key][scope_id]]
    all_used_cnts = all_used_cnts.concat(used_cnts[key])
    // if (valid_histSort[0] / valid_histSort[thBinId + 1] >= 5) {
    //     // 识别频次最高的 bin
    //     const maxFreq = valid_histSort[thBinId + 1];
    //     for (let i = 0; i < hists[key][scope_id].length; i++) {
    //         if (hists[key][scope_id][i] > threshold) { // high_freq_bins 的逻辑替换
    //             // 计算自适应调整倍数，使调整后频次不超过 maxFreq
    //             const adjustmentFactor = hists[key][scope_id][i] / maxFreq
    //             // const adjustmentFactor = Math.floor(range_cnts[i] / maxFreq)
    //             process_rate[key][i] = Math.floor(adjustmentFactor * 10) / 10
    //             all_process_rate.push(process_rate[key][i])
    //             used_cnts[key][i] = Math.floor(hists[key][scope_id][i] / adjustmentFactor)
    //             all_used_cnts.push(used_cnts[key][i])
    //         }
    //     }
    // }
  }
  
  let svg_id = `extreme-${polarity}-dis`
  d3.select(`#${svg_id}`).selectAll("*").remove()
  let margin_left = 32, margin_right = 26, margin_top = 17, margin_bottom = 18
  let main_w = 240, main_h = 134
  let text_h = 18
  let svg_w = main_w + margin_left + margin_right
  let svg_h = main_h + margin_top + margin_bottom
  let svg = d3.select(`#${svg_id}`)
    .attr('width', svg_w)
    .attr('height', svg_h)
  
  let pattern_defs = svg.append("defs")
  let err_highlight_pattern = pattern_defs.append("pattern")
    .attr("id", "err_highlight_pattern")
    .attr("patternUnits", "userSpaceOnUse")
    .attr("width", 5)
    .attr("height", 5)
  err_highlight_pattern.append("line")
    .attr("x1", 0)
    .attr("y1", 0)
    .attr("x2", 5)
    .attr("y2", 5)
    .attr("stroke", "#0097A7")
    .attr("stroke-width", 1)

  // define scale
  let xScale = d3.scaleLinear()
    .domain([0, bin_edges.length-1])
    .range([0, main_w])
  let yScale = d3.scaleLog()
    .domain([1, d3.max(all_used_cnts)])
    .range([main_h-text_h, 0])
  let rateScale = d3.scaleLinear()
    .domain([0, d3.max(all_process_rate)*1.1])
    .range([main_h-text_h, 0])

  let xAxis = d3.axisBottom(xScale)
    .ticks(bin_edges.length)
    .tickFormat((d) => bin_edges[d])
  const yAxis = d3.axisLeft(yScale)
    .ticks(5)
    .tickFormat((d) => {
      const expData = Math.exp(d)
      return d3.format("~s")(d)
    })
  let rateAxis = d3.axisRight(rateScale)
      // .tickValues(process_rate)
      .ticks(5)
  
  svg.append('text')
    .attr('x', 4)
    .attr('y', 12)
    .attr('text-anchor', 'start')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('#Predictions')
  svg.append('text')
    .attr('x', svg_w/2+margin_left/2-4)
    .attr('y', svg_h-6)
    .attr('text-anchor', 'middle')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('Residual Range')
  // svg.append('text')
  //   .attr('x', svg_w-1)
  //   .attr('y', 12)
  //   .attr('text-anchor', 'end')
  //   .style('font-size', '12px')
  //   .attr('fill', '#333')
  //   .text('Multiples')
    
  let hist_plot = svg.append('g').attr('class', 'hist-plot')
    .attr("transform", `translate(${margin_left},${margin_top})`)
  let xAxis_g = hist_plot.append("g")
    .attr("transform", `translate(0, ${main_h-text_h})`)
    .call(xAxis);
  // xAxis_g.selectAll("text")
  //   .attr("transform", "translate(8,6) rotate(45)")
  //   .style("text-anchor", "middle"); // 设置文本锚点位置
  let yAxis_g = hist_plot.append("g")
    // .attr("transform", `translate(0, 0)`)
    .call(yAxis);
  // if (valid_histSort[0] / valid_histSort[thBinId + 1] >= 5) {
  //   let rateAxis_g = hist_plot.append("g")
  //     .attr("transform", `translate(${main_w},0)`)
  //     .call(rateAxis)
  // }

  let bin_width = xScale(1) - xScale(0)
  if (Object.keys(hists).length > 1) {
    bin_width /= 2
  }

  for (let k = 0; k < Object.keys(hists).length; ++k) {
    let key = Object.keys(hists)[k]
    let bins_g = hist_plot.append('g')

    bins_g.selectAll('rect')
      .data(used_cnts[key])
      .join('rect')
        .attr('transform', `translate(${k*bin_width}, 0)`)
        .attr('x', (d,i) => xScale(i)+1*(1-k))
        .attr('y', (d,i) => yScale(d))
        .attr('bin_id', (d,i) => i)
        .attr('width', (d,i) => bin_width-1)
        .attr('height', (d,i) => main_h-text_h-yScale(d))
        .attr('fill', (d,i) => {
          if (k == 0) {
            return "#0097A7"
          } else {
            return "url(#err_highlight_pattern)"
          }
        })
        .attr('opacity', (d,i) => {
          if (k == 0) {
            return 0.8
          } else {
            return 1
          }
        })
        .on('mouseover', (e,d) => {
          if (polarity == 'pos') {
            cur_pos_extreme_hover_val.value = hists[key][scope_id][d3.select(e.target).attr('bin_id')]
            d3.select('#pos-extreme-tooltip')
              .style('left', e.clientX+'px')
              .style('top', e.clientY+'px')
              .style('opacity', 1)
          } else {
            cur_neg_extreme_hover_val.value = hists[key][scope_id][d3.select(e.target).attr('bin_id')]
            d3.select('#neg-extreme-tooltip')
              .style('left', e.clientX+'px')
              .style('top', e.clientY+'px')
              .style('opacity', 1)
          }
        })
        .on('mouseout', (e,d) => {
          if (polarity == 'pos') {
            cur_pos_extreme_hover_val.value = -1
            d3.select('#pos-extreme-tooltip')
              .style('opacity', 0)
          } else {
            cur_neg_extreme_hover_val.value = -1
            d3.select('#neg-extreme-tooltip')
              .style('opacity', 0)
          }
        })
    
    // if (valid_histSort[0] / valid_histSort[thBinId + 1] >= 5) {
    //   let rete_marks = hist_plot.append('g')
    //   rete_marks.append('g').selectAll('line')
    //     .data(process_rate[key])
    //     .join('line')
    //       .attr('x1', (d,i) => xScale(i+0.5*k)+1*(1-k)).attr('x2', (d,i) => xScale(i+0.5*k)+bin_width-1+1*(1-k))
    //       .attr('y1', (d,i) => rateScale(d)).attr('y2', (d,i) => rateScale(d))
    //       .attr('stroke', (d,i) => {
    //         if (k == 0) {
    //           if (d>0) {
    //             // if (i < 2 || i > bin_edges.length-3) return '#fff'
    //             // else return '#666'
    //             return '#fff'
    //           } 
    //           else return 'none'
    //         } else {
    //           if (d>0) return '#333'
    //           else return 'none'
    //         }
    //       })
    //   rete_marks.append('g').selectAll('text')
    //     .data(process_rate[key])
    //     .join('text')
    //       .attr('x', (d,i) => xScale(i+0.5*k)+1*(1-k)+(bin_width-1)/2)
    //       .attr('y', (d,i) => rateScale(d)+4)
    //       .attr('text-anchor', 'middle')
    //       .attr('fill', (d,i) => {
    //         if (k == 0) {
    //           if (d>0) {
    //             // if (i < 2 || i > bin_edges.length-3) return '#fff'
    //             // else return '#666'
    //             return '#fff'
    //           } 
    //           else return 'none'
    //         } else {
    //           if (d>0) return '#333'
    //           else return 'none'
    //         }
    //       })
    //       .text('x')
    // }
  }
}

function drawErrHistEval(scope_id) {
  let residual_bins = inforStore.error_distributions.all_residual_bins
  let mid_bins = inforStore.error_distributions.all_mid_bins
  let pos_extreme_bins = inforStore.error_distributions.all_pos_extreme_bins
  let neg_extreme_bins = inforStore.error_distributions.all_neg_extreme_bins
  
  let residual_hists = inforStore.error_distributions.all_residual_hists
  let mid_hists = inforStore.error_distributions.all_mid_hists
  let pos_extreme_hists = inforStore.error_distributions.all_pos_extreme_hists
  let neg_extreme_hists = inforStore.error_distributions.all_neg_extreme_hists

  drawMidHist(residual_bins, residual_hists, scope_id)
  // // drawMidHist(mid_bins, mid_hists)
  drawExtremeHist(pos_extreme_bins, pos_extreme_hists, scope_id, 'pos')
  drawExtremeHist(neg_extreme_bins, neg_extreme_hists, scope_id, 'neg')

  // drawOriginalMidHist(residual_bins, residual_hists, scope_id)
  // drawMidHist(mid_bins, mid_hists)
  // drawLogExtremeHist(pos_extreme_bins, pos_extreme_hists, scope_id, 'pos')
  // drawLogExtremeHist(neg_extreme_bins, neg_extreme_hists, scope_id, 'neg')
}

function drawErrDisLegend() {
  let main_h = 10, main_w = 260, margin_left = 1, margin_right = 1, margin_bottom = 1, margin_top = 1
  let svg_h = main_h + margin_bottom + margin_top
  let svg_w = main_w + margin_left + margin_right
  
  let cur_svg_id = 'err-dis-legend'
  d3.select(`#${cur_svg_id}`).selectAll('*').remove()
  let svg = d3.select(`#${cur_svg_id}`)
    .attr('width', svg_w).attr('height', svg_h)
  let main_g = svg.append('g')
    .attr('transform', `translate(${margin_left+5*main_h}, ${margin_top})`)
  
  // 绘制颜色高亮
  if (inforStore.cur_baseline_model.length > 0) {
    // 绘制斜条纹和方块
    let textureLegend = main_g.append('g')
      .attr('transform', `translate(${main_h*2}, 0)`)
    textureLegend.append('rect')
      .attr('x', 0).attr('y', 0)
      .attr('width', main_h*1.4)
      .attr('height', main_h)
      .attr('fill', '#999')
      .attr('stroke', '#666')
    textureLegend.append('text')
      .attr('x', main_h*2)
      .attr('y', 9)
      .attr('text-anchor', 'start')
      .style('font-size', '12px')
      .attr('fill', '#333')
      .text('Focus')
    
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

    let textureLegend_base = main_g.append('g')
      .attr('transform', `translate(${main_h*14}, 0)`)
    textureLegend_base.append('rect')
      .attr('x', 0).attr('y', 0)
      .attr('width', main_h*1.4)
      .attr('height', main_h)
      .attr('fill', 'url(#err_pattern)')
      .attr('stroke', '#666')
    textureLegend_base.append('text')
      .attr('x', main_h*2)
      .attr('y', 9)
      .attr('text-anchor', 'start')
      .style('font-size', '12px')
      .attr('fill', '#333')
      .text('Baseline')
  }
}

let cur_mid_hover_val = ref('-1')
let cur_pos_extreme_hover_val = ref('-1')
let cur_neg_extreme_hover_val = ref('-1')

function formatNumber(num) {
  if (Math.abs(num) >= 1e6) {
      return (num / 1e6).toFixed(1) + 'M'; // 百万
  } else if (Math.abs(num) >= 1e3) {
      return (num / 1e3).toFixed(1) + 'K'; // 千
  } else {
      return num.toString(); // 小于千的数值保持原样
  }
}

</script>

<template>
  <div style="margin-top: 10px;">
    <svg id="err-dis-legend" width="1" height="1" ></svg>
    <div style="margin-left: 5px; margin">
      <div><div class="attr-title">Multi-Scope Residual Box Plot</div></div>
      <svg id="scope-err-boxes"></svg>
    </div>
    <div class="err-dis-block">
      <div class="title-line"><div class="attr-title">Normal Residual Distribution</div></div>
      <svg class="err-dis-svg" id="mid-dis"></svg>
    </div>
    <div class="err-dis-block">
      <div class="title-line"><div class="attr-title">Extreme Positive Residual Distribution</div></div>
      <svg class="err-dis-svg" id="extreme-pos-dis"></svg>
    </div>
    <div class="err-dis-block">
      <div class="title-line"><div class="attr-title">Extreme Negative Residual Distribution</div></div>
      <svg class="err-dis-svg" id="extreme-neg-dis"></svg>
    </div>
    <div v-if="cur_mid_hover_val != -1" id="mid-hist-tooltip">{{ formatNumber(cur_mid_hover_val) }}</div>
    <div v-if="cur_pos_extreme_hover_val != -1" id="pos-extreme-tooltip">{{ formatNumber(cur_pos_extreme_hover_val) }}</div>
    <div v-if="cur_neg_extreme_hover_val != -1" id="neg-extreme-tooltip">{{ formatNumber(cur_neg_extreme_hover_val) }}</div>
  </div>
</template>

<style scoped>
.title-line {
  margin-left: 5px;
}

.attr-title {
  font-weight: 700;
}

.err-dis-block {
  margin-top: 6px;
}

#mid-hist-tooltip,
#pos-extreme-tooltip,
#neg-extreme-tooltip {
  /* width: 440px; */
  position: absolute;
  padding: 10px;
  background-color: #fff;
  border: 1px solid #999;
  border-radius: 5px;
  pointer-events: none;
  opacity: 0;
}

</style>