<script setup>
import { ref, reactive, computed, watch, onMounted, onUpdated } from "vue"
import { useInforStore } from '@/stores/infor.js'
import getData from '@/services/index.js'
import * as d3 from 'd3'
import * as d3Sankey from 'd3-sankey'
import { valColorScheme_blue, valColorScheme_red, valColorScheme_fire } from '@/data/index.js'

const inforStore = useInforStore()

const select_ranges_whole = ref({})
let brush_extent = {}
let attr_bin_edges = {}
let ini_attr_distributions = {}

onMounted(() => {
  let cur_select_ranges = {}
  brush_extent = {}
  for (let attr of inforStore.dataset_configs.point_metadata) {
    cur_select_ranges[attr] = []
    // inforStore.attr_distributions[attr] = []
    select_ranges_whole.value[attr] = []
    brush_extent[attr] = [0, 0]
  }
  inforStore.select_ranges = cur_select_ranges
})

watch (() => inforStore.dataset_configs, (oldValue, newValue) => {
  let cur_select_ranges = {}
  brush_extent = {}
  for (let attr of inforStore.dataset_configs.point_metadata) {
    cur_select_ranges[attr] = []
    // inforStore.attr_distributions[attr] = []
    select_ranges_whole.value[attr] = []
    brush_extent[attr] = [0, 0]
  }
  inforStore.select_ranges = cur_select_ranges
})

watch (() => inforStore.cur_range_infor, (oldValue, newValue) => {
  for (let attr in inforStore.cur_range_infor) {
    ini_attr_distributions[attr] = inforStore.cur_range_infor[attr].filter((value, index) => !value['agg_range']).map(item => item.size)
    attr_bin_edges[attr] = inforStore.meta_attr_objs[attr]['bin_edges']
  }
  console.log('ini_attr_distributions', ini_attr_distributions);
})

function updateAttrDis() {
  if (Object.values(inforStore.select_ranges).every(value => value.length === 0)) {
    inforStore.attr_distributions = ini_attr_distributions
  } else {
    getData(inforStore, 'attr_distributions',  inforStore.cur_sel_data, JSON.stringify(inforStore.cur_sel_models), JSON.stringify(inforStore.dataset_configs), JSON.stringify(inforStore.forecast_scopes), JSON.stringify(attr_bin_edges), JSON.stringify(inforStore.select_ranges))
  }
}

watch (() => inforStore.attr_distributions, (oldValue, newValue) => {
  console.log('attr_distributions', inforStore.attr_distributions);
  filterSubsets()
})

const view_id = (view_str, view_id) => `${view_str}_${view_id}`
function findIndexByRange(arrayOfObjects, rangeArray) {
  for (let i = 0; i < arrayOfObjects.length; i++) {
    const objRange = arrayOfObjects[i].range;
    if (arraysAreEqual(objRange, rangeArray)) {
      return i; // 返回索引
    }
  }
  return -1; // 如果没有找到匹配的范围数组，则返回 -1
}
function arraysAreEqual(array1, array2) {
  for (let i = 0; i < array1.length; i++) {
    if (array1[i] !== array2[i]) {
      return false;
    }
  }
  return true;
}

function isArraySuperset(superset, subset) {
  return Object.values(subset).every(value => {
    if (value.length == 0) return true
    else return value.some(item => superset.includes(item))
  });
}

const filterSubsets = () => {
  let filtered_points
  filtered_points = inforStore.cur_subsets.filter((value, index) => {
    let subset_id = value['subset_id']
    if (subset_id.includes('-')) {
      let split_id = subset_id.split('-')
      if (split_id[0] == inforStore.cur_focus_subset) return true
      else return false
    } else return true
  })

  // 进行radviz中选中的匹配
  if (inforStore.sel_subset_points.length > 0) {
    filtered_points = inforStore.cur_subsets.filter((value, index) => inforStore.sel_subset_points.includes(index))
  }
  
  // 再进行字符串匹配筛选
  let matched_subsets = []
  if (Object.values(inforStore.select_ranges).every(value => value.length === 0)) matched_subsets = filtered_points
  else {
    for (let i = 0; i < filtered_points.length; ++i) {
      let isSuperset = isArraySuperset(filtered_points[i].subset_attrs, inforStore.select_ranges);
      if (isSuperset) matched_subsets.push(filtered_points[i])
    }
    // inforStore.filtered_subsets = matched_subsets
  }
  // 先进行阈值筛选
  let err_th = inforStore.err_abs_th
  let cur_filtered_subsets = matched_subsets.filter((obj) => obj.residual_abs >= err_th)
  inforStore.filtered_subsets = cur_filtered_subsets
}

let global_resi_max = ref(0)
watch (() => inforStore.filtered_subsets, (oldValue, newValue) => {
  global_resi_max.value = 0
  for (let attr in inforStore.cur_range_infor) {
    for (let i = 0; i < inforStore.cur_range_infor[attr].length; ++i) {
      inforStore.cur_range_infor[attr][i].cur_cnt = 0
      let cur_max = d3.max(inforStore.cur_range_infor[attr].map(item => item.abs_residual))
      if (cur_max > global_resi_max.value) global_resi_max.value = cur_max
    }
  }
  for (let i = 0; i < inforStore.filtered_subsets.length; ++i) {
    let range_val_obj = inforStore.filtered_subsets[i].range_val
    // console.log(i, range_val_obj)
    for (let attr in range_val_obj) {
      const range_id = findIndexByRange(inforStore.cur_range_infor[attr], range_val_obj[attr])
      inforStore.cur_range_infor[attr][range_id].cur_cnt += 1
    }
  }
  drawAttrDistribution()
})

onUpdated(() => {
  drawAttrDistribution()
})

function drawAttrDistribution() {
  let margin_left = 28, margin_right = 30, margin_top = 16, margin_bottom = 38
  let main_w = 190, main_h = 80
  let svg_w = main_w + margin_left + margin_right
  let svg_h = main_h + margin_top + margin_bottom
  for (let attr in inforStore.cur_range_infor) {
    // 获取相应的hist数据
    let range_cnts = inforStore.attr_distributions[attr]
    // console.log(attr, 'range_cnts', range_cnts);
    let range_errors = inforStore.cur_range_infor[attr].filter((value, index) => !value['agg_range']).map(item => item.abs_residual)
    let bin_edges = inforStore.meta_attr_objs[attr]['bin_edges']

    const histSort = [...range_cnts].sort((a, b) => b - a);
    const valid_histSort = histSort.filter(element => element !== 0);
    const threshold = d3.quantile(valid_histSort, 0.5);
    let thBinId = 0;
    for (let i = 0; i < valid_histSort.length - 1; i++) {
        if (valid_histSort[i] > threshold && valid_histSort[i + 1] <= threshold) {
            thBinId = i;
            break;
        }
    }
    let process_rate = Array(range_cnts.length).fill(0)
    // 自适应调整倍数
    const used_cnts = [...range_cnts]; // 复制 hist 数组
    if (valid_histSort[0] / valid_histSort[thBinId + 1] >= 5) {
        // 识别频次最高的 bin
        const maxFreq = valid_histSort[thBinId + 1];

        for (let i = 0; i < range_cnts.length; i++) {
            if (range_cnts[i] > threshold) { // high_freq_bins 的逻辑替换
                // 计算自适应调整倍数，使调整后频次不超过 maxFreq
                const adjustmentFactor = range_cnts[i] / maxFreq
                // const adjustmentFactor = Math.floor(range_cnts[i] / maxFreq)
                process_rate[i] = Math.floor(adjustmentFactor * 10) / 10
                used_cnts[i] = Math.floor(range_cnts[i] / adjustmentFactor)
            }
        }
    }

    let svg_id = view_id('attr-dis', inforStore.meta_attr_objs[attr].simple_str)
    d3.select(`#${svg_id}`).selectAll('*').remove()
    let svg = d3.select(`#${svg_id}`)
      .attr('width', svg_w)
      .attr('height', svg_h)
    let hist_plot = svg.append('g')
      .attr('transform', `translate(${margin_left}, ${margin_top})`)

    let xScale = d3.scaleLinear()
      .domain([0, used_cnts.length])
      .range([0, main_w])
    let yScale = d3.scaleLinear()
      .domain([0, d3.max(used_cnts)])
      .range([main_h, 0])
    let rateScale = d3.scaleLinear()
      .domain([0, d3.max(process_rate)*1.1])
      .range([main_h, 0])
    // let errScale = d3.scaleSequential(d3.interpolateOranges)
    //   .domain([0, d3.max(range_errors)])

    let brush = d3.brushX()
      .extent([[0, main_h+margin_top+2], [main_w, svg_h-2]])
      // .on('brush', brushMove)
      .on("end", brushEnded);
    let brush_g = svg.append("g")
      .attr('transform', `translate(${margin_left}, 0)`)
      .call(brush)
      // .call(brush.move, defaultExtent);
    
    brush_g.selectAll('.selection')
      .data([brush_extent[attr]])
      .join('rect')
      .attr('class', 'selection')
      .attr('x', d => d[0])
      .attr('y', main_h+margin_top+2)
      .attr('width', d => d[1] - d[0])
      .attr('height', margin_bottom-4)
      .style('display', 'block')
    
    // let brushing = false; // 添加一个标志位
    // function brushMove(e) {
    //   // console.log(e);
    //   if (e && !brushing) {
    //     brushing = true; // 设置标志位，防止递归调用
    //     let selection = e.selection;
    //     let step = yScale(0) - yScale(1)
    //     let x0 = Math.round(selection[0] / step) * step;
    //     let x1 = Math.round(selection[1] / step) * step;
    //     // 更新选择框的位置
    //     brush_g.call(brush.move, [x0, x1]);
    //     brushing = false;
    //   }
    // }

    function brushEnded(e) {
      let selection = e.selection;
      if (selection) {
        let step = xScale(1) - xScale(0)
        brush_extent[attr][0] = Math.round(selection[0] / step) * step;
        brush_extent[attr][1] = Math.round(selection[1] / step) * step; 
        let x0 = xScale.invert(selection[0]);
        let x1 = xScale.invert(selection[1]);
        let x0_int = parseInt(Math.round(x0))
        let x1_int = parseInt(Math.round(x1))
        inforStore.select_ranges[attr] = []
        for (let i = x0_int; i < x1_int; ++i) {
          inforStore.select_ranges[attr].push(inforStore.cur_range_infor[attr][i].range_str)
        }
        // console.log(inforStore.select_ranges);
        // filterSubsets()
        updateAttrDis()
      } else {
        inforStore.select_ranges[attr] = []
        brush_extent[attr] = [0, 0]
        updateAttrDis()
        // filterSubsets()
      }
    }
    // let err_comp_patterns
    // if (inforStore.cur_baseline_model.length > 0) {
    //   let pattern_defs = svg.append("defs")
    //   err_comp_patterns = pattern_defs.selectAll("pattern")
    //     .data(range_errors)
    //     .join("pattern")
    //     .attr("id", (d,i) => `err_pattern-${i}`)
    //       .attr("patternUnits", "userSpaceOnUse")
    //       .attr("width", 5)
    //       .attr("height", 5)
    //   err_comp_patterns.append("line")
    //     .attr("x1", 0)
    //     .attr("y1", 0)
    //     .attr("x2", 5)
    //     .attr("y2", 5)
    //     .attr("stroke", (d,i) => {
    //       if (d > inforStore.err_abs_extreme_th) return inforStore.extreme_err_color_scale(d)
    //       else return inforStore.mild_err_color_scale(d)
    //     })
    //     .attr("stroke-width", 1)
    // }
    
    let err_box = hist_plot.append('g')
    err_box.selectAll('rect')
      .data(range_errors)
      .join('rect')
        .attr('x', (d,i) => xScale(i)+1)
        .attr('y', main_h+2)
        .attr('width', xScale(1)-xScale(0)-2)
        .attr('height', 6)
        .attr('fill', (d,i) => {
          if (d > inforStore.err_abs_extreme_th) return inforStore.extreme_err_color_scale(d)
          else return inforStore.mild_err_color_scale(d)
          // if (inforStore.cur_baseline_model.length > 0) {
          //   return `url(#err_pattern-${i})`
          // } else {
          //   if (d > inforStore.err_abs_extreme_th) return inforStore.extreme_err_color_scale(d)
          //   else return inforStore.mild_err_color_scale(d)
          // }
        })
        // .attr('stroke', (d,i) => {
        //   if (inforStore.cur_baseline_model.length > 0) return 'none'
        //   else return '#666'
        // })
        // .attr('stroke-width', 1)
    if (inforStore.cur_baseline_model.length > 0) {
      let range_errors_comp = inforStore.cur_range_infor[attr].filter((value, index) => !value['agg_range']).map(item => item.abs_residual_comp)
      let err_comp_box = hist_plot.append('g')
      err_comp_box.selectAll('rect')
        .data(range_errors_comp)
        .join('rect')
          .attr('x', (d,i) => xScale(i)+1)
          .attr('y', main_h+2+6+1)
          .attr('width', xScale(1)-xScale(0)-2)
          .attr('height', 6)
          .attr('fill', (d,i) => {
            // return inforStore.mild_err_color_scale(d)
            if (d > inforStore.err_abs_extreme_th) return inforStore.extreme_err_color_scale(d)
            else return inforStore.mild_err_color_scale(d)
          })
          .attr('stroke', '#666')
          .attr('stroke-width', 1)
    }

    let xAxis = d3.axisBottom(xScale)
      .ticks(bin_edges.length)
      .tickFormat(d => {
        const bin_edge = bin_edges[d]
        return d3.format("~s")(bin_edge)
      })
    let yAxis = d3.axisLeft(yScale)
      .ticks(5)
      .tickFormat(d3.format("~s"))
    let rateAxis = d3.axisRight(rateScale)
      // .tickValues(process_rate)
      .ticks(5)
      // .tickFormat(d3.format("~s"))
    let xAxis_g = hist_plot.append("g")
      .attr("transform", `translate(0,${main_h})`)
      .call(xAxis)
    let yAxis_g = hist_plot.append("g").call(yAxis)
    if (valid_histSort[0] / valid_histSort[thBinId + 1] >= 5) {
      let rateAxis_g = hist_plot.append("g")
        .attr("transform", `translate(${main_w},0)`)
        .call(rateAxis)
    }
    

    xAxis_g.selectAll("text")
      .attr("transform", () => {
        if (inforStore.cur_baseline_model.length > 0) return "translate(8,12) rotate(45)"
        else "translate(8,6) rotate(45)"
      })
      .style("text-anchor", "middle"); // 设置文本锚点位置
    yAxis_g.selectAll("text")
      .attr("transform", "translate(-9,-6) rotate(-45)")
      .style("text-anchor", "middle"); // 设置文本锚点位置
    
    // let rate_labels = svg.append('g')
    // // svg.append()
    // let rate_texts = rate_labels.selectAll('text')
    //   .data(process_rate)
    //   .join('text')
    //     .attr('x', (d,i) => margin_left + xScale(i+0.5))
    //     // .attr('y', (d,i) => yScale(used_cnts[i]))
    //     .attr('y', 12)
    //     .attr('text-anchor', 'middle')
    //     // .attr("transform", (d,i) => `translate(${xScale(0.5)},0) rotate(-45, ${margin_left + xScale(i+0.5)}, 10)`)
    //     .style('font-size', '10px')
    //     .attr('fill', '#333')
    //     .text((d,i) => {
    //       if (d > 0) return d
    //       else return ''
    //     })
    
    svg.append('text')
      .attr('x', 0)
      .attr('y', 10)
      .attr('text-anchor', 'start')
      .style('font-size', '12px')
      .attr('fill', '#333')
      .text('#Predictions')
    
    svg.append('text')
      .attr('x', svg_w/2)
      .attr('y', () => {
        if (inforStore.cur_baseline_model.length > 0) return svg_h
        else return svg_h-4
      } )
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .attr('fill', '#333')
      .text('Attribute Value')
    
    let bin_bars = hist_plot.append('g')
    bin_bars.selectAll('rect')
      .data(inforStore.cur_range_infor[attr])
      .join('rect')
        .attr('x', (d,i) => xScale(i)+1)
        .attr('y', (d,i) => yScale(used_cnts[i]))
        .attr('bin_id', (d,i) => i)
        .attr('width', (d,i) => xScale(i+1) - xScale(i)-1)
        .attr('height', (d,i) => main_h-yScale(used_cnts[i]))
        .attr('fill', (d,i) => {
          // if (inforStore.cur_range_infor[attr][i].cur_cnt > 0) return "#3182bd"
          // if (inforStore.cur_range_infor[attr][i].cur_cnt > 0) return "#0097A7"
          if (inforStore.cur_range_infor[attr][i].cur_cnt > 0) return "#0097A7"
          else return "#999"
        })
        .attr('stroke', (d,i) => {
          if (inforStore.select_ranges[attr].includes(d.range_str)) return '#333'
          else return 'none'
        })
        .attr('stroke-width', 1.5)
        .attr('opacity', 0.8)
        .on('click', (e,d) => {
          let cur_range_str = d.range_str
          if (inforStore.select_ranges[attr].includes(cur_range_str)) {
            let cur_index = inforStore.select_ranges[attr].indexOf(cur_range_str)
            if (cur_index !== -1) inforStore.select_ranges[attr].splice(cur_index, 1)
          } else {
            inforStore.select_ranges[attr].push(cur_range_str)
          }
          updateAttrDis()
          // filterSubsets()
        })
        .on('mouseover', (e,d) => {
          cur_hover_attr_val.value = range_cnts[d3.select(e.target).attr('bin_id')]
          d3.select('#attr-hist-tooltip')
            .style('left', e.clientX+'px')
            .style('top', e.clientY+'px')
            .style('opacity', 1)
        })
        .on('mouseout', (e,d) => {
          cur_hover_attr_val.value = -1
          d3.select('#attr-hist-tooltip')
            .style('opacity', 0)
        })
    if (valid_histSort[0] / valid_histSort[thBinId + 1] >= 5) {
      svg.append('text')
        .attr('x', svg_w-6)
        .attr('y', 10)
        .attr('text-anchor', 'end')
        .style('font-size', '12px')
        .attr('fill', '#333')
        .text('Multiples')

      let rete_marks = hist_plot.append('g')
      rete_marks.append('g').selectAll('line')
        .data(process_rate)
        .join('line')
          .attr('x1', (d,i) => xScale(i)).attr('x2', (d,i) => xScale(i+1))
          .attr('y1', (d,i) => rateScale(d)).attr('y2', (d,i) => rateScale(d))
          .attr('stroke', (d,i) => {
            if (d>0) return '#fff'
            else return 'none'
          })
      rete_marks.append('g').selectAll('text')
        .data(process_rate)
        .join('text')
          .attr('x', (d,i) => xScale(i+0.5))
          .attr('y', (d,i) => rateScale(d)+4)
          .attr('text-anchor', 'middle')
          .attr('fill', (d,i) => {
            if (d>0) return '#fff'
            else return 'none'
          })
          .text('x')
    }
  }
}


function drawAttrDistributionLog() {
  let margin_left = 28, margin_right = 10, margin_top = 4, margin_bottom = 28
  let main_w = 190, main_h = 80
  let svg_w = main_w + margin_left + margin_right
  let svg_h = main_h + margin_top + margin_bottom
  for (let attr in inforStore.cur_range_infor) {
    // 获取相应的hist数据
    let range_cnts = inforStore.cur_range_infor[attr].map(item => item.size)
    let range_errors = inforStore.cur_range_infor[attr].map(item => item.abs_residual)
    let bin_edges = inforStore.meta_attr_objs[attr]['bin_edges']

    const epsilon = 1e-10
    const logCnts = range_cnts.map(f => Math.log(f + epsilon))

    let used_cnts
    if (inforStore.dataset_configs.distribution_params.log_transform) used_cnts = logCnts
    else used_cnts = range_cnts

    let svg_id = view_id('attr-dis', inforStore.meta_attr_objs[attr].simple_str)
    d3.select(`#${svg_id}`).selectAll('*').remove()
    let svg = d3.select(`#${svg_id}`)
      .attr('width', svg_w)
      .attr('height', svg_h)
    let hist_plot = svg.append('g')
      .attr('transform', `translate(${margin_left}, ${margin_top})`)

    let xScale = d3.scaleLinear()
      .domain([0, used_cnts.length])
      .range([0, main_w])
    let yScale = d3.scaleLinear()
      .domain([0, d3.max(used_cnts)])
      .range([main_h, 0])

    let brush = d3.brushX()
      .extent([[0, main_h+margin_top+2], [main_w, svg_h-2]])
      .on('brush', brushMove)
      .on("end", brushEnded);
    let brush_g = svg.append("g")
      .attr('transform', `translate(${margin_left}, 0)`)
      .call(brush)
      // .call(brush.move, defaultExtent);
    let brushing = false; // 添加一个标志位

    function brushMove(e) {
      // console.log(e);
      if (e && !brushing) {
        brushing = true; // 设置标志位，防止递归调用
        let selection = e.selection;
        let step = xScale(1) - xScale(0)
        let x0 = Math.floor(selection[0] / step) * step;
        let x1 = Math.ceil(selection[1] / step) * step;
        // 更新选择框的位置
        brush_g.call(brush.move, [x0, x1])
        brushing = false;
      }
    }
    
    function brushEnded(e) {
      let selection = e.selection;
      if (selection) {
        let x0 = xScale.invert(selection[0]);
        let x1 = xScale.invert(selection[1]);
        let x0_int = parseInt(Math.round(x0))
        let x1_int = parseInt(Math.round(x1))
        for (let i = x0_int; i < x1_int; ++i) {
          inforStore.select_ranges[attr].push(inforStore.cur_range_infor[attr][i].range_str)
        }
        console.log(inforStore.select_ranges);
        filterSubsets()
      } else {
        inforStore.select_ranges[attr] = []
        filterSubsets()
      }
    }

    let xAxis = d3.axisBottom(xScale)
      .ticks(bin_edges.length)
      .tickFormat((d) => bin_edges[d])
    let yAxis = d3.axisLeft(yScale)
      .ticks(5)
      .tickFormat(d => {
        const expData = Math.exp(d)
        if (inforStore.dataset_configs.distribution_params.log_transform) return d3.format("~s")(expData)
        else return d3.format("~s")
      })
    let xAxis_g = hist_plot.append("g")
      .attr("transform", `translate(0,${main_h})`)
      .call(xAxis)
    let yAxis_g = hist_plot.append("g").call(yAxis)
    xAxis_g.selectAll("text")
      .attr("transform", "translate(8,6) rotate(45)")
      .style("text-anchor", "middle"); // 设置文本锚点位置
    let bin_bars = hist_plot.append('g')
    bin_bars.selectAll('rect')
      .data(inforStore.cur_range_infor[attr])
      .join('rect')
        .attr('x', (d,i) => xScale(i)+1)
        .attr('y', (d,i) => yScale(used_cnts[i]))
        .attr('width', (d,i) => xScale(i+1) - xScale(i)-1)
        .attr('height', (d,i) => main_h-yScale(used_cnts[i]))
        .attr('fill', (d,i) => {
          if (inforStore.cur_range_infor[attr][i].cur_cnt > 0) return "#3182bd"
          else return "#bcbcbc"
        })
        .attr('stroke', (d,i) => {
          if (inforStore.select_ranges[attr].includes(d.range_str)) return '#333'
          else return 'none'
        })
        .attr('stroke-width', 1.5)
        .attr('opacity', 0.8)
        .on('click', (e,d) => {
          let cur_range_str = d.range_str
          console.log(cur_range_str);
          if (inforStore.select_ranges[attr].includes(cur_range_str)) {
            let cur_index = inforStore.select_ranges[attr].indexOf(cur_range_str)
            if (cur_index !== -1) inforStore.select_ranges[attr].splice(cur_index, 1)
          } else {
            inforStore.select_ranges[attr].push(cur_range_str)
          }
          // filterSubsets()
          updateAttrDis()
        })
  }
}

let cur_hover_attr_val = ref(-1)
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
  <div>
    <div class="block-title">Attribute Distribution</div>
    <div class="attr-range-region">
      <div v-for="(value, key, index) in inforStore.meta_attr_objs" class="attr-dis-block" :key="index">
        <div class="attr-label">
          <span v-if="inforStore.meta_attr_objs[key].icon_type == 'value'" class="iconfont attr-label-icon">&#xe618; </span>
          <span v-if="inforStore.meta_attr_objs[key].icon_type == 'time'" class="iconfont attr-label-icon">&#xe634; </span>
          <span v-if="inforStore.meta_attr_objs[key].icon_type == 'space'" class="iconfont attr-label-icon">&#xe647; </span>
          <span class="attr_simple_str">{{ value.simple_str }}</span>
        </div>
        <svg class="attr-dis-svg" :id="view_id('attr-dis', value.simple_str)"></svg>
      </div>
    </div>
    <div v-if="cur_hover_attr_val != -1" id="attr-hist-tooltip">{{ formatNumber(cur_hover_attr_val) }}</div>
  </div>
</template>

<style scoped>
.attr-range-region {
  width: 860px;
  height: 336px;
  margin: 0 auto;
  margin-top: -4px;
  margin-left: 0px;
  overflow-x: auto;
  margin-bottom: 12px;
  display: flex;
  flex-wrap: wrap;
}

.block-title {
  font-size: 18px;
  font-weight: 700;
  color: #555;
  margin-left: 6px;
  margin-bottom: 4px;
}

.attr-dis-block {
  margin-bottom: 1px;
}

.attr-label-icon {
  font-size: 18px;
  color: #333;
}

.attr_simple_str {
  font-size: 14px;
  font-weight: 700;
  /* color: #333; */
}

.attr-dis-svg {
  margin-right: 20px;
  margin-bottom: 6px;
}

#attr-hist-tooltip {
  position: absolute;
  padding: 10px;
  background-color: #fff;
  border: 1px solid #999;
  border-radius: 5px;
  pointer-events: none;
  opacity: 0;
}

.subset-gen-btn {
  margin-left: 8px;
  width: 130px!important;
  font-size: 14px;
  padding: 1px 2px;
}
</style>