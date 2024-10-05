<script setup>
import { ref, reactive, computed, watch, onMounted, onUpdated } from "vue"
import { useInforStore } from '@/stores/infor.js'
import getData from '@/services/index.js'
import * as d3 from 'd3'
import { valColorScheme_blue, valColorScheme_red } from '@/data/index.js'
import SubSetsRadviz from './SubSetsRadviz.vue'
// import SubSetsProjection from './SubSetsProjection.vue'
import RangeAttrs from './RangeAttrs.vue'
import AttrDistributions from "./AttrDistributions.vue"
import RelatedSubsets from './RelatedSubsets.vue'

const inforStore = useInforStore()

const hist_view_id = (attr, id) => `hist-${attr}-${id}`
const attr_axis_id = attr => `table-axis-${attr}`
const all_subset_types = ref(['Point', 'Object'])
const all_subset_scopes = ref(['All Data', 'All Phases', 'Selected Phase'])

let ini_attr_distributions = {}
watch (() => inforStore.cur_subsets, (oldValue, newValue) => {
  for (let attr in inforStore.cur_range_infor) {
    ini_attr_distributions[attr] = inforStore.cur_range_infor[attr].filter((value, index) => !value['agg_range']).map(item => item.size)
  }
  inforStore.attr_distributions = ini_attr_distributions

  // 这里根据cur_focus_subset，去筛选掉
  inforStore.filtered_subsets = inforStore.cur_subsets.filter((value, index) => {
    let subset_id = value['subset_id']
    if (subset_id.includes('-')) {
      let split_id = subset_id.split('-')
      if (split_id[0] == inforStore.cur_focus_subset) return true
      else return false
    } else return true
  })
  
  // console.log(inforStore.filtered_subsets);
  // drawErrHistAxis()
  // for (let attr of inforStore.cur_st_attrs) {
  //   drawAttrAxis(attr)
  // }
})

watch (() => inforStore.currentIndex, (oldValue, newValue) => {
  if (inforStore.currentIndex == 0) go_back_valid.value = false
  else go_back_valid.value = true
  if (inforStore.currentIndex == inforStore.focused_subsets_list.length - 1) go_forward_valid.value = false
  else go_forward_valid.value = true
})

function goForward() {
    if (inforStore.currentIndex < inforStore.focused_subsets_list.length - 1) {
        inforStore.currentIndex += 1;
        inforStore.cur_focus_subset = inforStore.focused_subsets_list[inforStore.currentIndex]
    } else {
        console.log('Already at the last record.');
    }
}

// 后退到上一条记录
function goBackward() {
    if (inforStore.currentIndex > 0) {
        inforStore.currentIndex -= 1;
        inforStore.cur_focus_subset = inforStore.focused_subsets_list[inforStore.currentIndex]
    } else {
        console.log('Already at the first record.');
    }
}

watch (() => inforStore.cur_focus_subset, (oldValue, newValue) => {
  if (!inforStore.browsed_subsets.includes(inforStore.cur_focus_subset)) {
    inforStore.browsed_subsets.push(inforStore.cur_focus_subset)
  }
  
  let all_subset_ids = inforStore.filtered_subsets.map(item => item.subset_id)
  let subset_index = all_subset_ids.indexOf(inforStore.cur_focus_subset)

  scrollToIndex(subset_index)

  if (inforStore.cur_focus_subset == -1) {
    $('.subset-row').removeClass('subset-selected')
    inforStore.cur_attr_bin_errors = {}
  } else {
    if (inforStore.cur_focus_subset.includes('-')) {
      let split_id = inforStore.cur_focus_subset.split('-')
    } else {
      inforStore.cur_attr_bin_errors = {}
      $('.subset-row').removeClass('subset-selected')
      $(`.subset-row[subset_id=${inforStore.cur_focus_subset}]`).addClass('subset-selected')
    }

    getData(inforStore, 'attr_bin_errors',  inforStore.cur_sel_data, JSON.stringify(inforStore.cur_sel_models), inforStore.cur_focus_subset, JSON.stringify(inforStore.dataset_configs), JSON.stringify(inforStore.forecast_scopes))
  }

})


watch (() => inforStore.filtered_subsets, (oldValue, newValue) => {
  d3.selectAll('.subset-hist-view').selectAll('*').remove()
  for (let i = 0; i < inforStore.filtered_subsets.length; ++i) {
    for (let attr of inforStore.cur_st_attrs) {
      drawSubsetHist(inforStore.filtered_subsets[i].subset_id, attr)
    }
  }
})

watch (() => inforStore.select_ranges.length, (oldValue, newValue) => {
  filterSubsets()
})

watch (() => inforStore.cur_attr_bin_errors, (oldValue, newValue) => {
  // for (let attr of inforStore.cur_st_attrs) {
  //   let svg_id = hist_view_id(inforStore.meta_attr_objs[attr].simple_str, inforStore.cur_focus_subset)
  //   d3.select(`#${svg_id}`).selectAll('*').remove()
  //   drawSubsetHist(inforStore.cur_focus_subset, attr)
  // }
  if (inforStore.cur_focus_subset == -1 || Object.keys(inforStore.cur_attr_bin_errors).length > 0) {
    d3.selectAll('.subset-hist-view').selectAll('*').remove()
    for (let i = 0; i < inforStore.filtered_subsets.length; ++i) {
      for (let attr of inforStore.cur_st_attrs) {
        drawSubsetHist(inforStore.filtered_subsets[i].subset_id, attr)
      }
    }
  }
})

const onFilterErrTHChange = () => {
  filterSubsets()
}

function isArraySuperset(superset, subset) {
  return subset.every(value => superset.includes(value));
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

onUpdated(() => {
  d3.selectAll('.err-hist-svg').selectAll('*').remove()
  for (let i = 0; i < inforStore.filtered_subsets.length; ++i) {
    drawErrHist('normal', i)
    drawErrHist('extreme_pos', i)
    drawErrHist('extreme_neg', i)
    drawSubgroupSizeBar(i)
    drawErrorAbsBar(i)
  }
  if (inforStore.cur_focus_subset == -1 || Object.keys(inforStore.cur_attr_bin_errors).length > 0) {
    d3.selectAll('.subset-hist-view').selectAll('*').remove()
    for (let i = 0; i < inforStore.filtered_subsets.length; ++i) {
      for (let attr of inforStore.cur_st_attrs) {
        drawSubsetHist(inforStore.filtered_subsets[i].subset_id, attr)
      }
    }
  }
  if (inforStore.filtered_subsets.length > 0){
    for (let attr of inforStore.cur_st_attrs) {
      drawAttrAxis(attr)
    }
    drawSubgroupSizeAxis()
    drawErrorAbsAxis()
    drawErrHistAxis('normal')
    drawErrHistAxis('extreme_pos')
    drawErrHistAxis('extreme_neg')
    drawAbsResiLenged()
  }
})

const bin_width = 12

const drawErrHistAxis = (err_type) => {
  let val_bins, svg_id
  // 先判断err_type
  if (err_type == 'normal') {
    val_bins = inforStore.error_distributions.all_residual_bins
    svg_id = 'err-hist-svg-axis'
  }
  else if (err_type == 'extreme_pos') {
    val_bins = inforStore.error_distributions.all_pos_extreme_bins
    svg_id = 'extreme-pos-hist-axis'
  }
  else if (err_type == 'extreme_neg') {
    val_bins = inforStore.error_distributions.all_neg_extreme_bins
    svg_id = 'extreme-neg-hist-axis'
  }

  if (inforStore.cur_baseline_model.length == 0)
    val_bins = val_bins
  let width = (val_bins.length+1) * bin_width
  let height = 20
  let main_width = (val_bins.length-1) * bin_width
  let svg = d3.select(`#${svg_id}`)
    .attr('width', width)
    .attr('height', height)
  svg.selectAll('*').remove()
  let xScale = d3.scaleLinear()
    .domain([0, val_bins.length-1])
    .range([0, main_width])
  let xAxis = d3.axisTop(xScale)
    .ticks(val_bins.length)
    .tickFormat(d => val_bins[d])
    .tickSize(2.5)
  let axis_g = svg.append('g')
    .attr('id',  `axis_err_hist`)
    .attr('transform', `translate(${bin_width}, 19)`)
    .call(xAxis)
    .selectAll("text")
      .style("text-anchor", "middle")
      .style('font-size', '8px')
      .attr("dx", ".8em")
      .attr("dy", ".1em")
      .attr("transform", "rotate(-45)");
}

const drawAttrAxis = attr => {
  let bin_edges = inforStore.meta_attr_objs[attr].bin_edges
  let main_width = (bin_edges.length-1) * bin_width
  let simple_attr_str = inforStore.meta_attr_objs[attr].simple_str
  // console.log('simple_attr_str', simple_attr_str);
  let svg_id = attr_axis_id(simple_attr_str)
  let svg = d3.select(`#${svg_id}`)
  svg.selectAll('*').remove()

  let xScale = d3.scaleLinear()
    .domain([0, bin_edges.length-1])
    .range([0, main_width])
  let xAxis = d3.axisTop(xScale)
    .ticks(bin_edges.length)
    .tickFormat(d => {
      const bin_edge = bin_edges[d]
      if (bin_edges[d] > 1000) return d3.format("~s")(bin_edge)
      else return bin_edges[d]
    })
    .tickSize(2.5)
  let axis_g = svg.append('g')
    .attr('id',  `axis_${simple_attr_str}`)
    .attr('transform', `translate(${bin_width}, 19)`)
    .call(xAxis)
    .selectAll("text")
      .style("text-anchor", "middle")
      .style('font-size', '8px')
      .attr("dx", ".8em")
      .attr("dy", ".1em")
      .attr("transform", "rotate(-45)");
}

const drawSubsetHist = (subset_id, attr) => {
  // let cur_indices = inforStore.filtered_subsets[subset_id].indices
  // let cur_attr_data = inforStore.meta_attr_objs[attr].clean
  // let cur_attr_subset = cur_attr_data.filter((value, index) => cur_indices.includes(index))
  let filtered_subset_ids = inforStore.filtered_subsets.map(item => item.subset_id)
  let subset_index = filtered_subset_ids.indexOf(subset_id)
  let cur_attr_hist = inforStore.filtered_subsets[subset_index].attrs_hist[attr]
  let bin_edges = inforStore.meta_attr_objs[attr].bin_edges
  let width = (bin_edges.length-1)*bin_width
  let height = 24
  const margin = { top: 0, right: 0, bottom: 0, left: 0 };
  const main_width = width - margin.left - margin.right
  const main_height = height - margin.top - margin.bottom
  let simple_attr_str = inforStore.meta_attr_objs[attr].simple_str
  let svg_id = hist_view_id(simple_attr_str, subset_id)
  d3.select(`#${svg_id}`).selectAll('*').remove()
  let svg = d3.select(`#${svg_id}`)
    .attr('width', width)
    .attr('height', height)
  // svg.selectAll('*').remove()
  if (inforStore.meta_attr_objs[attr].bins.length == 0) return
  const subset_hist_g = svg.append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  const xScale = d3.scaleLinear()
    .domain([0, bin_edges.length-1])
    .range([0, main_width]);
  const yScale = d3.scaleLinear()
    .domain([0, d3.max(cur_attr_hist)])
    // .nice()
    .range([main_height, 0])
  // let focusErrScale
  // if (inforStore.cur_focus_subset != -1) {
  //   focusErrScale = d3.scaleSequential(d3.interpolateOranges)
  //     .domain([0, 400])
  // }
  // const bin_width = xScale(1) - xScale(0)
  let width_rate = 1
  let prev_model = inforStore.cur_focused_model
  let post_model = ""
  if (inforStore.cur_baseline_model.length > 0) {
    width_rate = 0.5
    prev_model = inforStore.cur_baseline_model
    post_model = inforStore.cur_focused_model
  }

  subset_hist_g.selectAll("rect")
    .data(cur_attr_hist)
    .join("rect")
    .attr("x", (d,i) => xScale(i) + bin_width*0.05)
    .attr("y", d => yScale(d))
    .attr("width", bin_width*0.9)
    .attr("height", d => main_height-yScale(d))
    .attr("fill", (d,i) => {
      if (subset_id == inforStore.cur_focus_subset) {
        let cur_err = inforStore.cur_attr_bin_errors[attr][i][prev_model]
        if (cur_err > inforStore.err_abs_extreme_th) return inforStore.extreme_err_color_scale(cur_err)
        else return inforStore.mild_err_color_scale(cur_err)
      }
      else return "#cecece"
    })
  
  if ((inforStore.cur_baseline_model.length > 0) && (inforStore.cur_focus_subset == subset_id)) {
    const subset_hist_comp_g = svg.append("g")
      .attr("transform", `translate(${margin.left+3},${margin.top})`);
    subset_hist_comp_g.selectAll("rect")
      .data(cur_attr_hist)
      .join("rect")
      .attr("x", (d,i) => xScale(i) + bin_width*0.05)
      .attr("y", d => yScale(d))
      .attr("width", bin_width*0.9 - 6)
      .attr("height", d => main_height-yScale(d))
      .attr("fill", (d,i) => {
        if (subset_id == inforStore.cur_focus_subset) {
          let cur_err = inforStore.cur_attr_bin_errors[attr][i][post_model]
          if (cur_err > inforStore.err_abs_extreme_th) return inforStore.extreme_err_color_scale(cur_err)
          else return inforStore.mild_err_color_scale(cur_err)
        }
        else return "#cecece"
      })
      .attr('stroke', '#666')
  }
  // console.log('frequent_sup_th', inforStore.preds_num, inforStore.dataset_configs.slice_params.frequent_sup_th);
  // console.log('hist sup_num', support_num, d3.max(cur_attr_hist));
  if (inforStore.cur_focus_subset == subset_id) {
    let support_num = inforStore.preds_num * inforStore.dataset_configs.slice_params.frequent_sup_th
    let sup_line  = subset_hist_g.append('line')
      .attr('x1', xScale(0))
      .attr('x2', xScale(main_width))
      .attr('y1', yScale(support_num))
      .attr('y2', yScale(support_num))
      .attr('stroke', '#999')
      .attr('stroke-dasharray', '4,4')
  }
}

const drawErrHist = (err_type, index) => {
  let val_bins, bin_means, hist_type
  // 先判断err_type
  if (err_type == 'normal') {
    val_bins = inforStore.error_distributions.all_residual_bins
    bin_means = inforStore.filtered_subsets[index].residual_hist_mean
    hist_type = 'residual_hist'
  }
  else if (err_type == 'extreme_pos') {
    val_bins = inforStore.error_distributions.all_pos_extreme_bins
    bin_means = inforStore.filtered_subsets[index].pos_extreme_hist_mean
    hist_type = 'pos_extreme_hist'
  }
  else if (err_type == 'extreme_neg') {
    val_bins = inforStore.error_distributions.all_neg_extreme_bins
    bin_means = inforStore.filtered_subsets[index].neg_extreme_hist_mean
    hist_type = 'neg_extreme_hist'
  }

  let width = (val_bins.length-1) * bin_width
  let height = 24
  let margin_l = 0, margin_r = 0, margin_t = 0, margin_b = 0
  let main_w = width - margin_l - margin_r
  let main_h = height - margin_t - margin_b
  let cur_subset = inforStore.filtered_subsets[index]
  let svg_id = errHistSvgID(err_type, index)
  let svg = d3.select('#' + svg_id)
    .attr('width', width)
    .attr("height", height)
  
  // 定义x，y轴的scale
  let hist_max = d3.max(cur_subset[hist_type])

  let width_rate = 1
  if (inforStore.cur_baseline_model.length > 0) {
    let all_hists = [...cur_subset[hist_type], ...cur_subset[`${hist_type}_comp`]]
    hist_max = d3.max(all_hists)
  }
  let xScale = d3.scaleLinear()
    .domain([0, val_bins.length-1])
    .range([0, main_w])
  let yScale = d3.scaleLinear()
    .domain([0, hist_max])
    .range([0, main_h])
  let err_hist = svg.append('g')
    .attr("transform", `translate(${margin_l},${margin_t})`);
  // let errColorScale = d3.scaleSequential(d3.interpolatePuOr)
  //     .domain([0, d3.max()])

  // let bin_w = xScale(1) - xScale(0)
  err_hist.selectAll('rect')
    .data(cur_subset[hist_type])
    .join('rect')
      .attr('x', (d,i) => xScale(i)+bin_width*0.05)
      .attr('y', (d,i) => main_h - yScale(d))
      .attr('width', width_rate * bin_width*0.9)
      .attr('height', (d,i) => yScale(d))
      .attr('fill', (d,i) => {
        if (Math.abs(bin_means[i]) > inforStore.err_abs_extreme_th) return inforStore.extreme_err_color_scale(Math.abs(bin_means[i]))
        else return inforStore.mild_err_color_scale(Math.abs(bin_means[i]))
      })
      .attr('stroke', 'none')
  
  if (inforStore.cur_baseline_model.length > 0) {
    let err_hist_comp = svg.append('g')
      .attr("transform", `translate(${margin_l+3},${margin_t})`);
    err_hist_comp.selectAll('rect')
      .data(cur_subset[`${hist_type}_comp`])
      .join('rect')
        .attr('x', (d,i) => xScale(i)+bin_width*0.05)
        .attr('y', (d,i) => main_h - yScale(d))
        .attr('width', width_rate * bin_width*0.9 - 6)
        .attr('height', (d,i) => yScale(d))
        .attr('fill', (d,i) => {
          if (Math.abs(bin_means[i]) > inforStore.err_abs_extreme_th) return inforStore.extreme_err_color_scale(Math.abs(bin_means[i]))
          else return inforStore.mild_err_color_scale(Math.abs(bin_means[i]))
        })
        .attr('stroke', '#666')
  }
  
  let zero_line = err_hist.append('line')
    .attr('x1', xScale(val_bins.indexOf(0)))
    .attr('x2', xScale(val_bins.indexOf(0)))
    .attr('y1', 0)
    .attr('y2', main_h)
    .attr('stroke', valColorScheme_red[1])
    .attr('stroke-dasharray', '4,4')
  if (inforStore.cur_focus_subset == cur_subset.subset_id) {
    let support_num = inforStore.preds_num * inforStore.dataset_configs.slice_params.frequent_sup_th
    let sup_line  = err_hist.append('line')
      .attr('x1', xScale(0))
      .attr('x2', xScale(main_w))
      .attr('y1', yScale(support_num))
      .attr('y2', yScale(support_num))
      .attr('stroke', '#999')
      .attr('stroke-dasharray', '4,4')
  }
}

function drawErrorLegends() {
  inforStore.extreme_err_color_scale
}



const errHistSvgID = (err_type, index) => `err-hist-${err_type}-${index}`
const col_width = attr => {
  let bin_edges = inforStore.meta_attr_objs[attr].bin_edges
  return (bin_edges.length+1) * bin_width
}

function CollectSubset(subset_id, subset_infor) {
  let index = inforStore.subset_collections.indexOf(subset_id);
  if (index > -1) {
    inforStore.subset_collections.splice(index, 1);
  } else {
    inforStore.subset_collections.push(subset_id)
  }
}

function allCollectSubsetsChosen() {
  let cur_filter_ids = inforStore.filtered_subsets.map(item => item.subset_id)
  return cur_filter_ids.every(item => inforStore.subset_collections.includes(item))
}

function allAugSubsetsChosen() {
  let cur_filter_ids = inforStore.filtered_subsets.map(item => item.subset_id)
  return cur_filter_ids.every(item => inforStore.cur_aug_subsets.includes(item))
}

function collectAllSubsets() {
  let cur_filter_ids = inforStore.filtered_subsets.map(item => item.subset_id)
  if (cur_filter_ids.every(item => inforStore.subset_collections.includes(item))) {
    const set_ids = new Set(cur_filter_ids);
    inforStore.subset_collections = inforStore.subset_collections.filter(item => !set_ids.has(item));
  } else {
    inforStore.subset_collections = [...new Set([...inforStore.subset_collections, ...cur_filter_ids])];
  }
}

function enhanceAllSubsets() {
  let cur_filter_ids = inforStore.filtered_subsets.map(item => item.subset_id)
  if (cur_filter_ids.every(item => inforStore.cur_aug_subsets.includes(item))) {
    const set_ids = new Set(cur_filter_ids);
    inforStore.cur_aug_subsets = inforStore.cur_aug_subsets.filter(item => !set_ids.has(item));
  } else {
    inforStore.cur_aug_subsets = [...new Set([...inforStore.cur_aug_subsets, ...cur_filter_ids])];
  }
}

function SelectSubset(subset_id, subset_infor) {
  if (subset_id == inforStore.cur_focus_subset) {
    inforStore.cur_focus_subset = -1
  } else {
    if (subset_id.includes('-')) {
      let split_id = subset_id.split('-')
    } else {
      inforStore.cur_focus_subset = subset_id
      inforStore.focused_subsets_list = inforStore.focused_subsets_list.slice(0, inforStore.currentIndex + 1)
      inforStore.focused_subsets_list.push(inforStore.cur_focus_subset)
      inforStore.currentIndex = inforStore.focused_subsets_list.length-1
    }
  }
}


function AddMixupSubset(subset_id, subset_infor) {
  let index = inforStore.cur_aug_subsets.indexOf(subset_id);

  if (index > -1) {
    inforStore.cur_aug_subsets.splice(index, 1);
    inforStore.cur_aug_subsets_infor.splice(index, 1);
  } else {
    inforStore.cur_aug_subsets.push(subset_id)
    inforStore.cur_aug_subsets_infor.push(subset_infor);
  }
}

function DataAugmentation() {
  console.log('data_augmentation')
  inforStore.all_focus_conditions = []
  inforStore.cur_aug_subsets_infor.forEach(subset_infor => {
    if ("contain_subsets" in subset_infor) {
      subset_infor.contain_subsets.forEach(subsubset => {
        inforStore.all_.push(subsubset.range_val)
      });
    } else {
      inforStore.all_focus_conditions.push(subset_infor.range_val)
    }
  })

  getData(inforStore, 'data_augmentation_by_slices', inforStore.cur_sel_data, inforStore.cur_focused_model, JSON.stringify(inforStore.cur_aug_subsets), JSON.stringify(inforStore.all_focus_conditions), JSON.stringify(inforStore.aug_params))
}

const onSubsetTypeChange = item => {
  // value是自动变化的，这里需要
  inforStore.subset_type = item
}
const onSubsetScopeChange = item => {
  // value是自动变化的，这里需要
  inforStore.dataset_configs.slice_params.data_scope = item
}

function percentile(arr, p) {
    // 排序数组
    arr.sort(function(a, b) { return a - b; });

    // 计算分位数位置
    const index = (p / 100) * (arr.length - 1);
    const lowerIndex = Math.floor(index);
    const upperIndex = Math.ceil(index);

    // 如果 index 恰好是整数，直接取对应元素
    if (lowerIndex === upperIndex) {
        return arr[index];
    }

    // 否则插值计算
    const weight = index - lowerIndex;
    return arr[lowerIndex] * (1 - weight) + arr[upperIndex] * weight;
}

function drawErrorAbsAxis() {
  // 获取subgroup size数据
  let all_error_abs = inforStore.cur_subsets.map(item => item.residual_abs)
  
  let main_width = 8 * bin_width
  let svg_id = "error-abs-svg-axis"
  let svg = d3.select(`#${svg_id}`)
  svg.selectAll('*').remove()
  svg.attr('width', main_width+6)

  let xScale = d3.scaleLinear()
    .domain([0, d3.max(all_error_abs)])
    .range([0, main_width])
  let xAxis = d3.axisTop(xScale)
    .ticks(6)
    .tickFormat(d3.format("~s"))
    .tickSize(2.5)
  let axis_g = svg.append('g')
    .attr('id',  `axis_subgroup_size`)
    .attr('transform', `translate(${bin_width}, 19)`)
    .call(xAxis)
    .selectAll("text")
      .style("text-anchor", "middle")
      .style('font-size', '8px')
      .attr("dx", ".8em")
      .attr("dy", ".1em")
      .attr("transform", "rotate(-45)");
}

function drawSubgroupSizeAxis() {
  // 获取subgroup size数据
  let subgroup_sizes = inforStore.cur_subsets.map(item => item.sup_num)
  let pos_res_nums = inforStore.cur_subsets.map(item => item.pos_res_num)
  let neg_res_num = inforStore.cur_subsets.map(item => item.neg_res_num)
  
  let main_width = 8 * bin_width
  let svg_id = "subgroup-size-svg-axis"
  let svg = d3.select(`#${svg_id}`)
  svg.selectAll('*').remove()
  svg.attr('width', main_width+6)

  let xScale = d3.scaleLinear()
    .domain([0, percentile(subgroup_sizes, 50)])
    .range([0, main_width])
  let xAxis = d3.axisTop(xScale)
    .ticks(6)
    .tickFormat(d3.format("~s"))
    .tickSize(2.5)
  let axis_g = svg.append('g')
    .attr('id',  `axis_subgroup_size`)
    .attr('transform', `translate(${bin_width}, 19)`)
    .call(xAxis)
    .selectAll("text")
      .style("text-anchor", "middle")
      .style('font-size', '8px')
      .attr("dx", ".8em")
      .attr("dy", ".1em")
      .attr("transform", "rotate(-45)");
}

function drawErrorAbsBar(index) {
  let all_error_abs = inforStore.cur_subsets.map(item => item.residual_abs)
  let all_error_abs_comp
  let cur_error_abs = inforStore.filtered_subsets[index].residual_abs
  let main_width = 8 * bin_width
  let bar_height = 10
  let svg_id = hist_view_id('error_abs', index)
  let svg = d3.select(`#${svg_id}`)
  svg.selectAll('*').remove()
  svg.attr('width', main_width)
    .attr('height', () => {
      if (inforStore.cur_baseline_model.length > 0) return 1.5 * bar_height
      else return bar_height
    })

  let xScale
  xScale = d3.scaleLinear()
    .domain([0, d3.max(all_error_abs)])
    .range([0, main_width])
  if (inforStore.cur_baseline_model.length > 0) {
    all_error_abs_comp = inforStore.cur_subsets.map(item => item.residual_abs_comp)
    let all_errors = [...all_error_abs, ...all_error_abs_comp]
    xScale = d3.scaleLinear()
      .domain([0, d3.max(all_errors)])
      .range([0, main_width])
  }

  if (inforStore.cur_baseline_model.length > 0) {
    bar_height *= 0.7
  }
  

  let bar_g = svg.append('g')
    .attr('transform', `translate(${bin_width}, 0)`)
  bar_g.append('rect')
    .attr('x', 0).attr('y', 0)
    .attr('width', xScale(cur_error_abs))
    .attr('height', bar_height)
    .attr('fill', () => {
      if (cur_error_abs > inforStore.err_abs_extreme_th) {
        return inforStore.extreme_err_color_scale(cur_error_abs)
      } else {
        return inforStore.mild_err_color_scale(cur_error_abs)
      }
    })
  if (inforStore.cur_baseline_model.length > 0) {
    let cur_error_comp_abs = inforStore.filtered_subsets[index].residual_abs_comp
    let bar_comp_g = svg.append('g')
    .attr('transform', `translate(${bin_width}, 0)`)
    bar_comp_g.append('rect')
      .attr('x', 0.5).attr('y', bar_height+1)
      .attr('width', xScale(cur_error_comp_abs))
      .attr('height', bar_height)
      .attr('fill', () => {
        if (cur_error_comp_abs > inforStore.err_abs_extreme_th) {
          return inforStore.extreme_err_color_scale(cur_error_comp_abs)
        } else {
          return inforStore.mild_err_color_scale(cur_error_comp_abs)
        }
      })
      .attr('stroke', '#666')
  }
}

function drawSubgroupSizeBar(index) {
  let subgroup_sizes = inforStore.cur_subsets.map(item => item.sup_num)
  let cur_subgroup_size = inforStore.filtered_subsets[index].sup_num
  let cur_pos_res_num = inforStore.filtered_subsets[index].pos_res_num
  let cur_neg_res_num = inforStore.filtered_subsets[index].neg_res_num
  let main_width = 8 * bin_width
  let bar_height = 10
  let svg_id = hist_view_id('subgroup_size', index)
  let svg = d3.select(`#${svg_id}`)
  svg.selectAll('*').remove()
  svg.attr('width', main_width)
    .attr('height', bar_height)

  let size_percentile = percentile(subgroup_sizes, 50)
  let xScale = d3.scaleLinear()
    .domain([0, size_percentile])
    .range([0, main_width])

  let pattern_defs = svg.append("defs")
  let pos_pattern = pattern_defs.append("pattern")
    .attr("id", "pos_pattern")
    .attr("patternUnits", "userSpaceOnUse")
    .attr("width", 5)
    .attr("height", 5)
  pos_pattern.append("line")
    .attr("x1", 0)
    .attr("y1", 5)
    .attr("x2", 5)
    .attr("y2", 0)
    .attr("stroke", "currentColor")
    .attr("stroke-width", 1)
  let neg_pattern = pattern_defs.append("pattern")
    .attr("id", "neg_pattern")
    .attr("patternUnits", "userSpaceOnUse")
    .attr("width", 5)
    .attr("height", 5)
  neg_pattern.append("line")
    .attr("x1", 0)
    .attr("y1", 0)
    .attr("x2", 5)
    .attr("y2", 5)
    .attr("stroke", "currentColor")
    .attr("stroke-width", 1)
  
  let bar_g = svg.append('g')
    .attr('transform', `translate(${bin_width}, 0)`)
  bar_g.append('rect')
    .attr('x', 0).attr('y', 0)
    .attr('width', xScale(cur_subgroup_size))
    .attr('height', bar_height)
    .attr('fill', '#0097A7')
  if (cur_subgroup_size > size_percentile) {
    bar_g.append('circle')
      .attr('cx', main_width-1.6*bin_width).attr('cy', bar_height/2)
      .attr('r', 3)
      .attr('fill', '#333')
  }
}

function scrollToIndex(index) {
  // 获取容器和表格元素
  const dataTable = document.getElementById('subset-table');
  
  // 获取要滚动到的目标行
  const targetRow = dataTable.rows[index];

  // 如果目标行存在，计算它的相对偏移位置并滚动
  if (targetRow) {
      // 计算目标行相对于容器顶部的偏移量
      const offsetTop = targetRow.offsetTop;
      
      // 将容器的滚动条滚动到目标行的位置
      // if (Math.abs(offsetTop-dataTable.offsetTop) > 200) 
      document.getElementById('table-region').scrollTop = offsetTop-42;
  } else {
      console.error("Invalid index: no such row in the table.");
  }
}

let collection_shown = ref(false)
let augment_shown = ref(false)
let viewed_shown = ref(false)
let unviewed_shown = ref(false)

function showViewed() {
  viewed_shown.value = !viewed_shown.value
}

function showUnviewed() {
  unviewed_shown.value = !unviewed_shown.value
}

function showCollection() {
  collection_shown.value = !collection_shown.value
}

function showAugment() {
  augment_shown.value = !augment_shown.value
}

watch (() => viewed_shown.value, (oldValue, newValue) => {
  if (viewed_shown.value) {
    // let collected_subsets = inforStore.cur_subsets.filter(item => inforStore.browsed_subsets.includes(item.subset_id))
    let collected_subsets = inforStore.filtered_subsets.filter(item => inforStore.browsed_subsets.includes(item.subset_id))
    inforStore.filtered_subsets = collected_subsets
  } else {
    filterSubsets()
  }
})

watch (() => unviewed_shown.value, (oldValue, newValue) => {
  if (unviewed_shown.value) {
    // let collected_subsets = inforStore.cur_subsets.filter(item => !inforStore.browsed_subsets.includes(item.subset_id))
    let collected_subsets = inforStore.filtered_subsets.filter(item => !inforStore.browsed_subsets.includes(item.subset_id))
    inforStore.filtered_subsets = collected_subsets
  } else {
    filterSubsets()
  }
})

watch (() => collection_shown.value, (oldValue, newValue) => {
  if (collection_shown.value) {
    let collected_subsets = inforStore.cur_subsets.filter(item => inforStore.subset_collections.includes(item.subset_id))
    inforStore.filtered_subsets = collected_subsets
  } else {
    filterSubsets()
  }
})

watch (() => augment_shown.value, (oldValue, newValue) => {
  if (augment_shown.value) {
    let aug_subsets = inforStore.cur_subsets.filter(item => inforStore.cur_aug_subsets.includes(item.subset_id))
    console.log('aug_subsets', aug_subsets);
    inforStore.filtered_subsets = aug_subsets
  } else {
    filterSubsets()
  }
})

function saveSubgroupCollection() {
  if (inforStore.cur_baseline_model.length == 0) {
    getData(inforStore, 'save_subgroup_collections', inforStore.cur_sel_data, inforStore.cur_focused_model, JSON.stringify(inforStore.subset_collections), JSON.stringify(inforStore.dataset_configs))
  } else {
    getData(inforStore, 'save_subgroup_collections', inforStore.cur_sel_data, inforStore.cur_baseline_model, JSON.stringify(inforStore.subset_collections), JSON.stringify(inforStore.dataset_configs))
  }
}

function SubsetGeneration() {

}

function drawAbsResiLenged() {
  let main_h = 12, main_w = 210, margin_left = 15, margin_right = 1, margin_bottom = 0, margin_top = 0
  let svg_h = main_h + margin_bottom + margin_top
  let svg_w = main_w + margin_left + margin_right
  
  let cur_svg_id = 'abs-residual-legend'
  d3.select(`#${cur_svg_id}`).selectAll('*').remove()
  let svg = d3.select(`#${cur_svg_id}`)
    .attr('width', svg_w).attr('height', svg_h)
  let main_g = svg.append('g')
    .attr('transform', `translate(${margin_left}, ${margin_top})`)
  
  let resi_legend_len = 80
  let residual_legend = main_g.append('g')

  let blueScale = d3.scaleSequential()
    .domain([0,1])
    .interpolator(d3.interpolateBlues)
  let colorScale = d3.scaleSequential()
    .domain([0,1])
    .interpolator(d3.interpolateYlOrRd)
  let mild_err_color = d3.scaleSequential()
    .domain([0, resi_legend_len])
    .interpolator(d3.interpolate(
      blueScale(0.0),
      blueScale(0.7)
    ));
  let extreme_err_color = d3.scaleSequential()
    .domain([0, resi_legend_len])
    .interpolator(d3.interpolate(
        colorScale(0.4),
        colorScale(1.0)
    ));
  
  residual_legend.append('text')
    .attr('x', -4)
    .attr('y', 10)
    .attr('text-anchor', 'end')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('0')
  residual_legend.append('g').selectAll('rect')
    .data(Array(resi_legend_len).fill(1))
    .join('rect')
      .attr('x', (d,i) => i)
      .attr('y', 0)
      .attr('width', 1)
      .attr('height', 12)
      .attr('fill', (d,i) => mild_err_color(i))

  residual_legend.append('text')
    .attr('x', resi_legend_len+1)
    .attr('y', 10)
    .attr('text-anchor', 'start')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text(inforStore.err_abs_extreme_th)

  residual_legend.append('g')
    .attr('transform', `translate(${resi_legend_len + 16})`)
    .selectAll('rect')
    .data(Array(resi_legend_len).fill(1))
    .join('rect')
      .attr('x', (d,i) => i)
      .attr('y', 0)
      .attr('width', 1)
      .attr('height', 12)
      .attr('fill', (d,i) => extreme_err_color(i))

  residual_legend.append('text')
    .attr('x', resi_legend_len * 2 + 19)
    .attr('y', 10)
    .attr('text-anchor', 'start')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('max')
}

let go_back_valid = ref(false)
let go_forward_valid = ref(false)

</script>

<template>
  <div class="models-container">
    <div class="title-layer">
      <div class="title">Subgroup Module</div>
      <!-- subset类型选择 -->
      <!-- <div class="data-dropdown">
        <label class="form-label"><span class="attr-title">Subset Type: </span></label>
        <button class="btn dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false">{{ inforStore.subset_type }}</button>
        <ul class="dropdown-menu">
          <li v-for="(item, index) in all_subset_types" :value="item" @click="onSubsetTypeChange(item)" class='dropdown-item' :key="index">
            <div class="li-data-name">{{ item }}</div>
          </li>
        </ul>
      </div> -->
      <!-- subset范围选择 -->
      <!-- <div class="data-dropdown" id="subset-scope-dropdown" v-if="Object.keys(inforStore.dataset_configs) != 0">
        <label class="form-label"><span class="attr-title">Subset Scope: </span></label>
        <button class="btn dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false">{{ inforStore.dataset_configs.slice_params.data_scope }}</button>
        <ul class="dropdown-menu">
          <li v-for="(item, index) in all_subset_scopes" :value="item" @click="onSubsetScopeChange(item)" class='dropdown-item' :key="index">
            <div class="li-data-name">{{ item }}</div>
          </li>
        </ul>
      </div> -->
      <!-- <div id="subset-config-icon" class="iconfont" data-bs-toggle="collapse" data-bs-target="#subset-config-region" aria-controls="subset-config-region">&#xe8ca;</div> -->
      <!-- <div class="collapse" id="subset-config-region">
      </div> -->
      <div class="params-control">
        <label class="form-label"><span class="attr-title">Filter_Residual_TH: </span></label>
        <input class="form-control" type="text" v-model="inforStore.err_abs_th" @change="onFilterErrTHChange">
      </div>

      <div class="collection-control" style="font-weight:700" @click="showViewed()" :showing="viewed_shown">
        <span class="iconfont" style="font-size: 20px; margin-right: 3px;">&#xe624;</span> <span>Viewed ({{inforStore.browsed_subsets.length}})</span> 
      </div>

      <div class="collection-control" style="font-weight:700" @click="showUnviewed()" :showing="unviewed_shown">
        <span class="iconfont" style="font-size: 20px; margin-right: 3px;">&#xe626;</span> <span>Unviewed ({{inforStore.cur_subsets.length - inforStore.browsed_subsets.length}})</span> 
      </div>

      <div class="collection-control" style="font-weight:700" @click="showCollection()" :showing="collection_shown">
        <span class="iconfont" style="font-size: 22px; margin-right: 3px;">&#xe625;</span> <span>Collection ({{inforStore.subset_collections.length}})</span> 
      </div>
      <div class="iconfont" id='save-subgroup-collection' @click="saveSubgroupCollection()">&#xe63c;</div>
      
      <div class="collection-control" style="font-weight:700" @click="showAugment()" :aug_showing="augment_shown">
        <span class="iconfont" style="font-size: 22px;">&#xe625;</span> <span>Subgroups to Augment ({{inforStore.cur_aug_subsets.length}})</span> 
      </div>
      <div class="mixup-subsets">
        <!-- <div>Subsets to Mixup: {{ inforStore.cur_aug_subsets }}</div> -->
        <button type="button" class="btn btn-outline-primary data-aug-btn" @click="DataAugmentation()">Data Augmentation</button>
      </div>
      <div class="collection-control" style="font-weight:700" @click="showCollection()" :showing="collection_shown">
        <span>abs_residual: </span> 
        <svg id="abs-residual-legend" width="1" height="1"></svg>
      </div>
      <!-- <button type="button" class="btn btn-outline-primary subset-gen-btn" @click="SubsetGeneration()">Generate Subgroup</button> -->
    </div>
    <div v-if="Object.keys(inforStore.dataset_configs) != 0" style="display: flex;">
      <SubSetsRadviz />
      <AttrDistributions />
      <!-- <RangeAttrs /> -->
    </div>
    <div class="main-region" v-if="inforStore.cur_subsets.length > 0">
      <div class="block-title">
        Subgroup Table
        <div style="margin-left: 20px">
          <span class="iconfont explore-icon" id="go-back-icon" @click="goBackward()" :valid="go_back_valid">&#xe856;</span>
          <span class="iconfont explore-icon" id="go-forward-icon" @click="goForward()" :valid="go_forward_valid" >&#xe857;</span>
        </div>
      </div>
      <div id="table-region">
        <table class="table table-fixed-header subset-list" id="subset-table">
          <thead>
            <tr>
              <th scope="col">ID</th>
              <th scope="col">
                Actions
                <div style="margin-bottom: -10px;"> 
                  <span class="iconfont subset-action-icon" style="visibility: hidden;">&#xe624;</span>
                  <span class="iconfont subset-action-icon" :chosen="allCollectSubsetsChosen()" @click="collectAllSubsets()">&#xe61c;</span>
                  <span class="iconfont subset-action-icon" :chosen="allAugSubsetsChosen()" @click="enhanceAllSubsets()">&#xe658;</span>
                </div>
              </th>
              <th scope="col" v-for="(item, index) in inforStore.cur_st_attrs" :key="index">
                <div>
                  <span v-if="inforStore.meta_attr_objs[item].icon_type == 'value'" class="iconfont">&#xe618;</span>
                  <span v-if="inforStore.meta_attr_objs[item].icon_type == 'time'" class="iconfont">&#xe634;</span>
                  <span v-if="inforStore.meta_attr_objs[item].icon_type == 'space'" class="iconfont">&#xe647;</span>
                  {{ inforStore.meta_attr_objs[item].simple_str }}
                </div>
                <svg class="attr-axis" :id="attr_axis_id(inforStore.meta_attr_objs[item].simple_str)" :width="col_width(item)"></svg>
              </th>
              <th scope="col">
                Size
                <svg class="attr-axis" id="subgroup-size-svg-axis"></svg>
              </th>
              <!-- <th scope="col">sup_rate</th> -->
              <th scope="col">
                Error(abs)
                <svg class="attr-axis" id="error-abs-svg-axis"></svg>
              </th>
              <th scope="col">
                Res_Hist
                <svg class="attr-axis" id="err-hist-svg-axis"></svg>
              </th>
              <th scope="col">
                Ex_Pos_Res_Hist
                <svg class="attr-axis" id="extreme-pos-hist-axis"></svg>
              </th>
              <th scope="col">
                Ex_Neg_Res_Hist
                <svg class="attr-axis" id="extreme-neg-hist-axis"></svg>
              </th>
              <!-- <th scope="col">error_pos</th>
              <th scope="col">error_neg</th> -->
              <!-- error_abs
              error_pos
              error_neg -->
            </tr>
          </thead>
          <tbody>
            <!-- <div> -->
              <!-- <tr v-for="(item, index) in inforStore.filtered_subsets" :key="index" class="subset-row" @click="onSubsetClick(index, item)"> -->
              <tr v-for="(item, index) in inforStore.filtered_subsets" :key="index" class="subset-row" :subset_id="item.subset_id">
                <td>{{ item.subset_id }}</td>
                <td class="subset-actions">
                  <span class="iconfont subset-action-icon" :browsed="inforStore.cur_focus_subset!=item.subset_id && inforStore.browsed_subsets.includes(item.subset_id)" :chosen="inforStore.cur_focus_subset==item.subset_id" @click="SelectSubset(item.subset_id, item)">&#xe624;</span>
                  <span class="iconfont subset-action-icon" :chosen="inforStore.subset_collections.includes(item.subset_id)" @click="CollectSubset(item.subset_id, item)">&#xe61c;</span>
                  <span class="iconfont subset-action-icon" :chosen="inforStore.cur_aug_subsets.includes(item.subset_id)" @click="AddMixupSubset(item.subset_id, item)">&#xe658;</span>
                  
                </td>
                <td v-for="(attr, i) in inforStore.cur_st_attrs" :key="i">
                  <div v-if="(!item.hasOwnProperty('contain_subsets')) && item.range_val.hasOwnProperty(attr)">{{ item.range_val[attr] }}</div>
                  <div v-else><svg class="subset-hist-view" :id="hist_view_id(inforStore.meta_attr_objs[attr].simple_str, item.subset_id)"></svg></div>
                </td>
                <!-- <td>{{ item.neg_res_num }}/{{ item.pos_res_num }}</td> -->
                <!-- <td>{{ item.sup_rate }}</td> -->
                <td>
                  <div style="font-size: 14px; margin-bottom: -6px;">{{ item.sup_rate }}</div>
                  <svg class="subgroup-size-svg" :id="hist_view_id('subgroup_size', index)"></svg>
                  <!-- <svg class="sup_rate-svg" :id="hist_view_id('sup_rate', index)"></svg>
                  <svg class="pos_neg-svg" :id="hist_view_id('pos_neg', index)"></svg> -->
                </td>
                <td>
                  <div v-if="inforStore.cur_baseline_model.length > 0" style="font-size: 12px; margin-bottom: -6px;">{{ `${item.residual_abs.toFixed(2)} / ${item.residual_abs_comp.toFixed(2)}`}}</div>
                  <div v-if="inforStore.cur_baseline_model.length == 0" style="font-size: 14px; margin-bottom: -6px;">{{ item.residual_abs.toFixed(2) }}</div>
                  <svg class="subgroup-size-svg" :id="hist_view_id('error_abs', index)"></svg>
                </td>
                <td><svg class="err-hist-svg" :id="errHistSvgID('normal', index)"></svg></td>
                <td><svg class="err-hist-svg" :id="errHistSvgID('extreme_pos', index)"></svg></td>
                <td><svg class="err-hist-svg" :id="errHistSvgID('extreme_neg', index)"></svg></td>
                <!-- <td>{{ item.error_pos }}</td>
                <td>{{ item.error_neg }}</td> -->
              </tr>
            <!-- </div> -->
          </tbody>
        </table>
      </div>
    </div>

    <RelatedSubsets />
  </div>

</template>

<style scoped>
.models-container {
  width: 1604px;
  /* width: 1520px; */
  /* width: 1400px; */
  height: 1076px !important;
  border: solid 1px #c2c5c5;
  padding: 0;
  border-radius: 6px;
  /* padding: 1px; */
  margin: 2px;
}

#table-region {
  margin-top: 16px;
  overflow-y: auto;
  height: 424px;
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
  display: flex;
  align-items: center;
  /* justify-content: space-between; */
}

.title {
  font: 700 20px "Arial";
  /* letter-spacing: 1px; */
  color: #333;
  /* z-index: 99999999999; */
}

#subset-filter-region {
  /* height: 400px !important;
  width: 600px !important; */
  border: solid 1px #c2c5c5;
  border-radius: 6px;
  position: absolute;
  background-color: #fff;
  z-index: 999;
  padding: 0;
  margin-top: 460px;
  margin-left: -12px
  /* z-index: 999; */
  /* display: flex; */
}

.params-control {
  display: flex;
  margin-left: 20px;
}

.params-control .form-control {
  width: 50px;
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

.form-label {
  margin-bottom: 0;
  font-weight: 700;
}


/* .subset-list tr{
  display: flex;
  justify-content: center;
} */

.subset-row {
  width: 100px;
}

.subset-selected {
  /* background-color: #cad8e5 !important; */
  border: solid 2px #333;
}

.table td, .table th {
  text-align: center; /* 水平居中 */
  vertical-align: middle; /* 垂直居中 */
  padding: 8px 3px 8px 3px;
}

/* .table thead th {
  border-bottom-color: #999;
} */

.table-fixed-header thead {
  position: sticky;
  top: 0;
  background-color: white;
  /* z-index: 1000; */
}

.table-param-row {
  display: flex;
  align-items: center;
}

.err-hist-svg {
  display: block;
  margin: 0 auto;
}

.subset-hist-view {
  display: block;
  margin: 0 auto;
  height: 24px;
}

.attr-axis {
  display: block;
  margin: 0 auto;
  height: 20px;
}

/* .table-param-row {
  margin-left: 10px;
} */

.subset-action-icon {
  font-size: 20px;
  color: #999;
  margin: 0 3px;
  font-weight: 400 !important;
}
.subset-action-icon:hover {
  cursor: pointer;
  color: #0097A7;
}
.subset-action-icon[chosen=true] {
  color: #0097A7;
}

.subset-action-icon[browsed=true] {
  color: #6a51a3;
}

.data-aug-btn {
  margin-left: 8px;
  width: 130px!important;
  font-size: 14px;
  padding: 1px 1px;
}

.subset-gen-btn {
  font-size: 14px;
  width: 136px!important;
  margin-left: 16px;
  padding: 1px 1px;
}


#subset-filter-icon,
#subset-config-icon {
  margin-left: 16px;
  font-size: 22px;
  font-weight: 700;
  color: #999;
}

#subset-filter-icon:hover,
#subset-config-icon:hover {
  cursor: pointer;
  color: #1a73e8;
}



.data-dropdown {
  margin-left: 20px;
}

.data-dropdown .dropdown-toggle {
  width: 80px !important;
  height: 30px;
  padding: 0px 2px -30px 0px !important;
  margin-bottom: 10px;
  border-bottom: solid 1px #9c9c9c;
  color:#1a73e8;
  border-radius: 0;
  font-size: 14px;
  /* text-align: left; */
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

#subset-scope-dropdown .dropdown-toggle {
  width: 120px !important;
}

.main-region {
  margin-top: -30px;
}

.block-title {
  display: flex;
  align-items: center;
  font-size: 18px;
  font-weight: 700;
  color: #555;
  margin-left: 10px;
  margin-bottom: -18px;
}

.collection-control {
  margin-left: 24px;
  display: flex;
  align-items: center;
}

.collection-control[showing=true] {
  color: #0097A7;
}

.collection-control[aug_showing=true] {
  color: #0097A7;
}

#save-subgroup-collection:hover,
.collection-control:hover {
  cursor: pointer;
  /* color: #0097A7; */
  color: #1a73e8;
}
#save-subgroup-collection {
  font-size: 20px;
  margin-left:8px;
  font-weight: 700;
}

.explore-icon {
  font-size: 20px;
  margin-left: 8px;
  opacity: 1;
  color: #999;
}

.explore-icon:hover {
  cursor: pointer;
  color: #0097A7;
}

.explore-icon[valid=false] {
  opacity: 0;
}

</style>