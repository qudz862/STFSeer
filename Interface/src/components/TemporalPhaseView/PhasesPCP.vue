<script setup>
import { ref, reactive, computed, watch, onMounted, onUpdated } from "vue"
import { useInforStore } from '@/stores/infor.js'
import getData from '@/services/index.js'
import * as d3 from 'd3'
import { valColorScheme_blue, valColorScheme_red, valColorScheme_fire } from '@/data/index.js'

const inforStore = useInforStore()
let cur_filter_conditions = ref({})

onMounted(() => {
  drawPhasesPCP()
})

onUpdated(() => {
  // drawPhasesPCP()
})

watch (() => inforStore.phase_pcp_dims, (oldValue, newValue) => {
  console.log('inforStore.phase_pcp_dims', inforStore.phase_pcp_dims);
})


let focused_phase_attr = ref('life_span')

let dim_objs = {}

function drawPhasesPCP() {
  d3.select('#phases-pcp-svg').selectAll('*').remove()
  let margin_left = 40, margin_right = 30, margin_top = 19, margin_bottom = 7
  let main_w = 1520, main_h = 110
  let svg_w = main_w + margin_left + margin_right
  let svg_h = main_h + margin_top + margin_bottom
  let pcp_svg = d3.select('#phases-pcp-svg')
    .attr('width', svg_w)
    .attr('height', svg_h)
    .append("g")
    .attr("transform", `translate(${margin_left},${margin_top})`)
  
  let phases_objs = inforStore.st_phase_events.phases[inforStore.cur_focused_model]
  
  phases_objs.forEach(obj => {
    obj['selected'] = 0
  })

  // 为每个轴定义scale
  let yScales = {}
  console.log('phase_pcp_dims', Object.keys(inforStore.phase_pcp_dims), inforStore.phase_pcp_dims);
  Object.keys(inforStore.phase_pcp_dims).forEach(dim => {
    console.log(dim);
    cur_filter_conditions.value[dim] = []
    let cur_dim_val = phases_objs.map(item => item[dim])
    // console.log('cur_dim_val', cur_dim_val);
    dim_objs[dim] = {}
    let dim_range = d3.extent(cur_dim_val)
    // dim_range[0] = Math.floor(dim_range[0]/dim_objs[dim].step)*dim_objs[dim].step
    // dim_range[1] = Math.ceil(dim_range[1]/dim_objs[dim].step)*dim_objs[dim].step
    dim_objs[dim].range = dim_range
    dim_objs[dim].bin_edges = inforStore.phase_pcp_dims[dim]
    dim_objs[dim].bin_edges[dim_objs[dim].bin_edges.length-1] += 1
    dim_objs[dim].bins = Array.from({length: dim_objs[dim].bin_edges.length-1}, (v,k) => [dim_objs[dim].bin_edges[k], dim_objs[dim].bin_edges[k+1]])
    yScales[dim] = d3.scaleLinear()
      .domain([0, dim_objs[dim].bins.length])
      .range([main_h, 0])
  })

  // 基于颜色划分的范围，根据hist的bin构建堆叠图，数据就是每个bin中包含的颜色数量

  // 设置不同范围的colorbar
  // let stack_colors = ['#f0f0f0', '#d9d9d9', '#bdbdbd', '#969696', '#737373', '#525252', '#252525', '#000000']
  // let stack_colors = ['#8e0152', '#c51b7d', '#de77ae', '#f1b6da', '#b8e186', '#7fbc41', '#4d9221', '#276419']
  // let stack_colors = ['#67001f', '#b2182b', '#d6604d', '#f4a582', 
  // '#fddbc7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061']
  // let stack_colors = ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#92c5de', '#4393c3', '#2166ac', '#053061']
  // let stack_colors = ['#8c510a', '#bf812d', '#dfc27d', '#f6e8c3', '#c7eae5', '#80cdc1', '#35978f', '#01665e']
  // let stack_colors = ['#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4']
  // let stack_colors = ['#b35806', '#e08214', '#fdb863', '#fee0b6', '#d8daeb', '#b2abd2', '#8073ac', '#542788']

  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  //原来：
  // let stack_colors = ['#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac']
  let stack_colors = ['#b2182b','#d6604d','#f4a582','#fddbc7','#e0e0e0','#bababa','#878787','#4d4d4d']

  // let binColorScale = d3.scaleQuantize()
  //   .domain(dim_objs[focused_phase_attr.value].range)
  //   .range(stack_colors)
  let binColorScale = d3.scaleQuantize()
    .domain([0, dim_objs[focused_phase_attr.value].bins.length])
    .range(stack_colors)
  let binValScale = d3.scaleQuantize()
    .domain(dim_objs[focused_phase_attr.value].range)
    .range(d3.range(stack_colors.length))
  let stackColorScale = d3.scaleOrdinal()
    .domain(d3.range(stack_colors.length))
    .range(stack_colors);

  let all_stack_data = {}
  for (let key in dim_objs) {
    all_stack_data[key] = []
    for (let i = 0; i < dim_objs[key].bins.length; ++i) {
      all_stack_data[key].push([])
      for (let j = 0; j < stack_colors.length; ++j) {
        all_stack_data[key][i].push(0)
      }
    }
    for (let i = 0; i < phases_objs.length; ++i) {
      let phase_obj = phases_objs[i]
      let bin_idx = -1 
      for (let j = 0; j < dim_objs[key].bins.length; ++j) {
        if ((phase_obj[key] >= dim_objs[key].bins[j][0]) && (phase_obj[key] < dim_objs[key].bins[j][1])) {
          bin_idx = j
          break
        }
      }
      let focus_bin_idx = -1
      for (let j = 0; j < dim_objs[focused_phase_attr.value].bins.length; ++j) {
        if ((phase_obj[focused_phase_attr.value] >= dim_objs[focused_phase_attr.value].bins[j][0]) && (phase_obj[focused_phase_attr.value] < dim_objs[focused_phase_attr.value].bins[j][1])) {
          focus_bin_idx = j
          break
        }
      }
      // all_stack_data[key][bin_idx][stack_colors.indexOf(binColorScale(phase_obj[focused_phase_attr.value]))] += 1
      all_stack_data[key][bin_idx][focus_bin_idx] += 1
    }
  }

  let attr_series = {}
  let attr_stack = d3.stack()
    .keys(d3.range(stack_colors.length))
    .order(d3.stackOrderNone)
    .offset(d3.stackOffsetNone);
  for (let key in dim_objs) {
    attr_series[key] = attr_stack(all_stack_data[key])
  }

  let tmp_dims = Object.keys(inforStore.phase_pcp_dims).slice()
  tmp_dims.push('tmp_dim')
  let xScale = d3.scalePoint()
    .domain(tmp_dims)
    .range([0, main_w])
  let axisInterval = xScale(Object.keys(inforStore.phase_pcp_dims)[1]) - xScale(Object.keys(inforStore.phase_pcp_dims)[0])

  // 添加轴标签
  let textElements = pcp_svg.selectAll(".axis-label")
    .data(Object.keys(inforStore.phase_pcp_dims))
    .enter().append("text")
    .attr("id", (d,i) => `axis-label-${i}`)
    .attr("transform", d => `translate(${xScale(d)}, -10)`)
    .text(d => d)
    .attr('text-anchor', 'middle')
    .style('font-size', '12px')
    .style('font-weight', (d,i) => d == focused_phase_attr.value ? 900 : 400)
    // .attr('stroke', (d,i) => d == focused_phase_attr.value ? valColorScheme_fire[4] : '#333')
    .style('cursor', 'pointer')
    .on('click', (e,d) => {
      d3.selectAll('.axis-label').style('font-weight', 400)
      // d3.selectAll('.axis-label').attr('stroke', '#333')
      d3.select(e.target).style('font-weight', 900)
      // d3.select(e.target).attr('stroke', valColorScheme_fire[4])
      focused_phase_attr.value = d
      drawPhasesPCP()
    })

  // 绘制轴的堆叠图
  Object.keys(inforStore.phase_pcp_dims).forEach(dim => {
    let axis_hist_g = pcp_svg.append('g')
      .attr('transform', `translate(${xScale(dim), 0})`)
    console.log(xScale(dim));
    let cur_y_step = 1.0*main_h / dim_objs[dim].bins.length
    const stackXScale = d3.scaleLinear()
      .domain([0, d3.max(attr_series[dim], d => d3.max(d, d => d[1]))])
      .range([0, axisInterval*0.8]);
    let axis_hist = axis_hist_g.selectAll("g")
      .data(attr_series[dim])
      .join("g")
        .attr('stack_bin_id', (d ,i) => `${i}`)
        .attr("fill", (d, i) => stackColorScale(i))
        .selectAll("rect")
        .data(d => d)
        .join("rect")
          .attr('class', 'stack-rect')
          .attr('attr_bin_id', (d, i) => `${i}`)
          .attr("x", (d, i) => xScale(dim) + stackXScale(d[0]))
          .attr("y", (d,i) => yScales[dim](i)-cur_y_step)
          .attr('stroke', 'none')
          .attr("height", cur_y_step)
          .attr("width", d => stackXScale(d[1] - d[0]))
          .on('click', (e, d) => {
            let cur_rect = d3.select(e.target)
            if (cur_rect.attr('stroke') != 'none'){
              cur_rect.attr('stroke', 'none')
              pcp_svg.selectAll(".line").remove()
              let all_phases_indices = phases_objs.map((item,index) => index)
              inforStore.cur_filtered_phases = phases_objs.slice()
              inforStore.cur_filtered_phases_indices = all_phases_indices.slice()
              return
            }
            d3.selectAll('.stack-rect').attr('stroke', 'none')
            cur_rect.attr('stroke', 'black')
            let attr_bin_id = parseInt(cur_rect.attr('attr_bin_id'))
            let stack_bin_id = parseInt(d3.select(cur_rect.node().parentNode).attr('stack_bin_id'))
            // console.log(attr_bin_id, stack_bin_id);
            phases_objs.forEach(obj => {
              obj.selected = 0
              let cur_attr_bin = dim_objs[dim].bins[attr_bin_id]
              let cur_focus_bin = dim_objs[focused_phase_attr.value].bins[stack_bin_id]
              let cur_focus_val = obj[focused_phase_attr.value]
              // console.log(cur_attr_bin, cur_focus_bin);
              if (obj[dim] >= cur_attr_bin[0] && obj[dim] < cur_attr_bin[1] && cur_focus_val >= cur_focus_bin[0] && cur_focus_val < cur_focus_bin[1]) {
                obj.selected = 1
              }
            })
            let all_phases_indices = phases_objs.map((item,index) => index)
            let filtered_phase_objs = phases_objs.filter(item => item.selected == 1)
            console.log('filtered_phase_objs', filtered_phase_objs);
            let filtered_phase_indices = all_phases_indices.filter(index => phases_objs[index].selected == 1)
            inforStore.cur_filtered_phases = filtered_phase_objs
            inforStore.cur_filtered_phases_indices = filtered_phase_indices
            // console.log(inforStore.cur_filtered_phases)
            // inforStore.cur_filtered_events = inforStore.cur_filtered_phases.reduce((accu, obj) => accu.concat(obj.evolution_events), [])
            pcp_svg.selectAll(".line").remove()
            lines.data(filtered_phase_objs)
              .join('path')
              .attr("class", "line")
              .attr("d", d => d3.line()(Object.keys(inforStore.phase_pcp_dims).map(dim => {
                let dim_bins = dim_objs[dim].bins
                let y_id = -1
                for (let i = 0; i < dim_bins.length; ++i) {
                  if (d[dim] < dim_bins[i][1]) {
                    y_id = i
                    break
                  }
                }
                let cur_y_step = 1.0*main_h / dim_objs[dim].bins.length
                return [xScale(dim), yScales[dim](y_id-0.5)-cur_y_step]
              })))  
              .attr("fill", 'none')
              .attr("stroke", (d,i) => '#ababab')
              .attr("stroke-width", 2)
              .attr("opacity", 0.9)
          })
  })

  // 添加路径
  let sel_phases_objs = phases_objs.filter(item => item.selected == 1)
  let lines = pcp_svg.selectAll(".line")
    .data(sel_phases_objs)
    .join("path")
    .attr("class", "line")
    .attr("d", d => d3.line()(Object.keys(inforStore.phase_pcp_dims).map(dim => {
      let dim_bins = dim_objs[dim].bins
      let y_id = -1
      for (let i = 0; i < dim_bins.length; ++i) {
        if (d[dim] < dim_bins[i][1]) {
          y_id = i
          break
        }
      }
      let cur_y_step = 1.0*main_h / dim_objs[dim].bins.length
      return [xScale(dim), yScales[dim](y_id-0.5)-cur_y_step]
    })))  
    .attr("fill", 'none')
    .attr("stroke", (d,i) => '#ababab')
    .attr("stroke-width", 2)
    .attr("opacity", 0.9)

  // 创建刷选器
  const brushes = {}
  const brushes_g = {}
  const brush_layer_g = pcp_svg.append('g')
  let brush_w = 28
  let brushing = false; // 添加一个标志位
  Object.keys(inforStore.phase_pcp_dims).forEach(dim => {
    brushes[dim] = d3.brushY()
      .extent([[xScale(dim)-brush_w, 0], [xScale(dim)+brush_w, main_h]])
      .on('brush', (e) => {
        let cur_dim = dim
        if (e && !brushing) {
          brushing = true; // 设置标志位，防止递归调用
          let selection = e.selection
          let step = yScales[cur_dim](0) - yScales[cur_dim](1)
          let y0 = Math.floor(selection[0] / step) * step
          let y1 = Math.ceil(selection[1] / step) * step
          // 更新选择框的位置
          brushes_g[cur_dim].call(brushes[cur_dim].move, [y0, y1])
          brushing = false
        }
      })
      .on("end", (e) => {
        let selection = e.selection;
        let cur_dim = dim
        if (selection) {
          let y0 = yScales[cur_dim].invert(selection[0])
          let y1 = yScales[cur_dim].invert(selection[1])
          let y0_int = parseInt(Math.round(y0))
          let y1_int = parseInt(Math.round(y1))
          cur_filter_conditions.value[cur_dim] = [dim_objs[cur_dim].bin_edges[y1_int], dim_objs[cur_dim].bin_edges[y0_int]] 
          // 在这里可以执行你的操作，例如根据范围重新渲染图表等
        } else {
          cur_filter_conditions.value[cur_dim] = []
        }
        let sel_cnt = 0
        phases_objs.forEach(obj => {
          obj.selected = 1
          for (let dim in cur_filter_conditions.value) {
            if (cur_filter_conditions.value[dim] == []) continue
            let cur_condition = cur_filter_conditions.value[dim]
            if (obj[dim] < cur_condition[0] || obj[dim] > cur_condition[1]) {
              obj.selected = 0
              break
            }
          }
          sel_cnt += obj.selected
        })
        let all_phases_indices = phases_objs.map((item,index) => index)
        let filtered_phase_objs = phases_objs.filter(item => item.selected == 1)
        let filtered_phase_indices = all_phases_indices.filter(index => phases_objs[index].selected == 1)
        inforStore.cur_filtered_phases = filtered_phase_objs
        inforStore.cur_filtered_phases_indices = filtered_phase_indices
        // console.log(inforStore.cur_filtered_phases)
        // inforStore.cur_filtered_events = inforStore.cur_filtered_phases.reduce((accu, obj) => accu.concat(obj.evolution_events), [])
        pcp_svg.selectAll(".line").remove()
        if (sel_cnt != phases_objs.length) 
          lines.data(filtered_phase_objs)
            .join('path')
            .attr("class", "line")
            .attr("d", d => {
              return d3.line()(Object.keys(inforStore.phase_pcp_dims).map(dim => {
                let dim_bins = dim_objs[dim].bins
                let y_id = -1
                for (let i = 0; i < dim_bins.length; ++i) {
                  if (d[dim] < dim_bins[i][1]) {
                    y_id = i
                    break
                  }
                }
                let cur_y_step = 1.0*main_h / dim_objs[dim].bins.length
                return [xScale(dim), yScales[dim](y_id-0.5)-cur_y_step]
              }))
            })  
            .attr("fill", 'none')
            .attr("stroke", (d,i) => '#ababab')
            .attr("stroke-width", 2)
            .attr("opacity", 0.9)
      })
    brushes_g[dim] = brush_layer_g.append("g")
      // .attr('transform', `translate(${xScale(dim)}, 0)`)
      .attr('class', `brush-${dim}`)
      .call(brushes[dim])
    brushes_g[dim].select('.overlay')
      .attr('dim', dim)
  })

  // 创建轴
  const yAxes = {};
  let pcp_axes = {}
  Object.keys(inforStore.phase_pcp_dims).forEach(dim => {
    yAxes[dim] = d3.axisLeft(yScales[dim])
      .ticks(dim_objs[dim].bin_edges.length)
      .tickSize(2.0) // 设置刻度线长度
      .tickFormat(d => dim_objs[dim].bin_edges[d])
    // 添加轴
    pcp_axes[dim] = pcp_svg.append("g")
      .attr("class", "pcp-axis")
      .attr("transform", "translate(" + xScale(dim) + ",0)")
      .call(yAxes[dim]);
    // 添加样式...
    pcp_axes[dim].selectAll('text')
      .style("font-size", "9.5px"); // 设置字号大小
    pcp_axes[dim].selectAll(".tick line")
      .style("stroke-width", "1.5px"); // 设置刻度线的宽度
    pcp_axes[dim].selectAll(".domain")
      .style("stroke-width", "1.5px"); // 设置轴线的宽度
  })
}
</script>

<template>
  <div>
    <svg id="phases-pcp-svg"></svg>
  </div>
</template>

<style scoped>
#instance-legends {
  margin-left: 20px;
}
</style>