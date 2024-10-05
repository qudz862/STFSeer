<script setup>
import { ref, reactive, computed, watch, onMounted, onUpdated } from "vue"
import { useInforStore } from '@/stores/infor.js'
import getData from '@/services/index.js'
import * as d3 from 'd3'
import { valColorScheme_blue, valColorScheme_red, valColorScheme_fire } from '@/data/index.js'

const inforStore = useInforStore()
let cur_filter_conditions = ref({})

onMounted(() => {
  
})

function flattenFirstLevel(obj) {
  let result = {};
  
  // 遍历对象的第一层级
  for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
      // 将第二层级的所有属性合并到新的对象中
      Object.assign(result, obj[key]);
    }
  }

  return result;
}

function flattenFirstLevelInArray(array) {
  return array.map(item => {
    if (typeof item === 'object' && !Array.isArray(item)) {
      return flattenFirstLevel(item);
    } else {
      throw new Error('数组中的元素必须是具有两个层级的对象');
    }
  });
}

let flat_events_list = ref([])

watch (() => inforStore.cur_filtered_events, (oldValue, newValue) => {
  flat_events_list.value = flattenFirstLevelInArray(inforStore.cur_filtered_events)

  drawEventsPCP()
})

function drawEventsPCP() {
  d3.select('#events-pcp-svg').selectAll('*').remove()
  let margin_left = 70, margin_right = 70, margin_top = 18, margin_bottom = 10
  let main_w = 1020, main_h = 160
  let svg_w = main_w + margin_left + margin_right
  let svg_h = main_h + margin_top + margin_bottom
  let pcp_svg = d3.select('#events-pcp-svg')
    .attr('width', svg_w)
    .attr('height', svg_h)
    .append("g")
    .attr("transform", `translate(${margin_left},${margin_top})`)
  
  let events_objs = flat_events_list.value
  let dims = ['type', 'change_type', 'pre_val', 'mean_change_val', 'cur_val', 'total_change_abs', 'pre_grid_num', 'change_grid_num', 'cur_grid_num']
  let dim_objs = {
    "type": {
      "type": "category",
      "bin_edges": ["Forming", "Merging", "Continuing", "Growing", "Shape Changing", "Shrinking", "Splitting", "Dissolving"]
    },
    "change_type": {
      "type": "category",
      "bin_edges": ["Increasing", "Stable", "Decreasing"]
    },
    "pre_grid_num": {
      "type": "continuous"
    },
    "cur_grid_num": {
      "type": "continuous"
    },
    "change_grid_num": {
      "type": "continuous"
    },
    "pre_val": {
      "type": "continuous"
    },
    "cur_val": {
      "type": "continuous"
    },
    "mean_change_val": {
      "type": "continuous"
    },
    "total_change_abs": {
      "type": "continuous"
    }
  }
  events_objs.forEach(obj => {
    obj['selected'] = 1
  })

  // 为每个轴定义scale
  let yScales = {}
  dims.forEach(dim => {
    cur_filter_conditions.value[dim] = []
    if (dim_objs[dim].type == 'continuous') {
      let cur_dim_val = events_objs.map(item => item[dim])
      yScales[dim] = d3.scaleLinear()
        .domain(d3.extent(cur_dim_val))
        .range([main_h, 0])
    }
    if (dim_objs[dim].type == 'category') {
      yScales[dim] = d3.scalePoint()
        .domain(dim_objs[dim].bin_edges)
        .range([main_h, 0])
    }
  })
  let xScale = d3.scalePoint()
    .domain(dims)
    .range([0, main_w])

  // 添加轴标签
  let textElements = pcp_svg.selectAll(".axis-label")
    .data(dims)
    .enter().append("text")
    .attr("id", (d,i) => `axis-label-${i}`)
    .attr("transform", d => `translate(${xScale(d)}, -10)`)
    .text(d => d)
    .attr('text-anchor', 'middle')
    .style('font-size', '12px')

  
  // 添加路径
  let lines = pcp_svg.selectAll(".line")
    .data(events_objs)
    .join("path")
    .attr("class", "line")
    .attr("d", d => d3.line()(dims.map(dim => [xScale(dim), yScales[dim](d[dim])])))  
    .attr("fill", 'none')
    .attr("stroke", (d,i) => {
      if (d.selected == 1) return valColorScheme_blue[3]
      else return '#999'
    })
    .attr("stroke-width", 1)
    .attr("opacity", 0.5)

  // 创建刷选器
  const brushes = {}
  const brushes_g = {}
  let brush_w = 40
  // dims.forEach(dim => {
  //   brushes[dim] = d3.brushY()
  //     .extent([[xScale(dim)-brush_w/2, 0], [xScale(dim)+brush_w/2, main_h]])
  //     .on('brush', brushMove)
  //     .on("end", brushEnded)
  //   brushes_g[dim] = pcp_svg.append("g")
  //     // .attr('transform', `translate(${xScale(dim)}, 0)`)
  //     .attr('class', `brush-${dim}`)
  //     .call(brushes[dim])
  //   brushes_g[dim].select('.overlay')
  //     .attr('dim', dim)
  // })

  // let brushing = false; // 添加一个标志位
  // function brushMove(e) {
  //   console.log(e);
  //   let cur_dim = d3.select(e.sourceEvent.target).attr('dim')
  //   if (e && !brushing) {
  //     brushing = true; // 设置标志位，防止递归调用
  //     let selection = e.selection
  //     let step = 1
  //     let y0 = Math.floor(selection[0] / step) * step
  //     let y1 = Math.ceil(selection[1] / step) * step
  //     // 更新选择框的位置
  //     brushes_g[cur_dim].call(brushes[cur_dim].move, [y0, y1])
  //     brushing = false
  //   }
  // }
  // function brushEnded(e) {
  //   let selection = e.selection;
  //   let cur_dim = d3.select(e.sourceEvent.target).attr('dim')
  //   console.log('end', cur_dim);
  //   if (selection) {
  //     let y0 = yScales[cur_dim].invert(selection[0]);
  //     let y1 = yScales[cur_dim].invert(selection[1]);
  //     let y0_int = parseInt(Math.round(y0))
  //     let y1_int = parseInt(Math.round(y1))

  //     cur_filter_conditions.value[cur_dim] = [y1_int, y0_int]
  //     // 在这里可以执行你的操作，例如根据范围重新渲染图表等
  //   } else {
  //     cur_filter_conditions.value[cur_dim] = []
  //   }
  //   events_objs.forEach(obj => {
  //     obj.selected = 1
  //     for (let dim in cur_filter_conditions.value) {
  //       if (cur_filter_conditions.value[dim] == []) continue
  //       let cur_condition = cur_filter_conditions.value[dim]
  //       if (obj[dim] < cur_condition[0] || obj[dim] > cur_condition[1]) {
  //         obj.selected = 0
  //         break
  //       }
  //     }
  //   })
  //   let filtered_phase_objs = events_objs.filter(item => item.selected == 1)
  //   inforStore.cur_filtered_phases = filtered_phase_objs
  //   inforStore.cur_filtered_events = inforStore.cur_filtered_phases.reduce((accu, obj) => accu.concat(obj.evolution_events), [])
  //   // pcp_svg.selectAll(".line").selectAll('*').remove()
  //   lines.attr("stroke", (d,i) => {
  //     if (d.selected == 1) return valColorScheme_blue[3]
  //     else return '#999'
  //   })
  // }

  // 创建轴
  const yAxes = {};
  let pcp_axes = {}
  dims.forEach(dim => {
    yAxes[dim] = d3.axisLeft(yScales[dim])
      .tickSize(2.0) // 设置刻度线长度
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
    <svg id="events-pcp-svg"></svg>
  </div>
</template>

<style scoped>
#instance-legends {
  margin-left: 20px;
}
</style>