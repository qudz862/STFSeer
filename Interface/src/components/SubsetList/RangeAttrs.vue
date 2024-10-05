<script setup>
import { ref, reactive, computed, watch, onMounted, onUpdated } from "vue"
import { useInforStore } from '@/stores/infor.js'
import getData from '@/services/index.js'
import * as d3 from 'd3'
import * as d3Sankey from 'd3-sankey'
import { valColorScheme_blue, valColorScheme_red, valColorScheme_fire } from '@/data/index.js'

const inforStore = useInforStore()

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
  if (inforStore.select_ranges.length == 0) matched_subsets = filtered_points
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
  console.log('cur_range_infor', inforStore.cur_range_infor);
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
  // console.log(inforStore.cur_range_infor);
  drawRangeRows()
  // drawRangeSankey()
})

onUpdated(() => {
  drawRangeRows()
  // drawRangeSankey()
})

function drawRangeRows() {
  // 设置svg
  let margin_left = 8, margin_right = 5, margin_top = 4, margin_bottom = 4
  let max_row_width = 0
  for (let attr in inforStore.cur_range_infor) {
    let svg_id = view_id('ranges', attr)
    d3.select(`#${svg_id}`).selectAll('*').remove()
    let valid_ranges = inforStore.cur_range_infor[attr].filter(item => item.cur_cnt > 0)
    // let valid_ranges = inforStore.cur_range_infor[attr]
    let cell_w = 80, cell_h = 25
    let cell_num = valid_ranges.length
    let main_w = cell_num * (cell_w+10)
    let main_h = cell_h
    let svg_w = main_w + margin_left + margin_right
    if (svg_w > max_row_width) max_row_width = svg_w
    let svg_h = main_h + margin_top + margin_bottom
    let svg = d3.select(`#${svg_id}`)
      .attr('width', svg_w)
      .attr('height', svg_h)
    let g = svg.append('g')
      .attr('transform', `translate(${margin_left}, ${margin_top})`)
    let range_cells = g.selectAll('g')
      .data(valid_ranges)
      .join('g')
        .attr('range-str', d => d.range_str)
        .attr('transform', (d,i) => `translate(${i*(cell_w+8)}, 0)`)
        .style('cursor', 'pointer')
        .on('click', (e,d) => {
          // let cur_range_str = d3.select(e.target).node().parentNode.getAttribute('range-str')
          let cur_range_str = d.range_str
          if (inforStore.select_ranges.includes(cur_range_str)) {
            let cur_index = inforStore.select_ranges.indexOf(cur_range_str)
            if (cur_index !== -1) inforStore.select_ranges.splice(cur_index, 1)
          } else {
            inforStore.select_ranges.push(cur_range_str)
          }
          filterSubsets()
        })
    let node_block_w = 80
    let node_block_h = 24
    let err_rect_w = 8
    range_cells.append('rect')
      .attr('selected', 'false')
      .attr('x', 0).attr('y', 0)
      .attr('width', node_block_w)
      .attr('height', node_block_h)
      .attr('fill', '#fff')
      .attr('stroke', d => {
        if (inforStore.select_ranges.includes(d.range_str)) return '#333'
        else return '#666'
      })
      .attr('stroke-width', d => {
        if (inforStore.select_ranges.includes(d.range_str)) return 3
        else return 1
      })

    let coverScale = d3.scaleLinear()
      .domain([0,1])
      .range([0, node_block_w-err_rect_w])
      // let errColorScale = d3.scaleSequential(d3.interpolateRdBu)
      //   .domain(inforStore.global_err_range
      // console.log(root.descendants());
      // let abs_residual = root.descendants().map(item => item.data.abs_residual)
    let errColorScale = d3.scaleQuantize()
      // .domain([0, inforStore.global_err_range[1]])
      .domain([0, global_resi_max.value])
      .range(valColorScheme_fire)
    range_cells.append('rect')
      .attr('x', err_rect_w).attr('y', 0.5)
      .attr('width', node_block_w-err_rect_w-0.5)
      .attr('height', 6)
      .attr('fill', '#cecece')
      .attr('stroke', 'none')
    range_cells.append('rect')
      .attr('x', err_rect_w).attr('y', 0.5)
      .attr('width', d => coverScale(d.cover))
      .attr('height', 6)
      .attr('fill', valColorScheme_blue[3])
      .attr('stroke', 'none')
    range_cells.append('rect')
      .attr('x', 0.5).attr('y', 0.5)
      .attr('width', err_rect_w-0.5)
      .attr('height', node_block_h-1)
      .attr('fill', d => errColorScale(d.abs_residual))
      .attr('stroke', 'none')
      // node.append('rect')
      //   .attr('x', err_rect_w).attr('y', -12)
      //   .attr('width', err_rect_w)
      //   .attr('height', node_block_h)
      //   .attr('fill', 'blue')  
    range_cells.append('rect')
      .attr('id', (d,i) => `highlight-${d.range_str}`)
      .attr('x', 0).attr('y', 0)
      .attr('width', node_block_w)
      .attr('height', node_block_h)
      .attr('fill', 'none')
      .attr('stroke', 'none')
    range_cells.append("text")
      .attr("dy", "1.6em")
      .attr("x", d => {
        let str = `[${d.range[0]}, ${d.range[1]}]`
        return 45 - str.length*2.7
      })
      .attr("text-anchor", d => (d.children ? "start" : "start"))
      .text(d => `[${d.range[0]}, ${d.range[1]}]`)
      .style('font-size', 12)
      .attr('fill', '#333')
      // .clone(true)
      // .lower()
      // .attr("stroke", "white")
  }
  $('.attr-range-row').css('width', `${max_row_width + 50}px`) 
}

function drawRangeSankey() {
  const width = 600;
  const height = 300;
  const svg = d3.select("#test-sankey")
    .attr('width', width)
    .attr('height', height)
  
  
}
</script>

<template>
  <div class="attr-range-region">
    <div v-for="(value, key, index) in inforStore.meta_attr_objs" class="attr-range-row" :key="index">
      <div class="attr-label">
        <span v-if="inforStore.meta_attr_objs[key].icon_type == 'value'" class="iconfont attr-label-icon">&#xe618; </span>
        <span v-if="inforStore.meta_attr_objs[key].icon_type == 'time'" class="iconfont attr-label-icon">&#xe634; </span>
        <span v-if="inforStore.meta_attr_objs[key].icon_type == 'space'" class="iconfont attr-label-icon">&#xe647; </span>
        <span class="attr_simple_str">{{ value.simple_str }}</span>
      </div>
      <svg :id="view_id('ranges', key)"></svg>
    </div>
    <!-- <svg id="test-sankey"></svg> -->
  </div>
</template>

<style scoped>
.block-title {
  font-size: 18px;
  font-weight: 700;
  color: #333;
  margin-left: 6px;
  margin-bottom: 4px;
}

.attr-range-region {
  width: 860px;
  height: 240px;
  margin: 0 auto;
  margin-left: 0px;
  overflow-x: auto;
  margin-bottom: 12px;
}
/* 
#subsets-radviz {
  display: block;
  margin: 0 auto;
} */
.attr-range-row {
  display: flex;
  align-items: center;
  margin-bottom: 5px;
}

.attr-label {
  width: 48px;
}

.attr-label-icon {
  font-size: 20px;
  color: #333;
}

.attr_simple_str {
  font-size: 15px;
  font-weight: 700;
  /* color: #333; */
}

</style>