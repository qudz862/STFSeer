<script setup>
import { ref, reactive, computed, watch, onMounted, onUpdated } from "vue"
import { useInforStore } from '@/stores/infor.js'
import getData from '@/services/index.js'
import * as d3 from 'd3'
import { valColorScheme_blue, valColorScheme_red } from '@/data/index.js'

const inforStore = useInforStore()

watch (() => inforStore.cur_subsets, (oldValue, newValue) => {
  drawSubsetsProjection()
})

watch (() => inforStore.filtered_subsets, (oldValue, newValue) => {
  drawSubsetsProjection()
})

function getDistance(p1, p2) {
  let dx = p2[0] - p1[0];
  let dy = p2[1] - p1[1];
  return Math.sqrt(Math.pow(dx, 2) + Math.pow(dy, 2));
}
function findNeighbors(points, targetPointIndex, distanceThreshold) {
    const neighbors = [];
    const targetPoint = points[targetPointIndex];
    for (let i = 0; i < points.length; i++) {
        if (i !== targetPointIndex && getDistance(points[i], targetPoint) <= distanceThreshold) {
            neighbors.push(i);
        }
    }
    return neighbors;
}

function expandCluster(points, targetPointIndex, clusterIndices, distanceThreshold, minPts) {
    const neighbors = findNeighbors(points, targetPointIndex, distanceThreshold);
    for (const neighborIndex of neighbors) {
        if (!points[neighborIndex].visited) {
            points[neighborIndex].visited = true;
            const neighborNeighbors = findNeighbors(points, neighborIndex, distanceThreshold);
            if (neighborNeighbors.length >= minPts) {
                expandCluster(points, neighborIndex, clusterIndices, distanceThreshold, minPts);
            }
        }
        if (!points[neighborIndex].clustered) {
            clusterIndices.push(neighborIndex);
            points[neighborIndex].clustered = true;
        }
    }
}
function DBSCAN(points, targetPointIndex, distanceThreshold, minPts) {
    for (let i = 0; i < points.length; i++) {
        points[i].visited = false;
        points[i].clustered = false;
    }
    const clusterIndices = [targetPointIndex];
    expandCluster(points, targetPointIndex, clusterIndices, distanceThreshold, minPts);
    return clusterIndices;
}

function isArraySuperset(superset, subset) {
  return subset.every(value => {
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

function drawSubsetsProjection() {
  d3.select('#subsets-projection').selectAll('*').remove()
  // 需要获取的数据信息：误差范围段数据数目、子集在各误差范围段的分布、子集自身的误差、样本数目等信息
  let subsets_proj = inforStore.phase_subsets_proj
  let subsets_links = inforStore.phase_subsets_links
  // 设置svg
  let margin_left = 10, margin_right = 10, margin_top = 10, margin_bottom = 10
  let main_w = 690, main_h = 190
  let svg_w = main_w + margin_left + margin_right
  let svg_h = main_h + margin_top + margin_bottom
  let svg = d3.select('#subsets-projection')
    .attr('width', svg_w)
    .attr('height', svg_h)
  svg.append('rect')
    .attr('x', 0).attr('y', 0)
    .attr('width', svg_w).attr('height', svg_h)
    .attr('fill', '#fff')
    .on('click', (e) => {
      // d3.selectAll('.subset-point').attr('fill', '#666')
      inforStore.sel_subset_points = []
      filterSubsets()
    })
  let proj_x = subsets_proj.map(item => item[0])
  let proj_y = subsets_proj.map(item => item[1])
  let xScale = d3.scaleLinear()
    .domain(d3.extent(proj_x))
    .range([0, main_w])
  let yScale = d3.scaleLinear()
    .domain(d3.extent(proj_y))
    .range([0, main_h])
  
  let links_g = svg.append('g')
    .attr('transform', `translate(${margin_left}, ${margin_top})`)
  links_g.selectAll('line')
    .data(subsets_links)
    .join('line')
      .attr('x1', (d,i) => xScale(proj_x[d[0]]))
      .attr('y1', (d,i) => yScale(proj_y[d[0]]))
      .attr('x2', (d,i) => xScale(proj_x[d[1]]))
      .attr('y2', (d,i) => yScale(proj_y[d[1]]))
      .attr('stroke', '#666')

  let points_g = svg.append('g')
    .attr('transform', `translate(${margin_left}, ${margin_top})`)
  points_g.selectAll('circle')
    .data(subsets_proj)
    .join('circle')
      .attr('cx', (d,i) => xScale(d[0]))
      .attr('cy', (d,i) => yScale(d[1]))
      .attr('r', 3)
      .attr('fill', '#999')
      .attr('opacity', '0.5')
}

</script>

<template>
  <svg id="subsets-projection"></svg>
</template>

<style scoped>
#subsets-projection {
  display: block;
  margin: 0 auto;
  margin-top: -24px;
}
</style>