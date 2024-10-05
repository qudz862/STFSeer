<script setup>
import { ref, reactive, computed, watch, onMounted, onUpdated } from "vue"
import { useInforStore } from '@/stores/infor.js'
import getData from '@/services/index.js'
import * as d3 from 'd3'
import { valColorScheme_blue, valColorScheme_red } from '@/data/index.js'

const inforStore = useInforStore()

let val_bins
let anchor_angles
let anchor_locs
let resi_hist_rate
let anchor_weights
let radviz_coords
let eps_th = 0

let margin_left = 10, margin_right = 10, margin_top = 10, margin_bottom = 34
let main_w = 710, main_h = 260
let svg_w = main_w + margin_left + margin_right
let svg_h = main_h + margin_top + margin_bottom
let radius = 300

function calculateKDistance(points, k) {  
    const distances = [];  
    for (let i = 0; i < points.length; i++) {  
        let sortedDistances = points.map((p, j) => {  
            if (i !== j) {  
                return Math.sqrt(Math.pow(p[0] - points[i][0], 2) + Math.pow(p[1] - points[i][1], 2));  
            }  
            return Infinity; // 排除自己  
        }).sort((a, b) => a - b).slice(0, k); // 取前k个  
  
        // 取第k个距离（如果k存在）  
        if (sortedDistances.length >= k) {  
            distances.push(sortedDistances[k - 1]);  
        } else {  
            // 如果少于k个邻居，可能需要特别处理  
            distances.push(Infinity);  
        }  
    }  
    return distances;  
}

function findMedian(arr) {
    // 首先对数组进行排序  
    arr.sort((a, b) => a - b);  
  
    const mid = Math.floor(arr.length / 4);  
    
    return arr[mid];  
}

function createWeightArray(size) {
    const weights = [];
    let sum = 0;

    // 生成初始的权重数组，使得两端大中间小，并且差异较小
    for (let i = 0; i < size; i++) {
        let weight = 0.5 + 0.5 * Math.abs((2 * i / (size - 1)) - 1);
        weights.push(weight);
        sum += weight;
    }

    // 计算当前的平均值
    const currentAverage = sum / size;

    // 调整权重数组，使其平均值为1
    for (let i = 0; i < size; i++) {
        weights[i] /= currentAverage;
    }

    return weights;
}

watch (() => inforStore.cur_subsets, (oldValue, newValue) => {
  console.log('cur_subsets');
  if (inforStore.cur_baseline_model.length == 0)
    val_bins = inforStore.error_distributions.all_residual_bins
  else val_bins = inforStore.error_distributions.all_residual_bins

  // anchor_angles = d3.range(-Math.PI/3, Math.PI/3+0.1, Math.PI*2/3 / (val_bins.length-2));
  anchor_angles = d3.range(-Math.PI*4/9, Math.PI*4/9+0.1, Math.PI / (val_bins.length-3));
  anchor_locs = anchor_angles.map(item => [radius*Math.sin(item), -radius * Math.cos(item)])
  resi_hist_rate = inforStore.cur_subsets.map(item => item.residual_hist_normalize)

  // 计算各个锚点的权重
  anchor_weights = createWeightArray(anchor_locs.length)
  console.log(anchor_weights)
  radviz_coords = radvizCoordinates(anchor_locs, resi_hist_rate)
  // let star_coords = starCoordinates(anchor_locs, resi_hist_rate)

  // 计算k-distance
  let kDistances = calculateKDistance(radviz_coords, 4)
  eps_th = findMedian(kDistances)
  console.log('eps_th', eps_th);

  drawSubsetsRadviz()
})

watch (() => inforStore.filtered_subsets, (oldValue, newValue) => {
  if (eps_th != 0) drawSubsetsRadviz()
})

function starCoordinates(anchors, hist_data) {
  let coords = []
  for (let k = 0; k < hist_data.length; ++k) {
    let p = [0, 0]
    for (let i = 0; i < anchors.length; ++i) {
      p[0] += anchors[i][0] * hist_data[k][i]
      p[1] += anchors[i][1] * hist_data[k][i]
    }
    let hist_sum = d3.sum(hist_data[k])
    p[0] /= hist_sum
    p[1] /= hist_sum
    p[0] *= 0.95
    p[1] *= 0.95
    coords.push(p)
  }
  
  return coords
}

function radvizCoordinates(anchors, hist_data) {
  let coords = []
  for (let k = 0; k < hist_data.length; ++k) {
    let p = [0, 0]
    for (let i = 0; i < anchors.length; ++i) {
      p[0] += anchors[i][0] * hist_data[k][i]
      p[1] += anchors[i][1] * hist_data[k][i]
    }
    let hist_sum = d3.sum(hist_data[k])
    p[0] /= hist_sum
    p[1] /= hist_sum
    p[0] *= 0.95
    p[1] *= 0.95

    coords.push(p)
  }
  
  return coords
}

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

watch (() => inforStore.cur_focus_subset, (oldValue, newValue) => {
  drawSubsetsRadviz()
})

function drawSubsetsRadviz() {
  d3.select('#subsets-radviz').selectAll('*').remove()
  // 需要获取的数据信息：误差范围段数据数目、子集在各误差范围段的分布、子集自身的误差、样本数目等信息

  // console.log(val_bins);
  // 设置svg
  let svg = d3.select('#subsets-radviz')
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
      // drawSubsetsRadviz()
    })
  let arc = d3.arc()
    .innerRadius(radius-2)
    .outerRadius(radius+2)
    .startAngle(-Math.PI*4/9)
    .endAngle(Math.PI*4/9); // Math.PI 表示半圆
  let arc_g = svg.append('g')
    .attr("transform", `translate(${svg_w/2},${margin_top+main_h*1.2})`)
    // .attr("transform", `translate(${svg_w/2},${margin_top+main_h*1.9})`)
  let arc_line = arc_g.append("path")
    .attr("d", arc)
    .attr("fill", "#cecece"); // 可以设置填充颜色
  
  let arc_anchors_g = arc_g.append('g')
  let arc_anchors = arc_anchors_g.selectAll('g')
    .data(anchor_angles)
    .join('g')
      .attr("transform", (d,i) => `translate(${radius * Math.sin(d)},${-radius * Math.cos(d)})`)
  arc_anchors.append('circle')
    .attr('cx', 0)
    .attr('cy', 0)
    .attr('r', 4)
    .attr('fill', (d,i) => {
      let val = (Math.abs(val_bins[i]) + Math.abs(val_bins[i+1])) / 2
      if ((i == 0) || (i == anchor_angles.length-1)) {
        return inforStore.extreme_err_color_scale(val)
      } else {
        return inforStore.mild_err_color_scale(val)
      }
      
    })
  arc_anchors.append('circle')
    .attr('cx', 0)
    .attr('cy', 0)
    .attr('r', 6)
    .attr('fill', 'none')
    .attr('stroke-width', 1)
    .attr('stroke', '#333')
  arc_anchors.append('text')
    // .attr('transform', (d,i) => `rotate(${(i*2/3)/(val_bins.length-2)*180-90*2/3})`)
    .attr('transform', (d,i) => `rotate(${(i*8/9)/(val_bins.length-4)*180-90*8/9})`)
    .attr('x', 0)
    .attr('y', 0)
    .attr('dy', -10)
    // .attr('text-anchor', (d,i) => {
    //   if (i == 0) return 'start'
    //   else if (i == val_bins.length-2) return 'end'
    //   else return 'middle'
    // })
    .attr('text-anchor', 'middle')
    .style('font-size', '12px')
    .text((d,i) => `[${val_bins[i]},${val_bins[i+1]}]`)
  
  // 绘制子集点
  let subsets_g = svg.append('g')
    // .attr("transform", `translate(${svg_w/2},${margin_top+main_h*1.9})`)
    .attr("transform", `translate(${svg_w/2},${margin_top+main_h*1.2})`)
  let subsets_glyphs = subsets_g.selectAll('g')
    .data(radviz_coords)
    .join('g')
      .attr("transform", d => `translate(${d[0]},${d[1]})`)
  subsets_glyphs.append('circle')
    .attr('class', 'subset-point')
    .attr('id', (d,i) => `subset-${i}`)
    .attr('cx', 0)
    .attr('cy', 0)
    .attr('r', 3.5)
    .attr('fill', (d,i) => {
      // console.log(inforStore.sel_subset_points);
      if (inforStore.filtered_subsets.length == inforStore.cur_subsets.length) {
        if (inforStore.cur_focus_subset == -1) return '#666'
        else if (inforStore.cur_subsets[i].subset_id == inforStore.cur_focus_subset) return '#0097A7'
        else return '#666'
      }
      let sel_points = inforStore.filtered_subsets.map((item,index) => (parseInt(item['subset_id'])-1))
      // if (sel_points.includes(i)) return '#3182bd'
      if (sel_points.includes(i)) return '#0097A7'
      else return '#666'
    })
    .attr('opacity', 0.5)
    .on('click', (e, d) => {
      // d3.selectAll('.subset-point').attr('fill', '#666')
      let subset_id = parseInt(d3.select(e.target).attr('id').split('-')[1])
      // console.log(d, radviz_coords[subset_id], getDistance(d, radviz_coords[subset_id]));
      console.log(main_h*0.05);
      inforStore.sel_subset_points = DBSCAN(radviz_coords, subset_id, eps_th, 1)
      filterSubsets()
      // let sel_points = inforStore.filtered_subsets.map((item,index) => item['subset_id'])
      // for (let i = 0; i < sel_points.length; ++i) {
      //   d3.select(`#subset-${sel_points[i]}`).attr('fill', valColorScheme_red[1])
      // }
      // drawSubsetsRadviz()
      // inforStore.filtered_subsets = inforStore.cur_subsets.filter((value, index) => sel_points.includes(index))
    })

}

</script>

<template>
  <div style="width: 730px;">
    <div class="block-title">Error Distribution Overview</div>
    <svg id="subsets-radviz"></svg>
  </div>
  
</template>

<style scoped>
#subsets-radviz {
  display: block;
  margin: 0 auto;
  margin-top: -20px;
}

.block-title {
  font-size: 18px;
  font-weight: 700;
  color: #555;
  margin-left: 10px;
  margin-bottom: 22px;
}
</style>