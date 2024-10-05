<script setup>
import { ref, reactive, computed, watch, onMounted, onUpdated } from "vue"
import { useInforStore } from '@/stores/infor.js'
import getData from '@/services/index.js'
import * as d3 from 'd3'
import { valColorScheme_blue, valColorScheme_red } from '@/data/index.js'
import { Cursor } from "mapbox-gl-controls/lib/ImageControl/types"

const inforStore = useInforStore()

let all_subset_ids
let subset_index
let cur_subset

function countCommonKeys(arr1, arr2) {
    // 将第一个数组转换为集合
    const set1 = new Set(arr1);

    // 遍历第二个数组，统计在集合中存在的元素数量
    let count = 0;
    arr2.forEach(value => {
        if (set1.has(value)) {
            count++;
        }
    });

    return count;
}

function countCommonValues(arr1, arr2) {
  let count = 0;
  for (let arr_i of arr1) {
    for (let arr_j of arr2) {
      if (arraysEqual(arr_i, arr_j)) count += 1
    }
  }

  return count;
}

function arraysEqual(arr1, arr2) {
    // 如果长度不同，则数组不相同
    if (arr1.length !== arr2.length) {
        return false;
    }

    // 对数组进行排序
    const sortedArr1 = arr1.slice().sort();
    const sortedArr2 = arr2.slice().sort();

    // 比较排序后的数组
    for (let i = 0; i < sortedArr1.length; i++) {
        if (sortedArr1[i] !== sortedArr2[i]) {
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
  console.log(inforStore.select_ranges);
  if (Object.values(inforStore.select_ranges).every(value => value.length === 0)) matched_subsets = filtered_points
  else {
    let pure_select_ranges = []
    for (let attr in inforStore.select_ranges) {
      pure_select_ranges = pure_select_ranges.concat(inforStore.select_ranges[attr])
    }
    console.log(pure_select_ranges);
    for (let i = 0; i < filtered_points.length; ++i) {
      let isSuperset = isArraySuperset(filtered_points[i].subset_attrs, pure_select_ranges);
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
  if (inforStore.cur_focus_subset == -1) {
    inforStore.cur_related_subsets = {
      'contained_subgroups': [],
      'replace_subgroups': [],
      'drill_down_subgroups': []
    }
  } else {
    let contained_subgroups = []
    let replace_subgroups = []
    let drill_down_subgroups = []

    // 判断和获得contain_subsets
    all_subset_ids = inforStore.cur_subsets.map(item => item.subset_id)
    subset_index = all_subset_ids.indexOf(inforStore.cur_focus_subset)
    cur_subset = inforStore.cur_subsets[subset_index]
    if (cur_subset.hasOwnProperty('contain_subsets')) {
      contained_subgroups = cur_subset.contain_subsets
    }

    for (let i = 0; i < inforStore.cur_subsets.length; ++i) {
      // 根据range_val，判断和获得replace_subgroups
      if (inforStore.cur_subsets[i].subset_id == inforStore.cur_focus_subset) continue
      // if (Object.keys(inforStore.cur_subsets[i].range_val).length == Object.keys(cur_subset.range_val).length) {
      if (arraysEqual(Object.keys(inforStore.cur_subsets[i].range_val), Object.keys(cur_subset.range_val))) {
        // console.log('arraysEqual!');
        let commonCnts = countCommonValues(Object.values(inforStore.cur_subsets[i].range_val), Object.values(cur_subset.range_val))
        if (commonCnts == Object.keys(inforStore.cur_subsets[i].range_val).length - 1) {
          // console.log('commonCnts', commonCnts);
          replace_subgroups.push(inforStore.cur_subsets[i])
        }
      }
      // 根据range_val，判断和获得drill_down_subgroups
      if (Object.keys(inforStore.cur_subsets[i].range_val).length > Object.keys(cur_subset.range_val).length) {
        // console.log('length greater');
        let commonCnts = countCommonValues(Object.values(inforStore.cur_subsets[i].range_val), Object.values(cur_subset.range_val))
        let commonKeyCnts = countCommonKeys(Object.keys(inforStore.cur_subsets[i].range_val), Object.keys(cur_subset.range_val))
        if ((commonKeyCnts > 0) && (commonCnts > 0) && (commonKeyCnts == commonCnts)) {
          drill_down_subgroups.push(inforStore.cur_subsets[i])
        }
      }
    }

    inforStore.cur_related_subsets = {
      'contained_subgroups': contained_subgroups,
      'replace_subgroups': replace_subgroups,
      'drill_down_subgroups': drill_down_subgroups
    }
  }
})

let size_opacities = reactive({
  'contained_subgroups': [],
  'replace_subgroups': [],
  'drill_down_subgroups': []
})
let error_opacities = reactive({
  'contained_subgroups': [],
  'replace_subgroups': [],
  'drill_down_subgroups': []
})
let error_comp_opacities = reactive({
  'contained_subgroups': [],
  'replace_subgroups': [],
  'drill_down_subgroups': []
})

watch (() => inforStore.cur_related_subsets, (oldValue, newValue) => {
  let all_related_subsets = [...inforStore.cur_related_subsets.contained_subgroups, ...inforStore.cur_related_subsets.replace_subgroups, ...inforStore.cur_related_subsets.drill_down_subgroups]
  // console.log('all_related_subsets', all_related_subsets.length);
  let all_related_sizes = all_related_subsets.map(item => Math.abs(item.sup_rate - cur_subset.sup_rate))
  let all_related_errors = all_related_subsets.map(item => Math.abs(item.residual_abs - cur_subset.residual_abs))
  let all_related_errors_comp = all_related_subsets.map(item => Math.abs(item.residual_abs_comp - cur_subset.residual_abs_comp))
  for (let key in size_opacities) {
    for (let subset of inforStore.cur_related_subsets[key]) {
      let size_opacity = 0.4 + Math.abs(subset.sup_rate - cur_subset.sup_rate) / d3.max(all_related_sizes) * 0.6
      let error_opacity = 0.4 + Math.abs(subset.residual_abs - cur_subset.residual_abs) / d3.max(all_related_errors) * 0.6
      let error_opacity_comp = 0.4 + Math.abs(subset.residual_abs_comp - cur_subset.residual_abs_comp) / d3.max(all_related_errors_comp) * 0.6
      size_opacities[key].push(size_opacity)
      error_opacities[key].push(error_opacity)
      error_comp_opacities[key].push(error_opacity_comp)
    }
  }
})

function arrayIn2DArray(twoDArray, oneDArray) {
    return twoDArray.some(row => arraysEqual(row, oneDArray));
}

// function arraysEqual(arr1, arr2) {
//     if (arr1.length !== arr2.length) return false;
//     return arr1.every((value, index) => value === arr2[index]);
// }

function checkReplaceSubset(value, key) {
  all_subset_ids = inforStore.cur_subsets.map(item => item.subset_id)
  subset_index = all_subset_ids.indexOf(inforStore.cur_focus_subset)
  cur_subset = inforStore.cur_subsets[subset_index]
  if (arrayIn2DArray(Object.values(cur_subset.range_val), value)) return false
  else return true
}

function checkDrillDownSubset(value, key) {
  all_subset_ids = inforStore.cur_subsets.map(item => item.subset_id)
  subset_index = all_subset_ids.indexOf(inforStore.cur_focus_subset)
  cur_subset = inforStore.cur_subsets[subset_index]
  if (Object.keys(cur_subset.range_val).includes(key)) return false
  else return true
}

function subsetSizeChange(subset) {
  if (subset.sup_rate > cur_subset.sup_rate) {
    return `${parseFloat((subset.sup_rate - cur_subset.sup_rate).toFixed(4))}`
  } else {
    return `${parseFloat((cur_subset.sup_rate - subset.sup_rate).toFixed(4))}`
  }
}

function subsetSizeOpacity(subset) {
  let all_related_subsets = [...inforStore.cur_related_subsets.contained_subgroups, ...inforStore.cur_related_subsets.contained_subgroups, ...inforStore.cur_related_subsets.contained_subgroups]
  let all_related_sizes = all_related_subsets.map(item => Math.abs(item.sup_rate - cur_subset.sup_rate))
  let opacity = Math.abs(subset.sup_rate - cur_subset.sup_rate) / d3.max(all_related_sizes) + 0.05
  return opacity
}

function subsetErrorChange(subset) {
  if (subset.residual_abs > cur_subset.residual_abs) {
    return `${parseFloat((subset.residual_abs - cur_subset.residual_abs).toFixed(4))}`
  } else {
    return `${parseFloat((cur_subset.residual_abs - subset.residual_abs).toFixed(4))}`
  }
}

function subsetErrorOpacity(subset) {
  let all_related_subsets = [...inforStore.cur_related_subsets.contained_subgroups, ...inforStore.cur_related_subsets.contained_subgroups, ...inforStore.cur_related_subsets.contained_subgroups]
  let all_related_errors = all_related_subsets.map(item => Math.abs(item.residual_abs - cur_subset.residual_abs))
  let opacity = Math.abs(subset.residual_abs - cur_subset.residual_abs) / d3.max(all_related_errors) + 0.05
  return opacity
}

function judgeInOrDe(subset, metric) {
  if (subset[metric] > cur_subset[metric]) return 'value-increase'
  else if (subset[metric] < cur_subset[metric]) return 'value-decrease'
  else return 'value-equal'
}

function judgeIncreaseFlag(subset, metric) {
  if (subset[metric] > cur_subset[metric]) return 'increase'
  else if (subset[metric] < cur_subset[metric]) return 'decrease'
  else return 'equal'
}

function selectRelatedSubset(subset) {
  inforStore.cur_focus_subset = subset.subset_id
  // let all_subset_ids = inforStore.filtered_subsets.map(item => item.subset_id)
  // let subset_index = all_subset_ids.indexOf(subset.subset_id)

  inforStore.focused_subsets_list = inforStore.focused_subsets_list.slice(0, inforStore.currentIndex + 1)
  inforStore.focused_subsets_list.push(inforStore.cur_focus_subset)
  inforStore.currentIndex = inforStore.focused_subsets_list.length-1
  
}

let sel_type = ref('none')

watch (() => sel_type.value, (oldValue, newValue) => {
  $('.subset-row').removeClass('subset-selected')
  $(`.subset-row[subset_id=${inforStore.cur_focus_subset}]`).addClass('subset-selected')
})



function selContainedSubgroups() {
  if (sel_type.value == 'contained') {
    sel_type.value = 'none'
    filterSubsets()
  } else {
    sel_type.value = 'contained'
    inforStore.filtered_subsets = inforStore.cur_subsets.filter((value, index) => (inforStore.cur_focus_subset == value.subset_id)).concat(inforStore.cur_related_subsets.contained_subgroups) 
  } 
}

function selSameLevelSubgroups() {
  if (sel_type.value == 'same-level') {
    sel_type.value = 'none'
    filterSubsets()
  } else {
    sel_type.value = 'same-level'
    inforStore.filtered_subsets = inforStore.cur_subsets.filter((value, index) => (inforStore.cur_focus_subset == value.subset_id)).concat(inforStore.cur_related_subsets.replace_subgroups) 
  }
}

function selDrillDownSubgroups() {
  if (sel_type.value == 'drill-down') {
    sel_type.value = 'none'
    filterSubsets()
  } else {
    sel_type.value = 'drill-down'
    inforStore.filtered_subsets = inforStore.cur_subsets.filter((value, index) => (inforStore.cur_focus_subset == value.subset_id)).concat(inforStore.cur_related_subsets.drill_down_subgroups) 
  }
}

</script>

<template>
  <div class="related-subset-block">
    <div class="type-block" v-if="inforStore.cur_related_subsets.contained_subgroups.length > 0">
      <div class="block-title" @click="selContainedSubgroups()" :select="sel_type=='contained'">Contained Subgroups ({{ inforStore.cur_related_subsets.contained_subgroups.length }})</div>
      <div class="subset-card-container">
        <div class="subset-card" v-for="(item, index1) in inforStore.cur_related_subsets.contained_subgroups" :key="index1" @click="selectRelatedSubset(item)">
          <div class="card-attrs"> 
            <div v-for="(value, key, index2) in item.range_val" :key="index2" :class="{ 'highlight-attr': checkReplaceSubset(value, key) }" >
              <span v-if="inforStore.meta_attr_objs[key].icon_type == 'value'" class="iconfont">&#xe618;</span>
              <span v-if="inforStore.meta_attr_objs[key].icon_type == 'time'" class="iconfont">&#xe634;</span>
              <span v-if="inforStore.meta_attr_objs[key].icon_type == 'space'" class="iconfont">&#xe647;</span>&nbsp;
              <span>{{ inforStore.meta_attr_objs[key].simple_str }}</span>&nbsp;
              <span>[{{value[0]}},{{value[1]}}]</span>
            </div>
          </div>
          <div class="vertical-seg"></div>
          <div class="card-metrics">
            <div style="display: flex; align-items: center;">Support: {{ item.sup_rate }} <span style="display: flex; align-items: center;" v-if="judgeIncreaseFlag(item, 'sup_rate') != 'equal'" :class="judgeInOrDe(item, 'sup_rate')">&nbsp;(
              <span class="iconfont increase-icon" v-if="size_opacities.contained_subgroups.length > 0 && judgeIncreaseFlag(item, 'sup_rate') == 'increase'" :style="{opacity: size_opacities.contained_subgroups[index1]}">&#xe622;</span>
              <span class="iconfont decrease-icon" v-if="size_opacities.contained_subgroups.length > 0 && judgeIncreaseFlag(item, 'sup_rate') == 'decrease'" :style="{opacity: size_opacities.contained_subgroups[index1]}">&#xe623;</span>
              <!-- {{ subsetSizeChange(item) }} -->
              )
            </span></div>

            <div style="display: flex; align-items: center;"><span v-if="inforStore.cur_baseline_model.length>0">Base_Error:</span><span v-else>Abs_Error:</span> {{ item.residual_abs }} <span style="display: flex; align-items: center;" v-if="judgeIncreaseFlag(item, 'residual_abs') != 'equal'" :class="judgeInOrDe(item, 'residual_abs')">&nbsp;(
              <span class="iconfont increase-icon" v-if="error_opacities.contained_subgroups.length > 0 && judgeIncreaseFlag(item, 'residual_abs') == 'increase'" :style="{opacity: error_opacities.contained_subgroups[index1]}">&#xe622;</span>
              <span class="iconfont decrease-icon" v-if="error_opacities.contained_subgroups.length > 0 && judgeIncreaseFlag(item, 'residual_abs') == 'decrease'" :style="{opacity: error_opacities.contained_subgroups[index1]}">&#xe623;</span>
              <!-- {{ subsetErrorChange(item) }} -->
              )
            </span></div>
            <div v-if="inforStore.cur_baseline_model.length > 0" style="display: flex; align-items: center;">Focus_Error: {{ item.residual_abs_comp }} <span style="display: flex; align-items: center;" v-if="judgeIncreaseFlag(item, 'residual_abs_comp') != 'equal'" :class="judgeInOrDe(item, 'residual_abs_comp')">&nbsp;(
              <span class="iconfont increase-icon" v-if="error_comp_opacities.contained_subgroups.length > 0 && judgeIncreaseFlag(item, 'residual_abs_comp') == 'increase'" :style="{opacity: error_comp_opacities.contained_subgroups[index1]}">&#xe622;</span>
              <span class="iconfont decrease-icon" v-if="error_comp_opacities.contained_subgroups.length > 0 && judgeIncreaseFlag(item, 'residual_abs_comp') == 'decrease'" :style="{opacity: error_comp_opacities.contained_subgroups[index1]}">&#xe623;</span>
              <!-- {{ subsetErrorChange(item) }} -->
              )
            </span></div>
          </div>
        </div>
      </div>
    </div>
    <div class="type-block" v-if="inforStore.cur_related_subsets.replace_subgroups.length > 0">
      <div class="block-title" @click="selSameLevelSubgroups()" :select="sel_type=='same-level'">Same-Level Subgroups ({{ inforStore.cur_related_subsets.replace_subgroups.length }})</div>
      <div class="subset-card-container">
        <div class="subset-card" v-for="(item, index1) in inforStore.cur_related_subsets.replace_subgroups" :key="index1" @click="selectRelatedSubset(item)">
          <div class="card-attrs"> 
            <div v-for="(value, key, index2) in item.range_val" :key="index2" :class="{ 'highlight-attr': checkReplaceSubset(value, key) }" >
              <span v-if="inforStore.meta_attr_objs[key].icon_type == 'value'" class="iconfont">&#xe618;</span>
              <span v-if="inforStore.meta_attr_objs[key].icon_type == 'time'" class="iconfont">&#xe634;</span>
              <span v-if="inforStore.meta_attr_objs[key].icon_type == 'space'" class="iconfont">&#xe647;</span>&nbsp;
              <span>{{ inforStore.meta_attr_objs[key].simple_str }}</span>&nbsp;
              <span>[{{value[0]}},{{value[1]}}]</span>
            </div>
          </div>
          <div class="vertical-seg"></div>
          <div class="card-metrics">
            <div style="display: flex; align-items: center;">Support: {{ item.sup_rate }} <span style="display: flex; align-items: center;" v-if="judgeIncreaseFlag(item, 'sup_rate') != 'equal'" :class="judgeInOrDe(item, 'sup_rate')">&nbsp;(
              <span class="iconfont increase-icon" v-if="size_opacities.replace_subgroups.length > 0 && judgeIncreaseFlag(item, 'sup_rate') == 'increase'" :style="{opacity: size_opacities.replace_subgroups[index1]}">&#xe622;</span>
              <span class="iconfont decrease-icon" v-if="size_opacities.replace_subgroups.length > 0 && judgeIncreaseFlag(item, 'sup_rate') == 'decrease'" :style="{opacity: size_opacities.replace_subgroups[index1]}">&#xe623;</span>
              <!-- {{ subsetSizeChange(item) }} -->
              )
            </span></div>

            <div style="display: flex; align-items: center;"><span v-if="inforStore.cur_baseline_model.length>0">Base_Error:</span><span v-else>Abs_Error:</span> {{ item.residual_abs }} <span style="display: flex; align-items: center;" v-if="judgeIncreaseFlag(item, 'residual_abs') != 'equal'" :class="judgeInOrDe(item, 'residual_abs')">&nbsp;(
              <span class="iconfont increase-icon" v-if="error_opacities.replace_subgroups.length > 0 && judgeIncreaseFlag(item, 'residual_abs') == 'increase'" :style="{opacity: error_opacities.replace_subgroups[index1]}">&#xe622;</span>
              <span class="iconfont decrease-icon" v-if="error_opacities.replace_subgroups.length > 0 && judgeIncreaseFlag(item, 'residual_abs') == 'decrease'" :style="{opacity: error_opacities.replace_subgroups[index1]}">&#xe623;</span>
              <!-- {{ subsetErrorChange(item) }} -->
              )
            </span></div>
            <div v-if="inforStore.cur_baseline_model.length > 0" style="display: flex; align-items: center;">Focus_Error: {{ item.residual_abs_comp }} <span style="display: flex; align-items: center;" v-if="judgeIncreaseFlag(item, 'residual_abs_comp') != 'equal'" :class="judgeInOrDe(item, 'residual_abs_comp')">&nbsp;(
              <span class="iconfont increase-icon" v-if="error_comp_opacities.replace_subgroups.length > 0 && judgeIncreaseFlag(item, 'residual_abs_comp') == 'increase'" :style="{opacity: error_comp_opacities.replace_subgroups[index1]}">&#xe622;</span>
              <span class="iconfont decrease-icon" v-if="error_comp_opacities.replace_subgroups.length > 0 && judgeIncreaseFlag(item, 'residual_abs_comp') == 'decrease'" :style="{opacity: error_comp_opacities.replace_subgroups[index1]}">&#xe623;</span>
              <!-- {{ subsetErrorChange(item) }} -->
              )
            </span></div>
          </div>
        </div>
      </div>
    </div>
    <div class="type-block" v-if="inforStore.cur_related_subsets.drill_down_subgroups.length > 0">
      <div class="block-title" @click="selDrillDownSubgroups()" :select="sel_type=='drill-down'">Drill-Down Subgroups ({{ inforStore.cur_related_subsets.drill_down_subgroups.length }})</div>
      <div class="subset-card-container">
        <div class="subset-card" v-for="(item, index1) in inforStore.cur_related_subsets.drill_down_subgroups" :key="index1" @click="selectRelatedSubset(item)">
          <div class="card-attrs"> 
            <div v-for="(value, key, index2) in item.range_val" :key="index2" :class="{ 'highlight-attr': checkDrillDownSubset(value, key) }" >
              <span v-if="inforStore.meta_attr_objs[key].icon_type == 'value'" class="iconfont">&#xe618;</span>
              <span v-if="inforStore.meta_attr_objs[key].icon_type == 'time'" class="iconfont">&#xe634;</span>
              <span v-if="inforStore.meta_attr_objs[key].icon_type == 'space'" class="iconfont">&#xe647;</span>&nbsp;
              <span>{{ inforStore.meta_attr_objs[key].simple_str }}</span>&nbsp;
              <span>[{{value[0]}},{{value[1]}}]</span>
            </div>
          </div>
          <div class="vertical-seg"></div>
          <div class="card-metrics">
            <div style="display: flex; align-items: center;">Support: {{ item.sup_rate }} <span style="display: flex; align-items: center;" v-if="judgeIncreaseFlag(item, 'sup_rate') != 'equal'" :class="judgeInOrDe(item, 'sup_rate')">&nbsp;(
              <span class="iconfont increase-icon" v-if="size_opacities.drill_down_subgroups.length > 0 && judgeIncreaseFlag(item, 'sup_rate') == 'increase'" :style="{opacity: size_opacities.drill_down_subgroups[index1]}">&#xe622;</span>
              <span class="iconfont decrease-icon" v-if="size_opacities.drill_down_subgroups.length > 0 && judgeIncreaseFlag(item, 'sup_rate') == 'decrease'" :style="{opacity: size_opacities.drill_down_subgroups[index1]}">&#xe623;</span>
              <!-- {{ subsetSizeChange(item) }} -->
              )
            </span></div>

            <div style="display: flex; align-items: center;"><span v-if="inforStore.cur_baseline_model.length>0">Base_Error:</span><span v-else>Abs_Error:</span> {{ item.residual_abs }} <span style="display: flex; align-items: center;" v-if="judgeIncreaseFlag(item, 'residual_abs') != 'equal'" :class="judgeInOrDe(item, 'residual_abs')">&nbsp;(
              <span class="iconfont increase-icon" v-if="error_opacities.drill_down_subgroups.length > 0 && judgeIncreaseFlag(item, 'residual_abs') == 'increase'" :style="{opacity: error_opacities.drill_down_subgroups[index1]}">&#xe622;</span>
              <span class="iconfont decrease-icon" v-if="error_opacities.drill_down_subgroups.length > 0 && judgeIncreaseFlag(item, 'residual_abs') == 'decrease'" :style="{opacity: error_opacities.drill_down_subgroups[index1]}">&#xe623;</span>
              <!-- {{ subsetErrorChange(item) }} -->
              )
            </span></div>
            <div v-if="inforStore.cur_baseline_model.length > 0" style="display: flex; align-items: center;">Focus_Error: {{ item.residual_abs_comp }} <span style="display: flex; align-items: center;" v-if="judgeIncreaseFlag(item, 'residual_abs_comp') != 'equal'" :class="judgeInOrDe(item, 'residual_abs_comp')">&nbsp;(
              <span class="iconfont increase-icon" v-if="error_comp_opacities.drill_down_subgroups.length > 0 && judgeIncreaseFlag(item, 'residual_abs_comp') == 'increase'" :style="{opacity: error_comp_opacities.drill_down_subgroups[index1]}">&#xe622;</span>
              <span class="iconfont decrease-icon" v-if="error_comp_opacities.drill_down_subgroups.length > 0 && judgeIncreaseFlag(item, 'residual_abs_comp') == 'decrease'" :style="{opacity: error_comp_opacities.drill_down_subgroups[index1]}">&#xe623;</span>
              <!-- {{ subsetErrorChange(item) }} -->
              )
            </span></div>
          </div>
        </div>
      </div>
    </div>
  </div>
  
</template>

<style scoped>
.related-subset-block {
  display: flex;
  padding: 2px 4px;
  margin-top: 10px;
  /* justify-content: flex-start; */
}

.type-block {
  margin: 0 5px;
}

.highlight-attr {
  color: #0097A7;
  font-weight: 700;
}

.block-title {
  font-size: 18px;
  font-weight: 700;
  color: #555;
  margin-left: 6px;
  margin-bottom: 4px;
}

.block-title:hover {
  cursor: pointer;
  color: #0097A7;
}

.block-title[select=true] {
  color: #0097A7;
}

.subset-card-container {
  display: flex;
  justify-content: flex-start;
  align-items: start;
  flex-wrap: wrap;
  /* flex-basis: calc(33.333%); */
  /* flex: 1 1 auto; */
  max-width: 1844px;
  min-width: 430px;
  max-height: 194px;
  width:fit-content;
  overflow-y: auto;
}

.subset-card {
  display: flex;
  /* justify-content: space-between; */
  /* align-items: center; */
  border: solid 1px #bcbcbc;
  border-radius: 6px;
  margin-left: 6px;
  margin-right: 2px;
  margin-bottom: 8px;
  padding: 4px 6px;
  cursor: pointer;
  transition: box-shadow 0.3s ease; /* 添加过渡效果 */  
}

.subset-card:hover {
  box-shadow: 0 0 5px 2px rgba(170, 170, 170, 0.5); /* 鼠标滑过时的阴影 */
}

.card-attrs {
  height: 90px;
  display: grid;
  place-items: center;
  font-weight: 700;
  color: #333;
  /* height: 84px;
  line-height: 84px;
  text-align: center; */
}

.card-metrics {
  height: 90px;
  display: grid;
  /* place-items: center; */
  font-weight: 700;
}

.vertical-seg {
  width:1px;
  /* height: 60px; */
  background-color: #bcbcbc;
  margin-top: 6px;
  margin-bottom: 6px;
  margin-left: 8px;
  margin-right: 8px;
}

.value-increase {
  color: #a63603;
}

.value-decrease {
  color: #08519c;
}

.value-equal {
  color: #333;
}

.increase-icon {
  font-size: 22px;
  margin-top: -2px;
  margin-bottom: -4px;
}
.decrease-icon {
  font-size: 22px;
  margin-top: -4px;
  margin-bottom: -6px;
}

</style>