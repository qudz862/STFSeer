<script setup>
import { ref, reactive, computed, watch, onMounted, onUpdated } from "vue"
import { useInforStore } from '@/stores/infor.js'
import * as d3 from 'd3'
import { valColorScheme_blue, valColorScheme_red, valColorScheme_fire, valColorScheme_double} from '@/data/index.js'

const props = defineProps({
  model_name: String,
  model_type: String,
  step_error: Object,
  model_parameters: Object
})

const inforStore = useInforStore()

const view_id = (view_str, model_name) => `${view_str}_${model_name}`

const disPlotConfig = {
  width: 272,
  height: 182,
  left_padding: 30,
  right_padding: 4,
  top_padding: 20,
  bottom_padding: 12
}

onMounted(() => {
  // getModelInfor(props.model_name)
  drawStepErrBoxPlot()
})

onUpdated(() => {
  // drawStepErrBoxPlot()
})

function drawStepErrBoxPlot() {
  let svg_id = view_id('step_error', props.model_name)
  let err_dis_svg = d3.select(`#${svg_id}`)
  err_dis_svg.selectAll('*').remove()
  let step_err_infor_list = props.step_error.step_err_infor_list
  let val_bins = props.step_error.val_bins
  let mid_bins = props.step_error.mid_bins
  let err_abs_max = Math.max(-val_bins[0], val_bins[val_bins.length-1])
  inforStore.global_err_range = [-err_abs_max, err_abs_max]
  inforStore.focus_err_range = [val_bins[0], val_bins[val_bins.length-1]]
  let mild_outliers_num_range = props.step_error.mild_outliers_num_range
  let extreme_outliers_num_range = props.step_error.extreme_outliers_num_range
  let text_h = 18
  // let val_bin_h = 16
  // if (val_bins.length > 15) val_bin_h = 11.5
  // disPlotConfig.height = val_bin_h * (val_bins.length-1) + text_h-disPlotConfig.top_padding-disPlotConfig.bottom_padding
  // err_dis_svg.attr('height', disPlotConfig.height)
  // console.log(inforStore.cur_model_parameters.output_window, step_width);
  let xScale = d3.scaleLinear()
    .domain([0.4, props.model_parameters.output_window+0.6])
    .range([disPlotConfig.left_padding, disPlotConfig.width-disPlotConfig.right_padding])
  let step_width = xScale(1.4) - xScale(0.4)
  // let yScale = d3.scaleLinear()
  //   .domain([, ])
  //   .range([step_width/2, disPlotConfig.width-step_width/2])
  err_dis_svg.append('text')
    .attr('x', 0)
    .attr('y', 12)
    .attr('text-anchor', 'start')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('Residuals')
  err_dis_svg.append('text')
    .attr('x', disPlotConfig.width/2)
    .attr('y', disPlotConfig.height-2)
    .attr('text-anchor', 'middle')
    .style('font-size', '12px')
    .attr('fill', '#333')
    .text('Forecast step')
  let xAxis = d3.axisBottom(xScale)
    .ticks(inforStore.model_parameters.output_window)
    .tickFormat(d => d)
  let xAxis_g = err_dis_svg.append("g")
    .attr("transform", `translate(0, ${disPlotConfig.height-text_h-disPlotConfig.bottom_padding})`) // 将X轴移至底部
    .call(xAxis);
  let yScale = d3.scaleLinear()
    .domain([0, val_bins.length-1])
    .range([disPlotConfig.height-text_h-disPlotConfig.top_padding-disPlotConfig.bottom_padding, 0])
  let range_height = yScale(0) - yScale(1)
  let yMidScale = d3.scaleLinear()
    .domain([mid_bins[0], mid_bins[mid_bins.length-1]])
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
  let mildRScale = d3.scaleQuantize()
    .domain(mild_outliers_num_range)
    .range([0.18, 0.3, 0.42])
  let extremeRScale = d3.scaleQuantize()
    .domain(extreme_outliers_num_range)
    .range([0.18, 0.3, 0.42])
  
  let brush = d3.brushY()
    .extent([[0, 0], [disPlotConfig.left_padding, disPlotConfig.height-text_h-disPlotConfig.bottom_padding-disPlotConfig.top_padding]])
    // .handleSize(xScale(inforStore.step_len))
    .on('brush', brushMove)
    .on("end", brushEnded);
  let brush_g = err_dis_svg.append("g")
    .attr('transform', `translate(0, ${disPlotConfig.top_padding+0.5})`)
    .call(brush)
    // .call(brush.move, defaultExtent);
  let brushing = false; // 添加一个标志位

  function brushMove(e) {
    // console.log(e);
    if (e && !brushing) {
      brushing = true; // 设置标志位，防止递归调用
      let selection = e.selection;
      let step = yScale(0) - yScale(1)
      let y0 = Math.floor(selection[0] / step) * step;
      let y1 = Math.ceil(selection[1] / step) * step;
      // 更新选择框的位置
      brush_g.call(brush.move, [y0, y1]);
      brushing = false;
    }
  }
  
  function brushEnded(e) {
    let selection = e.selection;
    if (selection) {
      let y0 = yScale.invert(selection[0]);
      let y1 = yScale.invert(selection[1]);
      let y0_int = parseInt(Math.round(y0))
      let y1_int = parseInt(Math.round(y1))

      inforStore.focus_err_range = [val_bins[y1_int], val_bins[y0_int]]
      console.log(inforStore.focus_err_range);
      // 在这里可以执行你的操作，例如根据范围重新渲染图表等
    } else {
      inforStore.focus_err_range = [val_bins[0], val_bins[val_bins.length-1]]
    }
  }

  for (let i = 1; i < step_err_infor_list.length+1; ++i) {
    let step_err_infor = step_err_infor_list[i-1]
    let step_err_box = err_dis_svg.append('g').attr('class', 'step-err-box')
      .attr('transform', `translate(0, ${disPlotConfig.top_padding})`)    
    // 绘制箱线图
    step_err_box.append('line')
      .attr('x1', xScale(i))
      .attr('x2', xScale(i))
      .attr('y1', yMidScale(step_err_infor.lower_whisker))
      .attr('y2', yMidScale(step_err_infor.upper_whisker))
      .attr('stroke', '#999')
    step_err_box.append('rect')
      .attr('x', d => xScale(i) - 0.4 * step_width)
      .attr('y', d => yMidScale(step_err_infor.percentiles[2]))
      .attr('width', d => step_width * 0.8)
      .attr('height', d => yMidScale(step_err_infor.percentiles[0]) - yMidScale(step_err_infor.percentiles[2]))
      .attr('fill', '#cecece')
      .attr('stroke', '#999')
    step_err_box.append('line')
      .attr('x1', d => xScale(i) - 0.4 * step_width)
      .attr('x2', d => xScale(i) + 0.4 * step_width)
      .attr('y1', d => yMidScale(step_err_infor.percentiles[1]))
      .attr('y2', d => yMidScale(step_err_infor.percentiles[1]))
      .attr('stroke', '#333')
    step_err_box.append('line')
      .attr('x1', d => xScale(i) - 0.4 * step_width)
      .attr('x2', d => xScale(i) + 0.4 * step_width)
      .attr('y1', d => yMidScale(step_err_infor.lower_whisker))
      .attr('y2', d => yMidScale(step_err_infor.lower_whisker))
      .attr('stroke', '#333')
    step_err_box.append('line')
      .attr('x1', d => xScale(i) - 0.4 * step_width)
      .attr('x2', d => xScale(i) + 0.4 * step_width)
      .attr('y1', d => yMidScale(step_err_infor.upper_whisker))
      .attr('y2', d => yMidScale(step_err_infor.upper_whisker))
      .attr('stroke', '#333')
    
    // 绘制异常标记线
    // step_err_box.append('line')
    //   .attr('x1', d => xScale(0.4))
    //   .attr('x2', d => xScale(inforStore.cur_model_parameters.output_window+0.6))
    //   .attr('y1', d => yScale(1))
    //   .attr('y2', d => yScale(1))
    //   .attr('stroke', '#a50f15')
    //   .attr('stroke-width', 0.2)
    //   .attr('stroke-dasharray', '5,5')
    step_err_box.append('line')
      .attr('x1', d => xScale(0.4))
      .attr('x2', d => xScale(props.model_parameters.output_window+0.6))
      .attr('y1', d => yScale(val_bins.length-3))
      .attr('y2', d => yScale(val_bins.length-3))
      .attr('stroke', '#ef3b2c')
      .attr('stroke-width', 0.2)
      .attr('stroke-dasharray', '5,5')
    // step_err_box.append('line')
    //   .attr('x1', d => xScale(0.4))
    //   .attr('x2', d => xScale(inforStore.cur_model_parameters.output_window+0.6))
    //   .attr('y1', d => yScale(val_bins.length-2))
    //   .attr('y2', d => yScale(val_bins.length-2))
    //   .attr('stroke', '#a50f15')
    //   .attr('stroke-width', 0.2)
    //   .attr('stroke-dasharray', '5,5')
    step_err_box.append('line')
      .attr('x1', d => xScale(0.4))
      .attr('x2', d => xScale(props.model_parameters.output_window+0.6))
      .attr('y1', d => yScale(2))
      .attr('y2', d => yScale(2))
      .attr('stroke', '#ef3b2c')
      .attr('stroke-width', 0.2)
      .attr('stroke-dasharray', '5,5')
    
    // 绘制异常点
    let outlier_color = '#999'
    let outlier_stroke_width = 1
    step_err_box.append('circle')
      .attr('cx', d => xScale(i))
      .attr('cy', d => (yScale(1) + range_height / 2))
      .attr('r', d => extremeRScale(step_err_infor.extreme_neg_outliers_num)*range_height)
      .attr('fill', outlier_color)
      .attr('stroke', 'none')
    step_err_box.append('circle')
      .attr('cx', d => xScale(i))
      .attr('cy', d => (yScale(2) + range_height / 2))
      .attr('r', d => mildRScale(step_err_infor.mild_neg_outliers_num)*range_height - outlier_stroke_width/2)
      .attr('fill', '#fff')
      .attr('stroke', outlier_color)
      .attr('stroke-width', outlier_stroke_width)
    step_err_box.append('circle')
      .attr('cx', d => xScale(i))
      .attr('cy', d => (yScale(val_bins.length-1) + range_height / 2))
      .attr('r', d => extremeRScale(step_err_infor.extreme_pos_outliers_num)*range_height)
      .attr('fill', outlier_color)
      .attr('stroke', 'none')
    step_err_box.append('circle')
      .attr('cx', d => xScale(i))
      .attr('cy', d => (yScale(val_bins.length-2) + range_height / 2))
      .attr('r', d => mildRScale(step_err_infor.mild_pos_outliers_num)*range_height - outlier_stroke_width/2)
      .attr('fill', '#fff')
      .attr('stroke', outlier_color)
      .attr('stroke-width', outlier_stroke_width)
  }
}

</script>

<template>
  <div class="model-card-block" :id="view_id('model-card', model_name)">
    <!-- <div>{{ model_name }}</div> -->
    <Popper placement="right">
      <span class="model_name" :id="view_id('model-name', model_name)">{{ model_name }}</span>
      <template #content>
        <div class="model-tooltip-title">{{ model_name }}</div>
        <div>Year: <span class="model-tooltip-text">{{ inforStore.model_infor[model_type].Year }}</span></div>
        <div>Publication: <span class="model-tooltip-text">{{ inforStore.model_infor[model_type].Publication }}</span></div>
        <div>Introduction: <span class="model-tooltip-text">{{ inforStore.model_infor[model_type].Introduction }}</span></div>
        <div>Spatial_module: <span class="model-tooltip-text">{{ inforStore.model_infor[model_type].Modules.Spatial_module }}</span></div>
        <div>Temporal_module: <span class="model-tooltip-text">{{ inforStore.model_infor[model_type].Modules.Temporal_module }}</span></div>
      </template>
    </Popper>
    <svg class="step-err-svg" :id="view_id('step_error', model_name)" width="272" height="182"></svg>
  </div>
  
</template>

<style scoped>
.model-card-block {
  border: 1px solid #cecece; /* 灰色边框 */
  border-radius: 8px;
  transition: box-shadow 0.1s; /* 添加过渡效果 */
  cursor: pointer;
  width: 282px;
  margin: 0 auto;
  margin-top: 2px;
  margin-bottom: 6px;
  padding: 5px;
}

.model-card-block:hover {
  box-shadow: 0 0 6px rgba(180, 180, 180, 0.8); /* 添加一圈混色荧光 */
}

.model_name {
  text-decoration: underline;
  cursor: pointer;
}

.step-err-svg {
  margin: 0 auto;
}

.model-selected {
  border: 2px solid #1a73e8; /* 蓝色边框 */
}
</style>