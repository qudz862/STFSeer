<script setup>
import { ref, reactive, computed, watch, onMounted, onUpdated } from "vue"
import { useInforStore } from '@/stores/infor.js'
import getData from '@/services/index.js'
import * as d3 from 'd3'
import { valColorScheme_blue, valColorScheme_red, valColorScheme_fire } from '@/data/index.js'

const inforStore = useInforStore()

watch (() => inforStore.cur_phase_data, (oldValue, newValue) => {
  drawPhaseSeqOverview()
})

function drawPhaseSeqOverview() {
  d3.select('#phase-seq-overview').selectAll("*").remove()
  let margin_left = 10, margin_right = 10, margin_top = 10, margin_bottom = 10
  let phase_raw_data = inforStore.cur_phase_data.phase_raw_data
  let phase_events = inforStore.cur_filtered_phases[inforStore.cur_phase_sorted_id].evolution_events
  let event_row_h = 80
  let cell_w = 80, cell_h = 80
  let main_w = phase_raw_data.length * (cell_w+2)
  let main_h = cell_h + event_row_h
  let svg_w = main_w + margin_left + margin_right
  let svg_h = main_h + margin_top + margin_bottom
  let svg = d3.select('#phase-seq-overview')
    .attr('width', svg_w)
    .attr('height', svg_h)
    .append("g")
    .attr("transform", `translate(${margin_left},${margin_top})`)
  let event_row = svg.append('g')
  let event_cells = event_row.selectAll('g')
    .data(phase_events)
    .join('g')
      .attr("transform", (d,i) => `translate(${d.step_id*(cell_w+2)}, 0)`)
  let area_event_g = event_cells.append('g')
  let h_gap = 1
  area_event_g.append('text')
    .attr('x', cell_w/2)
    .attr('y', (d,i) => {
      if (i >= 1 && phase_events[i].step_id == phase_events[i-1].step_id ) h_gap += 1
      else h_gap = 1
      return h_gap*20
    })
    .attr('text-anchor', 'middle')
    .style('font-size', '10px')
    .attr('fill', '#333')
    .text((d,i) => `${d.area_event.type}/${d.area_event.cur_grid_num}/${d.union_locs.length}`)
  // let intensity_events_g = event_cells.append('g')
  // let intensity_event_g = intensity_events_g.append('text')
  //   .attr("transform", (d,i) => `translate(0, ${40+i*20})`)  
  //   .attr('x', cell_w/2)
  //   .attr('y', 0)
  //   .attr('text-anchor', 'middle')
  //   .style('font-size', '12px')
  //   .attr('fill', '#333')
  //   .text((d,i) => `${d.intensity_event.change_type}`)


  let data_seq = svg.append('g')
    .attr("transform", `translate(0,${event_row_h})`)

  let loc_coords_x = inforStore.cur_data_infor.space.loc_list.map(item => item.geometry.coordinates[0])
  let loc_coords_y = inforStore.cur_data_infor.space.loc_list.map(item => item.geometry.coordinates[1])
  let white_rate = 0.08
  let loc_x_scale = d3.scaleLinear()
        .domain([Math.min(...loc_coords_x), Math.max(...loc_coords_x)])
        .range([cell_w*(white_rate), cell_w*(1-white_rate)])
  let loc_y_scale = d3.scaleLinear()
        .domain([Math.min(...loc_coords_y), Math.max(...loc_coords_y)])
        .range([cell_h*(1-white_rate), cell_h*white_rate])
  let valColor = d3.scaleLinear()
    .domain([0, 300])
    .range(['#efedf5', '#4a1486'])

  let data_cells = data_seq.selectAll('g')
    .data(phase_raw_data)
    .join('g')
      .attr('transform', (d,i) => `translate(${i*(cell_w+2)}, 0)`)

  data_cells.append('rect')
    .attr('x', 0).attr('y', 0)
    .attr('width', cell_w).attr('height', cell_h)
    .attr('fill', 'none')
    .attr('stroke', (d,i) => '#bababa')
  data_cells.selectAll('circle')
    .data(d => d)
    .join('circle')
      .attr('cx', (d,i) => loc_x_scale(loc_coords_x[i]))
      .attr('cy', (d,i) => loc_y_scale(loc_coords_y[i]))
      .attr('loc_id', (d,i) => i)
      .attr('r', 1.5)
      .attr('fill', (d,i) => valColor(d))
      .attr('stroke', (d,i) => {
        if (d >= inforStore.dataset_configs.focus_th) return '#333'
        else return 'none'
      })
}

</script>

<template>
  <svg id="phase-seq-overview"></svg>
</template>

<style scoped>

</style>