<script setup>
import { ref, reactive, computed, watch, onMounted, onUpdated, getCurrentInstance } from "vue";
import getData from '@/services/index.js'
import mapboxgl from "mapbox-gl";
import turf from 'turf'
import MapboxDraw from "@mapbox/mapbox-gl-draw"
import { RulerControl, CompassControl, StylesControl, ZoomControl, InspectControl } from 'mapbox-gl-controls'
import { useInforStore } from '@/stores/infor.js'
import * as d3 from 'd3'
import $ from 'jquery'

const inforStore = useInforStore()

// 地图相关设置
let map
let draw
let selPointList = []

let sel_sets = ref(['train', 'val', 'test'])

let space_infor = ref({})
let time_infor = ref({})
// let feature_infor = ref({})

let sel_start_time = ref("")
let sel_end_time = ref("")


onMounted(() => {
  // draw_th_sketch()
  
})

getData(inforStore, 'existed_task_data_model')

let sel_task_name = ref('air_quality_pred')
let sel_task_id = ref(-1)

let sel_data_name = ref('Choose Data')
let cur_data_names = ref([])
let sel_data_id = ref(-1)

watch (() => inforStore.existed_task_data_model, (oldValue, newValue) => {
  cur_data_names.value = Object.keys(inforStore.existed_task_data_model[sel_task_name.value])
})

const onDataCardClick = card_index => {
  if (card_index == -1) {
    sel_data_id.value = -1
    sel_data_name.value = "Choose Data"
    // inforStore.cur_model_names = []
    inforStore.cur_window_sizes = []
    inforStore.cur_sel_data = ""
  }
  else {
    sel_data_id.value = card_index
    sel_data_name.value = cur_data_names.value[card_index]
    // inforStore.cur_model_names = inforStore.existed_task_data_model[sel_task_name.value][sel_data_name.value].models
    let window_strs = Object.keys(inforStore.existed_task_data_model[sel_task_name.value][sel_data_name.value].forecast_steps)
    let window_objs = []
    for (let i = 0; i < window_strs.length; ++i) {
      let split_windows = window_strs[i].split('-')
      let window_obj = {"Input Steps": split_windows[0], "Forecast Steps": split_windows[1], "Lead Steps": split_windows[2]}
      window_objs.push(window_obj)
    }
    inforStore.cur_window_sizes = window_objs
    inforStore.cur_sel_data = sel_data_name.value

    getData(inforStore, 'cur_data_infor', sel_task_name.value, sel_data_name.value)
  }
}

mapboxgl.accessToken = 'pk.eyJ1IjoiZGV6aGFudmlzIiwiYSI6ImNraThnYWoxcDA1aXkycnMzMGxhcDcxeGgifQ.pbnOr8oKR894OJ3seHIayg'

const initDraw = () => {
    draw = new MapboxDraw()
    // let navigation = new mapboxgl.NavigationControl()
    // let scale_ctl =  new mapboxgl.ScaleControl({
    //     maxWidth: 80,
    //     unit: 'imperial'
    // })
    let full_screen_ctl = new mapboxgl.FullscreenControl()
    // map.addControl(navigation, 'bottom-right')
    map.addControl(full_screen_ctl, 'bottom-left')
    // map.addControl(scale_ctl, 'bottom-right')

    map.addControl(draw, 'bottom-left')

    map.on('draw.create', updateArea)
    map.on('draw.update', updateArea)
    map.on('draw.delete', updateArea)
}

watch (() => inforStore.cur_sel_task, (oldValue, newValue) => {
  cur_error_configs.value = inforStore.error_configs[inforStore.cur_sel_task]
})

watch (() => inforStore.cur_config_file, (oldValue, newValue) => {
  getData(inforStore, 'dataset_configs', inforStore.cur_sel_data, inforStore.cur_config_file)
})

watch (() => inforStore.dataset_configs, (oldValue, newValue) => {
  // getData(inforStore, 'st_phase_events', inforStore.cur_sel_data, inforStore.dataset_configs.focus_th, JSON.stringify(inforStore.dataset_configs.phase_params), JSON.stringify(inforStore.dataset_configs.focus_levels), JSON.stringify(inforStore.dataset_configs.event_params))
})

watch (() => inforStore.cur_data_infor, (oldValue, newValue) => {
  console.log(inforStore.cur_data_infor);
  inforStore.cur_config_file = inforStore.cur_data_infor.default_config_file
  space_infor.value.loc_num = inforStore.cur_data_infor.space.loc_list.length
  space_infor.value.type = inforStore.cur_data_infor.space.loc_list[0].properties.type + 's'
  time_infor.value.time_num = inforStore.cur_data_infor.time.time_num
  time_infor.value.input_window = inforStore.cur_data_infor.time.input_window
  time_infor.value.output_window = inforStore.cur_data_infor.time.output_window
  time_infor.value.start_time = inforStore.dataset_infor[inforStore.cur_sel_data].Duration.start
  time_infor.value.end_time = inforStore.dataset_infor[inforStore.cur_sel_data].Duration.end
  let range_num = inforStore.dataset_infor[inforStore.cur_sel_data].Duration.length
  time_infor.value.time_range = inforStore.dataset_infor[inforStore.cur_sel_data].Duration
  inforStore.cur_sel_time = {
    start: time_infor.value.start_time,
    end: time_infor.value.end_time
  }
  sel_start_time.value = time_infor.value.start_time
  sel_end_time.value = time_infor.value.end_time
  inforStore.feature_infor.num = inforStore.cur_data_infor.features.num
  inforStore.feature_infor.input = inforStore.cur_data_infor.features.input.join(', ')
  inforStore.feature_infor.output = inforStore.cur_data_infor.features.output.join(', ')

  map = new mapboxgl.Map({
      container: 'map-container',
      // style: 'mapbox://styles/dezhanvis/ckmcv57z60gjd17rq1jvowlcr',
      center: inforStore.cur_data_infor.space.loc_center,
      // pitch: 80,
      // bearing: 41,
      'zoom': 6,
      style: 'mapbox://styles/mapbox/satellite-streets-v10',
      zoomControl: true
    })
  map._logoControl && map.removeControl(map._logoControl);  // 去除mapbox标志
  map.on('style.load', () => {
    map.addSource('mapbox-dem', {
      'type': 'raster-dem',
      'url': 'mapbox://mapbox.mapbox-terrain-dem-v1',
      'tileSize': 256,
      'maxzoom': 14
      });
    map.setTerrain({ 'source': 'mapbox-dem', 'exaggeration': 4 });

    map.addSource("points", {
      "type": "geojson",
      "data": {
        "type": "FeatureCollection",
        "features": inforStore.cur_data_infor.space.loc_list
      }
    })

    map.addSource("grid_borders", {
      "type": "geojson",
      "data": {
        "type": "FeatureCollection",
        "features": inforStore.cur_data_infor.space.grid_border_geojson
      }
    })

    map.addLayer({
      "id": "points",
      "type": "circle",
      "source": "points",
      "paint": {
        'circle-radius': 4.5,
        'circle-color': '#ffc107'
      }
    });

    map.addLayer({
      "id": "points-selected",
      "type": "circle",
      "source": "points",
      "paint": {
        'circle-radius': 4.5,
        'circle-color': '#eb6877'
      },
      "filter": ["in", "loc_id", '']  /* 过滤器，名字为空的数据才显示，也就是默认不使用该layer  */
    });

    map.addLayer({
      "id": "grids",
      "type": "line",   /* symbol类型layer，一般用来绘制点*/
      "source": "grid_borders",
      paint: {
        'line-color': "#ffffff",
        "line-width": 1.5
      }
    })
  });

  initDraw()

  // map.on('click', 'points', function(e) {
  //   let coordinates = e.features[0].geometry.coordinates.slice();
    
  //   // 在这里你可以使用 coordinates 进行你的操作，比如显示在页面上
  //   console.log('Hovered Coordinates:', coordinates);
  // })

  // $('.attr-value').css('font-weight', 700)
  // console.log(inforStore.cur_data_infor);
})

const updateArea = (e) => {
    //   map.setFilter("points-selected", ["==", "group", ""]);
  if (e.type === 'draw.delete') {
    map.setFilter("points-selected", ["in", "loc_id", ""]);
    // map.setFilter("points-selected", ["==", "group", ""]);
    selPointList = []
    // store.commit('setData', {
    //   field: 'select_point',
    //   data: pointList
    // })
    return
  }

  let data = draw.getAll()
  if (data.features.length > 0) {
    let userPolygon = e.features[0];
    let polygonBoundingBox = turf.bbox(userPolygon);
    let southWest = [polygonBoundingBox[0], polygonBoundingBox[1]];
    let northEast = [polygonBoundingBox[2], polygonBoundingBox[3]];
    let northEastPointPixel = map.project(northEast);
    let southWestPointPixel = map.project(southWest);
    let features = map.queryRenderedFeatures([southWestPointPixel, northEastPointPixel], { layers: ['points'] });
    let filter = features.reduce(function(memo, feature) {
      if (! (undefined === turf.intersect(feature, userPolygon))) {
        // only add the property, if the feature intersects with the polygon drawn by the user
        // memo.push(feature.properties.title);
        memo.push(feature.properties.loc_id);
      }
      return memo;
    }, []);
    // sel_filter = ["in", "loc_id", ...filter]
    selPointList = filter
    // store.commit('setData', {
    //   field: 'select_point',
    //   data: pointList
    // })
    map.setFilter("points-selected", ["in", "loc_id", ...filter]);
  } else {
    // answer.innerHTML = '';
  }
}

function selConfigFile(item) {
  inforStore.cur_config_file = item.config_name
}

function drawTargetValDist() {
  
}
</script>

<template>
  <div class="select-block">
    <span class="iconfont data-icon select-icon">&#xe603;</span>
    <div class="data-dropdown">
      <button class="btn dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false">{{ sel_data_name }}</button>
      <ul class="dropdown-menu" id="dropdown-choose-task">
        <li v-for="(item, index) in cur_data_names" :value="item" @click="onDataCardClick(index)" class='dropdown-item' :key="index">
          <div class="li-data-name">{{ item }} - {{ inforStore.dataset_infor[item].Data_type }}</div>
          <div class="li-data-description">{{ inforStore.dataset_infor[item].Description }}</div>
        </li>
      </ul>
    </div>
    <Popper placement="right">
      <div class="data-infor-label iconfont">&#xe744;</div>
      <template #content>
        <div class="data-infor-tooltip">
          <div style="font-size: 20px; font-weight: 700; margin: 6px 0 0 10px">Dataset Information</div>
          <div class="data_infor_title"><span class="attr-title">Space</span>: <span class="attr-value">{{ space_infor.loc_num }} {{ space_infor.type }}</span></div>
          <div class="map-container" id="map-container"></div>
          <div class="data_infor_title">
            <div><span class="attr-title">Time</span>: <span class="attr-value">{{ time_infor.start_time }} ~ {{ time_infor.end_time }} ({{ time_infor.time_num }} timestamps)</span></div>  
          </div> 
          <!-- <div class="data_infor_block">
            <div class="data-infor-row">
              <div>Train: <span class="attr-value">{{ time_infor.train_range }}</span></div>
              <div style="display: flex;">
                <div style="margin-right: 16px;">Valid: <span class="attr-value">{{ time_infor.valid_range }}</span></div>
                <div>Test: <span class="attr-value">{{ time_infor.test_range }}</span></div>
              </div>
            </div>
          </div> -->
          <div class="data_infor_title">
            <div><span class="attr-title">Feature</span>: {{ inforStore.feature_infor.num }} features</div>
          </div>
          <div class="data_infor_block">
            <div>Input: <span class="attr-value">{{ inforStore.feature_infor.input }}</span></div>
            <div>Output: <span class="attr-value">{{ inforStore.feature_infor.output }}</span></div>
          </div>
        </div>
      </template>
    </Popper>
  </div>
  <div class="data_infor_title">
    
  </div>
  <!-- <div class="select-block">
    <span class="iconfont select-icon">&#xe60a;</span>
    <div class="data-dropdown">
      <button class="btn dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false">
        {{ inforStore.cur_config_file }}
      </button>
      <ul class="dropdown-menu" v-if="'config_files' in inforStore.cur_data_infor">
        <li v-for="(item, index) in inforStore.cur_data_infor.config_files" :value="item" class='dropdown-item' :key="index" @click="selConfigFile(item)">
          <div class="li-data-name">{{ item }}</div>
        </li>
      </ul>
    </div>
  </div> -->
  <div class="select-block">
    <label class="form-label"><span class="attr-title">Configs: </span></label>
    <div class="config-dropdown">
      <button class="btn dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false">{{ inforStore.cur_config_file }}</button>
      <ul class="dropdown-menu" v-if="'config_files' in inforStore.cur_data_infor">
        <li v-for="(item, index) in inforStore.cur_data_infor.config_files" :value="item" class='dropdown-item' :key="index" @click="selConfigFile(item)">
          <div class="li-data-name">{{ item.config_name }}</div>
          <div class="li-data-description">{{ item.descriptions }}</div>
        </li>
      </ul>
    </div>
    <Popper placement="right">
      <div class="iconfont config-icon">&#xe74d;</div>
      <template #content>
        <!-- 设置model failure的rules -->
          aaa
      </template>
    </Popper>
  </div>
  <div class="config-seg-line"></div>
  <!-- <svg id="target-val-dist"></svg> -->
  <!-- <div class="config-seg-line"></div> -->
</template>

<style scoped>
@import url("https://api.tiles.mapbox.com/mapbox-gl-js/v2.14.1/mapbox-gl.css");
@import url("https://api.mapbox.com/mapbox-gl-js/plugins/mapbox-gl-draw/v1.3.0/mapbox-gl-draw.css");

.select-block {
  display: flex;
  margin-left: 10px;
  align-items: center;
  margin-top: 8px;
  margin-bottom: -4px;
}

.select-icon {
  margin-top: 2px;
  font-size: 20px;
  margin-right: 6px;
}
.config-dropdown .dropdown-toggle {
  color: #1a73e8;
  width: 200px !important;
  height: 24px;
  margin-top: -10px;
  /* margin-bottom: -6px; */
  /* width: 120px; */
  padding: 0px 2px 0 4px !important;
  border-bottom: solid 1px #9c9c9c;
  border-radius: 0;
  font-size: 14px;
  /* text-align: left; */
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.config-dropdown .dropdown-toggle::after {
    margin-left: 0.6em !important;
}

.config-dropdown .dropdown-item {
  border-bottom: solid 1px #cecece;
  font-size: 14px;
  width: 268px;
  cursor: pointer;
  white-space: normal;
}

.config-dropdown .dropdown-item:hover {
  background-color: #cecece;
}


.dropdown-toggle::after {
    margin-left: 0.6em !important;
}

.dropdown-item {
  border-bottom: solid 1px #cecece;
  font-size: 14px;
  max-width: 480px;
  cursor: pointer;
  white-space: normal;
}

.dropdown-item:hover {
  background-color: #cecece;
}

.li-data-name {
    font-size: 14px;
}

.li-data-description {
    font-size: 12px;
    color: #777;
}

.config-icon {
  margin-left: 7px;
  font-size: 20px;
  color: #777;
}
.config-icon:hover {
  cursor: pointer;
  color:#1a73e8;
}

.mapboxgl-ctrl-fullscreen {
    /* background-color: #000 !important; */
    background-image: url("/src/assets/MapIcons/mapboxgl-ctrl-fullscreen.svg") !important;
}
.map-container {
  /* margin: 0 auto; */
  /* left: 300; */
  /* display: flex; */
  width: 324px !important;
  height: 306px !important;
  margin-left: 10px;
  margin-top: 3px;
  border: solid 1px #bcbcbc;
}

.data_infor_title {
  margin-top: 3px;
  margin-left: 10px;
  font-size: 14px;
  width: 276px;
  display: flex;
  /* justify-content: space-between; */
}

.data_infor_block {
  /* display: flex; */
  /* justify-content: space-between; */
  margin-left: 10px;
  font-size: 14px;
  font-weight: 400;
  margin-top: 1px;
}

.data-infor-row {
  width: 286px;
  /* display: flex; */
  /* justify-content: space-between; */
  padding-right: 12px;
}

.attr-value {
  color: #1a73e8;
}
.attr-title {
  font-weight: 700;
}

.form-row {
  display: flex;
  /* margin-left: 10px; */
  align-items: center;
  margin-bottom: 6px;
  /* width: 280px; */
  /* justify-content: space-between; */
}

.time-pre-text {
  width: 46px;
  text-align: left;
}

.data-dropdown .dropdown-toggle,
.task-dropdown .dropdown-toggle {
  color: #1a73e8;
  width: 230px !important;
  max-width: 230px;
  height: 30px;
  /* width: 120px; */
  padding: 0px 2px 0 4px !important;
  border-bottom: solid 1px #9c9c9c;
  border-radius: 0;
  font-size: 14px;
  /* text-align: left; */
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.th-form-row,
.config-load-row {
  width: 276px;
  height: 26px;
  margin-left: 10px;
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  font-size: 14px;
  justify-content: space-between;
  align-items: center;
}

.time-scope-control {
  width: 180px;
  height: 24px;
  display: flex;
  align-items: center;
}
.config-load-row .form-control {
  width: 168px;
  height: 22px !important;
  /* width: 120px; */
  padding: 2px 4px !important;
  margin-left: 4px;
  border: none;
  border-bottom: solid 1px #9c9c9c;
  border-radius: 0;
  font-size: 14px;
  text-align: center;
  color:#1a73e8;
  /* text-align: left; */
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.th-form-row label {
  margin-top: 8px;
  display: block;
}

.focus-type-dropdown .dropdown-toggle {
  width: 130px !important;
  height: 30px;
  /* width: 120px; */
  margin-top: -6px;
  padding: 6px 2px 0 4px !important;
  border-bottom: solid 1px #9c9c9c;
  border-radius: 0;
  font-size: 14px;
  /* text-align: left; */
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

/* .dropdown-menu,
.dropdown-item {
  font-size: 14px;
  width: 130px !important;
  cursor: pointer;
  white-space: normal;
} */


.config-seg-line {
  height: 1px;
  width: 274px;
  background-color: #bcbcbc;
  margin: 0 auto;
  margin-top: 8px;
  margin-bottom: 8px;
}

#save-cur-config {
  
  font-size: 14px;
  font-weight: 700;
  color: #777;
  text-decoration: underline;
}
#save-cur-config:hover {
  color: #1a73e8;
  cursor: pointer;
}

.error-config-title {
  display: flex;
  /* justify-content: space-between; */
  align-items: center;
}

.error-config-title span:first-child {
  margin-right: 6px;
}

.save-config-description {
  font-size: 14px;
}

.li-config-description {
  font-size: 12px;
  color: #777;
}
.save-config-name {
  font-size: 14px;
  display: flex;
  align-items: center;
  margin-bottom: 6px;
}

.save-config-name .form-control {
  width: 190px !important;
  height: 22px;
  /* width: 120px; */
  margin-left: 4px;
  padding: 0px 2px 0 4px !important;
  border: none;
  border-bottom: solid 1px #9c9c9c;
  border-radius: 0;
  font-size: 14px;
  text-align: center;
}

.save-config-row {
  margin-top: 6px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.save-config-state {  
  font-size: 14px;
  /* font-weight: 700; */
  color: #157347;
}

#save-config-btn {
  font-size: 14px;
  padding: 3px 6px !important;
}

.data-infor-label {
  /* text-decoration: underline; */
  /* font: italic; */
  font-weight: 700;
  color: #9c9c9c;
  cursor: pointer;
  font-size: 19px;
  margin-left: 7px;
}

.data-infor-label:hover {
  color: #1a73e8;
}
</style>