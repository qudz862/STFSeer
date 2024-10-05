import {
  getExistedTaskDataModel,
  getCurDataInfor,
  getDatasetConfigs,
  getModelInfor,
  getModelParameters,
  processModelPreds,
  getGridBorders,
  getSTPhaseEvents,
  getPhasesIndicators,
  getPhaseDetails,
  get_error_distributions,
  get_attr_distributions,
  get_attr_bin_errors,
  getInstanceSeqs,
  getLocFeatures,
  getPhaseData,
  getEventSubsets,
  getPointSubsets,
  get_slices_indices,
  getModelErrMtx,
  getDataAugmentation,
  getInputOutputData,
  saveSubsetCollection,
  getLocInstanceInfor,
  save_subgroup_collections,
  save_phase_collections
} from './request'

import { cityGroup } from '@/data'

import { strToJson } from '@/libs/utils'
import { json } from 'd3-fetch'
import * as d3 from 'd3'

export default async (store, field, ...args) => {
  let data = null
  let res = null
  switch (field) {
    case 'existed_task_data_model':
      data = await getExistedTaskDataModel(...args)
      res = data
      store.$patch((state) => {
        state.existed_task_data_model = res.existed_task_data_model
        state.dataset_infor = res.dataset_infor
        state.model_infor = res.model_infor
        state.error_configs = res.error_configs
      })
      break
    case 'point_loc':
      data = await getPointLocs(...args)
      // 构建为geojson的feature格式
      res = []
      data.forEach((item,index,array)=>{
        let feature = {
          "type": "Feature",
          "geometry": {
            "type": "Point",
            "coordinates": [item['lng'], item['lat']]
          },
          "properties": {
            "loc_id": item['loc_id'],
            "focused": false,
            // "elevation": item['elevation']
          }
        }
        res.push(feature)
      })
      break
    case 'grid_borders':
      data = await getGridBorders(...args)
      res = []
      data.forEach((item,index,array)=>{
        let feature = {
          "type": "Feature",
          "geometry": {
            "type": "LineString",
            "coordinates": item['border_points']
          },
          "properties": {
            "loc_id": item['loc_id']
          }
        }
        res.push(feature)
      })
      break
    case 'cur_data_infor':
      data = await getCurDataInfor(...args)
      data = JSON.parse(data.ordered_data_str)
      let loc_geojson = []
      let grid_border_geojson = []
      let loc_region_list = []
      data.space.loc_list.forEach((item,index,array)=>{
        let feature = {
          "type": "Feature",
          "geometry": {
            "type": "Point",
            "coordinates": [item.coordinates[0], item.coordinates[1]]
          },
          "properties": {
            "loc_id": item.geo_id,
            "focused": false,
            "type": item.type
            // "elevation": item['elevation']
          }
        }
        loc_geojson.push(feature)
        const url = `https://api.mapbox.com/geocoding/v5/mapbox.places/${item.coordinates[0]},${item.coordinates[1]}.json?access_token=pk.eyJ1IjoiZGV6aGFudmlzIiwiYSI6ImNraThnYWoxcDA1aXkycnMzMGxhcDcxeGgifQ.pbnOr8oKR894OJ3seHIayg`
        fetch(url)
          .then(response => response.json())
          .then(data => {
            if (data.features && data.features.length > 0) {
              let address = data.features[1].place_name
              if (!isNaN(address[0]) && address[0] !== ' ') address = address.slice(8, address.length)
              loc_region_list.push(address)
            } else {
              console.log("No results found");
            }
          })
          .catch(error => {
            console.error("Error fetching data:", error)
          })
      })
      data.space.grid_borders.forEach((item,index,array)=>{
        let feature = {
          "type": "Feature",
          "geometry": {
            "type": "LineString",
            "coordinates": item['border_points']
          },
          "properties": {
            "loc_id": item['loc_id']
          }
        }
        grid_border_geojson.push(feature)
      })

      res = data
      res.space.loc_list = loc_geojson
      res.space.grid_border_geojson = grid_border_geojson
      store.$patch((state) => {
        state.cur_data_infor = res
        state.loc_regions = loc_region_list
      })
      // console.log(store.cur_data_infor);
      break
    case 'dataset_configs':
      data = await getDatasetConfigs(...args)
      res = data.ordered_data_str
      store.$patch((state) => {
        state.dataset_configs = JSON.parse(res)
        state.cur_st_attrs = state.dataset_configs.point_metadata
      })
      break
    case 'model_infor':
      data = await getModelInfor(...args)
      res = data
      store.$patch((state) => {
        state.model_infor = res.models_infor
        state.range_infor = res.ranges_infor
      })
      break
    case 'input_output_data':
      data = await getInputOutputData(...args)
      res = data
      // store.$patch((state) => {
      // })
      break
    case 'save_subgroup_collection':
      data = await saveSubsetCollection(...args)
      break
    case 'model_parameters':
      data = await getModelParameters(...args)
      res = data
      store.$patch((state) => {
        state.model_parameters = res.model_configs
        state.subset_collections = res.saves_collections
      })
      break
    case 'process_model_preds':
      data = await processModelPreds(...args)
      res = data
      store.$patch((state) => {
        state.process_preds_state += 1
        state.preds_num = res.preds_num
      })
      break
    case 'model_err_mtx':
      data = await getModelErrMtx(...args)
      res = data
      store.$patch((state) => {
        state.model_err_mtx = res
      })
      break
    case 'data_items':
      data = await getDataItems(...args)
      res = data
      store.$patch((state) => {
        state.cur_data_items = res
      })
      break
    case 'save_error_config':
      data = await saveErrorConfig(...args)
      res = data
      store.$patch((state) => {
        state.config_save_state = res
      })
      break
    case 'overview_indicators':
      data = await getOverviewIndicators(...args)
      res = data
      store.$patch((state) => {
        const merged_indicators = Object.assign({}, state.cur_overview_indicators, res);
        state.cur_overview_indicators = merged_indicators
      })
      break
    case 'st_patterns':
      data = await getSTPatterns(...args)
      res = data
      store.$patch((state) => {
        state.cur_st_patterns = res
      })
      break
    case 'event_series':
      data = await getEventSeries(...args)
      res = data
      store.$patch((state) => {
        state.cur_event_series = res
      })
      break
    case 'st_phase_events':
      data = await getSTPhaseEvents(...args)
      res = data
      store.$patch((state) => {
        state.st_phase_events = res
        state.phase_pcp_dims = res.phases_attr_bin_edges[state.cur_focused_model]
        state.cur_filtered_phases = res.phases[state.cur_focused_model]
        console.log('phases', res.phases);
        // state.cur_filtered_events = res.events_clean
        state.cur_filtered_phases_indices = state.cur_filtered_phases.map((item,index) => index)
        // state.cur_filtered_events = res.events_outliers
        // state.cur_filtered_events = res.events_list
        // console.log(state.cur_filtered_events.length);
        // state.cur_filtered_events = state.cur_filtered_phases.reduce((accu, obj) => accu.concat(obj.evolution_events), [])
      })
      break
    case 'phases_indicators':
      data = await getPhasesIndicators(...args)
      res = data
      store.$patch((state) => {
        state.error_indicators = res
        // state.events_indicators = res.events_indicators
        for (let key in res.phases_indicators) {
          let tmp_space_mae_range = [0, 0]
          for (let i = 0; i < res.phases_indicators[key].length; ++i) {
            let space_mae = res.phases_indicators[key][i].space_abs_residual
            let space_mae_max = d3.max(space_mae)
            if (space_mae_max > tmp_space_mae_range[1]) {
              tmp_space_mae_range[1] = space_mae_max
            }
          }
          for (let i = 0; i < res.phases_indicators[key].length; ++i) {
            res.phases_indicators[key][i]['global_space_mae_range'] = tmp_space_mae_range
          }
        }
        state.phases_indicators = res.phases_indicators
        // console.log(state.phases_indicators);
      })
      break
    case 'event_subsets':
      data = await getEventSubsets(...args)
      res = data
      store.$patch((state) => {
        // state.event_subsets = res
        state.event_attr_objs = res
      })
      break
    case 'slices_indices':
      data = await get_slices_indices(...args)
      break
    case 'find_point_slices':
      data = await getPointSubsets(...args)
      res = data
      store.$patch((state) => {
        state.phase_clean_res_abs_mean = res.clean_residual_abs_mean
        let tmp_attr_objs = {}
        for (const attr of state.dataset_configs.point_metadata) {
          tmp_attr_objs[attr] = res.meta_attr_objs[attr]
          let split_attr = attr.split('-')
          if (split_attr[0].includes('.')) {
              // 删除所有的 '.'
              split_attr[0] = split_attr[0].replace(/\./g, '');
          }
          // console.log('split_attr[0]', split_attr, split_attr[0]);
          if (split_attr[1] == "target_val") {
            if (split_attr.length == 3)
              tmp_attr_objs[attr].simple_str = `V_${split_attr[0]}_${split_attr[2]}`
            else tmp_attr_objs[attr].simple_str = `V_${split_attr[0]}`
            tmp_attr_objs[attr].icon_type = 'value'
          }
          if (split_attr[1] == "temporal_state_vals") {
            if (split_attr.length == 3)
              tmp_attr_objs[attr].simple_str = `T_${split_attr[0]}_${split_attr[2]}`
            else tmp_attr_objs[attr].simple_str = `T_${split_attr[0]}`
            tmp_attr_objs[attr].icon_type = 'time'
          }
          if (split_attr[1] == "temporal_context_mean") {
            if (split_attr.length == 3)
              tmp_attr_objs[attr].simple_str = `TM_${split_attr[0]}_${split_attr[2]}`
            else tmp_attr_objs[attr].simple_str = `TM_${split_attr[0]}`
            tmp_attr_objs[attr].icon_type = 'time'
          }
          if (split_attr[1] == "temporal_context_trend") {
            if (split_attr.length == 3)
              tmp_attr_objs[attr].simple_str = `TT_${split_attr[0]}_${split_attr[2]}`
            else tmp_attr_objs[attr].simple_str = `TT_${split_attr[0]}`
            tmp_attr_objs[attr].icon_type = 'time'
          } 
          if (split_attr[1] == "space_comp_state_vals") {
            if (split_attr.length == 3)
              tmp_attr_objs[attr].simple_str = `SC_${split_attr[0]}_${split_attr[2]}`
            else tmp_attr_objs[attr].simple_str = `SC_${split_attr[0]}`
            tmp_attr_objs[attr].icon_type = 'space'
          }
          if (split_attr[1] == "space_diff_state_vals") {
            if  (split_attr.length == 3)
              tmp_attr_objs[attr].simple_str = `SD_${split_attr[0]}_${split_attr[2]}`
            else tmp_attr_objs[attr].simple_str = `SD_${split_attr[0]}`
            tmp_attr_objs[attr].icon_type = 'space'
          }
        }
        state.meta_attr_objs = tmp_attr_objs
        state.cur_range_infor = res.range_infor
        state.cur_subsets = res.subsets
        state.cached_slice_id = res.cached_slice_id
        // console.log('cur_range_infor', state.cur_range_infor)
        // console.log('meta_attr_objs', state.meta_attr_objs)
        // console.log('cur_subsets', state.cur_subsets)
        if (state.cur_baseline_model.length > 0) state.cur_subset_model = state.cur_baseline_model
        else state.cur_subset_model = state.cur_focused_model

        // state.phase_subsets_proj = res.subsets_proj
        // state.phase_subsets_links = res.subset_links
      })
      break
    case 'phase_details':
      data = await getPhaseDetails(...args)
      res = data
      store.$patch((state) => {
        state.phase_details_infor = res
        state.sel_phase_details = res.phase_details
        state.cur_event_pcp_dims = res.phase_details[state.cur_focused_model].event_attr_bin_edges
        console.log('state.cur_event_pcp_dims', state.cur_event_pcp_dims);
      })
      break
    case 'error_distributions':
      data = await get_error_distributions(...args)
      res = data.ordered_data_str
      store.$patch((state) => {
        state.error_distributions = JSON.parse(res)
      })
      break
    case 'attr_distributions':
      data = await get_attr_distributions(...args)
      res = data
      store.$patch((state) => {
        state.attr_distributions = res
      })
      break
    case 'attr_bin_errors':
      data = await get_attr_bin_errors(...args)
      res = data
      store.$patch((state) => {
        state.cur_attr_bin_errors = res
      })
      break
    case 'phase_data':
      data = await getPhaseData(...args)
      res = data
      store.$patch((state) => {
        state.cur_phase_data = res
      })
      break
    case 'instance_seqs':
      data = await getInstanceSeqs(...args)
      res = data
      store.$patch((state) => {
        state.cur_instance_seqs = res
      })
      break
    case 'loc_features':
      data = await getLocFeatures(...args)
      res = data
      store.$patch((state) => {
        let show_features = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "U", "V", "TEMP", "RH", "PSFC"]
        let reorderedObject = {};
        show_features.forEach(function(key) {
            if (res.hasOwnProperty(key)) {
                reorderedObject[key] = res[key];
            }
        });
        state.cur_loc_features = reorderedObject
      })
      break
    case 'data_augmentation_by_slices':
      data = await getDataAugmentation(...args)
      // res = data
      // store.$patch((state) => {
      //   state.cur_instance_seqs = res
      // })
      break
    case 'loc_instance_infor':
      data = await getLocInstanceInfor(...args)
      res = data
      store.$patch((state) => {
        state.loc_instance_infor = res
      })
      break
    case 'save_subgroup_collections':
      data = await save_subgroup_collections(...args)
      break
    case 'save_subgroup_collections':
      data = await save_phase_collections(...args)
      break
    default:
      break
  }
  
  return 0
}