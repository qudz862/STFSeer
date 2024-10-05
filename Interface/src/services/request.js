import { axiosGet, axiosPost } from '@/libs/http'

const ip_address = "http://192.168.1.2:1024"
const trajectory_process_ip = "http://192.168.1.2:1036"

function getExistedTaskDataModel() {
  return new Promise((resolve, reject) => {
    axiosGet({
      url: `${ip_address}/task_data_model_infor`,
      success (data) {
        resolve(data)
      },
      error (err) {
        reject(err)
      }
    })
  })
}

function getCurDataInfor(task, data_name) {
  return new Promise((resolve, reject) => {
    axiosGet({
      url: `${ip_address}/cur_data_infor/${task}/${data_name}`,
      success (data) {
        resolve(data)
      },
      error (err) {
        reject(err)
      }
    })
  })
}

function getDatasetConfigs (data_name, config_name) {
  return new Promise((resolve, reject) => {
    axiosGet({
      url: `${ip_address}/dataset_configs/${data_name}/${config_name}`,
      success (data) {
        resolve(data)
      },
      error (err) {
        reject(err)
      }
    })
  })
}

function getInputOutputData(data_name, input_step, output_step, output_offset) {
  return new Promise((resolve, reject) => {
    axiosGet({
      url: `${ip_address}/input_output_data/${data_name}/${input_step}/${output_step}/${output_offset}`,
      success (data) {
        resolve(data)
      },
      error (err) {
        reject(err)
      }
    })
  })
}

function getModelParameters(data_name, model_names, focused_model, baseline_model) {
  return new Promise((resolve, reject) => {
    axiosGet({
      url: `${ip_address}/model_parameters/${data_name}/${model_names}/${focused_model}/${baseline_model}`,
      success (data) {
        resolve(data)
      },
      error (err) {
        reject(err)
      }
    })
  })
}

function processModelPreds(data_name, model_names, focus_th, focus_levels) {
  return new Promise((resolve, reject) => {
    axiosGet({
      url: `${ip_address}/process_model_preds/${data_name}/${model_names}/${focus_th}/${focus_levels}`,
      success (data) {
        resolve(data)
      },
      error (err) {
        reject(err)
      }
    })
  })
}

function get_error_distributions(data_name, model_names, configs, forecast_scopes, inspect_type) {
  return new Promise((resolve, reject) => {
    axiosGet({
      url: `${ip_address}/error_distributions/${data_name}/${model_names}/${configs}/${forecast_scopes}/${inspect_type}`,
      success (data) {
        resolve(data)
      },
      error (err) {
        reject(err)
      }
    })
  })
}

function get_attr_distributions(data_name, model_names, configs, forecast_scopes, attr_bin_edges, select_ranges) {
  return new Promise((resolve, reject) => {
    axiosGet({
      url: `${ip_address}/attr_distributions/${data_name}/${model_names}/${configs}/${forecast_scopes}/${attr_bin_edges}/${select_ranges}`,
      success (data) {
        resolve(data)
      },
      error (err) {
        reject(err)
      }
    })
  })
}

function get_attr_bin_errors(data_name, model_names, subset_id, configs, forecast_scopes) {
  return new Promise((resolve, reject) => {
    axiosGet({
      url: `${ip_address}/attr_bin_errors/${data_name}/${model_names}/${subset_id}/${configs}/${forecast_scopes}`,
      success (data) {
        resolve(data)
      },
      error (err) {
        reject(err)
      }
    })
  })
}

function getModelInfor(data_name, model_name, failure_rules, scope_seg_th) {
  return new Promise((resolve, reject) => {
    axiosGet({
      url: `${ip_address}/model_infor/${data_name}/${model_name}/${failure_rules}/${scope_seg_th}`,
      success (data) {
        resolve(data)
      },
      error (err) {
        reject(err)
      }
    })
  })
}

function getPhaseData(phase_id) {
  return new Promise((resolve, reject) => {
    axiosGet({
      url: `${ip_address}/phase_data/${phase_id}`,
      success (data) {
        resolve(data)
      },
      error (err) {
        reject(err)
      }
    })
  })
}

function getSTPhaseEvents(data_name, model_names, focus_th, min_length, bin_list, event_params, forecast_scopes) {
  return new Promise((resolve, reject) => {
    axiosGet({
      url: `${ip_address}/st_phase_events/${data_name}/${model_names}/${focus_th}/${min_length}/${bin_list}/${event_params}/${forecast_scopes}`,
      success (data) {
        resolve(data)
      },
      error (err) {
        reject(err)
      }
    })
  })
}

function getPhasesIndicators(data_name, model_names, bin_edges, event_params, focused_scope) {
  return new Promise((resolve, reject) => {
    axiosGet({
      url: `${ip_address}/phases_indicators/${data_name}/${model_names}/${bin_edges}/${event_params}/${focused_scope}`,
      success (data) {
        resolve(data)
      },
      error (err) {
        reject(err)
      }
    })
  })
}

function getEventSubsets(model_names, model_name, fore_step, range_mode, params) {
  return new Promise((resolve, reject) => {
    axiosGet({
      url: `${ip_address}/event_subsets/${model_names}/${model_name}/${fore_step}/${range_mode}/${params}`,
      success (data) {
        resolve(data)
      },
      error (err) {
        reject(err)
      }
    })
  })
}

function getPointSubsets(data_name, baseline_model_name, focused_model_name, configs, focused_scopes) {
  return new Promise((resolve, reject) => {
    axiosGet({
      url: `${ip_address}/find_point_slices/${data_name}/${baseline_model_name}/${focused_model_name}/${configs}/${focused_scopes}`,
      success (data) {
        resolve(data)
      },
      error (err) {
        reject(err)
      }
    })
  })
}

function get_slices_indices(data_name, model_name, cached_slice_id) {
  return new Promise((resolve, reject) => {
    axiosGet({
      url: `${ip_address}/slices_indices/${data_name}/${model_name}/${cached_slice_id}`,
      success (data) {
        resolve(data)
      },
      error (err) {
        reject(err)
      }
    })
  })
}



function getPhaseDetails(model_names, phase_id, bin_edges, focused_val, focused_scopes, focused_scope_id) {
  return new Promise((resolve, reject) => {
    axiosGet({
      url: `${ip_address}/phase_details/${model_names}/${phase_id}/${bin_edges}/${focused_val}/${focused_scopes}/${focused_scope_id}`,
      success (data) {
        resolve(data)
      },
      error (err) {
        reject(err)
      }
    })
  })
}

function getGridBorders(data_name) {
  return new Promise((resolve, reject) => {
    axiosGet({
      url: `${ip_address}/grid_borders/${data_name}`,
      success (data) {
        resolve(data)
      },
      error (err) {
        reject(err)
      }
    })
  })
}

function getInstanceSeqs(model_names, stamp_id, loc_id, focused_scope) {
  return new Promise((resolve, reject) => {
    axiosGet({
      url: `${ip_address}/instance_seqs/${model_names}/${stamp_id}/${loc_id}/${focused_scope}`,
      success (data) {
        resolve(data)
      },
      error (err) {
        reject(err)
      }
    })
  })
}

function getLocFeatures(loc_id, stamp_id) {
  return new Promise((resolve, reject) => {
    axiosGet({
      url: `${ip_address}/loc_features/${loc_id}/${stamp_id}`,
      success (data) {
        resolve(data)
      },
      error (err) {
        reject(err)
      }
    })
  })
}

function getModelErrMtx(data_name, model_names, failure_rules, scope_seg_th) {
  return new Promise((resolve, reject) => {
    axiosGet({
      url: `${ip_address}/model_err_mtx/${data_name}/${model_names}/${failure_rules}/${scope_seg_th}`,
      success (data) {
        resolve(data)
      },
      error (err) {
        reject(err)
      }
    })
  })
}

function getDataAugmentation(data_name, model_name, sel_aug_subsets, focus_conditions, aug_params) {
  return new Promise((resolve, reject) => {
    axiosGet({
      url: `${ip_address}/data_augmentation_by_slices/${data_name}/${model_name}/${sel_aug_subsets}/${focus_conditions}/${aug_params}`,
      success (data) {
        resolve(data)
      },
      error (err) {
        reject(err)
      }
    })
  })
}

function saveSubsetCollection(data_name, model_name, configs, subgroup_collection) {
  return new Promise((resolve, reject) => {
    axiosGet({
      url: `${ip_address}/save_subgroup_collection/${data_name}/${model_name}/${configs}/${subgroup_collection}`,
      success (data) {
        resolve(data)
      },
      error (err) {
        reject(err)
      }
    })
  })
}

function getLocInstanceInfor(model_names, phase_id, step_id, loc_id) {
  return new Promise((resolve, reject) => {
    axiosGet({
      url: `${ip_address}/loc_instance_infor/${model_names}/${phase_id}/${step_id}/${loc_id}`,
      success (data) {
        resolve(data)
      },
      error (err) {
        reject(err)
      }
    })
  })
}

function save_subgroup_collections (data_name, model_name, subgroup_collections, cur_configs) {
  return new Promise((resolve, reject) => {
    axiosGet({
      url: `${ip_address}/save_subgroup_collections/${data_name}/${model_name}/${subgroup_collections}/${cur_configs}`,
      success (data) {
        resolve(data)
      },
      error (err) {
        reject(err)
      }
    })
  })
}

function save_phase_collections (data_name, phase_collections, cur_configs) {
  return new Promise((resolve, reject) => {
    axiosGet({
      url: `${ip_address}/save_phase_collections/${data_name}/${phase_collections}/${cur_configs}`,
      success (data) {
        resolve(data)
      },
      error (err) {
        reject(err)
      }
    })
  })
}

export {
  getExistedTaskDataModel,
  getCurDataInfor,
  getDatasetConfigs,
  getInputOutputData,
  getModelInfor,
  getSTPhaseEvents,
  getPhasesIndicators,
  getPhaseDetails,
  getGridBorders,
  getModelParameters,
  processModelPreds,
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
  saveSubsetCollection,
  getLocInstanceInfor,
  save_subgroup_collections,
  save_phase_collections
}