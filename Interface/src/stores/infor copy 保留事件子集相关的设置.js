import { ref, computed } from 'vue'
import { defineStore } from 'pinia'

export const useInforStore = defineStore('infor', () => {
  const existed_task_data_model = ref({})
  const cur_sel_task = ref("air_quality_pred")
  const cur_sel_data = ref("")
  const cur_model_names = ref([])
  const model_parameters = ref({})
  const model_infor = ref({})
  const error_configs = ref({})  // 暂时用不到
  const dataset_infor = ref({})
  const feature_infor = ref({
    'input': "",
    'output': ""
  })
  const indicator_schemes = ref({
    "Scope|Range-MAE": {
      "indicators": ["Scope-whole&Range-whole(MAE)"],
      "description": "Evaluating the union of range subgroups and scope subgroups using Mean Absolute Error(MAE)."
    },
    "Scope|Range-MSE": {
      "indicators": ["Scope-whole&Range-whole(MSE)"],
      "description": "Evaluating the union of range subgroups and scope subgroups using Mean Squared Error(MSE)."
    },  
    "Scope|Range-RMSE": {
      "indicators": ["Scope-whole&Range-whole(RMSE)"],
      "description": "Evaluating the union of range subgroups and scope subgroups using Rooted Mean Squared Error(RMSE)."
    },
    "Scope|Range-MAPE": {
      "indicators": ["Scope-whole&Range-whole(MAPE)"],
      "description": "Evaluating the union of range subgroups and scope subgroups using Mean Absolute Percent Error(MAPE)."
    },
    "Scope|Range-R2": {
      "indicators": ["Scope-whole&Range-whole(R2)"],
      "description": "Evaluating the union of range subgroups and scope subgroups using Coefficient of Determination(R2)."
    },
    "Scope|Range-EVAR": {
      "indicators": ["Scope-whole&Range-whole(EVAR)"],
      "description": "Evaluating the union of range subgroups and scope subgroups using Explained variance score(EVAR)."
    },
    "Whole-All": {
      "indicators": ["Scope-whole&Range-whole(MAE)", "Scope-whole&Range-whole(MSE)", "Scope-whole&Range-whole(RMSE)", "Scope-whole&Range-whole(MAPE)", "Scope-whole&Range-whole(R2)", "Scope-whole&Range-whole(EVAR)"],
      "description": "Evaluating the whole dataset using all metrics."
    }
  })
  const cur_sel_time = ref({})   // 暂时用不到
  const cur_sel_model = ref('Choose Model')
  const cur_sel_fore_step = ref(10)
  const cur_window_sizes = ref([])
  const cur_sel_window_size = ref('Choose Forecast Steps')
  const cur_err_dis = ref({})
  const focus_err_range = ref([0, 0])
  const global_err_range = ref([0, 0])
  // const cur_sel_err_th = ref(0)
  const cur_sel_failure_rules = ref([])
  const cur_sel_scope_th = ref(3)
  const cur_sel_indicator_scheme = ref("")
  // const cur_sel_focus_type = ref('equal')
  // const cur_sel_focus_th = ref(0)
  const cur_sel_indicators = ref([])
  const sel_phase_details = ref([])
  const sel_subset_details = ref([])
  const cur_detail_type = 'global'
  const cur_sel_corr_th = ref(0.6)
  const cur_overview_subgroups = ref({
    'time_scope': [],
    'val_range': []
  })

  const indicators_list = ref(['POD', 'FAR', 'Multi_accuracy', 'Residual_abs'])
  const cur_timeline_indicator = ref('Residual_abs')
  const cur_grid_indicator = ref('residual')

  const cur_data_infor = ref({})
  const cur_model_infor = ref([])
  
  const cur_range_infor = ref({})
  const cur_data_items = ref({})
  const cur_overview_indicators = ref({})
  const error_indicators = ref({
    'phases_indicators': {},
    'events_indicators': {},
    'clean_events_indicators': {},
    'outlier_events_indicators': {}
  })
  const phases_indicators = ref({})
  const cur_phase_indicators = []
  const events_indicators = ref([])

  const config_save_state = ref("")

  const cur_st_patterns = ref([])
  const cur_event_series = ref({
    'self_level_series': [],
    'temporal_state_series': [],
    'space_state_series': []
  })
  const st_phase_events = ref({})
  const cur_filtered_phases = ref([])
  const cur_filtered_events = ref([])
  const cur_filtered_phases_indices = ref([])
  // const cur_val_bins = ref([0, 50, 100, 150, 200, 300, 400, 500, 2000])
  const cur_val_bins = ref([0, 35, 75, 115, 150, 250, 350, 500, 2000])
  const cur_focused_scope = ref([1,12])
  const cur_st_attrs = ref(["target_val", "temporal_state_vals", "space_comp_state_vals", "space_diff_state_vals"])
  const analysis_type = ref('Error-Guided')
  const phase_params = ref({
    'focus_th': 115,
    'min_length': 8
  })
  const range_params = ref({
    'sup_th': 0.01,
    'div_th': 0.2,
    'bin_num': 10,
    'area_step': 1,
    'val_step': 5,
    'focus_indicator': 'max_residual',
    'step_len': 10
  })
  const subset_params = ref({
    'frequent_sup_th': 0.005,
    'err_diff_th': 10,
    'purity_diff_th': 0,
  })
  const organize_params = ref({
    'focus_purity_th': 0.1,
    'err_abs_th': 0
  })
  const event_params = ref({
    'dis_th': 25,
    'min_grid_num': 1,
    'val_change_th': 0.1,
    'area_change_th': 0.1,
    'area_remain_th': 0.9
  })

  const cur_subsets = ref({})  // 暂时用不到
  const cur_ranges = ref({})  // 暂时用不到
  const cur_sel_subset_id = ref(-1)  // 暂时用不到
  const cur_subset_data_ids = ref([])  // 暂时用不到

  const attrs_bins = ref({})  // 暂时用不到
  const attrs_hists = ref({})  // 暂时用不到
  const select_ranges = ref([])  // 暂时用不到

  const cur_x_attr = ref('Ground Truth')  // 暂时用不到
  const cur_y_attr = ref('Prediction')  // 暂时用不到
  const cur_bin_color = ref('Count')  // 暂时用不到
  const cur_x_bin_edges = ref([0, 35, 75, 115, 150, 250, 350, 500, 2000])  // 暂时用不到
  const cur_y_bin_edges = ref([0, 35, 75, 115, 150, 250, 350, 500, 2000])  // 暂时用不到

  const cur_sel_condition = ref({
    loc_id: -1,
    x_condition: {},
    y_condition: {}
  })  // 暂时用不到
  const cur_instance_details = ref({})
  const cur_sel_stamp_objs = ref({})
  const cur_sel_stamp_objs_single = ref({})
  const cur_instance_seqs = ref([])
  const cur_phase_id = ref(-1)
  const cur_phase_sorted_id = ref(-1)
  const ins_inspect_type = ref('')
  const cur_loc_features = ref({})

  const tile_resi_edge_range = ref([0, 0])
  const loc_feature_conditions = ref({
    'loc_id': -1,
    'stamp_id': -1,
    'seq_id': -1,
    'step_id': -1,
    'pred_num': 0
  })
  const cur_feature_hover_step = ref(-1)
  const cur_phase_data = ref({})
  const event_subsets = ref({})
  const event_attr_objs = ref({})
  const cur_range_mode = ref("supervised")
  // const cur_range_mode = ref("equal_val")

  return { existed_task_data_model, 
    error_configs,
    dataset_infor,
    feature_infor,
    model_infor,
    indicator_schemes,
    cur_sel_task,
    cur_sel_data,
    cur_sel_time,
    cur_model_names,
    cur_sel_model,
    cur_window_sizes,
    cur_sel_window_size,
    cur_err_dis,
    cur_st_attrs,
    global_err_range,
    focus_err_range,
    cur_sel_failure_rules,
    cur_sel_scope_th,
    cur_sel_corr_th,
    cur_sel_indicator_scheme,
    cur_sel_indicators,
    cur_overview_subgroups,
    cur_data_infor, 
    cur_model_infor,
    model_parameters,
    cur_range_infor,
    cur_data_items,
    config_save_state,
    cur_overview_indicators,
    error_indicators,
    phases_indicators,
    events_indicators,
    sel_phase_details,
    sel_subset_details,
    cur_detail_type,
    cur_st_patterns,
    cur_event_series,
    st_phase_events,
    cur_filtered_phases,
    cur_filtered_events,
    cur_subsets,
    cur_sel_subset_id,
    cur_ranges,
    analysis_type,
    phase_params,
    range_params,
    subset_params,
    organize_params,
    event_params,
    cur_subset_data_ids,
    cur_val_bins,
    indicators_list,
    cur_timeline_indicator,
    cur_grid_indicator,
    attrs_bins,
    attrs_hists,
    select_ranges,
    cur_x_attr,
    cur_y_attr,
    cur_bin_color,
    cur_x_bin_edges,
    cur_y_bin_edges,
    cur_instance_details,
    cur_sel_condition,
    cur_sel_stamp_objs,
    cur_instance_seqs,
    cur_phase_id,
    cur_phase_sorted_id,
    ins_inspect_type,
    cur_loc_features,
    tile_resi_edge_range,
    cur_sel_stamp_objs_single,
    loc_feature_conditions,
    cur_feature_hover_step,
    cur_phase_data,
    event_subsets,
    event_attr_objs,
    cur_range_mode,
    cur_sel_fore_step,
    cur_filtered_phases_indices,
    cur_phase_indicators,
    cur_focused_scope
  }
})
