import { ref, computed } from 'vue'
import { defineStore } from 'pinia'

export const useInforStore = defineStore('infor', () => {
  const existed_task_data_model = ref({})
  const cur_sel_task = ref("air_quality_pred")
  const cur_sel_data = ref("")
  const cur_config_file = ref("Select Config File")
  const dataset_configs = ref({})
  const forecast_scopes = ref([])
  const cur_focused_scope = ref(0)
  const cur_model_names = ref([])
  const cur_focused_model = ref("")
  const cur_baseline_model = ref("")
  const cur_subset_model = ref("")
  const cur_sel_models = ref(["", ""])
  const model_parameters = ref({})
  const model_infor = ref({})
  const error_configs = ref({})
  const dataset_infor = ref({})
  const feature_infor = ref({
    'input': "",
    'output': ""
  })
  const cur_sel_model = ref('Choose Model')
  const cur_sel_fore_step = ref(10)
  const cur_window_sizes = ref([])
  const cur_sel_window_size = ref('Choose Forecast Steps')
  const cur_err_dis = ref({})
  const error_distributions = ref({})
  const attr_distributions = ref({})
  const cur_attr_bin_errors = ref({})
  const focus_err_range = ref([0, 0])
  const global_err_range = ref([0, 0])
  const err_abs_th = ref(0)
  // const cur_sel_err_th = ref(0)
  const sel_error_config = ref('Select config file')
  const cur_sel_failure_rules = ref([])
  const cur_sel_indicator_scheme = ref("")
  // const cur_sel_focus_type = ref('equal')
  const cur_sel_indicators = ref([])
  const sel_phase_details = ref({})
  const phase_details_infor = ref({})
  const sel_subset_details = ref([])
  const cur_detail_type = 'global'
  const cur_sel_corr_th = ref(0.6)
  const cur_overview_subgroups = ref({
    'time_scope': [],
    'val_range': []
  })
  const loc_regions = ref([])

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
  const cur_phase_indicators = ref([])
  const events_indicators = ref([])

  const config_save_state = ref("")

  const select_ranges = ref({})
  const cur_event_series = ref({
    'self_level_series': [],
    'temporal_state_series': [],
    'space_state_series': []
  })
  const st_phase_events = ref({})
  const cur_filtered_phases = ref([])
  const cur_filtered_events = ref([])
  const cur_filtered_phases_indices = ref([])
  const event_types = ref(["Forming", "Merging", "Continuing", "Growing", "Shape Changing", "Shrinking", "Splitting", "Dissolving"])
  const sel_event_types = ref([])
  const cur_st_attrs = ref([])
  const input_feats = ref(['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'RH', 'PSFC'])
  const type_feats = ref({
    'Pollutants': ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3'],
    'Weather': ['U', 'V', 'TEMP', 'RH', 'PSFC'],
    // 'Space': ['elevation']
  })
  const analysis_type = ref('Error-Guided')
  const phase_params = ref({
    'min_length': 8,
    'max_gap_len': 3
  })
  const cur_range_mode = ref("supervised")
  // const cur_range_mode = ref("equal_val")
  // const cur_range_mode = ref("equal_fre")
  const range_params = ref({
    'sup_th': 0.01,
    'div_th': 0.2,
    'bin_num': 10,
    'val_step': 5,
    'step_len': 10
  })
  const subset_params = ref({
    'frequent_sup_th': 0.005,
    'err_diff_th': 0.1,
    'div_th': 0.5,
    'redundancy_th': 0.4
  })
  const aug_params = ref({
    'EMD': 'MEMD',
    'imf_num': 5,
    'tol': 0.05,
    'EMD_rounds': 2,
    'mixup': 'balanced_mixup',
    'mixup_rate': 1.0,
  })

  const cur_instance_details = ref({})
  const cur_sel_stamp_objs = ref({})
  const cur_sel_stamp_objs_single = ref({})
  const cur_instance_seqs = ref([])
  const cur_phase_id = ref(-1)
  const cur_phase_sorted_id = ref('none')
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
  const cur_subsets = ref([])
  const cached_slice_id = ref(-1)
  const filtered_subsets = ref([])
  const sel_subset_points = ref([])
  const phase_clean_res_abs_mean = ref(0)
  const meta_attr_objs = ref({})
  const cur_subset_st_indices = ref([])
  const phase_subsets_proj = ref([])
  const phase_subsets_links = ref([])
  const cur_shown_steps = ref([0, 3, 6, 9, 11])
  const model_err_mtx = ref({})

  const cur_sel_condition = ref({
    loc_id: -1,
    x_condition: {},
    y_condition: {}
  }) 

  const cur_focus_subset = ref(-1)
  const browsed_subsets = ref([])
  const browsed_phases = ref([])
  const subset_collections = ref([])
  const phase_collections = ref([])
  const event_records = ref({})
  const sel_event_record = ref(-1)
  const sorted_phase_ids = ref([])
  const cur_aug_subsets = ref([])
  const cur_aug_subsets_infor = ref([])
  const all_focus_conditions = ref([])

  const subset_type = ref('Point')
  
  const mild_err_color_scale = ref({})
  const process_preds_state = ref(0)
  const extreme_err_color_scale = ref({})
  const err_abs_extreme_th = ref(0)
  const preds_num = ref(0)

  const phase_pcp_dims = ref({})
  const cur_event_pcp_dims = ref({})
  const cur_sel_event_steps = ref([])

  const interface_type = ref(['Subgroup', 'Phase&Event'])
  const cur_interface_type = ref('Subgroup')
  const cur_related_subsets = ref({
    'contained_subgroups': [],
    'replace_subgroups': [],
    'drill_down_subgroups': []
  })
  const loc_instance_infor = ref({})
  const focused_subsets_list = ref([])
  const currentIndex = ref(-1)

  return { existed_task_data_model, 
    error_configs,
    dataset_infor,
    cur_config_file,
    dataset_configs,
    forecast_scopes,
    cur_focused_scope,
    feature_infor,
    model_infor,
    cur_sel_task,
    cur_sel_data,
    cur_model_names,
    cur_sel_model,
    cur_window_sizes,
    cur_sel_window_size,
    cur_err_dis,
    error_distributions,
    attr_distributions,
    cur_attr_bin_errors,
    cur_st_attrs,
    global_err_range,
    focus_err_range,
    err_abs_th,
    cur_sel_failure_rules,
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
    phase_details_infor,
    sel_subset_details,
    cur_detail_type,
    cur_event_series,
    st_phase_events,
    cur_filtered_phases,
    cur_filtered_events,
    analysis_type,
    phase_params,
    range_params,
    subset_params,
    indicators_list,
    cur_timeline_indicator,
    cur_grid_indicator,
    select_ranges,
    cur_instance_details,
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
    cur_subsets,
    filtered_subsets,
    cached_slice_id,
    sel_subset_points,
    phase_clean_res_abs_mean,
    meta_attr_objs,
    input_feats,
    cur_subset_st_indices,
    loc_regions,
    type_feats,
    event_types,
    phase_subsets_proj,
    phase_subsets_links,
    cur_shown_steps,
    model_err_mtx,
    cur_focused_model,
    cur_baseline_model,
    cur_sel_models,
    sel_error_config,
    cur_focus_subset,
    browsed_subsets,
    browsed_phases,
    subset_collections,
    phase_collections,
    cur_sel_condition,
    cur_aug_subsets,
    cur_aug_subsets_infor,
    all_focus_conditions,
    cur_subset_model,
    subset_type,
    aug_params,
    mild_err_color_scale,
    extreme_err_color_scale,
    process_preds_state,
    err_abs_extreme_th,
    preds_num,
    phase_pcp_dims,
    cur_event_pcp_dims,
    cur_sel_event_steps,
    interface_type,
    cur_interface_type,
    cur_related_subsets,
    loc_instance_infor,
    sel_event_types,
    event_records,
    sel_event_record,
    focused_subsets_list,
    currentIndex
  }
})
