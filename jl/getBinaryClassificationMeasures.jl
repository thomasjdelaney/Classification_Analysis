"""
Get lost of measurements for predictive classification

Arguments:  data = array, the actual data
            preds = array, the predicted classifications
            window_size = Int, window to apply to the predictions
Returns:    accuracy, tp_rate, tn_rate, precision, npv, fn_rate, fp_rate, fdr, fo_rate
"""
function getBinaryClassificationMeasures(data, preds, window_size=0)
  spike_inds = find(data .== 1)
  pred_inds = find(preds .== 1)
  all_indices = collect(1:length(data))
  true_positives, false_negatives, false_positives, true_negatives =  getHitsAndMisses(spike_inds, pred_inds, all_indices, window_size)
  num_true_positives = length(true_positives)
  num_true_negatives = length(true_negatives)
  num_false_positives = length(false_positives)
  num_false_negatives = length(false_positives)
  accuracy = (num_true_positives + num_true_negatives)/(num_true_positives + num_true_negatives + num_false_positives + num_false_negatives)
  tp_rate = num_true_positives/(num_true_positives + num_false_negatives) # aka recall, sensitivity, hit rate
  tn_rate = num_true_negatives/(num_true_negatives + num_false_positives) # aka specificty
  precision = num_true_positives/(num_true_positives + num_false_positives) # aka positive predictive value
  npv = num_true_negatives/(num_true_negatives + num_false_negatives) # negative predictive value
  fn_rate = num_false_negatives/(num_false_negatives + num_true_positives) # aka miss rate
  fp_rate = num_false_positives/(num_false_positives + num_true_negatives) # aka fall-out
  fdr = num_false_positives/(num_false_positives + num_true_positives) # false discovery rate
  fo_rate = num_false_negatives/(num_false_negatives + num_true_negatives) # false omission rate
  return accuracy, tp_rate, tn_rate, precision, npv, fn_rate, fp_rate, fdr, fo_rate
end
