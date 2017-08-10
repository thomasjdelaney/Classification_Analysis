"""
For applying a window to the spike predictions. The window is extended into
the frames after the spikeframe. If a spike prediction falls within that window,
the prediction is classified as a 'hit'. Otherwise, without a window
(window_size=0) the prediction would be classified as a 'fault'.

hits + misses = actual
hits + faults = predicted
hits + misses + faults + correct_negatives = all_indices
all_indices - (actual + predicted) = correct_negs

Arguments:  actual, the actual spikes (or non-spikes)
            predicted, the predicted spikes (or non-spikes)
            all_indices, all the possible spikes
            window_size = the size of the window to use
Returns:    hits, correct predictions
            misses, spikes (or non-spikes) not predicted
            faults, incorrect predictions
            correct_negs, correct predictions of a non-spike
"""
function getHitsAndMisses(actual::Array{Int}, predicted::Array{Int}, all_indices::Array{Int}, window_size::Int)
    if window_size <= 0
        hits = intersect(actual, predicted)
        misses = setdiff(actual, predicted)
        faults = setdiff(predicted, actual)
        correct_negs = setdiff(all_indices, [actual; predicted])
    else
        hits, faults = Int[], Int[]
        for p in predicted
            p_window = collect(p - (0:window_size))
            spikes_in_window = intersect(actual, p_window)
            if length(spikes_in_window) > 0 # we have a spike in the window
                matched_spike = minimum(spikes_in_window) # match to the earliest spike
                actual = filter(x->x!=matched_spike, actual)
                append!(hits, matched_spike)
            else
                append!(faults, p)
            end
        end
        misses = setdiff(actual, hits)
        correct_negs = setdiff(all_indices, [hits; misses; faults])
    end
    return hits, misses, faults, correct_negs
end
