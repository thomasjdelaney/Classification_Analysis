using DataFrames, PyPlot

m = readtable(homedir() * "/Classification_Analysis/csv/8.classification_measures.paninski.knn.model.csv")
d = readtable(homedir() * "/Classification_Analysis/csv/8.classification_measures.paninski.knn.train.csv")

common_traces = intersect(m[:trace_num], d[:trace_num])
model = m[[tn in common_traces for tn in m[:trace_num]], :tp_rate]
data = d[[tn in common_traces for tn in d[:trace_num]], :tp_rate]
datasets = [data, model]

PyPlot.boxplot([data, model])
PyPlot.plot([ones(data), 2*ones(model)], datasets, "r.", markersize=15)
PyPlot.xticks([1,2], ["data", "simulated"], fontsize="28")
PyPlot.yticks(fontsize="28")
PyPlot.title("Spike inference using the 'l zero' algorithm", fontsize="28")
PyPlot.ylabel("Recall", fontsize="28")
PyPlot.grid()

labels = map(string, collect(1:length(model)))
for i = 1:length(datasets)
  dataset = datasets[i]
  for j = 1:length(dataset)
    PyPlot.annotate(labels[j], xy=(i, dataset[j]), fontsize="20")
  end
end
