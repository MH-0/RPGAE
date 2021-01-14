import numpy as np
import os
import matplotlib.pyplot as plt

datasets = os.listdir("..\\data\\")

best_models = {}
for dataset in datasets:
    dataset_best_models = np.load("..\\data\\" + dataset + "\\scores\\best_models.npy",
                                allow_pickle=True).item()
    for feature in dataset_best_models:
        if not feature in best_models:
            best_models[feature]= {}
        for model in dataset_best_models[feature]:
            if not model in best_models[feature]:
                best_models[feature][model] = 0
            best_models[feature][model] += dataset_best_models[feature][model]

features =[]
models = {}
for feature in best_models:
    print(feature)
    features.append(feature.replace("_"," "))
    for model in best_models[feature]:
        print(model, best_models[feature][model])
        if not model in models:
            models[model] = []
        models[model].append(best_models[feature][model])


x = np.arange(len(features))  # the label locations
print(x)
width = 0.08  # the width of the bars

fig, ax = plt.subplots()
rects = []
bar_index = 0
for model in models:
    rects.append(ax.bar(x - (width /len(models.keys())) + (bar_index * width), models[model], width, label= model))
    bar_index +=1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number of times the model came on top')
ax.set_title('')
ax.set_xticks(x)
ax.set_xticklabels(features)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

for rect in rects:
    autolabel(rect)

fig.tight_layout()

plt.show()
