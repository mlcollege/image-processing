import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np


def get_info_df(labels, predictions, class_id2class_name_mapping, images=None):
    if labels.shape != predictions.shape:
        raise AttributeError("Labels and preds shape mismatch")
    example_count = labels.shape[0]

    label_class_ids = np.argmax(labels, axis=1).tolist()
    label_class_names = [class_id2class_name_mapping[c_id] for c_id in label_class_ids]
    label_class_scores = predictions[np.arange(example_count), label_class_ids].tolist()

    predicted_class_ids_top1 = np.argmax(predictions, axis=1).tolist()
    predicted_class_names_top1 = [class_id2class_name_mapping[c_id] for c_id in predicted_class_ids_top1]
    predicted_class_scores_top1 = np.max(predictions, axis=1).tolist()

    predicted_class_ids_top3 = np.argsort(predictions, axis=1)[:, -3:].tolist()
    predicted_class_names_top3 = []

    selection_class_ids_top3 = []
    for label_class_id, predicted_class_ids_triplet in zip(label_class_ids, predicted_class_ids_top3):
        if label_class_id in predicted_class_ids_triplet:
            class_id = label_class_id
        else:
            class_id = predicted_class_ids_triplet[0]
        selection_class_ids_top3.append(class_id)
        predicted_class_names_top3.append(class_id2class_name_mapping[class_id])
    predicted_class_scores_top3 = predictions[np.arange(example_count), selection_class_ids_top3].tolist()

    if images is None:
        images = [None for img in range(example_count)]

    return pd.DataFrame(
        {'label_class_name': label_class_names,
         'label_class_score': label_class_scores,
         'predicted_class_name_top1': predicted_class_names_top1,
         'predicted_class_score_top1': predicted_class_scores_top1,
         'predicted_class_name_top3': predicted_class_names_top3,
         'predicted_class_score_top3': predicted_class_scores_top3,
         'image': list(images)}
    )


def get_recall(df, class_name):
    true_positives = len(df[(df.label_class_name == class_name) & (df.predicted_class_name_top1 == class_name)])
    trues = len(df[(df.label_class_name == class_name)])
    return round(true_positives / trues * 100, 2)


def get_precision(df, class_name):
    true_positives = len(df[(df.label_class_name == class_name) & (df.predicted_class_name_top1 == class_name)])
    positives = len(df[(df.predicted_class_name_top1 == class_name)])
    if positives == 0:
        positives = 1
    return round(true_positives / positives * 100, 2)


def get_accuracy(df, use_top3=False):
    if use_top3:
        return round(float(np.mean((df.label_class_name == df.predicted_class_name_top3).astype(int))) * 100, 2)
    return round(float(np.mean((df.label_class_name == df.predicted_class_name_top1).astype(int))) * 100, 2)

def get_rec_prec(df, class_id2class_name_mapping):
    return pd.DataFrame(
        {
            "class_name": [class_name for class_name in class_id2class_name_mapping.values()],
            "recall": [get_recall(df, class_name) for class_name in class_id2class_name_mapping.values()],
            "precision": [get_precision(df, class_name) for class_name in class_id2class_name_mapping.values()]
        })


def get_false_positives(df, label_class_name, predicted_class_name=None):
    if predicted_class_name is None:
        condition = (df['label_class_name'] == label_class_name) & (df['predicted_class_name_top1'] != label_class_name)
    else:
        condition = (df['label_class_name'] == label_class_name) & (df['predicted_class_name_top1'] == predicted_class_name)
    return df[condition].sort_values(by='predicted_class_score_top1', ascending=False)


def plot_examples(df, image_shape=(28, 28)):
    examples_count = min(25, len(df))
    cols = 5
    rows = np.ceil(examples_count / cols)

    fig = plt.figure(figsize=(20, 25))
    for img_id in range(examples_count):
        ax = plt.subplot(rows, cols, img_id + 1)

        img = df.image.iloc[img_id]
        if img is None:
            continue
        img = img.reshape(image_shape)
        prediction_name = df.predicted_class_name_top1.iloc[img_id]
        prediction_score = df.predicted_class_score_top1.iloc[img_id]

        ax.imshow(img, cmap='gray')
        ax.set_title("{0}: {1}".format(prediction_name, round(prediction_score, 2)))
        ax.axes.set_axis_off()


def plot_coocurance_matrix(df, use_log=False, use_top3=False):
    if use_top3:
        coocurance_cols = ['label_class_name', 'predicted_class_name_top3']
    else:
        coocurance_cols = ['label_class_name', 'predicted_class_name_top1']
    coocurance_df = pd.pivot_table(df[coocurance_cols], index=coocurance_cols[0],
                                   columns=coocurance_cols[1], aggfunc=len, fill_value=0)
    if use_log:
        coocurance_df = np.log(coocurance_df)
        coocurance_df = coocurance_df.replace([-np.inf], 0)

    coocurance_df = coocurance_df.div(coocurance_df.sum(axis=1), axis=0) * 100
    coocurance_df = coocurance_df.round(2)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    sns.heatmap(coocurance_df, ax=ax, annot=True, linewidths=.5,
                cbar_kws={"orientation": "horizontal"}, cmap="YlGnBu")
