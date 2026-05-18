import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
from collections import Counter
from sklearn.model_selection import train_test_split
from matplotlib.patches import Rectangle, Patch
import logging
logger = logging.getLogger(__name__)

# helper functions
def transform_prediction(Y_pred, pure_threshold):
    Y_pred_transformed = []
    
    for row in Y_pred:
        if row.max() > pure_threshold:
            # Pure prediction: only the first max index gets 1
            one_hot = np.zeros_like(row, dtype=float)
            one_hot[np.argmax(row)] = 1.0
            Y_pred_transformed.append(one_hot)
        else:
            # Mixed prediction: top 2 indices get 0.5
            top2 = np.argsort(row)[-2:]
            mixed = np.zeros_like(row, dtype=float)
            mixed[top2] = 0.5
            Y_pred_transformed.append(mixed)
    
    Y_pred_transformed = np.array(Y_pred_transformed)
    
    # --- Validation step ---
    row_sums = Y_pred_transformed.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        raise ValueError("Validation failed: some rows do not sum to 1.0")
    
    # Check pure-case rows have exactly one '1'
    pure_mask = (Y_pred.max(axis=1) > pure_threshold)
    pure_rows = Y_pred_transformed[pure_mask]
    if not np.all((pure_rows.sum(axis=1) == 1) & (pure_rows.max(axis=1) == 1)):
        raise ValueError("Validation failed: some pure rows are not strictly one-hot")
    return Y_pred_transformed

def reconstruct_label(row):
    nonzero = row[row > 0].index.tolist()
    if len(nonzero) == 1:
        return nonzero[0]
    elif len(nonzero) == 2:
        return f"{nonzero[0]} / {nonzero[1]}"
    else:
        return "unknown"

def get_class_label(row):
    present = row[row > 0].index.tolist()
    if len(present) == 1:
        return present[0]
    else:
        return "-".join(sorted(present))

def sort_labels_with_pure_first(labels):
    def sort_key(label):
        parts = label.split("-")
        return (len(parts), label)  # (1, ...) = purebred first, then mixed
    return sorted(labels, key=sort_key)

def parse_label(label: str):
    if not isinstance(label, str) or label.lower() == "unknown":
        return []
    parts = [p.strip() for p in label.split("/") if p.strip()]
    return parts

def extract_chr_number(path: str) -> int:
    m = re.search(r'ch(\d+)', path)
    return int(m.group(1)) if m else -1

def group_snps_by_chr(snps):
    by_chr = {}
    for s in snps:
        m = re.match(r'chr(\d+):', s)
        if not m:
            continue
        by_chr.setdefault(int(m.group(1)), []).append(s)
    return by_chr

def prediction_analysis(Y_pred, Y_test, pure_threshold, mute = False):
    # === Transform Prediction
    Y_pred_transformed = transform_prediction(Y_pred, pure_threshold)

    if not isinstance(Y_test, pd.DataFrame):
        Y_test = pd.DataFrame(Y_test)
    Y_test_array = Y_test.to_numpy()

    # === Metric 1: Strict Accuracy
    strict_correct = np.all(Y_pred_transformed == Y_test_array, axis=1)
    strict_accuracy = np.mean(strict_correct)

    # === Metric 2: Loose Accuracy
    loose_correct = []
    for true_row, pred_row in zip(Y_test_array, Y_pred_transformed):
        true_indices = set(np.where(true_row > 0)[0])
        pred_indices = set(np.where(pred_row > 0)[0])
        loose_correct.append(len(true_indices & pred_indices) > 0)
    loose_correct = np.array(loose_correct)
    loose_accuracy = np.mean(loose_correct)

    # === Create Metadata
    metadata = pd.DataFrame(index=Y_test.index)

    metadata["strict_wrong"] = ~strict_correct
    metadata["loose_wrong"] = ~loose_correct
    if mute == False:
        logger.info(f'Purity Threshold: {pure_threshold}')
        logger.info(f"Strict Accuracy: {strict_accuracy:.2%}")
        logger.info(f"Loose Accuracy: {loose_accuracy:.2%}")
        logger.info(f"Number of Wrong Samples - Strict: {np.sum(~strict_correct)}, Loose: {np.sum(~loose_correct)}")

    return strict_accuracy, loose_accuracy, metadata

def pure_threshold_search(Y_pred, Y_test, pure_thresholds = list(np.arange(0, 1.01, 0.01)), mute = False, svg_save_path = None):
    strict_accuracies = []
    loose_accuracies = []
    for pure_threshold in pure_thresholds:
        acc1, acc2, _ = prediction_analysis(Y_pred, Y_test, pure_threshold, mute = mute)
        strict_accuracies.append(acc1)
        loose_accuracies.append(acc2)
    if mute == False:
        plt.figure(figsize=(10, 6))
        plt.plot(pure_thresholds, strict_accuracies, label="Strict Accuracy")
        plt.plot(pure_thresholds, loose_accuracies, label="Loose Accuracy")
        plt.xlabel("Pure-Mixed Threshold")
        plt.ylabel("Accuracy")
        plt.legend()
        if svg_save_path is not None:
            plt.savefig(f'{svg_save_path}/Purity_threshold_search.svg', format="svg", transparent=True)
        else:
            plt.show()
    return strict_accuracies, loose_accuracies

def best_pure_threshold_v1(pure_thresholds, strict_accuracies, loose_accuracies):
    pure_thresholds = np.array(pure_thresholds)
    strict_accuracies = np.array(strict_accuracies)
    loose_accuracies = np.array(loose_accuracies)

    max_loose = np.max(loose_accuracies)

    candidates = np.where(loose_accuracies == max_loose)[0]

    strict_subset = strict_accuracies[candidates]
    max_strict = np.max(strict_subset)
    best_indices = candidates[np.where(strict_subset == max_strict)[0]]
    best_candidate = best_indices[-1]  # larger index preferred

    return pure_thresholds[best_candidate]

def plot_confusion_matrix_highlighted(Y_test, Y_pred, pure_threshold, metadata, save_path = None, save_plot = False):
    if not isinstance(Y_test, pd.DataFrame):
        Y_test = pd.DataFrame(Y_test)
    # === Transform and classify
    Y_pred_transformed = transform_prediction(Y_pred, pure_threshold)
    Y_test_array = Y_test.to_numpy()
    Y_pred_array = np.array(Y_pred_transformed)

    # Get class labels
    Y_true_labels = Y_test.apply(get_class_label, axis=1)
    Y_pred_labels = pd.DataFrame(Y_pred_transformed, index=Y_test.index, columns=Y_test.columns)
    Y_pred_labels = Y_pred_labels.apply(get_class_label, axis=1)

    true_classes = set(Y_true_labels)
    pred_classes = set(Y_pred_labels)
    all_classes = sort_labels_with_pure_first(true_classes | pred_classes)
    label_to_idx = {label: i for i, label in enumerate(all_classes)}
    n_classes = len(all_classes)

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y_true_labels, Y_pred_labels, labels=all_classes)

    # Red borders for wrong predictions
    wrong_flags = metadata["strict_wrong"].to_numpy()
    red_cells = set()
    for idx, (true, pred) in enumerate(zip(Y_true_labels, Y_pred_labels)):
        if wrong_flags[idx]:
            i = label_to_idx[true]
            j = label_to_idx[pred]
            red_cells.add((i, j))

    def is_pure(label): return len(label.split("-")) == 1
    def has_overlap(label1, label2): return bool(set(label1.split("-")) & set(label2.split("-")))

    true_is_pure = (Y_test_array.max(axis=1) == 1.0)
    pred_is_pure = (Y_pred_array.max(axis=1) == 1.0)

    # === SECTION STATS
    from collections import defaultdict
    section_stats = defaultdict(lambda: {"total": 0, "strict_correct": 0, "loose_correct": 0})
    for i, (tp, pp) in enumerate(zip(true_is_pure, pred_is_pure)):
        if tp and pp:
            section = "True Pure → Pred Pure"
        elif tp and not pp:
            section = "True Pure → Pred Mixed"
        elif not tp and pp:
            section = "True Mixed → Pred Pure"
        else:
            section = "True Mixed → Pred Mixed"
        section_stats[section]["total"] += 1
        if not metadata["strict_wrong"].iloc[i]:
            section_stats[section]["strict_correct"] += 1
        if not metadata["loose_wrong"].iloc[i]:
            section_stats[section]["loose_correct"] += 1

    section_df = pd.DataFrame([
        {
            "Section": k,
            "Total": v["total"],
            "Strict Correct": v["strict_correct"],
            "Strict Accuracy": v["strict_correct"] / v["total"] if v["total"] else 0,
            "Loose Correct": v["loose_correct"],
            "Loose Accuracy": v["loose_correct"] / v["total"] if v["total"] else 0
        }
        for k, v in section_stats.items()
    ])

    # === PER-CLASS STATS + WRONG PRED SUMMARY
    class_stats = defaultdict(lambda: {
        "total": 0, "strict_correct": 0, "loose_correct": 0, "wrong_pred_detail": Counter()
    })

    for i in range(len(Y_test)):
        true_label = Y_true_labels.iloc[i]
        pred_label = Y_pred_labels.iloc[i]
        class_stats[true_label]["total"] += 1
        if not metadata["strict_wrong"].iloc[i]:
            class_stats[true_label]["strict_correct"] += 1
        if not metadata["loose_wrong"].iloc[i]:
            class_stats[true_label]["loose_correct"] += 1
        if true_label != pred_label:  # record only wrong predictions
            class_stats[true_label]["wrong_pred_detail"][pred_label] += 1

    def format_wrong(counter: Counter) -> str:
        if not counter:
            return ""
        return ";".join([f"{k}:{v}" for k, v in counter.items()])

    class_df = pd.DataFrame([
        {
            "Class Label": k,
            "Total": v["total"],
            "Strict Correct": v["strict_correct"],
            "Strict Accuracy": (v["strict_correct"] / v["total"]) if v["total"] else 0.0,
            "Loose Correct": v["loose_correct"],
            "Loose Accuracy": (v["loose_correct"] / v["total"]) if v["total"] else 0.0,
            "Wrong Prediction": format_wrong(v["wrong_pred_detail"])
        }
        for k, v in class_stats.items()
    ]).sort_values("Class Label")

    # === Plotting
    if save_plot:
        fig, ax = plt.subplots(figsize=(max(10, 0.4 * n_classes), max(8, 0.4 * n_classes)))
        ax.set_xlim(0, n_classes)
        ax.set_ylim(0, n_classes)
        ax.invert_yaxis()

        for i, true_label in enumerate(all_classes):
            for j, pred_label in enumerate(all_classes):
                count = cm[i, j]
                #if is_pure(true_label) and is_pure(pred_label):
                if true_label == pred_label:
                    facecolor = "lightgreen"
                elif has_overlap(true_label, pred_label):
                    facecolor = "moccasin"
                else:
                    facecolor = "lightblue"
                rect = Rectangle((j, i), 1, 1, facecolor=facecolor, edgecolor='gray')
                ax.add_patch(rect)
                if count > 0:
                    ax.text(j + 0.5, i + 0.5, str(count), ha="center", va="center", fontsize=10)

        for (i, j) in red_cells:
            ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=2))

        legend_elements = [
            Patch(facecolor='lightgreen', edgecolor='gray', label='Correct Prediction'),
            Patch(facecolor='moccasin', edgecolor='gray', label='Loosely Wrong'),
            Patch(facecolor='lightblue', edgecolor='gray', label='Strictly Wrong'),
            Patch(facecolor='none', edgecolor='red', lw=2, label='Wrong Prediction')
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=4, frameon=False)

        ax.set_xticks(np.arange(n_classes) + 0.5)
        ax.set_yticks(np.arange(n_classes) + 0.5)
        ax.set_xticklabels([l + "*" if l not in true_classes else l for l in all_classes], rotation=45, ha="right")
        ax.set_yticklabels(all_classes)
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        ax.set_title(f"Prediction Map", pad=40)

    # === Save files if path provided
    if save_path:
        save_table_path = f"{save_path}/Table"
        save_figure_path = f"{save_path}/Figure"
        os.makedirs(save_table_path, exist_ok=True)
        os.makedirs(save_figure_path, exist_ok=True)
        
        per_section_csv = os.path.join(save_table_path, f"Per_section_performance.csv")
        per_class_csv = os.path.join(save_table_path, f"Per_class_prediction_details.csv")

        section_df.to_csv(per_section_csv, index=False)
        class_df.to_csv(per_class_csv, index=False)
        logger.info(f"\nPer-section stats saved to: {per_section_csv}")
        logger.info(f"Per-class prediction stats saved to: {per_class_csv}")

        if save_plot: 
            plot_file = os.path.join(save_figure_path, f"Prediction_map.svg")
            plt.savefig(plot_file, format="svg", transparent=True)
            logger.info(f"Prediction_map saved to: {plot_file}")
        else:
            logger.info("The prediction map is not generated.")

def safe_train_test_split(X, Y, test_size=0.3, random_state=42):
    rng = np.random.default_rng(random_state)
    assert X.index.equals(Y.index), "X and Y must have identical indices (dog_id alignment)."
    def get_label_type(row):
        nonzero_idx = np.nonzero(row)[0]
        if len(nonzero_idx) == 1:
            return f"pure_{nonzero_idx[0]}"
        elif len(nonzero_idx) == 2 and np.allclose(row.iloc[nonzero_idx], 0.5):
            return f"mix_{tuple(sorted(nonzero_idx))}"
        else:
            return f"other_{hash(tuple(np.round(row, 3)))}"

    y_proxy = Y.apply(get_label_type, axis=1)

    value_counts = y_proxy.value_counts()
    rare_classes = value_counts[value_counts == 1].index
    rare_mask = y_proxy.isin(rare_classes)
    normal_mask = ~rare_mask

    X_rare, Y_rare = X[rare_mask], Y[rare_mask]
    X_normal, Y_normal = X[normal_mask], Y[normal_mask]
    y_proxy_normal = y_proxy[normal_mask]

    try:
        X_train_n, X_test_n, Y_train_n, Y_test_n = train_test_split(
            X_normal, Y_normal,
            test_size=test_size,
            stratify=y_proxy_normal,
            random_state=random_state
        )
    except ValueError as e:
        X_train_n, X_test_n, Y_train_n, Y_test_n = train_test_split(
            X_normal, Y_normal,
            test_size=test_size,
            random_state=random_state
        )

    mask = rng.random(len(X_rare)) < test_size
    X_test_r, Y_test_r = X_rare[mask], Y_rare[mask]
    X_train_r, Y_train_r = X_rare[~mask], Y_rare[~mask]
    y_proxy_rare_train = y_proxy[rare_mask][~mask]
    y_proxy_rare_test = y_proxy[rare_mask][mask]

    X_train = pd.concat([X_train_n, X_train_r]).sort_index()
    X_test  = pd.concat([X_test_n,  X_test_r]).sort_index()
    Y_train = pd.concat([Y_train_n, Y_train_r]).sort_index()
    Y_test  = pd.concat([Y_test_n,  Y_test_r]).sort_index()

    train_ids = pd.DataFrame(X_train.index, columns=["dog_id"])
    test_ids  = pd.DataFrame(X_test.index,  columns=["dog_id"])

    y_proxy_train = pd.concat([y_proxy_normal.loc[X_train_n.index], y_proxy_rare_train])
    y_proxy_test  = pd.concat([y_proxy_normal.loc[X_test_n.index],  y_proxy_rare_test])

    train_counts = y_proxy_train.value_counts().rename("Train")
    test_counts  = y_proxy_test.value_counts().rename("Test")
    summary = pd.concat([train_counts, test_counts], axis=1).fillna(0).astype(int)
    summary["Total"] = summary["Train"] + summary["Test"]
    summary = summary.sort_index()

    n_pure = sum("pure_" in s for s in summary.index)
    n_mix  = sum("mix_"  in s for s in summary.index)

    return X_train, X_test, Y_train, Y_test, summary