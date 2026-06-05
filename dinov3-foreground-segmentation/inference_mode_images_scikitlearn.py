import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import joblib

import wandb
import argparse

# Defines the ArgumentParser object
parser = argparse.ArgumentParser()

# Input / output parameters
parser.add_argument("--model_type", type=str, default='LR-SKLearn-LOOCV')
parser.add_argument("--run", type=int, default=1)
parser.add_argument("--train_data_path", type=str, default="./output_npy/512/train/")
parser.add_argument("--test_data_path", type=str, default="./output_npy/512/test/")
parser.add_argument("--output_path", type=str, default="./results/loocv_sklearn/")
parser.add_argument("--image_size", type=int, default=512)
parser.add_argument("--patch_size", type=int, default=16)
parser.add_argument("--max_iter", type=int, default=1000)
parser.add_argument("--C_values", nargs='+', type=float,
                    default=[0.01, 0.1, 1.0, 10.0])
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--test_images_path", type=str,
                    default="./data/test/images/")
parser.add_argument("--test_masks_path", type=str,
                    default="./data/test/masks/")
# ------------------------------------------------------------------------


def compute_fg_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float, float, float, float]:
    """Foreground semantic segmentation metrics (positive class = foreground = 1)."""
    dsc = metrics.f1_score(y_true, y_pred, zero_division=1)
    iou = metrics.jaccard_score(y_true, y_pred, zero_division=1)
    acc = metrics.accuracy_score(y_true, y_pred)
    pre = metrics.precision_score(y_true, y_pred, zero_division=1)
    rec = metrics.recall_score(y_true, y_pred, zero_division=1)
    return dsc, iou, acc, pre, rec


def loocv_hyperparameter_search(
    X: np.ndarray,
    y: np.ndarray,
    image_index: np.ndarray,
    C_values: list[float],
    max_iter: int,
    threshold: float,
) -> tuple[float, float, pd.DataFrame]:
    """Search over C using leave-one-image-out cross validation.

    One full image (all its patches) is held out per fold.
    Returns best C, best mean LOOCV DSC, and a DataFrame of all fold results.
    """
    unique_images = np.unique(image_index)
    best_C, best_mean_dsc = C_values[0], -1.0
    all_fold_records = []

    for C in C_values:
        fold_dscs = []
        for img_id in unique_images:
            val_mask = image_index == img_id
            train_mask = ~val_mask

            X_tr = X[train_mask]
            y_tr = y[train_mask].astype(int)
            X_val = X[val_mask]
            y_val_bin = y[val_mask].astype(int)

            model = LogisticRegression(C=C, max_iter=max_iter, solver='lbfgs')
            model.fit(X_tr, y_tr)
            pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (pred_proba >= threshold).astype(int)

            dsc, iou, acc, pre, rec = compute_fg_metrics(y_val_bin, y_pred)
            fold_dscs.append(dsc)
            all_fold_records.append({
                'C': C, 'fold_image_id': int(img_id),
                'dsc': dsc, 'iou': iou, 'accuracy': acc,
                'precision': pre, 'recall': rec,
            })

        mean_dsc = float(np.mean(fold_dscs))
        print(f"  LOOCV  C={C:.0e}  mean_dsc={mean_dsc:.4f}")

        if mean_dsc > best_mean_dsc:
            best_mean_dsc = mean_dsc
            best_C = C

    loocv_df = pd.DataFrame(all_fold_records)
    return best_C, best_mean_dsc, loocv_df


def save_loocv_c_plot(loocv_df: pd.DataFrame, output_path: str) -> None:
    mean_dsc_per_C = loocv_df.groupby('C')['dsc'].mean()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    bars = ax.bar(
        [f'{c:.0e}' for c in mean_dsc_per_C.index],
        mean_dsc_per_C.values,
        color='#2196F3',
    )
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('C (regularisation)')
    ax.set_ylabel('Mean DSC')
    ax.set_title('LOOCV Mean DSC — C Regularisation Search')
    for bar, val in zip(bars, mean_dsc_per_C.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'loocv_c_search.png'))
    plt.close()


def save_metrics_bar_chart(overall: dict, output_path: str) -> None:
    names = ['DSC', 'IoU', 'Accuracy', 'Precision', 'Recall']
    vals = [overall['mean_dsc'], overall['mean_iou'], overall['mean_accuracy'],
            overall['mean_precision'], overall['mean_recall']]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    bars = ax.bar(names, vals,
                  color=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336'])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Overall Test Metrics — Foreground Semantic Segmentation')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'overall_metrics_bar_chart.png'))
    plt.close()


def save_metrics_bar_chart_orig_gt(overall: dict, output_path: str) -> None:
    names = ['DSC', 'IoU', 'Accuracy', 'Precision', 'Recall']
    vals = [overall['mean_dsc_orig_gt'], overall['mean_iou_orig_gt'],
            overall['mean_accuracy_orig_gt'], overall['mean_precision_orig_gt'],
            overall['mean_recall_orig_gt']]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    bars = ax.bar(names, vals,
                  color=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336'])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Overall Test Metrics vs Original GT — Foreground Semantic Segmentation')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'overall_metrics_bar_chart_orig_gt.png'))
    plt.close()


# Main function to sequence the Python script source code
def main():
    args = parser.parse_args()
    h_patches = args.image_size // args.patch_size

    os.makedirs(args.output_path, exist_ok=True)

    wandb.init(
        project=f"DINOv3 {args.model_type} LOOCV Foreground Segmentation ScikitLearn",
        name=f"Run {args.run}",
    )

    ##############################################################################################
    # Load training / LOOCV data
    X = np.load(os.path.join(args.train_data_path, 'X_train.npy'))
    y_raw = np.load(os.path.join(args.train_data_path, 'y_train.npy'))
    image_index = np.round(
        np.load(os.path.join(args.train_data_path, 'image_index.npy'))
    ).astype(int)
    y = (y_raw > 0.5).astype(np.float32)

    print(f"Training data  X: {X.shape}  y: {y.shape}  "
          f"n_images: {np.unique(image_index).size}")

    ##############################################################################################
    # LOOCV hyperparameter search
    print("\n--- LOOCV Hyperparameter Search ---")
    best_C, best_dsc, loocv_df = loocv_hyperparameter_search(
        X, y, image_index,
        args.C_values,
        args.max_iter,
        args.threshold,
    )
    print(f"\nBest hyperparameters:  C={best_C}  LOOCV DSC={best_dsc:.4f}")

    loocv_df.to_csv(os.path.join(args.output_path, 'loocv_results.csv'), index=False)
    save_loocv_c_plot(loocv_df, args.output_path)
    wandb.log({'best_C': best_C, 'best_loocv_dsc': best_dsc})

    ##############################################################################################
    # Train final model on all training data with best hyperparameters
    print("\n--- Training Final Model on All Training Data ---")
    final_model = LogisticRegression(C=best_C, max_iter=args.max_iter, solver='lbfgs')
    final_model.fit(X, y.astype(int))
    model_save_path = os.path.join(args.output_path, 'logistic_regression_model.joblib')
    joblib.dump(final_model, model_save_path)
    print(f"Final model saved to {model_save_path}")

    ##############################################################################################
    # Load test data
    X_test = np.load(os.path.join(args.test_data_path, 'X_test.npy'))
    y_test_raw = np.load(os.path.join(args.test_data_path, 'y_test.npy'))
    image_index_test = np.round(
        np.load(os.path.join(args.test_data_path, 'image_index_test.npy'))
    ).astype(int)
    y_test = (y_test_raw > 0.5).astype(np.float32)

    print(f"\nTest data  X: {X_test.shape}  y: {y_test.shape}  "
          f"n_images: {np.unique(image_index_test).size}")

    ##############################################################################################
    # Per-image test evaluation
    # Sorted file lists — order must match the sorted order used during feature extraction
    _img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    test_img_files = sorted(
        f for f in os.listdir(args.test_images_path)
        if f.lower().endswith(_img_exts)
    )
    test_mask_files = sorted(
        f for f in os.listdir(args.test_masks_path)
        if f.lower().endswith(_img_exts)
    )

    unique_test_images = np.unique(image_index_test)
    test_records = []
    dsc_sum = iou_sum = acc_sum = pre_sum = rec_sum = 0.0
    dsc_og_sum = iou_og_sum = acc_og_sum = pre_og_sum = rec_og_sum = 0.0

    for file_idx, img_id in enumerate(unique_test_images):
        img_mask = image_index_test == img_id
        X_img = X_test[img_mask]
        y_true_bin = y_test[img_mask].astype(int)
        n_patches = int(img_mask.sum())
        w_patches = n_patches // h_patches

        pred_proba = final_model.predict_proba(X_img)[:, 1]

        y_pred = (pred_proba >= args.threshold).astype(int)
        dsc, iou, acc, pre, rec = compute_fg_metrics(y_true_bin, y_pred)

        # Reshape to patch grid
        pred_map = pred_proba.reshape(h_patches, w_patches)
        binary_mask = y_pred.reshape(h_patches, w_patches)
        gt_mask = y_true_bin.reshape(h_patches, w_patches)

        # ---- Original image and pre-conversion mask outputs ----
        orig_img = Image.open(
            os.path.join(args.test_images_path, test_img_files[file_idx])
        ).convert('RGB')
        orig_mask = Image.open(
            os.path.join(args.test_masks_path, test_mask_files[file_idx])
        )
        img_w, img_h = orig_img.size

        # Binarise original mask at image resolution: any non-black pixel = foreground
        orig_mask_arr = np.array(
            orig_mask.convert('RGB').resize((img_w, img_h), Image.NEAREST)
        )
        orig_mask_bin = (orig_mask_arr.max(axis=2) > 0).astype(int).flatten()

        # 1. Input image
        orig_img.save(
            os.path.join(args.output_path, f'input_image_{img_id}.png'))

        # 2. Input segmentation mask before pixel value conversion
        orig_mask.save(
            os.path.join(args.output_path, f'input_mask_pre_conversion_{img_id}.png'))

        # Upscale binary mask from patch grid to original image dimensions
        binary_mask_up = np.array(
            Image.fromarray((binary_mask * 255).astype(np.uint8)).resize(
                (img_w, img_h), Image.NEAREST
            )
        )
        fg_pixels = binary_mask_up > 0

        # Metrics against original ground truth mask at image resolution
        y_pred_img = fg_pixels.astype(int).flatten()
        dsc_og, iou_og, acc_og, pre_og, rec_og = compute_fg_metrics(orig_mask_bin, y_pred_img)

        # Now append to test_records with all metrics calculated
        test_records.append({
            'image_id': int(img_id),
            'dsc': dsc, 'iou': iou, 'accuracy': acc,
            'precision': pre, 'recall': rec,
            'dsc_orig_gt': dsc_og, 'iou_orig_gt': iou_og, 'accuracy_orig_gt': acc_og,
            'precision_orig_gt': pre_og, 'recall_orig_gt': rec_og,
        })

        print(f"\nTest Image {img_id}  DSC={dsc:.4f}  IoU={iou:.4f}  "
              f"Acc={acc:.4f}  Pre={pre:.4f}  Rec={rec:.4f}")
        print(f"  [orig GT]  DSC={dsc_og:.4f}  IoU={iou_og:.4f}  "
              f"Acc={acc_og:.4f}  Pre={pre_og:.4f}  Rec={rec_og:.4f}")
        wandb.log({f'test_image_{img_id}/dsc': dsc, f'test_image_{img_id}/iou': iou,
                   f'test_image_{img_id}/accuracy': acc,
                   f'test_image_{img_id}/precision': pre,
                   f'test_image_{img_id}/recall': rec,
                   f'test_image_{img_id}/dsc_orig_gt': dsc_og,
                   f'test_image_{img_id}/iou_orig_gt': iou_og,
                   f'test_image_{img_id}/accuracy_orig_gt': acc_og,
                   f'test_image_{img_id}/precision_orig_gt': pre_og,
                   f'test_image_{img_id}/recall_orig_gt': rec_og})

        dsc_sum += dsc
        iou_sum += iou
        acc_sum += acc
        pre_sum += pre
        rec_sum += rec
        dsc_og_sum += dsc_og
        iou_og_sum += iou_og
        acc_og_sum += acc_og
        pre_og_sum += pre_og
        rec_og_sum += rec_og

        # 3. Binary segmentation mask: foreground=red, background=black
        rgb_seg_mask = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        rgb_seg_mask[fg_pixels] = [255, 0, 0]
        plt.figure(dpi=300)
        plt.imshow(rgb_seg_mask)
        plt.axis('off')
        plt.savefig(
            os.path.join(args.output_path, f'binary_mask_red_image_{img_id}.png'),
            bbox_inches='tight', pad_inches=0)
        plt.close()

        # 4. Binary segmentation mask overlaid on input image (red foreground, alpha=0.5)
        img_arr = np.array(orig_img, dtype=np.float32)
        overlay = img_arr.copy()
        overlay[fg_pixels, 0] = img_arr[fg_pixels, 0] * 0.5 + 255 * 0.5
        overlay[fg_pixels, 1] = img_arr[fg_pixels, 1] * 0.5
        overlay[fg_pixels, 2] = img_arr[fg_pixels, 2] * 0.5
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        plt.figure(dpi=300)
        plt.imshow(overlay)
        plt.axis('off')
        plt.savefig(
            os.path.join(args.output_path, f'overlay_red_image_{img_id}.png'),
            bbox_inches='tight', pad_inches=0)
        plt.close()
        # ---- End original image and pre-conversion mask outputs ----

        # Save binary segmentation mask as .npy
        np.save(os.path.join(args.output_path, f'binary_mask_image_{img_id}.npy'),
                binary_mask)

        # Segmentation result plot: ground truth | foreground score | binary prediction
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=300)
        axes[0].imshow(gt_mask, cmap='gray')
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        axes[1].imshow(pred_map, cmap='hot')
        axes[1].set_title('Foreground Score')
        axes[1].axis('off')
        axes[2].imshow(binary_mask, cmap='gray')
        axes[2].set_title('Binary Prediction')
        axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(
            os.path.join(args.output_path, f'segmentation_plot_image_{img_id}.png'))
        plt.close()

        # Binary mask standalone plot (clean, no axes)
        plt.figure(dpi=300)
        plt.imshow(binary_mask, cmap='gray')
        plt.axis('off')
        plt.savefig(
            os.path.join(args.output_path, f'binary_mask_image_{img_id}.png'),
            bbox_inches='tight', pad_inches=0)
        plt.close()

    ##############################################################################################
    # Overall test metrics
    n_test = float(len(unique_test_images))
    overall = {
        'mean_dsc':                  dsc_sum / n_test,
        'mean_iou':                  iou_sum / n_test,
        'mean_accuracy':             acc_sum / n_test,
        'mean_precision':            pre_sum / n_test,
        'mean_recall':               rec_sum / n_test,
        'mean_dsc_orig_gt':          dsc_og_sum / n_test,
        'mean_iou_orig_gt':          iou_og_sum / n_test,
        'mean_accuracy_orig_gt':     acc_og_sum / n_test,
        'mean_precision_orig_gt':    pre_og_sum / n_test,
        'mean_recall_orig_gt':       rec_og_sum / n_test,
    }
    print(f"\n--- Overall Test Metrics ---")
    for k, v in overall.items():
        print(f"  {k}: {v:.4f}")
    wandb.log(overall)

    pd.DataFrame(test_records).to_csv(
        os.path.join(args.output_path, 'test_metrics_per_image.csv'), index=False)
    pd.DataFrame([overall]).to_csv(
        os.path.join(args.output_path, 'test_metrics_overall.csv'), index=False)

    save_metrics_bar_chart(overall, args.output_path)
    save_metrics_bar_chart_orig_gt(overall, args.output_path)

    print("\nDINOv3 Logistic Regression SKLearn LOOCV Script Complete")
    wandb.finish()


# Executes the main method from the inference_mode_images_scikitlearn.py Python script
if __name__ == '__main__':
    main()
