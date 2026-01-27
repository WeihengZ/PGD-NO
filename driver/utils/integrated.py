import numpy as np
import vtk
import torch
from typing import Dict, List, Tuple
from utils.metric import denormalize_pressure

'''
For fast integrated value computation
'''

def compute_coefficients(polydata):

    coef = np.zeros(polydata.GetNumberOfPoints())
    pts = polydata.GetPoints()

    cells = polydata.GetPolys()
    cells.InitTraversal()
    id_list = vtk.vtkIdList()
    while cells.GetNextCell(id_list):
        if id_list.GetNumberOfIds() != 3:
            # should be triangles after filter
            print("Not a triangle")
            continue
        i0 = id_list.GetId(0)
        i1 = id_list.GetId(1)
        i2 = id_list.GetId(2)
        p0 = np.array(pts.GetPoint(i0))
        p1 = np.array(pts.GetPoint(i1))
        p2 = np.array(pts.GetPoint(i2))
        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        if area <= 0:
            continue
        
        # assign the coefficients to the nodes
        coef[i0] += area / 3.0
        coef[i1] += area / 3.0
        coef[i2] += area / 3.0

    return coef

'''
For total force computation
'''
def _forward_model(model_name: str, model, features_6d, coef_Cp, node_cluster_flags, seg_matrix):
    if model_name == 'transolver_seg':
        return model(features_6d, seg_matrix, coef_Cp)
    elif model_name == 'figconv':
        return model(features_6d, coef_Cp, node_cluster_flags)
    elif model_name == 'multi':
        return model(features_6d, coef_Cp, node_cluster_flags)
    else:
        return model(features_6d, coef_Cp)


def total_force_compute(
    train_loader,
    val_loader,
    test_loader,
    model,
    normalization_scalars,
    predicted_feature_name: str,
    model_name: str,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Compute total x-force (sum across nodes) for train and test splits.

    Returns two dicts with keys: 'sim_ids', 'gt_total', 'pred_total'.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    def run_split(data_loader):
        results = {'sim_ids': [], 'gt_total': [], 'pred_total': []}
        with torch.no_grad():
            for batch_data in data_loader:
                # Unpack following utils/ml.py ordering
                features_6d, node_cluster_flags, target, seg_matrix, coef_Cp, integrated_cp_actual, sim_ids = batch_data
                features_6d = features_6d.to(device)
                target = target.to(device)
                coef_Cp = coef_Cp.to(device)
                if model_name == 'transolver_seg':
                    seg_matrix = seg_matrix.to(device)

                outputs, _ = _forward_model(model_name, model, features_6d, coef_Cp, node_cluster_flags, seg_matrix)

                # Denormalize x-force and sum per sample across nodes
                outputs_denorm = denormalize_x_force(outputs, normalization_scalars)
                target_denorm = denormalize_x_force(target, normalization_scalars)

                # Sum per-sample across nodes; keep result as 1D even for batch_size=1
                total_force_pred = torch.sum(outputs_denorm, dim=1).reshape(-1)
                total_force_gt = torch.sum(target_denorm, dim=1).reshape(-1)

                # Normalize sim_ids into a python list aligned with batch
                if isinstance(sim_ids, list):
                    batch_sim_ids = sim_ids
                elif torch.is_tensor(sim_ids):
                    batch_sim_ids = sim_ids.detach().cpu().tolist()
                else:
                    # Fallback: repeat the single id for the batch
                    batch_sim_ids = [sim_ids] * total_force_gt.shape[0]

                batch_size = total_force_gt.shape[0]
                for i in range(batch_size):
                    results['sim_ids'].append(batch_sim_ids[i] if i < len(batch_sim_ids) else batch_sim_ids[0])
                    results['gt_total'].append(float(total_force_gt[i].detach().cpu().item()))
                    results['pred_total'].append(float(total_force_pred[i].detach().cpu().item()))
        return results

    train_forces = run_split(train_loader)
    val_forces = run_split(val_loader)
    test_forces = run_split(test_loader)
    return train_forces, val_forces, test_forces


def plot_force_comparison(
    train_forces: Dict[str, List[float]],
    val_forces: Dict[str, List[float]],
    test_forces: Dict[str, List[float]],
    predicted_feature_name: str,
    output_path: str = "force_comparison.png",
):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))

    # Training data scatter and annotations
    if train_forces['gt_total'] and train_forces['pred_total']:
        plt.scatter(train_forces['gt_total'], train_forces['pred_total'],
                    c='blue', s=80, alpha=0.7, label='Training', edgecolors='black')
        # annotate with sim ids
        for x, y, sid in zip(train_forces['gt_total'], train_forces['pred_total'], train_forces['sim_ids']):
            plt.annotate(str(sid), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8, color='blue')

    # Validation data scatter and annotations
    if val_forces['gt_total'] and val_forces['pred_total']:
        plt.scatter(val_forces['gt_total'], val_forces['pred_total'],
                    c='green', s=80, alpha=0.7, label='Validation', edgecolors='black')
        for x, y, sid in zip(val_forces['gt_total'], val_forces['pred_total'], val_forces['sim_ids']):
            plt.annotate(str(sid), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8, color='green')

    # Test data scatter and annotations
    if test_forces['gt_total'] and test_forces['pred_total']:
        plt.scatter(test_forces['gt_total'], test_forces['pred_total'],
                    c='red', s=80, alpha=0.7, label='Test', edgecolors='black')
        for x, y, sid in zip(test_forces['gt_total'], test_forces['pred_total'], test_forces['sim_ids']):
            plt.annotate(str(sid), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8, color='red')

    # Diagonal and ±1% relative error bounds
    all_gt = train_forces['gt_total'] + val_forces['gt_total'] + test_forces['gt_total']
    all_pred = train_forces['pred_total'] + val_forces['pred_total'] + test_forces['pred_total']
    if all_gt and all_pred:
        min_val = min(min(all_gt), min(all_pred))
        max_val = max(max(all_gt), max(all_pred))
        xline = [min_val, max_val]
        plt.plot(xline, xline, 'k--', alpha=0.6, label='Perfect (y=x)')
        plt.plot(xline, [v * 1.01 for v in xline], color='gray', linestyle=':', alpha=0.7, label='+1% bound')
        plt.plot(xline, [v * 0.99 for v in xline], color='gray', linestyle=':', alpha=0.7, label='-1% bound')

    plt.xlabel('Ground Truth Total X-Force', fontsize=12)
    plt.ylabel('Predicted Total X-Force', fontsize=12)
    plt.title(f'Total X-Force: Predicted vs Ground Truth ({predicted_feature_name})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Brief stats to stdout
    def summarize(split_name, data):
        if data['gt_total']:
            gt = np.array(data['gt_total'])
            pred = np.array(data['pred_total'])
            mae = np.mean(np.abs(pred - gt))
            rel = np.mean(np.abs(pred - gt) / (np.abs(gt) + 1e-12))
            print(f"{split_name}: MAE={mae:.6f}, Mean Rel Err={rel*100:.2f}% (n={len(gt)})")

    summarize('Train', train_forces)
    summarize('Val', val_forces)
    summarize('Test', test_forces)


def plot_total_force_comparison(train_loader, val_loader, test_loader, model, normalization_scalars, output_path):
    # Optional helper if needed later; not used by main.py
    train_forces, val_forces, test_forces = total_force_compute(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=model,
        normalization_scalars=normalization_scalars,
        predicted_feature_name='x_force',
        model_name='transolver',  # placeholder; call explicitly with desired model if used
    )
    plot_force_comparison(train_forces, val_forces, test_forces, 'x_force', output_path)

