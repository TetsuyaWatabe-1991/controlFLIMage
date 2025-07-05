import numpy as np
from skimage.io import imread, imsave
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.morphology import binary_opening, binary_closing, disk, erosion, dilation

def create_roi_from_spine_and_shaft(spine_path, shaft_path, 
                                    spine_threshold=0.15, 
                                    shaft_threshold=0.10,
                                    watershed_threshold=0.5,
                                    separation_pixels=1,  # 分離するピクセル数
                                    plot_them=True):
    spine_img = imread(spine_path, as_gray=True)
    shaft_img = imread(shaft_path, as_gray=True)

    # 閾値処理 -> マスク作成
    spine_mask = spine_img > spine_threshold
    shaft_mask = shaft_img > shaft_threshold

    # ImageJ風前処理：モルフォロジー操作でノイズ除去
    # 小さなノイズを除去
    spine_mask = binary_opening(spine_mask, disk(1))
    # 穴を埋める
    spine_mask = binary_closing(spine_mask, disk(2))
    
    shaft_mask = binary_opening(shaft_mask, disk(1))
    shaft_mask = binary_closing(shaft_mask, disk(2))

    # ImageJ風watershed実装
    # 1. 距離変換（ImageJ風：より細かい設定）
    distance = ndi.distance_transform_edt(spine_mask)
    
    # 2. ImageJ風マーカー作成（より単純なアプローチ）
    # 距離変換の結果を8-bitに正規化（ImageJ風）
    distance_array = np.array(distance)
    distance_norm = ((distance_array - distance_array.min()) / (distance_array.max() - distance_array.min()) * 255).astype(np.uint8)
    
    # 3. 局所最大値検出（ImageJのwatershedに近い方法）
    # より大きなフィルタサイズで局所最大値を検出
    local_max = ndi.maximum_filter(distance_norm, size=5) == distance_norm
    local_max = local_max & (distance_norm > watershed_threshold * 255)
    
    # 4. マーカーのラベリング
    markers = label(local_max)
    
    # 5. watershed実行（ImageJ風）
    # ImageJのwatershedは通常、マーカーなしで実行して物体を分離する
    # まず、マーカーなしでwatershedを実行（ImageJのデフォルト動作）
    spine_watershed = watershed(-distance_array, mask=spine_mask, 
                               connectivity=2, compactness=0)
    
    # マーカーがある場合は、マーカー付きwatershedも試す
    markers_array = np.array(markers)
    if np.sum(markers_array) > 0:
        # マーカー付きwatershed（より細かい制御が必要な場合）
        spine_watershed_marked = watershed(-distance_array, markers, mask=spine_mask, 
                                          connectivity=2, compactness=0)
        # マーカー付きの方が良い結果の場合は使用
        if np.max(spine_watershed_marked) > np.max(spine_watershed):
            spine_watershed = spine_watershed_marked
    
    # 物体間の分離を強化：1ピクセル離す
    # 各ラベルを個別に処理
    unique_labels = np.unique(spine_watershed)
    if len(unique_labels) > 1:  # 背景以外にラベルがある場合
        separated_watershed = np.zeros_like(spine_watershed)
        
        # 隣接関係をチェック
        eroded_labels = []  # Erode処理されたラベルの記録
        
        for label_id in unique_labels[1:]:  # 背景（0）以外を処理
            # 各ラベルのマスクを作成
            label_mask = (spine_watershed == label_id)
            
            # このラベルが隣接する他のラベルがあるかチェック
            # 膨張して他のラベルと接触するかどうかを確認
            dilated_mask = dilation(label_mask, disk(1))
            other_labels = np.unique(spine_watershed[dilated_mask])
            other_labels = other_labels[other_labels != label_id]  # 自分以外のラベル
            other_labels = other_labels[other_labels != 0]  # 背景を除く
            
            if len(other_labels) > 0:
                # 隣接する物体がある場合：1ピクセルErode
                eroded = erosion(label_mask, disk(1))
                separated_watershed[eroded] = label_id
                eroded_labels.append(label_id)
                print(f"ラベル {label_id}: 隣接物体あり → Erode処理実行")
            else:
                # 隣接する物体がない場合：そのまま
                separated_watershed[label_mask] = label_id
                print(f"ラベル {label_id}: 隣接物体なし → そのまま")
        
        spine_watershed = separated_watershed
        
        if len(eroded_labels) > 0:
            print(f"Erode処理された物体数: {len(eroded_labels)}")
        else:
            print("隣接する物体はありませんでした")
    
    # シャフトを引いてスパインのみのマスクに
    spine_only_mask = np.logical_and(spine_watershed > 0, np.logical_not(shaft_mask))

    # ラベリングしてROI抽出
    labeled = label(spine_watershed)
    regions = regionprops(labeled)

    # ROIの出力例（Bounding box）
    rois = []
    for region in regions:
        if region.area >= 10:  # 小さすぎるノイズ除去
            rois.append(region.bbox)  # (min_row, min_col, max_row, max_col)

    if plot_them:
        #plot spine_mask, shaft_mask, spine_watershed, spine_only_mask
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0, 0].imshow(spine_mask, cmap='gray')
        axes[0, 0].set_title("Spine Mask")
        axes[0, 0].axis('off')
        axes[0, 1].imshow(shaft_mask, cmap='gray')
        axes[0, 1].set_title("Shaft Mask")
        axes[0, 2].imshow(distance_norm, cmap='gray')
        axes[0, 2].set_title("Distance Transform (8-bit)")
        axes[0, 2].axis('off')
        axes[1, 0].imshow(local_max, cmap='gray')
        axes[1, 0].set_title("Local Maxima")
        axes[1, 0].axis('off')
        axes[1, 1].imshow(spine_watershed, cmap='nipy_spectral')
        axes[1, 1].set_title("Spine Watershed (Erode if Adjacent)")
        axes[1, 1].axis('off')
        axes[1, 2].imshow(spine_only_mask, cmap='gray')
        axes[1, 2].set_title("Spine Only Mask")
        axes[1, 2].axis('off')
        plt.tight_layout()
        plt.show()
        
        # 分離の効果を確認するための追加プロット
        print(f"検出されたスパイン数: {len(np.unique(spine_watershed)) - 1}")  # 背景（0）を除く

    return rois, spine_only_mask, spine_watershed

# 使用例
# spine_path = r"C:\Users\WatabeT\Desktop\20250701\auto1\lowmag2__highmag_2\lowmag2__highmag_2_19_S_spine.tif"
# shaft_path = r"C:\Users\WatabeT\Desktop\20250701\auto1\lowmag2__highmag_2\lowmag2__highmag_2_19_S_shaft.tif"
# shaft_path = r"C:\Users\WatabeT\Desktop\20250701\auto1\tif\deepd3\lowmag1__highmag_7__3_S_shaft.tif"
# spine_path = r"C:\Users\WatabeT\Desktop\20250701\auto1\tif\deepd3\lowmag1__highmag_7__3_S_spine.tif"

shaft_path = r"C:\Users\WatabeT\Desktop\20250701\auto1\tif\deepd3\lowmag1__highmag_3__2_S_shaft.tif"
spine_path = r"C:\Users\WatabeT\Desktop\20250701\auto1\tif\deepd3\lowmag1__highmag_3__2_S_spine.tif"
original_image_path = r"C:\Users\WatabeT\Desktop\20250701\auto1\tif\lowmag1__highmag_3__2.0_after_align.tif"

rois, spine_mask, watershed_result = create_roi_from_spine_and_shaft(spine_path, shaft_path, plot_them=True)

# 結果の確認
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 元のスパインマスク
axes[0].imshow(spine_mask, cmap='gray')
axes[0].set_title("Spine Mask")
axes[0].axis('off')

# watershed結果
axes[1].imshow(watershed_result, cmap='nipy_spectral')
axes[1].set_title("Watershed Result")
axes[1].axis('off')

# ROI表示
axes[2].imshow(spine_mask, cmap='gray')
for bbox in rois:
    minr, minc, maxr, maxc = bbox
    rect = Rectangle((minc, minr), maxc - minc, maxr - minr,
                     edgecolor='red', facecolor='none', linewidth=2)
    axes[2].add_patch(rect)
axes[2].set_title("Spine ROIs")
axes[2].axis('off')

plt.tight_layout()
plt.show()

print(f"検出されたROI数: {len(rois)}")