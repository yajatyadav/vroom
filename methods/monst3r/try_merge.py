import open3d as o3d
import numpy as np

# ——— Parameters ———
run1_path = "/home/username/cs280/vroom/monst3r/demo_tmp/cl_top_half_24_fps_55_to_60_500_win_16/scene.glb"
run2_path = "/home/username/cs280/vroom/monst3r/demo_tmp/cl_top_half_24_fps_59_to_64/scene.glb"
voxel_size = 0.05           # down‐sample voxel size (meters)
ransac_dist = voxel_size * 1.5
icp_dist    = voxel_size * 0.4

# ——— Helpers ———
def preprocess(pcd, voxel_size):
    # 1) Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # 2) Estimate normals
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2.0, max_nn=30))

    # 3) Compute FPFH features
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5.0, max_nn=100))
    return pcd_down, fpfh

# ——— Load & Preprocess ———
pcd1 = o3d.io.read_point_cloud(run1_path)
pcd2 = o3d.io.read_point_cloud(run2_path)
pcd1_down, fpfh1 = preprocess(pcd1, voxel_size)
pcd2_down, fpfh2 = preprocess(pcd2, voxel_size)

# ——— Global Registration via RANSAC ———
result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    source=pcd2_down, target=pcd1_down,
    source_feature=fpfh2,  target_feature=fpfh1,
    mutual_filter=True,
    max_correspondence_distance=ransac_dist,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    ransac_n=4,
    checkers=[
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(ransac_dist)
    ],
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
)
print("RANSAC fitness:", result_ransac.fitness,
      "inlier_rmse:", result_ransac.inlier_rmse)

# ——— Refine with ICP ———
result_icp = o3d.pipelines.registration.registration_icp(
    source=pcd2, target=pcd1,
    max_correspondence_distance=icp_dist,
    init=result_ransac.transformation,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
)
print("ICP fitness:", result_icp.fitness,
      "inlier_rmse:", result_icp.inlier_rmse)

# ——— Transform & Merge ———
pcd2.transform(result_icp.transformation)
merged = pcd1 + pcd2
# (optional) down‐sample merged for cleanliness
merged_down = merged.voxel_down_sample(voxel_size / 2.0)

# ——— Save & Visualize ———
o3d.io.write_point_cloud("merged.ply", merged_down)
print("Merged point cloud saved to merged.ply")
o3d.visualization.draw_geometries([merged_down])