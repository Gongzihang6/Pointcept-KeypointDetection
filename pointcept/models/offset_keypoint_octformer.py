import torch
import torch.nn as nn

from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch

try:
    import ocnn
    from ocnn.octree import Points
except ImportError:
    raise ImportError(
        "OffsetKeypointOctFormer 依赖于 ocnn 库，但在您的环境中未找到。\n"
        "请参考 Pointcept 文档或运行 `pip install ocnn` (需根据 CUDA 版本选择安装方式) 进行安装。"
    )

from pointcept.models.octformer.octformer_v1m1_base import (
    OctreeT,
    PatchEmbed,
    OctFormerStage,
    Downsample,
    OctFormerDecoder,
)


@MODELS.register_module("OffsetKeypointOctFormer")
class OffsetKeypointOctFormer(nn.Module):
    def __init__(
        self,
        in_channels=4,
        num_keypoints=6,
        hidden_dim=256,
        fpn_channels=168,
        channels=(96, 192, 384, 384),
        num_blocks=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 24),
        patch_size=26,
        stem_down=2,
        head_up=2,
        dilation=4,
        drop_path=0.5,
        nempty=True,
        octree_scale_factor=10.24,
        octree_depth=11,
        octree_full_depth=2,
        **kwargs,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.octree_scale_factor = octree_scale_factor
        self.octree_depth = octree_depth
        self.octree_full_depth = octree_full_depth
        self.stem_down = stem_down
        self.num_stages = len(num_blocks)
        self.patch_size = patch_size
        self.dilation = dilation
        self.nempty = nempty

        drop_ratio = torch.linspace(0, drop_path, sum(num_blocks)).tolist()

        self.patch_embed = PatchEmbed(in_channels, channels[0], stem_down, nempty)
        self.layers = torch.nn.ModuleList(
            [
                OctFormerStage(
                    dim=channels[i],
                    num_heads=num_heads[i],
                    patch_size=patch_size,
                    drop_path=drop_ratio[sum(num_blocks[:i]) : sum(num_blocks[: i + 1])],
                    dilation=dilation,
                    nempty=nempty,
                    num_blocks=num_blocks[i],
                )
                for i in range(self.num_stages)
            ]
        )
        self.downsamples = torch.nn.ModuleList(
            [
                Downsample(channels[i], channels[i + 1], kernel_size=[2], nempty=nempty)
                for i in range(self.num_stages - 1)
            ]
        )
        self.decoder = OctFormerDecoder(
            channels=channels, fpn_channel=fpn_channels, nempty=nempty, head_up=head_up
        )
        self.interp = ocnn.nn.OctreeInterp("nearest", nempty)

        output_dim = num_keypoints * 4
        self.head = nn.Sequential(
            nn.Linear(fpn_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

        self.reg_criterion = nn.L1Loss(reduction="none")
        self.cls_criterion = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, data_dict):
        coord = data_dict["coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"]

        if feat.shape[1] >= 3:
            normal = feat[:, :3]
        else:
            normal = torch.zeros_like(coord)

        batch = offset2batch(offset)
        point = Points(
            points=coord / self.octree_scale_factor,
            normals=normal,
            features=feat,
            batch_id=batch.unsqueeze(-1),
            batch_size=len(offset),
        )
        octree = ocnn.octree.Octree(
            depth=self.octree_depth,
            full_depth=self.octree_full_depth,
            batch_size=len(offset),
            device=coord.device,
        )
        octree.build_octree(point)
        octree.construct_all_neigh()

        current_feat = self.patch_embed(octree.features[octree.depth], octree, octree.depth)
        depth = octree.depth - self.stem_down

        octree_transformer = OctreeT(
            octree,
            self.patch_size,
            self.dilation,
            self.nempty,
            max_depth=depth,
            start_depth=depth - self.num_stages + 1,
        )

        features = {}
        for i in range(self.num_stages):
            depth_i = depth - i
            current_feat = self.layers[i](current_feat, octree_transformer, depth_i)
            features[depth_i] = current_feat
            if i < self.num_stages - 1:
                current_feat = self.downsamples[i](current_feat, octree_transformer, depth_i)

        point_features = self.decoder(features, octree)
        query_pts = torch.cat([point.points, point.batch_id], dim=1).contiguous()
        point_features = self.interp(point_features, octree, octree.depth, query_pts)

        pred_flat = self.head(point_features)
        pred = pred_flat.view(-1, self.num_keypoints, 4)

        result_dict = {}

        if "target" in data_dict:
            target = data_dict["target"]

            offset_gt = target[..., :3]
            mask_gt = target[..., 3]

            offset_pred = pred[..., :3]
            mask_logits = pred[..., 3]

            cls_loss = self.cls_criterion(mask_logits, mask_gt).mean()
            valid_mask = (mask_gt > 0.5).float()
            valid_mask_exp = valid_mask.unsqueeze(-1)

            raw_reg_loss = self.reg_criterion(offset_pred, offset_gt)
            masked_reg_loss = raw_reg_loss * valid_mask_exp
            reg_loss = masked_reg_loss.sum() / (valid_mask_exp.sum() * 3 + 1e-6)

            loss = cls_loss + reg_loss * 2.0
            result_dict["loss"] = loss

            if self.training:
                with torch.no_grad():
                    result_dict["train/cls_loss"] = cls_loss.item()
                    result_dict["train/reg_loss"] = reg_loss.item()
                    result_dict["train/offset_l1_err"] = (
                        torch.abs(offset_pred - offset_gt) * valid_mask_exp
                    ).sum() / (valid_mask_exp.sum() * 3 + 1e-6)

                    dist = torch.norm(offset_pred - offset_gt, p=2, dim=-1)
                    if "scale" in data_dict:
                        scale = data_dict["scale"]
                        if scale.ndim == 0:
                            scale = scale.view(1)
                        batch_idx = offset2batch(offset)
                        if scale.ndim == 1 and len(scale) > 1:
                            dist = dist * scale[batch_idx].unsqueeze(-1)
                        else:
                            dist = dist * scale.view(-1, 1)

                    valid_mask_f = valid_mask.float()
                    valid_sum = valid_mask_f.sum(dim=0).clamp(min=1e-6)
                    kp_real_dist = (dist * valid_mask_f).sum(dim=0) / valid_sum
                    kp_real_dist[valid_mask_f.sum(dim=0) == 0] = 0.0

                    result_dict["train/mean_dist"] = kp_real_dist.mean().item()
                    for i in range(self.num_keypoints):
                        result_dict[f"train/kp{i}_dist"] = kp_real_dist[i].item()

        if not self.training:
            final_pred = pred.clone()
            final_pred[..., 3] = torch.sigmoid(pred[..., 3])
            result_dict["pred"] = final_pred

        return result_dict