# powerline-geometry-perception

一个基于 **mmsegmentation** 的输电线几何感知前端项目，核心目标不是通用目标检测，而是面向输电线场景，稳定输出：

- **中心线位置**
- **局部几何走向**

当前主线版本已从轻量级 **ResNet18 + FPN** 双头基线，演化为基于 **SegFormer-B1** 的 **V2** 系统，并进一步加入了 **distance auxiliary supervision**、**大图滑窗测试** 与 **test-time geometric verifier**。

## 项目定位

本项目不是普通的框检测器，也不是面向粗目标的常规语义分割模型。

它更适合被理解为一个：

> **输电线几何感知前端**

用于解决以下问题：

- 细长导线的中心线定位
- 局部切向方向估计
- 复杂背景中相似直线结构的误检抑制
- 为后续无人机沿线飞行 / 巡检决策提供几何感知输入

## 当前主线（V2）

当前稳定主线如下：

```text
Image Patch / Image
  -> SegDataPreProcessor
  -> SegFormer-B1 / MixVisionTransformer Encoder
  -> SharedFusion
  -> CenterHead
  -> OrientationHead
  -> DistanceFieldHead（辅助，仅训练期）
  -> Slide / Whole Inference
  -> Optional Geometric Verifier
```

### 主输出

- **CenterHead**：二分类中心线预测（`background / wire-centerline`）
- **OrientationHead**：二维局部方向场（`ox, oy`）
- **DistanceFieldHead**：训练期几何辅助监督，用于约束特征学习

## 项目演变

### V1

- Backbone：**ResNet18**
- Neck：**FPN**
- Fusion：**SharedFusion**
- Heads：
  - CenterHead
  - OrientationHead
- 训练方式：针对 TTPLA 大图进行 patch 训练

### V2

- Backbone 升级为 **SegFormer-B1**
- 主线中 `neck=None`，不再强依赖固定 FPN
- SharedFusion 适配 transformer 多尺度特征
- 增加 **DistanceFieldHead** 作为辅助几何监督
- 增加 **slide inference** 用于大图测试
- 增加可选 **Geometric Verifier** 用于 test-time 杂线抑制

## 核心设计原则

1. **双头几何建模**  
   将“线在哪”和“线往哪走”分开建模。

2. **patch 训练**  
   面向 3840×2160 大图，保留细导线的局部高分辨率细节。

3. **几何辅助监督**  
   - orientation 仅在有效 centerline 区域监督
   - distance map 由 center mask 在线生成

4. **几何后处理**  
   通过 geometric verifier 对短小、孤立、偏方向的假线段进行过滤。

## 数据与标签

当前项目主要围绕 **TTPLA** 数据组织。

内部训练标签形式包括：

- `center/*.png`：中心线二值图
- `orient/*.npy`：局部二维方向场
- `images/*`：原始 RGB 图像

对于 V2 的 distance 辅助监督：

- `gt_distance_map` 不是人工标注
- 它是在数据加载阶段由 centerline mask **在线生成** 的

## 目录结构

```text
configs/
  powerline_v1/
projects/
  powerline_v1/
    datasets/
    models/
      heads/
      modules/
      segmentors/
    utils/
tools/
  train.py
  test.py
  debug/
work_dirs/
```

## 示例命令

### V2 + distance aux 训练

```bash
python tools/train.py configs/powerline_v1/powerline_v1_segformer_b1_aux_distance.py
```

### Debug 训练

```bash
python tools/train.py configs/powerline_v1/powerline_v1_segformer_b1_aux_distance_debug.py
```

### 大图滑窗测试

```bash
python tools/test.py \
  configs/powerline_v1/powerline_v1_segformer_b1_aux_distance_test_slide.py \
  work_dirs/powerline_v1_segformer_b1_aux_distance/iter_20000.pth
```

### 测试并保存可视化结果

```bash
python tools/test.py \
  configs/powerline_v1/powerline_v1_segformer_b1_aux_distance_test_slide.py \
  work_dirs/powerline_v1_segformer_b1_aux_distance/iter_20000.pth \
  --show-dir work_dirs/powerline_v1_segformer_b1_aux_distance/test_vis_iter_20000
```

### 检查辅助监督与预测结果

```bash
python tools/debug/inspect_aux_targets_and_preds.py \
  --config configs/powerline_v1/powerline_v1_segformer_b1_aux_distance_debug.py \
  --checkpoint work_dirs/.../iter_20000.pth \
  --num-samples 8 \
  --out-dir work_dirs/debug_vis
```

## Geometric Verifier 是什么

当前的 geometric verifier 是一个 **test-time 硬几何过滤模块**。

它会读取：

- centerline 概率图
- orientation 方向场
- 可选 distance 预测图

并基于以下条件进行组件级过滤：

- 面积 / 估计长度 / 细长度
- center 置信度
- 方向一致性
- 主方向聚类
- 可选的孤立短线过滤

它的作用是压制神经网络预测后仍然残留的：

- 短小杂线
- 孤立假线段
- 与主导线簇方向明显不一致的误检

## 当前状态

### 已完成主线

- ResNet18 + FPN 的 V1 基线：已完成
- SegFormer-B1 的 V2 主线：已完成
- distance 辅助监督：已完成并可训练
- 大图滑窗测试：已完成
- geometric verifier：已完成为 test-time 后处理模块

### 正在推进 / 探索中

- 更强的 skeleton / branch 级 verifier
- 更复杂背景下的误检抑制
- hard-negative 自动挖掘与回灌
- 比当前 distance 更丰富的几何辅助监督（如 attraction field）
- 与无人机巡检控制系统的进一步耦合

### 当前主线尚未真正启用

- **Tower auxiliary head**：代码骨架已写，但 tower 数据链尚未完整接通

## 本项目不是什么

本项目并不是一个通用 YOLO 风格目标检测器，也不是普通粗 mask 语义分割任务。

更准确地说，它是一个：

> **为输电线跟踪、沿线飞行与巡检服务的几何感知前端**

## 未来计划

- 将 geometric verifier 从组件级过滤升级为 branch-level / skeleton-level pruning
- 引入 hard-negative mining，持续压制易混淆线性背景
- 探索比当前 distance field 更强的几何监督形式
- 面向无人机沿线飞行与巡检任务，打通 sim-to-real 闭环
- 结合基于 PPO 的策略学习，逐步推进真机部署与自主飞行验证

## 致谢

- [OpenMMLab / mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
- TTPLA 及相关输电线视觉感知研究工作

## 说明

该仓库处于持续开发过程中。部分模块已属于稳定主线，另一些模块仍处于研究探索或工程占位阶段。
