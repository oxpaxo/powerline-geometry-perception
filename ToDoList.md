## 0. 给 Claude Agent 的项目引导（先读完再开始）  
  
你现在进入的是一个**基于 mmsegmentation 的输电线视觉检测前端项目**，目标不是普通语义分割，也不是普通目标检测，更不是 YOLO 框检测。这个项目服务于**无人机沿输电线飞行 / 巡检**的前端几何感知，因此模型需要尽量稳定地回答两件事：  
  
1. **线在哪**  
2. **线在这里往哪走**  
  
所以当前系统不是单头分割，而是一个**双头系统**：  


Image  
  -> Backbone  
  -> Neck（可选）  
  -> SharedFusion  
      -> Center Head  
      -> Orientation Head  
  -> Postprocess

其中：

- **Center Head**  
    输出二分类中心线结果：`background / wire-centerline`
- **Orientation Head**  
    输出二维方向向量场：`[ox, oy]`

这两个 head 是**并联**关系，不是串联关系。也就是：

shared_feat -> center_head  
shared_feat -> orient_head

训练时：

- center 用 GT center 监督
- orientation 只在 GT wire 区域内计算损失

推理时：

- center 的结果可以用来过滤 orientation
- 然后再做几何后处理

---

## 1. 当前项目的核心问题是什么

当前主要问题不是“线断一点”，而是：

> **当背景中存在与导线形态相似的长直亮色物体/边缘时，会出现明显误检。**

典型误检来源包括但不限于：

- 车道线
- 杆塔强边缘
- 横担直边
- 建筑长直边
- 其它高亮、细长、线状结构

而且这个误检是：

- center 会误检
- orientation 也会误检

这说明问题不只是后处理阈值，而更像是：

> **共享特征把“长直线 = 导线”学错了。**

所以当前项目主线不是大改标签体系，也不是先上三分类，而是：

1. 先把 backbone 从 `ResNet18 + FPN` 升级到 `SegFormer-B1`
2. 在保留双头逻辑和二分类 center 的前提下，加更合理的辅助监督
3. 用最小可控变量做 ablation
4. 再视结果决定是否继续增强后处理与样本闭环

---

## 2. 这轮工作的总目标是什么

本轮不是“把所有想到的东西一次性全做完”。

本轮目标是：

### 必做

1. 稳定当前 `SegFormer-B1` 双头版本 baseline
2. 新增 **distance / attraction 类辅助监督**  
    本轮优先实现 **distance field auxiliary supervision**
3. 新增 **tower auxiliary head**  
    但前提是本地 workspace 中 tower 标签或 tower 转换链是可用的
4. 让所有新增模块都可以通过 config 开关独立启用/关闭
5. 完成最小可运行 ablation

### 本轮不要完整做

6. 不要完整实现 **geometry verifier**
7. 不要完整实现 **hard-negative auto reinjection**

这两项本轮只允许：

- 设计接口
- 写占位文件
- 说明后续计划

不允许把系统复杂度一次性炸开。

---

## 3. 为什么本轮先做 distance/tower，而不是四项一起做

因为这四项属于不同层面：

- **distance / attraction auxiliary supervision**  
    属于训练期辅助监督，直接改善 shared feature 的几何感知能力
- **tower auxiliary head**  
    属于训练期上下文约束，帮助 backbone 学“什么场景里更像真实导线”
- **geometry verifier**  
    属于推理后处理增强，不是训练主干的一部分
- **hard-negative auto reinjection**  
    属于数据闭环增强，依赖上一版模型先足够稳定

如果四项一起上，会同时改变：

- 网络结构
- 监督信号
- 后处理链
- 训练样本分布

最后你无法判断到底是哪项起作用。

所以这轮必须控制变量，优先顺序是：

1. baseline
2. - distance auxiliary supervision
3. - tower auxiliary head
4. - tower + distance
5. geometry verifier 只留接口
6. hard-negative 只留接口

---

## 4. 你必须遵守的项目工作流

### 4.1 不要破坏现有工程组织

优先在这些目录中工作：

configs/powerline_v1/  
projects/powerline_v1/  
tools/debug/  
tools/powerline_v1/   # 如果需要新增脚本目录，可以创建

### 4.2 不要修改 mmseg 核心源码

除非绝对必要，否则不要改：

mmseg/  
mmengine/

### 4.3 保持现有任务定义

- center 仍然是二分类，不升级成三分类
- orientation 仍然是二维方向向量场
- 现有 CenterHead / OrientationHead 尽量兼容
- 当前 segmentor 仍然沿用双头主结构
- 不要改成 mmseg 原生单 decode head 方案

### 4.4 所有新增功能必须支持 config 开关

要能通过 config 明确打开/关闭：

- distance head
- tower head
- 后续 verifier 接口
- 后续 hard-negative 接口

这样才能做 ablation。

### 4.5 每改完一个文件，必须输出修改说明

每修改一个文件，必须说明：

1. 文件路径
2. 修改目的
3. 具体改动点
4. 为什么这样改

### 4.6 每完成一个阶段，必须做最小验证

不能只写代码不验证。至少要做：

- config build
- model build
- debug forward/loss
- 最小可视化或日志检查

---

## 5. 当前项目的已知结构与背景

当前项目是真实工程，不是理论目录。整体应沿用以下组织思路：

configs/powerline_v1/  
├── powerline_v1_r18_fpn.py  
└── powerline_v1_r18_fpn_debug.py  
  
projects/powerline_v1/  
├── __init__.py  
├── datasets/  
│   ├── __init__.py  
│   ├── powerline_dataset.py  
│   ├── transforms/  
│   │   ├── __init__.py  
│   │   ├── formatting.py  
│   │   ├── geom_transforms.py  
│   │   └── loading.py  
│   └── TTPLA/  
│       ├── center/  
│       ├── images/  
│       ├── line_mask/  
│       ├── orient/  
│       ├── splits/  
│       └── convert_ttpla_to_v1.py  
├── models/  
│   ├── __init__.py  
│   ├── heads/  
│   │   ├── __init__.py  
│   │   ├── centerline_head.py  
│   │   └── orientation_head.py  
│   ├── modules/  
│   │   ├── __init__.py  
│   │   └── shared_fusion.py  
│   └── segmentors/  
│       ├── __init__.py  
│       └── powerline_segmentor.py  
├── utils/  
│   ├── __init__.py  
│   └── postprocess.py  
└── visualization/  
    ├── __init__.py  
    └── powerline_visualizer.py

当前训练输入以 patch 为主，不是整图直接训练。原图较大，当前训练 crop 通常是：

crop_size = (512, 1024)

---

## 6. 本轮执行总原则

1. **先把 baseline 稳住**
2. **先做训练期增强，再做推理期和数据闭环增强**
3. **先做可解释、可 ablation 的改动**
4. **不要把 attraction / verifier / hard-negative 一次性全做深**
5. **每一步都留后路，确保代码可回退**
6. **优先写“最小可用版本”，不要一开始就堆复杂设计**

---

# 7. 正式 ToDoList

---

## A. 全局约束（必须遵守）

1. 不要修改 `mmseg/` 主树核心源码，除非绝对必要。
2. 所有新增逻辑优先放在：
    - `projects/powerline_v1/`
    - `configs/powerline_v1/`
    - `tools/debug/`
    - `tools/powerline_v1/`（如需新增脚本目录，可创建）
3. 保留当前任务定义：
    - center 仍是二分类：`background / wire-centerline`
    - orientation 仍是 `[H, W, 2]` 单位方向向量监督
    - 现有 `center_head / orientation_head` 保持兼容
4. 当前 segmentor 仍然沿用双头主结构，不改成 mmseg 原生单 decode head。
5. 所有新增功能必须支持“开关可控”，能通过 config 单独开/关，便于 ablation。
6. 每改完一个文件，都要输出：
    - 文件路径
    - 修改目的
    - 具体改动点
    - 为什么这样改
7. 每完成一个阶段，都要实际运行最小验证命令；不能只写代码不验证。
8. 优先保证代码可跑通、可回退、可 ablation；避免大而全重构。
9. 代码风格尽量沿用当前项目。
10. 如遇 tower 标签在本地 workspace 中不存在，不要硬猜数据格式；先检索现有数据和脚本，再决定是“自动转换”还是“暂时跳过 tower full training，只保留接口”。

---

## B. 先做的事：baseline sanity check（必须先完成）

### B1. 检查以下文件是否已经在正确位置、且 import 链正常

- `projects/powerline_v1/models/segmentors/powerline_segmentor.py`
- `projects/powerline_v1/models/modules/shared_fusion.py`
- `configs/powerline_v1/powerline_v1_segformer_b1.py`
- `configs/powerline_v1/powerline_v1_segformer_b1_debug.py`

### B2. 检查以下 `__init__.py` 导出链是否完整

- `projects/powerline_v1/__init__.py`
- `projects/powerline_v1/models/__init__.py`
- `projects/powerline_v1/models/heads/__init__.py`
- `projects/powerline_v1/models/modules/__init__.py`
- `projects/powerline_v1/models/segmentors/__init__.py`

若缺失导入，补齐。

### B3. 运行最小 build sanity check

要求至少完成：

- 能成功 build config
- 能成功 build model
- 能成功跑 1 个 debug iteration（哪怕只到 forward / loss）

### B4. 如果当前 SegFormer-B1 baseline 仍未完全跑通

先修到跑通，再继续后面的工作。

> 本轮不允许在 baseline 都没稳时继续叠加辅助模块。

---

## C. 第一阶段增强：distance field auxiliary supervision（本轮必须完成）

### 目标

利用现有 wire center / line mask 自动生成辅助监督，不新增人工标注。

> 本轮优先实现 **distance field auxiliary supervision**  
> attraction field 只留扩展位，不要求完整实现。

### C1. 先检索当前 workspace 中和数据相关的这些文件

- `projects/powerline_v1/datasets/powerline_dataset.py`
- `projects/powerline_v1/datasets/transforms/loading.py`
- `projects/powerline_v1/datasets/transforms/geom_transforms.py`
- `projects/powerline_v1/datasets/transforms/formatting.py`
- `projects/powerline_v1/datasets/TTPLA/convert_ttpla_to_v1.py`

同时检查：

- 当前 TTPLA 目录下是否已有 `line_mask / center` 可直接生成 distance target

### C2. 设计原则

- 不新增人工标注
- distance target 从已有 `center` 或 `line_mask` 自动派生
- 优先用“在线生成”方案，避免强依赖离线预处理
- 若在线生成太复杂或太慢，可增加一个可选的离线 cache 脚本，但训练代码必须兼容两种方式

### C3. 新增一个轻量辅助 head

建议文件：

- `projects/powerline_v1/models/heads/distance_field_head.py`

要求：

- 输入：`shared_feat [N, C, H, W]`
- 输出：`distance_pred [N, 1, H, W]`
- loss：优先使用 `SmoothL1` 或 `L1`
- 支持 mask / normalization

配置项中允许设置：

- `loss_distance_weight`
- `target_normalize_mode`
- `max_distance_clip`

### C4. 在 `models/heads/__init__.py` 中导出新 head

### C5. 在数据流中增加 distance target

优先方案：

- 在 transform 或 formatting 阶段，根据 `gt_sem_seg` 或 `line_mask` 生成 `distance map`
- 最终打包进 `SegDataSample`，例如命名为 `gt_distance_map`

要求：

- 与 `crop / flip` 等几何增强严格同步
- 若 resize 发生，distance target 的 resize 策略要合理
- 对于 patch 内无正样本的情况也要健壮

### C6. 在 `PowerLineSegmentor` 中接入可选 auxiliary head

新增可选参数：

- `distance_head: Optional[dict] = None`

新增：

- `with_distance_head` 属性

要求：

- `loss()` 在存在 `gt_distance_map` 且启用 `distance_head` 时计算 auxiliary loss
- `predict()` 中可以不输出 distance map，或仅在 debug 模式输出
- 默认不影响现有推理接口

### C7. 配置层新增

- `configs/powerline_v1/powerline_v1_segformer_b1_aux_distance.py`
- `configs/powerline_v1/powerline_v1_segformer_b1_aux_distance_debug.py`

要求：

- 继承自当前 segformer baseline config
- 只增加 `distance_head` 和相关数据打包配置
- 其它尽量不改

### C8. 做最小 debug 验证

必须完成：

- build 通过
- 500 iter debug 可跑
- loss log 中出现 `distance loss`
- 保存若干可视化结果，至少包括：
    - 原图
    - gt center
    - gt distance
    - pred center
    - pred distance

### C9. attraction field 的处理方式

本轮不要完整重做 attraction field 全套。

只做下面两件事：

1. 在代码注释 / TODO 中明确：  
    distance head 未来如何扩展为 2-channel attraction field
2. 若结构上很容易兼容，可在接口层预留：
    - `task_type='distance'/'attraction'`

但默认只启用 `distance`，不要把本轮实现复杂化。

---

## D. 第一阶段增强：tower auxiliary head（本轮尽量完成，但要先检查数据可用性）

### 目标

利用 tower 语义给 backbone/shared feature 增加场景上下文约束，减少把车道线/建筑边误判为 wire。

### D1. 先检查 tower 数据是否实际可用

检查：

- 当前 TTPLA 本地目录里是否已有 tower mask / tower annotation 转换结果
- `convert_ttpla_to_v1.py` 是否能从原始 TTPLA 标注转换出 tower mask
- workspace 中是否还保留原始 TTPLA 标注文件

### D2. 根据检查结果分两种情况处理

#### 情况 1：tower 标签可直接获得或可自动转换

=> 完整实现 tower auxiliary head

#### 情况 2：tower 标签在当前 workspace 中不可获得

=> 不要胡编数据格式，不要强行实现训练闭环

只完成：

- tower auxiliary head 代码骨架
- segmentor 可选接入接口
- config 占位
- README/TODO 写清楚“需要 tower mask 或原始 TTPLA 标注才能启用”

### D3. 若 tower 数据可用，新增

- `projects/powerline_v1/models/heads/tower_head.py`

建议：

- `tower_head` 做 1-channel binary segmentation 即可
- 不做多类别复杂化
- loss 可沿用 `BCE + Dice`
- 尽量参考现有 `centerline_head` 写法
- 输入仍然是 `shared_feat`

### D4. 数据流

- 在 `loading / formatting` 里支持读取 tower GT
- 命名例如：`gt_tower_seg`
- 所有几何变换与 `crop / flip` 必须同步

### D5. 在 `PowerLineSegmentor` 中增加可选

- `tower_head: Optional[dict] = None`
- `with_tower_head` 属性

要求：

- `loss()` 中在启用时计算 tower loss
- `predict()` 默认可以不输出 tower 结果，或仅在 debug 中保留

### D6. 新增 config

- `configs/powerline_v1/powerline_v1_segformer_b1_aux_tower.py`
- `configs/powerline_v1/powerline_v1_segformer_b1_aux_tower_debug.py`

### D7. 若 tower 与 distance 都可用，再新增联合版本

- `configs/powerline_v1/powerline_v1_segformer_b1_aux_tower_distance.py`
- `configs/powerline_v1/powerline_v1_segformer_b1_aux_tower_distance_debug.py`

### D8. 做 debug 验证

必须完成：

- build 通过
- debug 训练可跑
- loss log 中出现 `tower loss`
- 保存若干 tower 可视化结果

---

## E. 必须完成的 ablation（本轮必须完成）

至少准备并验证这 4 组配置：

### 1. SegFormer baseline

- `powerline_v1_segformer_b1.py`
- `powerline_v1_segformer_b1_debug.py`

### 2. SegFormer + distance aux

- `powerline_v1_segformer_b1_aux_distance.py`
- `powerline_v1_segformer_b1_aux_distance_debug.py`

### 3. SegFormer + tower aux

- `powerline_v1_segformer_b1_aux_tower.py`
- `powerline_v1_segformer_b1_aux_tower_debug.py`

若 tower 数据不可用：

- 保留 config 占位
- 在最终报告中明确“未启用原因”

### 4. SegFormer + tower aux + distance aux

- `powerline_v1_segformer_b1_aux_tower_distance.py`
- `powerline_v1_segformer_b1_aux_tower_distance_debug.py`

若 tower 数据不可用：

- 该组同样只保留占位并说明原因

### 每组至少完成

- config build
- model build
- debug forward/loss
- 1 轮最小可视化

---

## F. 可视化与调试脚本（本轮必须做）

新增至少一个 debug 脚本，建议：

- `tools/debug/inspect_aux_targets_and_preds.py`

### 功能要求

从指定 `config + checkpoint` 或随机初始化模型读取一批样本，保存以下图像到 debug 输出目录：

- input image
- gt center
- pred center prob / binary
- gt orient（可箭头或 hsv）
- pred orient（可箭头或 hsv）
- gt distance（若启用）
- pred distance（若启用）
- gt tower（若启用）
- pred tower（若启用）

### 要求

- 脚本参数可配置
- 输出目录清晰
- 不依赖手工改脚本常量

---

## G. geometry verifier（本轮不要完整实现）

本轮只允许做“规划 + 占位”，不要完整做功能。

### 允许做

1. 在 `projects/powerline_v1/utils/` 下新增占位文件，例如：
    - `geometric_verifier.py`
2. 在文件头写清楚后续计划：
    - 输入 `center_prob + pred_orient_map`
    - 候选线生成
    - 连续性 / 方向一致性 / 支持度 / 长度筛选
3. 若 `postprocess.py` 中适合，保留一个未来可插拔接口，例如：
    - `use_geometric_verifier: bool = False`
    - `verifier_cfg = {...}`

但默认必须关闭，且不影响当前结果。

4. 写 README / TODO，解释 verifier 不是本轮实现项。

### 禁止做

- 不要在本轮写大量复杂拟合逻辑
- 不要把现有 postprocess 全部推翻
- 不要把推理链条一次性改炸

---

## H. hard-negative auto reinjection（本轮不要完整实现）

本轮只允许做“规划 + 占位”，不要完整实现训练闭环。

### 允许做

1. 新增占位脚本，例如：
    - `tools/powerline_v1/mine_hard_negatives.py`
2. 在脚本中先实现最基本的命令行框架、参数解析、输入输出约定
3. 在注释 / README 中明确后续方案：
    - 用当前模型跑 `train / extra images`
    - 通过 GT center 的膨胀区域外高置信预测挖掘 false positive 区域
    - 导出 patch list / crop metadata
    - 后续用于 sampler 或额外训练 split
4. 若很容易，可增加“只导出候选 hard-negative 坐标，不接训练”的最小版本

### 禁止做

- 不要本轮就把 `dataset / sampler / 训练闭环` 全部串起来
- 不要在没有验证 `baseline + aux` 的前提下直接修改训练数据分布

---

## I. 最终交付物（本轮必须输出）

### I1. 代码文件

至少应包含（视 tower 数据可用性而定）：

- 更新后的 segmentor
- 新增 `distance head`
- `tower head`（可用则完整，不可用则骨架）
- 数据流相关修改
- 新 configs
- debug 可视化脚本
- verifier 占位文件
- hard-negative mining 占位脚本

### I2. 运行结果

至少输出：

- baseline debug 通过日志
- aux_distance debug 通过日志
- tower 相关 debug 通过日志或不可启用原因
- 若联合版可用，联合版 debug 通过日志
- 若失败，给出精确报错和原因

### I3. 最终报告（必须简洁但完整）

包含：

1. 改了哪些文件
2. 每个文件改动目的
3. 哪些功能已完整实现
4. 哪些功能只做了占位，为什么
5. tower 数据是否在本地 workspace 可用
6. 建议下一轮优先继续做什么

---

## J. 本轮成功标准

满足以下条件才算本轮完成：

1. `SegFormer-B1 baseline` 能正常 build + debug train
2. `distance auxiliary` 版本能正常 build + debug train
3. `tower auxiliary` 若数据可用，则能正常 build + debug train
4. 所有辅助模块都能通过 config 开关独立启用/关闭
5. `geometry verifier` 和 `hard-negative auto reinjection` 只留占位，不把系统复杂度一次性炸开
6. 最终能清晰支持后续对比：
    - baseline
    - +distance
    - +tower
    - +tower+distance

---

## K. 建议的执行顺序（严格按顺序做）

1. 检查当前 workspace 文件与 import 链
2. baseline sanity check
3. distance auxiliary supervision
4. tower auxiliary head
5. 4 组配置与最小 ablation
6. 可视化脚本
7. verifier 占位接口
8. hard-negative 占位脚本
9. 最终报告

---

## L. 现在开始执行

从下面三步开始：

1. 检查当前 workspace 文件与 import 链
2. 跑通 SegFormer-B1 baseline sanity check
3. 实现并验证 distance field auxiliary supervision

完成后，再继续 tower auxiliary head。