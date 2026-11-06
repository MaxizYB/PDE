# CDMC 阶段性交接仓库

交接日期：2026-11-06。这是现有研究的独立快照，不是一个重新设计的方法版本。

## 接手顺序

1. 读 [当前研究与进度](handoff/CURRENT_HANDOFF.md)：研究对象、确定的骨架、近期结果和未完成工作。
2. 读 [历史教训与用户要求](handoff/HISTORY_AND_REQUIREMENTS.md)：不要再走的弯路，以及近期讨论的取舍。
3. 读 [运行与服务器资产](handoff/RUNNING_AND_SERVER.md)：代码入口、数据与权重位置、复现命令。
4. 按需要阅读原文，不要一次性加载所有历史文件。

## 当前最重要的原文

| 用途 | 文件 |
| --- | --- |
| 当前论文叙事、方法与已有机制证据 | [论文初稿](CDMC/reports/CDMC_NATURE_MANUSCRIPT_EXTENDED_20261027.md) |
| 论文主张、证据与相关工作骨架 | [论证骨架](CDMC/reports/CDMC_PAPER_ARGUMENT_BLUEPRINT.md) |
| 128 维统一响应字典 | [方法说明](CDMC/experiments/multiscale_elliptic_v1/refinement/README.md) |
| TB 强基线之后的稳定空间改进 | [方法](CDMC/experiments/multiscale_elliptic_v1/refinement/FRAME_SPACE_METHOD.md)、[测试报告](CDMC/experiments/multiscale_elliptic_v1/results/stable_space_20261104/REPORT.md) |
| 原始 TB、HINTS 对比 | [TB](CDMC/experiments/multiscale_elliptic_v1/results/trunk_basis_20261104/REPORT.md)、[HINTS](CDMC/experiments/multiscale_elliptic_v1/results/hints_20261104/REPORT.md) |
| 研究循环与上下文管理 | [研究规范](Nature/RESEARCH_PROTOCOL.md) |

论文初稿主要对应 11 月 3 日的统一响应字典；11 月 4 日的 TB 对照与 11 月 6 日的未完成探索以交接说明及对应实验报告为准，不能把几轮版本混成同一个已验证方法。

## 目录

~~~text
AGENTS.md                       原指导文件的脱敏副本
handoff/                        交接、历史教训、资产与完整性清单
CDMC/experiments/multiscale_elliptic_v1/
                                当前研究代码、近期基线和原始结果
CDMC/experiments/v2/             上述代码依赖的原数值/图层级实现
CDMC/reports/                   选定论文、汇报原文和配图
CDMC/methods/v1/                少量必要的历史方法与创新点说明
baselines/data/                 数据生成器和本地已有的小数据
Nature/                         研究规范与记录模板
tools/                          快照来源与验证工具，不属于方法本体
~~~

保留源项目相对目录，避免破坏现有导入和检查点引用。没有搬进全部 MetaPort 历史版本、虚拟环境、缓存或完整旧项目；没有重构研究代码。本地已有小检查点一并保留。服务器大数据和缺失权重按原路径使用，见运行说明。

## 完整性与安全

- 原项目文件不移动、不删除、不修改。复制文件的来源哈希见 [SNAPSHOT_MANIFEST.json](handoff/SNAPSHOT_MANIFEST.json)。
- 原文按原样复制；仅服务器 IP 和密码作脱敏，变动逐文件列在清单中。
- [AGENTS.md](AGENTS.md) 是脱敏副本，原项目指导文件没有改动。凭据由使用者私下配置，不写进仓库。
- 这是本地 Git 仓库，没有设置远程仓库或上传到第三方；第三方源码的现有许可和来源文件保留。对外发布前需核查第三方材料许可。
- 可运行 CPU 检查；GPU 训练仍需服务器环境及对应资产。未完成的实验不因打包变成已完成。

从仓库根检查快照：

~~~bash
python tools/check_snapshot.py
~~~

只有 CPU 的机器先用已有 NumPy/SciPy 环境执行运行说明中的检查，不必安装 PyTorch。
