"""Audit and aggregate the completed strong-baseline/transfer extension."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import statistics

EXP = Path(__file__).resolve().parents[1]
OLD = EXP/'results/refinement_20261103'
RUN = EXP/'results/extension_20261103'
KEYS = ['L2_percent', 'gradient_percent', 'flux_L2_percent', 'total_ms']


def read(path):
    return json.loads(path.read_text())


def values(doc, name):
    return [statistics.mean(row['methods'][name][key] for row in doc['samples'])
            for key in KEYS]


def formatted(v):
    return ' / '.join(f'{x:.6f}' for x in v[:3])+f'；{v[3]:.2f}'


def main():
    lines = ['# 多尺度椭圆后校正器：强基线、机制消融与网格迁移', '',
        '2026-11-03。该轮继承冻结128维校正器，新增公开DCNO实现、官方M2NO实现、同初始化消融、非规则三角网格迁移及固定小系统去偏。所有三项误差单位均为%，相对于一致的同网格P1参考解。', '',
        '## 1. 同条件神经基线与跨基底校正', '',
        '两类材料各256训练、32验证、32测试；四个NO均150epochs，batch8，相同训练归一化与L2+0.1导数监督。校正器始终使用原48个六尺度材料所训练的冻结权重，未在这些新NO上训练。', '',
        '数据沿用HANO文献的材料分布并重新生成一致P1标签。129²节点，单位方形、单位源项、零Dirichlet边界。时间为RTX4090、batch=1、输入已在GPU上的同步墙钟：每个样本重新执行NO、材料装配、字典生成、小系统分解及全部更新，再跨样本平均；一次共享网格计划单列。主表每方法一次预热、3次计时取中位数；第3节隔离计时采用5次。K2表示固定两轮完整校正。', '',
        '| 材料 | 基底 | 纯NO：场/导数/通量%；ms | NO＋校正K2：场/导数/通量%；ms | 场降误差 |',
        '| --- | --- | --- | --- | ---: |']
    csv_rows, sources = [], []
    for family in ('smooth', 'rough'):
        for method in ('FNO', 'MgNO', 'DCNO', 'M2NO'):
            if method in ('FNO', 'MgNO'):
                name = 'fno_h1' if method=='FNO' else 'mgno'
                path = OLD/family/f'frozen128_{name}'/'evaluation.json'
            else:
                path = RUN/family/f'{method}_corrected'/'evaluation.json'
            doc = read(path)
            assert len(doc['samples'])==32
            assert [r['index'] for r in doc['samples']]==list(range(32))
            assert Path(doc['checkpoint']).parts[-2]=='balanced128_train'
            before, after = values(doc, method), values(doc, 'dictionary_K2')
            lines.append(f'| {family} | {method} | {formatted(before)} | {formatted(after)} | {100*(1-after[0]/before[0]):.2f}% |')
            sources.append(path)
            for name in (method, 'dictionary_K1', 'dictionary_K2', 'dictionary_K4'):
                csv_rows.append(dict(experiment='baseline', dataset=family, method=method+' / '+name,
                    n=32, **dict(zip(KEYS, values(doc, name)))))
    lines += ['', 'DCNO模块取自固定提交的公开DCNO-ICLR2024代码，所用七个类与源文件AST一致；M2NO源文件逐字保留官方提交。M2NO使用官方Darcy的k4/c2/5层/[3,3,2,2,1]/LayerNorm配置，129节点输入零填充至160后裁回；DCNO为32通道、3个Fourier–dilated单元及官方解码器。这里比较同预算再训练，不将结果标为论文原始权重复现。', '',
        'M2NO的官方Darcy配置训练500epochs，本轮与其余模型统一为150epochs。表中直接报告这一预算下的实际结果；当前精度最强的新增对照为DCNO。', '',
        '进一步核查最佳验证权重的训练/验证/测试误差，区分拟合不足与泛化差距：', '',
        '| 材料 | 模型 | 训练场误差% | 验证场误差% | 测试场误差% |',
        '| --- | --- | ---: | ---: | ---: |']
    fit_path=RUN/'baseline_fit_audit.json'
    for row in read(fit_path):
        v=row['relative_L2_percent']
        lines.append(f'| {row["family"]} | {row["method"]} | {v["train"]:.4f} | {v["val"]:.4f} | {v["test"]:.4f} |')
    sources.append(fit_path)
    lines += ['', 'M2NO训练误差约0.31%/0.36%，测试为3.11%/5.44%；在本轮256样本条件下已经显现明显泛化差距，不能仅归因为epoch不足。其参数约1156万，DCNO约161万。完整预算复现与同预算对比属于两种不同评价。', '',
        '成熟AMG的匹配数值对照、逐材料重新设置与缓存应用时间继续保存在[上一轮主报告第4节](../refinement_20261103/RESULTS.md#4-与成熟数值基线的匹配比较)。本轮新增神经基线与校准实验，不重复运行已完成的数值对照。', '',
        '## 2. 容量与训练目标的独立检验', '',
        '两个128通道模型使用同一旧图网络初始化、相同输出初始化、相同48/12材料、4轮展开、20epochs和log目标；只改变最终读出的线性/非线性。测试统一使用六尺度独立后32材料。', '',
        '| 读出 | K2：场/导数/通量%；ms | K4：场/导数/通量%；ms |',
        '| --- | --- | --- |']
    for name, label in [('matched_linear', '线性128通道'), ('matched_nonlinear', '非线性128通道')]:
        path = RUN/(name+'_test')/'evaluation.json'
        doc = read(path)
        assert [r['index'] for r in doc['samples']]==list(range(32,64))
        lines.append(f'| {label} | {formatted(values(doc,"dictionary_K2"))} | {formatted(values(doc,"dictionary_K4"))} |')
        for method in ('dictionary_K2','dictionary_K4'):
            csv_rows.append(dict(experiment='readout_ablation',dataset='trigonometric_holdout',
                method=label+' / '+method,n=32,**dict(zip(KEYS,values(doc,method)))))
        sources.append(path)
    lines += ['', '线性读出的空间秩至多17；非线性读出解除该代数限制。此处使用匹配训练比较，区分结构表达能力与额外训练带来的收益。', '',
        '| 四轮训练目标 | K2：场/导数/通量%；ms | K4：场/导数/通量%；ms |',
        '| --- | --- | --- |']
    for path, label in [(RUN/'loss_rms4_test/evaluation.json', 'RMS'),
                        (OLD/'balanced128_holdout32/evaluation.json', '逐轮三输出log平均')]:
        doc = read(path)
        assert [r['index'] for r in doc['samples']]==list(range(32,64))
        lines.append(f'| {label} | {formatted(values(doc,"dictionary_K2"))} | {formatted(values(doc,"dictionary_K4"))} |')
        for method in ('dictionary_K2','dictionary_K4'):
            csv_rows.append(dict(experiment='loss_ablation',dataset='trigonometric_holdout',
                method=label+' / '+method,n=32,**dict(zip(KEYS,values(doc,method)))))
        sources.append(path)
    lines += ['', '这组从同一direct128权重继续训练20epochs，其余样本、种子、学习率与四轮展开保持相同。耗时记录来自各自运行；该消融重点比较误差。', '',
        '## 3. 固定小系统去偏：不增加全网格算子调用', '',
        r'原校准求解$Hc=q$，其中$H=G+\eta I$、$G=Z^TAZ$、$q=Z^Tr$、$\eta=10^{-5}$。正则化保证相关方向上的数值稳定，也会留下幅值偏差。固定$t$次小求解复用同一Cholesky分解：', '',
        '$$',
        r'd_0=H^{-1}q,\qquad d_{j+1}=H^{-1}(\eta d_j),\qquad c_t=\sum_{j=0}^{t-1}d_j.',
        '$$', '',
        r'在精确Gram代数下，对特征值$\lambda\ge0$，修正增益为$1-(\eta/(\lambda+\eta))^t\in[0,1]$。因此固定去偏逐步减少正则化偏差，并保持能量收缩。每多一次仅增加$O(r^2)$的三角求解，不增加$A$作用、图网络、字典或因子分解。该数值去偏是校准完善，不单独作为架构创新。', '',
        '| 材料，n=16 | 小求解次数 | K2场/导数/通量%；ms | 场误差逐样本优于原校准 |',
        '| --- | ---: | --- | ---: |']
    for family in ('smooth','rough','trigonometric'):
        path = RUN/f'calibration_isolated/{family}_FNO/evaluation.json'
        doc = read(path)
        assert len(doc['samples'])==16
        for count, name in [(1,'dictionary_K2'),(2,'dictionary_D2_K2'),(4,'dictionary_D4_K2')]:
            wins=sum(s['methods'][name]['L2_percent']<s['methods']['dictionary_K2']['L2_percent'] for s in doc['samples'])
            lines.append(f'| {family} | {count} | {formatted(values(doc,name))} | {wins}/16 |')
            csv_rows.append(dict(experiment='calibration',dataset=family,method=name,n=16,
                **dict(zip(KEYS, values(doc,name)))))
        sources.append(path)
    lines += ['', '该表来自其余本轮训练完成后，在空闲GPU5上顺序执行的计时，每个方法一次预热、5次同步计时取中位数。首轮GPU2运行出现主机/GPU负载波动，原始记录保留在`debias_*`，此处使用隔离计时。重复计时没有重复训练。代数审计20/20个非M矩阵SPD系统验证单次校准与旧实现逐位一致，2/4次去偏的能量误差依次不增。', '',
        '| 材料与强基底，n=16 | 原校准K2：场/导数/通量%；ms | 两次小求解K2：场/导数/通量%；ms |',
        '| --- | --- | --- |']
    for family in ('smooth','rough'):
        for base in ('DCNO','M2NO'):
            path=RUN/f'calibration_isolated/{family}_{base}/evaluation.json'
            doc=read(path)
            assert len(doc['samples'])==16
            lines.append(f'| {family} / {base} | {formatted(values(doc,"dictionary_K2"))} | {formatted(values(doc,"dictionary_D2_K2"))} |')
            for name in ('dictionary_K2','dictionary_D2_K2','dictionary_D4_K2'):
                csv_rows.append(dict(experiment='calibration',dataset=family+' / '+base,method=name,n=16,
                    **dict(zip(KEYS,values(doc,name)))))
            sources.append(path)
    lines += ['', '固定两次小求解作为推荐的轻量校准选项，在FNO、DCNO、M2NO与三类材料上统一使用；图网络与全部物理阶段保持原样。', '',
        '## 4. 冻结权重的非规则网格迁移', '',
        '将129²网格内部节点随机移动最多0.3个单元宽度，再用Delaunay重构连接；域仍为单位方形。材料节点值随节点搬移，在新网格重新生成P1真值。FNO在原规则网格预测，通过显式GPU双线性观测得到新网格初值；完整时间包含该观测。校正器、滤波参数与权重均不变。', '',
        '| 方法，n=16 | 场/导数/通量%；完整ms |',
        '| --- | --- |']
    path = RUN/'mesh_transfer/evaluation.json'
    doc = read(path)
    assert len(doc['samples'])==16
    for name in ('interpolated_FNO','filter_only_K2','dictionary_K2','dictionary_K4'):
        lines.append(f'| {name} | {formatted(values(doc,name))} |')
        csv_rows.append(dict(experiment='unstructured_transfer',dataset='rough',method=name,n=16,
            **dict(zip(KEYS, values(doc,name)))))
    sources.append(path)
    lines += ['', f'新网格一次计划构建{doc["mesh_plan_ms_once"]:.1f} ms；逐材料在线时间不含该共享网格计划。该实验检验后校正器对网格及连接变化的迁移，基底FNO依然在规则网格运行。', '',
        '## 5. 材料对比度外推', '',
        '保持粗糙两相测试前16个样本的相结构和网格，将低相2降至0.2、高相12不变，对比度从6升至60；重新生成一致P1真值。NO与校正器均不重训，NO实际接收新的系数场。', '',
        '| 方法 | 场/导数/通量%；完整ms |',
        '| --- | --- |']
    path=RUN/'contrast60/evaluation.json'
    doc=read(path)
    assert len(doc['samples'])==16
    for name in ('FNO','filter_only_K2','dictionary_K2','dictionary_D2_K2','dictionary_D4_K2','dictionary_K4'):
        lines.append(f'| {name} | {formatted(values(doc,name))} |')
        csv_rows.append(dict(experiment='contrast_transfer',dataset='rough_contrast60',
            method=name,n=16,**dict(zip(KEYS,values(doc,name)))))
    sources.append(path)
    path=RUN/'contrast_diagnosis.json'
    diagnosis=read(path)
    lines += ['', '对前4个配对材料进一步分离初值幅值、滤波与字典覆盖：', '',
        '| 材料对比度 | NO场误差% | 真值辅助最佳幅值后的场误差% | 预滤波场误差% | 字典一次捕获的剩余A能量 | 小系统4次捕获 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for case in diagnosis['cases']:
        rows=case['samples']
        no=statistics.mean(r['NO_percent'][0] for r in rows)
        oracle=statistics.mean(r['oracle_amplitude_percent'][0] for r in rows)
        filtered=statistics.mean(r['prefilter_percent'][0] for r in rows)
        captures=[statistics.mean(r['captured_energy_fraction'][k] for r in rows) for k in ('1','4')]
        contrast=60 if 'contrast60' in case['data'] else 6
        lines.append(f'| {contrast} | {no:.3f} | {oracle:.3f} | {filtered:.3f} | {100*captures[0]:.2f}% | {100*captures[1]:.2f}% |')
    lines += ['', '该诊断显示，两种机制同时变化：预测初值的形状误差增大，且相同字典对实际剩余误差的能量覆盖下降。真值辅助幅值拟合后，外推组场误差仍为46.36%；单独调幅不足以恢复方向。增加小求解只能把覆盖从57.15%提高到59.34%，主要缺口在空间覆盖，下一步应围绕材料连接结构与剩余方向建模。这里的真值辅助操作仅用于离线诊断。']
    sources.append(path)
    lines += ['', '## 6. 研究判断与下一步优先级', '',
        '1. 当前流程继续围绕正标量扩散控制的多尺度椭圆问题；按方程结构共享矩阵核、图表示和校准，不增加按材料名称选择的流程。',
        '2. 新版结构与多轮目标分别接受同条件消融；小系统去偏作为固定成本校准选项，保留原单次版本用于复现。',
        '3. 下一批主实验优先补独立细网格收敛评价：分别度量相对当前离散方程的代数误差和相对细化参考解的空间离散误差，再确定论文的最终分辨率。',
        '4. 扩大公开测试优先选择HANO公开多尺度标签、PDEBench Darcy，以及ELADO中的正定Poisson子集。新增数据必须明确系数、源项、边界与标签离散；完整物理重建条件比数据集数量更重要。',
        '5. 最终准出再进行多seed和重复训练；本轮每个科学配置只训练一次。', '',
        '## 7. 相关工作与实现来源', '',
        '- [DCNO论文](https://arxiv.org/abs/2408.00775)：Fourier全局交互与膨胀卷积局部交互结合，针对多尺度及高频信息。[公开实现](https://github.com/zhuhanqing/DCNO-ICLR2024)，固定提交见`vendor/dcno/PROVENANCE.json`。',
        '- [M2NO论文](https://arxiv.org/html/2406.04822v3)：预定义多小波传递与隐空间多级处理；也考察预条件用途。[官方实现](https://github.com/lizhihao2022/M2NO)，配置与提交保存在`vendor/m2no/`。',
        '- [ELADO](https://arxiv.org/html/2606.20771v1)：2026年提出以谱分布迁移、尾部及输入敏感性诊断椭圆算子失效。[公开数据](https://zenodo.org/records/20738464)；本轮完成适配性调研，没有将它计入已完成实验。', '',
        '运行入口：`baseline_train.py --help`、`train_compare.sh {smooth,rough} {DCNO,M2NO}`、`study.py evaluate --compare-calibration`、`mesh_transfer.py --help`。原始结果不覆盖；完整参数和训练日志保留在各运行目录。']
    (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    with (RUN/'SUMMARY.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['experiment','dataset','method','n',*KEYS])
        writer.writeheader();writer.writerows(csv_rows)
    checkpoint=OLD/'balanced128_train/model.pt'
    checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert checkpoint_hash==read(OLD/'AUDIT.json')['checkpoint_sha256']
    for a,b in [('matched_linear','matched_nonlinear'),('loss_rms4',None)]:
        first=read(RUN/a/'provenance.json')['arguments']
        second=read(RUN/b/'provenance.json')['arguments'] if b else read(OLD/'balanced128_train/provenance.json')['arguments']
        for key in ('initialize','seed','train_count','val_count','epochs','cycles','lr','rank','width'):
            assert first[key]==second[key], (a,b,key)
    audit=dict(completed=True, baseline_rows=8, unique_new_training_configurations=7,
        checkpoint_sha256=checkpoint_hash,matched_initialization_ablation_checked=True,
        structured_rows=len(csv_rows),
        sources={str(p.relative_to(EXP)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (RUN/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(RUN/'RESULTS.md')


if __name__=='__main__':
    main()
