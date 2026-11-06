"""Summarize the matched DeepONet-trunk correction-space experiment."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main(root):
    names = {'FNO':'FNO','filter_only_K2':'仅物理滤波',
             'TB80_native':'Trunk-basis 80（原始空间）',
             'TB80_shaped':'Trunk-basis 80（同材料平滑）',
             'TBqr_native':'Trunk-basis（QR筛选）',
             'CDMC128':'当前 CDMC 128','CDMC64':'已有 CDMC 64'}
    lines = ['# DeepONet trunk-basis 与 CDMC：学习空间直接对比', '',
        '日期：2026-11-04。平滑／粗糙两相 Darcy，129×129，分别32个未见测试样本。网络不重新训练。', '',
        '## 比较对象与流程', '',
        '采用 Kopaničáková & Karniadakis 的 trunk-basis（TB）构造：从预训练 DeepONet 取坐标主干函数，施加边界条件并 QR 正交化，使用当前矩阵建立小型 Galerkin 问题。预训练检查点复用本项目已完成的 HINTS 网络。', '',
        '本次固定 FNO 初值、细网格矩阵、物理预／后滤波与小系统求解，把中间校正空间替换为 TB。原论文以 TB 预条件器加速外层 Krylov；此处比较其空间构造在本项目固定次数后校正中的作用，未运行原论文的完整 Krylov 方案。', '',
        r'在线小系统为 $(Z^TAZ)c=Z^T(b-Au)$，更新 $u\leftarrow u+Zc$。实现沿用 CDMC 的列能量归一化、$10^{-5}I$ 正则与固定校准次数。TB 不调用 branch 预测系数，而由当前方程决定系数。', '',
        '- TB80 原始空间：保留全部80列 QR 基，无材料平滑。',
        '- TB80 同材料平滑：空间再经过与当前CDMC相同的材料平滑，隔离这部分处理的影响。',
        '- TBqr：保留 QR 对角幅值大于最大值 $10^{-10}$ 的列，记录实际维数。三个版本全部报告，不按测试真值选择。',
        '- CDMC128：原有 balanced128_train 检查点；CDMC64：该数据族已有 dictionary64 检查点。未修改它们的权重。',
        '- 全部版本使用同一 CDMC128 求解参数与 K=1/2/4。物理滤波对照保持相应滤波工作量。',
        '- trunk 用双精度计算和QR，随后以FP32进入共同在线矩阵作用；固定网格基及QR跨材料缓存。', '',
        '## 完整精度—时间结果', '',
        '误差为相对同一细网格 P1 真值的百分数。在线时间包含每次FNO、材料组装、空间设置、小系统分解及全部校正；固定网格计划和TB缓存单列。Batch=1，一次预热后3次同步计时中位数，再取样本均值。', '']
    all_rows=[]
    for family,title in (('smooth','平滑介质'),('rough','粗糙介质')):
        e=json.loads((root/family/'test32/evaluation.json').read_text())
        s=e['summary']
        lines += [f'### {title}', '',
                  '| 方法 | K | 场误差 % | 梯度误差 % | 通量误差 % | 总时间 ms |',
                  '|---|---:|---:|---:|---:|---:|']
        for name in s:
            if '_K' in name:
                base,k=name.rsplit('_K',1)
            else:
                base,k=name,'—'
            values=[s[name][key]['mean'] for key in ('L2_percent','gradient_percent','flux_L2_percent','total_ms')]
            lines.append('| '+names.get(base,base)+' | '+k+' | '+' | '.join(f'{v:.7g}' for v in values)+' |')
            all_rows.append(dict(dataset=family,method=name,**{key:value['mean'] for key,value in s[name].items()}))
        tb,cd=s['TB80_native_K2'],s['CDMC128_K2']
        ratios=[cd[key]['mean']/tb[key]['mean'] for key in ('L2_percent','gradient_percent','flux_L2_percent')]
        speed=cd['total_ms']['mean']/tb['total_ms']['mean']
        lines += ['', f'TB80原始空间K2的场、梯度、通量误差分别为当前CDMC128 K2的 1/{ratios[0]:.2f}、1/{ratios[1]:.2f}、1/{ratios[2]:.2f}，时间约快 {speed:.2f} 倍。', '',
                  f"QR筛选保留 {e['trunk']['stable_qr_columns']} / {e['trunk']['raw_columns']} 列；TB双精度QR正交性误差 {e['trunk']['orthogonality_error']:.3g}。", '',
                  f"一次性成本：网格计划 {e['mesh_plan_ms_once']:.2f} ms，装配计划 {e['assembly_plan_ms_once']:.2f} ms，TB检查点读取／坐标计算／QR缓存合计 {e['trunk']['cached_trunk_and_QR_ms_once']:.2f} ms。", '',
                  '离线机制诊断：在共同预滤波之后，单次空间投影捕获的实际误差能量比例（使用真值只作诊断，不参与在线预测）：', '',
                  '| 空间 | 维数 | 平均捕获比例 % |', '|---|---:|---:|']
        for space in e['diagnostics'][0]['spaces']:
            entries=[row['spaces'][space] for row in e['diagnostics']]
            lines.append(f"| {space} | {entries[0]['columns']} | {np.mean([x['captured_actual_error_energy_percent'] for x in entries]):.5f} |")
        audit_path=root/family/'qr_audit_val8.json'
        if audit_path.exists():
            q=json.loads(audit_path.read_text())['summary']
            lines += ['', '独立验证集8例的同空间QR诊断（保持全部128列、不训练）：', '',
                      '| CDMC128坐标表示 | K2场误差 % | 梯度误差 % | 通量误差 % |',
                      '|---|---:|---:|---:|']
            for name in ('original','same_space_QR'):
                vals=[q[name][key]['mean'] for key in ('L2_percent','gradient_percent','flux_L2_percent')]
                lines.append('| '+name+' | '+' | '.join(f'{v:.7g}' for v in vals)+' |')
        lines.append('')
    lines += ['## 本轮判断', '',
        '1. 直接预测逆作用的HINTS与提取其trunk进行方程投影的TB是不同强度的基线。本轮TB的性能显著强于此前HINTS直接修正，不能用HINTS结果替代空间类比较。',
        '2. 当前两类材料上，预训练固定trunk空间已经能够提供很强的后校正；当前CDMC的动态空间生成尚未形成对这个对照的优势。',
        '3. QR同空间诊断区分表示能力与数值校准：若同一空间经QR改善，部分损失来自坐标条件及正则，而非网络完全没有学到方向。该诊断未测在线QR速度，不构成新的速度结果。',
        '4. TB与CDMC的训练任务和预算不同：TB继承每类256材料／4096 RHS对的预训练，CDMC128继承已有六尺度材料训练。当前是部署效果与空间替换比较。', '',
        '## 复现与文件', '',
        '- baselines/trunk_basis/evaluate.py：完整测试入口；qr_audit.py：验证集数值诊断；report.py：生成本表和曲线。',
        '- 各数据族 test32/evaluation.json：逐样本、P95、频段、空间能量诊断、阶段计时与全部配置；summary.csv：主指标。',
        '- all_methods.csv：统一汇总；accuracy_time.png：三类误差—时间曲线。',
        '- 全部实验仅运行一个已训练检查点，不进行重复seed实验。', '',
        '## 文献', '',
        'Kopaničáková & Karniadakis, DeepONet Based Preconditioning Strategies for Solving Parametric Linear Systems of Equations，§4.2及§4.2.2： https://arxiv.org/html/2401.02016v2 。预训练网络采用已核对官方计算的HINTS结构，来源见 ../hints/vendor/PROVENANCE.json。', '']
    (root/'REPORT.md').write_text(chr(10).join(lines))
    keys=['dataset','method','L2_percent','gradient_percent','energy_percent','flux_L2_percent','total_ms']
    with (root/'all_methods.csv').open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=keys,extrasaction='ignore')
        w.writeheader();w.writerows(all_rows)
    plot(root,all_rows)


def plot(root,rows):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.ticker import NullFormatter
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for i,family in enumerate(('smooth','rough')):
        d={r['method']:r for r in rows if r['dataset']==family}
        for j,key in enumerate(('L2_percent','gradient_percent','flux_L2_percent')):
            ax=axes[i,j]
            for prefix,label,marker in [('TB80_native','Trunk-basis 80','o-'),('TB80_shaped','Trunk-basis, shaped','s--'),('CDMC128','Current CDMC 128','D-'),('CDMC64','Existing CDMC 64','^:')]:
                names=[f'{prefix}_K{k}' for k in (1,2,4)]
                ax.loglog([d[n]['total_ms'] for n in names],[d[n][key] for n in names],marker,label=label)
            title=('Field','Gradient','Flux')[j]
            ax.set(title=f'{family.capitalize()} / {title}',xlabel='Online time (ms)',ylabel='Relative error (%)')
            ax.set_xticks([7,10,15,20,25],labels=['7','10','15','20','25'])
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(True,which='both',alpha=.2)
            if i==0 and j==0:
                ax.legend(fontsize=8)
    fig.savefig(root/'accuracy_time.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
