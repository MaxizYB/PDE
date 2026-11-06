"""Build the HINTS comparison table and curves from completed evaluations."""
import argparse
import csv
import json
from pathlib import Path


def label(name):
    labels = {'FNO':'FNO', 'DeepONet':'DeepONet 单次预测',
              'dictionary_K1':'当前 CDMC K1', 'dictionary_K2':'当前 CDMC K2',
              'dictionary_K4':'当前 CDMC K4'}
    if name in labels:
        return labels[name]
    init, method, count = name.split('_')
    start = '零初值' if init == 'zero' else 'FNO 初值'
    return start + ' + ' + (f'HINTS {count[1:]} 块' if method == 'HINTS' else f'Jacobi {count[1:]} 步')


def main(root):
    lines = ['# HINTS 与当前 CDMC：同矩阵 Darcy 对比', '',
        '日期：2026-11-04。现有 129×129 P1-FEM Darcy，平滑介质与粗糙介质。', '',
        '## 设置与对齐', '',
        '- HINTS-Jacobi 每块：14 步阻尼 Jacobi（权重 0.8）+ 一次 DeepONet 残差校正，对应论文二维实验 1:14 比例。',
        '- 分别测试零初值与相同已训练 FNO 初值；Jacobi 对照每块执行 15 步，保持更新次数一致。',
        '- CDMC 使用已有 balanced128_train 检查点，本次不重新训练；所有方法在同一进程重新计时。',
        '- 同一细网格 P1 真值；场、梯度、能量和通量相对误差均以百分数报告。',
        '- 时间包含 FNO（若使用）、材料矩阵构建、传感器转换与全部更新。Batch=1，一次预热，3 次同步 wall-clock 的中位数，再取样本均值。几何计划及固定网格 DeepONet trunk 单独缓存。',
        '- 模型及交替流程依据官方代码接入；当前材料分布、P1 离散、变化 RHS 数据和归一化为本项目适配，并非论文原数据表的逐项复现。', '',
        'DeepONet branch：31×31 系数/源项双通道输入，4 层 stride-2 卷积与宽度 80 全连接层；trunk：宽度 80 的三层 tanh MLP，乘齐次 Dirichlet 边界因子。', '',
        r'FEM 残差是载荷向量，输入网络前转成源项 $f_r=M_{\mathrm{lump}}^{-1}(b-Au)$。训练标签来自当前细网格求解，65×65 节点仅用于 trunk 监督采样，不另求粗网格 PDE。', '',
        '## 测试结果', '']
    all_rows = []
    for family, title in (('smooth','平滑介质'), ('rough','粗糙介质')):
        run = root/family
        e = json.loads((run/'loaded_test32/evaluation.json').read_text())
        t = json.loads((run/'training.json').read_text())
        stats = e['summary']
        lines += [f'### {title}', '', f"未见测试样本 {e['count']} 个；GPU：{e['device']}。", '',
                  '| 方法 | 场误差 % | 梯度误差 % | 通量误差 % | 在线总时间 ms |',
                  '|---|---:|---:|---:|---:|']
        selected = ['FNO','DeepONet','zero_HINTS_B2','zero_HINTS_B8','zero_HINTS_B32',
                    'FNO_HINTS_B1','FNO_HINTS_B2','FNO_HINTS_B4','FNO_HINTS_B8',
                    'FNO_HINTS_B16','FNO_HINTS_B32','FNO_Jacobi_J30','FNO_Jacobi_J480',
                    'dictionary_K1','dictionary_K2','dictionary_K4']
        for name, info in stats.items():
            all_rows.append(dict(dataset=family,method=name,**{k:v['mean'] for k,v in info.items()}))
        for name in selected:
            values = [stats[name][k]['mean'] for k in ('L2_percent','gradient_percent','flux_L2_percent','total_ms')]
            lines.append('| '+label(name)+' | '+' | '.join(f'{v:.6g}' for v in values)+' |')
        cd, hi = stats['dictionary_K2'], stats['FNO_HINTS_B32']
        gain = [hi[k]['mean']/cd[k]['mean'] for k in ('L2_percent','gradient_percent','flux_L2_percent')]
        speed = hi['total_ms']['mean']/cd['total_ms']['mean']
        lines += ['', f'相对于 FNO + HINTS 32 块，CDMC K2 的场、梯度、通量误差分别低 {gain[0]:.2f}、{gain[1]:.2f}、{gain[2]:.2f} 倍，时间比（HINTS / CDMC）为 {speed:.2f}。', '',
            f"训练：{t['data']['splits']['train']['materials']} 个材料、{t['data']['splits']['train']['pairs']} 个变化 RHS 对；{t['epochs']} epochs；{t['training_seconds']:.1f} 秒。验证材料独立。", '',
            f"变化 RHS 验证集单次预测 L2：{100*t['best_validation_relative_L2']:.3f}%，不同于表中单位源项测试误差。", '',
            f"一次性成本：mesh plan {e['mesh_plan_ms_once']:.3f} ms；assembly plan {e['assembly_plan_ms_once']:.3f} ms；DeepONet trunk {e['trunk_cache_ms_once']:.3f} ms。", '',
            f"单位源项训练真值重算的最大相对差：{t['data']['splits']['train']['constant_solution_relative_error_max']:.3g}。", '']
    lines += ['## 解读与文件', '',
        'HINTS 的每块含 14 次数值平滑与一次神经调用；CDMC 的 cycle 具有不同结构，不能直接比较块数。零初值分支检验 HINTS 自身求解能力，FNO 初值分支检验同起点后校正性能。', '',
        '当前结论针对这两类材料与本次训练预算。HINTS 的 MG / Krylov 变体未纳入本次测试。', '',
        '- 有效结果：各数据集 loaded_test32/evaluation.json 与 summary.csv。JSON 保存逐样本、P95、残差、频段误差及阶段计时。',
        '- all_methods.csv：全部设置；accuracy_time.png：三类误差—时间曲线。',
        '- test32/ 为初次评测漏载权重的无效运行，不得引用；训练检查点不受影响。', '',
        '## 来源', '',
        '- Zhang et al., Blending Neural Operators and Relaxation Methods in PDE Numerical Solvers: https://arxiv.org/abs/2208.13273',
        '- 官方代码：https://github.com/kopanicakova/HINTS_precond；commit 与 SHA256 记录在 baselines/hints/vendor/PROVENANCE.json。', '']
    (root/'REPORT.md').write_text(chr(10).join(lines))
    keys = ['dataset','method','L2_percent','gradient_percent','energy_percent','flux_L2_percent','total_ms']
    with (root/'all_methods.csv').open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=keys,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)
    plot(root, all_rows)


def plot(root, rows):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.ticker import NullFormatter
    fig, axes = plt.subplots(2,3,figsize=(13,7.5),constrained_layout=True)
    for i, family in enumerate(('smooth','rough')):
        r = {row['method']:row for row in rows if row['dataset']==family}
        for j, (key, title) in enumerate((('L2_percent','Field'),('gradient_percent','Gradient'),('flux_L2_percent','Flux'))):
            ax = axes[i,j]
            for prefix, style, name in (('FNO_HINTS_B','o-','FNO + HINTS'),
                                       ('zero_HINTS_B','s--','HINTS, zero start'),
                                       ('FNO_Jacobi_J','^:','FNO + Jacobi'),
                                       ('dictionary_K','D-','FNO + CDMC')):
                names = sorted((n for n in r if n.startswith(prefix)),key=lambda n:int(n[len(prefix):]))
                ax.loglog([r[n]['total_ms'] for n in names],[r[n][key] for n in names],style,label=name)
            ax.scatter(r['FNO']['total_ms'],r['FNO'][key],marker='*',s=110,c='black',label='FNO')
            ax.set(title=f'{family.capitalize()} / {title}',xlabel='Online time (ms)',ylabel='Relative error (%)')
            ax.set_xticks([1,2,5,10,20,50], labels=['1','2','5','10','20','50'])
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(True,which='both',alpha=.2)
            if i==0 and j==0:
                ax.legend(fontsize=8)
    fig.savefig(root/'accuracy_time.png',dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
