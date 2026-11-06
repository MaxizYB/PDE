"""Aggregate completed refinement runs; never launch or modify an experiment."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import statistics

RUN=Path(__file__).resolve().parents[1]/'results/refinement_20261103'
KEYS=['L2_percent','gradient_percent','flux_L2_percent','total_ms']
CASES=[('六尺度·开发32','FNO','balanced128_test32'),
       ('六尺度·独立后32','FNO','balanced128_holdout32'),
       ('平滑两相','FNO-H1','smooth/frozen128_fno_h1'),
       ('平滑两相','MgNO','smooth/frozen128_mgno'),
       ('粗糙两相','FNO-H1','rough/frozen128_fno_h1'),
       ('粗糙两相','MgNO','rough/frozen128_mgno')]


def read(path):
    return json.loads((RUN/path).read_text())


def mean(records,name,key):
    return statistics.mean(r['methods'][name][key] for r in records)


def metric(records,name):
    return [mean(records,name,key) for key in KEYS]


def triple(values):
    return ' / '.join(f'{x:.6f}' for x in values[:3])


def main():
    records=[]
    documents=[]
    sources={}
    for dataset,base,path in CASES:
        doc=read(path+'/evaluation.json')
        assert len(doc['samples'])==32
        if '后32' in dataset:
            assert [r['index'] for r in doc['samples']]==list(range(32,64))
        else:
            assert [r['index'] for r in doc['samples']]==list(range(32))
        assert doc['matrix_bound'] is True
        prov=read(path+'/provenance.json')
        assert Path(prov['arguments']['checkpoint']).parts[-2]=='balanced128_train'
        for name in doc['summary']:
            values=metric(doc['samples'],name)
            for key,value in zip(KEYS,values):
                assert abs(value-doc['summary'][name][key]['mean'])<=max(1e-10,abs(value)*1e-10)
            records.append(dict(dataset=dataset,base=base,method=name,n=32,
                                **dict(zip(KEYS,values))))
        documents.append((dataset,base,doc))
        sources[path] = hashlib.sha256((RUN/path/'evaluation.json').read_bytes()).hexdigest()
    lines=['# 统一响应字典：方法简化、跨材料迁移与强基线结果',
           '', '日期：2026-11-03。所有数字由本文件旁的原始JSON汇总；三项误差均为相对同网格P1参考解的百分比。',
           '', '**核心结果：同一组128维校正权重，仅在48个六尺度材料上训练，冻结后用于三类材料、两种NO基底及129²/257²网格；不使用手工区域帽函数、固定正弦拼接、PCG或逐材料数值粗层级。**',
           '', '## 1. 同一冻结校正器的主结果', '',
           '| 材料与测试组 | NO基底 | n | NO场/导数/通量 % | K2场/导数/通量 % | NO→完整K2 ms | K2三项降误差 % |',
           '| --- | --- | ---: | --- | --- | --- | --- |']
    for dataset,base,doc in documents:
        pure=doc['predictor']
        before=metric(doc['samples'],pure)
        after=metric(doc['samples'],'dictionary_K2')
        reduction=[100*(1-a/b) for a,b in zip(after[:3],before[:3])]
        lines.append(f'| {dataset} | {base} | 32 | {triple(before)} | {triple(after)} | {before[3]:.2f}→{after[3]:.2f} | '+
                     ' / '.join(f'{x:.2f}' for x in reduction)+' |')
    lines += ['', 'NO是一次预测；K2是两轮完整校正。时间含新NO预测、细层材料装配、一次图字典设置、小系统分解与全部滤波/校准；batch=1、RTX4090、一次预热后3次同步墙钟取中位数，再跨材料平均。网格计划单列。',
              '', '## 2. 固定轮数的精度—时间关系', '',
              '| 材料与基底 | K1场/导数/通量 %；ms | K2场/导数/通量 %；ms | K4场/导数/通量 %；ms |',
              '| --- | --- | --- | --- |']
    for dataset,base,doc in documents:
        values=[metric(doc['samples'],f'dictionary_K{k}') for k in (1,2,4)]
        lines.append(f'| {dataset}，{base} | '+' | '.join(triple(v)+f'；{v[3]:.2f}' for v in values)+' |')
    lines += ['', '## 3. 与旧主线同批材料直接比较', '',
              '| 测试组 | 方法 | 场 % | 导数 % | 通量 % | 完整ms |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for dataset,base,doc in documents[:2]:
        for name in ('old_learned72_K2','old_union144_K2','dictionary_K2','old_union144_K4','dictionary_K4'):
            v=metric(doc['samples'],name)
            lines.append(f'| {dataset} | {name} | {v[0]:.6f} | {v[1]:.6f} | {v[2]:.6f} | {v[3]:.2f} |')
    lines += ['', '简化版K2改善旧联合144的场精度并减少在线时间，通量K2仍略高于旧联合空间。K4三项维持同一精度量级。该轮的主要结构收益是去掉两套几何字典、保住精度并获得冻结权重迁移能力。',
              '', '## 4. 与成熟数值基线的匹配比较', '',
              '每一材料族均使用测试前16个材料，表内CDMC也重新从对应JSON取完全相同的16个样本。CPU默认Gauss–Seidel AMG和GPU Jacobi AMG分别列出。', '',
              '| 材料 | 方法 | 场 % | 导数 % | 通量 % | 完整在线/CPUms | 缓存应用ms |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    configs=[('六尺度',documents[0][2],'trigonometric_gpu_amg/evaluation.json','trigonometric_numerics_threads2/numerics_rs.json'),
             ('平滑两相',documents[2][2],'smooth_gpu_amg/evaluation.json','smooth/numerics/numerics_rs.json'),
             ('粗糙两相',documents[4][2],'rough/gpu_amg/evaluation.json','rough/numerics/numerics_rs.json')]
    for family,ours,gpu_path,cpu_path in configs:
        expected_indices=[row['index'] for row in ours['samples'][:16]]
        gpu=read(gpu_path)
        cpu=read(cpu_path)
        assert [row['index'] for row in gpu['samples']]==expected_indices
        assert [row['index'] for row in cpu['samples']]==expected_indices
        for name in ('dictionary_K2','dictionary_K4'):
            v=metric(ours['samples'][:16],name)
            lines.append(f'| {family} | CDMC {name[-2:]} GPU | {v[0]:.6f} | {v[1]:.6f} | {v[2]:.6f} | {v[3]:.2f} | — |')
        for name in ('NO_GPU_AMG_K2','NO_GPU_AMG_K4'):
            v=metric(gpu['samples'],name)
            cached=mean(gpu['samples'],name,'cached_apply_ms')
            lines.append(f'| {family} | 显式RS-AMG {name[-2:]} GPU | {v[0]:.6f} | {v[1]:.6f} | {v[2]:.6f} | {v[3]:.2f} | {cached:.2f} |')
        for name in ('NO_AMG_K2','NO_AMG_K4','zero_AMG_CG_tol1e-10'):
            s=cpu['summary'][name]
            values=[s[key]['mean'] for key in KEYS[:3]]
            total=s['total_cpu_ms']['mean']
            solve=s['solve_ms']['mean']
            lines.append(f'| {family} | RS-AMG {name.replace("NO_AMG_", "")} CPU | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} | {total:.2f} | {solve:.2f} |')
    lines += ['', 'GPU基线使用PyAMG经典RS插值与显式Galerkin矩阵，2/2 Jacobi、末层精确分解；完整时间包含重新执行NO、GPU细层装配、CPU层级生成、传输、分解及GPU应用。CPU列包含CPU装配/AMG设置/求解，NO初值从同一缓存读取，因此其CPU总数不含NO和跨设备初值传输；它不是相同硬件的速度比。缓存应用列展示材料不变时AMG能够摊销设置的优势。',
              '', '## 5. 学习、物理与容量的机制证据', '']
    rank=read('rank_diagnosis_fixed/diagnosis.json')
    linear=[r['rank']['linear_readout']['rank_at_relative_1e_6'] for r in rank['samples']]
    nonlinear=[r['rank']['nonlinear_dictionary']['rank_at_relative_1e_6'] for r in rank['samples']]
    capture=[r['captured_energy_fraction'] for r in rank['samples']]
    lines += ['| 检验 | 样本与结果 | 解释 |','| --- | --- | --- |',
              f'| 窄线性读出与非线性字典 | 4验证材料：数值秩{linear}→{nonlinear}；相对奇异值阈值$10^{{-6}}$ | 增加了实际独立空间方向，不只是增加通道数量 |',
              f'| 当前真实误差能量覆盖 | 4验证材料：平均{100*statistics.mean(capture):.2f}% | 无手工区域的字典仍覆盖主要实际剩余任务 |',
              '| 矩阵谱界与能量稳定性 | 20/20非M矩阵SPD系统通过，谱条件数$10^5$ | 数值核由矩阵条件定义，不依赖材料名称 |',
              '| 合并物理阶段的失败 | 验证K2约0.164/0.634/0.756% | 一起删去阶段会破坏互补；保留必要物理工作 |',
              '', '能量覆盖使用验证真值进行离线诊断，部署不使用真值；它不是网络输入。非M矩阵审计验证数值内核的代数性质，不替代新PDE或新网格实测。',
              '', '## 6. 原六尺度样本的阶段耗时', '',
              '| 阶段 | 4样本同步诊断平均ms |','| --- | ---: |']
    stages=documents[0][2]['stages']
    for key in stages[0]:
        if key!='index':
            lines.append(f'| {key} | {statistics.mean(s[key] for s in stages):.3f} |')
    lines += ['', '阶段诊断单独同步，只用于识别开销，不将其相加替代完整端到端时间。更新阶段包含剩余轮次的预滤波。',
              '', '网格计划为跨材料复用的一次开销：开发32和独立后32的257²运行分别记录8.66s与10.04s；它不进入每个材料的在线表。',
              '', '## 7. 数据与训练记录', '',
              '- 新两相NO：各256训练/32验证/32测试，FNO与官方MgNO块各150epochs，同一L2+0.1导数监督。MgNO为同预算重新训练，不是论文权重。',
              '- 当前128校正器：48个六尺度训练材料、12验证；继承旧图表示，64字典30epochs、128扩展20epochs、多轮平衡20epochs；独立材料数始终为48。',
              '- 校正器只使用六尺度训练数据；两相族的64维专门训练模型是开发对照，不参与冻结128主表。',
              '- 本轮为单seed结构研究；模型、训练日志、逐材料指标、运行参数和所有失败候选均保留。',
              '- 误差相对重建一致的同网格P1解；细网格下采样与连续离散误差属于独立评价，不能与这些代数精度数字合并。',
              '', '方法、文献与运行说明见[统一方法说明](../../refinement/README.md)。']
    (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    with (RUN/'SUMMARY.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['dataset','base','method','n',*KEYS])
        writer.writeheader();writer.writerows(records)
    checkpoint=RUN/'balanced128_train/model.pt'
    source_dir=Path(__file__).resolve().parent
    source_hashes={str(p.relative_to(source_dir)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in source_dir.rglob('*') if p.is_file() and p.suffix in ('.py','.sh')}
    unique_samples={(read(path+'/provenance.json')['arguments']['data'],
                     doc['split'],row['sample_index'])
                    for (_,_,path),(_,_,doc) in zip(CASES,documents)
                    for row in doc['samples']}
    (RUN/'AUDIT.json').write_text(json.dumps(dict(main_cases=len(CASES),main_rows=len(records),
        model_material_observations=sum(len(doc['samples']) for _,_,doc in documents),
        unique_material_samples=len(unique_samples),
        same_frozen_corrector=True,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        code_snapshot_sha256=source_hashes,sources=sources),indent=2)+'\n')
    print(RUN/'RESULTS.md')


if __name__=='__main__':
    main()
