"""Consolidate all declared frame experiments, including unsuccessful pilots."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

KEYS=("L2_percent","gradient_percent","flux_L2_percent","total_ms")


def main(root):
    lines=["# TB 对照后的空间改进：方法、训练与最终测试", "",
        "日期：2026-11-04。两组129×129两相 Darcy，每组32个相同未见测试样本。保留原检查点，单seed快速迭代。", "",
        "## 方法变化", "",
        "先用FP64无正则Galerkin投影修复坐标正则损失；再以同一TB80参考空间为起点，由多级图网络生成空间外方向，构造80维有界适应空间。不是将80列和8列简单拼接，也不增加PCG迭代。数学、结构与数据协议见 refinement/FRAME_SPACE_METHOD.md。", "",
        "新方法继承TB已有预训练并额外进行NO缺陷训练。结果不能直接解释为同训练预算的架构胜出。TB是本项目共同后校正流程中的空间构造对照，不是原论文完整的外层Krylov求解器。", "",
        "## 计时与指标", "",
        "误差均为同一细网格P1真值上的相对误差百分数。在线总时间包括FNO、每材料细网格组装、空间生成/准备、小系统分解与全部校正。固定几何计划、参考空间QR和网格缓存不在在线总时间中，单列记录。Batch1，预热1次、同步墙钟测3次取中位数，再对32样本求均值。", ""]
    rows=[]
    for family,title in (("smooth","平滑介质"),("rough","粗糙介质")):
        lines += [f"## {title}", ""]
        for run,label in (("normal_full","64步预滤波：空间变化"),("normal_balanced","32步预滤波：空间与滤波配合")):
            path=root/family/run/"test32/evaluation.json"
            serial=root/family/run/"test32_serial/evaluation.json"
            if serial.exists():
                path=serial
            if not path.exists():
                lines += [f"### {label}","", "正式结果尚未生成。", ""]
                continue
            data=json.loads(path.read_text())
            if data["count"]!=32 or len(data["samples"])!=32:
                raise ValueError(f"Incomplete formal result: {path}")
            s=data["summary"]
            lines += [f"### {label}", "",f"检查点由验证集选择：epoch {data['selected_epoch']}。实际新方法预滤波 {data['options']['pre']} 步，下限参数 {data['options']['lower']}。结果来源：{path.relative_to(root)}。", "",
                "| 方法（均为K2） | 场误差 % | 梯度误差 % | 通量误差 % | 在线时间 ms |",
                "|---|---:|---:|---:|---:|"]
            labels={"CDMC_old":"原CDMC128/64步","CDMC_exact":"原CDMC128稳定投影/64步",
                    "TB_old":"原TB80/64步","TB_exact":"TB80稳定投影/64步",
                    "TB_same":"TB80稳定投影/32步配合频段","frame":"新空间"}
            for name,v in s.items():
                row=dict(dataset=family,run=run,method=name,**{k:v[k]["mean"] for k in KEYS})
                rows.append(row)
                if name.endswith("_K2"):
                    lines.append("| "+labels.get(name[:-3],name)+" | "+" | ".join(f"{row[k]:.7g}" for k in KEYS)+" |")
            f=s["frame_K2"]
            for base in ("TB_old_K2","TB_exact_K2","TB_same_K2","CDMC_old_K2"):
                if base not in s:
                    continue
                changes=[100*(1-f[k]["mean"]/s[base][k]["mean"]) for k in KEYS]
                wins=[sum(v["methods"]["frame_K2"][k]<v["methods"][base][k] for v in data["samples"]) for k in KEYS[:3]]
                lines += ["",f"相对 {base}：场/梯度/通量误差降低 {changes[0]:.1f}% / {changes[1]:.1f}% / {changes[2]:.1f}%；时间减少 {changes[3]:.1f}%（负值为增加）。三类误差逐样本胜出 {wins[0]}/32、{wins[1]}/32、{wins[2]}/32。"]
            lines += ["", "### 同一次运行的分阶段时间", "",
                "前4例单独同步剖析的均值；每阶段额外同步，不能直接代替整体多次测量时间。", "",
                "| 方法K2 | FNO | 组装 | 预滤波 | 空间生成 | 空间分解 | 校正及后滤波 |",
                "|---|---:|---:|---:|---:|---:|---:|"]
            for name in s:
                if not name.endswith("_K2"):
                    continue
                stages=[v for v in data["stages"] if v["method"]==name]
                if stages:
                    keys=("FNO_ms","fine_assembly_ms","prefilter_ms","space_generation_ms","space_factor_ms","correction_ms")
                    lines.append("| "+name+" | "+" | ".join(f"{np.mean([v[k] for v in stages]):.3f}" for k in keys)+" |")
            lines += ["", f"一次性计划：网格 {data['mesh_plan_ms_once']:.1f} ms，装配 {data['assembly_plan_ms_once']:.1f} ms，trunk读取/QR/参考缓存 {data['cache_ms_once']:.1f} ms。GPU：{data['device']}。", ""]
    conclusion=["## 本轮结论", "",
        "新空间明显改善了原CDMC，也在平滑介质上超过TB的K2精度；粗糙介质上与TB近乎持平，动态网络成本尚未转化为稳定收益。没有实现两类材料同时比TB更准、更快。", "",
        "32步联合版本能比原TB64步更快，但K2精度仍下降；同为32步时，新空间比TB更准，却仍需约3ms空间生成开销。不能将公共滤波改变的全部收益归因于学习。", "",
        "平滑介质首次并行测试的全方法计时一起偏慢，保留在test32中；最终报告采用空闲GPU单独复测的test32_serial。未重新训练或根据测试误差更换检查点。其他测试原样保留。", ""]
    lines[4:4]=conclusion
    lines += ["## 空间结构核验", "",
        "以下为独立验证集4例，无测试真值输入。能量Gram条件数为未做列能量缩放的当前矩阵Gram；不能将不同归一化协议下的条件数直接当作严格倍率比较。", "",
        "| 材料 | 新空间欧氏Gram条件数 | 新空间能量Gram条件数 | 参考空间能量Gram条件数 | 相对空间变化 | 变化秩 |",
        "|---|---:|---:|---:|---:|---:|"]
    for family in ("smooth","rough"):
        path=root/family/"normal_full/audit_val4/space_audit.json"
        if path.exists():
            a=json.loads(path.read_text())["summary"]
            values=[a[k]["mean"] for k in ("euclidean_gram_condition","learned_energy_gram_condition","reference_energy_gram_condition")]
            lines.append("| "+family+" | "+" | ".join(f"{v:.3f}" for v in values)+f" | {100*a['relative_frame_change']['mean']:.3f}% | {a['update_rank']['mean']:.0f} |")
    lines += ["", "空间外更新实际达到8秩，并非再次生成8列近乎相同的方向。粗糙介质最佳检查点仅进行很小的空间变化，而平滑介质变化明显。结合稳定的小系统与泛化结果，本轮剩余瓶颈更接近空间适应的可学习性和泛化，而不是原先的列病态。", "",
        "等维旋转还有一个结构限制：80维空间发生真正变化后，不能同时完整包含原80维参考空间。后续应优先检验保留参考空间、仅对未解释缺陷进行小秩物理正交补充的价值，而非再次从头生成更多高度相关模式。该建议尚未实现，不能记为本轮成果。", ""]
    lines += ["## 迭代过程记录与验证结果", "",
        "以下均为验证集，不能与32例测试均值直接混用。每个阶段保留最佳验证检查点，失败原型不删除。", "",
        "| 材料 | 阶段 | 训练/验证材料 | 最佳epoch | K2场/梯度/通量 % | 阶段耗时 s |",
        "|---|---|---:|---:|---|---:|"]
    for family in ("smooth","rough"):
        for run in ("frame_train","normal_train","static_train","normal_full","normal_fast","normal_balanced"):
            path=root/family/run/"training.json"
            if not path.exists():
                continue
            t=json.loads(path.read_text())
            best=min(t["history"],key=lambda x:x["score"])
            vals=" / ".join(f"{v:.7g}" for v in best["val_percent"][-1])
            lines.append(f"| {family} | {run} | {t['train_count']} / {t['val_count']} | {best['epoch']} | {vals} | {t['seconds']:.1f} |")
    lines += ["", "阶段含义：frame_train=有界局部图输运；normal_train=48材料空间外方向模型；static_train=只依赖几何的固定空间对照；normal_full=256材料继续训练、64步滤波；normal_fast=仅把原滤波阶数减半；normal_balanced=按Chebyshev相同衰减界将32步频带与学习空间配合、从参考空间重新训练。", "",
        "直接减半原滤波的版本未达到原TB64步精度，因此不作为主结果。固定空间对照和局部输运原型也未获得充分的验证收益。", "",
        "## 复现文件", "",
        "- 每个正式目录的 evaluation.json：逐样本、P95、阶段计时、硬件、滤波配置与检查点路径。", 
        "- all_methods.csv：所有正式K1/K2/K4指标；accuracy_time.png：三类误差—在线时间曲线。",
        "- refinement/frame_space.py：训练与评测；stable_space.py：数值稳定诊断；test_frame_algebra.py：无需PyTorch的代数检查。", ""]
    (root/"REPORT.md").write_text(chr(10).join(lines))
    with (root/"all_methods.csv").open("w") as stream:
        w=csv.DictWriter(stream,fieldnames=["dataset","run","method",*KEYS])
        w.writeheader()
        w.writerows(rows)
    if rows:
        plot(root,rows)


def plot(root,rows):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
    for i,family in enumerate(("smooth","rough")):
        configs=[("normal_full","TB_old","Original TB64","o--"),
                 ("normal_full","CDMC_old","Original CDMC128","D:"),
                 ("normal_full","frame","New frame / filter64","s-"),
                 ("normal_balanced","TB_same","TB / matched filter32","^--"),
                 ("normal_balanced","frame","New frame / filter32","v-")]
        for j,key in enumerate(KEYS[:3]):
            ax=axes[i,j]
            for run,prefix,label,style in configs:
                points=[v for v in rows if v["dataset"]==family and v["run"]==run and v["method"].startswith(prefix+"_K")]
                points.sort(key=lambda v:int(v["method"].rsplit("K",1)[1]))
                if points:
                    ax.loglog([v["total_ms"] for v in points],[v[key] for v in points],style,label=label)
            ax.set(title=family+" / "+("Field","Gradient","Flux")[j],xlabel="Online time (ms)",ylabel="Relative error (%)")
            ax.grid(True,which="both",alpha=.2)
            if i==0 and j==0:
                ax.legend(fontsize=8)
    fig.savefig(root/"accuracy_time.png",dpi=170)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    main(p.parse_args().root)
