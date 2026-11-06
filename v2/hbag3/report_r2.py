"""v2.3 reports: generated numeric tables, code-linked implementation and disk budget."""
from pathlib import Path
from dataclasses import asdict
import datetime,hashlib,json,math,shutil
import numpy as np
from .environment import ROOT
from .model import Config
from .plan_r2 import VERSION,SHA256
from .network import Shell
from .runner import write_json


def reference(file,token):
    lines=(ROOT/'hbag3'/file).read_text().splitlines()
    line=next(i+1 for i,x in enumerate(lines) if token in x)
    return '[%s:%d](hbag3/%s#L%d)'%(file,line,file,line)


def documentation():
    c=Config();dt=Shell(24,24).period_s/c.slots_per_orbit
    params=['# R2 v2.3 参数与数据边界','',
      '方案SHA256：`e40db49e1fb59dc8d6c2b238e07dbfa7b4e3f61379677d32512d80035f34818a`。所有新结果来自验证/正式运行原始记录；旧v1/v2结果均归档。R1的PARAMS.md保持不动。','',
      '| 参数 | 数值或定义 | 依据 |','|---|---|---|',
      '| 壳层 | 24×24，550 km，53°，全RAAN | 方案合成星座 |',
      '| 时隙 | %.9f s，96隙/轨 | R1既定时间网格 |'%dt,
      '| 流数 | 288 | v2.1取消576流备选 |',
      '| 候选集 | 3最少跳+1不同日照优先，K=4 | 共同星历；同跳数按距离 |',
      '| λ* | t=0，光学满容量，公共K=4，无能量约束 | 方法/RF/路由变体共用；旧距离K=3仅作诊断 |',
      '| load | 0.65；唯一条件备选0.5 | G0′只因交付率不足才用备选 |',
      '| MMPP | 每源城市独立；1200s平静×0.4/300s突发×3.4；Gamma(20,1/20) | 同城OD共享状态、各OD独立抖动；合成假设 |',
      '| 终端 | ON 25W+q_inc；SLEEP 2W；最大60W | 方案假设 |',
      '| 捕获 | 默认20s；容量乘max(1−tau_acq/dt,0) | ON功耗仍按整隙 |',
      '| 能量 | B=55W，H=500W，eta_c=eta_d=0.95，SOC0=0.95 | 方案假设 |',
      '| 安全 | DoDmax=0.30，Cmin=0.70Emax，Ecut=0.10Emax | Cmin处不裁剪 |',
      '| 储备 | fallback=63W；评估窗外再延伸2轨 | 不可行值保留 |',
      '| 标定 | V0/100MJ/1轨/10验证种子 | 完整日食窗口平均电池取能 |',
      '| 电池 | Emax=mean_eclipse_draw/(0.30rho_E) | rho=0.6/1.0/1.2/1.4/1.6 |',
      '| 主对比/P1 | 主1.2/1.4/1.6；P1默认1.4，E7为1.0 | 方案v2.1 |',
      '| 内层 | alpha=0.01，rho=0.02，200迭代，证书间隙1e-3 | 所有方法统一 |',
      '| 开关 | c=(pi_m+pi_n)23/100；保持2隙；低值连续2隙；A2每星≤1条/隙 | v2.3抗突发反事实值与路径唤醒值 |',
      '| 选参 | HBAG每档24点，10验证种子，2轨；三档各冻一套 | 共720个新配置×种子；未改方法沿用v2.2冻结参数，标明来源 |',
      '| Util-Sleep阈值 | 0.01/0.05/0.10 | 与其他方法同一选参准则 |',
      '| 种子 | 验证1000–1009；测试4000–4019 | G0′和E11′均通过才解锁 |',
      '| 附录A | D=max(0.95×500,(55+4×60)/0.95)=475W | 预算补偿tau D，价格用未补偿Ê |',
      '', '精确光学/RF函数沿用Terminal：1550nm、0.08m口径、10μrad半角、1000photons/bit、壁插效率0.2、额外2e-9J/bit、10Gbit/s为代表性假设，不声称硬件实测。城市人口源与合成渗透率/航空权重保持R1定义。β只指参考轨道面，逐面真实值保存。',
      '', '内部float64；原始浮点数组float32；安全阈值容差1e-6J在序列化前计算。求解器rate/1e9、power/100；落盘物理量J/W/s/bit/s。完整逐方法网格由hbag3.gate.validation_grid生成，冻结后保存tables_r2/E11_validation_grid.*。']
    (ROOT/'PARAMS_R2.md').write_text('\n'.join(params)+'\n')
    rows=[
      ('解锁',reference('runner.py','def seed_guard'),'验证限1000–1009；正式限4000–4019，且G0_prime/gate与E11_validation/frozen同源v2.3、均PASS。'),
      ('几何与公共候选',reference('environment.py','def catalogue'),'每隙+Grid/LOS/range；3最少跳Yen以距离打破并列，日照权重1+1.5×日食端点数，取首条不重复路径。'),
      ('参考容量',reference('environment.py','# v2.1 common reference'),'同t=0几何/OD形状，对光学公共K=4解最大并发LP；另计算R1距离K=3仅用于报告比值。'),
      ('MMPP',reference('environment.py','def city_mmpp_states'),'每源城市独立RNG与两状态链，同城各OD共享状态；Gamma按OD；全时段总需求均值归一化保持。'),
      ('储备',reference('energy.py','def reserve_envelope'),'S[t]=max(Cmin,S[t+1]−dt drift(Hlower,B+8W))；超评估窗2轨；不删除S>Emax。'),
      ('价格准备',reference('pricing.py','def prepare'),'w=alpha+pi；dual窗口入口取zero或prev；Z=max(E−S,0)，日食Ppace=8+eta_d Z×w当前/(dt sum w)；预测为当前需求乘未来公共K4几何路径的节点出现比例（含源宿、只计非空路径），w=潜在负载+0.1×窗口均值。权重和≤0或非有限时回退8+eta Z/(n dt)；日食P*=clip(eta_d Z/(n dt),0,232W)闭式；日照仅条件(a)不成立时按完整扩展储备曲线做14次二分。P_full=4(P_on−P_sleep)+vhat；vhat初值8W，仅Q≥4P_on时以0.2更新。条件(a)按P_full运行到日食仍能充满，或(b)P*≥P_full时mu=0且本隙不更新。其余日照Ppace=8+P*。'),
      ('P*储备可持续功率',reference('reserve_pricing_r23.py','def sustainable_sun_power'),'反推S^P，从窗口终点原储备开始；检查窗口内每个S^P≤Emax且初态≥S^P[t]；上限可行取上限、P=0不可行取0。条件(a)已成立时P*不求解，raw中记NaN并置computed=false；(b)只在(a)未成立处统计，分母另存。'),
      ('硬预算/A0',reference('control.py','raw_budget = load_budget'),'Qbar=load_budget(E,Snext,Hlower,dt)−B；无邻居或E≤Ecut休眠，任一端休眠则双端休眠。'),
      ('A1',reference('control.py','# A1:'),'Qbar<8W或Snext>Emax全休眠；按上一执行隙cf值升序关闭，直至固定终端功率可行。'),
      ('反事实值',reference('link_values.py','def counterfactual_values'),'g_p=sum w_tx slope(R) RATE/100；s_p=min剩余链路容量/RATE；每已用路径及其边，找不经该边、可用、余量最大（并列价最低）的备选；近期峰值dhat=max(d,exp(−dt/Torb) dhat_prev)，phi=dhat/d（d=0取1），r=min(phi x,s)；cf+=r(g_alt−g)^++(phi x−r)u′。sup仍用实际x。所有使用A1/A2 cf的方法无条件生效；V0/Guard/Util旧诊断不变。'),
      ('A2',reference('control.py','cost=(pi[a]+pi[z])'),'c只计pi；cf<kappa_off c连续2隙且ON≥2隙；按cf/c排序；任一端本隙已A2则跳过；pi双零不能触发。'),
      ('唤醒值',reference('link_values.py','def path_wake_values'),'按u′降序处理未满足需求；只考虑所有休眠边均满足保持时间、未被A0阻断且非本隙刚睡的路径。选休眠边最少者，并列取剩余方向捕获容量最大者；每商品只选一条路径，扣除各休眠方向容量，效用平均分给该路径休眠边。'),
      ('A3',reference('control.py','# A3:'),'睡眠≥2且非本隙刚睡；c≤kappa_off/2×max(last_cf,wake)或双端pi=0；硬方法另需两端余量≥28W；V1仅预算。'),
      ('A4',reference('control.py','# A4:'),'正需求无可用拓扑路径，沿最短候选救援；按商品编号、保持时间、整条路径原子预算检查。'),
      ('Guard/Util',reference('control.py',"self.method.startswith('Guard')"),'Guard以theta/theta+.05迟滞；Util双方向最大util连续2隙低于所选theta_u则睡眠，睡满5隙试探。'),
      ('捕获与K+1',reference('control.py','def usable_catalogue'),'ON需双端握手；新可用边全隙ON功耗、容量扣tau_acq；K路径全禁用时在可用图补Dijkstra。'),
      ('内层',reference('allocation.py','def problem_for_switch'),'log端到端效用−w增量功率−rho||x||²/2；固定功耗不在目标；容量含捕获；硬增量预算Qbar−fixed，其余140W硬件上限。'),
      ('CP/影子',reference('allocation.py','def allocate'),'CP对角预条件200迭代或相对证书≤1e-3，比例恢复；影子仅E2/rho1.4/HBAG；全部其他运行（包括E11/P1/P2）不做影子；central独立闭环。'),
      ('Clarabel等价缩放',reference('central_reference_r2.py','def solve'),'仅消去不可用、零需求或精确零预算强制为零的变量；由需求与资源约束取每路径上界u_j，令x_j=u_j z_j，正预算/需求行归一化为≤1。恢复原单位对偶，按原问题计算证书；保留全部小正预算。影子与central容差1e-9，E9参考1e-10。固定Clarabel数值尝试：步长0.90、0.95，以及静态正则1e-10下步长0.90、0.99；各次保持相同模型/容差。原问题证书达请求精度即停止，否则保留最小证书间隙解及实际精度；全部重试失败保存精确问题：影子标失败、该隙目标差为null并继续；central从有限失败迭代中选证书间隙最小者做可行恢复并标降级。若全部尝试均未返回有限迭代，显式降级为可行零速率；不另换求解器。逐隙保存状态、证书间隙、尝试次数和达标标记。'),
      ('物理保护',reference('allocation.py','def protect_bus'),'各方法相同：按到0J可供功率削减增量，递推仅裁0/Emax，保存spill/unserved。'),
      ('执行后更新',reference('control.py','def observe'),'用实际x及旧w/cap更新cf/sup与last_cf；休眠时不以0覆盖last；实际Q更新mu=max(mu+eta_mu(Q−Ppace)/100,0)。'),
      ('输出',reference('storage.py','def storage_tier'),'selection只summary+slots；正式4000–4004详细，其余仅节点/商品数组；path_nodes在公共inputs/paths注册表一份，方法存ID。'),
      ('E10在线计时',reference('timing_r2.py','def scaling_cell'),'所有快照及价格窗口所需的公共候选路径预先构建；路径总耗时/构建时隙数及每星每隙值单列。冷启动decide、可用路径、allocate、protect_bus逐项计时；当前需求的预测矩阵乘法仍在线。256–1584星20个新种子×4快照；5000星4000–4004×{24,48}；单线程，E10并行≤物理核一半。'),
      ('P2离线参考',reference('optional_r2.py','def fractional_mpc'),'共享分数开关及线性捕获松弛模型不变；t=0对H=96全轨只求解一次，逐隙执行同一计划并保存能量账本与solver_calls=1。与同cell HBAG配对，仅乐观参考，不称形式化上界。'),
      ('G0′/E11′',reference('gate.py','def execute'),'v2.3先核验验证种子1000上V0、V1、Lyapunov、HBAG-xi、HBAG-Lyap、Guard、Util-Sleep全部旧原始数组和科学指标逐比特一致；复用已归档哈希证明的v2.1标定与紧张度及E_max；用当前代码重跑60个硬方法功能检验，附宽裕与H275/beta60六轨验证诊断；只重选HBAG，其他方法参数复用v2.2；宽裕验证不参与选择，双PASS才正式运行。')]
    impl=['# R2 v2.3 实现与逐步骤代码','', '| 时序 | 代码 | 伪代码与约束 |','|---|---|---|']
    impl+=['|%s|%s|%s|'%r for r in rows]
    impl+=['','## 指标','',
      'Delivered=Σx RATE dt，FVR=1−Delivered/Offered，EE分别除ISL/系统真实供电能耗。ESR为隙末E≥Cmin−1e-6J比例。每轨DoD/越界深度和时长、预算强制休眠、brownout、ON比例、事件/翻转与捕获归因能耗均保存。捕获耗能已包含于ON固定功耗，不重复加入账本。',
      '', '主ON利用率=所有ON有向链路×隙的rate/捕获后容量均值（0容量隙记0）；另存额定容量利用率。日食终端分母含四个安装端口，无几何邻居记0。完整日食窗口两端必须在观察区间内且有阳照包围，边界截断窗口不用于电池标定。',
      '', '平均跳数=Σpath_bits×hops/Delivered；传播时延ms=Σpath_bits×path_distance_km/(Delivered×299792.458)×1000。断连按K和兜底均不可用的正需求商品×隙，计数量与需求占比；不混同容量饱和导致未服务。',
      '', '空洞：全端口SLEEP且源接入需求>0，分母只计源端。中继bit·跳仅路径内部节点，分日照/日食。日食进度使用前一轨星历确定起点前已过隙数，三等份和六等份统计ON端口数。E2另按星×日食窗口，在窗口首个可见时隙冻结当前需求预测，沿未来公共K4路径生成潜在负载序列，与实际执行路径经过该星的bit（含源宿各一次）计算跨时隙Pearson相关；截断窗口标记，恒定/短序列记未定义，不填0。逐窗口记录potential_load_windows.json（正式归档为逐比特无损.json.xz，可用compact_r2.read_json_archive读取），预测与实际序列保存在raw。溢出按spill/eta_c除总采能；另存储能侧spill。一阶方法报告nu>0；HBAG-central只报告nu>1e-6并标明阈值。另存P*、P_full、(a)/(b)标志及按日照/日食分组的归零比例；逐轨保存日照终端开启比例与溢出采能占比。E8 ESR展示至0.001%，不足100%但舍入会到100.000时标<100.000%。cf/sup分发送端日照/日食保存分位数；详细种子另存逐边。',
      '', '## 附录A（E8′）','',
      '原HBAG：t测量，t+tau执行过期动作，1s子步积分。HBAG+margin：t−tau测Ê，t准时执行，窗口与储备网格对齐；预算用Ê−tau D，价格用Ê。D=max(eta_c Hsun,(B+4P_el_max)/eta_d)，默认475W。分别报告两者安全、交付和实测max|E(t)−E(t−tau)|与tau D。G0′阶段不声称完成时延实验。',
      '', '## 未改变的证据边界','',
      '反事实重路由是方案规定的逐路径启发式；各替代流独立计算余量，不是额外求解的全局反事实最优。用每节点A2≤1约束逐步稀疏。源城市热点结构保留。G0′及E11′未完成前禁止正式结果；v2.1选参188个完成运行及v2.2全部产物已归档，不并入v2.3；标定与紧张度按方案明确复用。']
    (ROOT/'IMPLEMENTATION_R2.md').write_text('\n'.join(impl)+'\n')


def audit_r1():
    baseline=json.loads((ROOT/'results_r2/audit/r1_files_before.json').read_text())
    prefix=json.loads((ROOT/'results_r2/audit/progress_prefix.json').read_text());mismatch=[];concurrent=[];hashes=0
    for name,old in baseline.items():
        if name=='PROGRESS.md':continue
        p=ROOT/name
        if name=='EXPERIMENT_PLAN_R2.md' or name.startswith('manuscript_r2/'):
            if p.exists() and p.stat().st_mtime_ns!=old['mtime_ns']:concurrent.append(name)
            continue
        if not p.exists():mismatch.append(dict(path=name,reason='missing'));continue
        st=p.stat()
        if (st.st_size,st.st_mtime_ns,st.st_ctime_ns)!=(old['size'],old['mtime_ns'],old['ctime_ns']):mismatch.append(dict(path=name,reason='metadata_changed'))
        if 'sha256' in old:
            hashes+=1
            if hashlib.sha256(p.read_bytes()).hexdigest()!=old['sha256']:mismatch.append(dict(path=name,reason='sha256_changed'))
    prefix_ok=hashlib.sha256((ROOT/'PROGRESS.md').read_bytes()[:prefix['bytes']]).hexdigest()==prefix['sha256']
    record=dict(time=datetime.datetime.now().astimezone().isoformat(),baseline_files=len(baseline),source_document_hashes_checked=hashes,
        progress_original_prefix_unchanged=prefix_ok,mismatches=mismatch,concurrent_r2_edits=concurrent,passed=not mismatch and prefix_ok)
    write_json(ROOT/'results_r2/audit/r1_integrity.json',record);return record


def runtime_report():
    timings=[];groot=ROOT/'results_r2/G0_prime'
    for p in sorted(groot.glob('timing_load_*.json'),reverse=True):
        d=json.loads(p.read_text());row=d['pilot'];load=row['config']['load'];workers=d['workers']
        stem=('%.2g'%load).replace('.','p')
        complete=len(list((groot/('calibration_load_'+stem)).glob('seed_*/V0/summary.json')))
        timings.append(dict(load=load,run_s=row['runtime_s'],environment_s=row['environment_build_s'],workers=workers,
                            bytes=d['output_bytes'],completed=complete,remaining_s=math.ceil((10-complete)/workers)*row['runtime_s']))
    lines=['# R2 v2.3 时间与存储预算','', '只使用验证种子实测；并行≤10，空闲<25GB暂停；不写环境磁盘缓存。','',
      '| load | 首种子墙钟 s | 环境 s | 文件 MB | 完成/10 | workers | 同配置剩余粗估 s |',
      '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in timings:lines.append('|%g|%.3f|%.3f|%.3f|%d|%d|%.1f|'%(r['load'],r['run_s'],r['environment_s'],r['bytes']/1e6,r['completed'],r['workers'],r['remaining_s']))
    lines+=['','按已完成的验证实测持续更新；全部规模的验证测时完成前不给全套结束日期。E11′需720个HBAG两轨配置×种子，另有宽裕诊断；每种子环境一次构建后在内存复用。正式阶段开跑前，必须用验证样本的full/node/selection实测大小投影整个矩阵，≤10GB目标写入本文件；仅凭完整G0样本不能声称整个矩阵已满足10GB。']
    gate=groot/'gate.json'
    if gate.exists() and json.loads(gate.read_text())['state']=='STOP':lines+=['','G0′已STOP；无后续正式实验排队，不能给出全套完成倒计时。']
    projection=ROOT/'results_r2/audit/storage_projection.json'
    if projection.exists():
        record=json.loads(projection.read_text());record.pop('measured_samples',None)
        lines+=['','实测存储投影（逐样本原始测量见results_r2/audit/storage_projection.json）：','', '```json',json.dumps(record,indent=2,ensure_ascii=False),'```']
    (ROOT/'RUNTIME_ESTIMATE_R2.md').write_text('\n'.join(lines)+'\n')
    write_json(groot/'runtime_estimate.json',dict(timings=timings,time=datetime.datetime.now().astimezone().isoformat()))


def results_report():
    groot=ROOT/'results_r2/G0_prime';gate=groot/'gate.json'
    state=json.loads(gate.read_text()) if gate.exists() else dict(state='RUNNING')
    calibrations=[json.loads(p.read_text()) for p in sorted(groot.glob('calibration_load_*.json'),reverse=True)]
    selection=ROOT/'results_r2/E11_validation/frozen.json'
    phase_note='E11′已冻结；正式阶段的完成状态见下表及PROGRESS。' if selection.exists() else 'E11′尚未冻结；测试种子仍锁定。'
    lines=['# R2 v2.3 实验结果','', '**G0′状态：'+state['state']+'。** '+state.get('reason',phase_note),
      '', '## G0′负载与电池标定','', '576星、288商品、公共K=4、新λ*、按源城市独立MMPP，V0/100MJ/1轨，10验证种子1000–1009。此标定按v2.3第3.7条沿用既有结果（文件哈希保留），不充当独立测试结论。','',
      '| load | 平均ON利用率 | 交付率 | P95利用率 | 增量能耗占比 | 日食取能均值 kJ | P90 kJ |',
      '|---:|---:|---:|---:|---:|---:|---:|']
    tex=['% Generated by hbag3.report_r2 from G0_prime JSON','\\begin{tabular}{rrrrrrr}','\\toprule',
         'Load & Util. (\\%) & Delivered (\\%) & P95 (\\%) & Variable (\\%) & Mean (kJ) & P90 (kJ) \\\\','\\midrule']
    for c in calibrations:
        values=(c['load'],100*c['on_link_utilization_mean'],100*c['delivered_offered_ratio'],100*c['on_link_utilization_p95'],100*c['variable_energy_fraction'],c['eclipse_draw_mean_j']/1000,c['eclipse_draw_p90_j']/1000)
        lines.append('|%g|%.4f%%|%.4f%%|%.4f%%|%.4f%%|%.6f|%.6f|'%values)
        tex.append('%g & %.4f & %.4f & %.4f & %.4f & %.6f & %.6f \\\\'%values)
    tex+=['\\bottomrule','\\end{tabular}'];(ROOT/'tables_r2/G0_prime_calibration.tex').write_text('\n'.join(tex)+'\n')
    if calibrations:
        lines+=['','| load | seed | λ*_new Gbit/s | λ*_R1 Gbit/s | 比值 | 利用率 | 交付率 |','|---:|---:|---:|---:|---:|---:|---:|']
        for c in calibrations:
            for r in c['per_seed']:lines.append('|%g|%d|%.9f|%.9f|%.6f|%.4f%%|%.4f%%|'%(c['load'],r['seed'],r['reference_scale_bps']/1e9,r['reference_r1_bps']/1e9,r['reference_new_over_r1'],100*r['on_link_utilization_mean'],100*r['delivered_offered_ratio']))
    lines+=['','完整标定指标、五档电池容量、日食终端使用情况及闸门判定见GATE_R2.md；来源为results_r2/G0_prime/calibration_load_*/seed_*/inputs.npz、inputs/paths.npz、V0/raw.npz、slots.json、summary.json。',
      '', '## 后续状态','', '| 阶段 | 状态 |','|---|---|',
      '| E11′ | '+('已冻结' if (ROOT/'results_r2/E11_validation/frozen.json').exists() else '尚未完成；测试锁定')+' |',
      '| P0/P1/P2 | 只有G0′与E11′均通过后启动；完成记录见PROGRESS.md |',
      '', '不利结果原样保留；不能用V0标定推断HBAG比较优势。没有“R2 P0 DONE”就不代表P0完成。',
      '', '## R1与旧版归档','', 'R1科学产物保留，E0/E12按方案沿用；可重建cache/已获用户授权，清单先登记后删除。v2.2及更早结果与代码均归档，不进论文；v2.3复用标定、紧张度和未改方法参数，功能与HBAG选参重跑，所有正式方法使用4000–4019。R1核验见results_r2/audit/r1_integrity.json。',
      '', '```sh','python scripts/run_all_r2.py --priority ALL --workers 9','```',
      '', '入口必须先通过G0′和E11′。未通过时打印停止原因并保留原始输出，不自动扩网格或更改模型。']
    (ROOT/'RESULTS_R2.md').write_text('\n'.join(lines)+'\n')


def main():
    documentation();runtime_report();results_report();record=audit_r1()
    from .delivery_r2 import extend_gate,extend_results,extend_runtime
    extend_gate();extend_results();extend_runtime()
    from .runtime_v23 import refresh
    refresh()
    print(json.dumps(dict(r1_strict_integrity_passed=record['passed'],mismatch_count=len(record['mismatches']),strict_audit='results_r2/audit/r1_integrity.json'),ensure_ascii=False))


if __name__=='__main__':main()
