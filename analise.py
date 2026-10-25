"""Recalcula os números do artigo P1 x P1_SIM (KUKA KR 60-3) a partir dos CSVs brutos.

Uso: python -I analise.py
Grava resultados/numeros.json e figures/fig_{ciclos,torque_ciclo,tcp_path,torque_tempo,resultados}.pdf.
Termina com asserts: se algum número mudar sem motivo, o script falha.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy import stats

AQUI = Path(__file__).resolve().parent
BRUTOS = AQUI / "brutos"
RES = AQUI / "resultados"
FIGS = AQUI / "figures"
EIXOS = [f"A{i}" for i in range(1, 7)]
CONDS = {  # chave -> (arquivo, programa, velocidade)
    "P1_100": ("OP1_100_lote001_100ciclos.csv", "P1", 100),
    "P1_SIM_100": ("OP1_SIM_100_lote001_100ciclos.csv", "P1_SIM", 100),
    "P1_75": ("OP1_75_lote001_100ciclos.csv", "P1", 75),
    "P1_SIM_75": ("OP1_SIM_75_lote001_100ciclos.csv", "P1_SIM", 75),
}


def checar_hashes():
    for linha in (BRUTOS / "SHA256SUMS").read_text().splitlines():
        if not linha.strip():
            continue
        h, nome = linha.split(maxsplit=1)
        nome = nome.lstrip("*")
        real = hashlib.sha256((BRUTOS / nome).read_bytes()).hexdigest()
        if real != h:
            raise SystemExit(f"SHA256 não confere: {nome}")


def resumo(x):
    x = np.asarray(x, float)
    return {
        "n": int(x.size), "media": x.mean(), "dp": x.std(ddof=1), "mediana": float(np.median(x)),
        "min": x.min(), "max": x.max(), "lag1": float(np.corrcoef(x[:-1], x[1:])[0, 1]),
    }


def hedges_g(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp * (1 - 3 / (4 * (na + nb) - 9))


def cliff_delta(a, b):
    d = np.sign(np.subtract.outer(a, b))
    return float(d.mean())


def rms(x):
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2)))


def pico(x):
    return float(np.nanmax(np.abs(x)))


def var_pct(a, b):  # P1 -> P1_SIM
    return 100 * (b - a) / abs(a)


def analisar(chave):
    arq = CONDS[chave][0]
    df = pd.read_csv(BRUTOS / arq, encoding="utf-8-sig", low_memory=False)
    te = df["tempo_execucao"].to_numpy(float)
    ts = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    ts_s = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
    fim = np.flatnonzero(df["cycle_event"].astype(str).str.strip().eq("CYCLE_END").to_numpy())
    t_fim = te[fim]
    iv = np.diff(t_fim)
    iv_ts = np.diff(ts_s[fim])

    # Saltos do relógio do PC: timestamp andando para trás ou divergência súbita do tempo_execucao
    off = ts_s - (te - te[0])
    d_off = np.diff(off)
    d_ts = np.diff(ts_s)
    idx = np.flatnonzero((d_ts < 0) | (np.abs(d_off) > 0.5))
    saltos = [{
        "linha_csv": int(i + 2), "antes": df["timestamp"].iloc[i], "depois": df["timestamp"].iloc[i + 1],
        "delta_timestamp_s": float(d_ts[i]), "delta_tempo_execucao_s": float(te[i + 1] - te[i]),
        "salto_s": float(d_off[i]), "cycle_id": int(df["cycle_id"].iloc[i + 1]),
    } for i in idx]

    dt = np.diff(te)
    janela_b = slice(fim[0], fim[-1] + 1)
    torque = {}
    for e in EIXOS:
        x = df[f"{e}_torque_Nm"].to_numpy(float)
        rms_ciclo = [rms(x[fim[k] + 1: fim[k + 1] + 1]) for k in range(len(fim) - 1)]
        torque[e] = {
            "rms_arquivo": rms(x), "pico_arquivo": pico(x),
            "rms_janela": rms(x[janela_b]), "pico_janela": pico(x[janela_b]),
            "rms_ciclo_media": float(np.mean(rms_ciclo)), "rms_ciclo_dp": float(np.std(rms_ciclo, ddof=1)),
            "nan": int(np.isnan(x).sum()),
        }

    estados = df["estado_programa"].astype(str).str.strip()
    fora = df.loc[estados.ne("#P_ACTIVE"), ["estado_programa", "cycle_id"]]
    return {
        "arquivo": arq, "linhas": int(len(df)), "colunas": int(df.shape[1]), "n_cycle_end": int(fim.size),
        "intervalos": iv.tolist(), "ciclo": resumo(iv),
        "ciclo_timestamp": resumo(iv_ts),
        "amostragem": {"dt_mediano_s": float(np.median(dt)), "dt_medio_s": float(dt.mean()),
                       "dt_max_s": float(dt.max()), "taxa_hz": float(1 / np.median(dt))},
        "saltos_relogio": saltos, "torque": torque,
        "celulas_vazias_pct": float(100 * df.isna().to_numpy().mean()),
        "estados_nao_ativos": {"n": int(len(fora)),
                               "por_estado": fora["estado_programa"].value_counts().to_dict(),
                               "cycle_ids": sorted(set(int(c) for c in fora["cycle_id"]))},
        "extras": extras(df, fim),
        "extras8": extras8(df, fim),
        "extras9": extras9(df, fim),
        "_df": df, "_fim": fim,
    }


def unicos(serie):
    v = serie.dropna().value_counts().sort_index()
    return {str(k): int(n) for k, n in v.items()}


def extras(df, fim):
    """Seção 7: colunas de configuração, trajetória e logger, janela do 1º ao último CYCLE_END."""
    w = df.iloc[fim[0]: fim[-1] + 1]
    seqs = []
    for k in range(len(fim) - 1):  # ciclo k = linhas [fim[k], fim[k+1]); CYCLE_END abre o próximo ciclo
        p = df["ponto"].iloc[fim[k]: fim[k + 1]].astype(str).tolist()
        seqs.append(tuple(x for i, x in enumerate(p) if i == 0 or x != p[i - 1]))
    modal, n_modal = pd.Series(seqs).value_counts().index[0], int(pd.Series(seqs).value_counts().iloc[0])
    mov = np.diff(df["movement_id"].to_numpy()[fim])
    q = lambda c, pc: float(np.nanpercentile(w[c].to_numpy(float), pc))
    v = w["TCP_speed_m_s"].to_numpy(float)
    return {
        "config": {c: unicos(w[c]) for c in ("override_program_pct", "override_robot_pct", "payload_kg",
                                             "tool_id", "base_id")},
        "config_vazias_pct": {c: float(100 * w[c].isna().mean()) for c in ("override_program_pct", "payload_kg")},
        "pontos_distintos": sorted(set(w["ponto"].dropna().astype(str))),
        "seq_modal": list(modal), "seq_modal_n_ciclos": n_modal, "seq_distintas": len(set(seqs)),
        "mov_por_ciclo": {"media": float(mov.mean()), "min": int(mov.min()), "max": int(mov.max())},
        "tcp_speed": {"max": float(np.nanmax(v)), "media": float(np.nanmean(v))},
        "logger": {"sample_quality": unicos(w["sample_quality"]), "overrun": unicos(w["overrun"]),
                   "read_errors": unicos(w["read_errors"]),
                   **{f"{c}_{n}": q(c, pc) for c in ("jitter_ms", "read_time_ms", "sample_dt_s")
                      for n, pc in (("mediana", 50), ("p95", 95))},
                   "periodo_alvo_s_I": float(np.median(w["sample_dt_s"] - w["jitter_ms"] / 1000))},
        "vazias": {c: bool(df[c].isna().all()) for c in ("approx_cdis_mm", "approx_cptp_pct", "approx_cvel_pct",
                                                         "gear_jerk_pct")},
    }


def extras8(df, fim):
    """Seção 8. Horários e temperatura: arquivo inteiro; o resto: janela do 1º ao último CYCLE_END."""
    w = df.iloc[fim[0]: fim[-1] + 1]
    temp = {}
    for e in EIXOS:
        t = df[f"{e}_motor_temp_K"].dropna()
        temp[e] = {"inicio": float(t.iloc[0]), "fim": float(t.iloc[-1]), "n": int(t.size)} if t.size else None
    xyz = df[["TCP_X_meas", "TCP_Y_meas", "TCP_Z_meas"]].to_numpy(float) / 1000
    passo = np.r_[np.nan, np.linalg.norm(np.diff(xyz, axis=0), axis=1)]
    vel = df["TCP_speed_m_s"].to_numpy(float)
    comp, vmax = [], []
    for k in range(len(fim) - 1):  # passos que terminam em (fim[k], fim[k+1]]
        comp.append(np.nansum(passo[fim[k] + 1: fim[k + 1] + 1]))
        vmax.append(np.nanmax(vel[fim[k] + 1: fim[k + 1] + 1]))
    eixo = {}
    for e in EIXOS:
        tmax = w[f"{e}_torque_max_Nm"].dropna().unique()
        tq = w[f"{e}_torque_Nm"].to_numpy(float)
        x = w[f"{e}_current_pct"].to_numpy(float) * tmax[0] / 100
        ok = np.isfinite(x) & np.isfinite(tq)
        reg = stats.linregress(x[ok], tq[ok])
        eixo[e] = {"torque_max_valores": [float(v) for v in tmax],
                   "pico_sobre_max_pct": 100 * pico(tq) / tmax[0],
                   "usage_p95_pct": float(np.nanpercentile(w[f"{e}_torque_usage_pct"], 95)),
                   "speed_abs_p95_pct": float(np.nanpercentile(np.abs(w[f"{e}_speed_pct"]), 95)),
                   "reg_coef": float(reg.slope), "reg_r": float(reg.rvalue)}
    return {"inicio": df["timestamp"].iloc[0], "fim": df["timestamp"].iloc[-1], "temp_motor_K": temp,
            "caminho_tcp_m": {"media": float(np.mean(comp)), "dp": float(np.std(comp, ddof=1))},
            "vmax_tcp_ciclo_mediana": float(np.median(vmax)), "eixo": eixo}


TCP = ["TCP_X_meas", "TCP_Y_meas", "TCP_Z_meas"]  # exportadas em mm (faixa 0–1774); convertidas para m


def extras9(df, fim):
    """Seção 9 por condição; janela do 1º ao último CYCLE_END."""
    w = df.iloc[fim[0]: fim[-1] + 1]
    z = w["TCP_Z_meas"].to_numpy(float) / 1000
    te = df["tempo_execucao"].to_numpy(float)
    ang, tq = {}, {}
    for e in EIXOS:
        a = w[f"{e}_measured_deg"].to_numpy(float)
        ang[e] = {"media": float(np.nanmean(a)), "min": float(np.nanmin(a)), "max": float(np.nanmax(a))}
        x = w[f"{e}_torque_Nm"].to_numpy(float)
        x = x[np.isfinite(x)]
        mu, sg = float(x.mean()), float(x.std(ddof=0))
        xa = df[f"{e}_torque_Nm"].to_numpy(float)
        pc = [float(np.nanmax(np.abs(xa[fim[k] + 1: fim[k + 1] + 1]))) for k in range(len(fim) - 1)]
        rc = [rms(xa[fim[k] + 1: fim[k + 1] + 1]) for k in range(len(fim) - 1)]
        pc_alt = [float(np.nanmax(np.abs(xa[fim[k]: fim[k + 1]]))) for k in range(len(fim) - 1)]
        tq[e] = {"mu": mu, "abs_mu": abs(mu), "sigma_pop": sg, "rms": rms(x),
                 "rms2_menos_mu2_sigma2": rms(x) ** 2 - (mu ** 2 + sg ** 2),
                 "sat_speed_pct": float(100 * np.mean(np.abs(w[f"{e}_speed_pct"].to_numpy(float)) >= 99.9)),
                 "pico_ciclo_mediana": float(np.median(pc)),
                 "pico_ciclo_mediana_alt": float(np.median(pc_alt)),
                 "cv_rms_ciclo": float(np.std(rc, ddof=1) / np.mean(rc))}
    dtm = float(np.median(np.diff(te)))
    return {"z_max_m": float(np.nanmax(z)), "z_acima_0_9_pct": float(100 * np.nanmean(z > 0.9)),
            "angulos_deg": ang, "torque": tq,
            "media_telescopica_s": float((te[fim[-1]] - te[fim[0]]) / (len(fim) - 1)),
            "erro_quantizacao_s": dtm / (len(fim) - 1)}


EIXOS_DERIVA = ["A1", "A2", "A3", "A5"]


def secao11(R):
    """Seção 11: deriva do RMS por ciclo, caminho/velocidade do TCP, quantização máxima, evidência do relógio."""
    deriva, relogio, quant = {}, {}, {}
    for k, r in R.items():
        df, fim = r["_df"], r["_fim"]
        i = np.arange(len(fim) - 1)
        d = {}
        for e in EIXOS_DERIVA:
            x = df[f"{e}_torque_Nm"].to_numpy(float)
            rc = np.array([rms(x[fim[j] + 1: fim[j + 1] + 1]) for j in i])
            incl = np.polyfit(i, rc, 1)[0]  # OLS; variação do 1º ao 99º ciclo = inclinação × 98
            d[e] = 100 * incl * (i.size - 1) / rc.mean()
        e_max = max(d, key=lambda e: abs(d[e]))
        deriva[k] = {"variacao_pct": d, "max_abs_eixo": e_max, "max_abs_pct": d[e_max]}
        te = df["tempo_execucao"].to_numpy(float)
        sd = df["sample_dt_s"].to_numpy(float)
        dif = np.abs(np.diff(te) - sd[1:])
        dif = dif[np.isfinite(dif)]
        relogio[k] = {"inicio_s": float(te[0]), "monotonico": bool((np.diff(te) > 0).all()),
                      "mediana_abs_diff_menos_sample_dt_s": float(np.median(dif)),
                      "max_abs_diff_menos_sample_dt_s": float(dif.max())}
        dtmax = r["amostragem"]["dt_max_s"]
        quant[k] = {"dt_max_s": dtmax, "erro_quantizacao_max_s": dtmax / (len(fim) - 1),
                    "erro_quantizacao_mediano_s": r["extras9"]["erro_quantizacao_s"]}
    pior = max(quant, key=lambda k: quant[k]["dt_max_s"])
    tcp = {}
    for v in (100, 75):
        c = {p: R[f"{p}_{v}"]["extras8"]["caminho_tcp_m"]["media"] for p in ("P1", "P1_SIM")}
        t = {p: R[f"{p}_{v}"]["ciclo"]["media"] for p in ("P1", "P1_SIM")}
        tcp[str(v)] = {"caminho_P1_m": c["P1"], "caminho_SIM_m": c["P1_SIM"],
                       "variacao_caminho_pct": var_pct(c["P1"], c["P1_SIM"]),
                       "vel_media_P1_m_s": c["P1"] / t["P1"], "vel_media_SIM_m_s": c["P1_SIM"] / t["P1_SIM"]}
    return {"deriva_rms_ciclo": deriva, "tcp_caminho_velocidade": tcp, "quantizacao": quant,
            "quantizacao_pior": {"condicao": pior, **quant[pior]}, "relogio_tempo_execucao": relogio}


def seq_ciclos(df, fim):
    """Sequência de `ponto` por ciclo (linhas [fim[k], fim[k+1]), repetições colapsadas), igual à seção 7."""
    out = []
    for k in range(len(fim) - 1):
        p = df["ponto"].iloc[fim[k]: fim[k + 1]].astype(str).tolist()
        out.append(tuple(x for i, x in enumerate(p) if i == 0 or x != p[i - 1]))
    return out


def secao12(R):
    """Seção 12: desvios da sequência modal, RMS ponderado pelo tempo, razões 75/100, pré-janela, saturação de P1."""
    desvios = {}
    for k, r in R.items():
        seqs = seq_ciclos(r["_df"], r["_fim"])
        modal = pd.Series(seqs).value_counts().index[0]
        c = {"fora_modal": 0, "so_omissao": 0, "so_fronteira": 0, "omissao_e_fronteira": 0, "outro": 0}
        omit = {}
        for s in seqs:
            if s == modal:
                continue
            c["fora_modal"] += 1
            om = lambda t: [modal[i] for i in range(len(modal)) if modal[:i] + modal[i + 1:] == t]
            pts = om(s)
            # fronteira: o ciclo termina com o 1º ponto do seguinte (testada só se s, como está, não se explica)
            fr = not pts and len(s) > 1 and s[-1] == s[0] and (s[:-1] == modal or bool(om(s[:-1])))
            base = s[:-1] if fr else s
            pts = pts or (om(base) if fr else [])
            if pts:
                omit[pts[0]] = omit.get(pts[0], 0) + 1
            chave = ("omissao_e_fronteira" if fr else "so_omissao") if pts else (
                "so_fronteira" if fr and base == modal else "outro")
            c[chave] += 1
        desvios[k] = {**c, "um_ponto_omitido_total": c["so_omissao"] + c["omissao_e_fronteira"],
                      "pontos_omitidos": dict(sorted(omit.items(), key=lambda t: -t[1]))}
    pond = {}
    for k, r in R.items():
        df, fim = r["_df"], r["_fim"]
        te = df["tempo_execucao"].to_numpy(float)
        s = slice(fim[0], fim[-1] + 1)
        wt = {"tras": np.r_[np.nan, np.diff(te)][s], "central": np.r_[np.nan, (te[2:] - te[:-2]) / 2, np.nan][s]}
        pond[k] = {}
        for e in EIXOS:
            x = df[f"{e}_torque_Nm"].to_numpy(float)[s]
            pond[k][e] = {}
            for n, w in wt.items():
                ok = np.isfinite(x) & np.isfinite(w)
                pond[k][e][n] = float(np.sqrt(np.sum(w[ok] * x[ok] ** 2) / np.sum(w[ok])))
    drms = {}
    for v in (100, 75):
        drms[str(v)] = {}
        for e in EIXOS:
            a, b = pond[f"P1_{v}"][e], pond[f"P1_SIM_{v}"][e]
            am = var_pct(R[f"P1_{v}"]["torque"][e]["rms_janela"], R[f"P1_SIM_{v}"]["torque"][e]["rms_janela"])
            dt_, dc = var_pct(a["tras"], b["tras"]), var_pct(a["central"], b["central"])
            drms[str(v)][e] = {"amostral_pct": am, "ponderado_pct": dt_, "ponderado_central_pct": dc,
                               "dif_pp": dt_ - am, "dif_central_pp": dc - am}
    todos = [(v, e, d["dif_pp"]) for v, x in drms.items() for e, d in x.items()]
    vmax, emax, dmax = max(todos, key=lambda t: abs(t[2]))
    razoes = {p: {e: R[f"{p}_75"]["torque"][e]["rms_janela"] / R[f"{p}_100"]["torque"][e]["rms_janela"]
                  for e in EIXOS} for p in ("P1", "P1_SIM")}
    outros = [razoes[p][e] for p in razoes for e in ("A2", "A4", "A5", "A6")]
    pre = {}
    for k, r in R.items():
        st = r["_df"]["estado_programa"].astype(str).str.strip().iloc[: r["_fim"][0]]
        pre[k] = {"amostras_antes_1o_cycle_end": int(r["_fim"][0]), "em_P_STOP": int(st.eq("#P_STOP").sum()),
                  "nao_P_ACTIVE": int(st.ne("#P_ACTIVE").sum()),
                  "por_estado": {str(a): int(b) for a, b in st.value_counts().items()},
                  "nao_P_ACTIVE_no_arquivo": r["estados_nao_ativos"]["n"]}
    sat = {}
    for k in ("P1_100", "P1_75"):
        df, fim = R[k]["_df"], R[k]["_fim"]
        w = df.iloc[fim[0]: fim[-1] + 1]
        sat[k] = {}
        for e in EIXOS:
            sp = np.abs(w[f"{e}_speed_pct"].to_numpy(float))
            sat[k][e] = {"pct_amostras_abs_ge_100": float(100 * np.mean(sp >= 100)),
                         "max_abs_speed_pct": float(np.nanmax(sp))}
    smax = max(((k, e, d["max_abs_speed_pct"]) for k, x in sat.items() for e, d in x.items()), key=lambda t: t[2])
    return {"desvios_seq_modal": desvios, "rms_ponderado_tempo": pond, "delta_rms_ponderado": drms,
            "delta_rms_ponderado_dif_max": {"velocidade": vmax, "eixo": emax, "dif_pp": dmax},
            "razao_rms_75_100": razoes, "razao_rms_75_100_A2_A4_A5_A6_faixa": [min(outros), max(outros)],
            "pre_janela": pre, "saturacao_speed_P1": sat,
            "max_abs_speed_P1": {"condicao": smax[0], "eixo": smax[1], "pct": smax[2]}}


def ciclo_mediano(r):
    """Índice k do ciclo de duração mediana (99 intervalos -> mediana exata)."""
    iv = np.array(r["intervalos"])
    return int(np.argsort(iv)[len(iv) // 2])


def secao9(R):
    """Comparações cruzadas P1 x P1_SIM (100%) e ajuste T = a + b/ov."""
    pos = {}
    for k in ("P1_100", "P1_SIM_100"):
        df, fim = R[k]["_df"], R[k]["_fim"]
        w = df.iloc[fim[0]: fim[-1] + 1]
        pos[k] = (w[TCP].to_numpy(float), w.groupby("ponto")[TCP].mean())
    dist = {}
    for a, b in (("P1_100", "P1_SIM_100"), ("P1_SIM_100", "P1_100")):
        nuvem = pos[b][0]
        dist[a] = {p: float(np.min(np.linalg.norm(nuvem - xyz.to_numpy(), axis=1)))
                   for p, xyz in pos[a][1].iterrows()}
    ajuste = {}
    for p in ("P1", "P1_SIM"):
        t100, t75 = R[f"{p}_100"]["ciclo"]["media"], R[f"{p}_75"]["ciclo"]["media"]
        b = (t75 - t100) / (1 / 0.75 - 1)
        ajuste[p] = {"a_s": t100 - b, "b_s": b}
    medias = {k: {p: [float(v) for v in xyz] for p, xyz in pos[k][1].iterrows()} for k in pos}
    return {"dist_min_mm": dist, "pos_media_mm": medias, "ajuste_T_a_b_ov_I": ajuste}


def figura_tcp(R):
    """fig_tcp_path.pdf: (a) X-Y e (b) Z x ciclo normalizado, ciclo de duração mediana, 100%."""
    fig = plt.figure(figsize=(3.5, 2.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1], width_ratios=[1.25, 1])
    a1, aleg, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    aleg.axis("off")
    for p, ls, cor, nome in SERIES:
        k = f"{p}_100"
        df, fim = R[k]["_df"], R[k]["_fim"]
        c = ciclo_mediano(R[k])
        s = slice(fim[c], fim[c + 1] + 1)
        xyz = df[TCP].iloc[s].to_numpy(float) / 1000
        te = df["tempo_execucao"].to_numpy(float)[s]
        a1.plot(xyz[:, 0], xyz[:, 1], ls, color=cor, lw=1.1, label=nome)
        a2.plot(100 * (te - te[0]) / (te[-1] - te[0]), xyz[:, 2], ls, color=cor, lw=1.1)
    # locais de trabalho: posição média de P1_SIM em Xvia3 e em XP17-XP21
    m = R["P1_SIM_100"]["_df"].iloc[R["P1_SIM_100"]["_fim"][0]:]
    locais = [m.loc[m["ponto"].eq("Xvia3"), TCP].mean() / 1000,
              m.loc[m["ponto"].isin([f"XP{i}" for i in range(17, 22)]), TCP].mean() / 1000]
    for i, l in enumerate(locais):
        a1.plot(l.iloc[0], l.iloc[1], "s", ms=4, mfc="white", mec="black", mew=0.8, zorder=5,
                label="Work location" if i == 0 else None)
    a1.set_aspect("equal", adjustable="box")
    a1.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    a1.set_xlabel("X (m)", labelpad=1)
    a1.set_ylabel("Y (m)")
    a1.text(0.03, 0.97, "(a)", transform=a1.transAxes, va="top", ha="left")
    a2.axhline(0.9, ls=":", color="black", lw=0.7)
    a2.set_xlim(0, 100)
    a2.set_xlabel("Cycle (%)", labelpad=1)
    a2.set_ylabel("Z (m)")
    a2.text(0.99, 0.95, "(b)", transform=a2.transAxes, va="top", ha="right")
    for ax in (a1, a2):
        ax.grid(linewidth=0.3, color="0.85")
    h, lab = a1.get_legend_handles_labels()
    aleg.legend(h, lab, loc="center left", frameon=False, fontsize=8, handlelength=2.2)
    fig.tight_layout(h_pad=0.4)
    fig.savefig(FIGS / "fig_tcp_path.pdf")
    plt.close(fig)


def perfil_ciclo(r, eixo, grade):
    te = r["_df"]["tempo_execucao"].to_numpy(float)
    x = r["_df"][f"{eixo}_torque_Nm"].to_numpy(float)
    fim = r["_fim"]
    curvas = []
    for k in range(len(fim) - 1):
        s = slice(fim[k], fim[k + 1] + 1)
        fase = 100 * (te[s] - te[fim[k]]) / (te[fim[k + 1]] - te[fim[k]])
        ok = np.isfinite(x[s])
        curvas.append(np.interp(grade, fase[ok], x[s][ok]))
    c = np.array(curvas)
    return c.mean(0), c.std(0, ddof=1)


def figuras(R):
    # Times como o IEEEtran (fallback STIX/DejaVu fora do Windows); 8 pt no tamanho impresso
    plt.rcParams.update({"font.size": 8, "font.family": "serif",
                         "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "axes.linewidth": 0.6,
                         "pdf.fonttype": 42, "lines.linewidth": 0.9})
    # fig_ciclos: box plot dos 99 intervalos
    ordem = ["P1_100", "P1_SIM_100", "P1_75", "P1_SIM_75"]
    rot = ["P1\n100%", "P1_SIM\n100%", "P1\n75%", "P1_SIM\n75%"]
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    bp = ax.boxplot([R[k]["intervalos"] for k in ordem], patch_artist=True, widths=0.55,
                    medianprops={"color": "black"}, flierprops={"marker": "o", "markersize": 2.5})
    bp_labels = ax.set_xticks(range(1, 5), rot)
    for caixa, k in zip(bp["boxes"], ordem):
        sim = "SIM" in k
        caixa.set(facecolor="0.85" if sim else "white", linestyle="--" if sim else "-", hatch="///" if sim else None)
    ax.set_ylabel("Cycle interval (s)")
    ax.grid(axis="y", linewidth=0.3, color="0.8")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_ciclos.pdf")
    plt.close(fig)

    # fig_torque_ciclo: largura de página, 2×6 — linha (a) 100%, (b) 75%; colunas A1..A6; média ± 1 DP
    plt.rcParams.update({"xtick.labelsize": 8, "ytick.labelsize": 8})
    grade = np.linspace(0, 100, 201)
    fig, axs = plt.subplots(2, 6, figsize=(7.16, 3.1), sharex=True, sharey="col")
    for i, v in enumerate((100, 75)):
        for j, e in enumerate(EIXOS):
            ax = axs[i, j]
            for k, ls, cor, nome in [(f"P1_{v}", "-", "black", "P1"), (f"P1_SIM_{v}", "--", "0.45", r"P1$_\mathrm{SIM}$")]:
                m, s = perfil_ciclo(R[k], e, grade)
                ax.fill_between(grade, m - s, m + s, color=cor, alpha=0.18, linewidth=0)
                ax.plot(grade, m, ls, color=cor, label=nome)
            ax.grid(linewidth=0.3, color="0.85")
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 100])
            ax.set_xticks([50], minor=True)
            ax.grid(which="minor", linewidth=0.3, color="0.85")
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3))
            if i == 0:
                ax.set_title(e, fontsize=8, pad=3)
            else:
                ax.set_xlabel("Cycle (%)")
        axs[i, 0].set_ylabel(f"({'ab'[i]}) {v}%")
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, fontsize=8)
    fig.supylabel("Torque (exported as N·m)", fontsize=8, x=0.005)
    fig.align_ylabels(axs[:, 0])
    fig.tight_layout(rect=(0.015, 0, 1, 0.94), w_pad=0.4, h_pad=0.6)
    fig.subplots_adjust(left=fig.subplotpars.left - 0.022)  # tira o vão entre o rótulo comum e (a)/(b)
    fig.savefig(FIGS / "fig_torque_ciclo.pdf")
    plt.close(fig)


AZUL, LARANJA = "#1f4e79", "#c55a11"
SERIES = [("P1", "-", AZUL, "P1"), ("P1_SIM", "--", LARANJA, r"P1$_\mathrm{SIM}$")]


def perfil_tempo(r, eixo, grade):
    """Torque vs. tempo desde o CYCLE_END que abre o ciclo (relógio tempo_execucao); grade até o menor ciclo."""
    te = r["_df"]["tempo_execucao"].to_numpy(float)
    x = r["_df"][f"{eixo}_torque_Nm"].to_numpy(float)
    fim = r["_fim"]
    curvas = []
    for k in range(len(fim) - 1):
        s = slice(fim[k], fim[k + 1] + 1)
        ok = np.isfinite(x[s])
        curvas.append(np.interp(grade, (te[s] - te[fim[k]])[ok], x[s][ok]))
    c = np.array(curvas)
    return c.mean(0), c.std(0, ddof=1)


def figuras_novas(R, C):
    """fig_torque_tempo.pdf (página) e fig_resultados.pdf (coluna). Devolve números da seção 10."""
    plt.rcParams.update({"xtick.labelsize": 8, "ytick.labelsize": 8})
    trunc = {p: R[f"{p}_100"]["ciclo"]["min"] for p, *_ in SERIES}
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 2.05), sharex=True)
    for ax, e in zip(axs, ["A1", "A2", "A3", "A5"]):
        for p, ls, cor, nome in SERIES:
            g = np.linspace(0, trunc[p], 400)
            m, s = perfil_tempo(R[f"{p}_100"], e, g)
            ax.fill_between(g, m - s, m + s, color=cor, alpha=0.2, linewidth=0)
            ax.plot(g, m, ls, color=cor, lw=1.1, label=nome)
            ax.axvline(R[f"{p}_100"]["ciclo"]["media"], color=cor, ls=ls, lw=0.6, alpha=0.8)
        ax.set_title(e, fontsize=8, pad=3)
        ax.set_xlim(0, 14.4)
        ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))
        ax.grid(linewidth=0.3, color="0.85")
    axs[0].set_ylabel("Axis torque (exported, N·m)")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, fontsize=8)
    fig.supxlabel("Time since cycle start (s)", fontsize=8, y=0.005, va="bottom")
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=0.6)
    fig.subplots_adjust(bottom=0.2)  # tight_layout deixa vão grande acima do supxlabel
    fig.savefig(FIGS / "fig_torque_tempo.pdf")
    plt.close(fig)

    # fig_resultados: (a) intervalo x índice do ciclo; (b) RMS janela b a 100%, Δ% anotado
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.5, 2.9), height_ratios=[1, 1.2])
    reg = {}
    for p, ls, cor, nome in SERIES:
        for v, mfc in ((100, cor), (75, "white")):
            iv = np.array(R[f"{p}_{v}"]["intervalos"])
            i = np.arange(1, iv.size + 1)
            a.plot(i, iv, ls, color=cor, lw=0.8, marker="o", ms=1.8, mfc=mfc, mec=cor, mew=0.5,
                   label=f"{nome}, {v}%")
            lr = stats.linregress(i, iv)
            tq = stats.t.ppf(0.975, iv.size - 2)
            reg[f"{p}_{v}"] = {"inclinacao_ms_por_ciclo": 1000 * lr.slope, "ic95_ms_por_ciclo":
                               [1000 * (lr.slope - tq * lr.stderr), 1000 * (lr.slope + tq * lr.stderr)],
                               "p": float(lr.pvalue), "r2": float(lr.rvalue ** 2), "n": int(iv.size)}
    a.set_xlim(0, 100)
    a.set_ylim(7, 19)
    a.set_yticks([8, 10, 12, 14, 16, 18])
    a.set_xlabel("Cycle index", labelpad=1)
    a.set_ylabel("Cycle interval (s)")
    a.grid(linewidth=0.3, color="0.85")
    h, lab = a.get_legend_handles_labels()
    fig.legend([h[i] for i in (0, 2, 1, 3)], [lab[i] for i in (0, 2, 1, 3)], ncol=2, frameon=False, fontsize=8,
               loc="upper center", handlelength=2.2, columnspacing=1.2, borderaxespad=0.1)
    a.text(0.01, 0.03, "(a)", transform=a.transAxes, va="bottom", ha="left")
    x = np.arange(6)
    for d, v in ((-0.19, 100), (0.19, 75)):  # ΔRMS janela b, P1_SIM relativo a P1 (seção 4)
        y = [C[v]["torque"][e]["rms_janela"] for e in EIXOS]
        b.bar(x + d, y, 0.36, color="0.35" if v == 100 else "white", edgecolor="0.2",
              hatch=None if v == 100 else "////", linewidth=0.6, label=f"{v}%")
        for xi, yi in zip(x + d, y):
            b.text(xi, yi + (4 if yi >= 0 else -4), f"{yi:+.0f}".replace("-", "−"), ha="center",
                   va="bottom" if yi >= 0 else "top", fontsize=8, rotation=90)
    b.axhline(0, color="black", lw=0.6)
    b.set_ylim(-75, 205)
    b.set_yticks([0, 50, 100, 150])
    b.set_xlim(-0.6, 5.6)
    b.set_xticks(x, EIXOS)
    b.set_ylabel("ΔRMS torque (%)")
    b.grid(axis="y", linewidth=0.3, color="0.85")
    b.legend(ncol=2, frameon=False, fontsize=8, loc="upper right", borderaxespad=0.2, handlelength=1.4)
    b.text(0.01, 0.03, "(b)", transform=b.transAxes, va="bottom", ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=0.5)
    fig.align_ylabels((a, b))
    fig.savefig(FIGS / "fig_resultados.pdf")
    plt.close(fig)
    return {"truncamento_s": trunc, "regressao_intervalo_x_ciclo": reg}


def comparar(R):
    out = {}
    for v in (100, 75):
        a, b = np.array(R[f"P1_{v}"]["intervalos"]), np.array(R[f"P1_SIM_{v}"]["intervalos"])
        w = stats.ttest_ind(a, b, equal_var=False)
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        out[v] = {"reducao_pct": 100 * (1 - b.mean() / a.mean()), "welch_t": float(w.statistic),
                  "welch_gl": float(w.df), "welch_p": float(w.pvalue), "mw_U": float(u.statistic),
                  "mw_p": float(u.pvalue), "hedges_g": float(hedges_g(a, b)), "cliff_delta": cliff_delta(a, b),
                  "torque": {e: {m: var_pct(R[f"P1_{v}"]["torque"][e][m], R[f"P1_SIM_{v}"]["torque"][e][m])
                                 for m in ("rms_arquivo", "pico_arquivo", "rms_janela", "pico_janela",
                                           "rms_ciclo_media")} for e in EIXOS}}
    return out



def f(x, n=3):
    return f"{x:.{n}f}"


def escrever(R, C, S9, S10):
    limpo = {k: {kk: vv for kk, vv in r.items() if not kk.startswith("_")} for k, r in R.items()}
    json.dump({"fonte": "analise.py", "condicoes": limpo,
               "comparacao": {str(k): v for k, v in C.items()}, "secao9": S9,
               "secao10": {k: v for k, v in S10.items() if not k.startswith("_")}, "secao11": S10["_s11"], "secao12": S10["_s12"]},
              open(RES / "numeros.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)



def main():
    checar_hashes()
    RES.mkdir(exist_ok=True)
    R = {k: analisar(k) for k in CONDS}
    C = comparar(R)
    S9 = secao9(R)
    figuras(R)
    figura_tcp(R)
    S10 = figuras_novas(R, C)
    S10["_s11"] = secao11(R)
    S10["_s12"] = S12 = secao12(R)
    escrever(R, C, S9, S10)

    # Asserts: valores publicados no artigo
    for k, m, s in (("P1_100", 13.701, 0.125), ("P1_SIM_100", 8.915, 0.111),
                    ("P1_75", 17.935, 0.103), ("P1_SIM_75", 11.300, 0.127)):
        c = R[k]["ciclo"]
        assert c["n"] == 99 and R[k]["n_cycle_end"] == 100, k
        assert abs(c["media"] - m) < 0.0006 and abs(c["dp"] - s) < 0.0006, (k, c["media"], c["dp"])
    assert abs(C[100]["reducao_pct"] - 34.93) < 0.01, C[100]["reducao_pct"]
    assert abs(C[75]["reducao_pct"] - 36.99) < 0.01, C[75]["reducao_pct"]
    assert abs(R["P1_100"]["torque"]["A1"]["rms_arquivo"] - 593.69) < 0.01
    assert abs(R["P1_SIM_100"]["torque"]["A1"]["rms_arquivo"] - 1214.25) < 0.01
    assert abs(C[100]["torque"]["A1"]["rms_arquivo"] - 104.5) < 0.05
    assert abs(C[75]["torque"]["A4"]["pico_arquivo"] - 27.0) < 0.05
    assert [round(s["delta_timestamp_s"], 2) for s in R["P1_75"]["saltos_relogio"]] == [-2.75], "salto OP1_75"
    assert sum(len(r["saltos_relogio"]) for r in R.values()) == 1
    t75 = R["P1_75"]["ciclo_timestamp"]  # com o timestamp do PC, que tem o salto de relógio
    assert abs(t75["min"] - 14.84) < 0.005 and abs(t75["media"] - 17.904) < 0.0006 and abs(t75["dp"] - 0.330) < 0.0006
    for r in R.values():
        assert 0.25 <= r["amostragem"]["dt_mediano_s"] <= 0.28
    for k, r in R.items():
        x, vel = r["extras"], CONDS[k][2]
        assert x["config"]["override_program_pct"] == {f"{vel}.0": x["config"]["override_program_pct"].get(f"{vel}.0")}
        assert list(x["config"]["payload_kg"]) == ["30.0"], (k, x["config"]["payload_kg"])
        assert all(x["vazias"].values()), k
    x8 = R["P1_SIM_100"]["extras8"]["eixo"]["A1"]
    assert abs(x8["pico_sobre_max_pct"] - 72.4) < 0.05, x8["pico_sobre_max_pct"]
    assert abs(R["P1_100"]["extras8"]["eixo"]["A2"]["pico_sobre_max_pct"] - 58.2) < 0.05
    x9 = R["P1_100"]["extras9"]
    assert abs(x9["z_max_m"] - 1.695) < 0.001 and abs(x9["z_acima_0_9_pct"] - 55) < 1, x9
    assert R["P1_SIM_100"]["extras9"]["z_acima_0_9_pct"] == 0
    for r in R.values():
        assert all(abs(t["rms2_menos_mu2_sigma2"]) < 1e-6 * t["rms"] ** 2 for t in r["extras9"]["torque"].values())
        assert abs(r["extras9"]["media_telescopica_s"] - r["ciclo"]["media"]) < 1e-9
    assert abs(S10["truncamento_s"]["P1"] - 13.446) < 0.001 and abs(S10["truncamento_s"]["P1_SIM"] - 8.645) < 0.001
    S11 = secao11(R)
    dv = S11["deriva_rms_ciclo"]["P1_100"]
    for e, v in (("A1", -7.6), ("A2", -2.4), ("A3", -2.8), ("A5", -1.5)):
        assert abs(dv["variacao_pct"][e] - v) < 0.05, (e, dv["variacao_pct"][e])
    assert dv["max_abs_eixo"] == "A1"
    t = S11["tcp_caminho_velocidade"]
    assert abs(t["100"]["variacao_caminho_pct"] + 25.4) < 0.05, t["100"]
    assert [round(t[v][c], 2) for v in ("100", "75") for c in ("vel_media_P1_m_s", "vel_media_SIM_m_s")] == [0.64, 0.74, 0.50, 0.59]
    qp = S11["quantizacao_pior"]
    assert round(qp["dt_max_s"], 3) == 0.762 and round(qp["erro_quantizacao_max_s"], 4) == 0.0077, qp
    rl = S11["relogio_tempo_execucao"]
    assert all(c["monotonico"] and c["mediana_abs_diff_menos_sample_dt_s"] < 1e-9 for c in rl.values())
    assert sum(c["inicio_s"] < 1e-3 for c in rl.values()) == 3 and rl["P1_75"]["inicio_s"] > 1
    dsv = S12["desvios_seq_modal"]
    assert [(dsv[k]["fora_modal"], dsv[k]["um_ponto_omitido_total"], dsv[k]["so_fronteira"]) for k in CONDS] == \
        [(33, 33, 0), (34, 32, 2), (1, 1, 0), (1, 0, 1)], dsv
    assert dsv["P1_100"]["so_omissao"] == 29 and dsv["P1_100"]["omissao_e_fronteira"] == 4
    dm = S12["delta_rms_ponderado_dif_max"]
    assert (dm["velocidade"], dm["eixo"], round(dm["dif_pp"], 1)) == ("100", "A1", -2.0), dm
    assert [round(x, 2) for x in S12["razao_rms_75_100_A2_A4_A5_A6_faixa"]] == [0.71, 0.89]
    assert [(d["amostras_antes_1o_cycle_end"], d["em_P_STOP"], d["nao_P_ACTIVE"]) for d in S12["pre_janela"].values()] == \
        [(152, 75, 75), (134, 23, 83), (118, 0, 0), (113, 25, 38)]
    assert all(d["pct_amostras_abs_ge_100"] == 0 for x in S12["saturacao_speed_P1"].values() for d in x.values())
    assert round(S12["max_abs_speed_P1"]["pct"], 2) == 99.75
    print("OK: todos os asserts passaram")


if __name__ == "__main__":
    main()
