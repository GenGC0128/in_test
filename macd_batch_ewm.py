#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MACD 多周期批量评估脚本
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')


def calc_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def calc_macd(df, close_col='close_adj', fast=12, slow=26, signal=9):
    df = df.copy()
    df['DIF'] = calc_ema(df[close_col], fast) - calc_ema(df[close_col], slow)
    df['DEA'] = calc_ema(df['DIF'], signal)
    df['MACD'] = 2 * (df['DIF'] - df['DEA'])
    return df


def resample_to_weekly(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    weekly = pd.DataFrame()
    weekly['open_adj'] = df['open_adj'].resample('W').first()
    weekly['high_adj'] = df['high_adj'].resample('W').max()
    weekly['low_adj'] = df['low_adj'].resample('W').min()
    weekly['close_adj'] = df['close_adj'].resample('W').last()
    weekly['volume'] = df['volume'].resample('W').sum()
    return weekly.dropna().reset_index()


def resample_to_monthly(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    monthly = pd.DataFrame()
    monthly['open_adj'] = df['open_adj'].resample('ME').first()
    monthly['high_adj'] = df['high_adj'].resample('ME').max()
    monthly['low_adj'] = df['low_adj'].resample('ME').min()
    monthly['close_adj'] = df['close_adj'].resample('ME').last()
    monthly['volume'] = df['volume'].resample('ME').sum()
    return monthly.dropna().reset_index()


def detect_divergence(df, close_col='close_adj', lookback=30, window=2):
    if len(df) < lookback + 5:
        return "数据不足"

    recent = df.tail(lookback).reset_index(drop=True)
    close = recent[close_col]
    dif = recent['DIF']

    price_peaks, price_valleys = [], []
    for i in range(window, len(recent) - window):
        if all(close.iloc[i] >= close.iloc[i-j] for j in range(1, window+1)) and \
           all(close.iloc[i] >= close.iloc[i+j] for j in range(1, window+1)):
            price_peaks.append(i)
        if all(close.iloc[i] <= close.iloc[i-j] for j in range(1, window+1)) and \
           all(close.iloc[i] <= close.iloc[i+j] for j in range(1, window+1)):
            price_valleys.append(i)

    dif_peaks, dif_valleys = [], []
    for i in range(window, len(recent) - window):
        if all(dif.iloc[i] >= dif.iloc[i-j] for j in range(1, window+1)) and \
           all(dif.iloc[i] >= dif.iloc[i+j] for j in range(1, window+1)):
            dif_peaks.append(i)
        if all(dif.iloc[i] <= dif.iloc[i-j] for j in range(1, window+1)) and \
           all(dif.iloc[i] <= dif.iloc[i+j] for j in range(1, window+1)):
            dif_valleys.append(i)

    result = []
    if len(price_peaks) >= 2 and len(dif_peaks) >= 2:
        pp1, pp2 = price_peaks[-2], price_peaks[-1]
        dp1, dp2 = dif_peaks[-2], dif_peaks[-1]
        if close.iloc[pp2] > close.iloc[pp1] and dif.iloc[dp2] < dif.iloc[dp1]:
            result.append("顶背离")

    if len(price_valleys) >= 2 and len(dif_valleys) >= 2:
        pv1, pv2 = price_valleys[-2], price_valleys[-1]
        dv1, dv2 = dif_valleys[-2], dif_valleys[-1]
        if close.iloc[pv2] < close.iloc[pv1] and dif.iloc[dv2] > dif.iloc[dv1]:
            result.append("底背离")

    return " | ".join(result) if result else "无"


def analyze_macd_state(df, period_name="日线", lookback=30):
    min_required = 26 + 9 + lookback
    if len(df) < min_required:
        return {"周期": period_name, "error": f"数据不足({len(df)}根)"}

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    state = {"周期": period_name}

    dif, dea, macd = latest['DIF'], latest['DEA'], latest['MACD']
    close = latest['close_adj']

    state['收盘价'] = round(close, 2)
    state['DIF'] = round(dif, 4)
    state['DEA'] = round(dea, 4)
    state['MACD'] = round(macd, 4)

    if dif > 0 and dea > 0:
        state['零轴位置'] = '零轴上方'
        zone_score = 20
    elif dif < 0 and dea < 0:
        state['零轴位置'] = '零轴下方'
        zone_score = -20
    else:
        state['零轴位置'] = '零轴附近'
        zone_score = 0

    prev_dif, prev_dea = prev['DIF'], prev['DEA']
    if prev_dif <= prev_dea and dif > dea:
        state['交叉状态'] = '刚金叉'
        cross_score = 15
    elif prev_dif >= prev_dea and dif < dea:
        state['交叉状态'] = '刚死叉'
        cross_score = -15
    elif dif > dea:
        days = 0
        for i in range(1, min(20, len(df))):
            if df.iloc[-i]['DIF'] > df.iloc[-i]['DEA']:
                days += 1
            else:
                break
        state['交叉状态'] = f'金叉{days}天'
        cross_score = 10
    else:
        days = 0
        for i in range(1, min(20, len(df))):
            if df.iloc[-i]['DIF'] < df.iloc[-i]['DEA']:
                days += 1
            else:
                break
        state['交叉状态'] = f'死叉{days}天'
        cross_score = -10

    macd_series = df['MACD'].tail(6)
    if len(macd_series) >= 3:
        rate = macd_series.iloc[-1] - macd_series.iloc[-2]
        prev_rate = macd_series.iloc[-2] - macd_series.iloc[-3]
        accel = rate - prev_rate

        if macd > 0:
            if rate > 0 and accel > 0:
                state['动能状态'] = '红柱加速'
                mom_score = 15
            elif rate > 0:
                state['动能状态'] = '红柱减速'
                mom_score = 5
            else:
                state['动能状态'] = '红柱缩短'
                mom_score = -5
        else:
            if rate < 0 and accel < 0:
                state['动能状态'] = '绿柱加速'
                mom_score = -15
            elif rate < 0:
                state['动能状态'] = '绿柱减速'
                mom_score = -5
            else:
                state['动能状态'] = '绿柱缩短'
                mom_score = 10
    else:
        state['动能状态'] = '-'
        mom_score = 0

    div = detect_divergence(df, lookback=lookback)
    state['背离信号'] = div
    div_score = 20 if '底背离' in div else (-25 if '顶背离' in div else 0)

    score = 50 + zone_score + cross_score + mom_score + div_score
    state['趋势评分'] = max(0, min(100, score))

    if state['趋势评分'] >= 80:
        state['操作等级'] = 'A'
        state['操作建议'] = '强势多头'
    elif state['趋势评分'] >= 65:
        state['操作等级'] = 'B'
        state['操作建议'] = '偏多'
    elif state['趋势评分'] >= 50:
        state['操作等级'] = 'C'
        state['操作建议'] = '中性偏强'
    elif state['趋势评分'] >= 35:
        state['操作等级'] = 'D'
        state['操作建议'] = '中性偏弱'
    elif state['趋势评分'] >= 20:
        state['操作等级'] = 'E'
        state['操作建议'] = '偏弱'
    else:
        state['操作等级'] = 'F'
        state['操作建议'] = '弱势空头'

    return state


def multi_period_analysis(df_daily):
    daily = calc_macd(df_daily.copy())
    daily_state = analyze_macd_state(daily, "日线", lookback=min(30, len(daily)//2))

    weekly_df = resample_to_weekly(df_daily)
    if len(weekly_df) >= 35:
        weekly = calc_macd(weekly_df)
        weekly_state = analyze_macd_state(weekly, "周线", lookback=min(12, len(weekly)//2))
    else:
        weekly_state = {"周期": "周线", "error": f"数据不足({len(weekly_df)}根)"}

    monthly_df = resample_to_monthly(df_daily)
    if len(monthly_df) >= 10:
        monthly = calc_macd(monthly_df)
        monthly_state = analyze_macd_state(monthly, "月线", lookback=min(6, len(monthly)//2))
    else:
        monthly_state = {"周期": "月线", "error": f"数据不足({len(monthly_df)}根)"}

    return daily_state, weekly_state, monthly_state


def load_data(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()

    cleaned = [line.strip().strip('"') for line in raw_lines]

    first_data_line = cleaned[1] if len(cleaned) > 1 else cleaned[0]
    if '\t' in first_data_line:
        sep = '\t'
    elif ',' in first_data_line:
        sep = ','
    else:
        sep = '\s+'

    tmp = csv_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(cleaned))

    df = pd.read_csv(tmp, sep=sep, engine='python')
    df.columns = [c.strip().strip('"') for c in df.columns]

    date_col = None
    for c in ['date', 'Date', 'DATE', 'trade_date', 'datetime', 'time']:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        for c in df.columns:
            if 'date' in c.lower():
                date_col = c
                break

    if date_col is None:
        raise KeyError(f"找不到日期列: {df.columns.tolist()}")

    df.rename(columns={date_col: 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])

    os.remove(tmp)
    return df.sort_values('date').reset_index(drop=True)

def get_stock_names_dict():
    zhishu = {'sh.000001': '上证综指',
              'sh.000300': '沪深300',
              'sh.000905': '中证500',
              'sz.399001': '深证成指',
              'sz.399006': '创业板'
              }
    hs300 = pd.read_csv('d:/hs300_stocks.csv', encoding="gbk").set_index('code')['code_name'].to_dict()
    zz500 = pd.read_csv('d:/zz500_stocks.csv', encoding="gbk").set_index('code')['code_name'].to_dict()
    hs300.update(zz500)
    hs300.update(zhishu)
    return  hs300


# ==================== 策略建议（确保正确匹配）====================

def get_strategy_advice(monthly_zone, weekly_zone, daily_zone):
    """
    根据月/周/日三周期零轴位置，返回决策表匹配的一句话建议
    """
    # 标准化：提取核心方向
    def parse_zone(z):
        z = str(z)
        if '上方' in z:
            return '多'
        elif '下方' in z:
            return '空'
        else:
            return '-'

    m = parse_zone(monthly_zone)
    w = parse_zone(weekly_zone)
    d = parse_zone(daily_zone)

    # 完整三周期决策表
    advice_map = {
        ('多', '多', '多'): "💪 三周期共振多头，大力做多，持仓为主",
        ('多', '多', '空'): "🟡 大趋势好，短期调整，等日线企稳再介入",
        ('多', '空', '多'): "⚠️ 月线支撑反弹，周线没确认，轻仓试探见好就收",
        ('多', '空', '空'): "🛑 大趋势多，但中短期都在调整，观望等周线日线转好",
        ('空', '空', '空'): "⛔ 三周期共振空头，空仓不做",
        ('空', '空', '多'): "⛔ 大趋势空，只是反弹，不重仓，反弹是减亏机会",
        ('空', '多', '多'): "⚠️ 周线反弹，月线压制，短线可做别恋战",
        ('空', '多', '空'): "🛑 混沌期方向不明，观望等方向选择",
    }

    key = (m, w, d)
    if key in advice_map:
        return advice_map[key]

    # ========== 降级：两周期 ==========
    if d == '-' and m != '-' and w != '-':
        if m == '多' and w == '多': return "🟢 月周共振多头，日线待确认，可积极关注"
        if m == '多' and w == '空': return "🟡 月线多但周线调整，等日线企稳再介入"
        if m == '空' and w == '空': return "🔴 月周共振空头，回避"
        if m == '空' and w == '多': return "⚠️ 周线反弹，月线压制，谨慎参与"

    if w == '-' and m != '-' and d != '-':
        if m == '多' and d == '多': return "🟢 月线多头，日线顺势，可积极参与"
        if m == '多' and d == '空': return "🟡 大趋势多，短期调整，等日线企稳"
        if m == '空' and d == '空': return "🔴 月日共振空，回避"
        if m == '空' and d == '多': return "⚠️ 日线反弹，大趋势空，不重仓"

    if m == '-' and w != '-' and d != '-':
        if w == '多' and d == '多': return "🟢 周日共振多，积极参与"
        if w == '多' and d == '空': return "🟡 周线多，日线调整，等企稳"
        if w == '空' and d == '空': return "🔴 周日共振空，回避"
        if w == '空' and d == '多': return "⚠️ 日线反弹，周线空，谨慎"

    # 单周期
    if m == '多' or w == '多' or d == '多':
        return "🟡 部分多头信号，但数据不足，等确认"
    elif m == '空' or w == '空' or d == '空':
        return "🟠 部分空头信号，但数据不足，谨慎"
    else:
        return "⚪ 数据不足，无法判断"


# ==================== 批量分析 ====================

def extract_summary(code, daily, weekly, monthly):
    def safe_get(state, key, default="-"):
        if 'error' in state:
            return default
        return state.get(key, default)

    d_zone = safe_get(daily, '零轴位置', '-')
    d_cross = safe_get(daily, '交叉状态', '-')
    d_mom = safe_get(daily, '动能状态', '-')
    d_div = safe_get(daily, '背离信号', '-')
    d_score = safe_get(daily, '趋势评分', '-')
    d_grade = safe_get(daily, '操作等级', '-')

    w_zone = safe_get(weekly, '零轴位置', '-')
    w_score = safe_get(weekly, '趋势评分', '-')

    m_zone = safe_get(monthly, '零轴位置', '-')
    m_score = safe_get(monthly, '趋势评分', '-')

    zones = [z for z in [m_zone, w_zone, d_zone] if z != '-']
    scores = [s for s in [m_score, w_score, d_score] if isinstance(s, (int, float))]

    if len(zones) >= 2:
        if all('上方' in z for z in zones):
            resonance = "多头共振"
        elif all('下方' in z for z in zones):
            resonance = "空头共振"
        elif '上方' in zones[0] and '下方' in zones[-1]:
            resonance = "月多日空"
        elif '下方' in zones[0] and '上方' in zones[-1]:
            resonance = "月空日弹"
        else:
            resonance = "转换期"
    else:
        resonance = "数据不足"

    avg_score = round(sum(scores)/len(scores), 0) if scores else '-'

    # 策略建议（核心：调用决策表）
    strategy = get_strategy_advice(m_zone, w_zone, d_zone)

    signals = []
    if '底背离' in str(d_div): signals.append("底背离")
    if '顶背离' in str(d_div): signals.append("顶背离")
    if '刚金叉' in str(d_cross): signals.append("刚金叉")
    if '刚死叉' in str(d_cross): signals.append("刚死叉")
    if '绿柱缩短' in str(d_mom): signals.append("绿柱缩短")
    if '红柱缩短' in str(d_mom): signals.append("红柱缩短")
    signal_str = " | ".join(signals) if signals else "-"

    return {
        'code': code,
        '日线位置': d_zone,
        '日线交叉': d_cross,
        '日线动能': d_mom,
        '日线评分': d_score,
        '日线等级': d_grade,
        '周线位置': w_zone,
        '周线评分': w_score,
        '月线位置': m_zone,
        '月线评分': m_score,
        '平均评分': avg_score,
        '周期共振': resonance,
        '关键信号': signal_str,
        '策略建议': strategy,
        '日线背离': d_div,
    }


def batch_analyze(code_list, data_dir=".", suffix=".csv", verbose=False):
    stock_names = get_stock_names_dict()
    results = []
    errors = []

    for code in code_list:
        filepath = os.path.join(data_dir, code + suffix)

        if not os.path.exists(filepath):
            errors.append({"code": code, "reason": "文件不存在"})
            continue

        try:
            df = load_data(filepath)
            daily, weekly, monthly = multi_period_analysis(df)

            if verbose:
                print(f"\n{'='*60}")
                print(f"【{code}】 数据量: {len(df)}天")
                print(f"{'='*60}")
                for s in [monthly, weekly, daily]:
                    if 'error' in s:
                        print(f"  [{s['周期']}] {s['error']}")
                    else:
                        print(f"  [{s['周期']}] {s['零轴位置']} | {s['交叉状态']} | {s['动能状态']} | 评分:{s['趋势评分']}")

            summary = extract_summary(code, daily, weekly, monthly)
            summary['数据天数'] = len(df)
            results.append(summary)

        except Exception as e:
            errors.append({"code": code, "reason": str(e)})

    if results:
        df_result = pd.DataFrame(results)
        df_result['排序分'] = pd.to_numeric(df_result['平均评分'], errors='coerce').fillna(0)
        df_result = df_result.sort_values('排序分', ascending=False).drop('排序分', axis=1)

        df_result['stock_name'] = df_result['code'].map(stock_names)

        #print("\n" + "=" * 150)
        #print("MACD 多周期批量评估汇总表")
        #print("=" * 150)

        header = f"{'代码':<10} {'日线位置':<8} {'日线交叉':<8} {'日线动能':<8} {'日评分':>6} {'等级':<4} {'周线':<8} {'月线':<8} {'均分':>6} {'共振':<10} {'关键信号':<14} {'策略建议'}"
        #print(header)
        #print("-" * 150)

        #for _, row in df_result.iterrows():
            #line = (f"{row['code']:<10} {row['日线位置']:<8} {row['日线交叉']:<8} "
                    #f"{row['日线动能']:<8} {str(row['日线评分']):>6} {row['日线等级']:<4} "
                    #f"{row['周线位置']:<8} {row['月线位置']:<8} {str(row['平均评分']):>6} "
                    #f"{row['周期共振']:<10} {row['关键信号']:<14} {row['策略建议']}")
            #print(line)

        print("=" * 150)

        bullish = len(df_result[df_result['周期共振'] == '多头共振'])
        bearish = len(df_result[df_result['周期共振'] == '空头共振'])
        mixed = len(df_result) - bullish - bearish
        print(f"\n统计: 多头共振 {bullish} 只 | 空头共振 {bearish} 只 | 其他 {mixed} 只")

        watch = df_result[
            (df_result['周期共振'].isin(['多头共振', '月多日空'])) &
            (df_result['关键信号'] != '-') |
            (df_result['日线背离'].str.contains('底背离', na=False))
        ]
        if len(watch) > 0:
            print(f"\n⭐ 建议关注 ({len(watch)}只):")
            for _, row in watch.iterrows():
                print(f"   {row['code']}: {row['周期共振']} | {row['关键信号']} | 均分{row['平均评分']}")
                print(f"      → {row['策略建议']}")

        if errors:
            print(f"\n❌ 失败 ({len(errors)}只): " + ", ".join([e['code'] for e in errors]))

        return df_result
    else:
        print("没有成功分析的股票")
        return pd.DataFrame()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MACD 多周期批量分析')
    parser.add_argument('codes', nargs='+', help='股票代码列表')
    parser.add_argument('--dir', default='.', help='数据目录')
    parser.add_argument('--suffix', default='.csv', help='文件后缀')
    parser.add_argument('-v', '--verbose', action='store_true', help='打印每只详细报告')
    args = parser.parse_args()

    batch_analyze(args.codes, data_dir=args.dir, suffix=args.suffix, verbose=args.verbose)
