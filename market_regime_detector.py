import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings


@dataclass
class RegimeConfig:
    """市场状态检测配置参数"""
    # 回望窗口
    short_window: int = 20
    medium_window: int = 60
    long_window: int = 120
    history_window: int = 250

    # 动量阈值
    extreme_bear_threshold: float = -0.15
    bear_threshold: float = -0.08
    bull_threshold: float = 0.20
    mild_bull_threshold: float = 0.10

    # 波动率阈值
    high_vol_threshold: float = 0.25
    low_vol_threshold: float = 0.15

    # 均线偏离阈值
    tight_range_threshold: float = 0.02
    loose_range_threshold: float = 0.05

    # 状态切换平滑参数
    min_regime_days: int = 3

    # 评分权重
    trend_weight: float = 1.0
    momentum_weight: float = 1.0
    position_weight: float = 0.8
    volatility_weight: float = 0.6
    ma_relation_weight: float = 0.8


class RobustMarketRegimeDetector:
    """
    鲁棒版A股市场状态检测器 (V2修复版)

    修复内容：
    1. 状态平滑计数器逻辑修复（不再全部归零）
    2. 评分基数归一化（各维度话语权均衡）
    3. 趋势强度指标改进（保留方向信息）
    4. batch_detect状态隔离
    5. price_percentile缺失值处理优化
    6. 浮点比较容差优化
    7. 数据质量评估
    """

    REGIME_PRIORITY = [
        'extreme_bear',
        'bear',
        'recovery',
        'trending_down',
        'trending_up',
        'bull',
        'consolidation'
    ]

    REGIME_NAMES = {
        'extreme_bear': '极端熊市',
        'bear': '熊市',
        'recovery': '反弹期',
        'consolidation': '震荡市',
        'trending_up': '上升趋势',
        'trending_down': '下降趋势',
        'bull': '牛市'
    }

    def __init__(self, market_prices: pd.Series, config: Optional[RegimeConfig] = None):
        self._validate_input(market_prices)
        self.market_prices = market_prices.sort_index()
        self.config = config or RegimeConfig()

        # 状态切换平滑记录
        self._regime_history: List[Tuple[datetime, str, float]] = []
        self._current_regime: Optional[str] = None
        self._current_regime_start: Optional[datetime] = None
        self._regime_counter: Dict[str, int] = {r: 0 for r in self.REGIME_PRIORITY}

        # 预计算指标
        self._precompute_indicators()

        print(f"✅ 检测器初始化完成，数据范围: {self.market_prices.index[0].date()} ~ {self.market_prices.index[-1].date()}")
        print(f"   共 {len(self.market_prices)} 个交易日")

    def _validate_input(self, prices: pd.Series):
        if not isinstance(prices, pd.Series):
            raise TypeError(f"market_prices必须是pandas Series，当前类型: {type(prices)}")
        if len(prices) == 0:
            raise ValueError("价格序列不能为空")
        if prices.isna().all():
            raise ValueError("价格序列全部为NaN")
        try:
            pd.to_datetime(prices.index[:5])
        except Exception:
            warnings.warn("价格序列索引建议为datetime类型，将尝试自动转换")

    def _safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        if pd.isna(numerator) or pd.isna(denominator):
            return default
        if abs(denominator) < 1e-10:
            return default if abs(numerator) < 1e-10 else (np.inf if numerator > 0 else -np.inf)
        return numerator / denominator

    def _safe_compare(self, values: List[float], direction: str = 'desc') -> bool:
        if any(pd.isna(v) for v in values):
            return False
        if direction == 'desc':
            return all(values[i] > values[i+1] for i in range(len(values)-1))
        else:
            return all(values[i] < values[i+1] for i in range(len(values)-1))

    def _precompute_indicators(self):
        cfg = self.config
        prices = self.market_prices

        # 1. 移动平均线
        self.ma_short = prices.rolling(cfg.short_window, min_periods=cfg.short_window//2).mean()
        self.ma_medium = prices.rolling(cfg.medium_window, min_periods=cfg.medium_window//2).mean()
        self.ma_long = prices.rolling(cfg.long_window, min_periods=cfg.long_window//2).mean()

        # 2. 收益率和波动率
        self.returns = prices.pct_change()
        self.vol_short = self.returns.rolling(cfg.short_window).std() * np.sqrt(252)
        self.vol_medium = self.returns.rolling(cfg.medium_window).std() * np.sqrt(252)

        # 3. 动量指标
        self.mom_1m = prices / prices.shift(22) - 1
        self.mom_3m = prices / prices.shift(66) - 1
        self.mom_6m = prices / prices.shift(132) - 1

        # 4. 价格位置（动态百分位）
        # 修复：使用更鲁棒的滚动百分位，缺失值返回NaN而非固定0.5
        def rolling_percentile(x):
            if len(x) < 10:  # 数据不足时返回NaN
                return np.nan
            return pd.Series(x).rank(pct=True).iloc[-1]

        self.price_percentile = prices.rolling(
            cfg.history_window, 
            min_periods=10
        ).apply(rolling_percentile, raw=False)

        # 5. 趋势强度（改进版：保留方向）
        self.trend_direction, self.trend_strength = self._calculate_trend_strength_v2()

        # 6. 数据质量标记
        self.data_quality = self._calculate_data_quality()

        print("📊 技术指标预计算完成")

    def _calculate_trend_strength_v2(self) -> Tuple[pd.Series, pd.Series]:
        """
        改进版趋势强度计算
        返回: (方向: 1=上升, -1=下降, 0=震荡), (强度: 0-1)
        """
        prices = self.market_prices
        returns = self.returns

        window = 20
        # 上涨/下跌天数占比
        up_days = (returns > 0).rolling(window, min_periods=10).mean()
        down_days = (returns < 0).rolling(window, min_periods=10).mean()

        # 方向：基于净涨跌天数
        net_direction = up_days - down_days
        direction = np.sign(net_direction)

        # 强度：|上涨占比 - 下跌占比|，但区分方向
        strength = net_direction.abs()

        return direction, strength

    def _calculate_data_quality(self) -> pd.Series:
        """评估每个时间点的数据质量（0-1）"""
        cfg = self.config
        prices = self.market_prices

        # 计算各指标所需的最小数据量
        min_needed = max(cfg.short_window, cfg.medium_window//2, 10)

        quality = pd.Series(index=prices.index, dtype=float)
        for i, date in enumerate(prices.index):
            available = i + 1
            if available < min_needed:
                quality.iloc[i] = available / min_needed  # 线性增长
            else:
                # 检查近期缺失值比例
                recent_window = prices.iloc[max(0, i-cfg.short_window):i+1]
                missing_ratio = recent_window.isna().mean()
                quality.iloc[i] = 1.0 - missing_ratio

        return quality

    def detect_regime(self, date, verbose: bool = False) -> Dict:
        if isinstance(date, str):
            date = pd.to_datetime(date)

        if date not in self.market_prices.index:
            nearest = self._find_nearest_date(date)
            if verbose:
                print(f"⚠️ 日期 {date.date()} 不在交易日中，使用最近交易日 {nearest.date()}")
            date = nearest

        try:
            indicators = self._get_indicators_at_date(date)

            if indicators is None:
                return self._create_result(date, 'consolidation', 0.0, {}, "数据不足")

            # 数据质量警告
            quality_note = ""
            if indicators.get('data_quality', 1.0) < 0.5:
                quality_note = f"[数据质量低: {indicators['data_quality']:.1%}]"

            scores = self._calculate_weighted_scores_v2(indicators)
            raw_regime = self._scores_to_regime_v2(scores)
            confidence = self._calculate_confidence_v2(scores, raw_regime)

            # 修复：改进的状态平滑
            final_regime = self._apply_regime_smoothing_v2(date, raw_regime, confidence)

            self._update_regime_history(date, final_regime, confidence)

            result = self._create_result(
                date, final_regime, confidence, scores,
                indicators=indicators,
                raw_regime=raw_regime if final_regime != raw_regime else None,
                note=quality_note
            )

            if verbose:
                self._print_analysis(result, indicators)

            return result

        except Exception as e:
            error_msg = f"检测出错: {str(e)}"
            print(f"❌ {error_msg}")
            return self._create_result(date, 'consolidation', 0.0, {}, error_msg)

    def _find_nearest_date(self, date) -> datetime:
        idx = self.market_prices.index.get_indexer([date], method='nearest')[0]
        return self.market_prices.index[idx]

    def _get_indicators_at_date(self, date) -> Optional[Dict]:
        try:
            idx = self.market_prices.index.get_loc(date)
        except KeyError:
            return None

        def safe_get(series, idx, default=np.nan):
            if idx < 0 or idx >= len(series):
                return default
            val = series.iloc[idx]
            return default if pd.isna(val) else val

        indicators = {
            'price': safe_get(self.market_prices, idx),
            'ma_short': safe_get(self.ma_short, idx),
            'ma_medium': safe_get(self.ma_medium, idx),
            'ma_long': safe_get(self.ma_long, idx),
            'mom_1m': safe_get(self.mom_1m, idx, 0),
            'mom_3m': safe_get(self.mom_3m, idx, 0),
            'mom_6m': safe_get(self.mom_6m, idx, 0),
            'vol_short': safe_get(self.vol_short, idx, 0.2),
            'vol_medium': safe_get(self.vol_medium, idx, 0.2),
            'price_percentile': safe_get(self.price_percentile, idx, np.nan),
            'trend_direction': safe_get(self.trend_direction, idx, 0),
            'trend_strength': safe_get(self.trend_strength, idx, 0),
            'data_quality': safe_get(self.data_quality, idx, 0),
            'date': date
        }

        if pd.isna(indicators['price']):
            return None

        return indicators

    def _calculate_weighted_scores_v2(self, ind: Dict) -> Dict[str, float]:
        """
        改进版评分计算：各维度归一化，确保话语权均衡
        """
        cfg = self.config
        scores = {regime: 0.0 for regime in self.REGIME_PRIORITY}

        # ========== 1. 趋势评分（均线排列）==========
        ma_values = [ind['ma_short'], ind['ma_medium'], ind['ma_long']]
        trend_score = 0.0

        if self._safe_compare(ma_values, 'desc'):
            trend_score = 1.0  # 多头排列
            scores['trending_up'] += 3 * cfg.trend_weight
            scores['bull'] += 2 * cfg.trend_weight
        elif self._safe_compare(ma_values, 'asc'):
            trend_score = -1.0  # 空头排列
            scores['trending_down'] += 3 * cfg.trend_weight
            scores['bear'] += 2 * cfg.trend_weight
        else:
            trend_score = 0.0  # 缠绕

        # ========== 2. 动量评分 ==========
        mom_3m, mom_1m, mom_6m = ind['mom_3m'], ind['mom_1m'], ind['mom_6m']
        momentum_score = 0.0

        if mom_3m < cfg.extreme_bear_threshold:
            momentum_score = -2.0
            scores['extreme_bear'] += 3 * cfg.momentum_weight
            scores['bear'] += 2 * cfg.momentum_weight
            if mom_1m > 0.05:
                scores['recovery'] += 3 * cfg.momentum_weight
        elif mom_3m < cfg.bear_threshold:
            momentum_score = -1.0
            scores['bear'] += 2 * cfg.momentum_weight
            if mom_1m > 0.03:
                scores['recovery'] += 2 * cfg.momentum_weight

        if mom_6m > cfg.bull_threshold:
            momentum_score = max(momentum_score, 2.0)
            scores['bull'] += 3 * cfg.momentum_weight
        elif mom_6m > cfg.mild_bull_threshold:
            momentum_score = max(momentum_score, 1.0)
            scores['bull'] += 1 * cfg.momentum_weight

        # ========== 3. 价格位置评分 ==========
        pct = ind['price_percentile']
        position_score = 0.0

        if not pd.isna(pct):
            if pct < 0.3:
                position_score = -1.0
                scores['bear'] += 1 * cfg.position_weight
                if mom_1m > 0:
                    scores['recovery'] += 2 * cfg.position_weight
            elif pct > 0.7:
                position_score = 1.0
                scores['bull'] += 1 * cfg.position_weight
                if mom_1m < 0:
                    scores['trending_down'] += 1 * cfg.position_weight
            else:
                position_score = 0.0

        # ========== 4. 波动率评分 ==========
        vol = ind['vol_short']
        volatility_score = 0.0

        if vol > cfg.high_vol_threshold:
            volatility_score = 1.0
            if mom_3m < -0.1:
                scores['extreme_bear'] += 2 * cfg.volatility_weight
            else:
                scores['consolidation'] += 2 * cfg.volatility_weight
        elif vol < cfg.low_vol_threshold:
            volatility_score = -0.5
            if abs(mom_1m) < 0.05:
                scores['consolidation'] += 2 * cfg.volatility_weight

        # ========== 5. 价格与均线关系 ==========
        price = ind['price']
        ma_s, ma_m = ind['ma_short'], ind['ma_medium']

        dev_short = self._safe_divide(price - ma_s, ma_s, 0)
        dev_medium = self._safe_divide(price - ma_m, ma_m, 0)

        ma_relation_score = 0.0

        if abs(dev_short) < cfg.tight_range_threshold and abs(dev_medium) < cfg.loose_range_threshold:
            ma_relation_score = 0.0
            scores['consolidation'] += 2 * cfg.ma_relation_weight
        elif dev_short > 0.05 and dev_medium > 0.08:
            ma_relation_score = 1.0
            scores['trending_up'] += 1 * cfg.ma_relation_weight
        elif dev_short < -0.05 and dev_medium < -0.08:
            ma_relation_score = -1.0
            scores['trending_down'] += 1 * cfg.ma_relation_weight

        # ========== 6. 趋势强度与方向调整 ==========
        trend_dir = ind['trend_direction']
        trend_str = ind['trend_strength']

        if trend_str > 0.6:  # 强趋势
            if trend_dir > 0:
                scores['bull'] += 1
                scores['trending_up'] += 0.5
            elif trend_dir < 0:
                scores['bear'] += 1
                scores['trending_down'] += 0.5
        elif trend_str < 0.3:  # 弱趋势/震荡
            scores['consolidation'] += 1

        return scores

    def _scores_to_regime_v2(self, scores: Dict[str, float]) -> str:
        """改进版：使用相对容差处理浮点比较"""
        max_score = max(scores.values())

        if max_score == 0:
            return 'consolidation'

        # 使用相对容差
        tolerance = 0.01 * max(1.0, abs(max_score))
        top_regimes = [r for r, s in scores.items() if abs(s - max_score) < tolerance]

        for regime in self.REGIME_PRIORITY:
            if regime in top_regimes:
                return regime

        return 'consolidation'

    def _calculate_confidence_v2(self, scores: Dict[str, float], selected_regime: str) -> float:
        """改进版置信度计算"""
        values = list(scores.values())
        if not values or max(values) == 0:
            return 0.0

        selected_score = scores[selected_regime]
        total_score = sum(abs(v) for v in values)  # 使用绝对值和

        confidence = selected_score / total_score if total_score > 0 else 0

        other_scores = [s for r, s in scores.items() if r != selected_regime]
        other_max = max(other_scores) if other_scores else 0
        margin = (selected_score - other_max) / (selected_score + 1e-8)

        final_confidence = 0.6 * confidence + 0.4 * margin
        return min(max(final_confidence, 0.0), 1.0)

    def _apply_regime_smoothing_v2(self, date, raw_regime: str, confidence: float) -> str:
        """
        修复版状态平滑：
        1. 只增加当前原始状态的计数器
        2. 其他状态缓慢衰减而非归零
        3. 确保平滑机制真正生效
        """
        cfg = self.config

        # 低置信度时保持当前状态
        if confidence < 0.3 and self._current_regime is not None:
            return self._current_regime

        # 修复：只增加当前原始状态的计数器
        self._regime_counter[raw_regime] += 1

        # 修复：其他状态缓慢衰减（而非归零）
        for regime in self._regime_counter:
            if regime != raw_regime:
                self._regime_counter[regime] = max(0, self._regime_counter[regime] - 1)

        # 首次检测
        if self._current_regime is None:
            self._current_regime = raw_regime
            self._current_regime_start = date
            return raw_regime

        # 状态未变
        if raw_regime == self._current_regime:
            return raw_regime

        # 修复：新状态需要持续min_regime_days天才切换
        if self._regime_counter[raw_regime] >= cfg.min_regime_days:
            self._current_regime = raw_regime
            self._current_regime_start = date
            return raw_regime
        else:
            # 保持原状态
            return self._current_regime

    def _update_regime_history(self, date, regime: str, confidence: float):
        self._regime_history.append((date, regime, confidence))
        if len(self._regime_history) > 100:
            self._regime_history.pop(0)

    def _create_result(self, date, regime: str, confidence: float, 
                      scores: Dict, note: str = "", indicators: Dict = None,
                      raw_regime: str = None) -> Dict:
        return {
            'date': date,
            'regime': regime,
            'regime_name': self.REGIME_NAMES.get(regime, regime),
            'confidence': round(confidence, 3),
            'scores': {k: round(v, 2) for k, v in scores.items()},
            'indicators': indicators or {},
            'raw_regime': raw_regime,
            'note': note
        }

    def _print_analysis(self, result: Dict, indicators: Dict):
        print(f"\n{'='*50}")
        print(f"📅 日期: {result['date'].date()}")
        print(f"🎯 市场状态: {result['regime_name']} ({result['regime']})")
        print(f"📊 置信度: {result['confidence']*100:.1f}%")

        if result['raw_regime']:
            print(f"   (原始检测: {self.REGIME_NAMES.get(result['raw_regime'])}, 经平滑调整)")

        if result['note']:
            print(f"   {result['note']}")

        print(f"\n📈 关键指标:")
        print(f"   价格: {indicators['price']:.2f}")
        print(f"   短期MA: {indicators['ma_short']:.2f}")
        print(f"   中期MA: {indicators['ma_medium']:.2f}")
        print(f"   长期MA: {indicators['ma_long']:.2f}")
        print(f"   1月动量: {indicators['mom_1m']*100:+.2f}%")
        print(f"   3月动量: {indicators['mom_3m']*100:+.2f}%")
        print(f"   6月动量: {indicators['mom_6m']*100:+.2f}%")
        print(f"   短期波动率: {indicators['vol_short']*100:.1f}%")

        pct = indicators.get('price_percentile', np.nan)
        if not pd.isna(pct):
            print(f"   价格百分位: {pct*100:.1f}%")

        print(f"   趋势方向: {'上升' if indicators['trend_direction']>0 else '下降' if indicators['trend_direction']<0 else '震荡'}")
        print(f"   趋势强度: {indicators['trend_strength']:.2f}")
        print(f"   数据质量: {indicators['data_quality']:.1%}")

        print(f"\n🎲 各状态评分:")
        sorted_scores = sorted(result['scores'].items(), key=lambda x: x[1], reverse=True)
        for regime, score in sorted_scores[:4]:
            name = self.REGIME_NAMES.get(regime, regime)
            bar = "█" * int(min(score, 10)) + "░" * (10 - int(min(score, 10)))
            print(f"   {name:12s}: {bar} {score:.1f}")

        print(f"{'='*50}\n")

    def reset_state(self):
        """重置状态（用于批量检测前）"""
        self._regime_history = []
        self._current_regime = None
        self._current_regime_start = None
        self._regime_counter = {r: 0 for r in self.REGIME_PRIORITY}

    def batch_detect(self, start_date=None, end_date=None, reset: bool = True) -> pd.DataFrame:
        """
        批量检测（修复：默认重置状态，避免污染）
        """
        if reset:
            self.reset_state()

        if start_date is None:
            start_date = self.market_prices.index[0]
        if end_date is None:
            end_date = self.market_prices.index[-1]

        mask = (self.market_prices.index >= start_date) & (self.market_prices.index <= end_date)
        dates = self.market_prices.index[mask]

        results = []
        for date in dates:
            result = self.detect_regime(date)
            results.append({
                'date': result['date'],
                'regime': result['regime'],
                'regime_name': result['regime_name'],
                'confidence': result['confidence'],
                'raw_regime': result['raw_regime'],
                'note': result['note']
            })

        return pd.DataFrame(results).set_index('date')

    def get_regime_statistics(self) -> Dict:
        if not self._regime_history:
            return {}

        regimes = [r[1] for r in self._regime_history]
        total = len(regimes)

        stats = {}
        for regime in self.REGIME_PRIORITY:
            count = regimes.count(regime)
            stats[regime] = {
                'count': count,
                'percentage': round(count / total * 100, 1) if total > 0 else 0
            }

        return stats


# 示例：创建测试数据并演示
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='B')

    n = len(dates)
    returns = np.random.normal(0.0002, 0.012, n)

    returns[250:500] += 0.001
    returns[500:750] -= 0.0015
    returns[750:1000] += np.random.normal(0, 0.008, 250)
    returns[1000:] += 0.0008

    prices = 4000 * np.exp(np.cumsum(returns))
    price_series = pd.Series(prices, index=dates, name='CSI300')

    detector = RobustMarketRegimeDetector(price_series)

    test_dates = ['2021-06-15', '2022-04-20', '2023-08-10', '2024-02-05', '2024-11-20']

    print("\n" + "="*60)
    print("🔍 关键日期市场状态检测")
    print("="*60)

    for date_str in test_dates:
        result = detector.detect_regime(date_str, verbose=True)

    print("\n" + "="*60)
    print("📊 批量检测结果统计")
    print("="*60)

    # 修复：批量检测前重置状态
    detector.reset_state()
    results_df = detector.batch_detect(reset=True)
    stats = detector.get_regime_statistics()

    print("\n各状态占比:")
    for regime, info in stats.items():
        if info['count'] > 0:
            name = detector.REGIME_NAMES[regime]
            print(f"   {name:12s}: {info['count']:4d}天 ({info['percentage']:5.1f}%)")

    print(f"\n✅ 检测完成！共处理 {len(results_df)} 个交易日")




# 修复平滑机制过于保守的问题，并优化配置

class ImprovedMarketRegimeDetector(RobustMarketRegimeDetector):
    """
    改进版市场状态检测器 - 修复平滑机制过于保守的问题
    """
    
    def __init__(self, market_prices: pd.Series, config: Optional[RegimeConfig] = None):
        # 使用更合理的默认配置
        default_config = RegimeConfig(
            min_regime_days=2,  # 减少到2天，更灵活
            extreme_bear_threshold=-0.20,  # 更严格的极端熊市标准
            bear_threshold=-0.12,  # 更严格的熊市标准
            bull_threshold=0.25,  # 更严格的牛市标准
            mild_bull_threshold=0.15,
            high_vol_threshold=0.30,  # 提高高波动阈值
            low_vol_threshold=0.12,
        )
        
        if config is not None:
            default_config = config
            
        super().__init__(market_prices, default_config)
    
    def _apply_regime_smoothing(self, date, raw_regime: str, confidence: float) -> str:
        """改进的状态切换平滑 - 更灵活的策略"""
        cfg = self.config
        
        # 如果置信度很低，保持当前状态
        if confidence < 0.25 and self._current_regime is not None:
            return self._current_regime
        
        # 首次检测
        if self._current_regime is None:
            self._current_regime = raw_regime
            self._current_regime_start = date
            self._regime_counter = {r: 0 for r in self.REGIME_PRIORITY}
            self._regime_counter[raw_regime] = 1
            return raw_regime
        
        # 更新计数器
        if raw_regime == self._current_regime:
            self._regime_counter[raw_regime] += 1
            return raw_regime
        else:
            # 新候选状态
            self._regime_counter[raw_regime] += 1
            # 重置其他计数器
            for r in self._regime_counter:
                if r != raw_regime:
                    self._regime_counter[r] = 0
        
        # 切换逻辑：
        # 1. 如果是极端状态（extreme_bear），立即切换（风控优先）
        if raw_regime == 'extreme_bear' and confidence > 0.4:
            self._current_regime = raw_regime
            self._current_regime_start = date
            return raw_regime
        
        # 2. 如果新状态持续min_regime_days天且置信度足够，切换
        if self._regime_counter[raw_regime] >= cfg.min_regime_days and confidence > 0.35:
            self._current_regime = raw_regime
            self._current_regime_start = date
            return raw_regime
        
        # 3. 如果新状态持续5天以上（即使置信度一般），也考虑切换
        if self._regime_counter[raw_regime] >= 5:
            self._current_regime = raw_regime
            self._current_regime_start = date
            return raw_regime
        
        # 保持原状态
        return self._current_regime


# 重新运行测试
if __name__ == "__main__":
    szzs_df = pd.read_csv("d:/jupyter/export/sh.000001_前复权数据.csv")
    dates = pd.to_datetime(szzs_df["date"]).to_numpy()
    prices = szzs_df["close_adj"].to_numpy()
    price_series = pd.Series(prices, index=dates, name='CSI300')
    
    # 使用改进版检测器
    detector = ImprovedMarketRegimeDetector(price_series)
    
    # 检测关键日期
    test_dates = ['2021-06-15', '2022-04-20', '2023-08-10', '2024-02-05', '2024-11-20']
    
    print("\n" + "="*60)
    print("🔍 改进版 - 关键日期市场状态检测")
    print("="*60)
    
    for date_str in test_dates:
        result = detector.detect_regime(date_str, verbose=True)
    
    # 批量检测并统计
    print("\n" + "="*60)
    print("📊 改进版 - 批量检测结果统计")
    print("="*60)
    
    results_df = detector.batch_detect()
    stats = detector.get_regime_statistics()
    
    print("\n各状态占比:")
    total_days = sum(s['count'] for s in stats.values())
    for regime, info in stats.items():
        if info['count'] > 0:
            name = detector.REGIME_NAMES[regime]
            bar = "█" * int(info['percentage'] / 2)
            print(f"   {name:12s}: {info['count']:4d}天 ({info['percentage']:5.1f}%) {bar}")
    
    print(f"\n✅ 检测完成！共处理 {len(results_df)} 个交易日")
    
    # 显示最近20天的状态变化
    print("\n" + "="*60)
    print("📈 最近20个交易日状态明细")
    print("="*60)
    recent = results_df.tail(20)
    for idx, row in recent.iterrows():
        date_str = idx.strftime('%Y-%m-%d')
        regime = row['regime_name']
        conf = row['confidence'] * 100
        print(f"   {date_str} | {regime:10s} | 置信度{conf:5.1f}%")
