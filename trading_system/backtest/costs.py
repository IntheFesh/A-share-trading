"""成本六层。Phase 2(任务 2.1)。对应 v3.1 第四章。价格层:基于 raw 成交金额(INV-2)。

c_round = c_tax + c_exchange + c_commission + c_mincomm + c_spread + c_impact + c_failure
已核验官方下限:c_official = 印花税(卖 0.05%) + 经手费(双向 2×0.00341%) = 5.682 bp。
过户费、佣金率、最低佣金、滑点等为待核/可配置,从 config/costs.yaml 读。
最低佣金闸门 c_mincomm(Q)=min_per_side/Q;最小订单金额 Q>=q_min_amount 否则跳过该笔。
"""

from __future__ import annotations

from dataclasses import dataclass

# 已核验定值(勿改;来源见 v3.1 第四章)
STAMP_TAX_SELL = 0.0005          # 印花税:卖出单边 0.05%
EXCHANGE_FEE_ONEWAY = 0.0000341  # 经手费:单边 0.00341%
VERIFIED_FLOOR_BP = (STAMP_TAX_SELL + 2 * EXCHANGE_FEE_ONEWAY) * 1e4  # = 5.682 bp


@dataclass
class CostModel:
    """成本六层模型。费率用小数,金额用元,滑点/价差/冲击/失败用 bp。"""

    stamp_tax_sell: float = STAMP_TAX_SELL
    exchange_fee_oneway: float = EXCHANGE_FEE_ONEWAY
    transfer_fee_oneway: float = 0.00001     # 过户费(待核)
    commission_rate_oneway: float = 0.0003   # 佣金率(待核)
    min_commission_per_side: float = 5.0     # 最低佣金/边(元)
    q_min_amount_yuan: float = 16667.0       # 最小订单金额闸门
    slippage_bp: float = 10.0
    half_spread_bp: float = 0.0
    impact_bp: float = 0.0
    failure_bp: float = 0.0

    @classmethod
    def from_config(cls, cfg: dict) -> "CostModel":
        """从 config/costs.yaml(已加载为 dict)构造。"""
        vf = cfg.get("verified_floor", {})
        tf = cfg.get("transfer_fee", {})
        cm = cfg.get("commission", {})
        sl = cfg.get("slippage", {})
        sp = cfg.get("spread", {})
        im = cfg.get("impact", {})
        fa = cfg.get("failure", {})
        return cls(
            stamp_tax_sell=vf.get("stamp_tax_sell", STAMP_TAX_SELL),
            exchange_fee_oneway=vf.get("exchange_fee_oneway", EXCHANGE_FEE_ONEWAY),
            transfer_fee_oneway=tf.get("theta_transfer_oneway", 0.00001),
            commission_rate_oneway=cm.get("rate_oneway", 0.0003),
            min_commission_per_side=cm.get("min_per_side_yuan", 5.0),
            q_min_amount_yuan=cm.get("q_min_amount_yuan", 16667.0),
            slippage_bp=sl.get("default_bp", 10.0),
            half_spread_bp=sp.get("half_spread_bp", 0.0),
            impact_bp=im.get("cost_bp", 0.0) if isinstance(im.get("cost_bp"), (int, float)) else 0.0,
            failure_bp=fa.get("cost_bp", 0.0),
        )

    def commission_side(self, notional: float) -> float:
        """单边佣金(含最低佣金闸门),返回元。"""
        return max(self.commission_rate_oneway * notional, self.min_commission_per_side)

    def passes_min_order(self, notional: float) -> bool:
        """是否满足最小订单金额(不为摊薄佣金加仓)。"""
        return notional >= self.q_min_amount_yuan

    def round_trip_cost_fraction(self, notional: float, *, slippage_bp: "float | None" = None) -> float:
        """一笔往返(买+卖)的总成本占名义金额的比例。

        费率层:买边(经手+过户) + 卖边(经手+过户+印花) + 双边佣金(含最低佣金);
        变动层:滑点 + 半价差 + 冲击(每边各一次,往返×2) + 失败成本(一次)。
        """
        if notional <= 0:
            raise ValueError("notional 必须 > 0")
        rate_buy = self.exchange_fee_oneway + self.transfer_fee_oneway
        rate_sell = self.exchange_fee_oneway + self.transfer_fee_oneway + self.stamp_tax_sell
        comm = self.commission_side(notional) * 2.0
        abs_fee = (rate_buy + rate_sell) * notional + comm
        sl = (self.slippage_bp if slippage_bp is None else slippage_bp) * 1e-4
        var_cost = (sl + self.half_spread_bp * 1e-4 + self.impact_bp * 1e-4) * 2.0
        fail = self.failure_bp * 1e-4
        return abs_fee / notional + var_cost + fail
