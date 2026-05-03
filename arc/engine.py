class BSMPricer:
  """
  Black-Scholes-Merton
  """
  @staticmethod
  def price_european(S0: float, K: float, T:float, r:float, sigma: float, opt_type: OptionType) ->float:
    """
    利用1文件中的理论推导，先用BSM方法进行连续情况的类似定价。
    """
    if T<=0 :
      if opt_type == OptionType.CALL:
        return max(0.0, S0 - K)
      else:
        return max(0.0, K - S0)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2)*T)/(sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if opt_type == OptionType.CALL:
      price = S0


import numpy as np
from enum import Enum

class OptionType(Enum):
    CALL = 1
    PUT = -1

class ExerciseStyle(Enum):
    EUROPEAN = 1
    AMERICAN = 2

class PureDiscreteBinomialPricer:
    """
    带有 Delta (对冲比率) 计算模块的离散二叉树定价引擎。
    """
    def __init__(self, K: float, N: int, u: float, d: float, r: float):
        self.K = K
        self.N = N
        self.u = u
        self.d = d
        self.r = r
        
        self.discount = 1.0 / (1.0 + self.r)
        self.p = (1.0 + self.r - self.d) / (self.u - self.d)
        
        if self.p <= 0 or self.p >= 1:
            raise ValueError(f"无套利条件被打破！要求 {d} < {1+r} < {u}，当前 p={self.p:.4f}")

    def price_and_delta(self, current_t: int, current_S: float, opt_type: OptionType, style: ExerciseStyle):
        """
        返回一个元组：(期权价格, 当前节点的对冲比率 Delta)
        """
        if current_t > self.N or current_t < 0:
            raise ValueError(f"当前时间点 {current_t} 无效，必须在 0 到 {self.N} 之间。")
            
        remaining_steps = self.N - current_t
        
        # --- 极端情况：期权已到期，不再需要动态对冲 ---
        if remaining_steps == 0:
            if opt_type == OptionType.CALL:
                val = max(0.0, current_S - self.K)
                delta = 1.0 if current_S > self.K else 0.0 # 实值看涨 Delta 为 1
                return val, delta
            else:
                val = max(0.0, self.K - current_S)
                delta = -1.0 if current_S < self.K else 0.0 # 实值看跌 Delta 为 -1
                return val, delta

        # --- 1. 生成到期日终端节点 ---
        j = np.arange(0, remaining_steps + 1)
        ST = current_S * (self.u ** j) * (self.d ** (remaining_steps - j))
        
        # --- 2. 计算终端收益 Payoff ---
        if opt_type == OptionType.CALL:
            V = np.maximum(0, ST - self.K)
        elif opt_type == OptionType.PUT:
            V = np.maximum(0, self.K - ST)
        
        # 初始化 delta
        delta = 0.0 
            
        # --- 3. 离散贴现倒推求解 ---
        for i in range(remaining_steps - 1, -1, -1):
            
            # 【新增 Delta 计算板块】
            # 当倒推到最靠近当前时间点的那一层时 (即 i = 0)
            # 此时的 V[1] 就是上涨后的期权价值 V_u，V[0] 是下跌后的期权价值 V_d
            if i == 0:
                S_u = current_S * self.u
                S_d = current_S * self.d
                delta = (V[1] - V[0]) / (S_u - S_d)

            # 继续持有价值
            V_continuation = self.discount * (self.p * V[1:i+2] + (1 - self.p) * V[0:i+1])
            
            if style == ExerciseStyle.EUROPEAN:
                V = V_continuation
            
            elif style == ExerciseStyle.AMERICAN:
                j_current = np.arange(0, i + 1)
                S_current = current_S * (self.u ** j_current) * (self.d ** (i - j_current))
                
                if opt_type == OptionType.CALL:
                    V_intrinsic = np.maximum(0, S_current - self.K)
                else:
                    V_intrinsic = np.maximum(0, self.K - S_current)
                
                V = np.maximum(V_continuation, V_intrinsic)
                
        # 返回期权价格和 Delta
        return V[0], delta


