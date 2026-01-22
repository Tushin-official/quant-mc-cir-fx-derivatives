import numpy as np


class RangeAccrualPricer:
    """
    Прайсер Range Accrual.

    Параметры:
      - L: нижняя граница диапазона (может быть None => -inf)
      - U: верхняя граница диапазона (может быть None => +inf)
      - notional: номинал сделки

    Основной метод:
      price_from_paths(sim_result, obs_indices=None)
    """

    def __init__(self, L=None, U=None, notional=None):
        self.L = L
        self.U = U
        self.notional = notional

    def price_from_paths(self, sim_result):
        """
        sim_result — словарь, который возвращает Preprocess:

        {
          "t_grid",
          "S_paths",
          "r_rub_paths",
          "dt",
          ...
        }
        """
        S_paths = sim_result["S_paths"]
        n_sims = S_paths.shape[0]

        # индикатор нахождения в диапазоне
        L = self.L if self.L is not None else -np.inf
        U = self.U if self.U is not None else np.inf

        inside = (S_paths >= L) & (S_paths <= U)

        # доля времени в диапазоне по каждому сценарию
        w = inside.mean(axis=1)

        # payoff без дисконтирования
        payoff = self.notional * w

        price = payoff.mean()
        stderr = payoff.std(ddof=1) / np.sqrt(n_sims)
        return {
            "price": price,
            "stderr": stderr,
            "w_paths": w,
            "payoff_paths": payoff
        }
