import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class Preprocess:
    def __init__(
        self,
        ruonia_path,
        sofr_path,
        fx_path,
        date_col,
        ruonia_col,
        sofr_col,
        fx_col,
        valuation_date,
        rub_cir_start,
        corr_start,
        dt,
        n_sims,
        horizon_years
    ):
        self.ruonia_path = ruonia_path
        self.sofr_path = sofr_path
        self.fx_path = fx_path

        # параметры моделирования
        self.valuation_date = valuation_date
        self.rub_cir_start = rub_cir_start
        self.corr_start = corr_start
        self.dt = dt
        self.n_sims = n_sims
        self.horizon_years = horizon_years

        # хранилища
        self.series = {}
        self.start_values = {}  # скаляры на дату оценки
        self.params = {}        # параметры моделей (CIR/FX/corr)
        self.simulation = {}    # результаты симуляций

        self.date_col = date_col
        self.ruonia_col = ruonia_col
        self.sofr_col = sofr_col
        self.fx_col = fx_col

    # ЗАГРУЗКА ДАННЫХ

    def _load_ruonia(self):
        df = pd.read_excel(self.ruonia_path)

        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df[[self.date_col, self.ruonia_col]].dropna()
        df = df.rename(columns={self.ruonia_col: "rub"})
        df["rub"] = df["rub"] / 100.0

        df = df.set_index(self.date_col).sort_index()
        return df

    def _load_sofr(self):
        df = pd.read_csv(self.sofr_path)

        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df[[self.date_col, self.sofr_col]].dropna()
        df = df.rename(columns={self.sofr_col: "sofr"})
        df["sofr"] = df["sofr"] / 100.0

        df = df.set_index(self.date_col).sort_index()
        return df

    def _load_fx(self):
        df = pd.read_excel(self.fx_path)

        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df[[self.date_col, self.fx_col]].dropna()
        df = df.rename(columns={self.fx_col: "fx"})

        df = df.set_index(self.date_col).sort_index()
        return df

    def load_data(self):
        """Загрузка сырых временных рядов и подготовка их для калибровки/корреляций."""
        ru = self._load_ruonia()
        so = self._load_sofr()
        fx = self._load_fx()

        # длинная RUONIA для RUB CIR
        ru_long = ru[(ru.index >= self.rub_cir_start) &
                     (ru.index <= self.valuation_date)]

        # общий период для корреляций и калибровки USD CIR / FX
        joint = pd.concat(
            {
                "rub": ru["rub"],
                "sofr": so["sofr"],
                "fx": fx["fx"],
            },
            axis=1,
            join="inner",
        )
        joint = joint[(joint.index >= self.corr_start) &
                      (joint.index <= self.valuation_date)]

        # приращения для корреляций
        df_inc = pd.DataFrame(index=joint.index)
        df_inc["dr_rub"] = joint["rub"].diff()
        df_inc["dr_usd"] = joint["sofr"].diff()
        df_inc["dlog_fx"] = np.log(joint["fx"]).diff()
        df_inc = df_inc.dropna()

        # стартовые значения на дату оценки
        date = pd.to_datetime(self.valuation_date)
        r_rub_0 = float(ru.loc[date, "rub"])
        r_usd_0 = float(so.loc[date, "sofr"])
        S_0 = float(fx.loc[date, "fx"])

        self.series["ruonia_long"] = ru_long["rub"]
        self.series["joint"] = joint
        self.series["increments"] = df_inc

        self.start_values = {
            "r_rub_0": r_rub_0,
            "r_usd_0": r_usd_0,
            "S_0": S_0,
        }

    # БРОУНОВСКОЕ ДВИЖЕНИЕ

    def brownian_motion(self, T):
        return np.random.normal(0.0, 1.0, T)

    # CIR: ГЕНЕРАЦИЯ

    def _generate_CIR_process(self, dW: np.ndarray, kappa, theta, sigma, r_0):
        """
        Эйлеровская дискретизация CIR:
          r_{t+dt} = r_t + kappa*(theta - r_t)*dt + sigma*sqrt(r_t)*sqrt(dt)*dW
        """
        T = len(dW)
        dt = self.dt

        r_t = np.empty(T + 1)
        r_t[0] = r_0
        for t in range(1, T + 1):
            rt_prev = max(r_t[t - 1], 0.0)  #чтобы не было отрицательных
            dr_t = (
                kappa * (theta - rt_prev) * dt
                + sigma * np.sqrt(max(rt_prev, 0.0)) * np.sqrt(dt) * dW[t - 1]
            )
            r_t[t] = rt_prev + dr_t

        return r_t

    # CIR: КАЛИБРОВКА + БУТСТРЭП

    def _calibrate_cir(self, r_series, n_bootstrap=500):
        r = r_series.dropna().astype(float).values

        r_0 = r[0]

        dt = self.dt

        def calib_params(r_arr):
            r_arr = np.asarray(r_arr, dtype=float)
            # 1) убираем NaN
            r_arr = r_arr[~np.isnan(r_arr)]

            # 2) не даём ставкам быть <= 0
            r_arr_clipped = np.clip(r_arr, 1e-8, None)
            r_sqrt = np.sqrt(r_arr_clipped[:-1])

            # y = delta_r / sqrt(r_t)
            y = np.diff(r_arr) / r_sqrt

            # X = [1/sqrt(r_t), sqrt(r_t)]
            x1 = 1.0 / r_sqrt
            x2 = r_sqrt
            X = np.column_stack([x1, x2])

            reg = LinearRegression(fit_intercept=False)
            reg.fit(X, y)

            beta1, beta2 = reg.coef_  # y ≈ beta1 * x1 + beta2 * x2

            # beta1 = kappa*theta*dt
            # beta2 = -kappa*dt
            kappa_dt = -beta2
            theta = beta1 / kappa_dt

            kappa = kappa_dt / dt

            y_hat = reg.predict(X)
            eps = y - y_hat  # sigma * sqrt(dt) * N(0,1)
            c = np.std(eps, ddof=1)
            sigma = c / np.sqrt(dt)

            return kappa, theta, sigma

        # базовая оценка по реальному ряду
        kappa, theta, sigma = calib_params(r)

        kappa_list = [kappa]
        theta_list = [theta]
        sigma_list = [sigma]

        T_steps = len(r) - 1  # число шагов между наблюдениями

        for _ in range(n_bootstrap):
            dW = self.brownian_motion(T_steps)
            path = self._generate_CIR_process(dW, kappa, theta, sigma, r_0)
            path = np.clip(path, 1e-8, None)
            k_b, th_b, sig_b = calib_params(path)

            kappa_list.append(k_b)
            theta_list.append(th_b)
            sigma_list.append(sig_b)

        kappa_arr = np.array(kappa_list)
        theta_arr = np.array(theta_list)
        sigma_arr = np.array(sigma_list)

        kappa_emp_mean = kappa_arr.mean()
        theta_emp_mean = theta_arr.mean()
        sigma_emp_mean = sigma_arr.mean()

        kappa_emp_std = kappa_arr.std(ddof=1)
        theta_emp_std = theta_arr.std(ddof=1)
        sigma_emp_std = sigma_arr.std(ddof=1)

        bias_kappa = kappa_emp_mean - kappa
        bias_theta = theta_emp_mean - theta
        bias_sigma = sigma_emp_mean - sigma

        return {
            "kappa": [kappa, kappa_emp_mean, kappa_emp_std, bias_kappa],
            "theta": [theta, theta_emp_mean, theta_emp_std, bias_theta],
            "sigma": [sigma, sigma_emp_mean, sigma_emp_std, bias_sigma],
        }

    # GBM: ГЕНЕРАЦИЯ

    def _generate_GBM_process(self, dW: np.ndarray, mu, sigma, s_0):
        """
        Эйлеровская дискретизация GBM:
          S_{t+dt} = S_t + mu*S_t*dt + sigma*S_t*sqrt(dt)*dW
        """
        T = len(dW)
        dt = self.dt

        s_t = np.empty(T + 1)
        s_t[0] = s_0

        for t in range(1, T + 1):
            s_prev = s_t[t - 1]
            ds_t = mu * s_prev * dt + sigma * s_prev * np.sqrt(dt) * dW[t - 1]
            s_t[t] = s_prev + ds_t

        return s_t

    # FX (GBM): КАЛИБРОВКА + БУТСТРЭП

    def _calibrate_fx(self, S_series, n_bootstrap=500):
        s = S_series.dropna().astype(float)
        s = s[s > 0]

        s_0 = float(s.iloc[0])
        dt = self.dt

        def calib_params_fx(series: pd.Series):
            logS = np.log(series)
            X = logS.diff().dropna()
            mean_X = X.mean()
            var_X = X.var(ddof=1)

            sigma_hat = np.sqrt(var_X / dt)
            mu_hat = mean_X / dt + 0.5 * sigma_hat**2
            return mu_hat, sigma_hat

        mu, sigma = calib_params_fx(s)

        mu_list = [mu]
        sigma_list = [sigma]

        T_steps = len(s) - 1

        for _ in range(n_bootstrap):
            dW = self.brownian_motion(T_steps)
            path = self._generate_GBM_process(dW, mu, sigma, s_0)
            path_series = pd.Series(path)
            mu_b, sigma_b = calib_params_fx(path_series)

            mu_list.append(mu_b)
            sigma_list.append(sigma_b)

        mu_arr = np.array(mu_list)
        sigma_arr = np.array(sigma_list)

        mu_emp_mean = mu_arr.mean()
        sigma_emp_mean = sigma_arr.mean()

        mu_emp_std = mu_arr.std(ddof=1)
        sigma_emp_std = sigma_arr.std(ddof=1)

        bias_mu = mu_emp_mean - mu
        bias_sigma = sigma_emp_mean - sigma

        return {
            "mu": [mu, mu_emp_mean, mu_emp_std, bias_mu],
            "sigma": [sigma, sigma_emp_mean, sigma_emp_std, bias_sigma],
        }

    # ОБЩАЯ КАЛИБРОВКА

    def calibrate(self):
        ru_long = self.series["ruonia_long"]
        joint = self.series["joint"]
        df_inc = self.series["increments"]

        params_rub = self._calibrate_cir(ru_long)
        params_usd = self._calibrate_cir(joint["sofr"])
        params_fx = self._calibrate_fx(joint["fx"])

        corr = df_inc[["dr_rub", "dr_usd", "dlog_fx"]].corr().values

        self.params["cir_rub"] = params_rub
        self.params["cir_usd"] = params_usd
        self.params["fx"] = params_fx
        self.params["corr_matrix"] = corr

    
    def simulate(self):
        """
        Совместная симуляция траекторий r_RUB(t), r_USD(t), S(t) на горизонте horizon_years
        с шагом dt и учётом корреляционной матрицы.
        """

        # 1. Размерности и сетка времени
        dt = self.dt
        n_sims = self.n_sims
        horizon = self.horizon_years

        n_steps = int(horizon / dt)
        t_grid = np.linspace(0.0, horizon, n_steps + 1)

        # 2. Достаём параметры моделей

        # CIR RUB: берём историческую оценку (OLS)
        cir_rub = self.params["cir_rub"]
        kappa_rub = float(cir_rub["kappa"][0])
        theta_rub = float(cir_rub["theta"][0])
        sigma_rub = float(cir_rub["sigma"][0])

        # CIR USD: берём историческую оценку (OLS)
        cir_usd = self.params["cir_usd"]
        kappa_usd = float(cir_usd["kappa"][0])
        theta_usd = float(cir_usd["theta"][0])
        sigma_usd = float(cir_usd["sigma"][0])

        # FX: берём sigma из калибровки, mu = r_RUB - r_USD
        fx_params = self.params["fx"]
        sigma_fx = float(fx_params["sigma"][0])

        # корреляционная матрица 3x3
        corr = self.params["corr_matrix"]
        L = np.linalg.cholesky(corr)  # LL^T = corr

        # 3. Инициализация массивов траекторий

        r_rub_paths = np.empty((n_sims, n_steps + 1))
        r_usd_paths = np.empty((n_sims, n_steps + 1))
        S_paths = np.empty((n_sims, n_steps + 1))

        r_rub_0 = self.start_values["r_rub_0"]
        r_usd_0 = self.start_values["r_usd_0"]
        S_0 = self.start_values["S_0"]

        r_rub_paths[:, 0] = r_rub_0
        r_usd_paths[:, 0] = r_usd_0
        S_paths[:, 0] = S_0

        # 4. Основной цикл по времени

        for t_idx in range(n_steps):
            # независимые N(0,1)
            Z = np.random.normal(size=(n_sims, 3))
            # коррелированные шоки с корреляцией corr
            eps = Z @ L.T

            eps_rub = eps[:, 0]
            eps_usd = eps[:, 1]
            eps_fx  = eps[:, 2]

            r_rub_prev = r_rub_paths[:, t_idx]
            r_usd_prev = r_usd_paths[:, t_idx]
            S_prev     = S_paths[:, t_idx]

            # обновление RUB CIR
            r_rub_prev_clipped = np.clip(r_rub_prev, 1e-8, None)
            sqrt_rub = np.sqrt(r_rub_prev_clipped)

            dr_rub = (
                kappa_rub * (theta_rub - r_rub_prev) * dt
                + sigma_rub * sqrt_rub * np.sqrt(dt) * eps_rub
            )
            r_rub_new = r_rub_prev + dr_rub
            r_rub_new = np.clip(r_rub_new, 0.0, None)

            # обновление USD CIR
            r_usd_prev_clipped = np.clip(r_usd_prev, 1e-8, None)
            sqrt_usd = np.sqrt(r_usd_prev_clipped)

            dr_usd = (
                kappa_usd * (theta_usd - r_usd_prev) * dt
                + sigma_usd * sqrt_usd * np.sqrt(dt) * eps_usd
            )
            r_usd_new = r_usd_prev + dr_usd
            r_usd_new = np.clip(r_usd_new, 0.0, None)

            # риск-нейтральный дрейф FX
            # mu^Q_t = r_rub_t - r_usd_t (по траекториям на текущем шаге)
            mu_q = r_rub_prev - r_usd_prev

            dS = S_prev * (mu_q * dt + sigma_fx * np.sqrt(dt) * eps_fx)
            S_new = S_prev + dS
            S_new = np.clip(S_new, 1e-8, None)

            # запись нового шага
            r_rub_paths[:, t_idx + 1] = r_rub_new
            r_usd_paths[:, t_idx + 1] = r_usd_new
            S_paths[:, t_idx + 1]     = S_new

        # 5. Сохранение результатов
        self.simulation["t_grid"] = t_grid
        self.simulation["r_rub_paths"] = r_rub_paths
        self.simulation["r_usd_paths"] = r_usd_paths
        self.simulation["S_paths"] = S_paths


    def get_simulation_result(self):
        return {
            "t_grid": self.simulation["t_grid"],
            "S_paths": self.simulation["S_paths"],
            "r_rub_paths": self.simulation["r_rub_paths"],
            "r_usd_paths": self.simulation["r_usd_paths"],
            "params": self.params,
            "start_values": self.start_values,
            "dt": self.dt,
        }
