

def estimate_corr_matrix(
                self,
                loader: MarketDataLoader,
                start,
                end
            ):
        """
        Возвращает 3x3 матрицу корреляций между:
        (dr_rub, dr_usd, dlog_fx)
        в виде numpy.ndarray.
        """

        df_inc = loader.get_increments_for_correlations(start=start, end=end)
        corr = df_inc[["dr_rub", "dr_usd", "dlog_fx"]].corr()
        return corr.values