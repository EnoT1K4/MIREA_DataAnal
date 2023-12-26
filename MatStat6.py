import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import scipy.integrate as integrate
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
import random
import itertools

def set_random_seed(s):
    random.seed(s)
    np.random.seed(s)
SEED = 47
set_random_seed(SEED)
ts_sin = pd.read_csv(filepath_or_buffer='./Данные для 6 практики/ИНБО-10-21/вариант3/1.txt', header=None, names=['x'])
print(ts_sin.head())
sns.lineplot(data=ts_sin)
plt.show()
ts_linear = pd.read_csv(filepath_or_buffer='./Данные для 6 практики/ИНБО-10-21/вариант3/2.txt', header=None, names=['x'])
print(ts_linear.head())
ts_linear.plot()
plt.show()
def ljungbox_test(ts, m=5):
    n = len(ts)
    n_dif_k = np.full(shape=m, fill_value=n, dtype=float) - np.arange(1, m + 1, 1, dtype=float)
    r_k = smt.acf(ts, adjusted=True, nlags=m, qstat=False)[1:]
    return n * (n + 2) * (r_k ** 2 / n_dif_k).sum(), stats.chi2.ppf(df=m, q=0.975)
print('SMA')
def SMA(ts, m):
    ts_pad = pd.DataFrame(np.pad(ts, ((m, m), (0, 0)), mode='edge'), columns=['x'])
    return ts_pad.rolling(2 * m + 1, center=True).mean().dropna().reset_index(drop=True)
print(SMA(ts_sin, 1))
print(smt.acf(ts_sin, adjusted=True, nlags=5, qstat=True)[1][-1])
print(ljungbox_test(ts_sin))
print(acorr_ljungbox(ts_sin, lags=5)) # Not adjusted
lags = [3, 5, 7, 9]
print('m\tLjung-Box Test')
for l in lags:
    ts_sin_sma = SMA(ts_sin, l)
    print(f'{l}\t{ljungbox_test(ts_sin - ts_sin_sma)[0].round(3)}')
print('При m=3 статистика Льюнг-Бокса минимальна')
ts_sin_sma = SMA(ts_sin, 3)
sns.lineplot(y=ts_sin['x'], x=ts_sin.index, label='Original series')
sns.lineplot(y=ts_sin_sma['x'], x=ts_sin_sma.index, color='orange', label='SMA-smoothed series, m=3')
plt.legend()
plt.show()
print('Остатки модели на графике')
resid = ts_sin - ts_sin_sma
sns.lineplot(y=resid['x'], x=resid.index, label=r'Остатки: $y-\tilde{y}$')
plt.legend()
plt.show()
d = durbin_watson(resid)
print(d, 'Критерий Дарбина-Уотсона находится между значениями, поэтому принимается нулевая гипотеза $H_{0}$ о том, что отсутствует автокорреляция остатков')
print('________________________________________________________________')
lags = [3, 5, 7, 9]
print('m\tLjung-Box Test')
for l in lags:
    ts_linear_sma = SMA(ts_linear, l)
    print(f'{l}\t{ljungbox_test(ts_linear - ts_linear_sma)[0].round(3)}')
print('При m=9 статистика Льюнг-Бокса минимальна')
ts_linear_sma = SMA(ts_linear, 9)
sns.lineplot(y=ts_linear['x'], x=ts_linear.index, label='Original series')
sns.lineplot(y=ts_linear_sma['x'], x=ts_linear_sma.index, color='orange', label='SMA-smoothed series, m=9')
plt.legend()
plt.show()
resid = ts_linear - ts_linear_sma
sns.lineplot(y=resid['x'], x=resid.index, label=r'Остатки: $y-\tilde{y}$')
plt.legend()
plt.show()
d = durbin_watson(resid)
print(d, 'о наличии и о характере автокорреляции ничего сказать нельзя')
def WMA(ts, m, eps=0.3):
    e_i = np.exp(-eps * np.abs(np.arange(-m, m + 1, 1)))
    w_i = e_i / e_i.sum()
    ts_pad = pd.DataFrame(np.pad(ts, ((m, m), (0, 0)), mode='edge'), columns=['x'])
    return ts_pad.rolling(2 * m + 1, center=True).apply(lambda x: (x * w_i).sum()).dropna().reset_index(drop=True)
print('\n\nWMA\n\n') 
lags = [3, 5, 7, 9]
print('m\tLjung-Box Test')
for l in lags:
    ts_sin_wma = WMA(ts=ts_sin, m=l)
    print(f'{l}\t{ljungbox_test(ts_sin - ts_sin_wma)[0].round(3)}')
print('При m=3 статистика Льюнг-Бокса минимальна')
ts_sin_wma = WMA(ts=ts_sin, m=3)
sns.lineplot(y=ts_sin['x'], x=ts_sin.index, label='Original series')
sns.lineplot(y=ts_sin_wma['x'], x=ts_sin_wma.index, color='orange', label='WMA-smoothed series, m=3')
sns.lineplot(y=ts_sin_sma['x'], x=ts_sin_sma.index, color='green', label='SMA-smoothed series, m=3')
plt.legend()
plt.show()
resid = ts_sin - ts_sin_wma
sns.lineplot(y=resid['x'], x=resid.index, label=r'Residuals: $y-\tilde{y}$')
plt.legend()
plt.show()
d = durbin_watson(resid)
print(d,'принимается нулевая гипотеза H_0 о том, что отсутствует автокорреляция остатков' ) 
print('--------------------')
lags = [3, 5, 7, 9]
print('m\tLjung-Box Test')
for l in lags:
    ts_linear_wma = WMA(ts=ts_linear, m=l)
    print(f'{l}\t{ljungbox_test(ts_linear - ts_linear_wma)[0].round(3)}')
print('При m=9 статистика Льюнг-Бокса минимальна')
ts_linear_wma = WMA(ts=ts_linear, m=9)
sns.lineplot(y=ts_linear['x'], x=ts_linear.index, label='Original series')
sns.lineplot(y=ts_linear_wma['x'], x=ts_linear_wma.index, color='orange', label='WMA-smoothed series, m=9')
plt.legend()
plt.show()
resid = ts_linear - ts_linear_wma
sns.lineplot(y=resid['x'], x=resid.index, label=r'Residuals: $y-\tilde{y}$')
plt.legend()
plt.show()
d = durbin_watson(resid)
print(d, 'принимается альтернативная гипотеза $H_{1}$ о том, что существует отрицательная автокорреляция остатков')
def EMA(ts, a):
    return ts.ewm(alpha=a, adjust=False).mean()
print('\n\nEMA\n\n')
alphas = np.arange(0.1, 1, 0.1, dtype=float)
print('a\tLjung-Box Test')
for a in alphas:
    ts_sin_ema = EMA(ts_sin, a)
    print(f'{a.round(1)}\t{ljungbox_test(ts_sin - ts_sin_ema)[0].round(3)}')
print('При m=0.9 статистика Льюнг-Бокса минимальна')
ts_sin_ema = EMA(ts_sin, 0.9)
sns.lineplot(y=ts_sin['x'], x=ts_sin.index, label='Original series')
sns.lineplot(y=ts_sin_ema['x'], x=ts_sin_ema.index, color='red', label=r'EMA-smoothed series, $\alpha$=0.9')
plt.legend()
plt.show()
resid = ts_sin - ts_sin_ema
sns.lineplot(y=resid['x'], x=resid.index, label=r'Residuals: $y-\tilde{y}$')
plt.legend()
plt.show()
d = durbin_watson(resid)
print(d, 'принимается нулевая гипотеза H0 о том, что отсутствует автокорреляция остатков')
print('_----------------------')
alphas = np.arange(0.1, 1, 0.1, dtype=float)
print('a\tLjung-Box Test')
for a in alphas:
    ts_linear_ema = EMA(ts_linear, a)
    print(f'{a.round(1)}\t{ljungbox_test(ts_linear - ts_linear_ema)[0].round(3)}')
ts_linear_ema = EMA(ts_linear, 0.3)
sns.lineplot(y=ts_linear['x'], x=ts_linear.index, label='Original series')
sns.lineplot(y=ts_linear_ema['x'], x=ts_linear_ema.index, color='red', label=r'EMA-smoothed series, $\alpha$=0.3')
plt.legend()
plt.show()
resid = ts_linear - ts_linear_ema
sns.lineplot(y=resid['x'], x=resid.index, label=r'Residuals: $y-\tilde{y}$')
plt.legend()
plt.show()
d = durbin_watson(resid)
print(d, 'принимается альтернативная гипотеза H_1 о том, что существует положительная автокорреляция остатков')
def DEMA(ts, a, g):
    b_est = np.zeros(ts.shape)
    y_est = np.zeros(ts.shape)
    y_est[0] = ts.iat[0, 0]
    b_est[0] = ts.iat[1, 0] - ts.iat[0, 0]
    for i in range(1, len(ts)):
        y_est[i] = a * ts.iat[i, 0] + (1 - a) * (y_est[i - 1] + b_est[i - 1])
        b_est[i] = g * (y_est[i] - y_est[i - 1]) + (1 - g) * b_est[i - 1]
    return pd.DataFrame(y_est, columns=ts.columns)
print('\n\nDEMA\n\n')
alphas = np.arange(0.1, 1, 0.1, dtype=float)
gammas = np.arange(0.1, 1, 0.1, dtype=float)
print('a\tg\tLjung-Box Test')
minn = 100000000
best_a = 0
best_g = 0
for a, g in itertools.product(alphas, gammas):
    ts_sin_dema = DEMA(ts_sin, a, g)
    print(f'{a.round(1)}\t{g.round(1)}\t{ljungbox_test(ts_sin - ts_sin_dema)[0].round(3)}')
    if ljungbox_test(ts_sin - ts_sin_dema)[0].round(3) < minn:
        minn = ljungbox_test(ts_sin - ts_sin_dema)[0].round(3)
        best_a = a.round(1)
        best_g = g.round(1)
ts_sin_dema = DEMA(ts_sin, best_a, best_g)
sns.lineplot(y=ts_sin['x'], x=ts_sin.index, label='Original series')
sns.lineplot(y=ts_sin_dema['x'], x=ts_sin_dema.index, color='red', label=r'DEMA-smoothed series, $\alpha$=0.8, $\gamma$=0.9')
plt.legend()
plt.show()
resid = ts_sin - ts_sin_dema
sns.lineplot(y=resid['x'], x=resid.index, label=r'Residuals: $y-\tilde{y}$')
plt.legend()
plt.show()
d = durbin_watson(resid)
print(d, ' принимается нулевая гипотеза H_0 о том, что отсутствует автокорреляция остатков')
alphas = np.arange(0.1, 1, 0.1, dtype=float)
gammas = np.arange(0.1, 1, 0.1, dtype=float)
print('a\tg\tLjung-Box Test')
minn = 1823489237852934
for a, g in itertools.product(alphas, gammas):
    ts_linear_dema = DEMA(ts_linear, a, g)
    print(f'{a.round(1)}\t{g.round(1)}\t{ljungbox_test(ts_linear - ts_linear_dema)[0].round(3)}')
    if ljungbox_test(ts_linear - ts_linear_dema)[0].round(3) < minn:
        minn = ljungbox_test(ts_linear - ts_linear_dema)[0].round(3)
        best_a = a.round(1)
        best_g = g.round(1)
ts_linear_dema = DEMA(ts_linear, best_a, best_g)
sns.lineplot(y=ts_linear['x'], x=ts_linear.index, label='Original series')
sns.lineplot(y=ts_linear_dema['x'], x=ts_linear_dema.index, color='red', label=r'DEMA-smoothed series, $\alpha$=0.4, $\gamma$=0.5')
plt.legend()
plt.show()
resid = ts_linear - ts_linear_dema
sns.lineplot(y=resid['x'], x=resid.index, label=r'Residuals: $y-\tilde{y}$')
plt.legend()
plt.show()
d = durbin_watson(resid)
print(d, 'принимается нулевая гипотеза H_0 о том, что отсутствует автокорреляция остатков')