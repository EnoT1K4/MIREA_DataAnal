import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate, stats
from scipy.stats import norm, normaltest, f, chi2
import seaborn as sns
# Generate 30 realizations of a uniform random variable with 1000 elements
realizations = np.random.uniform(0, 1, size=(1000, 1000))

def get_hist(sample):
    sample = sample.reshape(-1)
    g = 1 + np.floor(np.log2(sample.size))
    sample_hist = pd.DataFrame(pd.cut(sample, bins=int(g)).value_counts().sort_index())
    sample_hist.iat[0, 0] += 1
    sample_hist.rename(columns={0: 'n_i'}, inplace=True)
    sample_hist['p_i'] = sample_hist['n_i'] / sample.size
    sample_borders = np.linspace(sample.min(), sample.max(), int(g) + 1)
    sample_borders = pd.DataFrame({'z_i-1': sample_borders[:-1], 'z_i': sample_borders[1:]})
    sample_hist.loc[:, 'x_(i)'] = ((sample_borders['z_i-1'] + sample_borders['z_i']) / 2).to_numpy()
    return sample_hist, sample_borders

#1.1 histogram
plt.figure()
plt.hist(realizations[0], bins=10, density=True, alpha=0.5, label='Sample')
plt.show()
#1.2 sum histogram
sums = np.sum(realizations, axis=1)
plt.hist(sums, bins=10)
plt.xlabel('Sum')
plt.ylabel('Frequency')
plt.title('Histogram of Sum of Uniform Random Variables')
plt.show()

#1.3 sum histogram plot and lines
mean = np.mean(sums)
std_dev = np.std(sums, ddof=1)

x = np.linspace(min(sums), max(sums), 100)

pdf = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-np.power(x - mean, 2) / (2 * np.power(std_dev, 2)))

plt.hist(sums, bins=10, density=True, alpha=0.5, label='Выборка')
plt.plot(x, pdf, color='red', label='Теоретическая плотность')
plt.xlabel('Значение')
plt.ylabel('Плотность')
plt.title('Гистограмма выборки с теоретической плотностью')
plt.legend()
plt.show()


#1.4 analyze with pirson

def test_normality(data):
    # Step 1: Calculate expected frequencies assuming normal distribution
    mean = np.mean(data)
    std = np.std(data)
    intervals = np.linspace(np.min(data), np.max(data), num=10)
    expected_freq = [len(data) * (stats.norm.cdf(intervals[i+1], loc=mean, scale=std) - stats.norm.cdf(intervals[i], loc=mean, scale=std)) for i in range(len(intervals)-1)]

    # Step 2: Determine observed frequencies
    observed_freq, _ = np.histogram(data, bins=intervals)

    # Step 3: Calculate chi-square statistic
    chi2_statistic = np.sum((observed_freq - expected_freq)**2 / expected_freq)
    
    # Step 4: Calculate degrees of freedom
    df = len(intervals) - 3

    # Step 5: Calculate critical value for desired significance level and df
    critical_value = chi2.ppf(0.95, df)

    # Step 6: Compare chi-square statistic with critical value
    if chi2_statistic > critical_value:
        print("Null hypothesis of normal distribution is rejected")
    else:
        print("Null hypothesis of normal distribution is not rejected")

    # Step 7: Calculate p-value
    p_value = chi2.sf(chi2_statistic, df)

    # Step 8: Compare p-value with significance level
    significance_level = 0.05
    if p_value < significance_level:
        print("Null hypothesis of normal distribution is rejected")
    else:
        print("Null hypothesis of normal distribution is not rejected")
    
print(test_normality(x))
#task 2
def generate_chi_squared_sample(df):
    normal_samples = np.random.standard_normal(df)
    chi_squared_sample = np.sum(normal_samples**2)
    return chi_squared_sample

# Example usage
df = 5  # degrees of freedom
sample = generate_chi_squared_sample(df)
#print(sample)

# Number of random realizations
n = 100

# Mean and standard deviation of the random variables
mu = 0
sigma = 1

# Generate the random realizations
L = np.random.normal(mu, sigma, n)

# Calculate the Z-scores
Z = (L - mu) / sigma

# Square the Z-scores to obtain the new sample
new_sample = Z**2

# Print the new sample
#print(new_sample)

# Set the degrees of freedom for the chi-square distribution
df = 3
# Plot the histogram
plt.hist(new_sample, bins='auto', density=True, alpha=0.7, rwidth=0.85, label='Sample')

# Calculate the theoretical density of the chi-square distribution
x = np.linspace(chi2.ppf(0.01, df), chi2.ppf(0.99, df), 100)
y = chi2.pdf(x, df)

# Plot the theoretical density
plt.plot(x, y, 'r', label='Theoretical Density')

# Customize the plot
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram with Theoretical Density')
plt.legend()

# Show the plot
plt.show()

#task3

def fisher_sample(d1, d2, size):
    # Генерируем две случайные выборки с помощью chi2 распределения
    y1 = np.random.chisquare(d1, size)
    y2 = np.random.chisquare(d2, size)

    # Вычисляем выборку распределения Фишера
    f_sample = (y1 / d1) / (y2 / d2)

    return f_sample

# Указываем степени свободы и размер выборки
d1 = 5
d2 = 10
size = 100

# Генерируем выборку распределения Фишера
sample3 = fisher_sample(d1, d2, size)

# Выводим результат
#print(sample3)

# Plotting the histogram
plt.hist(sample3, bins='auto', alpha=0.7, density=True, label='Sample')

# Generating the x-values for the theoretical density distribution
x = np.linspace(f.ppf(0.001, dfn=2, dfd=5), f.ppf(0.999, dfn=2, dfd=5), 100)

# Plotting the theoretical density distribution
plt.plot(x, f.pdf(x, dfn=2, dfd=5), 'r-', lw=2, label='Theoretical')

# Adding labels and titles
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Histogram with Theoretical Density Distribution')
plt.legend()

# Showing the plot
plt.show()

#task4

np.random.seed(42)

# Generate 2 samples from a t-distribution with 3 degrees of freedom
samples = np.random.standard_t(df=3, size=5)


#task4
k = 5
n = 700
samples_matr = np.random.normal(0, 1, (k, n))
t_sample = np.random.normal(0, 1, n) / np.sqrt((samples_matr ** 2).sum(axis=0) / k)
sns.histplot(t_sample, bins=9)

hist, borders = get_hist(t_sample)

def B(a, b):
    return integrate.quad(lambda x: x ** (a - 1) * (1 - x) ** (b - 1), 0, 1)[0]

def t_pdf(x, n):
    return (1 + x ** 2 / n) ** -((n + 1) / 2) / (np.sqrt(n) * B(0.5, n / 2))

n_i_theor = []
for i in range(hist.shape[0]):
    n_i = integrate.quad(t_pdf, borders['z_i-1'][i], borders['z_i'][i], args=(k))[0] * t_sample.size
    n_i_theor.append(n_i)
n_i_theor = np.array(n_i_theor)

x = np.linspace(t_sample.min(), t_sample.max(), 50)
y = stats.t.pdf(x, k)
fig=plt.figure(figsize=(8,6))
h = (borders['z_i'] - borders['z_i-1'])[0]
plt.bar(x=hist['x_(i)'], height=hist['p_i'] / h, width = h)
sns.lineplot(x=x, y=y, color='#FF8400')

plt.figure(figsize=(8, 6))
sns.barplot(x=hist.index, y=hist['n_i'], color='#0267C1', label='Гистограмма полученной выборки')
sns.barplot(x=hist.index, y=n_i_theor,
    linewidth=2, edgecolor="#FF8400", facecolor=(0, 0, 0, 0), label='Гистограмма теоретического распределения'
)

plt.legend()
plt.xlabel('$[z_{i-1}; z_{i}]$', loc='right')
plt.ylabel('$n_{i}$', loc='top')
plt.xticks(rotation=45)
plt.show()


x = np.linspace(t_sample.min(), t_sample.max(), 50)
y = stats.t.pdf(x, k)
fig=plt.figure(figsize=(8,6))
h = (borders['z_i'] - borders['z_i-1'])[0]
plt.bar(x=hist['x_(i)'], height=hist['p_i'] / h, width = h)
sns.lineplot(x=x, y=y, color='#FF8400')

plt.figure(figsize=(8, 6))
sns.barplot(x=hist.index, y=hist['n_i'], color='#0267C1', label='Гистограмма полученной выборки')
sns.barplot(x=hist.index, y=n_i_theor,
    linewidth=2, edgecolor="#FF8400", facecolor=(0, 0, 0, 0), label='Гистограмма теоретического распределения'
)

plt.legend()
plt.xlabel('$[z_{i-1}; z_{i}]$', loc='right')
plt.ylabel('$n_{i}$', loc='top')
plt.xticks(rotation=45)
plt.show()

x = np.linspace(t_sample.min(), t_sample.max(), 50)
y = stats.t.pdf(x, k)
fig=plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
sns.histplot(x=t_sample, ax=ax1)
sns.lineplot(x=x, y=y, color='#FF8400', ax=ax2)

plt.figure(figsize=(8, 6))
sns.barplot(x=hist.index, y=hist['n_i'], color='#0267C1', label='Гистограмма полученной выборки')
sns.barplot(x=hist.index, y=n_i_theor,
    linewidth=2, edgecolor="#FF8400", facecolor=(0, 0, 0, 0), label='Гистограмма теоретического распределения')

plt.legend()
plt.xlabel('$[z_{i-1}; z_{i}]$', loc='right')
plt.ylabel('$n_{i}$', loc='top')
plt.xticks(rotation=45)
plt.show()

chi = ((hist['n_i'] - n_i_theor) ** 2 / n_i_theor).sum()
crit_val = stats.chi2.ppf(df=hist.shape[0] - 1 - 1, q=0.95)
print(f'Chi square: {chi.round(3)}, crit: {crit_val.round(3)}'
      f'\nChi square < crit: {chi < crit_val}')