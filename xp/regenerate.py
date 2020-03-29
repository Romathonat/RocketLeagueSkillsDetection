import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#sns.set()
#sns.palplot(sns.cubehelix_palette())


sns.set(rc={'figure.figsize': (8, 6.5)}, font_scale=1.35)
palette = 'inferno_r'
#YlOrRd
#gist_heat
plt.figure()
df = pd.read_pickle('./wracc_datasets/result')
ax = sns.barplot(x='dataset', y='WRAcc', hue='Algorithm', data=df, palette=palette)
plt.savefig('./regenerate/iterations_boxplot.png')

plt.figure()
df = pd.read_pickle('./ground_truth/result')
red = ['#320656']
ax = sns.lineplot(data=df, x='iterations', y='WRAcc', hue='Algorithm', style='Algorithm', markers=True, palette=red)
plt.savefig('./regenerate/gt.png')

plt.figure()
df = pd.read_pickle('./theta/result')
ax = sns.lineplot(data=df, x='theta', y='WRAcc', hue='Algorithm', style='Algorithm', markers=True, palette=palette)
plt.savefig('./regenerate/over_theta.png')

plt.figure()
df = pd.read_pickle('./top_k/result')
ax = sns.lineplot(data=df, x='top_k', y='WRAcc', hue='Algorithm', style='Algorithm', markers=True, palette=palette)
plt.savefig('./regenerate/over_top_k.png')

plt.figure()
df = pd.read_pickle('./space_size/result')
ax = sns.lineplot(data=df, x='size', y='WRAcc', hue='Algorithm', style='Algorithm', markers=True, palette=palette)
ax.set(xlabel='Length max', ylabel='WRAcc')
plt.savefig('./regenerate/quality_vs_size.png')

plt.figure()
df = pd.read_pickle('./number_iterations_optima/result')
ax = sns.lineplot(data=df, x='iterations', y='cost', hue='dataset_name', style='dataset_name', dashes=False, markers=True, palette=palette)
plt.savefig('./regenerate/it_optima.png')

plt.figure()
df = pd.read_pickle('./lengths/result')
ax = sns.boxplot(x='dataset', y='Length', hue='Algorithm', data=df, palette=palette)
plt.savefig('./regenerate/boxplot.png')


sns.set(rc={'figure.figsize': (8, 6.5)}, font_scale=1.6)

plt.figure()
df = pd.read_pickle('./iterations_ucb/resultskating')
ax = sns.lineplot(data=df, x='iterations', y='WRAcc', hue='Algorithm', style='Algorithm', markers=True, palette=palette)
plt.savefig('./regenerate/over_iter_skating.png')

plt.figure()
df = pd.read_pickle('./iterations_ucb/resultsc2')
ax = sns.lineplot(data=df, x='iterations', y='WRAcc', hue='Algorithm', style='Algorithm', markers=True, palette=palette)
plt.savefig('./regenerate/over_iter_sc2.png')
#
plt.figure()
df = pd.read_pickle('./iterations_ucb/resultpromoters')
ax = sns.lineplot(data=df, x='iterations', y='WRAcc', hue='Algorithm', style='Algorithm', markers=True, palette=palette)
plt.savefig('./regenerate/over_iter_promoters.png')

# vs iterations

#plt.show()
