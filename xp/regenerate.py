import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#sns.set()
#sns.palplot(sns.cubehelix_palette())


sns.set(rc={'figure.figsize': (8, 6.5)}, font_scale=1.7)
sns.set(rc={'figure.figsize': (16, 13)}, font_scale=4.5)

#palette = 'inferno_r'
#YlOrRd
#gist_heat

plt.figure()
df = pd.read_pickle('./quality_over_pattern_number/result')
ax = sns.lineplot(data=df, x='Pattern number', y='Accuracy')
ax.set_ylabel('')
plt.savefig('./regenerate/qual_vs_pattern_number.pdf')

plt.figure()
df = pd.read_pickle('./wracc_vs_alpha/result')
ax = sns.lineplot(data=df, x='alpha', y='Patterns Mean WRAcc')
ax.set_ylabel('')
plt.savefig('./regenerate/wracc_vs_alpha.pdf')

plt.figure()
df = pd.read_pickle('./qual_vs_iter/result')
ax = sns.lineplot(data=df, x='Iteration number', y='Accuracy')
ax.set_ylabel('')
plt.savefig('./regenerate/qual_vs_iter.pdf')

plt.figure()
df = pd.read_pickle('./qual_vs_theta/result')
plt.savefig('./qual_vs_theta/qual_vs_theta.pdf')
ax = sns.lineplot(data=df, x='Theta', y='Accuracy')
ax.set_ylabel('')
plt.savefig('./regenerate/qual_vs_theta.pdf')

plt.figure()
df = pd.read_pickle('./barplot_classif/result')
ax = sns.barplot(x='Classifier', y='Accuracy', data=df)
ax.set_ylabel('')
plt.savefig('./regenerate/barplot_classif.pdf')

#plt.show()
