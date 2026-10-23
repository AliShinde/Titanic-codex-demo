"""Titanic survival EDA. Runs locally or in Google Colab (upload train.csv)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.backends.backend_pdf import PdfPages


def wilson_interval(successes, counts):
    """95% Wilson intervals for descriptive binomial proportions."""
    p = successes / counts
    z = 1.959964
    denominator = 1 + z**2 / counts
    center = (p + z**2 / (2 * counts)) / denominator
    half = z * np.sqrt(p * (1 - p) / counts + z**2 / (4 * counts**2)) / denominator
    return center - half, center + half


def analyze(csv_path='train.csv', output_dir='titanic_survival_outputs'):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    required = {'Age', 'Pclass', 'Survived'}
    if not required.issubset(df.columns):
        raise ValueError('Use train.csv: Age, Pclass and Survived are required.')
    if not df.Survived.isin([0, 1]).all():
        raise ValueError('Survived must contain only 0 and 1, without missing values.')
    if not df.Pclass.isin([1, 2, 3]).all() or (df.Age.dropna() < 0).any():
        raise ValueError('Unexpected passenger class or negative age.')

    labels = ['0–11', '12–17', '18–34', '35–59', '60+']
    df['Age group'] = pd.cut(df.Age, [0, 12, 18, 35, 60, np.inf],
                             labels=labels, right=False)
    by_class = df.groupby('Pclass').Survived.agg(n='size', survivors='sum', rate='mean')
    by_age = df.groupby('Age group', observed=False).Survived.agg(
        n='size', survivors='sum', rate='mean')
    joint = df.groupby(['Age group', 'Pclass'], observed=False).Survived.agg(
        n='size', survivors='sum', rate='mean')
    for name, table in [('by_class', by_class), ('by_age', by_age), ('by_age_and_class', joint)]:
        table.to_csv(output / f'{name}.csv')
    rates = joint.rate.unstack().reindex(index=labels, columns=[1, 2, 3])
    counts = joint.n.unstack().reindex(index=labels, columns=[1, 2, 3])
    total, survivors = len(df), int(df.Survived.sum())
    missing = int(df.Age.isna().sum())

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.titlesize': 13, 'axes.titleweight': 'bold',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': 'white'})
    fig = plt.figure(figsize=(12, 8.6), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.3])
    ax_class, ax_age, ax_joint = (fig.add_subplot(grid[0, 0]),
                                fig.add_subplot(grid[0, 1]),
                                fig.add_subplot(grid[1, :]))
    overall = survivors / total

    def bars(ax, table, names, colors, title, xlabel):
        positions = np.arange(len(table))
        lower, upper = wilson_interval(table.survivors.to_numpy(), table.n.to_numpy())
        ax.bar(positions, table.rate, color=colors, width=0.65)
        ax.errorbar(positions, table.rate,
                    yerr=np.vstack([table.rate - lower, upper - table.rate]),
                    fmt='none', ecolor='#334155', capsize=4, linewidth=1.2)
        for x, (_, row), high in zip(positions, table.iterrows(), upper):
            ax.text(x, high + .025, f'{row.rate:.1%}\n(n={int(row.n)})',
                    ha='center', va='bottom', fontsize=10)
        ax.axhline(overall, color='#64748b', linestyle='--', linewidth=1,
                   label=f'Overall: {overall:.1%}')
        ax.set(xticks=positions, xticklabels=names, ylim=(0, 1),
               title=title, xlabel=xlabel, ylabel='Survival rate')
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
        ax.legend(loc='upper right', fontsize=9, frameon=False)

    bars(ax_class, by_class, ['1st', '2nd', '3rd'],
         ['#0369a1', '#0891b2', '#94a3b8'], 'A  Survival by passenger class', 'Passenger class (Pclass)')
    bars(ax_age, by_age, labels, '#0891b2', 'B  Survival by age', 'Age group (years)')
    heat = ax_joint.imshow(rates.to_numpy(), cmap='YlGnBu', vmin=0, vmax=1, aspect='auto')
    for row in range(len(labels)):
        for col in range(3):
            rate, count = rates.iloc[row, col], int(counts.iloc[row, col])
            text = f'{rate:.1%}\n(n={count})' if count else 'No data'
            ax_joint.text(col, row, text, ha='center', va='center', fontsize=11,
                          color='white' if rate >= .6 else '#0f172a')
    ax_joint.set(xticks=[0, 1, 2], xticklabels=['1st class', '2nd class', '3rd class'],
                 yticks=range(5), yticklabels=labels, xlabel='Passenger class (Pclass)',
                 ylabel='Age group (years)', title='C  Survival by age and passenger class')
    fig.colorbar(heat, ax=ax_joint, label='Survival rate', format=PercentFormatter(1), pad=.02)
    fig.suptitle('Titanic survival: age and passenger class', fontsize=19, weight='bold')
    fig.supxlabel(f'Source: repository train.csv | {total:,} passengers | {survivors} survivors | '
                  f'Age panels exclude {missing} missing ages.\n'
                  'n = passengers in each group; error bars = 95% Wilson intervals. '
                  'Age bins are [0,12), [12,18), [18,35), [35,60), [60,∞).', fontsize=9)
    chart_path = output / 'titanic_survival_chart.png'
    fig.savefig(chart_path, dpi=200)
    fig.savefig(output / 'titanic_survival_chart.svg')

    gap = (by_class.loc[1, 'rate'] - by_class.loc[3, 'rate']) * 100
    report = plt.figure(figsize=(8.27, 11.69), facecolor='white')
    report.text(.07, .95, 'Titanic survival | Exploratory report', fontsize=19, weight='bold')
    report.text(.07, .922, f'{total} passengers • {survivors} survivors • Overall survival {overall:.1%}',
                fontsize=11, color='#475569')
    image_ax = report.add_axes([.04, .35, .92, .55])
    image_ax.imshow(plt.imread(chart_path))
    image_ax.axis('off')
    insight = (
        'Key findings\n\n'
        f'• Class shows a strong survival gradient: 1st {by_class.loc[1,"rate"]:.1%}, '
        f'2nd {by_class.loc[2,"rate"]:.1%}, 3rd {by_class.loc[3,"rate"]:.1%}.\n'
        f'  The first-to-third class gap is {gap:.1f} percentage points.\n\n'
        f'• Children aged 0–11 had {by_age.iloc[0].rate:.1%} survival (n={int(by_age.iloc[0].n)}), '
        f'versus {by_age.iloc[2].rate:.1%}\n'
        f'  for ages 18–34. Ages 60+ had {by_age.iloc[4].rate:.1%} survival (n={int(by_age.iloc[4].n)}). '
        'The age pattern is not monotonic.\n\n'
        '• Among children aged 0–11, survival was 100% in 2nd class (n=17),\n'
        '  but 40.4% in 3rd class (n=47). Age and class should be examined together.\n\n'
        'Interpretation and limits\n'
        f'Age is missing for {missing}/{total} passengers ({missing/total:.1%}); no ages were imputed.\n'
        'Class totals use all passengers; age panels use the 714 recorded ages.\n'
        'Small cells (especially older ages) are uncertain; 100% is an observed sample rate.\n'
        'These are descriptive associations, not causal effects. Sex and other factors\n'
        'may confound the patterns. Only train.csv has survival labels; test.csv is excluded.'
    )
    report.text(.07, .355, insight, fontsize=10, va='top', linespacing=1.35)
    report.text(.07, .035, 'Method: grouped observed survival proportions; 95% Wilson intervals on bars.',
                fontsize=8, color='#475569')
    with PdfPages(output / 'titanic_survival_report.pdf') as pdf:
        pdf.savefig(report)
    plt.close(report)
    print(f'Overall survival: {overall:.1%}; missing ages: {missing}')
    print(by_class.to_string())
    print(by_age.to_string())
    print(f'Chart, report and summary tables saved to: {output.resolve()}')
    return fig


if __name__ == '__main__':
    data_path = Path('train.csv')
    if not data_path.exists():
        try:
            from google.colab import files
        except ImportError as exc:
            raise FileNotFoundError('Place train.csv in the current directory.') from exc
        print('Upload the repository train.csv (not test.csv).')
        files.upload()
        if not data_path.exists():
            raise FileNotFoundError('Please upload a file named train.csv.')
    analyze(data_path)
    plt.show()
