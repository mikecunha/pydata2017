import matplotlib.pyplot as plt
import seaborn as sns


def plot_test(frame, title):
    """Scatter plot of effect size vs. p-value by embedding."""

    sns.set_context("notebook", font_scale=1.45)
    sns.set_style('ticks')
    splot = sns.lmplot(x='p_val', y='effect_size', data=frame,
                       fit_reg=False,
                       size=6,
                       aspect=2,
                       legend=False,
                       hue="embedding",
                       scatter_kws={"marker": "D",
                                    "s": 100})

    # Formatting
    plt.title(title)
    plt.xlabel('P-value')
    plt.ylabel('Effect Size')

    for i, txt in enumerate(frame.embedding):
        splot.ax.annotate('  '+txt, (frame.p_val.iat[i],
                          frame.effect_size.iat[i]))
