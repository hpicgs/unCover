# pyright: reportWildcardImportFromLibrary=false
from dominate.tags import *

# from https://colorbrewer2.org/?type=qualitative&scheme=Set3&n=12)
colors = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd','#ccebc5','#ffed6f']
# css for bar
styles = [
    # solid
    'background-color: {color}',
]

# loop through colors for each pattern-style
# then loop through pattern-styles
def _get_style(n: int):
    color = colors[n % len(colors)]
    return styles[int(n / len(colors)) % len(styles)].format(color=color)

# data: [(sentence/paragraph, [values])]
# labels: for [values], has to have same length
def stacked_bar(
    data: list[tuple[str, list[float]]],
    labels: list[str],
) -> tuple[div, div]:
    stack_height = max(sum(values) for _, values in data)

    chart = div()
    for sentence, values in data:
        with chart:
            with div(style='display: flex'):
                p(sentence, style='flex: 70%; padding-right: 1rem')
                with div(style='flex: 30%; display: flex; padding: 0.25rem 0'):
                    has_left_border = False
                    for n, v in enumerate(values):
                        width = 100 * v / stack_height
                        if width > 0:
                            span(style=f'flex: 0 1 auto; width: {width}%; box-sizing: border-box; background-color: {_get_style(n)}; border: 1px solid black; {"border-left: 0" if has_left_border else ""}')
                            has_left_border = True

    legend = div()
    for n, label in enumerate(labels):
        with legend:
            with p():
                span(style=f'display: inline-block; width: 1em; height: 1em; background-color: {_get_style(n)}; border: 1px solid black; margin-right: 0.25rem; border-radius: 9999px')
                span(label)

    return chart, legend
