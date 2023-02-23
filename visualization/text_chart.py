# pyright: reportWildcardImportFromLibrary=false
from dominate.tags import *

# data: [(sentence/paragraph, [values])]
# colors: CSS compatible color-strings; min same length as [values] (default
#     value from https://colorbrewer2.org/?type=qualitative&scheme=Set3&n=12)
# labels: for [values], has to have same length
def stacked_bar(
    data: list[tuple[str, list[float]]],
    labels: list[str],
    colors: list[str] = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd','#ccebc5','#ffed6f'],
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
                            span(style=f'flex: 0 1 auto; width: {width}%; box-sizing: border-box; background-color: {colors[n]}; border: 1px solid black; {"border-left: 0" if has_left_border else ""}')
                            has_left_border = True

    legend = div()
    for label, color in zip(labels, colors):
        with legend:
            with p():
                span(style=f'display: inline-block; width: 1em; height: 1em; background-color: {color}; border: 1px solid black; margin-right: 0.25rem; border-radius: 9999px')
                span(label)

    return chart, legend
