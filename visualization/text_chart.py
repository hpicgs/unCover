# pyright: reportWildcardImportFromLibrary=false
from dominate.tags import *

def stacked_bar(data: list[tuple[str, list[float]]], colors: list[str], labels: list[str]) -> str:
    stack_height = max(sum(values) for _, values in data)

    chart = div()
    for sentence, values in data:
        with chart:
            with div(style='display: flex'):
                p(sentence, style='flex: 70%; padding-right: 1rem')
                with div(style='flex: 30%; display: flex; padding: 0.25rem 0'):
                    for n, v in enumerate(values):
                        span(style=f'flex: 0 1 auto; width: {100 * v / stack_height}%; background-color: {colors[n]}')

    return chart.render()
