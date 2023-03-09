# pyright: reportWildcardImportFromLibrary=false
from dominate.tags import *

# from https://colorbrewer2.org/?type=qualitative&scheme=Set3&n=12)
_colors = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd','#ccebc5','#ffed6f']
# css styles for bars, see
# https://www.magicpattern.design/tools/css-backgrounds for patterns
_styles = [
    # solid
    'background-color: {color};',
    # diagonal stripes
    'background-color: {neutral}; background: repeating-linear-gradient( 45deg, {color}, {color} 6px, {neutral} 6px, {neutral} 12px );',
    # polka dots
    'background-color: {neutral}; background-image: radial-gradient({color} 6px, {neutral} 6px); background-size: 18px 18px;',
    # zig-zag
    '''
        background-color: {neutral};
        background-image:  linear-gradient(135deg, {color} 25%, transparent 25%), linear-gradient(225deg, {color} 25%, transparent 25%), linear-gradient(45deg, {color} 25%, transparent 25%), linear-gradient(315deg, {color} 25%, {neutral} 25%);
        background-position:  9px 0, 9px 0, 0 0, 0 0;
        background-size: 18px 18px;
        background-repeat: repeat;
    ''',
    # boxes
    '''
        background-color: {neutral};
        background-image:  linear-gradient({color} 6px, transparent 6px), linear-gradient(to right, {color} 6px, {neutral} 6px);
        background-size: 10px 10px;
    ''',
]

# loop through colors for each pattern-style
# then loop through pattern-styles
def _get_style(n: int):
    color = _colors[n % len(_colors)]
    return _styles[int(n / len(_colors)) % len(_styles)].format(color=color, neutral='#333')

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
                            span(style=f'flex: 0 1 auto; width: {width}%; box-sizing: border-box; {_get_style(n)}; border: 1px solid black; {"border-left: 0" if has_left_border else ""}')
                            has_left_border = True

    legend = div()
    for n, label in enumerate(labels):
        with legend:
            with p(style='display: flex; align-items: center'):
                span(style=f'display: inline-block; width: 2em; height: 2em; {_get_style(n)}; border: 1px solid black; margin-right: 0.5rem; border-radius: 9999px')
                span(label)

    return chart, legend
