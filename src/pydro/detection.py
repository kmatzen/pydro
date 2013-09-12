from pydro._detection import *

from pydro.core import *

import copy

__all__ = ['FilterImage', 'FilterPyramid', 'FilterModel']

def FilterPyramid (pyramid, filter):
    for level in pyramid:
        yield Filter(level, filter)

def FilterSymbol (pyramid, symbol):
    if symbol.filter is not None:
        filter = symbol.filter.blocklabel.w.reshape(symbol.filter.blocklabel.shape)
        filtered = FilterPyramid (pyramid, filter)
    else:
        filtered = None

    filtered_symbol = FilteredSymbol (symbol, filtered)

    for rule in filtered_symbol.rules:
        filtered_rhs = [FilterSymbol(pyramid, s) for s in rule.rhs]
        if isinstance(rule, DeformationRule):
            filtered_rule = FilteredDeformationRule(rule, filtered_symbol, filtered_rhs)
        elif isinstance(rule, StructuralRule):
            filtered_rule = FilteredStructuralRule(rule, filtered_symbol, filtered_rhs)
        else:
            raise Exception('unknown rule type')
        filtered_symbol.filtered_rules += [filtered_rule]

    return filtered_symbol

def FilterModel (pyramid, model):
    filtered_model = copy.deepcopy(model)

    filtered_model.start = FilterSymbol(pyramid, filtered_model.start)

    return filtered_model
