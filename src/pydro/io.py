import copy
import itertools
import Queue
import msgpack
import numpy
import zlib

from pydro.core import *

__all__ = ['LoadModel', 'SaveModel']


def _type_handler(obj):
    if isinstance(obj, numpy.int32):
        return int(obj)
    elif isinstance(obj, numpy.float32):
        return float(obj)
    elif isinstance(obj, numpy.ndarray):
        if obj.dtype == numpy.float32:
            return {
                '__ndarray__': True,
                'shape': obj.shape,
                'data': obj.tostring(),
                'type': 'f',
            }
        elif obj.dtype == numpy.int32:
            return {
                '__ndarray__': True,
                'shape': obj.shape,
                'data': obj.tostring(),
                'type': 'i',
            }
        else:
            raise Exception('unhandled ndarray dtype (%s)' % obj.dtype)
    else:
        raise Exception('unhandled msgpack type (%s)' % type(obj))


def _type_unpacker(obj):
    if '__ndarray__' in obj:
        if obj['type'] == 'f':
            dtype = numpy.float32
        elif obj['type'] == 'i':
            dtype = numpy.int32
        array = numpy.fromstring(obj['data'], dtype=dtype).reshape(obj['shape'])
        array.flags.writeable = False
        return array
    elif '__model__' in obj:
        return obj['__model__']
    elif '__filter__' in obj:
        return obj['__filter__']
    elif '__symbol__' in obj:
        return obj['__symbol__']
    elif '__rule__' in obj:
        if obj['__rule__']['type'] == 'S':
            return obj['__rule__']
        else:
            return obj['__rule__']
    elif '__loc__' in obj:
        return obj['__loc__']
    elif '__offset__' in obj:
        return obj['__offset__']
    elif '__block__' in obj:
        return obj['__block__']
    elif '__def__' in obj:
        return obj['__def__']
    elif '__features__' in obj:
        return obj['__features__']
    elif '__stats__' in obj:
        return obj['__stats__']
    else:
        return obj


def _denormalize_model(model):
    symbols = {}
    symbol_queue = Queue.Queue()
    symbol_queue.put(model.start)

    old_symbols = []

    while not symbol_queue.empty():
        symbol = symbol_queue.get()

        if symbol in symbols:
            raise Exception('cycle in symbols detected')

        old_symbols += [symbol]
        symbols[symbol] = len(old_symbols)

        for rule in symbol.rules:
            assert rule.lhs() == symbol
            if rule.lhs() not in symbols:
                raise Exception('lhs should have already been touched')

            for rhs_symbol in rule.rhs:
                symbol_queue.put(rhs_symbol)

    filters = {}
    filters[None] = None

    blocks = {}

    new_start = 1

    new_model = {
        'clss' : model.clss,
        'year' : model.year,
        'note' : model.note,
        'start' : new_start,
        'maxsize' : model.maxsize,
        'minsize' : model.minsize,
        'interval' : model.interval,
        'sbin' : model.sbin,
        'thresh' : model.thresh,
        'type' : model.type,
        'features' : model.features.__dict__,
        'stats' : model.stats.__dict__,
        'rules' : [],
        'symbols' : [],
        'filters' : [],
        'blocks' : [],
    }

    for old_symbol in old_symbols:
        if old_symbol.filter not in filters:
            if old_symbol.filter.blocklabel not in blocks:
                new_block = {
                    'w' : old_symbol.filter.blocklabel.w,
                    'lb' : old_symbol.filter.blocklabel.lb,
                    'learn' : old_symbol.filter.blocklabel.learn,
                    'reg_mult' : old_symbol.filter.blocklabel.reg_mult,
                    'dim' : old_symbol.filter.blocklabel.dim,
                    'type' : old_symbol.filter.blocklabel.type,
                }
                new_model['blocks'] += [new_block]
                blocks[old_symbol.filter.blocklabel] = len(new_model['blocks'])

            new_filter_blocklabel = blocks[old_symbol.filter.blocklabel]

            new_filter_symbol = symbols[old_symbol.filter.symbol()]

            new_filter = {
                'blocklabel' : new_filter_blocklabel,
                'size' : symbol.filter.size,
                'flip' : symbol.filter.flip,
                'symbol' : new_filter_symbol,
            }

            new_model['filters'] += [new_filter]
            filters[old_symbol.filter] = len(new_model['filters'])

        new_rules = []
        rule_group = old_symbol.rules
        for rule in rule_group:
            assert old_symbol == rule.lhs()
            new_lhs = symbols[rule.lhs()]
            new_rhs = [symbols[a] for a in rule.rhs]

            for block in rule.blocks:
                if block not in blocks:
                    new_block = {
                        'w' : block.w,
                        'lb' : block.lb,
                        'learn' : block.learn,
                        'reg_mult' : block.reg_mult,
                        'dim' : block.dim,
                        'type' : block.type,
                    }
                    new_model['blocks'] += [new_block]
                    blocks[block] = len(new_model['blocks'])

            new_loc_blocklabel = blocks[rule.loc.blocklabel]
            new_loc = {
                'blocklabel' : new_loc_blocklabel,
            }

            new_offset_blocklabel = blocks[rule.offset.blocklabel]
            new_offset = {
                'blocklabel' : new_offset_blocklabel,
            }

            new_blocks = [blocks[block] for block in rule.blocks]

            if isinstance(rule, DeformationRule):
                new_df_blocklabel = blocks[rule.df.blocklabel]

                new_df = {
                    'flip' : rule.df.flip,
                    'blocklabel' : new_df_blocklabel,
                }

                new_rule = {
                    'type' : rule.type,
                    'lhs' : new_lhs,
                    'rhs' : new_rhs,
                    'detwindow' : rule.detwindow,
                    'shiftwindow' : rule.shiftwindow,
                    'i' : rule.i,
                    'offset' : new_offset,
                    'loc' : new_loc,
                    'blocks' : new_blocks,
                    'df' : new_df,
                }
            else:
                new_rule = {
                    'type' : rule.type,
                    'lhs' : new_lhs,
                    'rhs' : new_rhs,
                    'detwindow' : rule.detwindow,
                    'shiftwindow' : rule.shiftwindow,
                    'i' : rule.i,
                    'anchor' : rule.anchor,
                    'offset' : new_offset,
                    'loc' : new_loc,
                    'blocks' : new_blocks,
                }
            new_rules += [new_rule]

        new_symbol = {
            'type' : symbol.type,
            'filter' : filters[old_symbol.filter],
        }

        new_model['symbols'] += [new_symbol]

        new_model['rules'] += [new_rules]

    return new_model


def _normalize_model(model):
    new_blocks = {i+1:Block(**block) for i, block in enumerate(model['blocks'])}

    new_rules = {}
    new_symbols = {}

    new_filters = {
        None: None,
    }
    for pos, filter in enumerate(model['filters']):
        block_idx = filter['blocklabel']
        block = new_blocks[block_idx]

        new_filter = Filter (
            blocklabel=block,
            size=filter['size'],
            flip=filter['flip'],
            symbol=None,
        )

        new_filters[pos+1] = new_filter

    symbol_rule_list = list(enumerate(itertools.izip(model['symbols'], model['rules'])))
    symbol_rule_list.reverse()

    for pos, (symbol, rules) in symbol_rule_list:
        new_rules = []
        for rule in rules:
            rhs_idx = rule['rhs']
            new_rhs = [new_symbols[a] for a in rhs_idx]

            loc_idx = rule['loc']['blocklabel']
            loc_block = new_blocks[loc_idx]

            new_loc = Loc(
                blocklabel=loc_block,
            )

            offset_idx = rule['offset']['blocklabel']
            offset_block = new_blocks[offset_idx]

            new_offset = Offset (
                blocklabel=offset_block,
            )

            block_idx = rule['blocks']
            new_rule_blocks = [new_blocks[a] for a in block_idx]

            if rule['type'] == 'D':
                df_idx = rule['df']['blocklabel']
                block = new_blocks[df_idx]

                new_df = Def(
                    blocklabel=block,
                    flip=rule['df']['flip'],
                )

                new_rule = DeformationRule (
                    type=rule['type'],
                    lhs=None,
                    rhs=new_rhs,
                    detwindow=rule['detwindow'],
                    shiftwindow=rule['shiftwindow'],
                    i=rule['i'],
                    offset=new_offset,
                    df=new_df,
                    loc=new_loc,
                    blocks=new_rule_blocks,
                )
            elif rule['type'] == 'S':
                new_rule = StructuralRule (
                    type=rule['type'],
                    lhs=None,
                    rhs=new_rhs,
                    detwindow=rule['detwindow'],
                    shiftwindow=rule['shiftwindow'],
                    i=rule['i'],
                    offset=new_offset,
                    anchor=rule['anchor'],
                    loc=new_loc,
                    blocks=new_rule_blocks,
                )

            new_rules += [new_rule]

        new_symbol = Symbol (
            type=symbol['type'],
            filter=new_filters[symbol['filter']],
            rules=new_rules,            
        )

        new_symbols[pos+1] = new_symbol

    for pos, (symbol, rules) in symbol_rule_list:
        new_symbol = new_symbols[pos+1]

        for rule, new_rule in itertools.izip(rules, new_symbol.rules):
            lhs_idx = rule['lhs']
            new_lhs = new_symbols[lhs_idx]

            new_rule.SetLHS (new_lhs)

            assert new_rule.lhs() == new_symbol

    for pos, filter in enumerate(model['filters']):
        new_filter = new_filters[pos+1]
        new_symbol = new_symbols[filter['symbol']]

        new_filter.SetSymbol (new_symbol)


    start_idx = model['start']
    new_start = new_symbols[start_idx]

    new_model = Model (
        clss=model['clss'],
        year=model['year'],
        note=model['note'],
        start=new_start,
        maxsize=model['maxsize'],
        minsize=model['minsize'],
        interval=model['interval'],
        sbin=model['sbin'],
        thresh=model['thresh'],
        type=model['type'],
        features=Features(**model['features']),
        stats=Stats(**model['stats']),
    )

    return new_model


def SaveModel(filename, model):
    model = _denormalize_model(model)
    packed = msgpack.packb(model, default=_type_handler)
    compressed = zlib.compress(packed)
    with open(filename, 'wb') as f:
        f.write(compressed)


def LoadModel(filename):
    with open(filename, 'rb') as f:
        compressed = f.read()
    packed = zlib.decompress(compressed)
    model = msgpack.unpackb(packed, object_hook=_type_unpacker)
    model = _normalize_model(model)
    return model
