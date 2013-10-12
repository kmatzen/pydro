import weakref
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
    elif isinstance(obj, Model):
        return {'__model__': obj.__dict__}
    elif isinstance(obj, Filter):
        return {'__filter__': obj.__dict__}
    elif isinstance(obj, Rule):
        return {'__rule__': obj.__dict__}
    elif isinstance(obj, Symbol):
        return {'__symbol__': obj.__dict__}
    elif isinstance(obj, Block):
        return {'__block__': obj.__dict__}
    elif isinstance(obj, Features):
        return {'__features__': obj.__dict__}
    elif isinstance(obj, Loc):
        return {'__loc__': obj.__dict__}
    elif isinstance(obj, Offset):
        return {'__offset__': obj.__dict__}
    elif isinstance(obj, Def):
        return {'__def__': obj.__dict__}
    elif isinstance(obj, Stats):
        return {'__stats__': obj.__dict__}
    else:
        raise Exception('unhandled msgpack type (%s)' % type(obj))


def _type_unpacker(obj):
    if '__ndarray__' in obj:
        if obj['type'] == 'f':
            dtype = numpy.float32
        elif obj['type'] == 'i':
            dtype = numpy.int32
        return numpy.fromstring(obj['data'], dtype=dtype).reshape(obj['shape'])
    elif '__model__' in obj:
        return Model(**obj['__model__'])
    elif '__filter__' in obj:
        return Filter(**obj['__filter__'])
    elif '__symbol__' in obj:
        return Symbol(**obj['__symbol__'])
    elif '__rule__' in obj:
        if obj['__rule__']['type'] == 'S':
            return StructuralRule(**obj['__rule__'])
        else:
            return DeformationRule(**obj['__rule__'])
    elif '__loc__' in obj:
        return Loc(**obj['__loc__'])
    elif '__offset__' in obj:
        return Offset(**obj['__offset__'])
    elif '__block__' in obj:
        return Block(**obj['__block__'])
    elif '__def__' in obj:
        return Def(**obj['__def__'])
    elif '__features__' in obj:
        return Features(**obj['__features__'])
    else:
        return obj


def _denormalize_model(model):
    model = copy.deepcopy(model)

    filters = {}
    filters_rev = {}
    blocks = {}
    blocks_rev = {}
    symbols = {}
    symbols_rev = {}
    symbol_queue = Queue.Queue()
    symbol_queue.put(model.start)
    while not symbol_queue.empty():
        symbol = symbol_queue.get()

        if symbol in symbols:
            raise Exception('cycle in symbols detected')

        symbol_idx = len(symbols) + 1
        symbols[symbol] = symbol_idx
        symbols_rev[symbol_idx] = symbol

        if symbol.filter is not None:
            if symbol.filter.blocklabel not in blocks:
                block_idx = len(blocks) + 1
                blocks[symbol.filter.blocklabel] = block_idx
                blocks_rev[block_idx] = symbol.filter.blocklabel
            block_idx = blocks[symbol.filter.blocklabel]
            symbol.filter.blocklabel = block_idx

            symbol.filter.symbol = symbol_idx

            if symbol.filter not in filters:
                filter_idx = len(filters) + 1
                filters[symbol.filter] = filter_idx
                filters_rev[filter_idx] = symbol.filter
                del symbol.filter._w

            filter_idx = filters[symbol.filter]
            symbol.filter = filter_idx

        for rule in symbol.rules:
            """
            if rule.lhs not in symbols:
                raise Exception('lhs should have already been touched')
            """

            for rhs_symbol in rule.rhs:
                symbol_queue.put(rhs_symbol)

    for symbol_idx in symbols_rev:
        symbol = symbols_rev[symbol_idx]
        rule_group = symbol.rules
        for rule in rule_group:
            #rule.lhs = symbols[rule.lhs]
            rule.rhs = [symbols[a] for a in rule.rhs]

            for block in rule.blocks:
                if block not in blocks:
                    block_idx = len(blocks) + 1
                    blocks[block] = block_idx
                    blocks_rev[block_idx] = block

            rule.loc.blocklabel = blocks[rule.loc.blocklabel]

            rule.offset.blocklabel = blocks[rule.offset.blocklabel]

            if isinstance(rule, DeformationRule):
                rule.df.blocklabel = blocks[rule.df.blocklabel]

            rule.blocks = [blocks[block] for block in rule.blocks]

    model.start = symbols[model.start]

    model.symbols = [symbols_rev[i + 1] for i in xrange(len(symbols_rev))]
    model.blocks = [blocks_rev[i + 1] for i in xrange(len(blocks_rev))]
    model.filters = [filters_rev[i + 1] for i in xrange(len(filters_rev))]

    model.rules = [s.rules for s in model.symbols]
    for s in model.symbols:
        del s.rules

    return model


def _normalize_model(model):
    model = copy.deepcopy(model)

    for filter in model.filters:
        block_idx = filter.blocklabel
        block = model.blocks[block_idx - 1]
        filter.blocklabel = block

        symbol_idx = filter.symbol
        symbol = model.symbols[symbol_idx - 1]
        filter.symbol = symbol

    for symbol in model.symbols:
        filter_idx = symbol.filter
        if filter_idx is not None:
            filter = model.filters[filter_idx - 1]
            symbol.filter = filter

    for symbol, rules in itertools.izip(model.symbols, model.rules):
        symbol.rules = rules

    for rule_group in model.rules:
        for rule in rule_group:
            """
            lhs_idx = rule.lhs
            lhs = model.symbols[lhs_idx - 1]
            rule.lhs = lhs
            """

            rhs_idx = rule.rhs
            rhs = [model.symbols[a - 1] for a in rhs_idx]
            rule.rhs = rhs

            if isinstance(rule, DeformationRule):
                df_idx = rule.df.blocklabel
                block = model.blocks[df_idx - 1]
                rule.df.blocklabel = block

            loc_idx = rule.loc.blocklabel
            loc = model.blocks[loc_idx - 1]
            rule.loc.blocklabel = loc

            offset_idx = rule.offset.blocklabel
            offset = model.blocks[offset_idx - 1]
            rule.offset.blocklabel = offset

            block_idx = rule.blocks
            blocks = [model.blocks[a - 1] for a in block_idx]
            rule.blocks = blocks

    start_idx = model.start
    start = model.symbols[start_idx - 1]
    model.start = start

    del model.blocks
    del model.filters
    del model.rules
    del model.symbols

    return model


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
