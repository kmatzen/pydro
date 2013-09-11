__all__ = ['Block', 'Def', 'DeformationRule', 'Feature', 'Filter', 'Loc', 'Model', 'Rule', 'Stats', 'StructuralRule', 'Symbol']

class Model(object):
    def __init__ (self, clss, year, note, filters, rules, symbols, start, maxsize, minsize,
                    interval, sbin, thresh, type, blocks, features, stats):
        self.clss = clss
        self.year = year
        self.note = note
        self.filters = filters
        self.rules = rules
        self.symbols = symbols
        self.start = start
        self.maxsize = maxsize
        self.minsize = minsize
        self.interval = interval
        self.sbin = sbin
        self.thresh = thresh
        self.type = type
        self.blocks = blocks
        self.features = features
        self.stats = stats

class Filter(object):
    def __init__ (self, blocklabel, size, flip, symbol):
        self.blocklabel = blocklabel
        self.size = size
        self.flip = flip
        self.symbol = symbol

class Rule(object):
    def __init__ (self, type, lhs, rhs, detwindow, shiftwindow, i, offset, loc, blocks):
        self.type = type
        self.lhs = lhs
        self.rhs = rhs
        self.detwindow = detwindow
        self.shiftwindow = shiftwindow
        self.i = i
        self.offset = offset
        self.loc = loc
        self.blocks = blocks

class DeformationRule(Rule):
    def __init__ (self, type, lhs, rhs, detwindow, shiftwindow, i, offset, df, loc, blocks):
        super(DeformationRule, self).__init__ (type, lhs, rhs, detwindow, shiftwindow, i, offset, loc, blocks)
        self.df = df

class StructuralRule(Rule):
    def __init__ (self, type, lhs, rhs, detwindow, shiftwindow, i, anchor, offset, loc, blocks):
        super(StructuralRule, self).__init__ (type, lhs, rhs, detwindow, shiftwindow, i, offset, loc, blocks)
        self.anchor = anchor

class Symbol(object):
    def __init__ (self, type, filter):
        self.type = type
        self.filter = filter
        self.rules = []

class Block(object):
    def __init__ (self, w, lb, learn, reg_mult, dim, shape, type):
        self.w = w
        self.lb = lb
        self.learn = learn
        self.reg_mult = reg_mult
        self.dim = dim  
        self.shape = shape
        self.type = type

class Feature(object):
    def __init__ (self, sbin, dim, truncation_dim, extra_octave, bias):
        self.sbin = sbin
        self.dim = dim
        self.truncation_dim = truncation_dim
        self.extra_octave = extra_octave
        self.bias = bias

class Stats(object):
    def __init__ (self, slave_problem_time, data_mining_time, pos_latent_time, filter_usage):
        self.slave_problem_time = slave_problem_time
        self.data_mining_time = data_mining_time
        self.pos_latent_time = pos_latent_time
        self.filter_usage = filter_usage

class Def(object):
    def __init__ (self, blocklabel, flip):
        self.blocklabel = blocklabel
        self.flip = flip

class Loc(object):
    def __init__ (self, blocklabel):
        self.blocklabel = blocklabel


