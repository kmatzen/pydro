from pydro.detection import *

import itertools
import copy
import numpy
from collections import namedtuple

__all__ = ['Offset', 'Block', 'Def', 'DeformationRule', 'Features', 'Filter', 'Loc', 'Model', 'Rule', 'Stats', 'StructuralRule', 'Symbol', 'FilteredSymbol', 'FilteredStructuralRule', 'FilteredDeformationRule']

Score = namedtuple('Score', 'score,scale')

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

    def Filter (self, pyramid):
        filtered_model = copy.deepcopy(self)

        filtered_model.start = filtered_model.start.Filter(pyramid, filtered_model)

        return filtered_model

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

    def __repr__ (self):
        return '%s: %s'%(self.type, len(self.rhs))

       

class DeformationRule(Rule):
    def __init__ (self, type, lhs, rhs, detwindow, shiftwindow, i, offset, df, loc, blocks):
        super(DeformationRule, self).__init__ (type, lhs, rhs, detwindow, shiftwindow, i, offset, loc, blocks)
        self.df = df

    def Filter (self, pyramid, model):
        return FilteredDeformationRule(self, pyramid, model)
 
class FilteredDeformationRule(DeformationRule):
    def __init__ (self, deformation_rule, pyramid, model):
        super(FilteredDeformationRule, self).__init__ (**deformation_rule.__dict__)
       
        self.filtered_rhs = [s.Filter(pyramid, model) for s in self.rhs]

        def_w = self.df.blocklabel.w.reshape(self.df.blocklabel.shape)
        
        assert len(self.filtered_rhs) == 1

        score = self.filtered_rhs[0].score

        bias = self.offset.blocklabel.w.reshape(self.offset.blocklabel.shape)
        loc_w = self.loc.blocklabel.w.reshape(self.loc.blocklabel.shape)

        loc_f = numpy.zeros((3, len(pyramid)), dtype=numpy.float32)
        loc_f[0,0:model.interval] = 1
        loc_f[1,model.interval:2*model.interval] = 1
        loc_f[2,2*model.interval:] = 1

        loc_scores = loc_w.dot(loc_f)

        ax, bx, ay, by = def_w.flatten().tolist()

        self.score = [Score(scale=ss.scale, score=DeformationCost(bias+s+ss.score, ax, bx, ay, by, 4)) for s,ss in itertools.izip(loc_scores.flatten(), score)]

class StructuralRule(Rule):
    def __init__ (self, type, lhs, rhs, detwindow, shiftwindow, i, anchor, offset, loc, blocks):
        super(StructuralRule, self).__init__ (type, lhs, rhs, detwindow, shiftwindow, i, offset, loc, blocks)
        self.anchor = anchor

    def Filter (self, pyramid, model):
        return FilteredStructuralRule(self, pyramid, model)

class FilteredStructuralRule(StructuralRule):
    def __init__ (self, structural_rule, pyramid, model):
        super(FilteredStructuralRule, self).__init__ (**structural_rule.__dict__)
        
        self.filtered_rhs = [s.Filter(pyramid, model) for s in self.rhs]

        bias = self.offset.blocklabel.w.reshape(self.offset.blocklabel.shape)*model.features.bias
        loc_w = self.loc.blocklabel.w.reshape(self.loc.blocklabel.shape)

        loc_f = numpy.zeros((3, len(pyramid)))
        loc_f[0,0:model.interval] = 1
        loc_f[1,model.interval:2*model.interval] = 1
        loc_f[2,2*model.interval:] = 1

        loc_scores = loc_w.dot(loc_f).flatten()

        self.score = [numpy.array([[float(bias + loc_score)]]) for loc_score in loc_scores]
      
        for anchor, filtered_symbol in itertools.izip(self.anchor, self.filtered_rhs):
            ax, ay, ds = anchor

            step = 2**ds

            score = [s.score for s in filtered_symbol.score]

            for i in xrange(len(score)):
                level = i - model.interval*ds
                
                if level >= 0:
                    endy = min(score[i].shape[0]-1, ay+step*(score[i].shape[0]-1))
                    endx = min(score[i].shape[1]-1, ax+step*(score[i].shape[1]-1))

                    iy = numpy.arange(ay, endy, step)
                    oy = (iy < 0).sum()
                    iy = iy[numpy.where(iy >= 0)]

                    ix = numpy.arange(ax, endx, step)
                    ox = (ix < 0).sum()
                    ix = ix[numpy.where(ix >= 0)]

                    sp = score[level][iy,:][:,ix]
                    sz = sp.shape

                    stmp = (-numpy.inf*numpy.ones(score[i].shape)).astype(numpy.float32)
                    stmp[oy:oy+sz[0], ox:ox+sz[1]] = sp

                    self.score[i] = stmp + self.score[i]
                else:
                    self.score[i] += numpy.array([[-numpy.inf]], dtype=numpy.float32)

        self.score = [Score(scale=l.scale, score=s) for l, s in itertools.izip(pyramid, self.score)]

class Symbol(object):
    def __init__ (self, type, filter, rules=[]):
        self.type = type
        self.filter = filter
        self.rules = rules

    def __repr__ (self):
        return '%s\t%s'%(self.type, '\n\t'.join(str(type(r)) for r in self.rules))

    def Filter (self, pyramid, model):
        if self.type == 'T' and isinstance(self, FilteredSymbol):
            assert len(self.rules) == 0
            return self

        filtered_symbol = FilteredSymbol (self, pyramid, model)

        return filtered_symbol

class FilteredSymbol(Symbol):
    def __init__ (self, symbol, pyramid, model):
        super(FilteredSymbol, self).__init__ (**symbol.__dict__)

        if self.filter is not None:
            filter = self.filter.blocklabel.w.reshape(self.filter.blocklabel.shape)
            self.filtered = FilterPyramid (pyramid, filter)
            self.score = [Score(scale=level.scale, score=filtered) for level, filtered in itertools.izip(pyramid, self.filtered)]
        else:
            self.filtered_rules = [r.Filter(pyramid, model) for r in self.rules]

            self.score = self.filtered_rules[0].score
            for filtered_rule in self.filtered_rules[1:]:
                for level in xrange(len(filtered_rule.score)):
                    self.score[level] = Score(scale=self.score[level].scale, score=numpy.max(numpy.dstack((self.score[level].score, filtered_rule.score[level].score)), axis=2))

        assert self.score is not None or self.filtered is not None

    def __repr__ (self):
        return '%s\t%s'%(self.type, '\n\t'.join(str(type(r)) for r in self.filtered_rules))

class Block(object):
    def __init__ (self, w, lb, learn, reg_mult, dim, shape, type):
        self.w = w
        self.lb = lb
        self.learn = learn
        self.reg_mult = reg_mult
        self.dim = dim  
        self.shape = shape
        self.type = type

class Features(object):
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

class Offset(object):
    def __init__ (self, blocklabel):
        self.blocklabel = blocklabel
