from pydro.detection import FilterPyramid, DeformationCost

import itertools
import numpy
from collections import namedtuple

__all__ = [
    'Offset',
    'Block',
    'Def',
    'DeformationRule',
    'Features',
    'Filter',
    'Loc',
    'Model',
    'Rule',
    'Stats',
    'StructuralRule',
    'Symbol',
    'FilteredSymbol',
    'FilteredStructuralRule',
    'FilteredDeformationRule',
    'TreeNode',
    'Leaf'
]

Score = namedtuple('Score', 'score,scale')
TreeNode = namedtuple('TreeNode', 'x,y,l,symbol,ds,s,children,rule')
Leaf = namedtuple('Leaf', 'x1,x2,y1,y2,scale')


class Model(object):

    def __init__(self, clss, year, note, start, maxsize, minsize,
                 interval, sbin, thresh, type, features, stats,
                 rules=None, symbols=None, filters=None, blocks=None):
        if rules is not None:
            self.rules = rules
        if symbols is not None:
            self.symbols = symbols
        if filters is not None:
            self.filters = filters
        if blocks is not None:
            self.blocks = blocks
        self.clss = clss
        self.year = year
        self.note = note
        self.start = start
        self.maxsize = maxsize
        self.minsize = minsize
        self.interval = interval
        self.sbin = sbin
        self.thresh = thresh
        self.type = type
        self.features = features
        self.stats = stats

    def Filter(self, pyramid):
        return FilteredModel(self, pyramid)


class FilteredModel (Model):

    def __init__(self, model, pyramid):
        super(FilteredModel, self).__init__(**model.__dict__)

        filtered_size = self.start.GetFilteredSize(pyramid)

        self.filtered_start = self.start.Filter(pyramid, self, filtered_size)

    def Parse(self, threshold, limit=None):
        X = numpy.array([], dtype=numpy.uint32)
        Y = numpy.array([], dtype=numpy.uint32)
        L = numpy.array([], dtype=numpy.uint32)
        S = numpy.array([], dtype=numpy.float32)
        for pos, level in enumerate(self.filtered_start.score):
            if isinstance(level.score, numpy.ndarray):
                Yi, Xi = numpy.where(level.score > threshold)
                Si = level.score[Yi, Xi].flatten()
                Li = pos * numpy.ones(Si.shape, dtype=numpy.uint32)
                X = numpy.hstack((X, Xi))
                Y = numpy.hstack((Y, Yi))
                S = numpy.hstack((S, Si))
                L = numpy.hstack((L, Li))

        if limit is None:
            limit = S.size

        order = list(enumerate(S))
        order.sort(key=lambda k: -k[1])
        order = numpy.array([o[0] for o in order[:limit]], dtype=numpy.uint32)

        X = X[order]
        Y = Y[order]
        L = L[order]
        S = S[order]

        detections = []

        assert len(X) == len(Y)
        assert len(X) == len(L)
        assert len(X) == len(S)
        for x, y, l, s in itertools.izip(X, Y, L, S):
            detections += [
                self.filtered_start.Parse(x=x, y=y, l=l, s=s, ds=0, model=self)
            ]

        return detections


class Filter(object):

    _p = numpy.array([
        9, 8, 7, 6, 5, 4, 3, 2, 1,
        0, 17, 16, 15, 14, 13, 12, 11, 10,
        18, 26, 25, 24, 23, 22, 21, 20, 19,
        29, 30, 27, 28,
        31,
    ])

    def __init__(self, blocklabel, size, flip, symbol):
        self.blocklabel = blocklabel
        self.size = size
        self.flip = flip
        self.symbol = symbol
        self._w = None

    def GetParameters (self):
        if self._w is None:
            if self.flip:
                self._w = self.blocklabel.w[:, ::-1, _p]
            else:
                self._w = self.blocklabel.w

        return self._w


class Rule(object):

    def __init__(self, type, lhs, rhs, detwindow, shiftwindow, i,
                 offset, loc, blocks, metadata=None):
        self.type = type
        self.lhs = lhs
        self.rhs = rhs
        self.detwindow = detwindow
        self.shiftwindow = shiftwindow
        self.i = i
        self.offset = offset
        self.loc = loc
        self.blocks = blocks
        self.metadata = metadata

    def __repr__(self):
        return '%s: %s' % (self.type, len(self.rhs))

    def GetFilteredSize(self, pyramid):
        size_pyramid = [(1, 1) for level in pyramid]

        for symbol in self.rhs:
            symbol_size_pyramid = symbol.GetFilteredSize(pyramid)

            for i in xrange(len(pyramid)):
                ymax, xmax = size_pyramid[i]
                ycurr, xcurr = symbol_size_pyramid[i]

                size_pyramid[i] = (max(ymax, ycurr), max(xmax, xcurr))

        return size_pyramid


class DeformationRule(Rule):

    def __init__(self, type, lhs, rhs, detwindow, shiftwindow, i,
                 offset, df, loc, blocks, metadata=None):
        super(DeformationRule, self).__init__(
            type, lhs, rhs, detwindow, shiftwindow, i, offset, loc, blocks, metadata
        )

        self.df = df

    def Filter(self, pyramid, model, filtered_size):
        return FilteredDeformationRule(self, pyramid, model, filtered_size)


class FilteredDeformationRule(DeformationRule):

    def __init__(self, deformation_rule, pyramid, model, filtered_size):
        super(FilteredDeformationRule, self).__init__(
            **deformation_rule.__dict__
        )

        self.filtered_rhs = [
            s.Filter(pyramid, model, filtered_size) for s in self.rhs
        ]

        def_w = self.df.GetParameters()

        assert len(self.filtered_rhs) == 1

        score = self.filtered_rhs[0].score

        bias = self.offset.GetParameters()
        loc_w = self.loc.GetParameters()

        loc_f = numpy.zeros((3, len(pyramid)), dtype=numpy.float32)
        loc_f[0, 0:model.interval] = 1
        loc_f[1, model.interval:2 * model.interval] = 1
        loc_f[2, 2 * model.interval:] = 1

        loc_scores = loc_w.dot(loc_f)

        ax, bx, ay, by = def_w.flatten().tolist()

        assert len(loc_scores.flatten()) == len(score)
        deformations = [
            DeformationCost(bias + s + ss.score, ax, bx, ay, by, 4)
            for s, ss in itertools.izip(loc_scores.flatten(), score)
        ]

        assert len(score) == len(deformations)
        self.score = [
            Score(scale=s.scale, score=d[0])
            for s, d in itertools.izip(score, deformations)
        ]
        self.Ix = [d[1] for d in deformations]
        self.Iy = [d[2] for d in deformations]

    def Parse(self, x, y, l, s, ds, model):
        Ix = self.Ix[l]
        Iy = self.Iy[l]

        nvp_y = y - model.maxsize[0] * ((1 << ds) - 1)
        nvp_x = x - model.maxsize[1] * ((1 << ds) - 1)

        rhs_nvp_x = Ix[nvp_y, nvp_x]
        rhs_nvp_y = Iy[nvp_y, nvp_x]

        rhs_x = rhs_nvp_x + model.maxsize[1] * ((1 << ds) - 1)
        rhs_y = rhs_nvp_y + model.maxsize[0] * ((1 << ds) - 1)

        symbol, = self.filtered_rhs

        nvp_x = rhs_x - model.maxsize[1] * ((1 << ds) - 1)
        nvp_y = rhs_y - model.maxsize[0] * ((1 << ds) - 1)
        rhs_s = symbol.score[l].score[nvp_y, nvp_x]

        children = [symbol.Parse(
            x=rhs_x,
            y=rhs_y,
            l=l,
            ds=ds,
            s=rhs_s,
            model=model,
        )]

        return children


class StructuralRule(Rule):

    def __init__(self, type, lhs, rhs, detwindow, shiftwindow, i, anchor,
                 offset, loc, blocks, metadata=None):
        super(StructuralRule, self).__init__(
            type, lhs, rhs, detwindow, shiftwindow, i, offset, loc, blocks, metadata
        )

        self.anchor = anchor

    def Filter(self, pyramid, model, filtered_size):
        return FilteredStructuralRule(self, pyramid, model, filtered_size)


class FilteredStructuralRule(StructuralRule):

    def __init__(self, structural_rule, pyramid, model, filtered_size):
        super(FilteredStructuralRule, self).__init__(
            **structural_rule.__dict__)

        self.filtered_rhs = [
            s.Filter(pyramid, model, filtered_size) for s in self.rhs
        ]

        bias = self.offset.GetParameters() * model.features.bias
        loc_w = self.loc.GetParameters()

        loc_f = numpy.zeros((3, len(pyramid)))
        loc_f[0, 0:model.interval] = 1
        loc_f[1, model.interval:2 * model.interval] = 1
        loc_f[2, 2 * model.interval:] = 1

        loc_scores = loc_w.dot(loc_f).flatten()

        assert len(filtered_size) == len(loc_scores.flatten())
        self.score = [
            float(bias + loc_score) * numpy.ones(size, dtype=numpy.float32)
            for size, loc_score
            in itertools.izip(filtered_size, loc_scores.flatten())
        ]

        assert len(self.anchor) == len(self.filtered_rhs)
        for anchor, filtered_symbol in itertools.izip(self.anchor, self.filtered_rhs):
            ax, ay, ds = anchor

            step = 2 ** ds

            virtpadx = (step - 1) * model.maxsize[1]
            virtpady = (step - 1) * model.maxsize[0]

            startx = ax - virtpadx + 1
            starty = ay - virtpady + 1

            score = [s.score for s in filtered_symbol.score]

            for i in xrange(len(score)):
                level = i - model.interval * ds

                if level >= 0:
                    endy = min(
                        score[level].shape[0],
                        starty + step * (self.score[i].shape[0] - 1)
                    )

                    endx = min(
                        score[level].shape[1],
                        startx + step * (self.score[i].shape[1] - 1)
                    )

                    iy = numpy.arange(starty, endy + 1, step)
                    oy = (iy < 1).sum()
                    iy = iy[numpy.where(iy >= 1)].flatten()

                    ix = numpy.arange(startx, endx + 1, step)
                    ox = (ix < 1).sum()
                    ix = ix[numpy.where(ix >= 1)].flatten()

                    sp = score[level][iy - 1, :][:, ix - 1]
                    sz = sp.shape

                    stmp = (-numpy.inf * numpy.ones(self.score[i].shape)).astype(
                        numpy.float32)
                    assert oy >= 0
                    assert ox >= 0
                    assert oy + sz[0] - 1 < stmp.shape[0]
                    assert ox + sz[1] - 1 < stmp.shape[1]
                    stmp[oy:oy + sz[0], ox:ox + sz[1]] = sp

                    self.score[i] += stmp
                else:
                    self.score[i][:] = -numpy.inf

        assert len(pyramid) == len(self.score)
        self.score = [
            Score(scale=l.scale, score=s)
            for l, s in itertools.izip(pyramid, self.score)
        ]

    def Parse(self, x, y, l, s, ds, model):
        assert len(self.anchor) == len(self.filtered_rhs)
        children = []
        for anchor, symbol in itertools.izip(self.anchor, self.filtered_rhs):
            ax, ay, ads = anchor

            rhs_x = x * (1 << ads) + ax
            rhs_y = y * (1 << ads) + ay
            rhs_l = l - model.interval * ads

            rhs_ds = ds + ads

            nvp_y = rhs_y - model.maxsize[0] * ((1 << rhs_ds) - 1)
            nvp_x = rhs_x - model.maxsize[1] * ((1 << rhs_ds) - 1)

            rhs_s = symbol.score[rhs_l].score[nvp_y, nvp_x]

            children += [
                symbol.Parse(
                    x=rhs_x,
                    y=rhs_y,
                    l=rhs_l,
                    s=rhs_s,
                    ds=rhs_ds,
                    model=model
                )
            ]

        return children


class Symbol(object):

    def __init__(self, type, filter, rules=[]):
        self.type = type
        self.filter = filter
        self.rules = rules

    def __repr__(self):
        return '%s\t%s' % (self.type, '\n\t'.join(str(type(r)) for r in self.rules))

    def Filter(self, pyramid, model, filtered_size):
        if self.type == 'T' and isinstance(self, FilteredSymbol):
            assert len(self.rules) == 0
            return self

        filtered_symbol = FilteredSymbol(self, pyramid, model, filtered_size)

        return filtered_symbol

    def GetFilteredSize(self, pyramid):
        if self.type == 'T':
            return [(
                level.features.shape[
                    0] - self.filter.GetParameters().shape[0] + 1,
                level.features.shape[
                    1] - self.filter.GetParameters().shape[1] + 1,
            ) for level in pyramid]

        else:
            size_pyramid = [(1, 1) for level in pyramid]

            for rule in self.rules:
                rule_size_pyramid = rule.GetFilteredSize(pyramid)

                for i in xrange(len(pyramid)):
                    ymax, xmax = size_pyramid[i]
                    ycurr, xcurr = rule_size_pyramid[i]

                    size_pyramid[i] = (max(ymax, ycurr), max(xmax, xcurr))

            return size_pyramid


class FilteredSymbol(Symbol):

    def __init__(self, symbol, pyramid, model, filtered_size):
        super(FilteredSymbol, self).__init__(**symbol.__dict__)

        if self.filter is not None:
            filter = self.filter.GetParameters()
            self.filtered = FilterPyramid(pyramid, filter, filtered_size)
            assert len(filtered_size) == len(self.filtered)
            assert len(pyramid) == len(self.filtered)
            self.score = [
                Score(scale=level.scale, score=filtered)
                for level, filtered in itertools.izip(pyramid, self.filtered)
            ]

            self.filtered_rules = []
        else:
            self.filtered_rules = [
                r.Filter(pyramid, model, filtered_size) for r in self.rules
            ]

            self.score = self.filtered_rules[0].score
            for filtered_rule in self.filtered_rules[1:]:
                self.score = [Score(
                    scale=level.scale,
                    score=numpy.max(
                        numpy.dstack((level.score, f.score)), axis=2),
                )
                    for level, f in itertools.izip(self.score, filtered_rule.score)]

        assert self.score is not None or self.filtered is not None

    def Parse(self, x, y, l, s, ds, model):
        if self.type == 'T':
            scale = model.sbin / self.score[l].scale

            x1 = (x - model.maxsize[1] * (1 << ds)) * scale
            y1 = (y - model.maxsize[0] * (1 << ds)) * scale
            x2 = x1 + self.filter.GetParameters().shape[1] * scale - 1
            y2 = y1 + self.filter.GetParameters().shape[0] * scale - 1

            leaf = Leaf(
                x1=x1,
                x2=x2,
                y1=y1,
                y2=y2,
                scale=scale,
            )

            return leaf
        else:
            selected_rule = None
            for rule in self.filtered_rules:
                nvp_y = y - model.maxsize[0] * ((1 << ds) - 1)
                nvp_x = x - model.maxsize[1] * ((1 << ds) - 1)

                score = rule.score[l].score[nvp_y, nvp_x]

                if score == s:
                    selected_rule = rule
                    break
            if selected_rule is None:
                raise Exception('Rule argmax not found')
            rule = selected_rule

            node = TreeNode(
                x=x,
                y=y,
                l=l,
                ds=ds,
                s=s,
                symbol=self,
                rule=selected_rule,
                children=rule.Parse(x=x, y=y, l=l, s=s, ds=ds, model=model),
            )

            return node

    def __repr__(self):
        return '%s\t%s' % (self.type, '\n\t'.join(
            str(type(r)) for r in self.filtered_rules
        ))


class Block(object):

    def __init__(self, w, lb, learn, reg_mult, dim, type):
        self.w = w
        self.lb = lb
        self.learn = learn
        self.reg_mult = reg_mult
        self.dim = dim
        self.type = type


class Features(object):

    def __init__(self, sbin, dim, truncation_dim, extra_octave, bias):
        self.sbin = sbin
        self.dim = dim
        self.truncation_dim = truncation_dim
        self.extra_octave = extra_octave
        self.bias = bias


class Stats(object):

    def __init__(self, slave_problem_time, data_mining_time, pos_latent_time,
                 filter_usage):
        self.slave_problem_time = slave_problem_time
        self.data_mining_time = data_mining_time
        self.pos_latent_time = pos_latent_time
        self.filter_usage = filter_usage


class Def(object):

    def __init__(self, blocklabel, flip):
        self.blocklabel = blocklabel
        self.flip = flip

    def GetParameters (self):
        return self.blocklabel.w
    

class Loc(object):

    def __init__(self, blocklabel):
        self.blocklabel = blocklabel

    def GetParameters (self):
        return self.blocklabel.w

class Offset(object):

    def __init__(self, blocklabel):
        self.blocklabel = blocklabel

    def GetParameters (self):
        return self.blocklabel.w
