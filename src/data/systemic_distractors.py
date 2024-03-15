import itertools
import random
import numpy as np

class SystematicDistractors():
    """
        Creates the items for a referential game.
        The items (target and distractors) consist of a 2-D grid of shape 2x2.
        (The current implementation does not generalize to other grid shapes)
        Each grid can host one figure in each of its positions (i.e. each grid may host a maximum of 4 figures).
        The distractors of a certain target are generated systematically.
        Example of distractor generation:
        * Target:
            +---+---+
            | @ |   |
            +---+---+
            |   | # |
            +---+---+
        * Distractors:
            ** Change figure 1:
                +---+---+
                | ~ |   |
                +---+---+
                |   | # |
                +---+---+
            ** Change figure 2:
                +---+---+
                | @ |   |
                +---+---+
                |   | * |
                +---+---+
            ** Toggle x coordinates:
                +---+---+
                |   | @ |
                +---+---+
                | # |   |
                +---+---+
            ** Toggle y coordinates:
                +---+---+
                |   | # |
                +---+---+
                | @ |   |
                +---+---+
            ** Toggle both coordinates (only for diagonal targets)
                +---+---+
                | # |   |
                +---+---+
                |   | @ |
                +---+---+

    """
    def __init__(self, figures:set, allow_repetitions:bool, n_figsxgrid=2, excluded_graphstrings=[]):
        """

        :param figures: collection of figures that can populate the grid
        :param allow_repetitions: if true, the same figure appear two or more times in the same grid
        :param n_figsxgrid: how many figures appear in each grid (default: 2)
        """
        if len(figures) < n_figsxgrid:
            raise Exception("The provided number of figures is smaller than the minimum number of figures per grid.")

        self.grid_size=4
        self.grid_shape=(2,2)
        self.figures=set(figures)
        self.nfigures=len(figures)
        self.allow_repetitions=allow_repetitions
        self.n_figsxgrid=n_figsxgrid
        self.targets=[]
        self.distractors=[]

        #Generate all possible targets
        self.targets=self.getTargets()

        #Generate all possible distractors for the target
        self.distractors=self.getDistractors(self.targets)
    
        self.post_adapter(excluded_graphstrings)

    def post_adapter(self, excluded_graphstrings):
        self.targets = [self.rebuild_graphstring(target) for target in self.targets]
        self.distractors = [[self.rebuild_graphstring(distractor) for distractor in distractors] for distractors in self.distractors]
        
        t = []
        d = []

        for i in range(len(self.targets)):
            target = self.targets[i]
            distractors = self.distractors[i]
            if target not in excluded_graphstrings:
                assert len(distractors) + 1 == len(set(distractors + [target]))
                t.append(target)
                d.append(distractors)
        
        self.targets = t
        self.distractors = d

    @staticmethod
    def rebuild_graphstring(input):
        s = ["0" for _ in range(4)]
        s[input[1][0]-1] = input[0][0]
        s[input[1][1]-1] = input[0][1]
        return "_".join(s)

    def getDistractors(self, targets):
        """
        Generate distractors for each target
        """
        distractors=[]
        for target in targets:
            distractors.append(self.getDistractors4Target(target))
        return distractors

    def getTargets(self):
        #Get all the possible grid arrangements (i.e. possible filled slots given n_figsxgrid)
        grid_positions=self.getPossibleGridArrangements()
        #Get all the possible combinations of n_figsxgrid figures which can appear in one grid
        figure_combs=self.getCombinationsFigures()
        #Combine figures with their position in the grid
        targets=itertools.product(figure_combs, grid_positions)
        #targets: iterator over ((fig1,fig2),(position_fig1, position_fig2))
        return list(targets)

    def getCombinationsFigures(self):
        """Get all the possible combinations of n_figsxgrid
        Example: figures=[1,2,3], combinations of 2: [(1,2),(1,3),(2,3)]
        :return: generator of combinations (tuples) of figure indices
        """
        if self.allow_repetitions:
            fig_combs=itertools.combinations_with_replacement(self.figures, r=self.n_figsxgrid)
        else:
            fig_combs=itertools.combinations(self.figures, r=self.n_figsxgrid)
        return fig_combs

    def getPossibleGridArrangements(self):
        """ Given a number of figures,
        returns all the possible ways in which they can be located in a grid
        :return: iterator with linear index of occupied position
         """
        return itertools.permutations(range(1, self.grid_size+1), self.n_figsxgrid)


    def prod2Grid(self, comb):
        """
        Represents grids as 2-dimensional arrays of integers.
        0 means empty; any other number represents the index of the figure that occupies
        such position in the grid.
        :param comb:
        :return:
        """
        gridList=np.zeros(self.grid_shape)
        idxs, figs=comb
        assert(len(idxs)==len(figs))

        for i,_ in enumerate(list(idxs)):
            x,y=self.pos2coords(idxs[i])
            gridList[x][y]=figs[i]
        return gridList

    def pos2coords(self, i):
        """
        From linear index to 2d indexing.
        :param i: Linear index (starting at 1!)
        :return: row_index, column_index
        """
        i-=1
        x = i // self.grid_shape[0]
        y = i % self.grid_shape[1]
        return x,y

    def coords2pos(self, x, y):
        """
        From to 2d indexing to linear index.
        :param x: row
        :param y: column
        :return: Linear index (starting at 1!)
        """
        return x * self.grid_shape[1] + y + 1

    def getDistractors4Target(self, target):
        """
        Given one target, get distractors that are systematically related.
        """
        distractors=[]
        #Replaced each figure (separately)
        for i in range(self.n_figsxgrid):
            distractors.append(self.replaceFigure(target, i))
        #Change x for every figure
        distractors.append(self.toggleCoord(target, 0))
        #Change y for every figure
        distractors.append(self.toggleCoord(target,1))
        #Change both x and y (only if target is diagonal)
        if self.isDiagonal(target):
            distractors.append(self.toggleCoord(self.toggleCoord(target,0), 1))
        return distractors


    def replaceFigure(self, target, fig_index):
        """
        Replaces figure with fig_index by another random figure.
        The new figure must be different form the current one.
        If allow_repetitions is off, the new figure is also different
        from any other figures in the target.
        :param target:
        :param fig_index:
        :return:
        """
        figs, positions = target
        new_figs=list(figs)

        if self.allow_repetitions:
            #Remove only current figure from the available options to substitute for
            options=self.figures - set([figs[fig_index]])
        else:
            #Remove current figures in the grid from options to substitute for
            options=self.figures-set(figs)

        #Pick another random figure
        new_figs[fig_index] = random.choice(list(options))

        return (tuple(new_figs), positions)

    @staticmethod
    def isDiagonal(target):
        """
        Returns true if the 2x2 grid consists of 2 figures placed on one diagonal.
        +---+---+      +---+---+
        | @ |   |  or  |   | @ |
        +---+---+      +---+---+
        |   | # |      | # |   |
        +---+---+      +---+---+
        """
        _, positions = target
        return (set(positions)==set((1,4)) or set(positions)==set((2,3)))

    def toggleCoord(self, target, axis):
        """
        Changes the horizontal (dim=0) or vertical (dim=1) coordinates of the target.
        E.g. dim=1:
        +---+---+           +---+---+
        | @ |   |  becomes  |   | @ |
        +---+---+           +---+---+
        |   | # |           | # |   |
        +---+---+           +---+---+
        E.g. dim=2:
        +---+---+           +---+---+
        | @ |   |  becomes  |   | # |
        +---+---+           +---+---+
        |   | # |           | @ |   |
        +---+---+           +---+---+
        """

        figs, positions = target
        new_positions=[]
        for i,position in enumerate(positions):
            x,y=self.pos2coords(position)
            x,y=[(x^1, y), (x, y^1)][axis]
            new_positions.append(self.coords2pos(x,y))
        distractor=(figs, tuple(new_positions))
        return distractor

if __name__=="__main__":
    game=SystematicDistractors(figures=['a', 'b', 'c'], allow_repetitions=False)
    print(game.targets[0])
    print(game.distractors[0])