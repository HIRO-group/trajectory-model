'''All the numbers are in inches'''
class Container:
    def __init__(self, diameter_b, height, diameter_u, fill_level):
        self.diameter_b = diameter_b
        self.height = height
        self.diameter_u = diameter_u
        self.fill_level = fill_level

class WineGlass(Container):
    low_fill = 0.3
    high_fill = 0.8
    def __init__(self, fill_level):
        super().__init__(diameter_b=3, height=4, diameter_u=3, fill_level=fill_level)

class FluteGlass(Container):
    low_fill = 0.5
    high_fill = 0.8
    def __init__(self, fill_level):
        super().__init__(diameter_b=0.5, height=5, diameter_u=1.8, fill_level=fill_level)


class BasicGlass(Container):
    low_fill = 0.3
    high_fill = 0.7
    def __init__(self, fill_level):
        super().__init__(diameter_b=2.5, height=3.7, diameter_u=3.2, fill_level=fill_level)


class RibbedCup(Container):
    low_fill = 0.3
    high_fill = 0.7
    def __init__(self, fill_level):
        super().__init__(diameter_b=2.7, height=3.4, diameter_u=3.2, fill_level=fill_level)

class TallCup(Container):
    low_fill = 0.5
    high_fill = 0.8
    def __init__(self, fill_level):
        super().__init__(diameter_b=2.4, height=6, diameter_u=3, fill_level=fill_level)


class CurvyWineGlass(Container):
    low_fill = 0.3
    high_fill = 0.7
    def __init__(self, fill_level):
        super().__init__(diameter_b=2, height=3.8, diameter_u=2.9, fill_level=fill_level)


class Flask(Container):
    low_fill = 0.6
    high_fill = 0.9
    def __init__(self, fill_level):
        super().__init__(diameter_b=3, height=5, diameter_u=1, fill_level=fill_level)


class Beaker(Container):
    low_fill = 0.5
    high_fill = 0.9
    def __init__(self, fill_level):
        super().__init__(diameter_b=2, height=3, diameter_u=2, fill_level=fill_level)


class Cylinder(Container):
    low_fill = 0.2
    high_fill = 0.8
    def __init__(self, fill_level):
        super().__init__(diameter_b=1, height=7, diameter_u=1, fill_level=fill_level)
