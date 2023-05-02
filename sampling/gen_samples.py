import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import List, Optional, Tuple
from skopt.space import Space
from skopt.sampler import Sobol
from skopt.sampler import Lhs
from skopt.sampler import Halton

def gen_lhs_sample(n_samples:int = 16, \
                space: List[Tuple[float, float] or Tuple[int, int]] \
                    = [(0.4, 2.2), (0.4, 0.9)], \
                criterion:int = 0, lhs_type:int = 1, \
                seed:int = 42) -> List[tuple[int or float, int or float]]:
    '''
    n_samples: number of samples  
    space: sample space. If inputs are integer sampled data points will be 
    integers  
    criterion: 0) maxmin, 1) correlation, 2) ratio, default is 0 
    // Input to Lhs function  
    lhs_type: 0) classic, 1) centered, default is 1  
    seed: default is 42  
      
    Returns the list of the sampled data points 
    '''
    space = Space(space)
    # Crietrion: maximin, correlation, ratio
    # lhs_type: classic, centered
    criterions = ["maximin", "correlation", "ratio"]
    lhs_types = ["classic", "centered"]
    lhs = Lhs(criterion=criterions[criterion], iterations=50000, \
             lhs_type=lhs_types[lhs_type])
    samples = lhs.generate(space.dimensions, n_samples, random_state = seed)
    return samples

def gen_halton_sample(n_samples = 16, space = [(0.4, 2.2), (0.4, 0.9)], \
                    seed = 42):
    space = Space(space)
    halton = Halton(min_skip = 25, max_skip = 100)
    samples = halton.generate(space.dimensions, n_samples, random_state = seed)
    return samples

def gen_sobol_sample(n_samples = 16, space = [(0.4, 2.2), (0.4, 0.9)], \
                seed = 42):
    space = Space(space)
    sobol = Sobol()
    samples = sobol.generate(space.dimensions, n_samples, random_state = seed)
    return samples

def plot_sample(data: List[Tuple[int or float, int or float]], 
                label: str,
                color: str,
                ax: plt.Axes = None,
                x_range: List[int or float] = None,
                y_range: List[int or float] = None,
                x_mark: List[int or float] = None,
                y_mark: List[int or float] = None,
                x_label: str = None,
                y_label: str = None,
                title: Optional[str] = None ) -> plt.Axes:
    
    x_data: List[int or float] = []
    y_data: List[int or float] = []
    
    for x, y in data:
        x_data.append(x)
        y_data.append(y)
    
    if ax == None:
        plt.rcParams.update({'font.size': 16})
        fig = plt.figure(constrained_layout=True,figsize=(8,5),dpi=600)
        gs = fig.add_gridspec(ncols=1, nrows=1, figure = fig)
        ax = fig.add_subplot(gs[0,0])
    
    if x_mark != None:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_mark[0]))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(x_mark[1]))
    
    if y_mark != None:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_mark[0]))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(y_mark[1]))
    
    if x_range != None:
        ax.set_xlim(x_range)
    
    if y_range != None:
        ax.set_ylim(y_range)
    
    if x_label != None:
        ax.set_xlabel(x_label)
    
    if y_label != None:
        ax.set_ylabel(y_label)
    
    if title != None:
        ax.set_title(title)
    
    ax.plot(x_data, y_data, color, label = label)
    # ax.grid()
    # ax.legend()
    
    return ax
 
def check_overlap(sample1, sample2):
    sample = set()
    for x, y in sample1:
        sample.add((x, y))
    
    for x, y in sample2:
        if (x, y) in sample:
            print(f'{x}, {y} duplicate')
    
    return

if __name__ == '__main__':
    space = [(0.2, 1.5), (0.2, 0.6)]
    x = gen_lhs_sample(10, space)
    x1 = []
    for cp, util in x:
        print(round(cp, 6), round(util, 6))
        x1.append([int(1000.0/cp), round(util, 6)])
    
    print(x1)
    x = gen_halton_sample(30)
    x1 = []
    for cp, util in x:
        print(round(cp, 6), round(util, 6))
        x1.append([int(1000.0/cp), round(util, 6)])
    
    print(x1)