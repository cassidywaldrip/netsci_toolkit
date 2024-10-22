# netsci_toolkit
A place to store useful network science functions. 

## Functions

### degree_preserving_randomization(G, n_iter=1000)

Perform degree-preserving randomization on a graph. 

Inputs: 

  G : networkx.Graph
      The input graph to be randomized. The graph can be directed or 
      undirected, but it must be simple (i.e., no self-loops or parallel edges).

  n_iter : int, optional (default=1000)
      The number of edge swap iterations to perform. A higher number of 
      iterations leads to more randomization, but the degree distribution 
      remains preserved. Typically, the number of iterations should be 
      proportional to the number of edges in the graph for sufficient 
      randomization.

Outputs: 

  G_random : networkx.Graph
      A randomized graph with the same degree distribution as the original 
      graph `G`, but with a shuffled edge structure.

### get_word_frequencies_from_strings(strings)

Create a dictionary of work usage frequencies.

Inputs: 

  strings : a list of strings

Outputs: 

  word_freqs : a dict mapping words (lowercase-d) to integer frequency counts.

### bfs(explore_queue, nodes_visited, graph)

Perform a breadth-first search on a graph. 

Inputs:

  explore_queue : list of nodes that are queued up to be explored

  nodes_visited: dict of nodes visited to distance from first node

Outputs: 

  nodes_visited: dict of nodes visited to distance from first node

### dfs(explore_stack, nodes_visited, graph)

Perform a depth-first search on a graph. 

Inputs:

  explore_queue : list of nodes that are queued up to be explored

  nodes_visited: dict of nodes visited to distance from first node

Outputs: 

  nodes_visited: dict of nodes visited to distance from first node

### transit_between_stations(source, target, city_graph)

Return the sequence of stations traversed between a source and target in a graph of a city, and report they type of transit between each station.

Inputs: 

  source : node id

  target : node id

  city_graph : directed graph of cities with appropriate attributes (see function code)

Outputs: 

  trans_type : dict of station to transit type

  duration: total duration of the trip

### degree_distribution(k, number_of_bins=15, log_binning=True, density=True)

Given a degree sequence, return the y values (probability) and the x values (support) of a degree distribution that you're going to plot.

Inputs:

  k: a list of nodes' degrees

  number_of_bins (int):
      length of output vectors
  
  log_binning (bool):
      if you are plotting on a log-log axis, then this is useful
  
  density (bool):
      whether to return counts or probability density (default: True)
      Note: probability densities integrate to 1 but do not sum to 1. 

Outputs: 

  hist, bins (np.ndarray):
      probability density if density=True node counts if density=False; binned edges

### get_prop_type(value, key=None)

Performs typing and value conversion for the graph_tool PropertyMap class. If a key is provided, it also ensures the key is in a format that can be used with the PropertyMap. Returns a tuple, (type name, value, key).

Inputs: 

  value : value to convert

  key : key in a format that can be used with PropertyMap

Outputs: 

  (type name, value, key) : tuple with the type name, value, and key

### nx2gt(nxG)

Converts a networkx graph to a graph-tool graph.

Inputs:

  nxG : a NetworkX graph

Outputs: 

  gt : a graph-tool graph

### get_colorblindness_colors(hex_col, colorblind_types='all')

Generates color representations for various types of colorblindness.

Inputs: 

  hex_col (str or tuple)
      The color you wish to check, in hex code format e.g. "#ffffff" or rgb
      format e.g. (1,255,20)

  colorblind_types (str or list)
      If "all", the function returns a dictionary with all of the following:
          Protanopia - ("Dichromat" family)
              The viewer sees no red.
          Deuteranopia - ("Dichromat" family)
              The viewer sees no green.
          Tritanopia - ("Dichromat" family)
              The viewer sees no blue.
          Protanomaly - ("Anomalous Trichromat" family)
              The viewer sees low amounts of red.
          Deuteranomaly - ("Anomalous Trichromat" family).
              The viewer sees low amounts of green.
          Tritanomaly - ("Anomalous Trichromat" family).
              The viewer sees low amounts of blue.
          Achromatopsia - ("Monochromat" family)
              The viewer sees no color at all.
          Achromatomaly - ("Monochromat" family)
              The viewer sees low amounts of color.
Outputs: 

  colorblind_output (dict)
      dictionary where the keys are the type of colorblindness and the values
      are the re-colored version of your original hex_col. This also includes
      a grayscale version of the color.
