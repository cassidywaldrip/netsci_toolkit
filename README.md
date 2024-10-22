# netsci_toolkit
A place to store useful network science functions. 

### degree_preserving_randomization(G, n_iter=1000)

Perform degree-preserving randomization on a graph. 

### get_word_frequencies_from_strings(strings)

Create a dictionary of work usage frequencies.

### bfs(explore_queue, nodes_visited, graph)

Perform a breadth-first search on a graph. 

### dfs(explore_stack, nodes_visited, graph)

Perform a depth-first search on a graph. 

### transit_between_stations(source, target, city_graph)

Return the sequence of stations traversed between a source and target in a graph of a city, and report they type of transit between each station.

### degree_distribution(k, number_of_bins=15, log_binning=True, density=True)

Given a degree sequence, return the y values (probability) and the x values (support) of a degree distribution that you're going to plot.

### Q_of_a_c(G, dict_partition, A, Q, M)

Calculate the modularity of a graph partition. 

### get_prop_type(value, key=None)

Performs typing and value conversion for the graph_tool PropertyMap class. If a key is provided, it also ensures the key is in a format that can be used with the PropertyMap. Returns a tuple, (type name, value, key).

### nx2gt(nxG)

Converts a networkx graph to a graph-tool graph.

### get_colorblindness_colors(hex_col, colorblind_types='all')

Generates color representations for various types of colorblindness.
