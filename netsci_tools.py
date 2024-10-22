import networkx as nx
import numpy as np
from collections import Counter
import graph_tool.all as gt
from bs4 import BeautifulSoup
import requests

def degree_preserving_randomization(G, n_iter=1000):
    """
    Perform degree-preserving randomization on a graph.

    Degree-preserving randomization, also known as edge swapping or rewiring, 
    is a method for creating randomized versions of a graph while preserving 
    the degree distribution of each node. This is achieved by repeatedly 
    swapping pairs of edges in the graph, ensuring that the degree (number of 
    edges connected) of each node remains unchanged. The result is a graph 
    with the same degree distribution but a randomized edge structure, which 
    can be used as a null model to compare with the original network.

    Parameters
    ----------
    G : networkx.Graph
        The input graph to be randomized. The graph can be directed or 
        undirected, but it must be simple (i.e., no self-loops or parallel edges).

    n_iter : int, optional (default=1000)
        The number of edge swap iterations to perform. A higher number of 
        iterations leads to more randomization, but the degree distribution 
        remains preserved. Typically, the number of iterations should be 
        proportional to the number of edges in the graph for sufficient 
        randomization.

    Returns
    -------
    G_random : networkx.Graph
        A randomized graph with the same degree distribution as the original 
        graph `G`, but with a shuffled edge structure.

    Notes
    -----
    - This method works by selecting two edges at random, say (u, v) and (x, y), 
      and attempting to swap them to (u, y) and (x, v) (or (u, x) and (v, y)), 
      ensuring that no self-loops or parallel edges are created in the process.
    - Degree-preserving randomization is particularly useful for creating null 
      models in network analysis, as it allows for the investigation of whether 
      specific network properties (e.g., clustering, path lengths) are a result 
      of the network's structure or just its degree distribution.
    - The effectiveness of randomization depends on the number of iterations 
      (`n_iter`). As a rule of thumb, using about 10 times the number of edges 
      in the graph for `n_iter` often provides sufficient randomization.
    
    Example
    -------
    >>> import networkx as nx
    >>> G = nx.erdos_renyi_graph(10, 0.5)
    >>> G_random = degree_preserving_randomization(G, n_iter=100)
    
    Citations
    ---------
    Milo, R., Shen-Orr, S., Itzkovitz, S., Kashtan, N., Chklovskii, D., & Alon, U. (2002). 
    Network motifs: simple building blocks of complex networks. *Science*, 298(5594), 824-827.
    
    Maslov, S., & Sneppen, K. (2002). Specificity and stability in topology of protein networks. 
    *Science*, 296(5569), 910-913.
    """

    G_random = G.copy()
    edges = list(G_random.edges())
    num_edges = len(edges)

    for _ in range(n_iter):
        # Select two random edges (u, v) and (x, y)
        edge1_id = np.random.choice(list(range(len(edges))))
        u, v = edges[edge1_id]
        edge2_id = np.random.choice(list(range(len(edges))))
        x, y = edges[edge2_id]

        # Avoid selecting the same edge pair or creating self-loops
        if len({u, v, x, y}) == 4:
            # Swap the edges with some probability
            if np.random.rand() > 0.5:
                # Swap (u, v) with (u, y) and (x, v)
                if not (G_random.has_edge(u, y) or G_random.has_edge(x, v)):
                    G_random.remove_edge(u, v)
                    G_random.remove_edge(x, y)
                    G_random.add_edge(u, y)
                    G_random.add_edge(x, v)
            else:
                # Swap (u, v) with (u, x) and (v, y)
                if not (G_random.has_edge(u, x) or G_random.has_edge(v, y)):
                    G_random.remove_edge(u, v)
                    G_random.remove_edge(x, y)
                    G_random.add_edge(u, x)
                    G_random.add_edge(v, y)

        # Update edge list after changes
        edges = list(G_random.edges())


    return G_random

def get_word_frequencies_from_strings(strings):
    """
    Given a list of strings, return a dict of word usage frequencies. 
    
    Input: strings, a list of strings
    Output: word_freqs, a dict mapping words (lowercased) to integer frequency counts.
    """

    STOPWORDS = set()
    with open('data/stopwords-en.txt', 'r') as f:
        for line in f.readlines():
            STOPWORDS.add(line.strip())

    word_counts = {} # word -> number of occurences

    words = []
    for d in strings:
        d_words = d.split()
        for w in d_words:
            words.append(w)

    for s in STOPWORDS:
        words = [i for i in words if i != s] 

    return Counter(words)

def bfs(explore_queue, nodes_visited, graph):
    if len(explore_queue) == 0:
        return nodes_visited
    else:
        current_node = explore_queue.pop(0)
        print('visiting node ' + str(current_node))
        for neighbor in graph.neighbors(current_node):
            if neighbor in nodes_visited:
                continue
            else:
                nodes_visited[neighbor] = nodes_visited[current_node] + 1
                explore_queue.append(neighbor)
        return bfs(explore_queue, nodes_visited, graph)
    
def dfs(explore_stack, nodes_visited, graph):
    if len(explore_stack) == 0:
        return nodes_visited
    else:
        current_node = explore_stack.pop(-1)
        print('visiting node {}'.format(str(current_node)))
        for neighbor in graph.neighbors(current_node):
            if neighbor in nodes_visited:
                continue
            else:
                nodes_visited[neighbor] = nodes_visited[current_node] + 1
                explore_stack.append(neighbor)
        return dfs(explore_stack, nodes_visited, graph)

def transit_between_stations(source, target, city_graph):
    """
    Given node IDs source and target (integers),
    find a path in city_graph (networkx directed graph) between the two stations.
    Indicate which transit type is used for each edge traversed, and indicate the total transit time. 
    Raise errors if the node IDs are invalid or no path exists.
    """
    try: 
        dist, path = nx.single_source_dijkstra(city_graph, source, target)
    except Exception as e:
        print(str(e))
        return

    trans_type = {}
    duration = 0
    for i in range(len(path) - 1):
        trans_type[city_graph.nodes[path[i]]['name']] = city_graph.edges[path[i], path[i+1]]['transit_type']
        duration += int(city_graph.edges[path[i], path[i+1]]['duration'])
    
    return trans_type, duration

def degree_distribution(k, number_of_bins=15, log_binning=True, density=True):
    """
    Given a degree sequence, return the y values (probability) and the
    x values (support) of a degree distribution that you're going to plot.
    
    Parameters
    ----------
    k: a list of nodes' degrees

    number_of_bins (int):
        length of output vectors
    
    log_binning (bool):
        if you are plotting on a log-log axis, then this is useful
    
    density (bool):
        whether to return counts or probability density (default: True)
        Note: probability densities integrate to 1 but do not sum to 1. 
        
    Returns
    -------
    hist, bins (np.ndarray):
        probability density if density=True node counts if density=False; binned edges
    
    """
    
    kmax = np.max(k)                    # get the maximum degree
    
    
    # Step 2: Then we'll need to construct bins
    if log_binning:
        # array of bin edges including rightmost and leftmost
        bins = np.logspace(0,np.log10(kmax+1),number_of_bins+1)
        bin_edges = []
        for ix in range(len(bins) - 1):
            bin_edges.append(np.exp((np.log(bins[ix])+np.log(bins[ix + 1]))/2))
    else:
        bins = np.linspace(0,kmax+1,num=number_of_bins+1)
        bin_edges = []
        for ix in range(len(bins) - 1):
            bin_edges.append((bins[ix] + bins[ix + 1]) / 2)
    # Step 3: Then we can compute the histogram using numpy
    hist, _ = np.histogram(k,bins,density=density)

    return bin_edges, hist

def get_prop_type(value, key=None):
    """
    Performs typing and value conversion for the graph_tool PropertyMap class.
    If a key is provided, it also ensures the key is in a format that can be
    used with the PropertyMap. Returns a tuple, (type name, value, key).

    This function is adapted from Benjamin Bengfort's blog post!
    https://bbengfort.github.io/2016/06/graph-tool-from-networkx/
    """
    if isinstance(key, str):  # Keep the key as a string
        key = key  # No encoding necessary in Python 3

    # Deal with the value
    if isinstance(value, bool):
        tname = 'bool'

    elif isinstance(value, int):
        tname = 'float'
        value = float(value)

    elif isinstance(value, float):
        tname = 'float'

    elif isinstance(value, str):
        tname = 'string'
        
    elif isinstance(value, bytes):
        tname = 'bytes'

    elif isinstance(value, dict):
        tname = 'object'

    else:
        tname = 'string'
        value = str(value)

    return tname, value, key


def nx2gt(nxG):
    """
    Converts a networkx graph to a graph-tool graph.
    """
    # Phase 0: Create a directed or undirected graph-tool Graph
    gtG = gt.Graph(directed=nxG.is_directed())

    # Add the Graph properties as "internal properties"
    for key, value in nxG.graph.items():
        # Convert the value and key into a type for graph-tool
        tname, value, key = get_prop_type(value, key)

        prop = gtG.new_graph_property(tname) # Create the PropertyMap
        gtG.graph_properties[key] = prop     # Set the PropertyMap
        gtG.graph_properties[key] = value    # Set the actual value

    # Phase 1: Add the vertex and edge property maps
    # Go through all nodes and edges and add seen properties
    # Add the node properties first
    nprops = set() # cache keys to only add properties once
    for node, data in nxG.nodes(data=True):

        # Go through all the properties if not seen and add them.
        for key, val in data.items():
            if key in nprops: continue # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key  = get_prop_type(val, key)

            prop = gtG.new_vertex_property(tname) # Create the PropertyMap
            gtG.vertex_properties[key] = prop     # Set the PropertyMap

            # Add the key to the already seen properties
            nprops.add(key)

    # Also add the node id: in NetworkX a node can be any hashable type, but
    # in graph-tool node are defined as indices. So we capture any strings
    # in a special PropertyMap called 'id' -- modify as needed!
    gtG.vertex_properties['id'] = gtG.new_vertex_property('string')

    # Add the edge properties second
    eprops = set() # cache keys to only add properties once
    for src, dst, data in nxG.edges(data=True):

        # Go through all the edge properties if not seen and add them.
        for key, val in data.items():
            if key in eprops: continue # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key = get_prop_type(val, key)

            prop = gtG.new_edge_property(tname) # Create the PropertyMap
            gtG.edge_properties[key] = prop     # Set the PropertyMap

            # Add the key to the already seen properties
            eprops.add(key)

    # Phase 2: Actually add all the nodes and vertices with their properties
    # Add the nodes
    vertices = {} # vertex mapping for tracking edges later
    for node, data in nxG.nodes(data=True):

        # Create the vertex and annotate for our edges later
        v = gtG.add_vertex()
        vertices[node] = v

        # Set the vertex properties, not forgetting the id property
        data['id'] = str(node)
        for key, value in data.items():
            gtG.vp[key][v] = value # vp is short for vertex_properties

    # Add the edges
    for src, dst, data in nxG.edges(data=True):

        # Look up the vertex structs from our vertices mapping and add edge.
        e = gtG.add_edge(vertices[src], vertices[dst])

        # Add the edge properties
        for key, value in data.items():
            gtG.ep[key][e] = value # ep is short for edge_properties

    # Done, finally!
    return gtG

namedColors = {'aliceblue':'#f0f8ff', 'antiquewhite':'#faebd7', 'aqua':'#0ff',
               'aquamarine':'#7fffd4', 'azure':'#f0ffff', 'beige':'#f5f5dc', 'bisque':'#ffe4c4',
               'black':'#000', 'blanchedalmond':'#ffebcd', 'blue':'#00f', 'blueviolet':'#8a2be2',
               'brown':'#a52a2a', 'burlywood':'#deb887', 'cadetblue':'#5f9ea0', 'chartreuse':'#7fff00',
               'chocolate':'#d2691e', 'coral':'#ff7f50', 'cornflowerblue':'#6495ed', 'cornsilk':'#fff8dc',
               'crimson':'#dc143c', 'cyan':'#0ff', 'darkblue':'#00008b', 'darkcyan':'#008b8b',
               'darkgoldenrod':'#b8860b', 'darkgray':'#a9a9a9', 'darkgrey':'#a9a9a9', 'darkgreen':'#006400',
               'darkkhaki':'#bdb76b', 'darkmagenta':'#8b008b', 'darkolivegreen':'#556b2f', 'darkorange':'#ff8c00',
               'darkorchid':'#9932cc', 'darkred':'#8b0000', 'darksalmon':'#e9967a', 'darkseagreen':'#8fbc8f',
               'darkslateblue':'#483d8b', 'darkslategray':'#2f4f4f', 'darkslategrey':'#2f4f4f',
               'darkturquoise':'#00ced1', 'darkviolet':'#9400d3', 'deeppink':'#ff1493', 'deepskyblue':'#00bfff',
               'dimgray':'#696969', 'dimgrey':'#696969', 'dodgerblue':'#1e90ff', 'firebrick':'#b22222',
               'floralwhite':'#fffaf0', 'forestgreen':'#228b22', 'fuchsia':'#f0f', 'gainsboro':'#dcdcdc',
               'ghostwhite':'#f8f8ff', 'gold':'#ffd700', 'goldenrod':'#daa520', 'gray':'#808080', 'grey':'#808080',
               'green':'#008000', 'greenyellow':'#adff2f', 'honeydew':'#f0fff0', 'hotpink':'#ff69b4',
               'indianred':'#cd5c5c', 'indigo':'#4b0082', 'ivory':'#fffff0', 'khaki':'#f0e68c', 'lavender':'#e6e6fa',
               'lavenderblush':'#fff0f5', 'lawngreen':'#7cfc00', 'lemonchiffon':'#fffacd', 'lightblue':'#add8e6',
               'lightcoral':'#f08080', 'lightcyan':'#e0ffff', 'lightgoldenrodyellow':'#fafad2', 'lightgray':'#d3d3d3',
               'lightgrey':'#d3d3d3', 'lightgreen':'#90ee90', 'lightpink':'#ffb6c1', 'lightsalmon':'#ffa07a',
               'lightseagreen':'#20b2aa', 'lightskyblue':'#87cefa', 'lightslategray':'#789', 'lightslategrey':'#789',
               'lightsteelblue':'#b0c4de', 'lightyellow':'#ffffe0', 'lime':'#0f0', 'limegreen':'#32cd32',
               'linen':'#faf0e6', 'magenta':'#f0f', 'maroon':'#800000', 'mediumaquamarine':'#66cdaa',
               'mediumblue':'#0000cd', 'mediumorchid':'#ba55d3', 'mediumpurple':'#9370d8', 'mediumseagreen':'#3cb371',
               'mediumslateblue':'#7b68ee', 'mediumspringgreen':'#00fa9a', 'mediumturquoise':'#48d1cc',
               'mediumvioletred':'#c71585', 'midnightblue':'#191970', 'mintcream':'#f5fffa', 'mistyrose':'#ffe4e1',
               'moccasin':'#ffe4b5', 'navajowhite':'#ffdead', 'navy':'#000080', 'oldlace':'#fdf5e6', 'olive':'#808000',
               'olivedrab':'#6b8e23', 'orange':'#ffa500', 'orangered':'#ff4500', 'orchid':'#da70d6',
               'palegoldenrod':'#eee8aa', 'palegreen':'#98fb98', 'paleturquoise':'#afeeee', 'palevioletred':'#d87093',
               'papayawhip':'#ffefd5', 'peachpuff':'#ffdab9', 'peru':'#cd853f', 'pink':'#ffc0cb', 'plum':'#dda0dd',
               'powderblue':'#b0e0e6', 'purple':'#800080', 'rebeccapurple':'#639', 'red':'#f00', 'rosybrown':'#bc8f8f',
               'royalblue':'#4169e1', 'saddlebrown':'#8b4513', 'salmon':'#fa8072', 'sandybrown':'#f4a460',
               'seagreen':'#2e8b57', 'seashell':'#fff5ee', 'sienna':'#a0522d', 'silver':'#c0c0c0', 'skyblue':'#87ceeb',
               'slateblue':'#6a5acd', 'slategray':'#708090', 'slategrey':'#708090', 'snow':'#fffafa',
               'springgreen':'#00ff7f', 'steelblue':'#4682b4', 'tan':'#d2b48c', 'teal':'#008080', 'thistle':'#d8bfd8',
               'tomato':'#ff6347', 'turquoise':'#40e0d0', 'violet':'#ee82ee', 'wheat':'#f5deb3', 'white':'#fff',
               'whitesmoke':'#f5f5f5', 'yellow':'#ff0', 'yellowgreen':'#9acd32'}

colorblind_mappings = {'Protanopia':'Dichromacy',
                       'Deuteranopia':'Dichromacy',
                       'Tritanopia':'Dichromacy',
                       'Protanomaly':'Trichromacy',
                       'Deuteranomaly':'Trichromacy',
                       'Tritanomaly':'Trichromacy',
                       'Achromatopsia':'Monochromacy',
                       'Achromatomaly':'Monochromacy'}

def rgb_to_hex(rgb):
    """
    Converts an RGB color to hex format.

    Parameters
    ----------
    rgb : tuple of ints
        A tuple containing the RGB values (R, G, B) where each value is in the
        range 0 to 255.

    Returns
    -------
    str
        The hex code representation of the RGB color.
    """
    r,g,b=rgb

    return '#%02x%02x%02x' % (r,g,b)

def hex_to_rgb(value):
    """
    Converts a hex color code to an RGB tuple.

    Parameters
    ----------
    value : str
        A hex code representing the color (e.g., "#ffffff").

    Returns
    -------
    tuple of ints
        A tuple containing the RGB values (R, G, B) where each value is in the range 0 to 255.
    """
    value = value.lstrip('#')
    lv = len(value)

    return tuple(int(value[i:i+lv//3], 16) for i in range(0, lv, lv//3))

def hex_to_grayscale(hex_col):
    """
    Converts a hex color code to its grayscale equivalent.

    Parameters
    ----------
    hex_col : str
        A hex code representing the color (e.g., "#ffffff").

    Returns
    -------
    str
        The grayscale value of the color as a normalized float (0.0 to 1.0).
    """
    img = hex_to_rgb(hex_col)
    R, G, B = img
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B    

    return '%.7f'%(imgGray/255)

def get_colorblindness_colors(hex_col, colorblind_types='all'):
    """
    Generates color representations for various types of colorblindness.

    Parameters
    ----------
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

    Returns
    -------
    colorblind_output (dict)
        dictionary where the keys are the type of colorblindness and the values
        are the re-colored version of your original hex_col. This also includes
        a grayscale version of the color.
    """
    all_vals = ['Original Color', 'Protanopia', 'Deuteranopia',
                'Protanomaly', 'Deuteranomaly',
                'Achromatopsia', 'Achromatomaly', 'Grayscale']

    if type(hex_col)!=str:
        if len(hex_col)!=3:
            print('Input a hex color please.')
            return ''
        else:
            hex_col = rgb_to_hex(hex_col)
    else:
        if "#" not in hex_col and len(hex_col)!=6:
            try:
                hex_col = namedColors[hex_col]
            except:
                print('Input a hex color please.')
                return ''

    base_url = 'https://convertingcolors.com/'
    hex_url = base_url + 'hex-color-%s.html'%hex_col.replace("#",'')
    print(hex_url)
    reqs = requests.get(hex_url)
    soup = BeautifulSoup(reqs.text, 'html.parser')

    colorblind_sec = soup.find_all('details',{'id':'blindness-simulation'})[0]
    colorblind_labels = [i.text for i in colorblind_sec.find_all('h3')]
    # colorblind_colors = np.unique([i.text for i in colorblind_sec.find_all('div')])
    tmp = np.unique([i.text for i in colorblind_sec.find_all('div')])
    colorblind_colors = [i for i in tmp for x in all_vals[1:] if x in i and all_vals[0] not in i]

    colorblind_output = {"Original Color":hex_col}

    # for i in colorblind_mappings.keys():
    #     for j in colorblind_colors:
    #         # if i in j:
    #         hex_col_j = j
    #         colorblind_output[i] = hex_col_j
    for i in colorblind_mappings.keys():
        for j in colorblind_colors:
            if i in j:
                hex_col_j = "#"+j.split('%')[-1]
                # hex_col_j = j.replace(i,'#')
                colorblind_output[i] = hex_col_j

    if colorblind_types!='all':
        if type(colorblind_types) == str:
            colorblind_types = [colorblind_types]

        new_out = {'Original Color':hex_col}
        for c in colorblind_types:
            new_out[c] = colorblind_output[c]

        colorblind_output = new_out

    colorblind_output['Grayscale'] = hex_to_grayscale(hex_col)
    for xx in all_vals:
        if xx not in list(colorblind_output.keys()):
            colorblind_output[xx] = hex_col

    return {hex_col:colorblind_output}