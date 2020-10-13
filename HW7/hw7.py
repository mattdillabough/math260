# Matt Dillabough - 10/13/2020

def read_graph_data(fname):
    graph = open(fname, "r")
    names = []
    adj = []

    for line in graph.readlines():
        values = line.split(" ")
        if values[0] == 'n':
            node = values[2].rstrip()
            names.append(node)
            adj.append([])
        else:
            u = int(values[1])
            v = int(values[2].rstrip())  # Removes /n
            adj[u].append(v)

    graph.close()

    return names, adj


if __name__ == "__main__":
    nodes, adjList = read_graph_data("graph.txt")
    print(nodes)
    print(adjList)
