import sys
import argparse
import os
import networkx as nx
from tabulate import tabulate


def main(argv):

    parser = argparse.ArgumentParser(prog='Automated ILP Scheduler')

    parser.add_argument('-l', '--latency', type=int)
    parser.add_argument('-m', '--memory_cost', type=int)
    parser.add_argument('-g', '--graph', type=argparse.FileType('r'))
    args = parser.parse_args()

    if args.graph is None:
        print("Error: No graph input found")
        exit()


    G = nx.read_weighted_edgelist(args.graph, nodetype=int, create_using=nx.DiGraph)
    nodes=get_nodes(G)

    for node in nodes:
        parents = sorted(list(G.predecessors(node)))
        s = nodes[0]
        if not parents and node != s: 
            G.add_edge(s, node)
    
    G=add_sink_node(G)
    
    edges = list(G.edges(data=True))

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    unit_times_asap = get_asap(G)

    t = sorted(list(G.nodes()))[-1] 

    asap_latency_cstr = unit_times_asap[t] - 1

    latency_cstr = args.latency if args.latency else asap_latency_cstr 
    unit_times_alap = get_alap(G, latency_cstr)

    if args.latency is None and args.memory_cost is None:
        print("Please input latency or memory constraint using arguments -l or -m ")
        exit()
    elif not args.latency and args.memory_cost:
        schedule_obj = "ML-MC"
        for id, edge in enumerate(edges):
            node1, node2, weight = edge
            weight_value = weight.get('weight', 1)
            weight = int(weight_value)

            if args.memory_cost<weight:
                raise Exception(f'Memory constraint makes the problem infeasible, the -m value is not enough for edge', node1, 'to', node2, 'with weight', weight)  
        
    elif args.latency and not args.memory_cost:
        schedule_obj = "MM-LC"
        if latency_cstr <=asap_latency_cstr: 
            some=asap_latency_cstr+1
            raise Exception(f'Latency constraint makes the problem infeasible, the -l value has to be at least', some)
        
    elif args.latency and args.memory_cost:
        schedule_obj = "both" 

    
    if schedule_obj == "ML-MC":
        min_latency=go_to_MLMC(schedule_obj, G, args, unit_times_asap, unit_times_alap, latency_cstr)
        if min_latency == '0':
            args.latency=asap_latency_cstr+1
            schedule_obj == "MM-LC"
            min_memory=go_to_MMLC(schedule_obj, G, args, unit_times_asap, unit_times_alap, latency_cstr)
            raise Exception(f'Memory constraint makes the problem infeasible, the -m value is not enough for minimum latency possible. The memory_cost has to be at least', min_memory)
        else:
            print(f"The minimized latency is ", min_latency)
    elif schedule_obj == "MM-LC":
        min_memory=go_to_MMLC(schedule_obj, G, args, unit_times_asap, unit_times_alap, latency_cstr)
        print(f"The minimized memory is ", min_memory)
    elif schedule_obj == "both":
        go_to_pareto("both", G, args, unit_times_asap, unit_times_alap, latency_cstr, asap_latency_cstr)
   

        

    

def go_to_MLMC(schedule_obj, graph, args, unit_times_asap, unit_times_alap, latency_cstr):
    
    print("Scheduling ML-MC")
    ilp = []
    variables = []
    crit_path_nodes = []
    crit_path_length=0
    ilp.append("Minimize")
    ilp.append("  func: L")
    ilp.append("Subject To")

    line = f"  value: M = {args.memory_cost}"
    ilp.append(line)

    generate_exec_cstrs(graph, unit_times_asap, unit_times_alap, ilp, variables)
    generate_dep_cstrs(graph, unit_times_asap, unit_times_alap, ilp)

    generate_lat_cstrs(graph, unit_times_asap, unit_times_alap, ilp)
    generate_mem_cstrs(graph, unit_times_asap, unit_times_alap, ilp, variables, latency_cstr)

    ilp.append("Binary")
    variable_set = "  " + " ".join(x for x in variables)
    ilp.append(variable_set)
    ilp.append("End")
    min_latency=execute(ilp, schedule_obj, graph)
    return min_latency
    

def go_to_MMLC(schedule_obj, graph, args, unit_times_asap, unit_times_alap, latency_cstr):

    print("Scheduling MM-LC")
    ilp = []
    variables = []
    crit_path_nodes = []
    crit_path_length=0
    ilp.append("Minimize")
    ilp.append("  func: M")
    ilp.append("Subject To")

    line = f"  value: L = {args.latency}"
    ilp.append(line)

    generate_exec_cstrs(graph, unit_times_asap, unit_times_alap, ilp, variables)
    generate_dep_cstrs(graph, unit_times_asap, unit_times_alap, ilp)

    generate_lat_cstrs(graph, unit_times_asap, unit_times_alap, ilp)
    generate_mem_cstrs(graph, unit_times_asap, unit_times_alap, ilp, variables, latency_cstr)

    ilp.append("Binary")
    variable_set = "  " + " ".join(x for x in variables)
    ilp.append(variable_set)
    ilp.append("End")
    min_memory=execute(ilp, schedule_obj, graph)
    return min_memory
    

def go_to_pareto(schedule_obj, graph, args, unit_times_asap, unit_times_alap, latency_cstr, asap_latency_cstr):
    print("both")

    if latency_cstr <=asap_latency_cstr: 
        some=asap_latency_cstr+1
        raise Exception(f'Latency constraint makes the problem infeasible, the -l value has to be at least', some)
    
    edges = list(graph.edges(data=True))
    for id, edge in enumerate(edges):
        node1, node2, weight = edge
        weight_value = weight.get('weight', 1)
        weight = int(weight_value)

        if args.memory_cost<weight:
            raise Exception(f'Memory constraint makes the problem infeasible, the -m value is not enough for edge', node1, 'to', node2, 'with weight', weight)  

    schedule_obj = "ML-MC"
    min_latency=go_to_MLMC(schedule_obj, graph, args, unit_times_asap, unit_times_alap, latency_cstr)

    
    if min_latency == '0':
        args.latency=asap_latency_cstr+1
        schedule_obj == "MM-LC"
        min_memory=go_to_MMLC(schedule_obj, graph, args, unit_times_asap, unit_times_alap, latency_cstr)
        raise Exception(f'Memory constraint makes the problem infeasible, the -m value is not enough for minimum latency possible. The memory_cost has to be at least', min_memory)
    else:
        print(f"The minimized latency for memory constraint ", args.memory_cost, 'is', min_latency)
        schedule_obj = "MM-LC"
        min_memory=go_to_MMLC(schedule_obj, graph, args, unit_times_asap, unit_times_alap, latency_cstr)
        print(f"The minimized memory for latency constraint ", args.latency, 'is', min_memory)
    
    
    
def execute(ilp, schedule_obj, graph):
    print("Executing ILP")
    lp_filename = rf"auto_{schedule_obj}.lp"
    generate_file(lp_filename, ilp)
    glpsol_dir = r"../../glpk-4.35/examples/glpsol" # NOTE: assumes glpk dir is two directories up (same dir as the repo)
    output_txt = f"{lp_filename[:-3]}.txt"
    os.system(rf"{glpsol_dir} --cpxlp {lp_filename} -o {output_txt} >/dev/null 2>&1")

    min_results = {"obj": 0, "counts": {}}
    with open(output_txt) as file:
        for line in file:
            line = line.split()
            if line and line[0] == 'Objective:':
                min_results["obj"] = line[3]

    if schedule_obj == "ML-MC":
        return min_results['obj']
    elif schedule_obj == "MM-LC":
        return min_results['obj']
    

def generate_file(filename, ilp):
    with open(filename, 'w') as f:
        for s in ilp:
            f.write("%s\n" % s)


def crit_path(G, unit_times_asap, unit_times_alap, crit_path_nodes):
    crit_path_length=0
    s = sorted(list(G.nodes()))[0] 
    t = sorted(list(G.nodes()))[-1] 

    nodes = get_nodes(G)
    for id, node in enumerate(nodes):
        id = 'n' if node == t else id
        start_time = unit_times_asap[node]
        end_time = unit_times_alap[node]
        if start_time == end_time: 
            if node != s and node != t: 
                crit_path_nodes.append(node)
                crit_path_length+=1
            continue
    return crit_path_length

def generate_exec_cstrs(graph, unit_times_asap, unit_times_alap, generated_ilp, variables):

    s = sorted(list(graph.nodes()))[0] 
    t = sorted(list(graph.nodes()))[-1] 

    nodes = get_nodes(graph)

    for id, node in enumerate(nodes):
        id = 'n' if node == t else id

        start_time = unit_times_asap[node]
        end_time = unit_times_alap[node]
        exec_cstr = []
        for time in range(start_time, end_time + 1):
            if node == s:
              exec_cstr.append(f"x_{id}_{time+1}")
              variables.append(f"x_{id}_{time+1}")
            else:
              exec_cstr.append(f"x_{id}_{time}")
              variables.append(f"x_{id}_{time}")

        exec_cstr = " + ".join(x for x in exec_cstr)
        line = f"  st{id}: {exec_cstr} = 1"
        generated_ilp.append(line)

def generate_dep_cstrs(graph, unit_times_asap, unit_times_alap, generated_ilp):

    s = sorted(list(graph.nodes()))[0] 
    t = sorted(list(graph.nodes()))[-1] 

    cstr_id = 0
    nodes = get_nodes(graph)
    for id, node in enumerate(nodes):
        id = 'n' if node == t else id

        parents = sorted(list(graph.predecessors(node)))
        s = nodes[0] 
        if not parents or parents[0] == s: 
            continue

        start_time = unit_times_asap[node]
        end_time = unit_times_alap[node]
        slack = end_time - start_time

        for parent in parents:
            parent_start_time = unit_times_asap[parent]
            parent_end_time = unit_times_alap[parent]
            parent_slack = parent_end_time - parent_start_time

            dep_cstr = []
            for time in range(start_time, end_time + 1):
                dep_cstr.append(f"{time} x_{id}_{time}")
            plus_count = len(dep_cstr) - 1

            for time in range(parent_start_time, parent_end_time + 1):
                parent_id = nodes.index(parent)
                dep_cstr.append(f"{time} x_{parent_id}_{time}")

            dep_cstr = " - ".join(x for x in dep_cstr)
            dep_cstr = dep_cstr.replace('-', '+', plus_count)
            line = f"  d_{parent_id}_{id}: {dep_cstr} >= 1"
            generated_ilp.append(line)
            cstr_id += 1

def generate_lat_cstrs(graph, unit_times_asap, unit_times_alap, generated_ilp):

    s = sorted(list(graph.nodes()))[0]
    t = sorted(list(graph.nodes()))[-1] 

    cstr_id = 0
    nodes = get_nodes(graph)
    for id, node in enumerate(nodes):
        id = 'n' if node == t else id

        s = nodes[0] 

        start_time = unit_times_asap[node]
        end_time = unit_times_alap[node]
        slack = end_time - start_time

        if node==s:
          continue

        lat_cstr = []
        for time in range(start_time, end_time + 1):
            lat_cstr.append(f"{time} x_{id}_{time}")


        lat_cstr = " + ".join(x for x in lat_cstr)

        line = f"  l_{id}: {lat_cstr} - L <= 0"
        generated_ilp.append(line)


def generate_mem_cstrs(graph, unit_times_asap, unit_times_alap, generated_ilp, variables, latency_cstr):
  terms = []

  nodes = get_nodes(graph)
  edges = list(graph.edges(data=True))
  num_nodes = graph.number_of_nodes()
  num_edges = graph.number_of_edges()

  s = sorted(list(graph.nodes()))[0] 
  t = sorted(list(graph.nodes()))[-1] 

  for i in range(1, latency_cstr):
      for id, edge in enumerate(edges):
        node1, node2, weight = edge
        weight_value = weight.get('weight', 1)
        weight = int(weight_value)

        if node2 == t or node1 == s:
          continue

        start_time_node1 = unit_times_asap[node1]
        end_time_node1 = unit_times_alap[node1]
        slack_node1 = end_time_node1 - start_time_node1

        start_time_node2 = unit_times_asap[node2]
        end_time_node2 = unit_times_alap[node2]
        slack_node2 = end_time_node2 - start_time_node2


        if start_time_node1 <= i and end_time_node2 >= i+1 :
          terms.append(f"{weight} y_{node1}_{node2}_{i}")
          variables.append(f"y_{node1}_{node2}_{i}")

          mem_cstr = []
          mem_cstry1=[]
          mem_cstrz1=[]
          mem_cstr.append(f"y_{node1}_{node2}_{i}")
          mem_cstry1.append(f"y_{node1}_{node2}_{i}")
          mem_cstrz1.append(f"y_{node1}_{node2}_{i}")
          for time in range(start_time_node1, min(end_time_node1+1, i+1)):
              mem_cstr.append(f"x_{node1}_{time}")
              mem_cstry1.append(f"x_{node1}_{time}")
          for time in range(max(i+1, start_time_node2), end_time_node2+1):
              mem_cstr.append(f"x_{node2}_{time}")
              mem_cstrz1.append(f"x_{node2}_{time}")
          mem_cstrx = " - ".join(x for x in mem_cstr)
          mem_cstry = " - ".join(x for x in mem_cstry1)
          mem_cstrz = " - ".join(x for x in mem_cstrz1)

          line1 = f"  n_{node1}_{node2}_{i}_x: {mem_cstrx} >= -1"
          line2 = f"  n_{node1}_{node2}_{i}_y: {mem_cstry} <= 0"
          line3 = f"  n_{node1}_{node2}_{i}_z: {mem_cstrz} <= 0"
          generated_ilp.append(line1)
          generated_ilp.append(line2)
          generated_ilp.append(line3)

      generated_ilp.append(f"  m{i}: {' + '.join(terms)} - M <= 0")
      terms = []

def get_asap(graph):

    unit_times_asap = {}
    seen = set()

    level = 0
    s = sorted(list(graph.nodes()))[0] 
    t = sorted(list(graph.nodes()))[-1]
    unit_times_asap[s] = level
    seen.add(s)

    children = sorted(list(graph.adj[s]))
    if not children:
        raise Exception('Invalid DFG, there are no children connected to source.')
    for child in children:
        #print(child)
        dfs(graph, child, unit_times_asap, seen, level)

    if len(graph.nodes()) > len(seen):
        raise Exception('Invalid DFG, there is at least one node that is untraversable from source.')

    
    return unit_times_asap

def dfs(graph, node, unit_times, seen, level):

    level += 1
    n_level = unit_times.get(node, -1)
    if level > n_level:
        unit_times[node] = level
    seen.add(node)
    #print(node)
    children = sorted(list(graph.adj[node]))
    for child in children:
        #print(child)
        dfs(graph, child, unit_times, seen, level)


def get_alap(graph, latency_cstr):

    unit_times_alap = {}
    seen = set()

    level = latency_cstr + 1 
    t = sorted(list(graph.nodes()))[-1] 
    unit_times_alap[t] = level
    seen.add(t)

    parents = sorted(list(graph.predecessors(t)))
    if not parents:
        raise Exception('Invalid DFG, there are no parents connected to sink.')
    for parent in parents:
        dfs_reverse(graph, parent, unit_times_alap, seen, level)

    if len(graph.nodes()) > len(seen):
        raise Exception('Invalid DFG, there is at least one node that is untraversable from sink.')

    return unit_times_alap


def dfs_reverse(graph, node, unit_times, seen, level):

    level -= 1
    n_level = unit_times.get(node, float('inf'))
    if level < n_level:
        unit_times[node] = level
    seen.add(node)
    parents = sorted(list(graph.predecessors(node)))
    for parent in parents:
        dfs_reverse(graph, parent, unit_times, seen, level)


def add_sink_node(graph):
    t = sorted(list(graph.nodes()))[-1]
    t=t+1
    graph.add_node(t)
    nodes = graph.nodes
    for node in nodes:
      if node != t:
        children = sorted(list(graph.adj[node]))
        if not children:
          graph.add_edge(node,t)

    return graph

def get_nodes(graph):

    s = sorted(list(graph.nodes()))[0] 
    t = sorted(list(graph.nodes()))[-1] 
    nodes = sorted(list(graph.nodes()))
    nodes.remove(s)
    nodes.remove(t)
    nodes.insert(0, s)
    nodes.append(t)
    return nodes


def generate_mem_cstrs(graph, unit_times_asap, unit_times_alap, generated_ilp, variables, latency_cstr):
    terms = []

    nodes = get_nodes(graph)
    edges = list(graph.edges(data=True))
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    s = sorted(list(graph.nodes()))[0] 
    t = sorted(list(graph.nodes()))[-1] 


    for i in range(1, latency_cstr):
        for id, edge in enumerate(edges):
          node1, node2, weight = edge
          weight_value = weight.get('weight', 1)
          weight = int(weight_value)

          if node2 == t or node1 == s:
            continue

          start_time_node1 = unit_times_asap[node1]
          end_time_node1 = unit_times_alap[node1]
          slack_node1 = end_time_node1 - start_time_node1

          start_time_node2 = unit_times_asap[node2]
          end_time_node2 = unit_times_alap[node2]
          slack_node2 = end_time_node2 - start_time_node2


          if start_time_node1 <= i and end_time_node2 >= i+1 :
            #generated_ilp.append(f"    accepting m{i}: y_{node1}_{node2}_{i} - M <= 0 \n")
            terms.append(f"{weight} y_{node1}_{node2}_{i}")
            variables.append(f"y_{node1}_{node2}_{i}")

            mem_cstr = []
            mem_cstry1=[]
            mem_cstrz1=[]
            mem_cstr.append(f"y_{node1}_{node2}_{i}")
            mem_cstry1.append(f"y_{node1}_{node2}_{i}")
            mem_cstrz1.append(f"y_{node1}_{node2}_{i}")
            for time in range(start_time_node1, min(end_time_node1+1, i+1)):
                mem_cstr.append(f"x_{node1}_{time}")
                mem_cstry1.append(f"x_{node1}_{time}")
            for time in range(max(i+1, start_time_node2), end_time_node2+1):
                mem_cstr.append(f"x_{node2}_{time}")
                mem_cstrz1.append(f"x_{node2}_{time}")
            mem_cstrx = " - ".join(x for x in mem_cstr)
            mem_cstry = " - ".join(x for x in mem_cstry1)
            mem_cstrz = " - ".join(x for x in mem_cstrz1)

            line1 = f"  n_{node1}_{node2}_{i}_x: {mem_cstrx} >= -1"
            line2 = f"  n_{node1}_{node2}_{i}_y: {mem_cstry} <= 0"
            line3 = f"  n_{node1}_{node2}_{i}_z: {mem_cstrz} <= 0"
            generated_ilp.append(line1)
            generated_ilp.append(line2)
            generated_ilp.append(line3)

        generated_ilp.append(f"  m{i}: {' + '.join(terms)} - M <= 0")
        terms = []

if __name__ == "__main__":
    main(sys.argv[1:])



