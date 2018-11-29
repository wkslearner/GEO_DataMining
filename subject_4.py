#!/usr/bin/python
# encoding=utf-8


'''获取节点间的关系字典'''
def node_relation(node_list,edge_list):
    '''
    :param node_list:
    :param edge_list:
    :return:
    '''
    relation_dict={}
    for node in node_list:
        edge_temp = []
        for edge in edge_list:
            edge_node_a,edge_node_b=edge
            if node==edge_node_a:
                edge_temp.append(edge_node_b)
            elif node==edge_node_b:
                edge_temp.append(edge_node_a)

        relation_dict[node]=edge_temp

    print(relation_dict)
    return relation_dict


'''深度优先搜索函数'''
def deep_first_search(node_list,edge_list,begin_vertex):
    '''
    :param node_list:
    :param edge_list:
    :param begin_vertex:
    :return:
    '''
    search_path=[];searched_node=[]
    searched_node.append(begin_vertex)
    relation_dict = node_relation(node_list, edge_list)

    while searched_node:
        node=searched_node.pop()
        search_path.append(node)
        for rl_node in relation_dict[node]:
            if rl_node not in search_path and rl_node not in searched_node:
                searched_node.append(rl_node)

    return search_path


if __name__ == "__main__":
    nodes = [i+1 for i in range(10)]
    edges=[(1, 2),(1, 3),(2, 4),(2, 5),(4, 8),(3, 6),(6, 7),
           (8,9),(7,10)]

    search_path=deep_first_search(nodes,edges,5)
    print(search_path)