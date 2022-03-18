#  openrouteservice key #: 5b3ce3597851110001cf6248d4f79d4ad7b54a69853c45c7a9423c1b
import os
import folium as fm
import shapely as sh
import fiona
import pandas as pd
import numpy as np
from sklearn.cluster import OPTICS, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import haversine_distances
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from folium.plugins import BeautifyIcon
import math
import pyproj
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString, mapping
from colorhash import ColorHash
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import json as js
import time as t
import igraph as ig
import progressbar as prog
import multiprocessing
from functools import partial
import sys
from tqdm import tqdm
import psutil
from contextlib import closing, redirect_stdout

def df_pool_proc(df, df_func, njobs=-1, **kwargs):
    if njobs == -1:
        njobs = multiprocessing.cpu_count() - 1
    results = []
    dfLen = len(df)

    with closing( multiprocessing.Pool(processes=njobs) ) as pool:
        results = [i for i in tqdm(pool.imap_unordered(partial((df_func), **kwargs), df.iterrows(), 5), total=dfLen)]

    pool.close()
    pool.join()

    recat = pd.concat([split for split in results], axis='columns')

    return recat


def build_solution_list(manager, routing, solution):
    """returns list of nodes in route order"""
    routeList = np.empty((0,2), int)
    index = routing.Start(0)
    dist = 0
    routeDistance = 0
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        prevIndex = index
        index = solution.Value(routing.NextVar(index))
        dist = routing.GetArcCostForVehicle(prevIndex, index, 0)
        routeDistance += dist
        routeList = np.append(routeList, [[node, dist]], axis=0)

    node = manager.IndexToNode(index)
    routeList = np.append(routeList, [[node, 0]], axis=0)

    return (routeList, routeDistance)

# using soulutionLLimit=1 for testing, 50 was DB's value 210316
#def MainRT(data, multi=False, maxUnits=0, solutionLimit=50, timeLimit=6000):
def MainRT(data, multi=False, maxUnits=0, solutionLimit=1, timeLimit=6000):

    """Entry point of the program."""
    # Instantiate the data problem.
    # data = create_data_model()

    # Create the routing index manager.
    otStart = t.perf_counter()
    print("starting route solver setup")
    print(data)
    print(data['depot'])
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)

        #if to_node == from_node:
        #print('to_node == from_node')
        #print('from_index', from_index)
        #print('to_index', to_index)

        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    #setup time / distance dimension
    if multi != False:
        dimension_name = 'Distance'
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            maxUnits,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name)

        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.solution_limit = solutionLimit
    search_parameters.time_limit.seconds = timeLimit
    search_parameters.log_search = True

    # Solve the problem.
    otSolve = t.perf_counter()
    print(f"route solver setup done in {otSolve - otStart:0.4f} seconds")
    solution = None
    if multi != False:
        while solution == None:
            solution = routing.SolveWithParameters(search_parameters)

    else:
        solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.

    if multi==False:
        if solution != None:

            # print_solution(manager, routing, solution)
            routeList, routeDistance = build_solution_list(manager, routing, solution)

            print('routeList', routeList)
            print('routeDistance', routeDistance)

            otEnd = t.perf_counter()
            print(f"route solved done in {otEnd - otSolve:0.4f} seconds")
            return (routeList, routeDistance)
        else:
            raise Exception("routing solution not found.")
    '''
    else:
        if solution != None:
            routeList, totDistList, totalDist = build_multisolution_list(data, manager, routing, solution)

            otEnd = t.perf_counter()
            print(f"route solved done in {otEnd - otSolve:0.4f} seconds")
            return (routeList, totDistList, totalDist)
        else:
            raise Exception("routing solution not found.")
    '''

def MainOW(data):
    """Entry point of the program."""
    # Instantiate the data problem.
    # data = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['starts'],
                                           data['ends'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        #print('from_node', from_node )
        #print('to_node', to_node)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.log_search = False

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        # print_solution(manager, routing, solution)
        routeList = build_solution_list(manager, routing, solution)

    #print("route_list", routeList)

    return routeList


def findBestUTMZone(df):
    #make sure to pick western-most point to generate best UTM projection
    for rIndex, row in df.iterrows():
        if rIndex == 0:
            longitude = row['Long']
            latitude = row['Lat']
            continue

        if (longitude > row['Long']):
            longitude = row['Long']

        if (latitude > row['Lat']):
            latitude = row['Lat']

    utm = math.ceil((longitude + 180.0) / 6.0)

    codes = [None, 32601, 32602, 32603, 32604, 32605,
            32606, 32607, 32608, 32609, 32610, 32611,
            32612, 32613, 32614, 32615, 32616, 32617,
            32618, 32619, 32620, 32621, 32622, 32623,
            32624, 32625, 32626, 32627, 32628, 32629,
            32630, 32631, 32632, 32633, 32634, 32635,
            32636, 32637, 32638, 32639, 32640, 32641,
            32642, 32643, 32644, 32645, 32646, 32647,
            32648, 32649, 32650, 32651, 32652, 32653,
            32654, 32655, 32656, 32657, 32658, 32659,
            32660, 32701, 32702, 32703, 32704, 32705,
            32706, 32707, 32708, 32709, 32710, 32711,
            32712, 32713, 32714, 32715, 32716, 32717,
            32718, 32719, 32720, 32721, 32722, 32723,
            32724, 32725, 32726, 32727, 32728, 32729,
            32730, 32731, 32732, 32733, 32734, 32735,
            32736, 32737, 32738, 32739, 32740, 32741,
            32742, 32743, 32744, 32745, 32746, 32747,
            32748, 32749, 32750, 32751, 32752, 32753,
            32754, 32755, 32756, 32757, 32758, 32759,
            32760]

    if latitude <= 0:
        hem = "S"
        epsg = codes[utm + 60]
    else:
        hem = "N"
        epsg = codes[utm]

    return (utm, hem, epsg)


def pandsToGeopandas(pdDF):

    pdDF['geometry'] = pdDF.apply(lambda z: Point(z.Long, z.Lat), axis=1)
    gpDF = gpd.GeoDataFrame(pdDF)
    gpDF.crs = pyproj.CRS('EPSG:4326')

    return gpDF


def haversineDistance(lonStart, latStart, lonEnd, latEnd):
    R = 6371000
    latStart = math.radians(latStart)
    latEnd = math.radians(latEnd)
    latDelta = math.radians(latEnd - latStart)
    lonDelta = math.radians(lonEnd - lonStart)

    a = (math.sin(latDelta / 2) ** 2) + (math.cos(latStart) * math.cos(latEnd)) * (math.sin(lonDelta / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c

    return d


def reprojectGeoPandasDF(gpDF, epsgCode):
    newgpDF = gpDF.to_crs(epsg=epsgCode)
    cs = pyproj.CRS.from_epsg(epsgCode)
    newgpDF.crs = cs.to_wkt(version='WKT1_ESRI')

    return newgpDF


def getPolesDataPlusDepot(in_poles_shp, dLat, dLon, nID, yID, xID):
    poles_data = gpd.read_file(in_poles_shp)

    poles_data = poles_data[['P_Tag', 'Long', 'Lat']]
    depot = pd.DataFrame([['depot', float(dLat), float(dLon)]], columns=['P_Tag', 'Lat', 'Long'])
    depot = gpd.GeoDataFrame(depot, geometry=gpd.points_from_xy(depot['Long'], depot['Lat']), crs=4326)

    poles_data = pd.concat([depot, poles_data], ignore_index=True)
    poles_data = gpd.GeoDataFrame(poles_data, geometry=gpd.points_from_xy(poles_data['Long'], poles_data['Lat']),
                                  crs=4326)

    utm, hem, epsg = findBestUTMZone(poles_data)

    geoGDF = poles_data
    utmGDF = poles_data.to_crs(epsg)

    return geoGDF, utmGDF


def clusterPoles(utmGDF):

    #utmGDF['label'] = [0 for _ in range(len(utmGDF))]
    utmGDF['X'] = utmGDF['geometry'].x
    utmGDF['Y'] = utmGDF['geometry'].y
    ddArray = utmGDF.to_numpy()
    locArray = np.vstack((utmGDF['X'], utmGDF['Y'])).transpose()
    # clustered = OPTICS(min_samples=5, max_eps=500, cluster_method='dbscan').fit_predict(locArray)
    clustered = DBSCAN(min_samples=1, eps=150).fit_predict(locArray)
    utmGDF['label'] = clustered[:]
    # newGDF = newGDF.sort_values('label')
    newGDF = pandsToGeopandas(utmGDF)
    newGDF.to_file('new_gdf.shp')

    return newGDF


def getOSMNXGraphOfBBox(nLat, sLat, eLon, wLon):
    ox.config(use_cache=True, log_console=True)
    G = ox.graph_from_bbox((nLat + 0.01), (sLat - 0.01), (eLon + 0.01), (wLon - 0.01), network_type='drive', simplify=True)
    nodes, edges =  ox.graph_to_gdfs(G)
    edges = edges.fillna(value={'maxspeed': '25 mph'})
    edges['maxspeedint'] = edges['maxspeed'].apply(lambda x: int(x.replace('mph', '')) if isinstance(x, str) else int(x[0].replace('mph', '')))
    edges['time'] = edges.apply(lambda x: round(x['length'] / (x['maxspeedint'] * 0.44704)), axis='columns')
    newG = ox.gdfs_to_graph(nodes, edges)
    #3120
    # fig, ax = ox.plot_graph(G)

    return newG



def sortPoleNodesForSingleEdges(polesGDF, nodesGDF):
    grpSort = []
    grpNum = -1
    polesGDF = polesGDF.set_index(['origU', 'origV']).sort_index()
    # polesGDF = polesGDF.drop(['geometry'], axis='columns')
    # polesGDF = polesGDF.rename(index=str, columns={'nGeometry': 'geometry'})
    for grp, indexed in polesGDF.groupby(level=[0,1]):
        grpNum += 1
        u = nodesGDF.loc[int(grp[0])]
        v = nodesGDF.loc[int(grp[1])]
        grpSort.append({'U': grp[0],
                        'V': grp[1],
                        'Groups': np.array([[u['y'], u['x'], int(u['osmid'])],
                                    [v['y'], v['x'], int(v['osmid'])]], dtype='O')})

        grpSort[grpNum]['Groups'] = np.append(grpSort[grpNum]['Groups'], np.vstack([indexed['geometry'].y, indexed['geometry'].x, indexed['osmid']]).transpose(), axis=0)
        try:
            otherGrp = polesGDF.loc[grp[1], grp[0]]
            grpSort[grpNum]['Groups'] = np.append(grpSort[grpNum]['Groups'], np.vstack([otherGrp['geometry'].y, otherGrp['geometry'].x, otherGrp['osmid']]).transpose(), axis=0)
        except:
            oGrpNum = grpNum
            grpNum += 1
            u = nodesGDF.loc[int(grp[1])]
            v = nodesGDF.loc[int(grp[0])]
            grpSort.append({'U': grp[1],
                            'V': grp[0],
                            'Groups': np.array([[u['y'], u['x'], int(u['osmid'])],
                                        [v['y'], v['x'], int(v['osmid'])]], dtype='O')})
            grpSort[grpNum]['Groups'] = np.append(grpSort[grpNum]['Groups'], grpSort[oGrpNum]['Groups'], axis=0)
            continue
        # grpSort[grpNum]['Groups'] = np.append(grpSort[grpNum]['Groups'], np.vstack([indexed['geometry'].y, indexed['geometry'].x, indexed['osmid']), axis=0)
    return grpSort


def poleStrToInt(string):
    # this creates unique ids from the pole tags
    inte = ""
    for i in range(0, len(string)):
        inte = inte + str(ord(string[i : i+1]))

    return int(inte)

'''
def intToPoleStr(integerID):
    pString = ""
    intStr = str(integerID)
    skip = False
    for i in range(0, len(intStr)):
        if skip:
            skip = False
            continue
        pString = pString + str(chr(int(intStr[i : i+2])))
        skip = True
    return pString
'''

def applyRouteToBuildEdges(grp, route, edges):
    index = -1
    routeLen = len(route)
    g = grp['Groups']
    for node in route:
        index += 1
        if index == 0:
            continue
        elif index == (routeLen-1):
            preNode = route[index - 1]
            node = route[index]
            break

        preNode = route[index - 1]
        node = route[index]
        postNode = route[index + 1]

        geometry = LineString(((float(g[preNode[0], 1]), float(g[preNode[0], 0])), (float(g[node[0], 1]), float(g[node[0], 0]))))

        edge = np.array([[g[preNode[0], 2],
                                g[node[0], 2],
                                0,
                                str(g[preNode[0], 2]) + "_" + str(g[node[0], 2]),
                                preNode[1],
                                geometry ]], dtype='O')
        edges = np.append(edges, edge, axis=0)
        preNode = node

    geometry = LineString(((float(g[preNode[0], 1]), float(g[preNode[0], 0])), (float(g[node[0], 1]), float(g[node[0], 0]))))
    edge = np.array([[g[preNode[0], 2],
                            g[node[0], 2],
                            0,
                            str(g[preNode[0], 2]) + "_" + str(g[node[0], 2]),
                            preNode[1],
                            geometry ]], dtype='O')
    edges = np.append(edges, edge, axis=0)

    return edges


def sortPoleGroupsFromUtoV(sortedGrps):
    edges = np.empty((0,6), dtype="O")
    for grp in sortedGrps:
        data = {}
        data['distance_matrix'] = haversine_distances((math.pi / 180) * grp['Groups'][0:, 0:2].astype('float64'))
        data['distance_matrix'] = np.around(data['distance_matrix'] * 6371000).astype('int32')
        data['num_vehicles'] = 1
        data['starts'] = [0]
        data['ends'] = [1]
        route = MainOW(data)[0].astype('int32')
        edges = applyRouteToBuildEdges(grp, route, edges)

    return edges


def adjustPolesToPointOnEdge(polesGDF, edgeGDFLUT):
    for grpIndex, grpVal in polesGDF.iterrows():
        grpLine = edgeGDFLUT.loc[[(grpVal['origU'], grpVal['origV'])], 'geometry'][0]
        polesGDF.at[grpIndex, 'geometry'] = grpLine.interpolate(grpLine.project(grpVal['geometry']))

    polesGDF['y'] = polesGDF['geometry'].y
    polesGDF['x'] = polesGDF['geometry'].x

    return polesGDF


def insertPolesIntoGraph(graph, poleGDF, networkID):
    #get closest edges of graph to each pole in order to connect graph edges to newly appended pole nodes, then concat the columns into pole graph
    print("finding closest edges for all pole points.  This can take a few minutes.")
    closestEdges = ox.geo_utils.get_nearest_edges(graph, Y=poleGDF['Lat'], X=poleGDF['Long'], method='balltree')
    cEdgesGDF = pd.DataFrame.from_records(closestEdges, columns=['origU', 'origV'])
    cePoleGDF = pd.concat([poleGDF, cEdgesGDF], axis=1)
    cePoleGDF = cePoleGDF.drop(['label', 'X', 'Y'], axis='columns')
    cePoleGDF = cePoleGDF.rename(index=str, columns={'Lat': 'y', 'Long': 'x', 'P_Tag': 'osmid'})
    cePoleGDF = cePoleGDF.set_index('osmid', drop=False)
    cePoleGDF = cePoleGDF.drop_duplicates('osmid')

    #make osmnx formatted GDF for appending to OSM GDF from graph.  Remember to change 'P_Tag' as 'osmid' if new pole CSV has different columns
    print("converting graph to GeoDataFrame")
    gdfsN, gdfsE = ox.save_load.graph_to_gdfs(graph)
    gdfseLUT = gdfsE.set_index(['u', 'v']).sort_index()
    iCePolesGDF = adjustPolesToPointOnEdge(cePoleGDF, gdfseLUT)
    oxFormGDF = iCePolesGDF.drop(['origU', 'origV'], axis='columns')

    oxFormGDF_slice = oxFormGDF.drop(['osmid'], axis='columns')
    oxFormGDF_slice.to_file('oxFormGDF_slice.shp')

    gdfsN = gdfsN.append(oxFormGDF)

    # outOxFormGDF = oxFormGDF.drop(['osmid'], axis='columns')


    # sort poles into groups of similar edge closeness
    print("sorting common edge poles for linear inclusion")
    sortedGrps = sortPoleNodesForSingleEdges(iCePolesGDF, gdfsN)
    sortedGrpsWithDMatrix = sortPoleGroupsFromUtoV(sortedGrps)
    grpEdgesGDF = pd.DataFrame.from_records(sortedGrpsWithDMatrix, columns=['u', 'v', 'key', 'osmid', 'length', 'geometry'])
    alignedGdfsE, grpEdgesGDF = gdfsE.align(grpEdgesGDF, axis='columns')
    nGdfsE = gdfsE.append(grpEdgesGDF)
    nGdfsE = nGdfsE.reset_index(drop=True)
    nGdfsE['oneway'] = nGdfsE['oneway'].apply(lambda x: True if x == 1 else False)
    nGdfsE = nGdfsE.fillna(value={'maxspeed': '25 mph'})
    nGdfsE['maxspeedint'] = nGdfsE['maxspeed'].apply(lambda x: int(x.replace('mph', '')) if isinstance(x, str) else int(x[0].replace('mph', '')))
    nGdfsE['time'] = nGdfsE.apply(lambda x: round(x['length'] / (x['maxspeedint'] * 0.44704)), axis='columns')
    gdfsN.gdf_name = 'unnamed_nodes'
    nGdfsE.gdf_name = 'unnamed_edges'

    #gdfsN.to_file('gdfsN.shp')

    nGdfsE_slice = nGdfsE[['u', 'v', 'length', 'geometry']]
    nGdfsE_slice.to_file('nGdfsE_slice.shp')
    nG = ox.save_load.gdfs_to_graph(gdfsN, nGdfsE)

    # filenamem param is folder name in data dir
    ox.save_graph_shapefile(nG, filename='insert_poles_into_graph')

    return nG, oxFormGDF


def network_distance_matrix(u, G, vs):
    """
            Calculates distance matrix

                    Parameters:
                            u (tuple):

                            G (igraph.Graph):

                            vs(list):

                    Returns:
                            d (pandas series): A pandas series with distances between pole nodes
        """


    print('u', u)

    u = u[0]
    dists = G.shortest_paths(source=G.vs.select(name=u)[0].index, target=vs[0], weights='length')
    d = pd.Series(dists[0], index=vs[1], name=u)

    print('d', d)

    return d


def network_time_matrix(u, G, vs):

    u = u[0]
    times = G.shortest_paths(source=G.vs.select(name=u)[0].index, target=vs[0], weights='time')
    d = pd.Series(times[0], index=vs[1], name=u)

    return d


def network_path_matrix(u, G, vs):

    u = u[0]
    # import pdb; pdb.set_trace()
    paths = G.get_shortest_paths(G.vs.select(name=u)[0].index, to=vs[0], weights='length')
    # pdb.set_trace()
    paths = convertIgPaths2NxPaths(paths, G)
    # pdb.set_trace()
    p = pd.Series(paths, index=vs[1], name=u)

    return p


def convertIgPaths2NxPaths(igPathsList, igGraph):
    L = []
    for i in igPathsList:
        p = []
        for j in i:
            p.append(igGraph.vs[j]['name'])
        L.append(p)

    return L


def convertIgPath2NxPath(igPathsList, igGraph):
    for i in igPathsList:
        p = []
        for j in i:
            p.append(igGraph.vs[j]['name'])

    return p


def convertToIgraph(nxGraph):
    g2 = ig.Graph.TupleList(nxGraph.edges(), directed=True)
    for i in g2.es:
        i['time'] = nxGraph.edges[g2.vs[i.source]['name'], g2.vs[i.target]['name'], 0]['time']
        i['length'] = nxGraph.edges[g2.vs[i.source]['name'], g2.vs[i.target]['name'], 0]['length']
        i['u'] = g2.vs[i.source]['name']
        i['v'] = g2.vs[i.target]['name']

    for j in g2.vs:
        j['x'] = nxGraph.nodes[j['name']]['x']
        j['y'] = nxGraph.nodes[j['name']]['y']

    return g2


def convertMdg2Dg(graph):
    GG = nx.DiGraph()
    for n, nbrs in graph.adjacency():
        for nbr, edict in nbrs.items():
            minvalue = min([d['length'] for d in edict.values()])
            GG.add_edge(n, nbr, weight = minvalue)

    return GG


def buildDistGraphMatrix(graph='nG', polesGDF='oxFormPoles', routeType="S"):

    """
            Calculates distance matrix

                    Parameters:
                            graph (igraph.Graph):

                            polesGDF (geopandas geodataframe):

                            routeType (str):

                    Returns:
                            ndm (pandas dataframe): A pandas dataframe distance matrix
        """

    nodes = pd.DataFrame(polesGDF['osmid'])
    # nodes.index = nodes.values
    ig2 = convertToIgraph(graph)
    tList = [[], []]
    for v in nodes.to_numpy():
        tList[0].append(ig2.vs.select(name=v[0])[0].index)
        tList[1].append(v[0])

    print('tList', tList)

    # populate the distance matrix
    if routeType == "S":
        node_dm = df_pool_proc(df=nodes, df_func=network_distance_matrix, njobs=-1, G=ig2, vs=tList)
    else:
        node_dm = df_pool_proc(df=nodes, df_func=network_time_matrix, njobs=-1, G=ig2, vs=tList)


    node_dm = node_dm.replace([np.inf, -np.inf], np.nan)
    node_dm = node_dm.fillna(1000000)
    node_dm = node_dm.astype(int)

    # reindex to create establishment-based net dist matrix
    ndm = node_dm.reindex(index=polesGDF['osmid'], columns=polesGDF['osmid'])

    print('type(ndm)', type(ndm))
    ndm.to_csv('ndm.csv')

    return ndm


def buildWayFromSolvedMatrix(solvedRoute, graph, poles, noUTurns=False):
    nodes = pd.DataFrame(poles['osmid'])
    # nodes.index = nodes.values
    ig2 = convertToIgraph(graph)
    # tList = [[], []]
    # for v in nodes.to_numpy():
    #     tList[0].append(ig2.vs.select(name=v[0])[0].index)
    #     tList[1].append(v[0])

    way = []
    count = -1
    for i in solvedRoute:
        count += 1
        if count == 0:
            end = i[0]
            continue
        start = end
        end = i[0]
        s = nodes.iloc[start]
        e = nodes.iloc[end]

        path = ig2.get_shortest_paths(ig2.vs.select(name=s[0])[0].index, to=ig2.vs.select(name=e[0])[0].index, weights='length')
        p = convertIgPath2NxPath(path, ig2)
        way.append(p)
    print('way', way)

    return way, ig2


def getTotalDistanceAndTime(pairedWay, graph):
    edges = ox.graph_to_gdfs(graph, nodes=False).set_index(['u', 'v']).sort_index()
    # totTime = np.array([edges.loc[[uv]]['time'].iloc[0] for uv in pairedWay]).sum()
    totTime = np.array([edges.loc[uv].iloc[0]['time'] for uv in pairedWay]).sum()
    # totDist = np.array([edges.loc[[uv]]['length'].iloc[0] for uv in pairedWay]).sum()
    totDist = np.array([edges.loc[uv].iloc[0]['length'] for uv in pairedWay]).sum()

    return (totTime, totDist)


def saveRouteToShp(shpPath, graph, multiRoute):
    #gettting wrong lines and not sure why.
    edges = ox.graph_to_gdfs(graph, nodes=False).set_index(['u', 'v']).sort_index()
    # lines = [edges.loc[[uv], 'geometry'].iloc[0] for uv in pairedWay]
    lines = [edges.loc[[uv], 'geometry'].iloc[0] for uv in multiRoute]
    routeLine = MultiLineString(lines)
    schema = {
    'geometry': 'MultiLineString',
    'properties': {'name': 'str'},
    }
    with fiona.open(shpPath, 'w', 'ESRI Shapefile', schema) as c:
        c.write({
            'geometry': mapping(routeLine),
            'properties': {'name': 'route'},
        })

    with fiona.open(shpPath + ".gpx", 'w', 'GPX', schema) as c:
        c.write({
            'geometry': mapping(routeLine),
            'properties': {'name': 'route'},
        })

    return


def setupRoundTrip(distMatrix, numOfCars=1):
    data = {}
    data['distance_matrix'] = distMatrix.to_numpy()
    data['num_vehicles'] = numOfCars
    data['depot'] = 0
    return data


def runSingleRouteMode(args, dMatrix, nG, oxFormPoles):
    data = setupRoundTrip(dMatrix)
    route, routeDistance = MainRT(data)
    route = route.astype('int32')
    way = buildWayFromSolvedMatrix(route, nG, oxFormPoles)
    pairedWay = makeAndCheckPairedWay(way, nG)
    totTime, totDist = getTotalDistanceAndTime(pairedWay, nG)

    return (totTime, totDist, route, way, pairedWay)


def makeAndCheckPairedWay(multiRoute, graph):
    mergedWays = [i[:-1] for i in multiRoute[0]]
    mergedWay = [item for sublist in mergedWays for item in sublist]
    pairedWay = zip(mergedWay[:-1], mergedWay[1:])
    loopPairedWay = list(pairedWay)
    listPairedWay = loopPairedWay
    edges = ox.graph_to_gdfs(graph, nodes=False).set_index(['u', 'v']).sort_index()
    index = -1
    for uv in loopPairedWay:
        index += 1
        try:
            test = edges.loc[uv].iloc[0]
        except:
            print("deleted erroneous point index ", listPairedWay.pop(index))

    print('listPairedWay', listPairedWay)
    return listPairedWay


def explode_route(route_shp):
    route_shp_gdf = gpd.read_file(route_shp)
    route_shp_gdf_sp = route_shp_gdf.explode()
    route_shp_gdf_sp.index = route_shp_gdf_sp.index.droplevel(0)
    route_shp_gdf_sp['order'] = route_shp_gdf_sp.index

    route_shp_gdf_sp.crs = 4326
    route_shp_gdf_sp['length_ft'] = route_shp_gdf_sp['geometry'].to_crs(6559).length
    route_shp_gdf_sp.to_file(route_shp + '_explode.shp')


def main(args):
    tMainStart = t.perf_counter()
    print("starting pole route solver")

    # GET GDFS FROM POLES
    geoGDF, utmGDF = getPolesDataPlusDepot(args.shp_path, args.depotY, args.depotX, args.networkID, args.YID, args.XID)

    # CLUSTER POLES
    tGetPoles = t.perf_counter()
    print(f"poles read and projected in {tGetPoles - tMainStart:0.4f} seconds")
    print("clustering poles")
    clusterUtmGDF = clusterPoles(utmGDF)
    #clusterUtmGDF['P_Tag'] = clusterUtmGDF['P_Tag'].apply(poleStrToInt)
    clusterUtmGDF['P_Tag'] = clusterUtmGDF.index


    # GET OSM DATA
    tClusterPoles = t.perf_counter()
    print(f"poles clustered in {tClusterPoles - tGetPoles:0.4f} seconds")
    print("downloading and saving NX graph of pole street network")
    bb = geoGDF.total_bounds
    G = getOSMNXGraphOfBBox(bb[3], bb[1], bb[2], bb[0])
    ox.save_graphml(G, filename="origGraph.graphml")

    orig_graph_df = pd.DataFrame.from_dict(G.nodes, orient='index')
    orig_graph_gdf = gpd.GeoDataFrame(orig_graph_df, geometry=orig_graph_df['geometry'], crs=4326)
    orig_graph_gdf.to_file('orig_graph_gdf.shp')

    # ADD POLES TO OSM DATA/GRAPH
    tNetworkDwnld = t.perf_counter()
    print(f"street network download, simplified and saved to '/data/origGraph.graphml' in {tNetworkDwnld - tClusterPoles:0.4f} seconds")
    print("inserting pole nodes into NX graph and building edge continuity")
    nG, oxFormPoles = insertPolesIntoGraph(G, clusterUtmGDF, 'P_Tag')
    ox.save_graphml(nG, filename="polesGraph.graphml")
    ox.save_graph_shapefile(nG, filename="polesGraph")


    #new_graph_df = pd.DataFrame.from_dict(nG.nodes, orient='index')
    #new_graph_gdf = gpd.GeoDataFrame(new_graph_df, geometry=new_graph_df['geometry'], crs=4326)
    #new_graph_gdf.to_file('new_graph_gdf.shp')

    tPolesInsert = t.perf_counter()
    print(f"pole nodes and edges inserted into graph in {tPolesInsert - tNetworkDwnld:0.4f} seconds")
    print("Building graph distance matrix")
    dMatrix = buildDistGraphMatrix(nG, oxFormPoles, args.routeType)

    # CALCULATE ROUTE
    tDistMatrix = t.perf_counter()
    print(f"graph distance matrix built in {tDistMatrix - tPolesInsert:0.4f} seconds")
    print("Finding shortest route to visit all points")
    #Original, Single route mode--only mode that works currently
    if args.multiDay == False and args.autoMode == False:
        totTime, totDist, route, way, pairedWay = runSingleRouteMode(args, dMatrix, nG, oxFormPoles)
        wTime = round(totTime / 3600.0, 2)
        wDist = round(totDist / 1609.34, 2)
        saveRouteToShp(args.shp_path + '_output.shp', nG, pairedWay)
        print(f"SHP written to: {args.shp_path + '_output.shp'}")
        print(f"GPX written to; {args.shp_path + '_output.shp.gpx'}")
        print(f"Total time for route = {wTime} hours")
        print(f"Total distance for route = {wDist} miles")
        fig, ax = ox.plot_graph_routes(nG, way[0])

        explode_route(args.shp_path + '_output.shp')

    tShortestPath = t.perf_counter()
    print(f"shortest route built in {tShortestPath - tDistMatrix:0.4f} seconds")

    tWayPlotting = t.perf_counter()
    print(f"optimized route plotted and saved in {tWayPlotting - tShortestPath:0.4f} seconds")
    print(f"total script runtime {tWayPlotting - tMainStart:0.4f} seconds")


if __name__ == "__main__":
    import argparse as ap


    parser = ap.ArgumentParser(description='Run route optimization on formatted poles shp provided by client')
    argGroup = parser.add_argument_group(title='Inputs')

    argGroup.add_argument('-i', dest='shp_path', required=True, type=str,
                        help='input path to formatted shp of pole locations.')
    argGroup.add_argument('-id', dest='networkID', default='P_Tag', required=True, type=str,
                        help='column name to be used for network ID.  This need to be unique amoung all poles. eg. "P_Tag".')
    argGroup.add_argument('-x', dest='XID', default='Long', required=True, type=str,
                        help='column name to be used for the Longitude values.  eg. "Long".')
    argGroup.add_argument('-y', dest='YID', default='Lat', required=True, type=str,
                        help='column name to be used for the Latitude values.  eg. "Lat".')
    argGroup.add_argument('-sx', dest='depotX', required=True, type=float,
                        help='Longitude of start / end location for round-trip routing. eg. "-122.655955".')
    argGroup.add_argument('-sy', dest='depotY', required=True, type=float,
                        help='Latitude of start / end location for round-trip routing. eg. "45.505681".')
    argGroup.add_argument('-t', dest='routeType', required=False, type=str, default="S",
                        help='type of route optimization.  Select from "S" = shortest route or "F" = fastest route.')
    argGroup.add_argument('-multi', dest='multiDay', type=int, required=False, default=False, metavar='NumOfSegments',
                        help='Option for running MANUAL multi-day optimization module.  User should designate '
                        'the number of segments (routes) the optimizer should break the job into.  '
                        'EXAMPLE: if the user wants to break the route into 10 days, the arguments would be "-multi 10".')
    argGroup.add_argument('-auto', dest='autoMode', required=False, type=int, default=False, metavar='NumOfUnits',
                        help='Use this argument to allow script to find the best number of days to break up the project based on the the maximum number of "units" for the route.  '
                        'Units are hours if the ROUTETYPE = "F" OR miles if the ROUTETYPE = "S".  Example: "-t "F" -auto 4" would find the minimum number of days to break the project up based on a 4 hour work day.'
                        'This may need to run a long time (overnight) if the project is large due to the optimizer needing to run multiple times to find the right balance of days and units.')


    args = parser.parse_args()
    main(args)