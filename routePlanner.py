#  openrouteservice key #: 5b3ce3597851110001cf6248d4f79d4ad7b54a69853c45c7a9423c1b
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
import geopandas as gp
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


# def _df_split(tup_arg, **kwargs):
#     split_ind, df_split, df_f_name = tup_arg
#     return (split_ind, getattr(df_split, df_f_name)(**kwargs))
#
#
# def df_multi_core(df, df_f_name, subset=None, njobs=-1, **kwargs):
#     if njobs == -1:
#         njobs = multiprocessing.cpu_count()
#     pool = multiprocessing.Pool(processes=njobs)
#
#     try:
#         splits = np.array_split(df[subset], njobs)
#     except (ValueError, KeyError):
#         splits = np.array_split(df, njobs)
#
#     pool_data = [(split_ind, df_split, df_f_name) for split_ind, df_split in enumerate(splits)]
#     results = []
#     results = (pool.map(partial(_df_split, **kwargs), pool_data))
#     pool.close()
#     pool.join()
#     import pdb; pdb.set_trace()
#     results = sorted(results, key=lambda x:x[0])
#     re = pd.concat([split[1] for split in results])
#
#     return results


def linecut(line, distance):
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]


# def create_data_model():
#     """Stores the data for the problem."""
#     data = {}
#     data['distance_matrix'] = [
#         [0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],
#         [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],
#         [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],
#         [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],
#         [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371],
#         [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999],
#         [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701],
#         [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099],
#         [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600],
#         [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162],
#         [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200],
#         [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504],
#         [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0],
#     ]  # yapf: disable
#     data['num_vehicles'] = 1
#     data['depot'] = 0
#     return data


# def print_solution(manager, routing, solution):
#     """Prints solution on console."""
#     print('Objective: {} miles'.format(solution.ObjectiveValue()))
#     index = routing.Start(0)
#     plan_output = 'Route for vehicle 0:\n'
#     route_distance = 0
#     while not routing.IsEnd(index):
#         plan_output += ' {} ->'.format(manager.IndexToNode(index))
#         previous_index = index
#         index = solution.Value(routing.NextVar(index))
#         route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
#     plan_output += ' {}\n'.format(manager.IndexToNode(index))
#     print(plan_output)
#     plan_output += 'Route distance: {}meters\n'.format(route_distance)
#     return


def build_solution_list(manager, routing, solution):
    """returns list of nodes in route order"""
    routeList = np.empty((0,2), int)
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        prevIndex = index
        index = solution.Value(routing.NextVar(index))
        dist = routing.GetArcCostForVehicle(prevIndex, index, 0)
        routeList = np.append(routeList, [[node, dist]], axis=0)

    node = manager.IndexToNode(index)
    routeList = np.append(routeList, [[node, 0]], axis=0)

    return routeList


def MainRT(data):
    """Entry point of the program."""
    # Instantiate the data problem.
    # data = create_data_model()

    # Create the routing index manager.
    otStart = t.perf_counter()
    print("starting route solver setup")
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.log_search = True

    # Solve the problem.
    otSolve = t.perf_counter()
    print(f"route solver setup done in {otSolve - otStart:0.4f} seconds")
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        # print_solution(manager, routing, solution)
        routeList = build_solution_list(manager, routing, solution)

    otEnd = t.perf_counter()
    print(f"route solved done in {otEnd - otSolve:0.4f} seconds")
    return routeList


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
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
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
    gpDF = gp.GeoDataFrame(pdDF)
    gpDF.crs = {'init': 'epsg:4326'}

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


def getPolesData(csvpath):

    poles_data = pd.read_csv(csvpath)
    # "C:/QSI_Git/SphericalProcessing/MM_RoutePlanning/testData/pre_capture_full.txt"
    utm, hem, epsg = findBestUTMZone(poles_data)
    geoDeliveries = pandsToGeopandas(poles_data)
    utmDeliveries = reprojectGeoPandasDF(geoDeliveries, epsg)

    return (geoDeliveries, utmDeliveries, utm, hem, epsg)


def clusterPoles(utmGDF):
    utmGDF['label'] = [0 for _ in range(len(utmGDF))]
    utmGDF['X'] = utmGDF['geometry'].x
    utmGDF['Y'] = utmGDF['geometry'].y
    ddArray = utmGDF.to_numpy()
    locArray = np.vstack((ddArray[0:, 6], ddArray[0:, 7])).transpose()
    # clustered = OPTICS(min_samples=5, max_eps=500, cluster_method='dbscan').fit_predict(locArray)
    clustered = DBSCAN(min_samples=1, eps=150).fit_predict(locArray)
    ddArray[:, 5] = clustered[:]
    newGDF = pd.DataFrame.from_records(ddArray, columns=utmGDF.columns)
    # newGDF = newGDF.sort_values('label')
    newGDF = pandsToGeopandas(newGDF)

    return newGDF


def returnFoliumOfPoleClusters(gdfWithClusters, startLocation):
    # Plot the locations on the map with more info in the ToolTip
    m = fm.Map(location=startLocation, zoom_start=8, tiles='OpenStreetMap') #location = [45.519573, -122.672306]
    for location in gdfWithClusters.itertuples():
        tooltip = fm.map.Tooltip("<h4><b>ID {}</b></p><p>Project Line: <b>{}</b></p>".format(
            location.P_Tag, location.Prj
        ))

        fm.Marker(
            location=[location.Lat, location.Long],
            tooltip=tooltip,
            icon=BeautifyIcon(
                icon_shape='marker',
                number=int(location.Index),
                spin=True,
                text_color='red',
                background_color=ColorHash(location.label).hex,
                inner_icon_style="font-size:12px;padding-top:-5px;"
            )
        ).add_to(m)
    print("number of clusters: " + str(len(gdfWithClusters)))

    depot = startLocation

    fm.Marker(
        location=depot,
        icon=fm.Icon(color="green", icon="bus", prefix='fa'),
        setZIndexOffset=1000
    ).add_to(m)

    m.save('index.html')

    return m


def getOSMNXGraphOfBBox(nLat, sLat, eLon, wLon):
    ox.config(use_cache=True, log_console=True)
    G = ox.graph_from_bbox((nLat + 0.01), (sLat - 0.01), (eLon + 0.01), (wLon - 0.01), network_type='drive', simplify=True) #3120
    # fig, ax = ox.plot_graph(G)

    return G


def sortPoleNodesForSingleEdgesOld(polesGDF, nodesGDF):
    grpSort = []
    grpTrack = []
    grpNum = -1
    for grpOne in polesGDF.itertuples():
        grpNum += 1
        u = nodesGDF.loc[grpOne.origU]
        v = nodesGDF.loc[grpOne.origV]
        grpSort.append({'U': grpOne.origU,
                        'V': grpOne.origV,
                        'Groups': np.array([[u['y'], u['x'], u['osmid']],
                                    [v['y'], v['x'], v['osmid']]])})
        grpSort[grpNum]['Groups'] = np.append(grpSort[grpNum]['Groups'], [[grpOne.nGeometry.y, grpOne.nGeometry.x, grpOne.osmid]], axis=0)
        for grpTwo in polesGDF.itertuples():
            if (grpOne.origU == grpTwo.origU) and (grpOne.origV == grpTwo.origV) and (grpOne.osmid != grpTwo.osmid):
                grpSort[grpNum]['Groups'] = np.append(grpSort[grpNum]['Groups'], [[grpTwo.nGeometry.y, grpTwo.nGeometry.x, grpTwo.osmid]], axis=0)
    return grpSort


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
    inte = ""
    for i in range(0, len(string)):
        inte = inte + str(ord(string[i : i+1]))

    return int(inte)


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

        # edge = np.array([[g[preNode[0], 2],
        #                         g[node[0], 2],
        #                         0,
        #                         str(g[preNode[0], 2]) + "-" + str(g[node[0], 2]),
        #                         np.NaN,
        #                         np.NaN,
        #                         0,
        #                         node[1],
        #                         geometry,
        #                         np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN ]], dtype='O')
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
        route = MainOW(data).astype('int32')
        edges = applyRouteToBuildEdges(grp, route, edges)

    return edges


def buildDistMatrix(poleEdgeGrp):

    return


def adjustPolesToPointOnEdge(polesGDF, edgeGDFLUT):

    for grpIndex, grpVal in polesGDF.iterrows():
        grpLine = edgeGDFLUT.loc[(grpVal['origU'], grpVal['origV']), 'geometry'][0]
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
    cePoleGDF = cePoleGDF.drop(['Prj', 'label', 'X', 'Y'], axis='columns')
    cePoleGDF = cePoleGDF.rename(index=str, columns={'Lat': 'y', 'Long': 'x', 'P_Tag': 'osmid'})
    cePoleGDF = cePoleGDF.set_index('osmid', drop=False)
    cePoleGDF = cePoleGDF.drop_duplicates('osmid')

    #make osmnx formatted GDF for appending to OSM GDF from graph.  Remember to change 'P_Tag' as 'osmid' if new pole CSV has different columns
    print("converting graph to GeoDataFrame")
    gdfsN, gdfsE = ox.save_load.graph_to_gdfs(graph)
    gdfseLUT = gdfsE.set_index(['u', 'v']).sort_index()
    iCePolesGDF = adjustPolesToPointOnEdge(cePoleGDF, gdfseLUT)
    oxFormGDF = iCePolesGDF.drop(['origU', 'origV'], axis='columns')
    # oxFormGDF = oxFormGDF.rename(index=str, columns={'nGeometry': 'geometry'})
    # oxFormGDF = oxFormGDF.set_index('osmid', drop=False)

    gdfsN = gdfsN.append(oxFormGDF)
    # gdfsN = gdfsN.reset_index()
    # gdfsN.to_file("gdfsn.geojson", driver='GeoJSON')
    outOxFormGDF = oxFormGDF.drop(['osmid'], axis='columns')
    # outOxFormGDF.to_file("oxFormGDF.geojson", driver='GeoJSON')

    # sort poles into groups of similar edge closeness
    print("sorting common edge poles for linear inclusion")
    sortedGrps = sortPoleNodesForSingleEdges(iCePolesGDF, gdfsN)
    sortedGrpsWithDMatrix = sortPoleGroupsFromUtoV(sortedGrps)
    grpEdgesGDF = pd.DataFrame.from_records(sortedGrpsWithDMatrix, columns=['u', 'v', 'key', 'osmid', 'length', 'geometry'])
    alignedGdfsE, grpEdgesGDF = gdfsE.align(grpEdgesGDF, axis='columns')
    nGdfsE = gdfsE.append(grpEdgesGDF)
    nGdfsE = nGdfsE.reset_index(drop=True)
    nGdfsE['oneway'] = nGdfsE['oneway'].apply(lambda x: True if x == 1 else False)
    gdfsN.gdf_name = 'unnamed_nodes'
    nGdfsE.gdf_name = 'unnamed_edges'
    nG = ox.save_load.gdfs_to_graph(gdfsN, nGdfsE)

    # ox.save_graph_shapefile(nG, filename='graph')

    return (nG, oxFormGDF)


def network_distance_matrix(u, G, vs):
    # def netDist(source, target):
    #     latStart = G.nodes[source]['y']
    #     lonStart = G.nodes[source]['x']
    #     latEnd = G.nodes[target]['y']
    #     lonEnd = G.nodes[target]['x']
    #     haversineDistance(lonStart, latStart, lonEnd, latEnd)
    #     return haversineDistance(lonStart, latStart, lonEnd, latEnd)
    # dG = convertMdg2Dg(G)
    # dists = [nx.bidirectional_dijkstra(G, source=u, target=v, weight='length') for v in vs]
    # ndists = [nx.dijkstra_path_length(G, source=u, target=v, weight='length') for v in vs]
    # dists = [nx.astar_path_length(dG, source=u, target=v, heuristic=netDist, weight='length') for v in vs]

    # nd = pd.Series(ndists, index=vs)

    # ig2 = convertToIgraph(G)
    # tList = []
    # for v in vs:
    #     tList.append(ig2.vs.select(name=v)[0].index)
    # tStartDmatrix = t.perf_counter()
    u = u[0]
    dists = G.shortest_paths(source=G.vs.select(name=u)[0].index, target=vs[0], weights='length')

    d = pd.Series(dists[0], index=vs[1], name=u)
    # tEndDmatrix = t.perf_counter()
    # print(f"found shortest length for {intToPoleStr(u.iloc[0])} in {tEndDmatrix - tStartDmatrix:0.4f} seconds")
    return d


def network_path_matrix(u, G, vs):
    # paths = [nx.dijkstra_path(G, source=u, target=v, weight='length') for v in vs]
    # ig2 = convertToIgraph(G)
    # tList = []
    # for v in vs:
    #     tList.append(ig2.vs.select(name=v)[0].index)
    # tStartDmatrix = t.perf_counter()
    u = u[0]
    # import pdb; pdb.set_trace()
    paths = G.get_shortest_paths(G.vs.select(name=u)[0].index, to=vs[0], weights='length')
    # pdb.set_trace()
    paths = convertIgPaths2NxPaths(paths, G)
    # pdb.set_trace()
    p = pd.Series(paths, index=vs[1], name=u)
    # tEndDmatrix = t.perf_counter()
    # print(f"found shortest path for {intToPoleStr(u.iloc[0])} in {tEndDmatrix - tStartDmatrix:0.4f} seconds")

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


def buildGraphMatrix(graph, polesGDF):

    nodes = pd.DataFrame(polesGDF['osmid'])
    # nodes.index = nodes.values
    ig2 = convertToIgraph(graph)
    tList = [[], []]
    for v in nodes.to_numpy():
        tList[0].append(ig2.vs.select(name=v[0])[0].index)
        tList[1].append(v[0])

    node_pm = df_pool_proc(df=nodes, df_func=network_path_matrix, njobs=-1, G=ig2, vs=tList)
    node_dm = df_pool_proc(df=nodes, df_func=network_distance_matrix, njobs=-1, G=ig2, vs=tList)

    # node_pm = nodes.apply(network_path_matrix, axis='columns', G=ig2, vs=tList)
    # node_dm = nodes.apply(network_distance_matrix, axis='columns', G=ig2, vs=tList)

    node_dm = node_dm.astype(int)

    # reindex to create establishment-based net dist matrix
    ndm = node_dm.reindex(index=polesGDF['osmid'], columns=polesGDF['osmid'])
    npm = node_pm.reindex(index=polesGDF['osmid'], columns=polesGDF['osmid'])

    return ndm, npm


def buildDistGraphMatrix(graph, polesGDF):

    nodes = pd.DataFrame(polesGDF['osmid'])
    # nodes.index = nodes.values
    ig2 = convertToIgraph(graph)
    tList = [[], []]
    for v in nodes.to_numpy():
        tList[0].append(ig2.vs.select(name=v[0])[0].index)
        tList[1].append(v[0])

    # node_pm = df_pool_proc(df=nodes, df_func=network_path_matrix, njobs=-1, G=ig2, vs=tList)
    node_dm = df_pool_proc(df=nodes, df_func=network_distance_matrix, njobs=-1, G=ig2, vs=tList)

    # node_pm = nodes.apply(network_path_matrix, axis='columns', G=ig2, vs=tList)
    # node_dm = nodes.apply(network_distance_matrix, axis='columns', G=ig2, vs=tList)

    node_dm = node_dm.replace([np.inf, -np.inf], np.nan)
    node_dm = node_dm.fillna(1000000)
    node_dm = node_dm.astype(int)

    # reindex to create establishment-based net dist matrix
    ndm = node_dm.reindex(index=polesGDF['osmid'], columns=polesGDF['osmid'])
    # npm = node_pm.reindex(index=polesGDF['osmid'], columns=polesGDF['osmid'])

    return ndm


def buildWayFromSolvedMatrix(solvedRoute, graph, poles):
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

    return way


def saveRouteToShp(shpPath, graph, multiRoute):
    #gettting wrong lines and not sure why.
    mergedWays = [i[:-1] for i in multiRoute]
    mergedWay = [item for sublist in mergedWays for item in sublist]
    pairedWay = zip(mergedWay[:-1], mergedWay[1:])
    edges = ox.graph_to_gdfs(graph, nodes=False).set_index(['u', 'v']).sort_index()
    lines = [edges.loc[uv, 'geometry'].iloc[0] for uv in pairedWay]
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


def main(args):
    tMainStart = t.perf_counter()
    print("starting pole route solver")

    geoGDF, utmGDF, utm, hem, epsg = getPolesData(args.csvPath)

    tGetPoles = t.perf_counter()
    print(f"poles read and projected in {tGetPoles - tMainStart:0.4f} seconds")
    print("clustering poles")
    clusterUtmGDF = clusterPoles(utmGDF)
    clusterUtmGDF['P_Tag'] = clusterUtmGDF['P_Tag'].apply(poleStrToInt)

    tClusterPoles = t.perf_counter()
    print(f"poles clustered in {tClusterPoles - tGetPoles:0.4f} seconds")
    print("downloading and saving NX graph of pole street network")
    bb = geoGDF.total_bounds
    G = getOSMNXGraphOfBBox(bb[3], bb[1], bb[2], bb[0])
    ox.save_graphml(G, filename="origGraph.graphml")

    tNetworkDwnld = t.perf_counter()
    print(f"street network download, simplified and saved to '/data/origGraph.graphml' in {tNetworkDwnld - tClusterPoles:0.4f} seconds")
    print("inserting pole nodes into NX graph and building edge continuity")
    nG, oxFormPoles = insertPolesIntoGraph(G, clusterUtmGDF, 'P_Tag')
    ox.save_graphml(nG, filename="polesGraph.graphml")
    # ox.save_graph_shapefile(nG, filename="polesGraph")

    tPolesInsert = t.perf_counter()
    print(f"pole nodes and edges inserted into graph in {tPolesInsert - tNetworkDwnld:0.4f} seconds")
    print("Building graph distance matrix")
    dMatrix = buildDistGraphMatrix(nG, oxFormPoles)

    tDistMatrix = t.perf_counter()
    print(f"graph distance matrix built in {tDistMatrix - tPolesInsert:0.4f} seconds")
    print("Finding shortest route to visit all points")
    data = {}
    data['distance_matrix'] = dMatrix.to_numpy()
    data['num_vehicles'] = 1
    data['depot'] = 0
    route = MainRT(data).astype('int32')

    tShortestPath = t.perf_counter()
    print(f"shortest route built in {tShortestPath - tDistMatrix:0.4f} seconds")
    print("plotting and saving optimized route")
    way = buildWayFromSolvedMatrix(route, nG, oxFormPoles)
    saveRouteToShp('./data/route.shp', nG, way)
    fig, ax = ox.plot_graph_routes(nG, way)

    tWayPlotting = t.perf_counter()
    print(f"optimized route plotted and saved in {tWayPlotting - tShortestPath:0.4f} seconds")
    print(f"total script runtime {tWayPlotting - tMainStart:0.4f} seconds")

    import pdb; pdb.set_trace()

    returnFoliumOfPoleClusters(clusterUtmGDF, [45.27718945, -123.0839672])


    # Define the vehicles
    # https://openrouteservice-py.readthedocs.io/en/latest/openrouteservice.html#openrouteservice.optimization.Vehicle
    vehicles = list()
    for idx in range(3):
        vehicles.append(
            ors.optimization.Vehicle(
                id=idx,
                start=list(reversed(depot)),
                end=list(reversed(depot)),
                # capacity=[300],
                time_window=[1553241600, 1553284800]  # Fri 8-20:00, expressed in POSIX timestamp
            )
        )

    # Next define the delivery stations
    # https://openrouteservice-py.readthedocs.io/en/latest/openrouteservice.html#openrouteservice.optimization.Job
    deliveries = list()
    for delivery in utmGDF.itertuples():
        deliveries.append(
            ors.optimization.Job(
                id=delivery.Index,
                location=[delivery.Long, delivery.Lat],
                # service=1200,  # Assume 20 minutes at each site
                # amount=[delivery.Needed_Amount],
                # time_windows=[[
                #     int(delivery.Open_From.timestamp()),  # VROOM expects UNIX timestamp
                #     int(delivery.Open_To.timestamp())
                # ]]
            )
        )

    # Initialize a client and make the request
    ors_client = ors.Client(key='5b3ce3597851110001cf6248d4f79d4ad7b54a69853c45c7a9423c1b')  # Get an API key from https://openrouteservice.org/dev/#/signup
    result = ors_client.optimization(
        jobs=deliveries,
        vehicles=vehicles,
        geometry=True
    )

    # Add the output to the map
    for color, route in zip(['green', 'red', 'blue'], result['routes']):
        decoded=ors.convert.decode_polyline(route['geometry'])  # Route geometry is encoded
        gj = fm.GeoJson(
            name='Vehicle {}'.format(route['vehicle']),
            data={"type": "FeatureCollection", "features": [{"type": "Feature",
                                                             "geometry": decoded,
                                                             "properties": {"color": color}
                                                            }]},
            style_function=lambda x: {"color": x['properties']['color']}
        )
        gj.add_child(fm.Tooltip(
            """<h4>Vehicle {vehicle}</h4>
            <b>Distance</b> {distance} m <br>
            <b>Duration</b> {duration} secs
            """.format(**route)
        ))
        gj.add_to(m)

    fm.LayerControl().add_to(m)
    # m
    m.save('index.html')
    return


#get PDX data from OpenStreetMap
if __name__ == "__main__":
    import argparse as ap

    parser = ap.ArgumentParser(description='Run route optimization on poles CSV provided by client')
    argGroup = parser.add_argument_group(title='Inputs')

    argGroup.add_argument('-i', dest='csvPath', required=True, type=str,
                        help='input path to CSV of pole locations.')
    argGroup.add_argument('-id', dest='networkID', required=False, type=str,
                        help='column name to be used for network ID.  This need to be unique amoung all poles. eg. "P_Tag".')

    args = parser.parse_args()
    # exampleMain()
    main(args)
