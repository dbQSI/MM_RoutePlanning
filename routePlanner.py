#  openrouteservice key #: 5b3ce3597851110001cf6248d4f79d4ad7b54a69853c45c7a9423c1b
import folium as fm
import shapely as sh
import pandas as pd
import numpy as np
from sklearn.cluster import OPTICS, DBSCAN
from sklearn.neighbors import NearestNeighbors
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
# import openrouteservice as ors
from folium.plugins import BeautifyIcon
import math
import pyproj
import geopandas as gp
from shapely.geometry import Point
from colorhash import ColorHash
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import json as js


def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = [
        [0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],
        [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],
        [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],
        [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],
        [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371],
        [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999],
        [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701],
        [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099],
        [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600],
        [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162],
        [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200],
        [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504],
        [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0],
    ]  # yapf: disable
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data


def print_solution(manager, routing, solution):
    """Prints solution on console."""
    print('Objective: {} miles'.format(solution.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = 'Route for vehicle 0:\n'
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += ' {}\n'.format(manager.IndexToNode(index))
    print(plan_output)
    plan_output += 'Route distance: {}miles\n'.format(route_distance)


def exampleMain():
    """Entry point of the program."""
    # Instantiate the data problem.
    data = create_data_model()

    # Create the routing index manager.
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

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(manager, routing, solution)



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
    newGDF = newGDF.sort_values('label')
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
    G = ox.graph_from_bbox(nLat, sLat, eLon, wLon, network_type='drive', simplify=True) #3120
    # fig, ax = ox.plot_graph(G)

    return G


def sortPoleNodesForSingleEdges(polesGDF, nodesGDF):
    grpSort = []
    grpNum = -1
    for grpOne in polesGDF.itertuples():
        grpNum += 1
        grpSort.append({'U': grpOne.origU, 'V': grpOne.origV, 'Groups': [], 'uN': nodesGDF.loc[grpOne.origU], 'vN': nodesGDF.loc[grpOne.origV]})
        grpSort[grpNum]['Groups'].append(grpOne)
        for grpTwo in polesGDF.itertuples():
            if (grpOne.origU == grpTwo.origU) and (grpOne.origV == grpTwo.origV):
                grpSort[grpNum]['Groups'].append(grpTwo)
    return grpSort


def buildDistMatrix(poleEdgeGrp):

    return


def insertPolesIntoGraph(graph, poleGDF, networkID):
    #make osmnx formatted GDF for appending to OSM GDF from graph.  Remember to change 'P_Tag' as 'osmid' if new pole CSV has different columns
    print("converting graph to GeoDataFrame")
    oxFormGDF = poleGDF.drop(['Prj', 'label', 'X', 'Y'], axis='columns')
    oxFormGDF = oxFormGDF.rename(index=str, columns={'Lat': 'y', 'Long': 'x', 'P_Tag': 'osmid'})
    oxFormGDF = oxFormGDF.set_index('osmid', drop=False)
    gdfsN, gdfsE = ox.save_load.graph_to_gdfs(graph)
    gdfsN = gdfsN.append(oxFormGDF)
    #get closest edges of graph to each pole in order to connect graph edges to newly appended pole nodes, then concat the columns into pole graph
    print("finding closest edges for all pole points.  This can take a few minutes.")
    closestEdges = ox.geo_utils.get_nearest_edges(graph, Y=poleGDF['Lat'], X=poleGDF['Long'], method='balltree')
    cEdgesGDF = pd.DataFrame.from_records(closestEdges, columns=['origU', 'origV'])
    poleGDF = pd.concat([poleGDF, cEdgesGDF], axis=1)
    poleGDF = poleGDF.set_index('P_Tag', drop=False)

    # sort poles into groups of similar edge closeness
    print("sorting common edge poles for linear inclusion")
    sortedGrps = sortPoleNodesForSingleEdges(poleGDF, gdfsN)
    sortedGrpsWithDMatrix = sortPoleGroupsFromUtoV(sortedGrps)
    import pdb; pdb.set_trace()
    #     pass



    return



def main(args):
    geoGDF, utmGDF, utm, hem, epsg = getPolesData(args.csvPath)
    clusterUtmGDF = clusterPoles(utmGDF)
    bb = geoGDF.total_bounds
    G = getOSMNXGraphOfBBox(bb[3], bb[1], bb[2], bb[0])
    insertPolesIntoGraph(G, clusterUtmGDF, 'P_Tag')

    returnFoliumOfPoleClusters(clusterUtmGDF, [45.27718945, -123.0839672])

    import pdb; pdb.set_trace()
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
    main(args)
