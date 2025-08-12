import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

class WastePredictor:
    def __init__(self, model_path=None):
        if model_path:
            self.load_model(model_path)
        else:
            self.model = RandomForestRegressor(n_estimators=100)
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, area, day_of_week, weather):
        # In a real app, this would be more sophisticated
        features = pd.DataFrame({
            'area': [area],
            'day_of_week': [day_of_week],
            'weather': [weather]
        })
        return self.model.predict(features)[0]
    
    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

class RouteOptimizer:
    def __init__(self):
        pass
    
    def optimize(self, bins):
        """Optimize collection route using OR-Tools"""
        # Create distance matrix
        locations = [(bin['lat'], bin['lng']) for bin in bins]
        distance_matrix = self._create_distance_matrix(locations)
        
        # Create routing model
        manager = pywrapcp.RoutingIndexManager(
            len(distance_matrix), 1, 0)  # 1 vehicle, depot at 0
        routing = pywrapcp.RoutingModel(manager)
        
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Solve
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        
        solution = routing.SolveWithParameters(search_parameters)
        
        # Extract route
        route = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
        
        return [bins[i] for i in route]
    
    def _create_distance_matrix(self, locations):
        # Simplified - in reality use Haversine or actual road distances
        size = len(locations)
        matrix = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                if i == j:
                    continue
                lat1, lon1 = locations[i]
                lat2, lon2 = locations[j]
                matrix[i][j] = self._haversine(lat1, lon1, lat2, lon2)
        return matrix
    
    def _haversine(self, lat1, lon1, lat2, lon2):
        # Calculate distance between two points on Earth
        from math import radians, sin, cos, sqrt, atan2
        R = 6371  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
