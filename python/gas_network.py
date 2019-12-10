#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from networkx import MultiGraph
from networkx.classes.function import set_node_attributes, set_edge_attributes, get_node_attributes, get_edge_attributes
from networkx.linalg.graphmatrix import incidence_matrix
from numpy import asarray


class GasNetwork(MultiGraph):
    def __init__(self, nodes, edges, compressors):
        edge_list, edge_definitions = list(edges.keys()), list(edges.values())
        nodes_container_type = not isinstance(nodes, list)
        node_numbers_types = not all([isinstance(nodes[i], int) for i in range(len(nodes))])
        edges_container_type = not isinstance(edges, dict)
        edge_numbers_types = not all([isinstance(edge_list[i], int) for i in range(len(edge_list))])
        edge_objects_types = not all([isinstance(edge_definitions[i], tuple) for i in range(len(edge_definitions))])
        edge_definitions_consistency = not all(
            [edge_definitions[i][0] and edge_definitions[i][1] in nodes for i in range(len(edge_definitions))])
        if nodes_container_type or node_numbers_types:
            raise TypeError("Check node input container object type is list, with elements of type int.")
        elif edges_container_type or edge_numbers_types or edge_objects_types:
            raise TypeError(
                "Check edges input container object type is dict, "
                "with keys of type int and values of type tuple.")
        elif edge_definitions_consistency:
            raise IOError(
                "Check consistency of edges definition - "
                "all edges must be defined using nodes included in the nodes list input.")
        else:
            super(GasNetwork, self).__init__()

            self.ref_nodes = nodes
            self.ref_edges = edges
            self.ref_compressors = compressors

            self.add_nodes_from(nodes)
            self.add_edges_from(edges.values())

            self._p_min_init = False
            self._p_max_init = False
            self._s_min_init = False
            self._s_max_init = False
            self._c_init = False
            self._VoLL_init = False
            self._cc_init = False
            self._d_init = False
            self._ref_flows_init = False

    @property
    def incidence_matrix(self):
        return asarray(incidence_matrix(self, nodelist=self.ref_nodes, edgelist=list(self.ref_edges.values()), oriented=True).todense())
    
    @property
    def ref_pipes(self):
        ref_keys, compr_keys = list(self.ref_edges.keys()), list(self.ref_compressors.keys())
        keys = [key for key in ref_keys if key not in compr_keys]
        values = [self.ref_edges[key] for key in keys]
        return dict(zip(keys, values))
        
    @property
    def minimum_pressure_bounds(self):
        if self._p_min_init:
            return get_node_attributes(self, "p_min")
        else:
            raise ValueError("Lower bounds on nodal pressures have not been initialised.")
    
    @minimum_pressure_bounds.setter
    def minimum_pressure_bounds(self, p_min):
        if isinstance(p_min, dict) and len(p_min.values()) == len(self.nodes):
            set_node_attributes(self, p_min, name="p_min")
            self._p_min_init = True
        else:
            raise IOError("Check input object type is dict and has as many values as there are edges.")
        return None
    
    @property
    def maximum_pressure_bounds(self):
        if self._p_max_init:
            return get_node_attributes(self, "p_max")
        else:
            raise ValueError("Upper bounds on nodal pressures have not been initialised.")
    
    @maximum_pressure_bounds.setter
    def maximum_pressure_bounds(self, p_max):
        if isinstance(p_max, dict) and len(p_max.values()) == len(self.nodes):
            set_node_attributes(self, p_max, name="p_max")
            self._p_max_init = True
        else:
            raise IOError("Check input object type is dict and has as many values as there are edges.")
        return None
    
    @property
    def minimum_nodal_injections(self):
        if self._s_min_init:
            return get_node_attributes(self, "s_min")
        else:
            raise ValueError("Nodal consumptions have not been initialised.")
    
    @minimum_nodal_injections.setter
    def minimum_nodal_injections(self, s_min):
        if isinstance(s_min, dict) and len(s_min.values()) == len(self.nodes):
            set_node_attributes(self, s_min, name="s_min")
            self._s_min_init = True
        else:
            raise IOError("Check input object type is dict and has as many values as there are edges.")
        return None
    
    @property
    def maximum_nodal_injections(self):
        if self._s_max_init:
            return get_node_attributes(self, "s_max")
        else:
            raise ValueError("Maximum nodal injections have not been initialised.")
    
    @maximum_nodal_injections.setter
    def maximum_nodal_injections(self, s_max):
        if isinstance(s_max, dict) and len(s_max.values()) == len(self.nodes):
            set_node_attributes(self, s_max, name="s_max")
            self._s_max_init = True
        else:
            raise IOError("Check input object type is dict and has as many values as there are edges.")
        return None
    
    @property
    def friction_coefficients(self):
        if self._c_init:
            attr = get_edge_attributes(self, "c")
            keys, values = list(self.ref_edges.keys()), list(attr.values())
            return dict(zip(keys, values))
        else:
            raise ValueError("Friction coefficients have not been initialised.")
            
    @friction_coefficients.setter
    def friction_coefficients(self, c):
        if isinstance(c, dict) and len(c.values()) == len(self.edges):
            keys, values = list(self.edges), list(c.values())
            attr = dict(zip(keys, values))
            set_edge_attributes(self, attr, name="c")
            self._c_init = True
        else:
            raise IOError("Check input object type is dict and has as many values as there are edges.")
        return None
    
    @property
    def minimum_pressure_ratio(self):
        if self._alpha_min_init:
            attr = get_edge_attributes(self, "alpha_min")
            keys, values = list(self.ref_edges.keys()), list(attr.values())
            return dict(zip(keys, values))
        else:
            raise ValueError("Minimum bounds on nodal pressures have not been initialised.")
    
    @minimum_pressure_ratio.setter
    def minimum_pressure_ratio(self, alpha_min):
        if isinstance(alpha_min, dict) and len(alpha_min.values()) == len(self.edges):
            keys, values = list(self.edges), list(alpha_min.values())
            attr = dict(zip(keys, values))
            set_edge_attributes(self, attr, name="alpha_min")
            self._alpha_min_init = True
        else:
            raise IOError("Check input object type is dict and has as many values as there are edges.")
        return None
    
    @property
    def maximum_pressure_ratio(self):
        if self._alpha_max_init:
            attr = get_edge_attributes(self, "alpha_max")
            keys, values = list(self.ref_edges.keys()), list(attr.values())
            return dict(zip(keys, values))
        else:
            raise ValueError("Upper bounds on nodal pressures have not been initialised.")
    
    @maximum_pressure_ratio.setter
    def maximum_pressure_ratio(self, alpha_max):
        if isinstance(alpha_max, dict) and len(alpha_max.values()) == len(self.edges):
            keys, values = list(self.edges), list(alpha_max.values())
            attr = dict(zip(keys, values))
            set_edge_attributes(self, attr, name="alpha_max")
            self._alpha_max_init = True
        else:
            raise IOError("Check input object type is dict and has as many values as there are edges.")
        return None
 
    @property
    def value_unserved_demand(self):
        if self._VoLL_init:
            return get_node_attributes(self, "VoLL")
        else:
            raise ValueError("Value of Lost Load (VoLL) for gas consumption has not been initialised.")
            
    @value_unserved_demand.setter
    def value_unserved_demand(self, VoLL):
        if isinstance(VoLL, dict) and len(VoLL.values()) == len(self.nodes):
            set_node_attributes(self, VoLL, name="VoLL")
            self._VoLL_init = True
        else:
            raise IOError("Check input object type is float and has value greater than 0.")
            
    @property
    def compression_costs(self):
        if self._cc_init:
            return self.cc_value
        else:
            raise ValueError("Compression costs have not been initialised.")
            
    @compression_costs.setter
    def compression_costs(self, cc):
        if isinstance(cc, float) or isinstance(cc, int) and cc > 0:
            self.cc_value = cc
            self._cc_init = True
        else:
            raise IOError("Check input object type is float and has value greater than 0.")
            
    @property
    def nodal_demands(self):
        if self._d_init:
            return get_node_attributes(self, "d")
        else:
            raise ValueError("Maximum nodal injections have not been initialised.")
    
    @nodal_demands.setter
    def nodal_demands(self, d):
        if isinstance(d, dict) and len(d.values()) == len(self.nodes):
            set_node_attributes(self, d, name="d")
            self._d_init = True
        else:
            raise IOError("Check input object type is dict and has as many values as there are edges.")
        return None
                           
    @property
    def reference_flows(self):
        if self._ref_flows_init:
            attr = get_edge_attributes(self, "ref_flows")
            keys, values = list(self.ref_edges.keys()), list(attr.values())
            return dict(zip(keys, values))
        else:
            raise ValueError("Reference flows have not been initialised.")
    
    @reference_flows.setter
    def reference_flows(self, ref_flows):
        if isinstance(ref_flows, dict) and len(ref_flows.values()) == len(self.edges):
            keys, values = list(self.edges), [2*val for val in list(ref_flows.values())]
            attr = dict(zip(keys, values))
            set_edge_attributes(self, attr, name="ref_flows")
            self._ref_flows_init = True
        else:
            raise IOError("Check input object type is dict and has as many values as there are edges.")
        return None
        
    