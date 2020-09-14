import numpy as np
from graphviz import Digraph
import json
import pydotplus

class OffloadingTask(object):
    def __init__(self, id_name, process_data_size, transmission_data_size, type_name, depth=0, heft_score=0 ):
        self.id_name = id_name
        self.processing_data_size = process_data_size
        self.transmission_data_size = transmission_data_size
        self.type_name = type_name
        self.depth = depth
        self.heft_score = heft_score
        self.all_locally_execute = 0.0
        self.all_mec_execute = 0.0

    def print_task(self):
        print("task id name: {}, task type name: {} task processing data size: {}, "
              "task transmission_data_size: {}".format(
                                self.id_name, self.type_name,
            self.processing_data_size, self.transmission_data_size))


class OffloadingDotParser(object):
    """
        Parser for Task_graphs

        Args:
            file_name: the graphviz file name
        """
    def __init__(self, file_name):
        self.succ_task_for_ids = {}
        self.pre_task_for_ids = {}

        self.dot_ob = pydotplus.graphviz.graph_from_dot_file(file_name)
        self._parse_task()
        self._parse_dependecies()
        self._calculate_depth_and_transimission_datasize()

    def _parse_task(self):
        jobs = self.dot_ob.get_node_list()
        self.task_list = [0] * len(jobs)

        for job in jobs:
            job_id = job.get_name()
            data_size = int(eval(job.obj_dict['attributes']['size']))
            communication_data_size = int(eval(job.obj_dict['attributes']['expect_size']))

            task = OffloadingTask(job_id, data_size, 0, "compute")
            task.transmission_data_size = communication_data_size
            id = int(job_id) - 1
            self.task_list[id] = task

    def _parse_dependecies(self):
        edge_list = self.dot_ob.get_edge_list()
        dependencies = []

        task_number = len(self.task_list)
        dependency_matrix = np.zeros(shape=(task_number, task_number),
                                     dtype=np.float32)

        for i in range(len(self.task_list)):
            self.pre_task_for_ids[i] = []
            self.succ_task_for_ids[i] = []
            dependency_matrix[i][i] = self.task_list[i].processing_data_size

        for edge in edge_list:
            source_id = int(edge.get_source()) - 1
            destination_id = int(edge.get_destination()) - 1
            data_size = int(eval(edge.obj_dict['attributes']['size']))

            self.pre_task_for_ids[destination_id].append(source_id)
            self.succ_task_for_ids[source_id].append(destination_id)

            dependency = [source_id, destination_id, data_size]

            dependency_matrix[source_id][destination_id] = data_size
            dependencies.append(dependency)

        self.dependencies = dependencies
        self.dependency_matrix = dependency_matrix

    def _calculate_depth_and_transimission_datasize(self):
        ids_to_depth = dict()

        def caluclate_depth_value(id):
            if id in ids_to_depth.keys():
                return ids_to_depth[id]
            else:
                if len(self.pre_task_for_ids[id]) != 0:
                    depth = 1 + max([caluclate_depth_value(pre_task_id) for
                                     pre_task_id in self.pre_task_for_ids[id]])
                else:
                    depth = 0

                ids_to_depth[id] = depth

            return ids_to_depth[id]

        for id in range(len(self.task_list)):
            ids_to_depth[id] = caluclate_depth_value(id)

        for id, depth in ids_to_depth.items():
            self.task_list[id].depth = depth

    def generate_task_list(self):
        return self.task_list

    def generate_dependency(self):
        return self.dependencies


class OffloadingTaskGraph(object):
    """
        The Parsed Task Graph

        Args:
            file_name: the raw file of the task graph
        """
    def __init__(self, file_name):
        self._parse_from_dot(file_name)

    # add task list to
    def _parse_from_dot(self, file_name):
        parser = OffloadingDotParser(file_name)
        task_list = parser.generate_task_list()

        self.task_number = len(task_list)
        self.dependency = np.zeros((self.task_number, self.task_number))
        self.task_list = []
        self.prioritize_sequence=[]

        self.pre_task_sets = []
        self.succ_task_sets = []
        self.task_finish_time = [0] * self.task_number
        self.edge_set = []

        for _ in range(self.task_number):
            self.pre_task_sets.append(set([]))
            self.succ_task_sets.append(set([]))
        # add task list to
        self.add_task_list(task_list)

        dependencies = parser.generate_dependency()

        for pair in dependencies:
            self.add_dependency(pair[0], pair[1], pair[2])

        # get max data size and min data size, used to feature scaling.
        self.max_data_size = np.max(self.dependency[self.dependency > 0.01])
        self.min_data_size = np.min(self.dependency[self.dependency > 0.01])

    def add_task_list(self, task_list):
        self.task_list = task_list

        for i in range(0, len(self.task_list)):
            self.dependency[i][i] = task_list[i].processing_data_size

    def norm_feature(self, data_size):
        return float(data_size - self.min_data_size) / float(self.max_data_size - self.min_data_size)

    def add_dependency(self, pre_task_index, succ_task_index, transmission_cost):
        self.dependency[pre_task_index][succ_task_index] = transmission_cost
        self.pre_task_sets[succ_task_index].add(pre_task_index)
        self.succ_task_sets[pre_task_index].add(succ_task_index)

        # for each edge, we use a five dimension vector to represent this
        edge = [pre_task_index,
                self.task_list[pre_task_index].depth,
                self.task_list[pre_task_index].processing_data_size,
                transmission_cost,
                succ_task_index,
                self.task_list[succ_task_index].depth,
                self.task_list[succ_task_index].processing_data_size]

        self.edge_set.append(edge)

    def encode_point_sequence(self):
        point_sequence = []
        for i in range(self.task_number):
            norm_processing_data_size = self.norm_feature(self.task_list[i].processing_data_size)
            norm_transmission_data_size = self.norm_feature(self.task_list[i].transmission_data_size)
            norm_data_size_list = [norm_processing_data_size, norm_transmission_data_size]
            # heft_score = [self.task_list[i].heft_score]

            pre_task_index_set = []
            succs_task_index_set = []

            for pre_task_index in range(0, i):
                if self.dependency[pre_task_index][i] > 0.1:
                    pre_task_index_set.append(pre_task_index)

            while (len(pre_task_index_set) < 6):
                pre_task_index_set.append(-1.0)

            for succs_task_index in range(i + 1, self.task_number):
                if self.dependency[i][succs_task_index] > 0.1:
                    succs_task_index_set.append(succs_task_index)

            while (len(succs_task_index_set) < 6):
                succs_task_index_set.append(-1.0)

            succs_task_index_set = succs_task_index_set[0:6]
            pre_task_index_set = pre_task_index_set[0:6]

            point_vector = norm_data_size_list + pre_task_index_set + succs_task_index_set
            point_sequence.append(point_vector)

        return point_sequence

    def encode_point_sequence_with_ranking(self, sorted_task):
        point_sequence = self.encode_point_sequence()

        prioritize_point_sequence = []
        for task_id in sorted_task:
            prioritize_point_sequence.append(point_sequence[task_id])

        return prioritize_point_sequence

    def encode_point_sequence_with_cost(self, resource_cluster):
        point_sequence = []
        for i in range(self.task_number):
            task = self.task_list[i]
            local_process_cost = task.processing_data_size / resource_cluster.mobile_process_capable
            up_link_cost = resource_cluster.up_transmission_cost(task.processing_data_size)
            mec_process_cost = task.processing_data_size / resource_cluster.mec_process_capble
            down_link_cost = resource_cluster.dl_transmission_cost(task.transmission_data_size)

            task_embeding_vector = [i, local_process_cost, up_link_cost,
                                    mec_process_cost, down_link_cost]

            pre_task_index_set = []
            succs_task_index_set = []

            for pre_task_index in range(0, i):
                if self.dependency[pre_task_index][i] > 0.1:
                    pre_task_index_set.append(pre_task_index)

            while (len(pre_task_index_set) < 6):
                pre_task_index_set.append(-1.0)

            for succs_task_index in range(i + 1, self.task_number):
                if self.dependency[i][succs_task_index] > 0.1:
                    succs_task_index_set.append(succs_task_index)

            while (len(succs_task_index_set) < 6):
                succs_task_index_set.append(-1.0)

            succs_task_index_set = succs_task_index_set[0:6]
            pre_task_index_set = pre_task_index_set[0:6]

            point_vector = task_embeding_vector + pre_task_index_set + succs_task_index_set
            point_sequence.append(point_vector)

        return point_sequence

    def encode_point_sequence_with_ranking_and_cost(self, sorted_task, resource_cluster):
        point_sequence = self.encode_point_sequence_with_cost(resource_cluster)

        prioritize_point_sequence = []
        for task_id in sorted_task:
            prioritize_point_sequence.append(point_sequence[task_id])

        return prioritize_point_sequence

    def encode_edge_sequence(self):
        edge_array = []
        for i in range(0, len(self.edge_set)):
            if i < len(self.edge_set):
                edge_array.append(self.edge_set[i])
            else:
                edge_array.append([0, 0, 0, 0, 0, 0, 0])

        # input edge sequence refers to start node index
        edge_array = sorted(edge_array)

        return edge_array

    def return_cost_metric(self):
        adj_matrix = np.array(self.dependency)
        cost_set = adj_matrix[np.nonzero(adj_matrix)]
        cost_set = cost_set[cost_set > 0.01]

        mean = np.mean(cost_set)
        std = np.std(cost_set)

        return mean, std

    def print_graphic(self):
        print(self.dependency)
        print("This is pre_task_sets:")
        print(self.pre_task_sets)
        print("This is edge set:")
        print(self.edge_set)

    def prioritize_tasks(self, resource_cluster):
        w = [0] * self.task_number
        for i, task in enumerate(self.task_list):
            t_locally = task.processing_data_size / resource_cluster.mobile_process_capable
            t_mec = resource_cluster.up_transmission_cost(task.processing_data_size) + \
                    task.processing_data_size / resource_cluster.mec_process_capble + \
                    resource_cluster.dl_transmission_cost(task.transmission_data_size)

            w[i] = min(t_locally, t_mec)

        rank_dict = [-1] * self.task_number
        def rank(task_index):
            if rank_dict[task_index] != -1:
                return rank_dict[task_index]

            if len(self.succ_task_sets[task_index]) == 0:
                rank_dict[task_index] = w[task_index]
                return rank_dict[task_index]
            else:
                rank_dict[task_index] = w[task_index] + max(rank(j) for j in self.succ_task_sets[task_index])
                return rank_dict[task_index]

        for i in range(self.task_number):
            rank(i)

        sort = np.argsort(rank_dict)[::-1]
        self.prioritize_sequence = sort
        return sort

    def render(self, path):
        dot = Digraph(comment='DAG')

        # str(self.task_list[i].running_time)
        for i in range(0, self.task_number):
            dot.node(str(i), str(i) + ":" + str(self.task_list[i].processing_data_size))

        for e in self.edge_set:
            dot.edge(str(e[0]), str(e[4]), constraint='true', label="%.6f" % e[3])

        dot.render(path, view=False)
