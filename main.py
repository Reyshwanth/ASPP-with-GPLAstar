import networkx as nx
import matplotlib.pyplot as plt
from networkx.generators.trees import prefix_tree
import numpy as np

from animationV2 import AnimateV2
from GPLAstar import GPLAstar
from centralizedAstar import centralizedAstar

import csv, time
from ultis import BestFirstSearch
from graph import graph





def main():
    planner()



class planner():
    def __init__(self):


        '''Define the instance'''
        self.generate_graph = graph(grid_size= [6,6], frac_imp=0.5, cuts = 0, unimp_cost_range = [10,16], 
                                imp_cost_range = [40,51], SV_cost_range = [1,2], service_cost_range = [1,6] )
        
        self.GV_start = self.generate_graph.GV_start
        self.GV_goal = self.generate_graph.GV_goal
        self.SV_start = self.generate_graph.SV_start
        self.Graph = self.generate_graph.G
        self.impeded_edges = self.generate_graph.impeded_edges
        self.perm_label_path ,self.lifted_graph_path = [], []

        self.colors = ['b','g','y','m','k','c']

        breakpoint()
        # input()
        print('GV_start',self.GV_start)
        print('GV_goal',self.GV_goal)
        print('SV_start',self.SV_start)
        print('Impeded_edges', len(self.impeded_edges), self.impeded_edges)

        self.upper_bound = nx.shortest_path_length(self.Graph, source=self.GV_goal[0], target=None, weight='impeded_cost', method='dijkstra')
        self.lower_bound = nx.shortest_path_length(self.Graph, source=self.GV_goal[0], target=None, weight='unimpeded_cost', method='dijkstra')

        UB = self.upper_bound[self.GV_start[0]]
        LB = self.lower_bound[self.GV_start[0]]
        print('upper bound', self.upper_bound[self.GV_start[0]])
        print('lower bound', self.lower_bound[self.GV_start[0]])

        upper_bound_search = BestFirstSearch(self.Graph,self.GV_goal[0])
        self.UB_parents, _ = upper_bound_search.use_algorithm()

        frac_imp_edges = len(self.impeded_edges)/len(self.Graph.edges)


        time_limit = 900


        print('############ Starting new GPLAstar algo ############# ')
        starttime = time.time()
        GPLAsim = GPLAstar(self.Graph,self.GV_start,self.GV_goal,self.SV_start,self.impeded_edges,self.upper_bound, self.UB_parents, self.lower_bound, time_limit)
        GPLAsim_cost = GPLAsim.UB_cost 
        GPLAsim_time = time.time()-starttime 

        print('Vehicle Trajectory (gv_pos, sv_pos, gv_time, sv_time)',GPLAsim.Final_path)

        print('Run Time ', GPLAsim_time,  'Cost ', GPLAsim_cost)


        self.perm_label_path=GPLAsim.Final_path

        print('############ Starting Centralized A* algo ############# ')
        starttime = time.time()
        cen_algo = centralizedAstar(self.Graph,self.GV_start,self.GV_goal,self.SV_start,self.impeded_edges,self.upper_bound,self.lower_bound)
        cen_cost = cen_algo.Final_cost
        cen_time = time.time()-starttime
        
        print('Vehicle Trajectory (gv_pos, sv_pos, gv_time, sv_time)',cen_algo.Final_path)

        print('Run Time ', cen_time,  'Cost ', cen_cost)





        print('##########################################################################')
        self.initialize_plot()
        self.animate_motion("nothing")


    def initialize_plot(self):
        fig, ax = plt.subplots()

        for i in range(len(self.SV_start)):
            ax.plot([],[],  markersize=12, marker='^', color=self.colors[i+len(self.GV_start)], 
                alpha=1.0, zorder=3, label="SV"+str(i+1))
        for i in range(len(self.GV_start)):
            ax.plot([],[],  markersize=12, marker='s', color=self.colors[i], 
                alpha=1.0, zorder=3, label="GV"+str(i+1))  
        plt.legend(loc='upper center',bbox_to_anchor=(0.65, 1.10), shadow=True, ncol=3)
        AnimateV2.init_figure(fig, ax)

        nodes = list(self.generate_graph.inverse_mapping.values())

        edge_x,edge_y = [],[]
        impedge_x,impedge_y = [],[]
        for edge in self.Graph.edges:
            if edge in self.impeded_edges:
                #breakpoint()
                impedge_x.append((self.generate_graph.inverse_mapping[edge[0]][0],self.generate_graph.inverse_mapping[edge[1]][0]))
                impedge_y.append((self.generate_graph.inverse_mapping[edge[0]][1],self.generate_graph.inverse_mapping[edge[1]][1]))
            else:
                #breakpoint()

                edge_x.append((self.generate_graph.inverse_mapping[edge[0]][0],self.generate_graph.inverse_mapping[edge[1]][0]))
                edge_y.append((self.generate_graph.inverse_mapping[edge[0]][1],self.generate_graph.inverse_mapping[edge[1]][1]))
        #breakpoint()
        AnimateV2.add("Nodes", [x[0] for x in nodes], [x[1] for x in nodes], draw_clean=True, markersize=1, marker='o', color='k', alpha=0.5)
        obstacle = np.array([obs for obs,node in self.Graph.nodes(data=True)if node['obs']==1])
        if len(obstacle) > 0:
            AnimateV2.add("Nodes", obstacle[:,0], obstacle[:,1], draw_clean=True, markersize=10, marker='o', color='k', alpha=1)
        

        
        for i in range(len(self.SV_start)):
            AnimateV2.add("SV_pos"+str(i), self.generate_graph.inverse_mapping[self.SV_start[i]], draw_clean=True,  markersize=10, marker='^', color=self.colors[i+len(self.GV_start)], alpha=1.0, zorder=3)

        for i in range(len(self.GV_start)):
            AnimateV2.add("GV_goal"+str(i), self.generate_graph.inverse_mapping[self.GV_goal[i]], draw_clean=True,  markersize=10, marker='o', color=self.colors[i], alpha=1.0, zorder=3)
            AnimateV2.add("GV_pos"+str(i), self.generate_graph.inverse_mapping[self.GV_start[i]], draw_clean=True,  markersize=10, marker='s', color=self.colors[i], alpha=1.0, zorder=3)
        

        plt.plot(np.array(edge_x).T, np.array(edge_y).T, linestyle='-', color='k', alpha=0.3) 
        plt.plot(np.array(impedge_x).T, np.array(impedge_y).T, linestyle='-', color='r')

        AnimateV2.update()

    def animate_motion(self, algo):
        print("animating")
        if algo == 'lifted_graph': 
            old_GV_pos, old_SV_pos = self.GV_start, self.SV_start
            #breakpoint()
            for pos in self.lifted_graph_path:
                GV_pos = pos[:len(self.GV_start)]
                SV_pos = pos[len(self.GV_start):]
                

                for i in range(len(GV_pos)):
                    AnimateV2.add("GV_pos"+str(i), self.generate_graph.inverse_mapping[GV_pos[i]], draw_clean=True,  markersize=10, marker='s', color=self.colors[i], alpha=1.0, zorder=3)
                    if old_GV_pos[i]!=GV_pos[i]:
                        edge = np.array((old_GV_pos[i],GV_pos[i]))
                        plt.plot(edge[:,0], edge[:,1], linestyle='-', color=self.colors[i],linewidth=2) 
                for i in range(len(SV_pos)):
                    AnimateV2.add("SV_pos"+str(i), self.generate_graph.inverse_mapping[SV_pos[i]], draw_clean=True,  markersize=10, marker='^', color=self.colors[i+len(self.GV_start)], alpha=1.0, zorder=3)
                    if old_SV_pos[i]!=SV_pos[i]:
                        edge = np.array((old_SV_pos[i],SV_pos[i]))
                        plt.plot(edge[:,0], edge[:,1], linestyle='-', color=self.colors[i+len(self.GV_start)],linewidth=2) 
                
                old_GV_pos, old_SV_pos = GV_pos, SV_pos
                AnimateV2.update()
                print(pos)
                # time.sleep(1)
                input()
        
        
        else:
            
            GV_path = [x[0] for x in self.perm_label_path]
            GV_time = [x[2] for x in self.perm_label_path]
            SV_path = [x[1] for x in self.perm_label_path]
            SV_time = [x[3] for x in self.perm_label_path]
            sim_time = GV_time[-1]
            speed_up = 1 
            start = time.time()
        
            SV_term = False

            gv_ptr, sv_ptr = 0, 0

            # GV_pos, SV_pos = np.array(GV_path[gv_ptr]), np.array(SV_path[sv_ptr])
            
            while time.time() - start <= sim_time:

                curr_time = time.time() - start

                if GV_time[gv_ptr] < curr_time:
                    gv_ptr += 1
                    if GV_time[gv_ptr]==GV_time[gv_ptr-1]:
                        gv_speed = 0
                    else:
                        gv_speed = (np.array(self.generate_graph.inverse_mapping[GV_path[gv_ptr]]) - np.array(self.generate_graph.inverse_mapping[GV_path[gv_ptr-1]]))/(GV_time[gv_ptr]-GV_time[gv_ptr-1])

                if SV_term == False:
                    if curr_time > SV_time[-1]:
                        if sv_ptr>0:
                            if (SV_path[sv_ptr-1],SV_path[sv_ptr]) in self.impeded_edges or (SV_path[sv_ptr],SV_path[sv_ptr-1]) in self.impeded_edges:
                                edge = np.array((SV_path[sv_ptr-1],SV_path[sv_ptr]))
                                plt.plot([self.generate_graph.inverse_mapping[edge[0]][0],
                                             self.generate_graph.inverse_mapping[edge[1]][0]],[self.generate_graph.inverse_mapping[edge[0]][1],
                                             self.generate_graph.inverse_mapping[edge[1]][1]], linestyle='-', color=self.colors[len(self.GV_start)],linewidth=2)
                        
                        sv_ptr = len(SV_path)
                        sv_speed = 0
                        SV_term = True

                        
                    
                    else:
                        if SV_time[sv_ptr] < curr_time:
                            sv_ptr += 1
                            sv_speed = (np.array(self.generate_graph.inverse_mapping[SV_path[sv_ptr]]) - np.array(self.generate_graph.inverse_mapping[SV_path[sv_ptr-1]]))/(SV_time[sv_ptr]-SV_time[sv_ptr-1])
                            
                            if sv_ptr>1:
                                if (SV_path[sv_ptr-2],SV_path[sv_ptr-1]) in self.impeded_edges or (SV_path[sv_ptr-1],SV_path[sv_ptr-2]) in self.impeded_edges:
                                    edge = np.array((SV_path[sv_ptr-2],SV_path[sv_ptr-1]))
                                    #breakpoint()
                                    plt.plot([self.generate_graph.inverse_mapping[edge[0]][0],
                                             self.generate_graph.inverse_mapping[edge[1]][0]],[self.generate_graph.inverse_mapping[edge[0]][1],
                                             self.generate_graph.inverse_mapping[edge[1]][1]], linestyle='-', color=self.colors[len(self.GV_start)],linewidth=2)


                GV_pos = np.array(self.generate_graph.inverse_mapping[GV_path[gv_ptr-1]]) + gv_speed*(curr_time - GV_time[gv_ptr-1])
                SV_pos = np.array(self.generate_graph.inverse_mapping[SV_path[sv_ptr-1]]) + sv_speed*(curr_time - SV_time[sv_ptr-1])


                AnimateV2.add("GV_pos"+str(0), GV_pos, draw_clean=True,  markersize=10, marker='s', color=self.colors[0], alpha=1.0, zorder=3)
                AnimateV2.add("SV_pos"+str(0), SV_pos, draw_clean=True,  markersize=10, marker='^', color=self.colors[len(self.GV_start)], alpha=1.0, zorder=3)

                AnimateV2.update()
                #breakpoint()






if __name__=='__main__':

	main()