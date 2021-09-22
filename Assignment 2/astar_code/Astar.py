from Map import *  # Small manipulations are done in "Map.py"
from queue import PriorityQueue
from PIL import ImageDraw, ImageFont

class Search_node:
    def __init__(self, state):
        """
        Initiate search node with state.
        """
        self.state = state  # Location and id of node
        self.g = 0  # Current cost to reach node from root node
        self.h = 0  # Heuristic
        self.f = self.g + self.h  # Total cost
        self.status = None  # Either 'OPEN', 'CLOSED' or 'PATH'
        self.parent = None  # A pointer to parent node
        self.children = []  # A list of children nodes
    
    def update_f(self):
        """
        Updates the total cost.
        """
        self.f = self.g + self.h
    
    def __lt__(self, other):
        """
        Overloads the `<` operator. This is used for the priority queue
        (binary heap) in the A* algorithm.
        """
        return self.f < other.f
    
    def __str__(self):
        """
        Enables printing useful information about the node.
        """
        return '\n'.join([f"{self.state},",
                          f"g = {self.g}, h = {self.h}, f = {self.f},",
                          f"status = {self.status}, " +
                          f"n_children = {len(self.children)}"])
    
    def __repr__(self):
        """
        Similar to `__str__()`.
        """
        return '\n'.join(["<class Search_node>", self.__str__()])
        

class Search_state:
    def __init__(self, pos, state_id):
        """
        Initiate search state which contains information about location and
        a unique id of the node. The id is used to check if the node has been
        previously discovered.
        """
        self.pos = pos      # Position of node
        self.id = state_id  # Unique id (integer) for the node at this position
    
    def __str__(self):
        """
        Enables printing useful information about the state.
        """
        return f"position = {self.pos}, id = {self.id}"
    
    def __repr__(self):
        """
        Similar to `__str__()`.
        """
        return '\n'.join(["<class Search_state>", self.__str__()])
    

class Bfs_Obj(Map_Obj):
    def __init__(self, task = 1):
        """
        Initiates Best First Search object that inherits attributes and methods
        from Map_Obj(). This class consists of a maze with a start position and
        a goal position. The A* algorithms tries to find the best path, where
        'best' is measured in cost of passing each node.
        """
        super(Bfs_Obj, self).__init__(task)  # Inherit attributes and methods
        self.ncols = self.int_map.shape[1]  # Number of columns in map
        self.root_node = self.create_root_node()  # Start node
        self.hash_table = {self.root_node.state.id: self.root_node}  # Table of
                                                                     # nodes
        self.images = [self.get_map_image()]  # List of images from search
        self.add_nodes_visited(self.images[0])  # Adds text in first image
        
    def create_root_node(self):
        """
        Creates the root node in the search graph.
        """
        pos = np.array(self.get_start_pos())
        state_id = self.get_node_id(pos)
        state = Search_state(pos, state_id)
        return Search_node(state)
    
    def get_node_id(self, pos):
        """
        Get node/state id.
        """
        return self.ncols*pos[0] + pos[1]
    
    def push(self, node, l):
        """
        Pushes node to list l, either a normal list, or a priority queue
        (binary heap), and updates the image.
        """
        if isinstance(l, PriorityQueue):
            l.put(node)
            node.status = 'OPEN'
        else:
            node.status = 'CLOSED'
            l.append(node)
        self.update_image(node)  # Adds an image to `self.images`
        
    def generate_all_successors(self, node):
        """
        Generates all nodes adjacent to current node that are reachable.
        """
        successors = []
        # Step up, right, down and left
        for step in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            new_pos = node.state.pos + step
            if self.get_cell_value(new_pos) != -1:
                new_id = self.get_node_id(new_pos)
                new_state = Search_state(new_pos, new_id)
                new_node = Search_node(new_state)
                successors.append(new_node)
        return successors
    
    def solution_check(self, node):
        """
        Checks if node is located at the goal position.
        """
        return np.array_equal(self.goal_pos, node.state.pos)
    
    def heuristic_evaluation(self, node):
        """
        Computes the distance between the goal node and the current node where
        the Manhattan distance is the distance measure.
        """
        return np.abs(self.goal_pos - node.state.pos).sum()
    
    def attach_and_eval(self, child, parent):
        """
        Attaches the child to the best parent so far and updates the cost and
        heuristics.
        """
        child.parent = parent
        child.g = parent.g + self.arc_cost(parent, child)
        child.h = self.heuristic_evaluation(child)
        child.update_f()
    
    def arc_cost(self, parent, child):
        """
        Computes the arc cost between parent and child node.
        """
        return self.get_cell_value(child.state.pos)
    
    def propagate_path_improvements(self, parent):
        """
        Updates the total costs of the predecessors.
        """
        for child in parent.children:
            arc_cost = self.arc_cost(parent, child)
            if parent.g + arc_cost < child.g:
                child.parent = parent
                child.g = parent.g + arc_cost
                child.update_f()
                self.propagate_path_improvements(child)
    
    def add_nodes_visited(self, image):
        """
        Adds text to image with information about the amount of nodes visited
        so far.
        """
        draw = ImageDraw.Draw(image)
        draw.rectangle(((0, 0), (345, 38)), (211, 33, 45))  # Clear content
                                                            # from previous
                                                            # image
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 42)  # Set font
        draw.text(
            xy   = (0, 0),
            text = f"Nodes visited: {len(self.hash_table) - 1}",
            fill = (255, 255, 255),
            font = font
        )
    
    def update_image(self, node):
        """
        Adds an image to the list of images including the new node and its
        status. If node is in start position or goal position, we skip.
        """
        colors = {
            'OPEN':   (250, 128, 114),  # Salmon
            'CLOSED': ( 25,  25, 112),  # Midnight blue
            'PATH':   (255, 255,   0)   # Yellow
        }
        
        im = self.images[-1].copy()  # Copy of last image
        pix = im.load()              # Pixels
        pos = node.state.pos                 # Position
        start_id = self.root_node.state.id   # State id
        self.goal_pos = self.get_goal_pos()  # Goal position
        goal_id = self.get_node_id(self.goal_pos)  # Goal id
        if not node.state.id in [start_id, goal_id]:
            self.add_nodes_visited(im)  # Adds text of nodes visited
            for i in range(self.scale):
                for j in range(self.scale):
                    pix[pos[1]*self.scale + i,
                        pos[0]*self.scale + j] = colors[node.status]
            self.images.append(im)
    
    def get_goal_node(self):
        """
        Fetches the goal node.
        """
        pos = np.array(self.get_goal_pos())  # Position
        state_id = self.get_node_id(pos)     # State id
        return self.hash_table[state_id]     # Goal node
    
    def add_path_to_image(self):
        """
        Add the least cost path to the image, node by node, starting from goal
        position until start position is reached.
        """
        node = self.get_goal_node()  # Goal node
        while not node.parent is None:
            node = node.parent
            node.status = 'PATH'     # Change status for correct color when
            self.update_image(node)  # updating the image
    
    def get_final_cost(self):
        """
        Computes the cost of the path going from start to goal.
        """
        node = self.get_goal_node()
        return node.g


def best_first_search(bfs):
    """
    A* algorithm taken from "Essentials of the A* Algorithm" document.
    """
    closed = []
    q_open = PriorityQueue()
    n0 = bfs.root_node
    n0.g = 0
    n0.h = bfs.heuristic_evaluation(n0)
    n0.update_f()
    bfs.push(n0, q_open)
    while not q_open.empty():
        X = q_open.get()
        bfs.push(X, closed)
        if bfs.solution_check(X):
            return 'SUCCESS'
        SUCC = bfs.generate_all_successors(X)
        for S in SUCC:
            if S.state.id in bfs.hash_table:
                S = bfs.hash_table[S.state.id]
            else:
                bfs.hash_table[S.state.id] = S
            X.children.append(S)
            if not (S.status in ['CLOSED', 'OPEN']):
                bfs.attach_and_eval(S, X)
                bfs.push(S, q_open)
            elif X.g + bfs.arc_cost(X, S) < S.g:
                bfs.attach_and_eval(S, X)
                if S.status == 'CLOSED':
                    bfs.propagate_path_improvements(S)
    return 'FAIL'

## Compute least cost paths of task 1, 2, 3 and 4
bfs_objs = []
for task in (1, 2, 3, 4):
    bfs = Bfs_Obj(task)
    print(best_first_search(bfs))
    bfs.add_path_to_image()
    print(f"Final cost: {bfs.get_final_cost()}")
    bfs_objs.append(bfs)



# ## Save animations and images
# path = "../output/"  # Edit to "" to save in current working directory

# ## Saves an (.gif) animation per task
# for i, bfs in enumerate(bfs_objs):
#     bfs.images[0].save(path + f"task{i + 1}.gif",
#         save_all = True,
#         append_images = bfs.images[1:],
#         optimize = False,
#         duration = 50,
#         loop = 0
#     )
#     print(f"{i + 1}/{len(bfs_objs)} animations saved!")


# ## Saves three (.png) images per task.
# for i, bfs in enumerate(bfs_objs):
#     snapshots = np.round(np.linspace(0, len(bfs.images) - 1, 3)).astype(int)
#     for j, shot in enumerate(snapshots):
#         bfs.images[shot].save(path + f"task{i + 1}_snapshot{j + 1}.png")
# print("Images saved!")